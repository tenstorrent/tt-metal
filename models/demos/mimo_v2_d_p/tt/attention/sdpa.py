# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chunked-prefill attention core over the block-cyclic SP KV cache (ring-joint SDPA, SP > 1).

* GA layers: full causal ring over the cached prefix; K head_dim 192, V head_dim 128 (the causal
  chunked path accepts DK != DV).
* SWA layers: one-hop compact-halo sliding ring (window 128) with per-head attention sink, V at 128 (the
  sliding path's VDH == DH check was relaxed to VDH <= DH; its kernels are generic in vDHt).

Requires chunk_size / SP >= window (one-hop halo) and chunk-aligned ``kv_actual`` for SWA (the sliding
ring needs ``logical_n`` on a ring-group boundary).
"""

import math
import os

import ttnn

TILE = 32


def _fidelity():
    return getattr(ttnn.MathFidelity, os.environ.get("MIMO_SDPA_FIDELITY", "HiFi2"))


def _largest_dividing(n, candidates):
    for c in candidates:
        if n % c == 0:
            return c
    return candidates[-1]


def ring_program_config(mesh_device, q_chunk=None, k_chunk=None, sliding=False, q_local=None, kv_local=None, k_split=1):
    """GA: q128 / k1024 measured best on BH 2x2 at 640 and 2048 tokens/chip (53.5% / 57.5% of HiFi2 peak at
    32k context vs 50.2% / 54.9% for q256/k512). q is shrunk to divide the per-device Q slab (Galaxy SP8:
    640-token slabs); k stays 1024 (the op masks a padded last K chunk). SWA: the sliding ring supports q 64/128, k 128.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    if q_chunk is None:
        q_chunk = (
            int(os.environ["MIMO_SDPA_Q_CHUNK"])
            if os.environ.get("MIMO_SDPA_Q_CHUNK")
            else (_largest_dividing(q_local, (128, 64, 32)) if q_local else 128)
        )
    if k_chunk is None:
        if sliding:
            k_chunk = 128
        elif os.environ.get("MIMO_SDPA_K_CHUNK"):
            k_chunk = int(os.environ["MIMO_SDPA_K_CHUNK"])
        else:
            # A padded (masked) last K chunk is much cheaper than a small k_chunk (k=256 on a 16640-token shard
            # cost 53.6% -> 46.9% FPU util), so keep 1024 unless the shard itself is shorter.
            k_chunk = 1024 if not kv_local or kv_local >= 1024 else max(128, 1 << (kv_local.bit_length() - 1))
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),  # last column = CCL workers
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=os.environ.get("MIMO_SDPA_EXP_APPROX", "0") == "1",
        ring_k_split=k_split,
    )


def default_k_split(window, chunk_local, kv_actual, sp, k_chunk=1024):
    """K split for GA (full-attention) layers: 2 at >= 2048 tokens per chip, 3 below (2048 tok/chip: 512 units ->
    1024 on 100 cores; 640: 160 -> 480), only when every partition gets K chunks from the prefix in every ring
    iteration (the op requires a valid chunk per partition). ``MIMO_SDPA_KSPLIT`` = N forces N, 0 / 1 turns it off."""
    env = os.environ.get("MIMO_SDPA_KSPLIT")
    if window is not None:
        return 1
    s = int(env) if env else (2 if chunk_local >= 2048 else 3)
    if s <= 1:
        return 1
    # every ring iteration's shard holds kv_actual / sp prefix tokens in whole chunk rounds: need s K chunks of them
    return s if kv_actual // sp >= s * k_chunk else 1


def compute_config(mesh_device, fidelity=None):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=fidelity or _fidelity(),
        math_approx_mode=False,
        fp32_dest_acc_en=False,  # sinks require the streaming (non-fp32-dest) compute path
        packer_l1_acc=False,
    )


def gather_seq_len(window, k_chunk, full_seq):
    if window is None:
        return full_seq
    return max(math.ceil((window - 1) / k_chunk) * k_chunk, TILE)


def ring_attention(
    tt_q,
    kv_cache,
    *,
    kv_actual,
    logical_n,
    window,
    sink,
    layer_slot,
    mesh_device,
    ccl_manager,
    sp_axis,
    scale,
    program_config=None,
    compute_kernel_config=None,
    k_split=1,
    merge=True,
):
    """q [1, nq_local, S_local, 192] (block-cyclic chunk, K/V already written at ``kv_actual``) ->
    [1, nq_local, S_local, v_dim]. ``k_split`` > 1 (GA only): every (head, Q chunk) is split over k_split key
    partitions on separate cores (even work over the grid), merged here by merge_k_split."""
    assert kv_cache.k.dtype == ttnn.bfloat8_b and kv_cache.v.dtype == ttnn.bfloat8_b
    sp = mesh_device.shape[sp_axis]
    pc = program_config or ring_program_config(
        mesh_device,
        sliding=window is not None,
        q_local=tt_q.shape[2],
        kv_local=kv_cache.max_seq_len // sp,
        k_split=k_split,
    )
    ckc = compute_kernel_config or compute_config(mesh_device)
    tp = mesh_device.shape[1 - sp_axis]
    n_kv = kv_cache.n_kv_local * tp
    bufseq = gather_seq_len(window, pc.k_chunk_size, kv_cache.max_seq_len)
    out, _, stats = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        kv_cache.k,
        kv_cache.v,
        None,
        None,
        None,
        persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(
            f"mimo_k_{bufseq}", n_kv, bufseq, kv_cache.k_dim, ttnn.bfloat8_b
        ),
        persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(
            f"mimo_v_{bufseq}", n_kv, bufseq, kv_cache.v_dim, ttnn.bfloat8_b
        ),
        joint_strategy="rear",
        logical_n=logical_n,
        program_config=pc,
        compute_kernel_config=ckc,
        dim=2,
        multi_device_global_semaphore=ccl_manager.ring_attention_ccl_semaphore_handles,
        num_links=ccl_manager.num_links,
        cluster_axis=sp_axis,
        mesh_device=mesh_device,
        topology=ccl_manager.topology,
        ccl_core_grid_offset=ccl_manager.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=scale,
        is_balanced=False,
        kv_cache_batch_idx=layer_slot,
        kv_actual_isl=kv_actual,
        attention_sink=sink,
        sliding_window_size=window,
    )
    if pc.ring_k_split > 1 and not merge:  # raw partitions (timing / debugging)
        return out, stats
    if pc.ring_k_split > 1 and os.environ.get("MIMO_KSPLIT_NOMERGE_DEBUG") == "1":  # hang isolation only (wrong values)
        part0 = ttnn.slice(out, [0, 0, 0, 0], [1, out.shape[1] // pc.ring_k_split, out.shape[2], out.shape[3]])
        out.deallocate(True)
        stats.deallocate(True)
        return part0
    if pc.ring_k_split > 1:
        merged = merge_k_split(out, stats, k_split=pc.ring_k_split, scale=scale)
        out.deallocate(True)
        stats.deallocate(True)
        return merged
    stats.deallocate(True)
    return out


def merge_k_split(o, stats, *, k_split, scale):
    """Combine the K-split partitions of ring_joint_sdpa: o [1, k_split * NH, S, DV] unnormalized partial outputs
    (virtual head p * NH + h), stats [1, k_split * NH, 2 S', 32] their running max m (rows [0, S), column 0) and
    sum l (rows [S', S' + S), the sum of the 32 per-column partials), with o_p = sum_j exp(scale (s_j - m_p)) v_j,
    l_p = sum_j exp(scale (s_j - m_p)). Exact merge: M = max_p m_p, a_p = exp(scale (m_p - M)),
    O = sum_p a_p o_p / sum_p a_p l_p -- one fused program (tt/kernels/sdpa_merge, fp32 DEST and intermediates)."""
    return _KSplitMerge.get(o.device(), k_split)(o, stats, scale)


class _KSplitMerge:
    """Fused K-split merge program over all worker cores; work item = (head, 32-row tile)."""

    KDIR = "models/demos/mimo_v2_d_p/tt/kernels/sdpa_merge"
    _cache = {}

    @classmethod
    def get(cls, mesh_device, k_split):
        key = (id(mesh_device), k_split)
        if key not in cls._cache:
            cls._cache[key] = cls(mesh_device, k_split)
        return cls._cache[key]

    def __init__(self, mesh_device, k_split):
        self.dev, self.S = mesh_device, k_split
        g = mesh_device.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(x, y) for y in range(g.y) for x in range(g.x)]
        self._prog = {}

    def _program(self, o, stats, out, scale):
        import struct

        S, P = self.S, len(self.cores)
        nh, St, DVt = o.shape[1] // S, o.shape[2] // 32, o.shape[3] // 32
        halft = stats.shape[2] // 64
        assert 2 * S + 1 <= 8, "the coefficient session holds 2 S + 1 fp32 tiles in DEST (full sync: 8)"
        crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in self.cores]
        )
        drt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        items = nh * St
        for me, c in enumerate(self.cores):
            drt[c.x][c.y] = [o.buffer_address(), stats.buffer_address(), out.buffer_address(), me]
            crt[c.x][c.y] = [len(range(me, items, P))]

        def cb(i, tiles, fmt, page):
            return ttnn.CBDescriptor(
                total_size=tiles * page,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)],
            )

        bf, f32 = ttnn.bfloat16, ttnn.float32
        cbs = [cb(3, 1, bf, 2048), cb(25, S, f32, 4096)]
        DEPTH = int(os.environ.get("MIMO_MERGE_DEPTH", "3"))  # items buffered per lane
        for lane in (0, 1):  # per data movement lane: max, sum, O, out
            cbs += [
                cb(4 * lane, DEPTH * S, bf, 2048),
                cb(1 + 4 * lane, DEPTH * S, bf, 2048),
                cb(2 + 4 * lane, DEPTH * S * DVt, bf, 2048),
                cb(16 + lane, DEPTH * DVt, bf, 2048),
            ]
        dm = lambda lane: ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1 if lane else ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.NOC_1 if lane else ttnn.NOC.NOC_0,
        )
        dbg = [(d, "1") for d in os.environ.get("MIMO_MERGE_DEFINES", "").split(",") if d]  # e.g. MERGE_NO_DRAM
        kd = lambda src, ct, rt, config: ttnn.KernelDescriptor(
            kernel_source=f"{self.KDIR}/{src}",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=crs,
            compile_time_args=ct,
            runtime_args=rt,
            defines=dbg,
            config=config,
        )
        scale_bits = struct.unpack("<I", struct.pack("<f", scale))[0]
        # both RISCs read DRAM (one lane each): a single NOC0 reader tops out near 320 GB/s on these tile reads
        kernels = [kd("merge_dm.cpp", [S, nh, St, DVt, halft, P, lane], drt, dm(lane)) for lane in (0, 1)]
        kernels.append(
            kd(
                "merge_compute.cpp",
                [S, DVt, scale_bits],
                crt,
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
                ),
            )
        )
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)

    def __call__(self, o, stats, scale):
        nh = o.shape[1] // self.S
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, nh, o.shape[2], o.shape[3]]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.dev,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        key = (o.buffer_address(), stats.buffer_address(), out.buffer_address(), tuple(o.shape), scale)
        if key not in self._prog:
            if len(self._prog) > 32:
                self._prog.clear()
            self._prog[key] = self._program(o, stats, out, scale)
        ttnn.generic_op([o, stats, out], self._prog[key])
        return out
