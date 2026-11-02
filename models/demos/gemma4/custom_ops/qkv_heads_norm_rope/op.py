# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``QkvHeadsNormRope``: the decode head split, the per-head q/k/v RMSNorm and the q/k RoPE straight from the fused
QKV projection output, as one program (the dense 31B / 12B head counts: 8 / 4 / 2 q heads and 4 / 2 / 1 kv heads
per chip at TP=4 / TP=8).

Input ``x`` is the projection output ``[1, 1, B, N]`` (bf16, TILE, **L1** interleaved, ``B <= 32`` users on one row
tile, ``N = (hq + 2 hkv) * D`` laid out ``[q heads | k heads | v heads]``), ``gq`` / ``gk`` the norm gammas
``[1, 1, 1, D]`` (bf16 TILE), ``cos`` / ``sin`` the position-gathered rows ``[1, 1, 32, D]`` (row ``u`` = user
``u``). Outputs ``q [1, B, hq, D]``, ``k`` / ``v [1, B, hkv, D]`` (bf16, TILE, L1 HEIGHT_SHARDED, one user per core,
shard ``[32, D]`` with the heads on the rows: the layout ``nlp_create_qkv_heads_decode`` emits, so SDPA and
``paged_update_cache`` take them unchanged).

One work unit per core = (user, tensor) when the grid holds 3 B cores (3 at batch 1, 96 at batch 32 on Blackhole),
else one user per core with its three tensors in sequence (32 cores at batch 32 on an 8x8 grid; ``grid.py``
plans it from the device grid), so a batch-1 step spreads the three tensors over three cores instead of one (the per-user form of ``qkv_norm_rope`` ran all three on one core,
~70 us vs ~46 us for the composite chain at batch 1). The reader gathers the user's row of every head's column
tiles of its tensor into head-row tiles; the compute is the v2 compute of ``custom_ops/qkv_norm_rope`` for one
tensor (``x^2`` on the FPU, FPU row reduce, ``rsqrt(sum / D + eps)`` on the SFPU, ``x * inv * gamma``, NeoX RoPE
on the FPU with the user's cos / sin row broadcast; v: norm only); the writer ships the tiles to the user's output
shard over the NoC. This replaces ``to_memory_config (DRAM -> L1), nlp_create_qkv_heads_decode`` + the
``qkv_norm_rope`` program (or the composite's ``to_memory_config x3, rms_norm x3, the q and k RoPE,
to_memory_config x2``): the head tensors are never materialized before the norm.
"""

import struct
from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid, core_range_set, plan_qkv_heads_units, row_major_cores

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


def heads_sharded_memory_config(device, batch: int, head_dim: int):
    """The L1 height-sharded config of the decode head tensors: one user per core (row-major core order, as
    ``nlp_create_qkv_heads_decode``), shard ``[32, head_dim]``."""
    gx, gy = compute_grid(device)
    core_grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(gx, gy), True)
    return ttnn.create_sharded_memory_config(
        shape=(TILE, head_dim),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class QkvHeadsNormRope:
    CB_X, CB_OUT = 0, 16
    CB_G, CB_COS, CB_SIN, CB_SCALER = 3, 5, 6, 7
    CB_SQ, CB_INV, CB_N = 8, 9, 10

    @staticmethod
    def golden(x: torch.Tensor, hq: int, hkv: int, head_dim: int, gq, gk, eps: float, cos, sin):
        """x [1, 1, B, N] -> (q [1, B, hq, D], k, v [1, B, hkv, D]): the head split, per-head RMSNorm (gamma on q
        and k) and NeoX RoPE on q and k (cos / sin [B, D] rows)."""
        B = x.shape[2]
        xf = x.float()
        q = xf[0, 0, :, : hq * head_dim].reshape(1, B, hq, head_dim)
        k = xf[0, 0, :, hq * head_dim : (hq + hkv) * head_dim].reshape(1, B, hkv, head_dim)
        v = xf[0, 0, :, (hq + hkv) * head_dim :].reshape(1, B, hkv, head_dim)

        def norm(t, g):
            n = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)
            return n * g.float() if g is not None else n

        def rope(t):
            c = cos[:B].float().reshape(1, B, 1, head_dim)
            s = sin[:B].float().reshape(1, B, 1, head_dim)
            rot = torch.cat([-t[..., head_dim // 2 :], t[..., : head_dim // 2]], dim=-1)
            return t * c + rot * s

        return rope(norm(q, gq)), rope(norm(k, gk)), norm(v, None)

    @classmethod
    def op(cls, x, hq: int, hkv: int, head_dim: int, gq, gk, cos, sin, eps: float):
        D = int(head_dim)
        assert (
            x.layout == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16 and not x.is_sharded()
        ), "x: bf16 TILE interleaved"
        assert x.memory_config().buffer_type == ttnn.BufferType.L1, "x must be in L1 (32-byte row reads)"
        shape = [int(v) for v in x.shape]
        assert len(shape) == 4 and shape[0] == 1 and shape[1] == 1 and 1 <= shape[2] <= TILE, f"x: {shape}"
        B, N = shape[2], shape[3]
        assert N == (hq + 2 * hkv) * D, f"N {N} != ({hq} + 2 * {hkv}) * {D}"
        assert D % (2 * TILE) == 0, f"head_dim {D} must be a multiple of 64"
        assert 1 <= hq <= TILE and 1 <= hkv <= TILE
        for name, t in (("gq", gq), ("gk", gk)):
            assert tuple(int(s) for s in t.shape) == (1, 1, 1, D) and t.layout == ttnn.TILE_LAYOUT, f"{name}: {t.shape}"
        for name, t in (("cos", cos), ("sin", sin)):
            ts = [int(s) for s in t.shape]
            assert ts[0] == 1 and ts[1] == 1 and ts[3] == D and B <= ts[2] <= TILE and t.layout == ttnn.TILE_LAYOUT, (
                name,
                ts,
            )
        cols_t = D // TILE
        half_t = cols_t // 2
        device = x.device()
        mem = heads_sharded_memory_config(device, B, D)
        outs = [
            ttnn.allocate_tensor_on_device(ttnn.Shape([1, B, h, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem)
            for h in (hq, hkv, hkv)
        ]
        grid = compute_grid(device)
        n_cores, per_core = plan_qkv_heads_units(grid, B)  # (user, tensor) per core, or a user's three tensors
        cores = row_major_cores(grid, n_cores)  # row-major, as the shards (shard core of user u = cores[u])
        core_set = core_range_set(cores)
        shard_noc = [device.worker_core_from_logical_core(cores[u]) for u in range(B)]
        tile_bytes = TILE * TILE * 2
        f32_tile = TILE * TILE * 4

        def cb(idx, pages, page=tile_bytes, fmt=ttnn.bfloat16):
            return ttnn.CBDescriptor(
                total_size=pages * page,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
            )

        cbs = [
            cb(cls.CB_X, cols_t),
            cb(cls.CB_OUT, cols_t),
            cb(cls.CB_G, cols_t),
            cb(cls.CB_COS, cols_t),
            cb(cls.CB_SIN, cols_t),
            cb(cls.CB_SCALER, 1 + 2 * cols_t),  # the scaler + the cos / sin staging pages
            cb(cls.CB_SQ, cols_t, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_INV, 1, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_N, cols_t),
        ]
        reader_ct = [cls.CB_X, cls.CB_G, cls.CB_COS, cls.CB_SIN, cls.CB_SCALER, cols_t, hq, hkv]
        for t in (x, gq, gk, cos, sin):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [cls.CB_OUT, cols_t]
        eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
        inv_d_bits = struct.unpack("<I", struct.pack("<f", 1.0 / D))[0]
        compute_ct = [
            cls.CB_X, cls.CB_OUT, cls.CB_G, cls.CB_COS, cls.CB_SIN, cls.CB_SCALER, cls.CB_SQ, cls.CB_INV, cols_t, half_t,
            eps_bits, inv_d_bits, cls.CB_N,
        ]  # fmt: skip
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        out_addrs = [o.buffer_address() for o in outs]
        for i, core in enumerate(cores):
            u, t0 = (i // 3, i % 3) if per_core == 1 else (i, 0)
            reader_rt[core.x][core.y] = [
                x.buffer_address(), gq.buffer_address(), gk.buffer_address(), cos.buffer_address(),
                sin.buffer_address(), u, t0, per_core,
            ]  # fmt: skip
            writer_rt[core.x][core.y] = [shard_noc[u].x, shard_noc[u].y, t0, per_core] + out_addrs
            compute_rt[core.x][core.y] = [t0, per_core]
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_qkv_heads_norm_rope.cpp"),
                core_ranges=core_set,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_qkv_heads_norm_rope.cpp"),
                core_ranges=core_set,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_qkv_heads_norm_rope.cpp"),
                core_ranges=core_set,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_cfg,
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ttnn.generic_op([x, gq, gk, cos, sin] + outs, program)
        return outs[0], outs[1], outs[2]
