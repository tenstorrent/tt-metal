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

Per core (user) the reader gathers the user's row of every head's column tiles into the head-row tiles of the
norm / RoPE compute of ``custom_ops/qkv_norm_rope`` (its v2 compute and writer kernels are reused unchanged):
``x^2`` on the FPU, FPU row reduce, ``rsqrt(sum / D + eps)`` on the SFPU, ``x * inv * gamma``, NeoX RoPE on the
FPU with the user's cos / sin row broadcast. This replaces ``to_memory_config (DRAM -> L1), nlp_create_qkv_heads_
decode`` + the ``qkv_norm_rope`` program (or the composite's ``to_memory_config x3, rms_norm x3, the q and k RoPE,
to_memory_config x2``): the head tensors are never materialized before the norm.
"""

import struct
from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
QKV_KERNEL_DIR = Path(__file__).resolve().parent.parent / "qkv_norm_rope" / "kernels"
TILE = 32


def heads_sharded_memory_config(device, batch: int, head_dim: int):
    """The L1 height-sharded config of the decode head tensors: one user per core (row-major core order, as
    ``nlp_create_qkv_heads_decode``), shard ``[32, head_dim]``."""
    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(batch, grid, True)
    return ttnn.create_sharded_memory_config(
        shape=(TILE, head_dim),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class QkvHeadsNormRope:
    # reader-filled head tiles
    CB_Q, CB_K, CB_V = 0, 1, 2
    # tensor-backed output shards
    CB_QO, CB_KO, CB_VO = 16, 17, 18
    # streamed inputs
    CB_GQ, CB_GK, CB_COS, CB_SIN = 3, 4, 5, 6
    CB_SCALER = 7
    # intermediates
    CB_SQ = 8
    CB_INV = 9
    CB_N = 10

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
        grid = mem.shard_spec.grid
        cores = []
        for rng in grid.ranges():
            for y in range(rng.start.y, rng.end.y + 1):
                for xx in range(rng.start.x, rng.end.x + 1):
                    cores.append(ttnn.CoreCoord(xx, y))
        assert len(cores) == B, f"{len(cores)} shard cores for {B} users"
        tile_bytes = TILE * TILE * 2
        f32_tile = TILE * TILE * 4

        def cb(idx, pages, page=tile_bytes, fmt=ttnn.bfloat16):
            return ttnn.CBDescriptor(
                total_size=pages * page,
                core_ranges=grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
            )

        cbs = [
            cb(cls.CB_Q, cols_t),
            cb(cls.CB_K, cols_t),
            cb(cls.CB_V, cols_t),
            ttnn.cb_descriptor_from_sharded_tensor(cls.CB_QO, outs[0]),
            ttnn.cb_descriptor_from_sharded_tensor(cls.CB_KO, outs[1]),
            ttnn.cb_descriptor_from_sharded_tensor(cls.CB_VO, outs[2]),
            cb(cls.CB_GQ, cols_t),
            cb(cls.CB_GK, cols_t),
            cb(cls.CB_COS, cols_t),
            cb(cls.CB_SIN, cols_t),
            cb(cls.CB_SCALER, 2),
            cb(cls.CB_SQ, cols_t, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_INV, 1, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_N, cols_t),  # the v2 compute: bf16 normalized columns between the norm and the RoPE stage
        ]
        reader_ct = [cls.CB_Q, cls.CB_K, cls.CB_V, cls.CB_GQ, cls.CB_GK, cls.CB_COS, cls.CB_SIN, cls.CB_SCALER,
                     cols_t, hq, hkv]  # fmt: skip
        for t in (x, gq, gk, cos, sin):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [cls.CB_QO, cls.CB_KO, cls.CB_VO, cols_t]
        eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
        inv_d_bits = struct.unpack("<I", struct.pack("<f", 1.0 / D))[0]
        compute_ct = [
            cls.CB_Q, cls.CB_K, cls.CB_V, cls.CB_QO, cls.CB_KO, cls.CB_VO, cls.CB_GQ, cls.CB_GK, cls.CB_COS, cls.CB_SIN,
            cls.CB_SCALER, cls.CB_SQ, cls.CB_INV, cols_t, half_t, eps_bits, inv_d_bits, cls.CB_N,
        ]  # fmt: skip
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for u, core in enumerate(cores):
            reader_rt[core.x][core.y] = [
                x.buffer_address(), gq.buffer_address(), gk.buffer_address(), cos.buffer_address(),
                sin.buffer_address(), u,
            ]  # fmt: skip
            writer_rt[core.x][core.y] = [0]
            compute_rt[core.x][core.y] = [0]
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_qkv_heads_norm_rope.cpp"),
                core_ranges=grid,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(QKV_KERNEL_DIR / "writer_qkv_norm_rope.cpp"),
                core_ranges=grid,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(QKV_KERNEL_DIR / "compute_qkv_norm_rope_v2.cpp"),
                core_ranges=grid,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_cfg,
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ttnn.generic_op([x, gq, gk, cos, sin] + outs, program)
        return outs[0], outs[1], outs[2]
