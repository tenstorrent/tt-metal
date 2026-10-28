# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``QkvNormRope``: the decode per-head q/k/v RMSNorm (gamma on q and k, none on v) and the q/k RoPE as one program
on the height-sharded head tensors ``nlp_create_qkv_heads_decode`` produces.

``q`` ``[1, B, Hq, D]``, ``k`` / ``v`` ``[1, B, Hkv, D]`` (bf16, TILE, L1 HEIGHT_SHARDED: one user per core, shard
``[32, D]`` with the heads on the rows, padded to the tile), ``gq`` / ``gk`` the norm gammas ``[1, 1, 1, D]`` (bf16,
TILE, DRAM: one row-0 tile per column), ``cos`` / ``sin`` the position-gathered tables ``[1, 1, 32, D]`` (bf16,
TILE, DRAM: row ``u`` = user ``u``'s row, the ``ttnn.embedding`` output of the model's per-step gather, so the
per-step transpose / slice / head-repeat of the composite RoPE are not needed).
Outputs are written in place over the input shards by default (``GEMMA4_QKV_NORM_ROPE_INPLACE=1``: the op returns
``q, k, v`` themselves; the compute consumes every input column before it packs the first output column of that
tensor, see the kernel), or into fresh tensors in the inputs' sharded memory configs (``=0``). SDPA takes the q
output sharded and ``paged_update_cache`` takes the k / v outputs as before. In place keeps the attention's L1
peak at three live head shards per core: with six (inputs plus fresh outputs) the head_dim-512 global layers
at 32 users pushed the lowest live L1 buffer under the static circular-buffer region of the following
``paged_update_cache`` program ("statically allocated dataflow buffers clash with L1 buffers").

Per core (user) and tensor: the shard tiles are bound to circular buffers in place (no copy); compute squares
every column tile (FPU, fp32 dest), reduces the row sums (FPU reduce with a unit scaler), takes
``rsqrt(sum / D + eps)`` on the SFPU, scales the columns by it (SFPU column broadcast) and by gamma (row
broadcast) into a bf16 CB, then rotates with the FPU row-broadcast of the user's cos / sin row: ``out_a = n_a
cos_a - n_b sin_a``, ``out_b = n_b cos_b + n_a sin_b`` on fp32 dest, one bf16 pack (the norm and RoPE stages are
``custom_ops/kernels/norm_rope_stages.hpp``, shared with the heads and prefill forms). This replaces
``to_memory_config x3 (sharded -> DRAM), rms_norm x3, the q and k RoPE (7 programs each), to_memory_config x2
(k, v back to sharded)``: 21 programs -> 1, and the three head tensors never leave L1.
"""

import os
from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


class QkvNormRope:
    # pinned (tensor-backed) CBs
    CB_Q, CB_K, CB_V = 0, 1, 2
    CB_QO, CB_KO, CB_VO = 16, 17, 18
    # streamed inputs
    CB_GQ, CB_GK, CB_COS, CB_SIN = 3, 4, 5, 6
    CB_SCALER = 7
    # intermediates
    CB_SQ = 8  # fp32 x^2 tiles
    CB_INV = 9  # fp32 rsqrt column tile
    CB_N = 10  # bf16 normalized columns between the norm windows and the RoPE windows

    @staticmethod
    def golden(x: torch.Tensor, gamma, eps: float, cos=None, sin=None) -> torch.Tensor:
        """x [1,B,H,D]; gamma [D] or None; cos/sin [1,B,1,D] or None: HF per-head RMSNorm then NeoX RoPE."""
        xf = x.float()
        n = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        if gamma is not None:
            n = n * gamma.float()
        if cos is None:
            return n
        d = n.shape[-1]
        rot = torch.cat([-n[..., d // 2 :], n[..., : d // 2]], dim=-1)
        return n * cos.float() + rot * sin.float()

    @staticmethod
    def in_place() -> bool:
        """Outputs over the input shards (``GEMMA4_QKV_NORM_ROPE_INPLACE``, default 1) or fresh output tensors (0)."""
        return os.environ.get("GEMMA4_QKV_NORM_ROPE_INPLACE", "1") == "1"

    @classmethod
    def op(cls, q, k, v, gq, gk, cos, sin, eps: float, in_place: bool = None):
        """Returns ``(q_out, k_out, v_out)``; with ``in_place`` these are ``q, k, v`` themselves."""
        in_place = cls.in_place() if in_place is None else bool(in_place)
        B, D = int(q.shape[1]), int(q.shape[3])
        for name, t in (("q", q), ("k", k), ("v", v)):
            assert t.layout == ttnn.TILE_LAYOUT and t.dtype == ttnn.bfloat16, f"{name} must be bf16 TILE"
            assert t.is_sharded() and t.memory_config().buffer_type == ttnn.BufferType.L1, f"{name} must be L1 sharded"
            assert int(t.shape[1]) == B and int(t.shape[3]) == D and t.padded_shape[2] == TILE, f"{name}: {t.shape}"
            assert list(t.memory_config().shard_spec.shape) == [TILE, D], f"{name}: one user per core expected"
        assert D % (2 * TILE) == 0, f"head_dim {D} must be a multiple of 64"
        for name, t in (("gq", gq), ("gk", gk)):
            assert tuple(int(s) for s in t.shape) == (1, 1, 1, D) and t.layout == ttnn.TILE_LAYOUT, f"{name}: {t.shape}"
        for name, t in (("cos", cos), ("sin", sin)):
            ts = [int(s) for s in t.shape]
            assert ts[0] == 1 and ts[1] == 1 and ts[3] == D, f"{name}: {ts}"
            assert B <= ts[2] <= TILE and t.layout == ttnn.TILE_LAYOUT, f"{name}: one row tile of users"
        cols_t = D // TILE
        half_t = cols_t // 2
        device = q.device()
        if in_place:
            outs = [q, k, v]
        else:
            outs = [
                ttnn.allocate_tensor_on_device(t.shape, t.dtype, t.layout, device, t.memory_config()) for t in (q, k, v)
            ]
        q_grid = q.memory_config().shard_spec.grid
        assert k.memory_config().shard_spec.grid == q_grid and v.memory_config().shard_spec.grid == q_grid
        cores = []
        for rng in q_grid.ranges():
            for y in range(rng.start.y, rng.end.y + 1):
                for x in range(rng.start.x, rng.end.x + 1):
                    cores.append(ttnn.CoreCoord(x, y))
        assert len(cores) == B, f"{len(cores)} shard cores for {B} users"
        tile_bytes = TILE * TILE * 2
        f32_tile = TILE * TILE * 4

        def cb(idx, pages, page=tile_bytes, fmt=ttnn.bfloat16):
            return ttnn.CBDescriptor(
                total_size=pages * page,
                core_ranges=q_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
            )

        cbs = [
            ttnn.cb_descriptor_from_sharded_tensor(cls.CB_Q, q),
            ttnn.cb_descriptor_from_sharded_tensor(cls.CB_K, k),
            ttnn.cb_descriptor_from_sharded_tensor(cls.CB_V, v),
            ttnn.cb_descriptor_from_sharded_tensor(cls.CB_QO, outs[0]),
            ttnn.cb_descriptor_from_sharded_tensor(cls.CB_KO, outs[1]),
            ttnn.cb_descriptor_from_sharded_tensor(cls.CB_VO, outs[2]),
            cb(cls.CB_GQ, cols_t),
            cb(cls.CB_GK, cols_t),
            cb(cls.CB_COS, cols_t),
            cb(cls.CB_SIN, cols_t),
            cb(cls.CB_SCALER, 2),  # the scaler tile + a staging tile for the cos / sin row copies
            cb(cls.CB_SQ, cols_t, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_INV, 1, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_N, cols_t),  # the normalized q / k columns (bf16) between the norm and the RoPE stages
        ]
        reader_ct = [cls.CB_Q, cls.CB_K, cls.CB_V, cls.CB_GQ, cls.CB_GK, cls.CB_COS, cls.CB_SIN, cls.CB_SCALER, cols_t]
        for t in (gq, gk, cos, sin):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [cls.CB_QO, cls.CB_KO, cls.CB_VO, cols_t]
        import struct

        eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
        inv_d_bits = struct.unpack("<I", struct.pack("<f", 1.0 / D))[0]
        compute_ct = [
            cls.CB_Q, cls.CB_K, cls.CB_V, cls.CB_QO, cls.CB_KO, cls.CB_VO, cls.CB_GQ, cls.CB_GK, cls.CB_COS, cls.CB_SIN,
            cls.CB_SCALER, cls.CB_SQ, cls.CB_INV, cols_t, half_t, eps_bits, inv_d_bits, cls.CB_N,
        ]  # fmt: skip
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for u, core in enumerate(cores):
            reader_rt[core.x][core.y] = [
                gq.buffer_address(), gk.buffer_address(), cos.buffer_address(), sin.buffer_address(), u
            ]  # fmt: skip
            writer_rt[core.x][core.y] = [0]
            compute_rt[core.x][core.y] = [0]
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_qkv_norm_rope.cpp"),
                core_ranges=q_grid,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_qkv_norm_rope.cpp"),
                core_ranges=q_grid,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_qkv_norm_rope.cpp"),
                core_ranges=q_grid,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_cfg,
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ttnn.generic_op([q, k, v, gq, gk, cos, sin] + ([] if in_place else outs), program)
        return outs[0], outs[1], outs[2]
