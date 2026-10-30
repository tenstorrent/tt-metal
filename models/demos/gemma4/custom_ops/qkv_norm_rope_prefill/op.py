# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``QkvNormRopePrefill``: the prefill per-head q/k/v RMSNorm (gamma on q and k, none on v) and the q/k RoPE as one
program on the ``[B, H, S, D]`` head tensors ``nlp_create_qkv_heads`` produces (bf16, TILE, DRAM or L1 interleaved).

Replaces, per layer, ``rms_norm x3`` with their flatten / unflatten ``reshape`` pairs (not views: the heads dim is
tile-padded) and the two ``rotary_embedding`` programs: 11 programs -> 1. The work unit is one ``[32, D]`` tile row
of one head (32 consecutive positions); units are dealt round-robin over the compute grid, each core streams its
units' column tiles through L1 (reader -> compute -> writer, nothing stays sharded). Per unit the compute squares
every column tile (FPU, fp32 dest), reduces the row sums (FPU reduce with a unit scaler), takes
``rsqrt(sum / D + eps)`` on the SFPU, scales the columns by it (SFPU column broadcast) and by gamma (row broadcast
of the gamma tile's row 0), then rotates elementwise with the cos / sin tile rows of the same positions
(``out_a = n_a cos_a - n_b sin_a``, ``out_b = n_b cos_b + n_a sin_b``, HF full-width rotate_half; the partial-rotary
global layers carry identity padding in their caches, as the composite ``rotary_embedding`` relies on). v is
normed only. Numerics: HiFi4 + fp32 accumulation like the composite prefill norms (``fp32_accumulate=True``).

``cos`` / ``sin``: the layer's ``[1, 1, S_max, D]`` caches; ``row_tile_offset`` selects the first position tile
(chunked prefill at offset ``32 * row_tile_offset``). ``k`` / ``v`` may be None (KV-shared layers: q only).
Outputs are fresh tensors in the inputs' memory configs (``q_out, k_out, v_out``).
"""

import hashlib
import struct
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.custom_ops.grid import compute_grid
from models.demos.gemma4.tt.mesh_orientation import qkv_norm_rope_prefill_enabled

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


def plan_units(kind_heads, B, rt, cols_t, row_tile_offset, n_cores_available):
    """The work split: one unit per (kind, batch, head, 32-row tile), ``(kind, first tile of the row, first cos/sin
    tile)``, dealt round-robin over ``min(n_cores_available, units)`` cores in the grid's row-major core order. Pure:
    the device op and the host tests share it."""
    units = []
    for kind, H in kind_heads:
        for b in range(B):
            for h in range(H):
                for r in range(rt):
                    row = (b * H + h) * rt + r  # tile row of the [B*H*S/32, D/32] tile grid
                    units.append((kind, row * cols_t, (row_tile_offset + r) * cols_t))
    n_cores = min(int(n_cores_available), len(units))
    return n_cores, [units[i::n_cores] for i in range(n_cores)]


def program_cache_key(structural_hash: int, units_per_core) -> int:
    """A 64-bit program-cache key: generic_op's structural hash combined with each core's unit count (the per-core
    runtime-arg layout). Same structure and layout -> same key (cache hit, runtime args overridden in place); any
    core with a different unit count -> a different program."""
    h = hashlib.blake2b(digest_size=8)
    h.update(int(structural_hash).to_bytes(8, "little", signed=False))
    h.update(len(units_per_core).to_bytes(4, "little"))
    for n in units_per_core:
        h.update(int(n).to_bytes(4, "little"))
    return int.from_bytes(h.digest(), "little")


class QkvNormRopePrefill:
    CB_X, CB_OUT = 0, 16
    CB_GQ, CB_GK, CB_COS, CB_SIN, CB_SCALER = 1, 2, 3, 4, 5
    CB_SQ, CB_INV, CB_N = 6, 7, 8
    KIND_Q, KIND_K, KIND_V = 0, 1, 2

    @staticmethod
    def enabled() -> bool:
        """``GEMMA4_QKV_NORM_ROPE_PREFILL=1`` on Blackhole (opt-in; refused with a warning on Wormhole, where the
        kernels are not validated -- see mesh_orientation.qkv_norm_rope_prefill_enabled)."""
        return qkv_norm_rope_prefill_enabled(is_blackhole())

    @staticmethod
    def golden(x: torch.Tensor, gamma, eps: float, cos=None, sin=None) -> torch.Tensor:
        """x [B, H, S, D]; gamma [D] or None; cos / sin [1, 1, S, D] or None: HF per-head RMSNorm then RoPE."""
        xf = x.float()
        n = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        if gamma is not None:
            n = n * gamma.float()
        if cos is None:
            return n
        d = n.shape[-1]
        rot = torch.cat([-n[..., d // 2 :], n[..., : d // 2]], dim=-1)
        return n * cos.float() + rot * sin.float()

    @classmethod
    def op(cls, q, k, v, gq, gk, cos, sin, eps: float, row_tile_offset: int = 0, core_grid=None):
        device = q.device()
        B, Hq, S, D = (int(s) for s in q.shape)
        assert (
            q.layout == ttnn.TILE_LAYOUT and q.dtype == ttnn.bfloat16 and not q.is_sharded()
        ), "q: bf16 TILE interleaved"
        assert S % TILE == 0 and D % (2 * TILE) == 0, f"S {S} must be a multiple of 32, D {D} of 64"
        tensors = [(cls.KIND_Q, q)]
        for kind, t in ((cls.KIND_K, k), (cls.KIND_V, v)):
            if t is not None:
                ts = [int(s) for s in t.shape]
                assert ts[0] == B and ts[2] == S and ts[3] == D and t.layout == ttnn.TILE_LAYOUT, f"{ts}"
                assert t.dtype == ttnn.bfloat16 and not t.is_sharded()
                tensors.append((kind, t))
        for name, g in (("gq", gq), ("gk", gk)):
            assert tuple(int(s) for s in g.shape) == (1, 1, 1, D) and g.layout == ttnn.TILE_LAYOUT, f"{name}: {g.shape}"
        for name, t in (("cos", cos), ("sin", sin)):
            ts = [int(s) for s in t.shape]
            assert ts[0] == 1 and ts[1] == 1 and ts[3] == D and t.layout == ttnn.TILE_LAYOUT, f"{name}: {ts}"
            assert ts[2] >= TILE * row_tile_offset + S, f"{name}: {ts[2]} rows < offset {32 * row_tile_offset} + S {S}"
        cols_t, half_t, rt = D // TILE, D // (2 * TILE), S // TILE
        outs = {
            kind: ttnn.allocate_tensor_on_device(t.shape, t.dtype, t.layout, device, t.memory_config())
            for kind, t in tensors
        }
        gx, gy = (int(core_grid.x), int(core_grid.y)) if core_grid is not None else compute_grid(device)
        n_cores, per_core = plan_units(
            [(kind, int(t.shape[1])) for kind, t in tensors], B, rt, cols_t, row_tile_offset, gx * gy
        )
        all_cores = [ttnn.CoreCoord(x, y) for y in range(gy) for x in range(gx)]
        cores = all_cores[:n_cores]
        core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        tile_bytes, f32_tile = TILE * TILE * 2, TILE * TILE * 4

        def cb(idx, pages, page=tile_bytes, fmt=ttnn.bfloat16):
            return ttnn.CBDescriptor(
                total_size=pages * page,
                core_ranges=core_ranges,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
            )

        cbs = [
            cb(cls.CB_X, 2 * cols_t),  # double-buffered input row
            cb(cls.CB_OUT, 2 * cols_t),
            cb(cls.CB_GQ, cols_t),
            cb(cls.CB_GK, cols_t),
            cb(cls.CB_COS, 2 * cols_t),
            cb(cls.CB_SIN, 2 * cols_t),
            cb(cls.CB_SCALER, 1),
            cb(cls.CB_SQ, cols_t, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_INV, 1, page=f32_tile, fmt=ttnn.float32),
            cb(cls.CB_N, cols_t),
        ]
        kq, kk, kv = (outs.get(cls.KIND_Q), outs.get(cls.KIND_K), outs.get(cls.KIND_V))
        # absent k / v: the accessor args and addresses of q stand in (never dereferenced for a missing kind)
        tk = k if k is not None else q
        tv = v if v is not None else q
        ok, ov = (kk if kk is not None else kq), (kv if kv is not None else kq)
        reader_ct = [cls.CB_X, cls.CB_GQ, cls.CB_GK, cls.CB_COS, cls.CB_SIN, cls.CB_SCALER, cols_t]
        for t in (q, tk, tv, gq, gk, cos, sin):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [cls.CB_OUT, cols_t]
        for t in (kq, ok, ov):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        eps_bits = struct.unpack("<I", struct.pack("<f", float(eps)))[0]
        inv_d_bits = struct.unpack("<I", struct.pack("<f", 1.0 / D))[0]
        compute_ct = [
            cls.CB_X, cls.CB_OUT, cls.CB_GQ, cls.CB_GK, cls.CB_COS, cls.CB_SIN, cls.CB_SCALER, cls.CB_SQ, cls.CB_INV,
            cls.CB_N, cols_t, half_t, eps_bits, inv_d_bits,
        ]  # fmt: skip
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, cu in zip(cores, per_core):
            r_args = [
                q.buffer_address(), tk.buffer_address(), tv.buffer_address(), gq.buffer_address(),
                gk.buffer_address(), cos.buffer_address(), sin.buffer_address(), len(cu),
            ]  # fmt: skip
            w_args = [kq.buffer_address(), ok.buffer_address(), ov.buffer_address(), len(cu)]
            c_args = [len(cu)]
            for kind, tile0, rope0 in cu:
                r_args += [kind, tile0, rope0]
                w_args += [kind, tile0]
                c_args.append(kind)
            reader_rt[core.x][core.y] = r_args
            writer_rt[core.x][core.y] = w_args
            compute_rt[core.x][core.y] = c_args
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_qkv_norm_rope_prefill.cpp"),
                core_ranges=core_ranges,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_qkv_norm_rope_prefill.cpp"),
                core_ranges=core_ranges,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_qkv_norm_rope_prefill.cpp"),
                core_ranges=core_ranges,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_cfg,
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        # generic_op's structural hash covers kernel sources, compile-time args, core ranges and CBs but not the
        # per-core runtime-arg counts, and here every core carries a length-dependent number of units (3 / 2 / 1
        # args each for the reader / writer / compute). A later call on the same core set with more units per core
        # (a 2048-row prefill after a 1024-row one) would hit the cached program and override arguments it never
        # had ("Index 14 is larger than runtime args size 14"), so the cache key carries the unit layout too.
        program.custom_program_hash = program_cache_key(
            int(ttnn.compute_program_descriptor_hash(program)), [len(cu) for cu in per_core]
        )
        io = [q, tk, tv, gq, gk, cos, sin, kq, ok, ov]
        ttnn.generic_op(io, program)
        return kq, outs.get(cls.KIND_K), outs.get(cls.KIND_V)
