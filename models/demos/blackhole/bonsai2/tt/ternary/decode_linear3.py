# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ternary decode linear v3: y[n] = sum_kb s[n, kb] * sum_{k in kb} x'[k] t[k, n] on one activation row.

Weights in Bonsai's own values: t as 2-bit Bfp2_b tile data with a forced exponent (pack.py), one record per output
tile column (its K tiles in order, 256 B each; a multiple of 4 KB for every Bonsai K), stored as 4-KB pages so a
core's consecutive pages rotate over the DRAM banks. Scales: s as IEEE fp16, one [1, 32] tile (64 B) per (output
tile column, kb), the 32 columns in order. Compute: kernels/tdec3_compute.cpp.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.core_ranges import available, core_xy, row_major_crs, virtual_rects
from models.demos.blackhole.bonsai2.tt.ternary.pack import TILE, pack_ternary_tiles

KBLOCK = 128
PAGE_BYTES = 4096
KERNELS = "models/demos/blackhole/bonsai2/tt/ternary/kernels/"


def pack_weights(t_kn: np.ndarray) -> np.ndarray:
    """t_kn [K, N] int8 in {-1, 0, 1} -> [N/32 * K/512, 1024] uint32: record nt = tiles (0..Kt-1, nt)."""
    K, N = t_kn.shape
    Kt, Nt = K // TILE, N // TILE
    assert Kt % 16 == 0, K
    tiles = t_kn.reshape(Kt, TILE, Nt, TILE).transpose(2, 0, 1, 3)  # [Nt, Kt, 32, 32]
    return pack_ternary_tiles(tiles).reshape(-1, PAGE_BYTES // 4)


def pack_scales(s_nb: np.ndarray, fp32: bool) -> np.ndarray:
    """s_nb [N, KB] float16 -> [N/32, KB * 32] uint16 (or float32): record nt = for each kb the 32 columns' scales."""
    N, KB = s_nb.shape
    r = np.ascontiguousarray(s_nb.reshape(N // TILE, TILE, KB).transpose(0, 2, 1)).reshape(N // TILE, KB * TILE)
    return r.astype(np.float32) if fp32 else r.view(np.uint16)


def pack_glu_records(rec_gate: np.ndarray, rec_up: np.ndarray, ppr: int) -> np.ndarray:
    """Interleave two inline record sets (same K, N) per output tile: gate tile i, up tile i, ... (GLU_PAIRS)."""
    nt = rec_gate.shape[0] // ppr
    g = rec_gate.reshape(nt, ppr * rec_gate.shape[1])
    u = rec_up.reshape(nt, ppr * rec_up.shape[1])
    return np.ascontiguousarray(np.stack([g, u], 1).reshape(2 * nt * ppr, rec_gate.shape[1]))


def pack_scales_group16(s_nb: np.ndarray) -> np.ndarray:
    """s_nb [N, KB] float16 -> [N/32, G * 1024] uint16, G = ceil(KB/32) fp16 32x32 tiles per output tile column (face
    order): kb 32g + j of group g at tile row j; rows past KB zero."""
    N, KB = s_nb.shape
    Nt, G = N // TILE, -(-KB // 32)
    t = np.zeros((Nt, G, TILE, TILE), dtype=np.float16)
    for kb in range(KB):
        g, j = divmod(kb, 32)
        t[:, g, j, :] = s_nb[:, kb].reshape(Nt, TILE)
    faces = t.reshape(Nt, G, 2, 16, 2, 16).transpose(0, 1, 2, 4, 3, 5)  # [Nt, G, fr, fc, 16, 16]
    return np.ascontiguousarray(faces).reshape(Nt, G * TILE * TILE).view(np.uint16)


def pack_records_inline(t_kn: np.ndarray, s_nb: np.ndarray) -> tuple[np.ndarray, int]:
    """Weights with each group's scale tile inline: record nt = for each group g of 32 kbs, its fp16 scale tile (2 KB,
    pack_scales_group16 layout) then its kbs' ternary tiles (256 B each); zero padding to a multiple of 4 KB.
    Returns ([N/32 * pages_per_record, 1024] uint32, pages_per_record)."""
    K, N = t_kn.shape
    Kt, Nt, KB = K // TILE, N // TILE, K // KBLOCK
    G = -(-KB // 32)
    tiles = t_kn.reshape(Kt, TILE, Nt, TILE).transpose(2, 0, 1, 3)
    w = pack_ternary_tiles(tiles)  # [Nt, Kt, 64]
    sc = pack_scales_group16(s_nb).view(np.uint32).reshape(Nt, G, 512)
    parts = []
    for g in range(G):
        parts.append(sc[:, g])
        k0, k1 = g * 128, min(Kt, (g + 1) * 128)
        parts.append(w[:, k0:k1].reshape(Nt, -1))
    rec = np.concatenate(parts, axis=1)
    pad = -rec.shape[1] % (PAGE_BYTES // 4)
    rec = np.concatenate([rec, np.zeros((Nt, pad), dtype=np.uint32)], axis=1)
    return rec.reshape(-1, PAGE_BYTES // 4), rec.shape[1] * 4 // PAGE_BYTES


def reference(x_rot: torch.Tensor, t_kn: np.ndarray, s_nb: np.ndarray) -> torch.Tensor:
    K, N = t_kn.shape
    xb = x_rot.float().reshape(K // KBLOCK, KBLOCK)
    tb = torch.from_numpy(t_kn).float().reshape(K // KBLOCK, KBLOCK, N)
    p = torch.einsum("bk,bkn->bn", xb, tb)
    return (p * torch.from_numpy(s_nb.astype(np.float32)).T).sum(0)


# Both RISCs stream alternate 4-KB pages (inline group16 records). Off: no faster than one RISC per record
# (17408 x 4160: 68.9 vs 67.6 us; the stream reaches 356 GB/s either way), and the gate/up GLU pairs built on it
# were slower (162.7 us vs 2 x 75).
PAGE_SPLIT = os.environ.get("BONSAI_PAGE_SPLIT", "0") == "1"
COMMON_ARGS = os.environ.get("BONSAI_COMMON_ARGS", "1") == "1"  # one set of runtime args for all cores
BATCH4 = os.environ.get("BONSAI_BATCH4", "1") == "1"  # 4 kbs per SFPU pass: 17408 x 5120 89.7 -> 78.2 us


_XL1 = {}


def _xl1_buffer(device, K, crs, cores):
    """A fused rotation's x' buffer (one per K and core count, shared by every layer): per linear core K bf16, then
    the flag (16 B), the hub's counter (16 B) and its "1" word; zeroed once, the reader clears the flag after use."""
    key = (K, cores)
    if key not in _XL1:
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, [1, K + 256], ttnn.ShardOrientation.ROW_MAJOR),
        )
        init = torch.zeros(cores, K + 256, dtype=torch.int16)
        init[:, K + 16] = 1  # the hub's "1" word (2 K + 32 bytes in): the source of its flag multicast, set up front
        _XL1[key] = ttnn.from_torch(
            init.view(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=mc,
        )
    return _XL1[key]


STAGE_INFLIGHT = int(os.environ.get("BONSAI_STAGE_INFLIGHT", "1"))  # staging reads in flight per RISC


@dataclass
class TernaryDecodeLinear3:
    K: int
    N: int
    w: ttnn.Tensor
    s: ttnn.Tensor
    cores: int

    scale_fp32: bool = False
    group16: bool = False
    inline_scales: bool = False
    ppr: int = 0
    glu: bool = False

    @classmethod
    def from_packed(cls, device, rec: np.ndarray, ppr: int, K: int, N: int, cores=None):
        """Upload records made by pack_records_inline (host packing and device upload can happen apart)."""
        w = ttnn.from_torch(
            torch.from_numpy(rec.view(np.int32)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        grid = device.compute_with_storage_grid_size()
        lin = cls(K, N, w, None, cores or grid.x * grid.y, False, True, True, ppr)
        lin.rec_host = rec  # for stream_pages
        return lin

    @classmethod
    def from_host(cls, device, t_kn, s_nb, cores=None, scale_fp32=False, group16=False, inline_scales=False):
        if inline_scales:
            rec, ppr = pack_records_inline(t_kn, s_nb)
            w = ttnn.from_torch(
                torch.from_numpy(rec.view(np.int32)),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            grid = device.compute_with_storage_grid_size()
            return cls(t_kn.shape[0], t_kn.shape[1], w, None, cores or grid.x * grid.y, False, True, True, ppr)
        w = ttnn.from_torch(
            torch.from_numpy(pack_weights(t_kn).view(np.int32)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sp = pack_scales_group16(s_nb) if group16 else pack_scales(s_nb, scale_fp32)
        s = ttnn.from_torch(
            torch.from_numpy(sp if scale_fp32 else sp.view(np.int16)),
            dtype=ttnn.float32 if scale_fp32 else ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        grid = device.compute_with_storage_grid_size()
        return cls(t_kn.shape[0], t_kn.shape[1], w, s, cores or grid.x * grid.y, scale_fp32, group16)

    def __call__(
        self,
        x_rot,
        inflight=int(os.environ.get("BONSAI_TDEC_INFLIGHT", "3")),  # reads in flight per RISC: 3 beat 8 by ~4% (traced)
        sfpu_iter=10,
        dbg=False,
        sfpu_reinit=True,
        extra_defines=(),
        dm_defines=(),
        lead=(1, 1, TILE),
        split=0,
        residual=None,
        shapes=None,
        qg_heads=0,
        rot=None,
        out_l1=False,
        stage=None,
        x_local=False,
        stream=None,
        host_pf=None,
        kvp=None,
    ):
        """stage: this linear's weights already copied into L1 by stage_kernels(stage) in an earlier op.
        stream: its weights from the weight stream's ring (bonsai2/tt/stream); streamed linears take it by
        themselves while the stream is active.
        host_pf: a GDN step's input prefetch (GDNStep.prefetch_spec) for this op's writer to run first.
        kvp: the decode SDPA's K / V prefetch (stream/decode.py kv_prefetch) for this op's writer to run first.
        split > 0: return two tensors, output columns [0, split) and [split, N) (split a multiple of 32).
        x_rot: [C, K] bf16 ROW_MAJOR, C copies of the rotated activation (one per DRAM bank: page c is in bank
        c mod 8) -> [32, N] bf16 TILE, row 0 = y.
        residual: a tensor of the output's shape added before the bf16 rounding (out = residual + y).
        shapes: the output shape(s) instead of lead + [N] (one tile row: e.g. head-major [1, heads, 1, D], whose pages
        are the token-major ones). qg_heads: per head [q | gate] columns -> (q [1, heads, 1, D], gate [1, 1, heads D]).
        """
        device = (x_rot if x_rot is not None else rot[1]).device()
        K, N = self.K, self.N
        if stream is None and getattr(self, "streamed", False):
            from models.demos.blackhole.bonsai2.tt.stream import decode as stream_decode

            stream = stream_decode.stream()
        if getattr(self, "page_split", PAGE_SPLIT) and self.inline_scales and self.group16:
            # Both RISCs stream alternate 4-KB pages (kernels/tdec_stream.hpp stream_pages).
            dm_defines = list(dm_defines) + [("PAGE_SPLIT", "1")]
            extra_defines = list(extra_defines) + [("PAGE_SPLIT", "1")]
        if BATCH4 and self.inline_scales and self.group16 and K % 512 == 0:
            # 4 kbs per SFPU pass (kernels/tdec3_compute.cpp scale_acc4)
            extra_defines = list(extra_defines) + [("BATCH4", "1")]
        glu = getattr(self, "glu", False)
        if rot is not None:
            assert COMMON_ARGS and self.inline_scales and self.group16
            dm_defines = list(dm_defines) + [("X_HUB", "1")]  # x' from the fused rotation (tdec_stream.hpp XHub)
        elif x_local:
            # x_rot is a fused rotation's hub buffer (last_xl1) from the previous op, which ran on (a superset of)
            # this op's cores: each core copies x' from its own L1
            dm_defines = list(dm_defines) + [("X_PAGED", "1"), ("X_LOCAL", "1")]
        elif x_rot is not None and list(x_rot.shape) == [K // 1024, 1024]:
            dm_defines = list(dm_defines) + [("X_PAGED", "1")]  # x' as 2-KB pages in L1 (rotate.py XPAGED)
        if residual is not None:
            assert COMMON_ARGS and not glu and not split and self.inline_scales and self.group16
            dm_defines = list(dm_defines) + [("RESIDUAL", "1")]
            extra_defines = list(extra_defines) + [("RESIDUAL", "1")]
        if glu:
            # (gate, up) record pairs -> silu(gate) * up, one output tile per pair (pack_glu_records).
            assert self.inline_scales and self.group16 and not split, "GLU needs inline group16 records"
            dm_defines = list(dm_defines) + [("GLU_PAIRS", "1")]
            extra_defines = list(extra_defines) + [("GLU_PAIRS", "1")]
        Kt, Nt, KB = K // TILE, N // TILE, K // KBLOCK
        ppr = self.ppr if self.inline_scales else Kt * 256 // PAGE_BYTES
        # lead: the output's leading logical dims; any of them tile-pad to one 32-row tile row, so the pages match.
        if qg_heads:
            D = N // qg_heads // 2
            shapes = ([1, qg_heads, 1, D], [1, 1, qg_heads * D])
            dm_defines = list(dm_defines) + [("QG_HT", str(D // TILE))]
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(
                list(shapes[0]) if shapes else list(lead) + [(split or (N // 2 if glu else N)) * (3 if dbg else 1)]
            ),
            ttnn.float32 if dbg else ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.L1_MEMORY_CONFIG if out_l1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        out2 = (
            ttnn.allocate_tensor_on_device(
                ttnn.Shape(list(shapes[1]) if shapes else list(lead) + [N - split]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                device,
                ttnn.L1_MEMORY_CONFIG if out_l1 else ttnn.DRAM_MEMORY_CONFIG,
            )
            if split or qg_heads
            else None
        )
        grid = device.compute_with_storage_grid_size()
        unit = 2 if glu else 1  # a core takes whole (gate, up) pairs
        # the fewest cores with the same per-core maximum: fewer concurrent weight streams contend less in DRAM
        # (traced, 130 -> 80 cores: 17408x5120 78.4 -> 68.7 us, 6144x5120 30.4 -> 26.9; GLU 5120x17408 at 109 cores:
        # 125.8 -> 119.2)
        cores, spans = self._spans()
        coords = [core_xy(c, grid.x) for c in range(cores)]
        crs = row_major_crs(cores, grid.x)  # rectangles: one range per core costs ~50 us per launch
        max_n = max(n for _, n in spans)
        tiny = ttnn.TileDescriptor(1, 32, False)

        def cb(index, fmt, page, pages, tile=None):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page, tile=tile)
                ],
            )

        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        if not COMMON_ARGS:  # per-core args (COMMON_ARGS replaces them below)
            for cc, (s, n) in zip(coords, spans):
                s_addr = 0 if self.inline_scales else self.s.buffer_address()
                rd[cc.x][cc.y] = [
                    x_rot.buffer_address(),
                    s_addr,
                    self.w.buffer_address(),
                    s,
                    n,
                    (cc.x + cc.y * grid.x) % x_rot.shape[0],
                ]
                wr[cc.x][cc.y] = [self.w.buffer_address(), out.buffer_address(), s, n, 0] + (
                    [out2.buffer_address()] if split else []
                )
                cp[cc.x][cc.y] = [n]
        rd_c, wr_c, cp_c = [], [], []
        if COMMON_ARGS:
            # Same args on every core (kernels/core_span.hpp): the dispatcher writes them once, not once per core.
            s_addr = 0 if self.inline_scales else self.s.buffer_address()
            span = [Nt, cores, grid.x, unit]
            if rot is not None:
                xl1 = _xl1_buffer(device, K, crs, cores)
                self.last_xl1 = xl1  # x' stays there for the next op on these cores (x_local=)
                rd_c = [xl1.buffer_address(), s_addr, self.w.buffer_address()] + span + [1]
            else:
                rd_c = [x_rot.buffer_address(), s_addr, self.w.buffer_address()] + span + [x_rot.shape[0]]
            if residual is not None:
                rd_c.append(residual.buffer_address())  # tdec3_reader.cpp RESIDUAL
            if rot is not None:
                rects = virtual_rects(device, crs)
                rd_c += [rot[0].B, len(rects)] + [v for r in rects for v in r]  # tdec3_reader.cpp X_HUB
            wr_c = (
                [self.w.buffer_address(), out.buffer_address()]
                + span
                + [out2.buffer_address() if out2 is not None else 0]
            )
            cp_c = span
            rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            dm_defines = list(dm_defines) + [("COMMON_ARGS", "1")]
            extra_defines = list(extra_defines) + [("COMMON_ARGS", "1")]
        if stream is not None:
            # weights from the weight-stream ring (bonsai2/tt/stream; this linear's pages added to its Plan)
            assert COMMON_ARGS and self.inline_scales and not getattr(self, "page_split", PAGE_SPLIT)
            sd = [("W_STREAM", "1"), ("STREAM_ARG", str(len(rd_c)))] + stream.defines()
            dm_defines = list(dm_defines) + sd
            extra_defines = list(extra_defines) + stream.defines()
            rd_c = rd_c + stream.consumer_args()
        if stage is not None:
            assert COMMON_ARGS
            dm_defines = list(dm_defines) + [
                ("W_STAGED", "1"),
                ("STAGE_ARG_RD", str(len(rd_c))),
                ("STAGE_ARG_WR", str(len(wr_c))),
            ]
            rd_c = rd_c + [stage.buffer_address()]
            wr_c = wr_c + [stage.buffer_address()]
        rd_ct = [Kt, KB, ppr, inflight]
        for t in (x_rot if x_rot is not None else self.w, self.w if self.inline_scales else self.s, self.w):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        if residual is not None:
            rd_ct += ttnn.TensorAccessorArgs(residual).get_compile_time_args()
        wr_ct = [ppr, inflight] + ttnn.TensorAccessorArgs(self.w).get_compile_time_args()
        wr_ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
        if split or qg_heads:
            assert split % TILE == 0, split
            wr_ct += ttnn.TensorAccessorArgs(out2).get_compile_time_args()
        pf_tensors = []
        if host_pf is not None:  # the next GDN step's input prefetch on this op's writer (tdec3_writer.cpp)
            assert COMMON_ARGS
            pf_common, pf_ct, pf_tensors, _ = host_pf
            dm_defines = list(dm_defines) + [
                ("STEP_PREFETCH", "1"),
                ("PF_CT", str(len(wr_ct))),
                ("PF_ARG", str(len(wr_c))),
            ]
            wr_ct = wr_ct + pf_ct
            wr_c = wr_c + pf_common
        if kvp is not None:  # the decode SDPA's K / V prefetch on this op's writer (sdpa_dec/kernels/kv_prefetch.hpp)
            assert COMMON_ARGS
            kv_common, kv_ct, kv_tensors = kvp
            dm_defines = list(dm_defines) + [
                ("KV_PREFETCH", "1"),
                ("KVP_CT", str(len(wr_ct))),
                ("KVP_ARG", str(len(wr_c))),
            ]
            wr_ct = wr_ct + kv_ct
            wr_c = wr_c + kv_common
            pf_tensors = list(pf_tensors) + list(kv_tensors)
        rkern, rcbs, rins, rsems = [], [], [], []
        if rot is not None:
            rotation, rx, ru, rsig = rot
            spare = (
                available(grid.x * grid.y) - cores
            )  # a NormRotation spreads its blocks over these; a DecodeRotation needs B
            assert spare > 0 and (spare >= rotation.B or hasattr(rotation, "eps")), (cores, rotation.B)
            hub = device.worker_core_from_logical_core(core_xy(0, grid.x))
            rkern, rcbs, rins, rsems = rotation.fused_parts(
                rx, ru, rsig, cores, grid.x, (hub.x, hub.y), xl1.buffer_address(), spare=spare
            )
        # records per weight CB: two for the DRAM streams (a record starts at a half: no read wraps); one from the
        # weight-stream ring (its reader fills both CBs, a record per CB at a time; leaves L1 to the ring)
        wrec = int(os.environ.get("STREAM_WREC", "1")) if stream is not None else 2
        program = ttnn.ProgramDescriptor(
            kernels=rkern
            + [
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "tdec3_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    defines=list(dm_defines)
                    + ([("INLINE_SCALES", "1")] if self.inline_scales else [])
                    + [("S_TILE_BYTES", "128" if self.scale_fp32 else "64")]
                    + ([("S_RECORD_BYTES", str(-(-KB // 32) * 2048))] if self.group16 else []),
                    runtime_args=rd,
                    common_runtime_args=rd_c,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "tdec3_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=wr_ct,
                    defines=list(dm_defines)
                    + ([("INLINE_SCALES", "1")] if self.inline_scales else [])
                    + [("OUT_PER_NT", "3" if dbg else "1"), ("SPLIT_NT", str(split // TILE))]
                    + [("S_PER_NT", str(-(-KB // 32) if self.group16 else KB))],
                    runtime_args=wr,
                    common_runtime_args=wr_c,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "tdec3_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[Kt, KB],
                    defines=[("SFPU_ITER", str(sfpu_iter)), ("SCALE_FP32", "1" if self.scale_fp32 else "0")]
                    + [("SFPU_REINIT", "1" if sfpu_reinit else "0")]
                    + ([("SCALE_GROUP16", "1")] if self.group16 else [])
                    + ([("INLINE_SCALES", "1"), ("RECORD_PAGES", str(ppr * 16))] if self.inline_scales else [])
                    + list(extra_defines)
                    + ([("DBG_FIRST_KB", "1")] if dbg else []),
                    runtime_args=cp,
                    common_runtime_args=cp_c,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, dst_full_sync_en=True
                    ),
                ),
            ],
            semaphores=rsems,
            cbs=rcbs
            + [
                cb(0, ttnn.bfloat16, 64, Kt, tiny),
                cb(1, ttnn.bfloat8_b, 256, wrec * ppr * 16),
                cb(2, ttnn.float32, 2048, max_n * -(-KB // 32))
                if self.group16
                else cb(2, ttnn.float32, 128 if self.scale_fp32 else 64, max_n * KB, tiny),
                cb(6, ttnn.bfloat8_b, 256, wrec * ppr * 16),
            ]
            + ([cb(3, ttnn.bfloat16, 2048, max_n)] if residual is not None else [])
            + [
                cb(16, ttnn.float32 if dbg else ttnn.bfloat16, 4096 if dbg else 2048, max_n * (3 if dbg else 1)),
            ],
        )
        res = ttnn.generic_op(
            (rins + [xl1] if rot is not None else [x_rot])
            + ([] if self.inline_scales else [self.s])
            + ([residual] if residual is not None else [])
            + list(pf_tensors)
            + [self.w, out]
            + ([out2] if out2 is not None else []),
            program,
        )
        return (out, out2) if out2 is not None else res  # generic_op returns the last tensor of its list

    def _spans(self):
        """(cores, [(first output tile, tiles)] per core): __call__'s split of the output tiles over its cores."""
        unit = 2 if getattr(self, "glu", False) else 1
        Nt = self.N // TILE
        per = -(-(Nt // unit) // min(self.cores, Nt // unit))
        cores = -(-(Nt // unit) // per)
        if getattr(self, "spread", False):  # every core (weight stream: each sender's receivers all take part)
            cores = min(self.cores, Nt // unit)
        split = getattr(self, "split", None)  # the weight stream's sender-balanced split (stream.Stream.span_split)
        if split is not None and len(split) == cores:
            return cores, [(unit * split[c][0], unit * split[c][1]) for c in range(cores)]
        base, extra = divmod(Nt // unit, cores)
        spans, start = [], 0
        for c in range(cores):
            n = unit * (base + (1 if c < extra else 0))
            spans.append((start, n))
            start += n
        return cores, spans

    def _cores(self, device):
        """(cores, max output tiles per core) of __call__'s split (unit 1)."""
        Nt = self.N // TILE
        per = -(-Nt // min(self.cores, Nt))
        if getattr(self, "spread", False):
            return min(self.cores, Nt), per
        return -(-Nt // per), per

    def alloc_stage(self, device):
        """An L1 buffer for stage_kernels: each of this linear's cores holds its own weight pages at one address."""
        assert not getattr(self, "glu", False) and self.inline_scales
        cores, per = self._cores(device)
        grid = device.compute_with_storage_grid_size()
        rows = per * self.ppr  # 4-KB pages per core
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(row_major_crs(cores, grid.x), [rows, 1024], ttnn.ShardOrientation.ROW_MAJOR),
        )
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([cores * rows, 1024]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, mc
        )

    def stage_kernels(self, device, stage):
        """Kernels for another op's program that copy this linear's weight pages into stage (alloc_stage) on this
        linear's cores (kernels/tdec3_stage.cpp, both data-movement RISCs); __call__(stage=stage) then reads them
        from L1. The other op must not use these cores."""
        cores, _ = self._cores(device)
        grid = device.compute_with_storage_grid_size()
        crs = row_major_crs(cores, grid.x)
        common = [self.w.buffer_address(), stage.buffer_address(), self.N // TILE, cores, grid.x, 1]
        ct = ttnn.TensorAccessorArgs(self.w).get_compile_time_args()
        return crs, [
            ttnn.KernelDescriptor(
                kernel_source=KERNELS + "tdec3_stage.cpp",
                core_ranges=crs,
                compile_time_args=[self.ppr, parity, STAGE_INFLIGHT] + ct,
                runtime_args=ttnn.RuntimeArgs(),
                common_runtime_args=common,
                config=ttnn.ReaderConfigDescriptor() if parity == 0 else ttnn.WriterConfigDescriptor(),
            )
            for parity in (0, 1)
        ]

    def stream_pages(self, device):
        """{receiver index: this linear's 4-KB record pages it reads} (for stream.Plan.add), as __call__ splits it."""
        rec = np.asarray(self.rec_host).reshape(-1, 1024)
        cores, spans = self._spans()
        return {c: rec[f * self.ppr : (f + n) * self.ppr] for c, (f, n) in enumerate(spans)}

    def rotated(
        self,
        rotation,
        x,
        lead=(1, 1, TILE),
        u=None,
        sig=False,
        residual=None,
        out_l1=False,
        split=0,
        stage=None,
        qg_heads=0,
        host_pf=None,
        kvp=None,
    ):
        """This linear on rotation(x) (rotation(x, u=u, sig=sig)) in one op: the rotation runs on the cores after this
        linear's and hands x' over through core 0 (no rotation op, no x' round trip; the weight stream starts at
        once). x (and u) should be in L1: the rotation's reads then do not queue behind the weight streams."""
        return self(
            None,
            lead=lead,
            rot=(rotation, x, u, sig),
            residual=residual,
            out_l1=out_l1,
            split=split,
            stage=stage,
            qg_heads=qg_heads,
            host_pf=host_pf,
            kvp=kvp,
        )
