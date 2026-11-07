# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact ternary prefill linear: Y = X' W'^T with Bonsai's own ternary weights, no rounding of any weight.

W'[n, k] = t[k, n] * s[n, k // 128], t in {-1, 0, 1}, s fp16 (the PQ2_0 values, decode_linear3's records). The 2-bit
codes are matmul in1 tiles (Bfp2_b, exponent forced to 127: exactly -1/0/+1; any fidelity is exact on a 1-bit SrcA);
each 128-wide K block's product goes to its own DEST tile and the SFPU adds it, times the column's fp16 scale, into an
fp32 accumulator (kernels/tpre_compute.cpp). The weights are split by output tile column over all cores and stay in
L1; the activation rows are multicast to every core (kernels/tpre_reader.cpp).

X' is the rotated activation blockdiag(H_1024) D x (the GGUF folds the rotation into the weights)."""

import os

import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import KBLOCK, TILE, pack_records_inline

TILE_BYTES = 256  # a ternary tile of the decode records: 2-bit codes, no exponent section (exponent forced to 127)

KERNELS = "models/demos/blackhole/bonsai2/tt/tprefill/kernels/"
NB = 2
L1_BUDGET = 1300 * 1024


class TernaryPrefillLinear:
    """Reads the decode linear's own records (decode_linear3.pack_records_inline): no second copy of the weights."""

    def __init__(self, device, rec: ttnn.Tensor, ppr: int, K: int, N: int, glu=False):
        """glu: the records are (gate i, up i) pairs (decode_linear3.pack_glu_records, N = 2 x the output width); the
        output is up * silu(gate)."""
        self.device, self.rec, self.ppr, self.K, self.N, self.glu = device, rec, ppr, K, N, glu

    @classmethod
    def from_host(cls, device, t_kn, s_nb):
        rec, ppr = pack_records_inline(t_kn, s_nb)
        w = ttnn.from_torch(
            torch.from_numpy(rec.view(np.int32)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return cls(device, w, ppr, t_kn.shape[0], t_kn.shape[1])

    @classmethod
    def from_decode(cls, lin):
        """Share a TernaryDecodeLinear3's records (inline group16 scales; GLU pairs too)."""
        assert lin.inline_scales and lin.group16
        return cls(lin.w.device(), lin.w, lin.ppr, lin.K, lin.N, glu=getattr(lin, "glu", False))

    def __call__(self, x):
        """x: [..., T, K] bf16 TILE (T a multiple of 32) -> [..., T, N] bf16 TILE."""
        device, K, N = self.device, self.K, self.N
        lead = list(x.shape)[:-2]
        T = list(x.shape)[-2]
        assert list(x.shape)[-1] == K and T % TILE == 0, (x.shape, K)
        Kt, Nt, KB, Mt = K // TILE, N // TILE, K // KBLOCK, T // TILE
        G = -(-KB // 32)
        CH = next(c for c in (32, 16, 8, 4) if Kt % c == 0)
        glu = self.glu
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(lead + [T, N // 2 if glu else N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device
        )
        grid = device.compute_with_storage_grid_size()
        ncores = grid.x * grid.y
        unit = 2 if glu else 1  # a core takes whole (gate, up) pairs
        base, extra = divmod(Nt // unit, ncores)
        counts = [unit * (base + (1 if c < extra else 0)) for c in range(ncores)]
        n_max = max(counts)
        defs = [d for d in os.environ.get("BONSAI_TPRE_DEFINES", "").split(",") if d]
        if os.environ.get("BONSAI_TPRE_ZACC", "1") == "1":
            defs.append("ZACC")  # the P tiles cleared by ZEROACC (kernels/tpre_compute.cpp)
        nbat = 1 if "OVERLAP" in defs else NB  # columns per compute batch
        if glu:
            assert nbat == 2, "GLU pairs need two columns per batch"
        resident = n_max > nbat
        w_bytes = n_max * (Kt * TILE_BYTES + G * 2048)
        x_bytes = (2 * Kt if resident else 4 * CH) * 2048
        assert w_bytes + x_bytes < L1_BUDGET, f"K={K} N={N}: weights {w_bytes} + x {x_bytes} B per core"
        coords = [ttnn.CoreCoord(c % grid.x, c // grid.x) for c in range(ncores)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(coords[0], coords[-1])])
        v0 = device.worker_core_from_logical_core(coords[0])
        rects = _rects(device, grid, v0)
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        start = 0
        for c, (cc, n) in enumerate(zip(coords, counts)):
            rd[cc.x][cc.y] = [start, n, 1 if c == 0 else 0]
            wr[cc.x][cc.y] = [start, n]
            cp[cc.x][cc.y] = [n]
            start += n
        rd_c = [x.buffer_address(), self.rec.buffer_address(), self.ppr, Mt, v0.x, v0.y]
        rd_c += [ncores, len(rects)]
        for r in rects:
            rd_c += r
        wr_c = [out.buffer_address(), Nt // unit, Mt]
        dm = [("GLU_PAIRS", "1")] if glu else []
        rd_ct = [Kt, CH, G, nbat] + ttnn.TensorAccessorArgs(x).get_compile_time_args()
        rd_ct += ttnn.TensorAccessorArgs(self.rec).get_compile_time_args()
        wr_ct = ttnn.TensorAccessorArgs(out).get_compile_time_args()

        def cb(index, fmt, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "tpre_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    runtime_args=rd,
                    common_runtime_args=rd_c,
                    config=ttnn.WriterConfigDescriptor(),  # BRISC, NOC 0: the multicast rectangle is start -> end
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "tpre_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=wr_ct,
                    defines=dm,
                    runtime_args=wr,
                    common_runtime_args=wr_c,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "tpre_compute.cpp",
                    core_ranges=crs,
                    compile_time_args=[Kt, KB, CH, 1 if resident else 0, nbat],
                    defines=dm + [(d, "1") for d in defs],
                    runtime_args=cp,
                    common_runtime_args=[Mt],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, dst_full_sync_en=True
                    ),
                ),
            ],
            semaphores=[
                ttnn.SemaphoreDescriptor(id=0, core_ranges=crs, initial_value=0),
                ttnn.SemaphoreDescriptor(id=1, core_ranges=crs, initial_value=0),
            ],
            cbs=[
                cb(0, ttnn.bfloat16, 2048, 2 * Kt if resident else 4 * CH),
                cb(1, ttnn.bfloat8_b, TILE_BYTES, max(n_max, 1) * Kt),
                cb(3, ttnn.float32, 2048, max(n_max, 1) * G),
                cb(16, ttnn.bfloat16, 2048, 2 * NB),
            ],
        )
        return ttnn.generic_op([x, self.rec, out], program)


def _rects(device, grid, sender):
    """The core grid as multicast rectangles of adjacent virtual columns: [x0, y0, x1, y1, destinations excluding the
    sender] each."""
    vx = [device.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x for x in range(grid.x)]
    vy = [device.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y for y in range(grid.y)]
    assert vy == list(range(vy[0], vy[0] + grid.y)), vy
    runs, start = [], 0
    for i in range(1, grid.x + 1):
        if i == grid.x or vx[i] != vx[i - 1] + 1:
            runs.append((vx[start], vx[i - 1]))
            start = i
    out = []
    for x0, x1 in runs:
        n = (x1 - x0 + 1) * grid.y
        if x0 <= sender.x <= x1:
            n -= 1
        out.append([x0, vy[0], x1, vy[-1], n])
    return out


def reference(x_rot: torch.Tensor, t_kn: np.ndarray, s_nb: np.ndarray) -> torch.Tensor:
    """fp32 Y = X' W'^T computed block by block (x_rot [T, K])."""
    K, N = t_kn.shape
    x = x_rot.float()
    y = torch.zeros(x.shape[0], N)
    s = torch.from_numpy(s_nb.astype(np.float32))
    for kb in range(K // KBLOCK):
        tb = torch.from_numpy(t_kn[kb * KBLOCK : (kb + 1) * KBLOCK].astype(np.float32))
        y += (x[:, kb * KBLOCK : (kb + 1) * KBLOCK] @ tb) * s[:, kb]
    return y


def prefill_linear(lin, rec_host):
    """The prefill linear sharing a TernaryDecodeLinear3's records: the 2D shared-fold layout (linear6.py, scales rounded
    to 5 significant bits; ~10% faster than linear2.py at 4096 tokens), BONSAI_TPRE_CLASS=2: the exact 2D layout
    (linear2.py), BONSAI_TPRE_2D=0: the 1D one."""
    if os.environ.get("BONSAI_TPRE_2D", "1") == "1":
        from models.demos.blackhole.bonsai2.tt.tprefill.linear2 import TernaryPrefillLinear2

        if os.environ.get("BONSAI_TPRE_CLASS", "6") == "6":
            from models.demos.blackhole.bonsai2.tt.tprefill.linear6 import TernaryPrefillLinear6

            return TernaryPrefillLinear6.from_decode(lin, rec_host)

        return TernaryPrefillLinear2.from_decode(lin, rec_host)
    return TernaryPrefillLinear.from_decode(lin)
