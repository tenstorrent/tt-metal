# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Fused MoE expert step for Inkling-2-Mini decode (one Blackhole card): one program does, for the 8 routed
experts, gate_up matmul -> softcapped SwiGLU x routing weight -> down matmul -> sum over experts.

Drop-in for ``Inkling2MoE.experts(x, weights, indices)`` (tt/moe.py): same inputs (x [1,1,1,H] bf16 TILE in L1, the
top-8 weights [1,1,1,8] bf16 and ids [1,1,1,8] uint16 TILE from ``route``), same output ([1,1,1,H] bf16 TILE L1,
rows 1..31 zero), and it reads the SAME device weight tensors (``moe.w_gate_up`` [1,E,H,2I], ``moe.w_down``
[1,E,I,H], TILE interleaved DRAM, BFP4 or BFP8): no extra weight memory, no conversion.

Layout (72 workers = the first 72 cores, row-major, of a rectangle (default logical (0,0)-(8,7)), plus one collector
core (default (9,0); it may be a spare core inside the rectangle). ``origin`` / ``shape`` / ``collector`` move it,
e.g. off the tensor prefetcher's receiver ring):
  worker L < 36  "gate core j=L": gate column j of every expert, then SwiGLU for hidden tile j of all 8 experts.
  worker L >= 36 "up core j=L-36": up column j (gate_up column 36+j); sends it to gate core j.
  every worker L: down output column L, K = 36 hidden tiles x 8 experts in one dest accumulation.
  The hidden of all experts (36 x 8 1x32 tiles, 18 KB) never leaves L1: gate cores -> collector -> multicast.
Matmuls (1x32 in0 tiles, fp32 dest): BFP4 weights use Blackhole ``custom_mm`` (LoFi, exact for BFP4 x bf16);
BFP8 weights use the standard ``matmul_block`` at ``bfp8_fidelity`` (default HiFi2, as tt/moe.py). gate_up and
down may differ in dtype (e.g. BFP4 gate_up + BFP8 down).
"""

import os

import torch

import ttnn

HIDDEN, INTER, TOP_K = 2304, 1152, 8
KT_GU, NT_GU = HIDDEN // 32, 2 * INTER // 32  # 72, 72
KT_DN, NT_DN = INTER // 32, HIDDEN // 32  # 36, 72
N_WORKERS = 72
N_GATE = INTER // 32  # 36
HID_TILES = KT_DN * TOP_K  # 288
KDIR = "models/demos/inkling2_mini/tt/fused/kernels"

CB_X, CB_W, CB_G, CB_U16, CB_WS16, CB_H16, CB_HID, CB_O, CB_Z, CB_SCR, CB_G32, CB_WD = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
)
SEM_U, SEM_G, SEM_R = 0, 1, 2


def _crs(c0, c1=None):
    c1 = c1 or c0
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c0), ttnn.CoreCoord(*c1))])


def stick_weights(device, w, dtype, n_banks=None):
    """Device-layout expert weight ``w`` [E, K, N] (torch; gate_up [E,H,2I] or down [E,I,H]) -> DRAM tensor in the
    stick layout: stick q = e * Nt + n (one output column of one expert, Kt tiles in K order) lies contiguous in DRAM
    bank q % n_banks, so a core reads it in a few long bursts instead of Kt scattered 576 B tiles.
    Same tiles, same bytes and size as the TILE tensor (whole tiles move; <= n_banks - 1 padding sticks)."""
    nb = n_banks or device.dram_grid_size().x
    E, K, N = w.shape
    kt, nt = K // 32, N // 32
    sticks = w.bfloat16().reshape(E, kt, 32, nt, 32).permute(0, 3, 1, 2, 4).reshape(E * nt, kt, 32, 32)
    q = sticks.shape[0]
    pad = (-q) % nb
    if pad:
        sticks = torch.cat([sticks, sticks.new_zeros(pad, kt, 32, 32)])
    # page p = nb * (m0 + i) + b with stick (q // nb, b): order (stick-in-bank, i, bank)
    pages = sticks.reshape(-1, nb, kt, 32, 32).permute(0, 2, 1, 3, 4).reshape(-1, 32)
    t = ttnn.from_torch(pages.reshape(1, 1, -1, 32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    return t


def _cores(origin, shape):
    """Row-major logical cores of the rectangle ``origin`` + ``shape``."""
    return [(origin[0] + i % shape[0], origin[1] + i // shape[0]) for i in range(shape[0] * shape[1])]


def _range_set(cores):
    """Core list -> CoreRangeSet of contiguous row runs (one range per run)."""
    runs = []
    for x, y in sorted(set(cores), key=lambda c: (c[1], c[0])):
        if runs and runs[-1][1] == y and runs[-1][2] == x - 1:
            runs[-1][2] = x
        else:
            runs.append([x, y, x])
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, y), ttnn.CoreCoord(b, y)) for a, y, b in runs])


class FusedMoeExperts:
    def __init__(
        self,
        device,
        w_gate_up,
        w_down,
        sbk=18,
        nbuf=3,
        sticks=False,
        origin=(0, 0),
        shape=(9, 8),
        collector=(9, 0),
        bfp8_fidelity=ttnn.MathFidelity.HiFi2,
    ):
        """``sticks``: w_gate_up / w_down come from ``stick_weights`` (faster reads); else the TILE tensors of
        tt/moe.py ([1,E,H,2I], [1,E,I,H]). Workers: first 72 cores of the ``origin``/``shape`` rectangle; the
        collector is ``collector`` (outside the rectangle, or one of its spare cores). ``bfp8_fidelity=None``
        runs BFP8 weights through LoFi custom_mm too (faster, less exact)."""
        assert KT_GU % sbk == 0 and HID_TILES % sbk == 0 and (not sticks or KT_DN % sbk == 0), sbk
        self.sticks, self.n_banks = sticks, device.dram_grid_size().x
        grid = device.compute_with_storage_grid_size()
        rect = _cores(origin, shape)
        assert len(rect) >= N_WORKERS, (origin, shape)
        assert all(x < grid.x and y < grid.y for x, y in rect + [tuple(collector)]), (rect[-1], collector, grid)
        collector = tuple(collector)
        assert collector not in rect[:N_WORKERS], collector
        if not sticks:
            assert tuple(w_gate_up.padded_shape)[-2:] == (HIDDEN, 2 * INTER), w_gate_up.padded_shape
            assert tuple(w_down.padded_shape)[-2:] == (INTER, HIDDEN), w_down.padded_shape
        self.device, self.w_gate_up, self.w_down, self.sbk, self.nbuf = device, w_gate_up, w_down, sbk, nbuf
        self.worker_list = rect[:N_WORKERS]
        self.workers = _range_set(self.worker_list)
        self.rect = rect
        self.collector = collector
        cb_cores = rect + ([] if collector in rect else [collector])
        self.all_cores = _range_set(cb_cores)
        self.noc = {}
        for c in cb_cores:
            p = device.worker_core_from_logical_core(ttnn.CoreCoord(*c))
            self.noc[c] = (p.x, p.y)
        self.mcast_dests = len(rect) - (1 if collector in rect else 0)
        bfp8 = ttnn.bfloat8_b
        self.std_gu = bfp8_fidelity is not None and w_gate_up.dtype == bfp8
        self.std_dn = bfp8_fidelity is not None and w_down.dtype == bfp8
        self.cbs = self._cbs(w_gate_up.dtype, w_down.dtype)
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=bfp8_fidelity if (self.std_gu or self.std_dn) else ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
        )

    def _cbs(self, gu_dtype, dn_dtype):
        tiny = ttnn.TileDescriptor(ttnn.Tile([1, 32]))
        bf = ttnn.bfloat16
        t32 = ttnn.TileDescriptor(ttnn.Tile([32, 32]))
        gtile = ttnn.Tile([32, 32]).get_tile_size(gu_dtype)
        dtile = ttnn.Tile([32, 32]).get_tile_size(dn_dtype)

        def cb(cores, total, *fmts):
            return ttnn.CBDescriptor(
                total_size=total,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=d, page_size=ps, tile=t) for i, d, ps, t in fmts
                ],
            )

        w = self.workers
        n = self.nbuf * self.sbk
        return [
            cb(w, KT_GU * 64, (CB_X, bf, 64, tiny)),
            cb(w, n * gtile, (CB_W, gu_dtype, gtile, t32)),
            cb(w, n * dtile, (CB_WD, dn_dtype, dtile, t32)),
            cb(w, 2048, (CB_G, bf, 64, tiny), (CB_G32, bf, 2048, t32)),
            cb(w, 2048, (CB_U16, bf, 2048, t32)),
            cb(w, 2048, (CB_WS16, bf, 2048, t32)),
            cb(w, 2048, (CB_H16, bf, 2048, t32)),
            cb(self.all_cores, HID_TILES * 64, (CB_HID, bf, 64, tiny)),
            cb(w, 64, (CB_O, bf, 64, tiny)),
            cb(w, 2048, (CB_Z, bf, 2048, t32)),
            cb(w, 64, (CB_SCR, ttnn.uint16, 64, tiny)),
        ]

    def __call__(self, x, weights, indices, debug=None):
        """x [1,1,1,H] bf16 (L1), weights / indices from Inkling2MoE.route -> [1,1,1,H] bf16 L1."""
        for t in (x, weights, indices):
            assert t.memory_config().buffer_type == ttnn.BufferType.L1, t.memory_config()
            assert t.layout == ttnn.TILE_LAYOUT
        assert x.dtype == ttnn.bfloat16 and weights.dtype == ttnn.bfloat16 and indices.dtype == ttnn.uint16
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, HIDDEN]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, ttnn.L1_MEMORY_CONFIG
        )
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        common = {"topk": TOP_K, "hid_tiles": HID_TILES}
        r_named = {
            "cb_x": CB_X, "cb_w": CB_W, "cb_wd": CB_WD, "cb_ws16": CB_WS16, "cb_scr": CB_SCR, "sbk": self.sbk, "nbuf": self.nbuf,
            "kt_gu": KT_GU, "nt_gu": NT_GU, "kt_dn": KT_DN, "nt_dn": NT_DN, "x_tiles": KT_GU, **common,
        }  # fmt: skip
        w_named = {
            "cb_g": CB_G, "cb_g32": CB_G32, "cb_u16": CB_U16, "cb_h16": CB_H16, "cb_hid": CB_HID, "cb_o": CB_O, "cb_z": CB_Z,
            "sem_u": SEM_U, "sem_g": SEM_G, "sem_r": SEM_R, "cb_x": CB_X, "cb_ws16": CB_WS16, **common,
        }  # fmt: skip
        c_named = {
            "cb_x": CB_X, "cb_w": CB_W, "cb_wd": CB_WD, "cb_g": CB_G, "cb_g32": CB_G32, "cb_u16": CB_U16, "cb_ws16": CB_WS16,
            "cb_h16": CB_H16, "cb_hid": CB_HID, "cb_o": CB_O, "sbk": self.sbk, "kt_gu": KT_GU, **common,
        }  # fmt: skip
        r_args, w_args, c_args = [], [], []
        cx, cy = self.noc[self.collector]
        for L, c in enumerate(self.worker_list):
            core = ttnn.CoreCoord(*c)
            gate = L < N_GATE
            j = L if gate else L - N_GATE
            col_gu = j if gate else N_GATE + j
            px, py = self.noc[self.worker_list[j]]
            r_args.append((core, [x.buffer_address(), weights.buffer_address(), indices.buffer_address(),
                                  self.w_gate_up.buffer_address(), self.w_down.buffer_address(),
                                  int(gate), col_gu, L]))  # fmt: skip
            w_args.append(
                (
                    core,
                    [int(gate), j, px, py, cx, cy, out.buffer_address(), L]
                    + ([debug.buffer_address()] if debug is not None else []),
                )
            )
            c_args.append((core, [int(gate)]))
        (x0, y0), (x1, y1) = self.noc[self.rect[0]], self.noc[self.rect[-1]]
        # shortcut: E12_PROBE=NO_MM|NO_READ (comma list) is a timing probe only (wrong output); drop it once tuned.
        probe = [(f"E12_PROBE_{p}", "1") for p in os.environ.get("E12_PROBE", "").split(",") if p]
        layout = [("E12_STICKS", "1")] if self.sticks else []
        r_named["n_banks"] = self.n_banks
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/moe_expert_reader.cpp",
                core_ranges=self.workers,
                compile_time_args=acc(x) + acc(weights) + acc(indices) + acc(self.w_gate_up) + acc(self.w_down),
                named_compile_time_args=list(r_named.items()),
                defines=probe + layout,
                runtime_args=r_args,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/moe_expert_writer.cpp",
                core_ranges=self.workers,
                compile_time_args=acc(out) + (acc(debug) if debug is not None else []),
                defines=([("E12_DEBUG", "1")] if debug is not None else []) + layout,
                named_compile_time_args=list(w_named.items()),
                runtime_args=w_args,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/moe_expert_compute.cpp",
                core_ranges=self.workers,
                named_compile_time_args=list(c_named.items()),
                defines=probe + [("E12_STD_GU", "1")] * self.std_gu + [("E12_STD_DN", "1")] * self.std_dn,
                runtime_args=c_args,
                config=self.compute_config,
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/moe_expert_collector.cpp",
                core_ranges=_crs(self.collector),
                named_compile_time_args=[
                    ("cb_hid", CB_HID),
                    ("sem_g", SEM_G),
                    ("sem_r", SEM_R),
                    ("n_senders", N_GATE),
                    ("hid_bytes", HID_TILES * 64),
                ],  # fmt: skip
                runtime_args=[(ttnn.CoreCoord(*self.collector), [x0, y0, x1, y1, self.mcast_dests])],
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]
        sems = [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=self.all_cores, initial_value=0) for i in (SEM_U, SEM_G, SEM_R)
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=self.cbs)
        io = [x, weights, indices, self.w_gate_up, self.w_down] + ([debug] if debug is not None else []) + [out]
        return ttnn.generic_op(io, prog)
