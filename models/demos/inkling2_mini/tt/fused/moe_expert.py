# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Fused MoE expert step for Inkling-2-Mini decode (one Blackhole card): one program does, for the 8 routed
experts, gate_up matmul -> softcapped SwiGLU x routing weight -> down matmul -> sum over experts.

Drop-in for ``Inkling2MoE.experts(x, weights, indices)`` (tt/moe.py): same inputs (x [1,1,1,H] bf16 TILE in L1, the
top-8 weights [1,1,1,8] bf16 and ids [1,1,1,8] uint16 TILE from ``route``), same output ([1,1,1,H] bf16 TILE L1,
rows 1..31 zero), and it reads the SAME device weight tensors (``moe.w_gate_up`` [1,E,H,2I], ``moe.w_down``
[1,E,I,H], TILE interleaved DRAM, BFP4 or BFP8): no extra weight memory, no conversion.

Layout (72 workers = logical (0,0)-(8,7), plus one collector core (9,0)):
  worker L < 36  "gate core j=L": gate column j of every expert, then SwiGLU for hidden tile j of all 8 experts.
  worker L >= 36 "up core j=L-36": up column j (gate_up column 36+j); sends it to gate core j.
  every worker L: down output column L, K = 36 hidden tiles x 8 experts in one dest accumulation.
  The hidden of all experts (36 x 8 1x32 tiles, 18 KB) never leaves L1: gate cores -> collector -> multicast.
Matmuls use Blackhole ``custom_mm`` (1x32 in0 tiles, LoFi, fp32 dest). LoFi is exact for BFP4 weights x bf16 x;
for BFP8 it drops weight mantissa bits (see E12-report.md).
"""

import ttnn

HIDDEN, INTER, TOP_K = 2304, 1152, 8
KT_GU, NT_GU = HIDDEN // 32, 2 * INTER // 32  # 72, 72
KT_DN, NT_DN = INTER // 32, HIDDEN // 32  # 36, 72
GRID_X, GRID_Y = 9, 8  # 72 workers
N_GATE = INTER // 32  # 36
HID_TILES = KT_DN * TOP_K  # 288
KDIR = "models/demos/inkling2_mini/tt/fused/kernels"

CB_X, CB_W, CB_G, CB_U16, CB_WS16, CB_H16, CB_HID, CB_O, CB_Z, CB_SCR, CB_G32 = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
SEM_U, SEM_G, SEM_R = 0, 1, 2


def _crs(c0, c1=None):
    c1 = c1 or c0
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c0), ttnn.CoreCoord(*c1))])


class FusedMoeExperts:
    def __init__(self, device, w_gate_up, w_down, sbk=18, nbuf=3):
        assert KT_GU % sbk == 0 and HID_TILES % sbk == 0, sbk
        grid = device.compute_with_storage_grid_size()
        assert grid.x >= GRID_X + 1 and grid.y >= GRID_Y, grid
        assert w_gate_up.dtype == w_down.dtype, (w_gate_up.dtype, w_down.dtype)
        assert tuple(w_gate_up.padded_shape)[-2:] == (HIDDEN, 2 * INTER), w_gate_up.padded_shape
        assert tuple(w_down.padded_shape)[-2:] == (INTER, HIDDEN), w_down.padded_shape
        self.device, self.w_gate_up, self.w_down, self.sbk, self.nbuf = device, w_gate_up, w_down, sbk, nbuf
        self.workers = _crs((0, 0), (GRID_X - 1, GRID_Y - 1))
        self.collector = (GRID_X, 0)
        self.all_cores = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1)),
                _crs(self.collector).ranges()[0],
            ]
        )
        self.noc = {}
        for y in range(GRID_Y):
            for x in range(GRID_X):
                p = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
                self.noc[(x, y)] = (p.x, p.y)
        p = device.worker_core_from_logical_core(ttnn.CoreCoord(*self.collector))
        self.noc[self.collector] = (p.x, p.y)
        self.cbs = self._cbs(w_gate_up.dtype)
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
        )

    def _cbs(self, wdtype):
        tiny = ttnn.TileDescriptor(ttnn.Tile([1, 32]))
        bf = ttnn.bfloat16
        t32 = ttnn.TileDescriptor(ttnn.Tile([32, 32]))
        wtile = ttnn.Tile([32, 32]).get_tile_size(wdtype)

        def cb(cores, total, *fmts):
            return ttnn.CBDescriptor(
                total_size=total,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=d, page_size=ps, tile=t) for i, d, ps, t in fmts
                ],
            )

        w = self.workers
        return [
            cb(w, KT_GU * 64, (CB_X, bf, 64, tiny)),
            cb(w, self.nbuf * self.sbk * wtile, (CB_W, wdtype, wtile, ttnn.TileDescriptor(ttnn.Tile([32, 32])))),
            cb(w, 2048, (CB_G, bf, 64, tiny), (CB_G32, bf, 2048, t32)),
            cb(w, 2048, (CB_U16, bf, 2048, t32)),
            cb(w, 2048, (CB_WS16, bf, 2048, t32)),
            cb(w, 2048, (CB_H16, bf, 2048, t32)),
            cb(self.all_cores, HID_TILES * 64, (CB_HID, bf, 64, tiny)),
            cb(w, 64, (CB_O, bf, 64, tiny)),
            cb(w, 2048, (CB_Z, bf, 2048, ttnn.TileDescriptor(ttnn.Tile([32, 32])))),
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
            "cb_x": CB_X, "cb_w": CB_W, "cb_ws16": CB_WS16, "cb_scr": CB_SCR, "sbk": self.sbk, "nbuf": self.nbuf,
            "kt_gu": KT_GU, "nt_gu": NT_GU, "kt_dn": KT_DN, "nt_dn": NT_DN, "x_tiles": KT_GU, **common,
        }  # fmt: skip
        w_named = {
            "cb_g": CB_G, "cb_g32": CB_G32, "cb_u16": CB_U16, "cb_h16": CB_H16, "cb_hid": CB_HID, "cb_o": CB_O, "cb_z": CB_Z,
            "sem_u": SEM_U, "sem_g": SEM_G, "sem_r": SEM_R, "cb_x": CB_X, "cb_ws16": CB_WS16, **common,
        }  # fmt: skip
        c_named = {
            "cb_x": CB_X, "cb_w": CB_W, "cb_g": CB_G, "cb_g32": CB_G32, "cb_u16": CB_U16, "cb_ws16": CB_WS16,
            "cb_h16": CB_H16, "cb_hid": CB_HID, "cb_o": CB_O, "sbk": self.sbk, "kt_gu": KT_GU, **common,
        }  # fmt: skip
        r_args, w_args, c_args = [], [], []
        cx, cy = self.noc[self.collector]
        for y in range(GRID_Y):
            for x_ in range(GRID_X):
                L = y * GRID_X + x_
                core = ttnn.CoreCoord(x_, y)
                gate = L < N_GATE
                j = L if gate else L - N_GATE
                col_gu = j if gate else N_GATE + j
                px, py = self.noc[(j % GRID_X, j // GRID_X)]
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
        (x0, y0), (x1, y1) = self.noc[(0, 0)], self.noc[(GRID_X - 1, GRID_Y - 1)]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/moe_expert_reader.cpp",
                core_ranges=self.workers,
                compile_time_args=acc(x) + acc(weights) + acc(indices) + acc(self.w_gate_up) + acc(self.w_down),
                named_compile_time_args=list(r_named.items()),
                runtime_args=r_args,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/moe_expert_writer.cpp",
                core_ranges=self.workers,
                compile_time_args=acc(out) + (acc(debug) if debug is not None else []),
                defines=[("E12_DEBUG", "1")] if debug is not None else [],
                named_compile_time_args=list(w_named.items()),
                runtime_args=w_args,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/moe_expert_compute.cpp",
                core_ranges=self.workers,
                named_compile_time_args=list(c_named.items()),
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
                runtime_args=[(ttnn.CoreCoord(*self.collector), [x0, y0, x1, y1, GRID_X * GRID_Y])],
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]
        sems = [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=self.all_cores, initial_value=0) for i in (SEM_U, SEM_G, SEM_R)
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=self.cbs)
        io = [x, weights, indices, self.w_gate_up, self.w_down] + ([debug] if debug is not None else []) + [out]
        return ttnn.generic_op(io, prog)
