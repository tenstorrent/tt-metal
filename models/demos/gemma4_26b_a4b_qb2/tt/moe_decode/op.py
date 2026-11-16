# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 routed-expert MoE decode (WS4; kernels in kernels/, overview in README.md).

Two programs per call, both reading only the k routed experts' weights:

* gate|up: one core per (routing slot s, column pair j): reads x row 0, the slot's expert id and routing weight from
  the packed routing row (``RouterTopk.op_packed``: ``[1, 1, 32, 2k]`` bf16, row 0 = weights then ids), streams that
  expert's gate column j and up column j (2 * kt tiles, contiguous), and writes ``h = gelu(x Wg_j) * (x Wu_j) * w_s``
  as one 1x32 tile of the h buffer.
* down: one core per output column: concatenates the k experts' h rows (k * kts tiles) and their down tiles of the
  column, so one custom_mm over K = k * kts gives ``sum_s w_s * h_s Wd_s``; writes row 0 of the output tile.

Weights are DRAM width-sharded over the banks (layouts below) so cores read large contiguous chunks and every bank
serves the same bytes for any routing. Per chip, tensor-parallel experts: ``I_chip`` intermediate
columns (a multiple of 32) of every expert.
"""

import struct
from pathlib import Path

import torch

import ttnn

TILE = 32
TINY_BYTES = 2 * TILE
KERNELS = Path(__file__).resolve().parent / "kernels"
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.bfloat16: 2048}


def bank_split(n, banks):
    """``n`` tiles over ``banks``: per bank ``(count, first)``; the first ``n % banks`` banks take one more."""
    base, extra = divmod(n, banks)
    counts = [base + (b < extra) for b in range(banks)]
    firsts = [sum(counts[:b]) for b in range(banks)]
    return list(zip(counts, firsts))


def _bank_columns(blocks):
    """Per-bank ``[rows_b, 32]`` blocks -> ``[max rows, banks * 32]`` (shorter banks zero-padded at the end)."""
    rows = max(b.shape[0] for b in blocks)
    return torch.cat([torch.nn.functional.pad(b, (0, 0, 0, rows - b.shape[0])) for b in blocks], dim=1)


def gate_up_local_layout(gate, up, banks, readers):
    """``gate`` / ``up`` ``[E, H, I]`` -> ``[rows, banks * 32]`` for the bank-local gate|up: column block b holds
    bank b's K slice (``bank_split(H / 32, banks)``) of every (expert, reader) as one chunk
    ``[g_j, u_j, g_j+1, u_j+1, ...][k]`` (reader r owns pairs ``[r * pr, (r + 1) * pr)``, pr = pairs / readers)."""
    E, H, inter = gate.shape
    kt, pairs = H // TILE, inter // TILE
    assert pairs % readers == 0
    pr = pairs // readers
    g, u = (w.reshape(E, kt, TILE, pairs, TILE).permute(0, 3, 1, 2, 4) for w in (gate, up))  # [E, pairs, kt, 32, 32]
    both = torch.stack((g, u), dim=2).reshape(E, readers, pr * 2, kt, TILE, TILE)  # [E, readers, n, kt, 32, 32]
    blocks = [both[:, :, :, k0 : k0 + ks].reshape(-1, TILE) for ks, k0 in bank_split(kt, banks)]
    return _bank_columns(blocks)


def down_layout(down, banks, bias=None, bias_row=None):
    """``down`` ``[E, I, H]`` -> ``[rows, banks * 32]``: column block b holds bank b's output columns
    (``bank_split(H / 32, banks)``) of every expert as ``[expert][column][k]`` tiles. ``bias`` ``[E, H]`` goes into
    row ``bias_row`` of each expert (a zero-padding row of I; the op then adds ``sum_s w_s * bias_s``)."""
    if bias is not None:
        assert bias_row is not None and not down[:, bias_row].any(), "bias_row must be a zero-padding row"
        down = down.clone()
        down[:, bias_row] = bias.to(down.dtype)
    E, inter, H = down.shape
    kts, cols = inter // TILE, H // TILE
    tiles = down.reshape(E, kts, TILE, cols, TILE).permute(0, 3, 1, 2, 4)  # [E, cols, kts, 32, 32]
    return _bank_columns([tiles[:, c0 : c0 + cpb].reshape(-1, TILE) for cpb, c0 in bank_split(cols, banks)])


def upload_per_chip(mesh, per_chip, dtype):
    """One ``[rows, banks * 32]`` layout per chip -> a mesh tensor ``[1, 1, rows, banks * 32]``, DRAM width-sharded
    with one ``[rows, 32]`` shard per bank."""
    banks = mesh.dram_grid_size().x
    rows = per_chip[0].shape[0]
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            (rows, TILE),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    return ttnn.from_torch(
        torch.stack(per_chip)[:, None],
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=memory,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def gate_up_bias_layout(bias_gate, bias_up):
    """``[E, I]`` gate / up biases -> ``[E, I / 32 * 64]``: per pair ``[gate 32 | up 32]``."""
    E, inter = bias_gate.shape
    return torch.stack(
        (bias_gate.reshape(E, inter // TILE, TILE), bias_up.reshape(E, inter // TILE, TILE)), dim=2
    ).reshape(E, -1)


def upload_rows_per_chip(mesh, per_chip):
    """One ``[E, W]`` bf16 row table per chip -> ``[1, 1, E, W]`` ROW_MAJOR DRAM (one page per row)."""
    return ttnn.from_torch(
        torch.stack(per_chip)[:, None],
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def tp_expert_layouts(state, *, tp, banks, inter_pad, local_readers=3):
    """Checkpoint experts (``experts.gate_up_proj [E, 2I, H]``, ``experts.down_proj [E, H, I]``) -> per chip, the
    tensor-parallel slice of the intermediate dimension (I padded to ``inter_pad``, split ``tp`` ways) in the
    bank-sharded gate|up and down layouts."""
    gate, up = state["experts.gate_up_proj"].chunk(2, dim=-2)
    pad = inter_pad - gate.shape[-2]
    gate, up = (torch.nn.functional.pad(t.transpose(-2, -1), (0, pad)) for t in (gate, up))  # [E, H, Ip]
    down = torch.nn.functional.pad(state["experts.down_proj"].transpose(-2, -1), (0, 0, 0, pad))  # [E, Ip, H]
    gates, ups, downs = gate.chunk(tp, dim=-1), up.chunk(tp, dim=-1), down.chunk(tp, dim=-2)
    gu = [gate_up_local_layout(g.contiguous(), u.contiguous(), banks, local_readers) for g, u in zip(gates, ups)]
    dn = [down_layout(d.contiguous(), banks) for d in downs]
    return gu, dn


class MoeDecodeB1:
    def __init__(
        self,
        mesh,
        *,
        hidden,
        inter_chip,
        k,
        weight_dtype=ttnn.bfloat8_b,
        down_dtype=None,
        grid=None,
        activation="geglu",
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
    ):
        """activation: "geglu" (Gemma: gelu(g) * u) or "swiglu" (gpt-oss: (clamp(u, -L, L) + 1) * g' sigmoid(alpha
        g'), g' = min(g, L))."""
        assert hidden % TILE == 0 and inter_chip % TILE == 0 and 1 <= k <= 32
        assert activation in ("geglu", "swiglu")
        self.compute_defines = []
        if activation == "swiglu":
            bits = lambda v: f"0x{struct.unpack('<I', struct.pack('<f', v))[0]:08x}u"  # noqa: E731
            self.compute_defines = [
                ("SWIGLU", "1"),
                ("SWIGLU_LIMIT", bits(swiglu_limit)),
                ("SWIGLU_NEG_LIMIT", bits(-swiglu_limit)),
                ("SWIGLU_ALPHA", bits(swiglu_alpha)),
                ("SWIGLU_INV_ALPHA", bits(1.0 / swiglu_alpha)),
            ]
        self.mesh, self.hidden, self.inter = mesh, hidden, inter_chip
        self.kt, self.pairs, self.kts, self.cols, self.k = (
            hidden // TILE,
            inter_chip // TILE,
            inter_chip // TILE,
            hidden // TILE,
            k,
        )
        self.dtype, self.tile_bytes = weight_dtype, TILE_BYTES[weight_dtype]
        self.down_dtype = down_dtype or weight_dtype
        self.down_tile_bytes = TILE_BYTES[self.down_dtype]
        g = mesh.compute_with_storage_grid_size()
        self.gx, self.gy = grid or (int(g.x), int(g.y))
        assert self.cols <= self.gx * self.gy
        self.route_page = 2 * k * 2
        self.banks = mesh.dram_grid_size().x
        self.local_readers = 3

    @staticmethod
    def _coalesce(cores):
        """Core list -> CoreRangeSet merged into rectangles: dispatch writes kernel binaries per range, and per-core
        ranges made the multi-core programs slow to launch under trace."""
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[0])])
        for c in cores[1:]:
            core_set = core_set.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))
        return core_set

    def _cores(self, n):
        cores = [ttnn.CoreCoord(i % self.gx, i // self.gx) for i in range(n)]
        return cores, self._coalesce(cores)

    @staticmethod
    def _cb(index, dtype, page, pages, cores, tiny=False):
        kwargs = dict(buffer_index=index, data_format=dtype, page_size=page)
        if tiny:
            kwargs["tile"] = ttnn.TileDescriptor(ttnn.Tile([1, TILE]))
        return ttnn.CBDescriptor(
            total_size=page * pages, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(**kwargs)]
        )

    @staticmethod
    def _kernels(name, cores, reader, writer, compute, compute_rt=None, defines=None, compute_defines=None):
        defines = list(defines or [])
        dm = ttnn.DataMovementConfigDescriptor
        return [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / f"{name}_reader.cpp"),
                core_ranges=cores,
                compile_time_args=reader[0],
                runtime_args=reader[1],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / f"{name}_writer.cpp"),
                core_ranges=cores,
                compile_time_args=writer[0],
                runtime_args=writer[1],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / f"{name}_compute.cpp"),
                core_ranges=cores,
                compile_time_args=compute,
                runtime_args=compute_rt if compute_rt is not None else [],
                defines=list(compute_defines or []),
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]

    def gate_up_local(self, x, route, w_gul, readers=3, bias=None):
        """Bank-local gate|up (kernels gu_local_*.cpp): ``readers`` cores on each DRAM bank's row read only that
        bank (its K slice of their columns of the routed experts) and push fp32 partial (g, u) tiles to the reducer
        core owning each (slot, pair) unit, which sums the banks and applies the activation and the routing weight.
        A reducer owns several units when the units outnumber the free cores."""
        from models.demos.gpt_oss.tt.fused_decode.stream import row_placement

        device = x.device()
        h = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.k * self.pairs, TILE]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
        assert self.pairs % readers == 0
        nr = 2 * self.pairs // readers
        split = bank_split(self.kt, self.banks)
        max_ks = max(ks for ks, _ in split)
        placed = row_placement(self.mesh, readers)
        reader_cores = [(placed[b][r], b, r) for b in range(self.banks) for r in range(readers)]
        used = {(c.x, c.y) for c, _, _ in reader_cores}
        free = [ttnn.CoreCoord(x_, y_) for y_ in range(self.gy) for x_ in range(self.gx) if (x_, y_) not in used]
        n_units = self.k * self.pairs
        per = -(-n_units // len(free))
        reducers = free[: -(-n_units // per)]
        owner = [(reducers[u // per], u % per) for u in range(n_units)]  # unit -> (core, local index)
        cores = [c for c, _, _ in reader_cores] + reducers
        core_set = self._coalesce(cores)
        cb_x, cb_w, cb_part, cb_recv, cb_hout, cb_route, cb_scr, cb_bias = 0, 1, 2, 3, 16, 5, 6, 7
        part = 4 * TILE
        cbs = [
            self._cb(cb_x, ttnn.bfloat16, TINY_BYTES, max_ks, core_set, tiny=True),
            self._cb(cb_w, self.dtype, self.tile_bytes, self.k * nr * max_ks, core_set),
            self._cb(cb_part, ttnn.float32, part, 8, core_set, tiny=True),
            self._cb(cb_recv, ttnn.float32, part, 2 * self.banks * per, core_set, tiny=True),
            self._cb(cb_hout, ttnn.bfloat16, TINY_BYTES, 2, core_set, tiny=True),
            self._cb(cb_route, ttnn.bfloat16, self.route_page, 1, core_set),
            self._cb(cb_scr, ttnn.bfloat16, self.route_page, 1, core_set),
            self._cb(cb_bias, ttnn.bfloat16, TINY_BYTES, 2, core_set, tiny=True),
        ]
        sem = 0
        sems = [ttnn.SemaphoreDescriptor(sem, ttnn.CoreType.WORKER, core_set, 0)]
        table = []
        for c, local in owner:
            n = device.worker_core_from_logical_core(c)
            table += [n.x, n.y, local]
        reader_ct = [cb_w, cb_route, cb_recv, 0, nr, readers, self.k, self.tile_bytes, self.route_page, sem, self.banks]
        reader_ct += ttnn.TensorAccessorArgs(route).get_compile_time_args()
        writer_ct = [cb_x, cb_part, cb_recv, cb_hout, cb_scr, nr, self.pairs, self.k, self.route_page, sem, self.banks]
        writer_ct += [cb_bias]
        for t in (x, h, route, bias if bias is not None else route):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cb_x, cb_w, cb_part, cb_recv, cb_hout, 0, nr, self.k, self.banks, cb_bias]
        bias_addr = bias.buffer_address() if bias is not None else 0
        defines = [("HAS_BIAS", "1")] if bias is not None else []
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        addrs = [x.buffer_address(), h.buffer_address(), route.buffer_address()]
        for c, b, r in reader_cores:
            ks, k0 = split[b]
            reader_rt[c.x][c.y] = [w_gul.buffer_address(), route.buffer_address(), 1, b, r, ks, 0, (b + r) & 3]
            writer_rt[c.x][c.y] = addrs + [1, b, r, ks, k0, 0, bias_addr] + table
            compute_rt[c.x][c.y] = [1, ks, 0]
        for i, c in enumerate(reducers):
            mine = [divmod(u, self.pairs) for u in range(i * per, min(n_units, (i + 1) * per))]
            reader_rt[c.x][c.y] = [w_gul.buffer_address(), route.buffer_address(), 2, 0, 0, 0, len(mine), 0]
            writer_rt[c.x][c.y] = addrs + [2, 0, 0, 0, 0, len(mine), bias_addr] + [v for sp in mine for v in sp]
            compute_rt[c.x][c.y] = [2, 0, len(mine)]
        kernels = self._kernels(
            "gu_local",
            core_set,
            (reader_ct, reader_rt),
            (writer_ct, writer_rt),
            compute_ct,
            compute_rt,
            defines=defines,
            compute_defines=self.compute_defines + defines,
        )
        io = [w_gul, x, route, h] + ([bias] if bias is not None else [])
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return h

    def down(self, h, route, w_down, out=None, bias_row=None):
        """``bias_row``: the down weights carry the experts' biases in this row (``down_layout(..., bias=...)``)."""
        device = h.device()
        if out is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 1, self.hidden]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
            )
        # bank b's local column r on the r-th reader of bank b, on the bank's row (gpt-oss row_placement)
        from models.demos.gpt_oss.tt.fused_decode.stream import row_placement

        split = bank_split(self.cols, self.banks)
        most, fewest = max(cpb for cpb, _ in split), min(cpb for cpb, _ in split)
        try:
            placed = row_placement(self.mesh, most)
        except AssertionError:
            # the banks' rows hold `fewest` per bank; the extra column of the larger banks goes to the free core
            # nearest the bank's own cores
            placed = row_placement(self.mesh, fewest)
            used = {(c.x, c.y) for cs in placed.values() for c in cs}
            for b, (cpb, _) in enumerate(split):
                while len(placed[b]) < cpb:
                    ref = placed[b][0]
                    c = min(
                        (
                            ttnn.CoreCoord(x_, y_)
                            for y_ in range(self.gy)
                            for x_ in range(self.gx)
                            if (x_, y_) not in used
                        ),
                        key=lambda c: (abs(c.y - ref.y), abs(c.x - ref.x)),
                    )
                    used.add((c.x, c.y))
                    placed[b].append(c)
        columns = [(placed[b][r], c0 + r, b, r, cpb) for b, (cpb, c0) in enumerate(split) for r in range(cpb)]
        cores = [c for c, *_ in columns]
        core_set = self._coalesce(cores)
        kt = self.k * self.kts
        cb_in0, cb_w, cb_route, cb_scr, cb_sync, cb_out = 0, 1, 2, 3, 4, 16
        cbs = [
            self._cb(cb_in0, ttnn.bfloat16, TINY_BYTES, kt, core_set, tiny=True),
            self._cb(cb_w, self.down_dtype, self.down_tile_bytes, kt, core_set),
            self._cb(cb_route, ttnn.bfloat16, self.route_page, 1, core_set),
            self._cb(cb_scr, ttnn.bfloat16, self.route_page, 1, core_set),
            self._cb(cb_sync, ttnn.bfloat16, 32, 1, core_set),
            self._cb(cb_out, ttnn.bfloat16, TINY_BYTES, 1, core_set, tiny=True),
        ]
        reader_ct = [
            cb_in0,
            cb_w,
            cb_route,
            self.kts,
            self.k,
            self.cols,
            self.down_tile_bytes,
            self.route_page,
            self.banks,
        ]
        reader_ct += [cb_sync, bias_row or 0]
        for t in (route, h):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [cb_out, cb_w, cb_sync, cb_scr, self.kts, self.k, self.cols, self.down_tile_bytes, self.route_page]
        writer_ct += [self.banks]
        for t in (out, route):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, col, bank, local, cpb in columns:
            reader_rt[core.x][core.y] = [
                w_down.buffer_address(),
                route.buffer_address(),
                h.buffer_address(),
                col,
                bank,
                local,
                cpb,
            ]
            writer_rt[core.x][core.y] = [
                out.buffer_address(),
                col,
                w_down.buffer_address(),
                route.buffer_address(),
                bank,
                local,
                cpb,
            ]
        defines = [("HAS_BIAS", "1")] if bias_row is not None else []
        kernels = self._kernels(
            "down",
            core_set,
            (reader_ct, reader_rt),
            (writer_ct, writer_rt),
            [cb_in0, cb_w, cb_out, kt],
            defines=defines,
        )
        ttnn.generic_op([w_down, route, h, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out

    def __call__(self, x, route, w_gu, w_down, bias_gu=None, down_bias_row=None):
        """``w_gu`` / ``w_down`` in the ``gate_up_local_layout`` / ``down_layout`` (``tp_expert_layouts``). Biases:
        ``bias_gu`` the ``gate_up_bias_layout`` rows (``upload_rows_per_chip``); the down bias rides in row
        ``down_bias_row`` of the down weights (``down_layout(..., bias=, bias_row=)``; on tensor-parallel chips put
        it on one chip only, the all-reduce sums the chips)."""
        h = self.gate_up_local(x, route, w_gu, readers=self.local_readers, bias=bias_gu)
        return self.down(h, route, w_down, bias_row=down_bias_row)


def router_topk_b1(logits, pes, k, out=None, core=None):
    """Batch-1 router tail on one core (kernels/router_topk_b1.cpp): FP32 logits ``[1, 1, 1, E]`` TILE ->
    top-k, softmax over the picks, times ``pes`` (``[1, 1, 1, E]`` bf16 ROW_MAJOR per-expert scale) -> the packed
    routing row ``[1, 1, 32, 2k]`` bf16 ROW_MAJOR (L1), row 0 = k weights then k global ids, as
    ``RouterTopk.op_packed`` writes it."""
    assert logits.dtype == ttnn.float32 and logits.layout == ttnn.TILE_LAYOUT
    E = int(logits.shape[-1])
    assert E % TILE == 0 and k <= 32
    device = logits.device()
    if out is None:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, 2 * k]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
    if core is None:
        g = device.compute_with_storage_grid_size()
        core = ttnn.CoreCoord(int(g.x) - 1, int(g.y) - 1)
    core_set = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
    scratch = E * 4 + E * 2 + 4 * k
    cb = ttnn.CBDescriptor(
        total_size=scratch,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=scratch)],
    )
    ct = [0, E, k]
    for t in (logits, pes, out):
        ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    rt = ttnn.RuntimeArgs()
    rt[core.x][core.y] = [logits.buffer_address(), pes.buffer_address(), out.buffer_address()]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / "router_topk_b1.cpp"),
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor(),  # BRISC: measured faster than NCRISC for this scalar code
    )
    ttnn.generic_op([logits, pes, out], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))
    return out
