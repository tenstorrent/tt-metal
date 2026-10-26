# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device side of the global KV pair split on one TP=8 replica (MIMO_KV_PAIR_SPLIT, vLLM mode; the layout and the
math are in reference/kv_pair_split.py).

Each chip of a global layer's pair (chips 2g, 2g + 1) stores its 64-position halves of every 128-position global
block, attends with its 16 query heads over them and scales its partial output by the merge weight from the two LSEs:

* ``positions``: one one-core program (kernels/attn/pair_positions.cpp) at the start of every decode and verify step
  writes each chip's write positions, SDPA positions and serial (one slot each) write positions from the step's global
  positions, which the trace advances with ``plus_one``. So the per-chip positions follow the global ones inside the
  trace, with nothing uploaded. The program keys on nothing that changes (generic_op hashes the runtime arguments'
  count, not their values), so it compiles in the decode trace's warm-up and serves the verify trace too.
* ``merge``: the partner's LSE (an all_gather of the LSE over the replica's row, then a tensor-args slice at the
  partner's index, a per-chip start), and ``alpha * o`` with ``alpha = 1 / (1 + exp(clamp(lse_p - lse, -80, 80)))``
  (reference.kv_pair_split.merge_weights_device): the pair's two scaled partials then add up to full attention in the
  row all-reduce after the o projection.
* ``merge_heads`` (decode): one launch. Each chip's exchange core sends its packed LSEs to the partner over the fabric
  (one hop) while the merge cores read o and the local LSE, then wait for the partner's packet and write the scaled,
  head-concatenated rows.

Every tensor here is persistent and made before any trace capture (TtMiMoModel.setup_vllm_kv_cache).
"""

import torch

import ttnn

from ..reference import kv_pair_split as ref
from . import kv as kv_lib
from .common import TILE
from .device_sampler import CB_SCRATCH, _cb, _is_dram, _kernel, _ranges

PARITY_WORDS = 16  # the per-chip parity page (64 bytes): word 0 the chip's parity, the rest zero
PARTNER_WORDS = 4  # the per-chip partner page (16 bytes): word 0 the partner's index, chip ^ 1
SLOT_BYTES = 2048  # one exchange slot: up to 32 users x 16 fp32 LSEs
PACKED_HEADS = 16  # LSEs per user in an exchange slot


def _align64(nbytes):
    return (nbytes + 63) // 64 * 64


class PairSplit:
    MERGE_CORES = 64  # merge_heads: rows of MERGE_GRID_X (as ConcatHeads)
    MERGE_GRID_X = 8
    EXCHANGE_CORE = (8, 0)  # merge_heads' fabric sender and receive slots, next to the merge grid
    SEND_CB = 2

    def __init__(self, plan, max_batch, serial_slots=0):
        """``serial_slots``: the serial write rows a verify step uses (0 without a drafter)."""
        tp = plan.tp
        assert tp % ref.REP == 0 and max_batch * 4 <= 1024 and serial_slots <= max_batch
        self.plan, self.max_batch, self.serial_slots = plan, max_batch, serial_slots
        chips = torch.arange(tp, dtype=torch.int32)
        parity = torch.zeros(tp, PARITY_WORDS, dtype=torch.int32)
        parity[:, 0] = chips % ref.REP
        # Per chip: int32 [PARITY_WORDS], word 0 its parity (the positions kernel reads one 64-byte page).
        self.parity = kv_lib.per_chip(plan, parity, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        # Per chip: int32 [1], its parity (pair-split chunked MLA prefill's kv_pair_parity_tensor).
        self.prefill_parity = kv_lib.per_chip(plan, parity[:, :1].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        partner = torch.zeros(tp, PARTNER_WORDS, dtype=torch.int32)
        partner[:, 0] = chips ^ 1
        # Per chip: the slice start (partner, 0, 0, 0) of its partner's LSE in the row's gathered LSEs.
        self.partner_start = kv_lib.per_chip(plan, partner, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        idle = torch.full((max_batch,), -1, dtype=torch.int32)
        self.write = kv_lib.replicated(plan, idle, ttnn.int32)
        self.sdpa = kv_lib.replicated(plan, idle, ttnn.int32)
        self.serial = [kv_lib.replicated(plan, idle, ttnn.int32) for _ in range(serial_slots)]
        self.core = ttnn.CoreCoord(0, 0)

        # merge_heads' pair exchange: two receive slots on the exchange core (alternated by call count), its arrival
        # semaphore (the partner's packets) and send count, and per merge RISC its call count. The counts are never
        # reset: every launch adds one to each, on every chip.
        device = plan.mesh_device
        rows = self.MERGE_CORES // self.MERGE_GRID_X
        self.merge_cores = [ttnn.CoreCoord(x, y) for y in range(rows) for x in range(self.MERGE_GRID_X)]
        self.merge_ranges = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.MERGE_GRID_X - 1, rows - 1))]
        )
        self.exchange_core = ttnn.CoreCoord(*self.EXCHANGE_CORE)
        self.exchange_ranges = _ranges([self.exchange_core])
        slot_rows = 2 * SLOT_BYTES // (TILE * 2)
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.exchange_ranges, [slot_rows, TILE], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.slots = ttnn.allocate_tensor_on_device(
            ttnn.Shape([slot_rows, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc
        )
        self.arrive = ttnn.create_global_semaphore(device, self.exchange_ranges, 0)
        self.sent = ttnn.create_global_semaphore(device, self.exchange_ranges, 0)
        self.merged = [ttnn.create_global_semaphore(device, self.merge_ranges, 0) for _ in range(2)]
        self.fabric_nodes = None

    # ------------------------------------------------------------------ device (trace-capturable)
    def positions(self, cur_pos):
        """The step's global positions ``cur_pos`` (int32 [B], DRAM row-major, -1 an idle slot) -> each chip's
        (write [B], sdpa [B], serial rows), the persistent buffers rewritten in place."""
        b, rows = self.max_batch, self.serial_slots
        assert tuple(cur_pos.shape) == (b,) and cur_pos.dtype == ttnn.int32 and _is_dram(cur_pos)
        assert cur_pos.layout == ttnn.ROW_MAJOR_LAYOUT
        outs = [self.write, self.sdpa, *self.serial]
        args = [cur_pos.buffer_address(), self.parity.buffer_address()] + [t.buffer_address() for t in outs]
        scratch = _align64(b * 4) + _align64(PARITY_WORDS * 4) + (2 + rows) * _align64(b * 4)
        cores = [self.core]
        kernel = _kernel(
            "attn/pair_positions.cpp",
            cores,
            _ranges(cores),
            [b, rows, CB_SCRATCH, PARITY_WORDS * 4],
            [args],
            ttnn.ReaderConfigDescriptor(),
        )
        program = ttnn.ProgramDescriptor(
            kernels=[kernel], semaphores=[], cbs=[_cb(CB_SCRATCH, scratch, _ranges(cores))]
        )
        ttnn.generic_op([cur_pos, self.parity, *outs], program)
        return self.write, self.sdpa, self.serial

    def partner(self, lse):
        """The pair partner's LSE, of ``lse``'s shape: the row's LSEs gathered along dim 0, then the partner's entry
        (a tensor-args slice: the start is per chip; the program keys on the shape only). The LSE is a TILE tensor
        whose last dim (and in decode, the 16 heads) does not fill its tiles, and the tensor-args slice checks the
        logical shape against whole tiles, so it slices a view of the whole tiles (logical = padded shape) and views
        the result back."""
        assert lse.shape[0] == 1, f"the LSE's dim 0 gathers the row: {tuple(lse.shape)}"
        gathered = self.plan.all_gather(lse, dim=0)
        tp = self.plan.tp
        tiles = ttnn.Shape(list(gathered.padded_shape))
        whole = ttnn.reshape(gathered, tiles, tiles)  # a view: same buffer, the padding counted in
        out = ttnn.slice(whole, self.partner_start, self.partner_start, slice_dim=0, num_devices=tp)
        gathered.deallocate(True)
        return ttnn.reshape(out, lse.shape, ttnn.Shape(list(out.padded_shape)))  # a view back to lse's shape

    def merge(self, o, lse, memory_config):
        """``alpha * o`` (bf16, of o's shape): o the chip's normalized partial attention (bf16, [.., rows, dv]), lse
        its fp32 LSE ([.., rows, 1], broadcast along o's last dim), alpha this chip's merge weight in fp32."""
        lse_p = self.partner(lse)
        d = ttnn.subtract(lse_p, lse, dtype=ttnn.float32, memory_config=memory_config)
        lse_p.deallocate(True)
        clamped = ttnn.clamp(d, min=-ref.LSE_CLAMP, max=ref.LSE_CLAMP, memory_config=memory_config)
        d.deallocate(True)
        e = ttnn.exp(clamped, memory_config=memory_config)
        clamped.deallocate(True)
        den = ttnn.add(e, 1.0, memory_config=memory_config)
        e.deallocate(True)
        alpha = ttnn.reciprocal(den, memory_config=memory_config)
        den.deallocate(True)
        out = ttnn.multiply(o, alpha, dtype=ttnn.bfloat16, memory_config=memory_config)
        alpha.deallocate(True)
        return out

    def _nodes(self):
        if self.fabric_nodes is None:
            device = self.plan.mesh_device
            rows, cols = self.plan.mesh_shape
            self.fabric_nodes = [
                [device.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for c in range(cols)] for r in range(rows)
            ]
        return self.fabric_nodes

    def merge_heads(self, o, lse, concat_heads):
        """Decode, from the SDPA's outputs to the o projection's input: ``o`` this chip's normalized partial attention
        [1, B, 16, 128] (bf16, L1) and ``lse`` its LSE [1, B, 16, 1] (fp32) -> the concatenated heads of ``alpha * o``
        [1, 1, B, 16 * 128] (bf16, L1; ``concat_heads`` an attention.ConcatHeads, for the heads and kept V columns).
        The caller frees ``o`` and ``lse``. One launch: each chip's exchange core (kernels/attn/pair_lse_send.cpp)
        sends its packed LSEs to the pair partner over the fabric, and the merge cores
        (kernels/attn/pair_merge_heads.cpp) wait for the partner's, compute alpha and write the scaled,
        head-concatenated rows (rows >= B zeros, as ConcatHeads)."""
        shape = list(o.shape)
        users = shape[1]
        assert len(shape) == 4 and shape[0] == 1 and 1 <= users <= TILE and shape[3] % TILE == 0
        assert o.layout == ttnn.TILE_LAYOUT and o.dtype == ttnn.bfloat16
        assert tuple(lse.shape) == (1, users, shape[2], 1), (tuple(lse.shape), shape)
        assert lse.layout == ttnn.TILE_LAYOUT and lse.dtype == ttnn.float32
        assert shape[2] <= PACKED_HEADS and users * PACKED_HEADS * 4 <= SLOT_BYTES
        assert self.plan.tp_axis == 1
        for t in (o, lse):
            assert t.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        d_tiles = shape[3] // TILE
        heads = concat_heads.num_heads if concat_heads is not None else shape[2]
        v_tiles = concat_heads.v_tiles if concat_heads is not None else d_tiles
        assert heads <= shape[2] and v_tiles <= d_tiles
        head_tiles = (shape[2] + TILE - 1) // TILE
        tiles = heads * v_tiles
        cores = self.MERGE_CORES
        # every merge core runs every call: its call count must advance with the partner's sends
        assert tiles % cores == 0, (tiles, cores)
        clamp = int(ref.LSE_CLAMP)
        assert clamp == ref.LSE_CLAMP

        device = self.plan.mesh_device
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, tiles * TILE]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.L1_MEMORY_CONFIG,
        )
        per_core = tiles // cores
        ex = self.exchange_core
        ex_noc = device.worker_core_from_logical_core(ex)
        slots_addr = self.slots.buffer_address()
        arrive_addr = ttnn.get_global_semaphore_address(self.arrive)
        args = [
            per_core,
            users,
            head_tiles * d_tiles,
            d_tiles,
            v_tiles,
            head_tiles,
            _is_dram(o),
            _is_dram(lse),
            clamp,
            SLOT_BYTES,
        ]
        # alignment slack, the polled count, per user its LSE row, the partner's packed LSEs and o row, two output faces
        scratch = 64 + 64 + users * (64 + 64 + 32) + 2 * 512
        configs = ((0, ttnn.WriterConfigDescriptor()), (1, ttnn.ReaderConfigDescriptor()))
        kernels = []
        for face, config in configs:
            count_addr = ttnn.get_global_semaphore_address(self.merged[face])
            rt = [
                [
                    o.buffer_address(),
                    lse.buffer_address(),
                    slots_addr,
                    arrive_addr,
                    count_addr,
                    out.buffer_address(),
                    i * per_core,
                    ex_noc.x,
                    ex_noc.y,
                ]
                for i in range(cores)
            ]
            kernels.append(
                _kernel(
                    "attn/pair_merge_heads.cpp", self.merge_cores, self.merge_ranges, [face, face] + args, rt, config
                )
            )
        cbs = [_cb(face, scratch, self.merge_ranges) for face, _ in configs]
        send_scratch = 64 + _align64(users * PACKED_HEADS * 4) + users * 1024
        cbs.append(_cb(self.SEND_CB, send_scratch, self.exchange_ranges))
        send_rt = [
            lse.buffer_address(),
            slots_addr,
            arrive_addr,
            ttnn.get_global_semaphore_address(self.sent),
            ex_noc.x,
            ex_noc.y,
        ]
        send_ct = [self.SEND_CB, users, head_tiles, _is_dram(lse), SLOT_BYTES]

        nodes = self._nodes()
        mesh_program = ttnn.MeshProgramDescriptor()
        n = len(kernels)
        for r, row_nodes in enumerate(nodes):
            for c in range(len(row_nodes)):
                send = _kernel(
                    "attn/pair_lse_send.cpp",
                    [ex],
                    self.exchange_ranges,
                    send_ct,
                    [list(send_rt)],
                    ttnn.WriterConfigDescriptor(),
                )
                program = ttnn.ProgramDescriptor(kernels=list(kernels) + [send], cbs=list(cbs), semaphores=[])
                fabric_args = ttnn.setup_fabric_connection(
                    src_fabric_node_id=row_nodes[c],
                    dst_fabric_node_id=row_nodes[c ^ 1],
                    link_idx=0,
                    program_descriptor=program,
                    worker_core=ex,
                )
                program.kernels[n].runtime_args[ex.x][ex.y].extend(fabric_args)
                coord = ttnn.MeshCoordinate(r, c)
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        ttnn.generic_op([o, lse, self.slots, out], mesh_program)
        return out
