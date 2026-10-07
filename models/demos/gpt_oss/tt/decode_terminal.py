# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Terminal path of the fused decode layers (tt/fused_decode.py): final norm -> LM head -> logits -> sampling.

LM head: a DRAM-streaming LinearStream (experts/stream.py) over a decode-only BFP8 copy of the LM-head weight, split
evenly over the TP devices (device d owns the vocab ids [d * vd, (d + 1) * vd), vd = vocab / tp rounded up to a tile;
ids >= vocab get a -1e30 bias). Its input is the last layer's pending boundary (the MoE all-reduce + residual + final
RMSNorm runs inside the op, decode_boundary.py consumer_parts), and it writes the logits "folded": local vocab tile n
goes to row n // row_tiles, tile n % row_tiles of a persistent [1, 1, 32, width] BF16 tensor (width = row_tiles * 32
rounded up to a power of two; the never-written tail stays -1e30). One token's logits thus fill all 32 rows of the
tile layout, so the row-wise top-k runs on 32 x width elements instead of 32 rows of the full vocab shard.

Sampling (split, top-k / top-p / temperature capable). Candidates, DECODE_TERMINAL_EXCHANGE = "fused" (default): the
LM-head op itself produces them (TerminalExchange): each LM-head writer core keeps the top-32 of its own logits, an
exchange core merges the 8 lists into the device's top-32 (kernels/terminal_exchange.cpp), the fabric sender of the
layer boundaries (kernels/boundary_sender.cpp) writes them into every device's receive slots, and the exchange core
writes the gathered 4 x 32 candidates as the sampler input; it also advances the decode position state (the plus_one
ops of the unfused path) when the model decodes with on-device sampling. "ops": ttnn.topk(k=32) per folded row ->
per-device merge (kernels/terminal_merge.cpp) -> two all_gathers. Then either ttnn.sampling (k / p / temp, the common
sampler's op) or, for greedy requests, the argmax over the gathered candidates (kernels/terminal_pick.cpp), written
straight into the decode token input (tt_out_tok). The candidate ids are exact: every device's top-32 is contained in
the union of the per-core (or per-row) top-32s, and the merges and the greedy pick order ties by the lowest vocab id.
"""

import torch
from loguru import logger

import ttnn
from models.common.sampling.generator import SamplingGenerator
from models.common.sampling.tt_sampling import TTSampling

from .decode_boundary import KERNEL_DIR
from .experts.stream import LinearStream, as_stream_tensor, linear_stream_rows, stream_linear_layout
from .fused_decode import (
    DECODE_GREEDY_SAMPLER,
    DECODE_TERMINAL_EXCHANGE,
    LM_HEAD_DECODE_WEIGHT_DTYPE,
    LM_HEAD_STREAM_PREFETCH,
    LM_HEAD_STREAM_READERS,
)

TILE = ttnn.TILE_SIZE
MASK = -1.0e30  # logit of a padded vocab id
TOPK = 32  # candidates per folded row, per device and per device after the merge (ttnn.sampling max_top_k)
MERGE_CORE = ttnn.CoreCoord(0, 0)
# The fused exchange's merge / fabric-send core: not an LM-head stream worker and not the boundary core (asserted).
EXCHANGE_CORE = ttnn.CoreCoord(4, 0)


def _cache_name(tensor_cache_path, name):
    return None if tensor_cache_path is None else f"{tensor_cache_path}/{name}"


class DecodeTerminal:
    """Persistent state and ops of the fused decode terminal path (module doc)."""

    def __init__(self, mesh_device, hf_config, lm_head_weight, tensor_cache_path):
        """lm_head_weight: the HF [vocab, hidden] LM-head weight, or None to load the streamed copy from the cache."""
        self.mesh_device = mesh_device
        self.tp = mesh_device.shape[1]
        self.vocab_size = hf_config.vocab_size
        self.hidden = hf_config.hidden_size
        self.vd = -(-self.vocab_size // self.tp // TILE) * TILE  # vocab ids per device
        self.n_tiles = self.vd // TILE
        self.row_tiles = -(-self.n_tiles // TILE)
        self.row_len = self.row_tiles * TILE
        self.width = 1 << (self.row_len - 1).bit_length()
        self.coords = [ttnn.MeshCoordinate(0, c) for c in range(self.tp)]
        banks = mesh_device.dram_grid_size().x

        self.op = LinearStream(
            mesh_device,
            self.hidden // TILE,
            self.vd,
            LM_HEAD_DECODE_WEIGHT_DTYPE,
            readers=LM_HEAD_STREAM_READERS,
            heads=(self.n_tiles, 0),
            head_tiles=self.row_tiles,
            prefetch_cols=LM_HEAD_STREAM_PREFETCH,
        )
        rows = linear_stream_rows(mesh_device, self.hidden, self.vd)
        layouts = None
        if lm_head_weight is not None:
            w = lm_head_weight.transpose(0, 1).float()  # [hidden, vocab]
            layouts = []
            for d in range(self.tp):
                lo, hi = d * self.vd, min((d + 1) * self.vd, self.vocab_size)
                wd = torch.zeros(self.hidden, self.vd)
                wd[:, : hi - lo] = w[:, lo:hi]
                bias = torch.zeros(self.vd)
                bias[hi - lo :] = MASK
                layouts.append(stream_linear_layout(wd, bias, banks))
            assert layouts[0].shape[0] == rows
        dt = {ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}[LM_HEAD_DECODE_WEIGHT_DTYPE]
        self.weights = as_stream_tensor(
            mesh_device,
            layouts,
            rows,
            LM_HEAD_DECODE_WEIGHT_DTYPE,
            _cache_name(tensor_cache_path, f"lm_head.weight_stream_{dt}_vd{self.vd}_b{banks}"),
        )

        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        shard = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        l1 = ttnn.L1_MEMORY_CONFIG

        def dev(t, dtype, layout, mapper=replicate):
            return ttnn.from_torch(
                t, device=mesh_device, dtype=dtype, layout=layout, memory_config=l1, mesh_mapper=mapper
            )

        # Persistent tensors (allocated at model construction, before any trace exists).
        self.folded = dev(torch.full((1, 1, TILE, self.width), MASK), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.topk_values = dev(torch.zeros(1, 1, TILE, TOPK), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.topk_indices = dev(torch.zeros(1, 1, TILE, TOPK, dtype=torch.int32), ttnn.uint16, ttnn.TILE_LAYOUT)
        # Merge outputs ("ops" candidates): row 0 = this device's candidates; rows 1..31 (the 31 unused sampling users)
        # hold distinct finite filler values (ttnn.sampling must not see a row of identical values) and id 0; the
        # all-gather carries them into the gathered rows.
        filler = torch.zeros(self.tp, 1, TILE, TOPK)
        for d in range(self.tp):
            filler[d, 0, 1:] = -(100.0 + d * TOPK + torch.arange(TOPK, dtype=torch.float32))
        self.cand_values = dev(filler, ttnn.bfloat16, ttnn.TILE_LAYOUT, shard)
        self.cand_ids = dev(
            torch.zeros(self.tp, 1, TILE, TOPK, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, shard
        )
        # Rows 1..31 (unused sampling users) keep distinct finite filler values: nothing writes them later.
        gathered_filler = torch.zeros(1, 1, TILE, TOPK * self.tp)
        gathered_filler[0, 0, 1:] = -(100.0 + torch.arange(TOPK * self.tp, dtype=torch.float32))
        self.gathered_values = dev(gathered_filler, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.gathered_ids = dev(
            torch.zeros(1, 1, TILE, TOPK * self.tp, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
        self.exchange = TerminalExchange(self) if DECODE_TERMINAL_EXCHANGE == "fused" else None
        logger.info(
            f"Fused decode terminal: LM head {self.hidden}x{self.vd} per device ({self.n_tiles} tiles, "
            f"folded {TILE}x{self.width}), candidates '{DECODE_TERMINAL_EXCHANGE}', greedy sampler '{DECODE_GREEDY_SAMPLER}'"
        )

    # ---- LM head ----

    def lm_head(self, x, sampling=False, current_pos=None, rot_idxs=None):
        """x: the last layer's PendingBoundary (or the flat normed hidden) -> the folded logits (persistent). With the
        fused exchange every call also produces the gathered sampler candidates; sampling (the model decodes with
        on-device sampling) also advances current_pos / rot_idxs by one (both or neither given)."""
        # Every call gathers the candidates (one program for both decode modes: no mode-specific LM-head program is
        # compiled after the first trace capture); only the position advance depends on `sampling`.
        exchange = self.exchange.bind(current_pos, rot_idxs, advance=sampling) if self.exchange is not None else None
        self.op(x, self.weights, (self.folded, self.folded, self.folded), exchange=exchange)
        return self.folded

    @property
    def advances_positions(self):
        """The LM head (sampling) advances the decode positions itself."""
        return self.exchange is not None

    def is_folded(self, logits):
        return isinstance(logits, ttnn.Tensor) and list(logits.shape) == [1, 1, TILE, self.width]

    def unfold_host(self, device_tensors):
        """Host view of the folded logits: per-device torch tensors [1, 1, 32, width] -> [vocab] logits."""
        parts = [t.reshape(TILE, self.width)[:, : self.row_len].reshape(-1)[: self.vd] for t in device_tensors]
        return torch.cat(parts)[: self.vocab_size]

    # ---- sampling ----

    def candidates(self, logits):
        """Folded logits -> gathered (values, ids): row 0 holds every device's top-32 (ordered by value, then id).
        With the fused exchange the LM-head op already wrote them: contract, the folded logits handed to the sampler
        come from the model's decode forward with on-device sampling (lm_head(sampling=True)); the generator's traced
        and eager on-device-sampling paths are the only callers. Under trace this cannot be checked from Python."""
        if self.exchange is not None:
            return self.gathered_values, self.gathered_ids
        ttnn.topk(
            logits,
            k=TOPK,
            dim=-1,
            largest=True,
            sorted=True,
            output_tensor=(self.topk_values, self.topk_indices),
        )
        self._merge()
        ttnn.all_gather(self.cand_values, dim=3, output_tensor=self.gathered_values)
        ttnn.all_gather(self.cand_ids, dim=3, output_tensor=self.gathered_ids)
        return self.gathered_values, self.gathered_ids

    def _single_core_program(self, kernel, ct, rt, cb_bytes):
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(MERGE_CORE, MERGE_CORE)])
        cb = ttnn.CBDescriptor(
            total_size=cb_bytes,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=cb_bytes)],
        )
        args = ttnn.RuntimeArgs()
        args[MERGE_CORE.x][MERGE_CORE.y] = rt
        k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / kernel),
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=args,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )
        return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb])

    def _merge(self):
        io = [self.topk_values, self.topk_indices, self.cand_values, self.cand_ids]
        ct = [0, self.row_len, TOPK]
        for t in io:
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        mesh_program = ttnn.MeshProgramDescriptor()
        for d, coord in enumerate(self.coords):
            rt = [t.buffer_address() for t in io] + [d * self.vd]
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = self._single_core_program(
                "terminal_merge.cpp", ct, rt, 4096 + 1024 + 128
            )
        ttnn.generic_op(io, mesh_program)

    def pick(self, values, ids, tok):
        """Greedy: the argmax of the gathered candidates (lowest id on ties) -> token slot 0 of tok."""
        io = [values, ids, tok]
        n_tiles = values.shape[-1] // TILE
        ct = [0, n_tiles]
        for t in io:
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = [t.buffer_address() for t in io]
        program = self._single_core_program("terminal_pick.cpp", ct, rt, n_tiles * 64 + n_tiles * TILE * 4 + 128)
        ttnn.generic_op(io, program)
        return tok

    def new_token_tensor(self):
        return ttnn.from_torch(
            torch.zeros(1, 1, 1, TILE, dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )


class TerminalExchange:
    """Top-k candidate merge + cross-device exchange fused into the LM-head op (module doc). Persistent state, all
    allocated at model construction on EXCHANGE_CORE (same addresses on every device): the writers' list buffer, this
    device's payload, the receive slots (one per device) and the global receive semaphore."""

    LISTS_SEM, SEND_SEM = 1, 2  # program semaphores (0 is the fused boundary's)

    def __init__(self, terminal):
        self.t = terminal
        mesh = terminal.mesh_device
        self.core = EXCHANGE_CORE
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(self.core, self.core)])
        writer_cores = {(c.x, c.y) for c, *_ in terminal.op.assignments}
        from .decode_boundary import BOUNDARY_CORE

        assert (self.core.x, self.core.y) not in writer_cores and (self.core.x, self.core.y) != BOUNDARY_CORE
        self.n_lists = len(terminal.op.assignments)
        self.ring = terminal.tp
        self.fwd_hops = self.ring // 2
        self.bwd_hops = self.ring - 1 - self.fwd_hops
        self.payload = TOPK * 2 + TOPK * 4
        phys = mesh.worker_core_from_logical_core(self.core)
        self.noc_xy = (phys.x, phys.y)

        def on_core(words):
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.cores, (1, words), ttnn.ShardOrientation.ROW_MAJOR),
            )
            return ttnn.from_torch(
                torch.zeros(1, 1, 1, words, dtype=torch.int32),
                device=mesh,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=mem,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        self.lists = on_core(self.n_lists * 2 * TOPK)
        self.payload_buf = on_core(self.payload // 4)
        self.slots = on_core(self.ring * self.payload // 4)
        self.recv_sem = ttnn.create_global_semaphore(mesh, self.cores, 0)
        self.coords = terminal.coords

    def bind(self, current_pos, rot_idxs, advance):
        return _ExchangeCall(self, current_pos, rot_idxs, advance)


class _ExchangeCall:
    """One LM-head call's exchange (the position tensors differ between eager calls)."""

    def __init__(self, ex, current_pos, rot_idxs, advance):
        assert (current_pos is None) == (rot_idxs is None)
        assert not advance or current_pos is not None, "advancing the positions needs the position tensors"
        self.ex, self.pos, self.rot, self.advance = ex, current_pos, rot_idxs, advance
        t = ex.t
        self.io = [ex.lists, ex.payload_buf, ex.slots, t.gathered_values, t.gathered_ids]
        if current_pos is not None:
            self.io += [current_pos, rot_idxs]

    def writer_compile_args(self):
        return [1, self.ex.noc_xy[0], self.ex.noc_xy[1], self.ex.LISTS_SEM]

    def writer_runtime_args(self, i):
        return [self.ex.lists.buffer_address(), i]

    @staticmethod
    def _page(t):
        """(page bytes, entries) of a single-page row-major position tensor."""
        n = 1
        for d in t.shape:
            n *= d
        return n * 4, n

    def program(self, build, writer_cores):
        ex, t = self.ex, self.ex.t
        mesh = t.mesh_device
        pos_page, pos_n = self._page(self.pos) if self.pos is not None else (0, 0)
        rot_page, rot_n = self._page(self.rot) if self.rot is not None else (0, 0)
        scr_bytes = max(64, (pos_page + 63) // 64 * 64 + rot_page)
        merger_ct = [ex.n_lists, ex.LISTS_SEM, ex.SEND_SEM, ex.ring, TOPK, 1 if self.pos is not None else 0]
        merger_ct += [pos_page, pos_n, rot_page, rot_n, 0]
        pos_t = self.pos if self.pos is not None else t.gathered_ids
        rot_t = self.rot if self.rot is not None else t.gathered_ids
        for tt_ in (t.gathered_values, t.gathered_ids, pos_t, rot_t):
            merger_ct += ttnn.TensorAccessorArgs(tt_).get_compile_time_args()
        mesh_program = ttnn.MeshProgramDescriptor()
        for d, coord in enumerate(ex.coords):
            program = build()
            program.semaphores = list(program.semaphores) + [
                ttnn.SemaphoreDescriptor(ex.LISTS_SEM, ttnn.CoreType.WORKER, writer_cores.merge(ex.cores), 0),
                ttnn.SemaphoreDescriptor(ex.SEND_SEM, ttnn.CoreType.WORKER, ex.cores, 0),
            ]
            program.cbs = list(program.cbs) + [
                ttnn.CBDescriptor(
                    total_size=scr_bytes,
                    core_ranges=ex.cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scr_bytes)
                    ],
                )
            ]
            mrt = ttnn.RuntimeArgs()
            mrt[ex.core.x][ex.core.y] = [
                ex.lists.buffer_address(),
                ex.payload_buf.buffer_address(),
                ex.slots.buffer_address(),
                t.gathered_values.buffer_address(),
                t.gathered_ids.buffer_address(),
                d * t.vd,
                ttnn.get_global_semaphore_address(ex.recv_sem),
                pos_t.buffer_address(),
                rot_t.buffer_address(),
                1 if self.advance else 0,
            ]
            merger = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "terminal_exchange.cpp"),
                core_ranges=ex.cores,
                compile_time_args=merger_ct,
                runtime_args=mrt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                ),
            )
            srt = ttnn.RuntimeArgs()
            srt[ex.core.x][ex.core.y] = [
                ex.payload_buf.buffer_address(),
                ex.slots.buffer_address() + d * ex.payload,
                ex.noc_xy[0],
                ex.noc_xy[1],
                ttnn.get_global_semaphore_address(ex.recv_sem),
            ]
            sender = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "boundary_sender.cpp"),
                core_ranges=ex.cores,
                compile_time_args=[ex.payload, ex.payload, ex.fwd_hops, ex.bwd_hops, 1, ex.SEND_SEM, 0, 1, 1],
                runtime_args=srt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
                ),
            )
            program.kernels = list(program.kernels) + [merger, sender]
            src = mesh.get_fabric_node_id(coord)
            fabric_args = []
            for step, hops in ((1, ex.fwd_hops), (-1, ex.bwd_hops)):
                fabric_args.append(1 if hops > 0 else 0)
                if hops > 0:
                    dst = ttnn.MeshCoordinate(coord[0], (coord[1] + step) % ex.ring)
                    fabric_args += ttnn.setup_fabric_connection(
                        src_fabric_node_id=src,
                        dst_fabric_node_id=mesh.get_fabric_node_id(dst),
                        link_idx=0,
                        program_descriptor=program,
                        worker_core=ex.core,
                    )
            program.kernels[len(program.kernels) - 1].runtime_args[ex.core.x][ex.core.y].extend(fabric_args)
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        return mesh_program


class FusedDecodeSampling(TTSampling):
    """TTSampling whose decode path consumes the folded logits of DecodeTerminal (module doc). Other logits (prefill
    sampling, which keeps the original LM head) take the stock TTSampling path; its force-argmax branch needs a CCL
    manager this model does not pass, so it is never used there."""

    def __init__(self, mesh_device, tt_ccl, args, terminal):
        super().__init__(mesh_device=mesh_device, tt_ccl=tt_ccl, args=args)
        self.terminal = terminal
        # Greedy requests (k=1, p in {0, 1}, temp=1) take the argmax of the gathered candidates instead of the
        # seeded ttnn.sampling draw (fused_decode.DECODE_GREEDY_SAMPLER). Keyed as force-argmax by SamplingGenerator.
        self._allow_force_argmax_sampling = DECODE_GREEDY_SAMPLER == "split_argmax"
        self._force_argmax_sampling = False
        self._params_key = None

    def reset_params(self, k, p, temp, enable_log_probs=None, num_logprobs=None, empty_slots=None):
        """Re-upload the parameter tensors only when the request parameters changed (the generator may reload the
        same parameters every step). TTSampling.reset_params skips the upload for force-argmax requests; the stock
        path (prefill sampling) still reads k / p / temp then, so upload them for every change and set the
        force-argmax flag afterwards."""
        key = repr((k, p, temp, enable_log_probs, num_logprobs, empty_slots))
        if key == self._params_key:
            return
        allow = self._allow_force_argmax_sampling
        self._allow_force_argmax_sampling = False
        try:
            super().reset_params(
                k, p, temp, enable_log_probs=enable_log_probs, num_logprobs=num_logprobs, empty_slots=empty_slots
            )
        finally:
            self._allow_force_argmax_sampling = allow
        k_n, temp_n = self._normalize_device_params(k, temp)
        self._force_argmax_sampling = self._is_force_argmax_sampling(k_n, p, temp_n)
        self._params_key = key

    def forward(self, x, tt_out_tok=None):
        if not self.terminal.is_folded(x):
            forced = self._force_argmax_sampling
            self._force_argmax_sampling = False
            try:
                return super().forward(x, tt_out_tok=tt_out_tok)
            finally:
                self._force_argmax_sampling = forced
        if self.log_probs_calculator.enable_log_probs:
            raise NotImplementedError(
                "log-probs are not implemented on the fused batch-1 decode path (tt/decode_terminal.py)"
            )
        values, ids = self.terminal.candidates(x)
        if self._force_argmax_sampling:
            tok = tt_out_tok if tt_out_tok is not None else self.terminal.new_token_tensor()
            return self.terminal.pick(values, ids, tok), None
        ttnn.manual_seed(
            seeds=self.seeds_tt_tensor, user_ids=self.user_ids_tt_tensor, sub_core_grids=self._sampling_sub_core_grids
        )
        tok = ttnn.sampling(
            values,
            ids,
            k=self.k_tensor,
            p=self.p_tensor,
            temp=self.temp_tensor,
            sub_core_grids=self._sampling_sub_core_grids,
            output_tensor=tt_out_tok,
        )
        return tok, None


class FusedSamplingGenerator(SamplingGenerator):
    """SamplingGenerator over FusedDecodeSampling. The fused decode path implements greedy and top-k / top-p /
    temperature sampling; penalties and log-probs (the batch-128 Galaxy serving configurations) are not implemented on
    it: a decode sample / capture with them active raises before any trace capture starts, and their decode warmup
    variants are skipped."""

    def __init__(self, *, args, mesh_device, tt_ccl, terminal, cq_id=0):
        super().__init__(args=args, mesh_device=mesh_device, tt_ccl=tt_ccl, cq_id=cq_id)
        self.tt_sampling = FusedDecodeSampling(mesh_device, tt_ccl, args, terminal)
        self.seed_manager = type(self.seed_manager)(
            self.tt_sampling,
            max_batch_size=self.tt_sampling.max_batch_size * self.tt_sampling._sampling_dp,
            salt_duplicate_seeds=getattr(args, "salt_duplicate_seeds", True),
        )
        self.terminal = terminal
        self._params_repr = None

    def reset_sampling_params(self, sampling_params, empty_slots=None):
        """Changed-only parameter upload; switching between greedy (force-argmax key) and sampled requests selects the
        other cached sampling trace instead of releasing them all (SamplingGenerator.reset_sampling_params does, on a
        force-argmax change; the trace slots are already keyed by force_argmax)."""
        key = repr((sampling_params, empty_slots))
        if key == self._params_repr:
            return
        self._in_param_reset = True
        try:
            super().reset_sampling_params(sampling_params, empty_slots=empty_slots)
        finally:
            self._in_param_reset = False
        self._params_repr = key

    def reset_trace(self):
        if getattr(self, "_in_param_reset", False):
            return
        super().reset_trace()

    def _run_sampling(self, logits, *, penalties_on, tt_out_tok, count_tokens=True):
        if penalties_on and self.terminal.is_folded(logits):
            raise NotImplementedError(
                "sampling penalties are not implemented on the fused batch-1 decode path (tt/decode_terminal.py)"
            )
        return super()._run_sampling(
            logits, penalties_on=penalties_on, tt_out_tok=tt_out_tok, count_tokens=count_tokens
        )

    def capture_trace(self, logits, *, tt_out_tok=None, skip_precompile=False):
        """SamplingGenerator.capture_trace without its whole-window corruptible_allocation_scope for the fused path:
        the fused sampling ops allocate nothing inside the capture (the pick and ttnn.sampling write the preallocated
        tt_out_tok, manual_seed has no output, the candidates are persistent), so the trace allocation tracker keeps
        checking the window. Other logits (stock path) keep the base behavior."""
        if not self.terminal.is_folded(logits) or tt_out_tok is None:
            return super().capture_trace(logits, tt_out_tok=tt_out_tok, skip_precompile=skip_precompile)
        self._check_supported()
        penalties_on = self._penalties_active
        log_probs_on = getattr(self, "_log_probs_active", False)
        key, slot = self._trace_slot(penalties_on, log_probs_on, self.tt_sampling.force_argmax_sampling)
        if not skip_precompile:
            self._run_sampling(logits, penalties_on=penalties_on, tt_out_tok=tt_out_tok, count_tokens=False)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=self.cq_id)
        sampled = self._run_sampling(logits, penalties_on=penalties_on, tt_out_tok=tt_out_tok)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=self.cq_id)
        ttnn.synchronize_device(self.mesh_device)
        slot["id"] = trace_id
        slot["input"] = logits
        slot["output"] = (tt_out_tok, sampled[-1]) if isinstance(sampled, tuple) else (tt_out_tok, sampled)
        slot["kwargs"] = {"tt_out_tok": tt_out_tok}
        return slot["output"]

    def _check_supported(self):
        """Penalties / log-probs requested for fused decode: reject before any trace capture starts (the generator's
        prefill warmup legitimately compiles them on the stock path, so the parameters themselves are accepted)."""
        if self._penalties_active or getattr(self, "_log_probs_active", False):
            raise NotImplementedError(
                "sampling penalties / log-probs are not implemented on the fused batch-1 decode path "
                "(tt/decode_terminal.py)"
            )

    def sample(self, logits, *, enable_trace=True, tt_out_tok=None, skip_precompile=False, count_tokens=True):
        if self.terminal.is_folded(logits):
            self._check_supported()
        return super().sample(
            logits,
            enable_trace=enable_trace,
            tt_out_tok=tt_out_tok,
            skip_precompile=skip_precompile,
            count_tokens=count_tokens,
        )

    def precompile(self, logits, *, tt_out_tok=None, all_configs=False):
        if not (all_configs and self.terminal.is_folded(logits)):
            return super().precompile(logits, tt_out_tok=tt_out_tok, all_configs=all_configs)
        # The fused path's variants: greedy pick (force-argmax key) and the ttnn.sampling draw.
        saved = self.tt_sampling._force_argmax_sampling
        try:
            for force_argmax in (False, True):
                if force_argmax and not self.tt_sampling._allow_force_argmax_sampling:
                    continue
                self.tt_sampling._force_argmax_sampling = force_argmax
                self._run_sampling(logits, penalties_on=False, tt_out_tok=tt_out_tok, count_tokens=False)
        finally:
            self.tt_sampling._force_argmax_sampling = saved
