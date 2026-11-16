# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM adapter: S1a's async multi-user adapter on resident_m5_4's API (S1).

Plan: bench/resident/s1/S1_PLAN_v3.md §1 (accepted); contract: bench/resident/m5/M1_INTERFACE_M5_ADDENDUM_2.md.
S1a's adapter (generator_resident_multi_async_vllm.py, accepted, unchanged) plus:

- The engine: resident_m5_4 by default, one R = 32 program per user-block size, EngineConfig.block = {32: (1, 2, 4,
  8)} (A2). Launches name their block (`launch(R, block=M)`, M in `blocks_for(R)`); `users_for` and `capacity_for`
  replace S0's assumptions (A1, A3).
- Capacity: `capacity_for(32)` (10,240 until K2b) bounds the profile. Warmup refuses a max_seq_len above it, and the
  draft limit and the capacity-edge decline read it.
- Mixed greedy and sampled steps (A4, lane S): the sampling trace ends with greedy_restore.cpp (tokens = greedy_mask ?
  argmax_ids : tokens, one core per chip), so a greedy row takes the hub argmax inside the replay. ttnn.where cannot do
  it: its ternary op needs TILE tensors, and the engine's tokens / argmax_ids / greedy_mask are ROW_MAJOR [1, 1, 1, 32].
  The adapter writes greedy_mask (write_greedy_mask) before every sampled replay whose greedy rows differ from the last
  ones written (an empty set included).
- server.log evidence (S1_PLAN_v3 §1.4): DRAM free per chip after warmup and after every novel prefill shape.
- GEMMA4_RESIDENT_ENGINE_TAPS=0 builds the engine without tap layers (S1_PLAN_v3 §1.5: the CP2 legs run policy never
  with no tap store, and X32 adds the tap term apart). The default keeps the drafter's tap layers.
- GEMMA4_RESIDENT_KV_POOL=b32 sizes the KV pool with B32's budget (generator_vllm.get_max_tokens_all_users, 528,384
  at c32: 4,160 blocks) instead of M1's from max_model_len (296,960 at 10,240: 2,352 blocks), so a comparison with B32
  preempts alike on prefill-heavy load (bench/resident/s1/S1_HOST_RESULTS_ADDENDUM_3.md).
- GEMMA4_RESIDENT_ENGINE_BLOCKS (default "1,2,4,8"): the R = 32 block sizes built. Each m5_4 block model keeps its own
  persistent L1 (~9.4 KB per core), and four of them put the ttnn model's L1 floor (1,457,280) below K2a's CB top and
  the ttnn prefill's CB region (bench/resident/s1/S1_HOST_RESULTS_ADDENDUM_1.md); "1" serves policy never only.
"""

import os
import time

from loguru import logger

from models.demos.gemma4_31b_qb2.tt import resident_taps
from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_async_vllm import Gemma4ResidentMultiAsyncForCausalLM
from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_vllm import MAX_USERS, VERIFY_BLOCKS
from models.demos.gemma4_31b_qb2.tt.generator_resident_vllm import DEFAULT_L1_TOP, RAW_WEIGHT_CACHE
from models.demos.gemma4_31b_qb2.tt.generator_vllm import Gemma4ForCausalLM

RESTORE_KERNEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "greedy_restore.cpp")


class Gemma4ResidentMultiS1ForCausalLM(Gemma4ResidentMultiAsyncForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.greedy_rows = None  # the greedy rows greedy_mask holds, or None before the first write
        self.restore_program = None  # (tensor addresses, greedy_restore.cpp's mesh program descriptor)
        self.warming_prefill = False  # inside M1's prefill bucket warmup
        if os.environ.get("GEMMA4_RESIDENT_ENGINE_TAPS", "1") == "0":
            self.tap_layers = ()

    # ------------------------------------------------------------------ engine

    @classmethod
    def get_max_tokens_all_users(cls, *, max_num_seqs, max_model_len=262144, **kwargs):
        own = super().get_max_tokens_all_users(max_num_seqs=max_num_seqs, max_model_len=max_model_len, **kwargs)
        if os.environ.get("GEMMA4_RESIDENT_KV_POOL", "") != "b32":
            return own
        return max(own, Gemma4ForCausalLM.get_max_tokens_all_users(max_num_seqs=max_num_seqs, **kwargs))

    def _build_engine(self):
        """resident_m5_4: one R = 32 program per block size (M1_INTERFACE_M5_ADDENDUM_2 A2)."""
        import importlib

        package = os.environ.get("GEMMA4_RESIDENT_ENGINE_PACKAGE", "resident_m5_4")
        resident = importlib.import_module(f"models.demos.gemma4_31b_qb2.{package}.engine")
        setup = self.generator.model.shared_setup
        blocks = self._engine_blocks()
        cfg = resident.EngineConfig(
            weight_cache=os.environ.get("GRES_CACHE", RAW_WEIGHT_CACHE),
            rows=self.ROWS,
            max_users=self.max_batch_size,
            capacity=self.max_seq_len,
            tap_layers=self.tap_layers,
            l1_top=int(os.environ.get("GRES_L1_TOP", DEFAULT_L1_TOP)),
            logits=True,
            embedding="resident",
            block={R: blocks for R in self.ROWS},
        )
        pools = resident.KVPools(
            kv=list(self.cache.kv),
            layer_group=list(self.groups),
            scratch=self._group_scratch(),
            rope={
                kind: (setup[("rope", kind)]["decode_cos"], setup[("rope", kind)]["decode_sin"])
                for kind in ("sliding_attention", "full_attention")
            },
        )
        logger.info("Building the resident multi-user engine from {}, rows {}, blocks {}, users {}", package, cfg.rows,
                    cfg.block, cfg.max_users)
        return resident.ResidentEngine(self.mesh, cfg, pools)

    @staticmethod
    def _engine_blocks():
        blocks = tuple(int(x) for x in os.environ.get("GEMMA4_RESIDENT_ENGINE_BLOCKS", "1,2,4,8").split(","))
        if 1 not in blocks or any(b not in VERIFY_BLOCKS for b in blocks):
            raise ValueError(f"GEMMA4_RESIDENT_ENGINE_BLOCKS={blocks}: must include 1, from {VERIFY_BLOCKS}")
        return blocks

    def _launch(self, R, M=1):
        """S1a's launch (deferred tap upkeep whose buffer this launch would overwrite runs first), with the block named
        (A2). The early drain is S1a's _launch, repeated: S1a hands the launch itself to S0, which read cfg.block_for."""
        blocks = tuple(self.engine.blocks_for(R))
        if M not in blocks:
            raise ValueError(f"The engine has no R = {R} program with {M}-row user blocks: {blocks}")
        upcoming = self.engine.launches
        early = [entry for entry in self.deferred_taps if entry[1].launch <= upcoming - 2]
        if early:
            self.deferred_taps = [entry for entry in self.deferred_taps if entry[1].launch > upcoming - 2]
            for state, step, c in early:
                state.deferred -= 1
            for state, step, c in early:
                if self.slots.states.get(state.slot) is state:
                    self._upkeep_rows(state, step, c)
            self.counters["deferred_tap_steps_before_launch"] += len(early)
        self.engine.launch(R, block=M)

    def _capacity(self):
        return int(self.engine.capacity_for(max(self.ROWS)))

    def _spec_limit(self):
        limit = self._capacity()
        return limit if self.drafter is None else min(limit, int(self.drafter.max_positions))

    def warmup_model_decode(self, kv_cache, enable_trace, **kwargs):
        """S0's warmup checks plus A3's: the profile's context fits the program's capacity. Then the DRAM line."""
        if self._engine_blocks() != VERIFY_BLOCKS and os.environ.get("GEMMA4_RESIDENT_SPEC_POLICY") != "never":
            raise ValueError(f"GEMMA4_RESIDENT_ENGINE_BLOCKS={self._engine_blocks()} cannot verify every draft size: "
                             "it serves GEMMA4_RESIDENT_SPEC_POLICY=never only")
        if self.prefill_tokens is None and self.max_seq_len > self._capacity():
            raise ValueError(f"max_seq_len {self.max_seq_len} is above the R = {max(self.ROWS)} program's capacity "
                             f"{self._capacity()} (M1_INTERFACE_M5_ADDENDUM_2 A3: 10,240 until K2b)")
        super().warmup_model_decode(kv_cache, enable_trace, **kwargs)
        self._log_dram("warmup")

    def _log_dram(self, reason):
        logger.info("DRAM free per chip after {}: {} B", reason, resident_taps.dram_free_per_chip(self.mesh))

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        """M1's bucket warmup (generator_resident_vllm.warmup_model_prefill) prefills every bucket from position 0 with
        the same first full-layer block, i.e. the same stable slot key: each bucket's slot is released after its
        prefill, as a finished request's is (S1a's release: tables back to scratch)."""
        self.warming_prefill = True
        try:
            return super().warmup_model_prefill(kv_cache, enable_trace, **kwargs)
        finally:
            self.warming_prefill = False

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        if max(int(x) for x in prompt_lens) > self._capacity():
            raise ValueError(f"Prompt lengths {list(prompt_lens)} pass the capacity {self._capacity()}")
        self._rebind_continuations(kwargs.get("page_tables_per_layer"), kwargs.get("start_pos"),
                                   kwargs.get("empty_slots"))
        shapes = len(self.generator.prefill_shapes)
        out = super().prefill_forward(tokens, page_table, kv_cache, prompt_lens, **kwargs)
        if len(self.generator.prefill_shapes) != shapes:
            self._log_dram(f"prefill {[int(x) for x in prompt_lens]}")
        if self.warming_prefill:
            for slot in kwargs.get("empty_slots") or range(len(prompt_lens)):
                self.release_request(int(slot))
        return out

    def _rebind_continuations(self, page_tables_per_layer, start_pos, empty_slots):
        """The plugin assigns prefill state slots per step (vllm_tt_plugin model_runner._alloc_prefill_state_slots,
        preferring each request's row), so a continuing chunk can arrive with another plugin slot than its last one,
        and a new request in the same step can get the one it held. S0's loop binds in row order and would find that
        slot still held by the continuation (CP2 hold 1, leg 2): every continuation is rebound first, as one
        permutation."""
        if page_tables_per_layer is None or start_pos is None or empty_slots is None:
            return
        source = page_tables_per_layer[self.full_layer]
        moving = {}
        for i, (start, plugin_slot) in enumerate(zip(start_pos, empty_slots)):
            slot = self.slots.by_key.get(self._key(source[i, 0])) if int(start) > 0 else None
            if slot is not None:
                moving[int(plugin_slot)] = slot
        if not moving or all(self.slots.plugin.get(p) == s for p, s in moving.items()):
            return
        engine = set(moving.values())
        self.slots.plugin = {p: s for p, s in self.slots.plugin.items() if s not in engine}
        for plugin_slot, slot in moving.items():
            other = self.slots.plugin.get(plugin_slot)
            if other is not None and other in self.slots.states:
                raise RuntimeError(f"Plugin slot {plugin_slot} is bound to engine slot {other}, not {slot}: "
                                   f"{self.slots.plugin}")
            self.slots.plugin[plugin_slot] = slot
        self.counters["prefill_rebinds"] += 1

    # ------------------------------------------------------------------ mixed greedy and sampled rows (A4)

    def _capture_sampling(self):
        """M1's capture (generator_resident_vllm._capture_sampling) with the greedy restore at the end of the trace:
        sampler(logits) -> tokens, seeds + 1, then greedy_restore.cpp."""
        import ttnn

        start = time.perf_counter()
        entry = {"logits": self.engine.logits, "tokens": self.engine.tokens}
        seeds = self.generator.sampler.seeds_tt_tensor
        saved = (ttnn.clone(seeds), ttnn.clone(self.engine.tokens))
        self.generator._sample_step(entry)
        self._greedy_restore_op()
        ttnn.copy(saved[0], seeds)
        ttnn.copy(saved[1], self.engine.tokens)
        del saved
        trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.generator._sample_step(entry)
        self._greedy_restore_op()
        ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
        self.sampling_trace = trace
        self.sampling_captures.append(time.perf_counter() - start)
        logger.info("Sampling trace (with the greedy restore) captured in {:.1f} ms (host)",
                    self.sampling_captures[-1] * 1000)

    def _greedy_restore_op(self):
        e = self.engine
        self._restore_program([e.greedy_mask, e.argmax_ids, e.tokens])

    def _restore_program(self, tensors):
        """greedy_restore.cpp on (greedy_mask, argmax_ids, tokens); the program is rebuilt when their addresses change
        (an engine rebuild)."""
        import ttnn

        addresses = tuple(t.buffer_address() for t in tensors)
        if self.restore_program is None or self.restore_program[0] != addresses:
            self.restore_program = (addresses, self._restore_descriptor(addresses))
        ttnn.generic_op(tensors, self.restore_program[1])

    def _restore_descriptor(self, addresses):
        """one core per chip, as resident_m5_4's table scatter (engine._scatter)"""
        import ttnn

        core = ttnn.CoreCoord(0, 0)
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        cb = ttnn.CBDescriptor(total_size=3 * 128, core_ranges=cores, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=3 * 128)])
        args = ttnn.RuntimeArgs()
        args[core.x][core.y] = list(addresses)
        kernel = ttnn.KernelDescriptor(
            kernel_source=RESTORE_KERNEL, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=cores,
            compile_time_args=[], runtime_args=args,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1,
                                                     noc=ttnn.NOC.RISCV_1_default))
        program = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])
        mesh_program = ttnn.MeshProgramDescriptor()
        for chip in range(self.mesh.get_num_devices()):
            coord = ttnn.MeshCoordinate(0, chip)
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program
        return mesh_program

    def _set_greedy_mask(self, rows):
        """greedy_mask = rows (the engine's write_greedy_mask), written only when it differs from what the device holds."""
        rows = tuple(sorted(rows))
        if rows == self.greedy_rows:
            return
        self.engine.write_greedy_mask([r in rows for r in range(MAX_USERS)])
        self.greedy_rows = rows
        self.counters["greedy_mask_writes"] += 1

    def _restore_greedy(self, rows):
        """The restore runs inside the sampling trace (_capture_sampling); greedy_mask was written before the replay."""
        if tuple(sorted(rows)) != self.greedy_rows:
            raise RuntimeError(f"greedy_mask holds {self.greedy_rows}, the step's greedy rows are {sorted(rows)}")

    def decode_forward(self, tokens, start_pos, page_table, kv_cache, *, sampling_params=None, spec_mode=None,
                       **kwargs):
        """Before a sampled ordinary step, greedy_mask = its greedy live rows (launch row r is decode row r)."""
        if spec_mode is None and sampling_params is not None:
            batch = self.max_batch_size
            positions = [int(p) for p in start_pos.reshape(-1)[:batch]]
            temps = self._rows(sampling_params.temperature, batch)
            if any(p >= 0 and t > 0 for p, t in zip(positions, temps)):
                self._set_greedy_mask([r for r, (p, t) in enumerate(zip(positions, temps)) if p >= 0 and t == 0])
        return super().decode_forward(tokens, start_pos, page_table, kv_cache, sampling_params=sampling_params,
                                      spec_mode=spec_mode, **kwargs)

    def _configure_verify_sampling(self, sampling_params, blocks, M, R):
        """S0's verify sampling, then greedy_mask = the greedy users' block rows."""
        super()._configure_verify_sampling(sampling_params, blocks, M, R)
        temps = self._rows(sampling_params.temperature, self.max_batch_size)
        self._set_greedy_mask([u * M + j for u, b, _, n, _, _ in blocks if temps[b] == 0 for j in range(n + 1)])
