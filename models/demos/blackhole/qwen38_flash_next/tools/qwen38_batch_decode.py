# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P1a bring-up driver: batched greedy decode of the PLAIN lane body, one token per lane, on the 1x4 Blackhole mesh.

``Qwen38PlainLaneDecoder`` sits beside an open vLLM adapter (``Qwen38ForCausalLM``, one resident slot).  It allocates a
B-lane state on the adapter's model, builds the lane pagers and the one-slot host pool and warms every program
eagerly (``prepare``, the chain's warm hook: the lane tensors must exist before the chain captures its traces), then
captures the four residue variants of the lane body (``capture`` after the chain opened) and checks the trace
allocation tracker.
A prompt is prefilled on the adapter's B=1 chain, evicted into the host slot and re-admitted into a lane
(``admit``); ``step`` replays the lane trace of the shared residue once and returns one new greedy token per lane.

Residue-aligned admissions only: every lane shares one position residue (position mod 4), so the prompts of one batch
must have lengths of the same residue (the test uses multiples of 4).  Not a scheduler, no MTP, no mid-batch admission.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Sequence

import torch
from loguru import logger
from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker, acknowledge_corruptible

import ttnn
from models.demos.blackhole.qwen38_flash_next.tools import resident_decode
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import Qwen38TracedChain
from models.demos.blackhole.qwen38_flash_next.ttnn.lanes import Qwen38LaneHostPool, Qwen38TTNNLanePager

RESIDUES = 4
WARM_TOKEN_ID = 1000  # any in-vocabulary id: the warm passes write garbage into lane state that a reset clears


def greedy_sampling_params() -> SimpleNamespace:
    """The plugin's ``TTSamplingParams`` for one greedy row (as demo/text_demo.py builds it)."""

    return SimpleNamespace(
        temperature=[0.0],
        top_k=[1],
        top_p=[0.0],
        presence_penalty=[0.0],
        frequency_penalty=[0.0],
        repetition_penalty=[1.0],
        seed=[None],
        num_logprobs=[0],
        enable_log_probs=[False],
    )


@contextmanager
def chain_warm_hook(hook):
    """Inject ``warm_hook`` into ``Qwen38TracedChain.open`` for the adapter's ``initialize_vllm_model`` (which has no
    parameter for it); the chain's own open calls it after its warm pass, before its captures."""

    original = Qwen38TracedChain.__dict__["open"]
    function = original.__func__

    def open_with_hook(cls, construction, **kwargs):
        kwargs.setdefault("warm_hook", hook)
        return function(cls, construction, **kwargs)

    Qwen38TracedChain.open = classmethod(open_with_hook)
    try:
        yield
    finally:
        Qwen38TracedChain.open = original


class Qwen38PlainLaneDecoder:
    """B lanes of the plain lane body over one adapter; see the module docstring."""

    def __init__(self, lanes: int) -> None:
        self.lanes = lanes
        self.vm = self.chain = self.mesh = self.model = None
        self.positions: list[int | None] = [None] * lanes  # P_u: the prompt length, the lane's start position
        self.contexts: list[Any] = [None] * lanes
        self.ls = None
        self.pager_g = self.pager_l = self.pool = None
        self.captures: list[Any] = []
        self.trace_ids: list[int] = []
        self.timing: dict[str, float] = {}
        self.released = False

    # -- the two build phases ----------------------------------------------------------------------

    def prepare(self, chain: Any) -> None:
        """The chain's warm hook (after its warm pass, before its own captures, misses allowed): the lane state, the
        pagers and host slot, the warm admissions and the eager body passes at every residue.  The lane tensors must
        exist before the chain's traces are captured, or the trace allocation tracker refuses those traces."""

        self.chain = chain
        self.mesh = mesh = chain.mesh
        self.model = model = chain.built_target.model
        lanes = self.lanes
        started = time.perf_counter()
        self.ls = ls = model.allocate_lane_state(lanes, indexer_form="wide")
        gl = model.generic_lane_layout(chain.state)
        ll = model.lane_layout(ls)
        self.pager_g = Qwen38TTNNLanePager(mesh, model.mesh_contract, gl)
        self.pager_l = Qwen38TTNNLanePager(mesh, model.mesh_contract, ll, buffers=self.pager_g)
        self.pool = Qwen38LaneHostPool(mesh, gl, slots=1)
        slot = self.pool.slots[0]
        self.pager_g.warm_moves(slot)
        self.pager_l.warm_moves(slot)
        ttnn.synchronize_device(mesh)
        self.timing["allocate_s"] = time.perf_counter() - started

        # Eager warm: every lane admitted once (compiles the admission programs), then one body pass per residue.
        started = time.perf_counter()
        zeros = [0] * lanes
        model.reset_lane_state_inplace(ls, zeros)
        for lane in range(lanes):
            model.admit_lane(ls, lane, 0, rope_shift=0)
        model.reset_lane_state_inplace(ls, zeros)
        contexts: Sequence[Any] = [None] * lanes
        tokens = [WARM_TOKEN_ID] * lanes
        for residue in range(RESIDUES):
            if ls.position.residue != residue:
                raise RuntimeError(f"warm pass {residue}: the position row's residue is {ls.position.residue}")
            contexts = model.write_lane_ple_rows(ls, tokens, contexts)
            model.write_lane_tokens(ls, tokens)
            output = model.forward_decode_lanes(ls)
            candidates, token_row = model.resolve_lane_tokens(output, ls)
            ttnn.synchronize_device(mesh)
            output.release_tensors()
            for tensor in (
                token_row,
                candidates.local_indices,
                candidates.local_values,
                candidates.packed,
                candidates.shard_row,
            ):
                if isinstance(tensor, ttnn.Tensor):
                    ttnn.deallocate(tensor)
        model.reset_lane_state_inplace(ls, zeros)
        ttnn.synchronize_device(mesh)
        self.timing["warm_s"] = time.perf_counter() - started

    def capture(self, vm: Any) -> None:
        """After the chain opened (its traces captured, the miss guard closed): the four residue captures of the lane
        body from reset positions, the tracker over every trace, the lane mirrors back at position 0."""

        self.vm = vm
        chain, mesh, model, ls = self.chain, self.mesh, self.model, self.ls
        if vm.chain is not chain or ls is None:
            raise RuntimeError("capture follows prepare on the same chain")
        if not chain.misses_forbidden:
            raise RuntimeError("the lane captures run under the chain's closed miss guard")
        started = time.perf_counter()
        ttnn.synchronize_device(mesh)
        guard = lambda label: resident_decode.forbid_trace_body_host_io_and_sync(phase=f"lanes {label}")  # noqa: E731
        epilogue = lambda out: model.resolve_lane_tokens(out, ls)  # noqa: E731
        for residue in range(RESIDUES):
            cap = model.capture_decode_lanes(
                ls, residue=residue, guard=guard, epilogue=epilogue, phase_observer=None, cq_id=0
            )
            self.captures.append(cap)
            self.trace_ids.append(next(iter(cap.trace_ids.values())))
        ttnn.synchronize_device(mesh)
        for tensor in (
            ls.token_row,
            ls.ple_rows.embedding_rows,
            ls.position.row,
            ls.position.shift_row,
            *self.pager_g.packs.values(),
            *self.pager_g.kv_stagings,
        ):
            acknowledge_corruptible(tensor)
        for trace_id in (*chain.trace_ids(), *self.trace_ids):
            TraceAllocationTracker.verify_before_replay(mesh, trace_id)
        entries = resident_decode.program_cache_count(mesh)
        logger.info(f"program cache entries after the lane captures {entries} (chain {chain.program_cache_entries})")
        if entries != chain.program_cache_entries:
            raise RuntimeError(f"lane captures compiled programs: {entries} vs {chain.program_cache_entries}")
        model.reset_lane_state_inplace(ls, [0] * self.lanes)  # the mirrors advanced 4 during capture
        ttnn.synchronize_device(mesh)
        self.timing["capture_s"] = time.perf_counter() - started
        logger.info(f"plain lane decoder ready: {self.lanes} lanes, timing {self.timing}")

    # -- admission ---------------------------------------------------------------------------------

    def admit(self, lane: int, prompt_ids: Sequence[int]) -> int:
        """Prefill ``prompt_ids`` on the B=1 chain, move its state into lane ``lane``; returns the first token t0
        (the greedy token after the prompt, to be consumed by the lane's first step)."""

        vm, model, ls = self.vm, self.model, self.ls
        length = len(prompt_ids)
        out, _ = vm.prefill_forward(
            tokens=torch.tensor([list(prompt_ids)], dtype=torch.int32),
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[length],
            start_pos=[0],
            empty_slots=[0],
            sampling_params=greedy_sampling_params(),
        )
        first_token = int(out.reshape(-1)[0])
        context = vm.slot.ple_context
        if len(vm.slot.committed) != length:
            raise RuntimeError(f"adapter committed {len(vm.slot.committed)} tokens, expected {length}")
        slot = self.pool.slots[0]
        self.pager_g.evict(0, slot)
        slot.position = length
        model.admit_lane(ls, lane, length, rope_shift=0)
        self.pager_l.readmit(slot, lane, position_row=None)
        ttnn.synchronize_device(self.mesh)
        vm.release_request(0)
        self.positions[lane] = length
        self.contexts[lane] = context
        return first_token

    # -- decode ------------------------------------------------------------------------------------

    def step(self, tokens: Sequence[int]) -> list[int]:
        """One lane-body replay consuming ``tokens`` (one per lane); returns the lanes' next greedy tokens."""

        model, ls, mesh = self.model, self.ls, self.mesh
        residue = ls.position.residue
        self.contexts = list(model.write_lane_ple_rows(ls, tokens, self.contexts))
        model.write_lane_tokens(ls, tokens)
        ttnn._ttnn_execute_trace(mesh, self.trace_ids[residue], cq_id=0, blocking=True)
        ls.position.advance_replayed()
        row = ttnn.to_torch(ttnn.get_device_tensors(ls.token_row)[0]).reshape(-1)[: self.lanes]
        return [int(v) for v in row.to(torch.float32).round().to(torch.int64).tolist()]

    # -- release -----------------------------------------------------------------------------------

    def release(self) -> None:
        """Lane traces, pagers, lane state, capture objects; the caller then calls ``vm.release_persistent_capture``."""

        if self.released or self.mesh is None:
            return
        self.released = True
        mesh = self.mesh
        ttnn.synchronize_device(mesh)
        mesh.set_program_cache_misses_allowed(True)
        try:
            for trace_id in self.trace_ids:
                ttnn.release_trace(mesh, trace_id)
            self.trace_ids.clear()
            for pager in (self.pager_l, self.pager_g):
                if pager is not None:
                    pager.release()
            if self.ls is not None:
                self.model.release_lane_state(self.ls)
                self.ls = None
            self.captures.clear()
        finally:
            mesh.set_program_cache_misses_allowed(False)
