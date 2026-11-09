# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefix caching and chunked prefill of the MTP lanes adapter (``QWEN38_MAMBA_SPEC=1``) on device, in-process.

``Qwen38MTPForCausalLM`` with 2 lanes, K = 3, block size 1024, allocated context 32768 and a pool of 112 blocks (KV blocks
1..71, state blocks 72..111), driven as the plugin drives it: ``allocate_kv_cache_per_layer`` with the per-layer entries the
plugin builds from ``get_kv_cache_spec`` (48 layers, three Mamba groups of 12 GDN layers named ``model.layers.<i>.linear_attn``,
slot j = position j inside the group), both warm-ups (eager, then traced), then ``prefill_forward`` with a ``mamba_state``
(windows, checkpoint blocks and positions) and ``intermediate_prefill_mask``, ``copy_kv_cache_blocks`` for a prefix hit,
``decode_forward`` and ``propose_draft_tokens`` with the plugin's accept walk.  The whole run is under the chain's closed
program-cache miss guard that the warm-ups leave (T6).

Gates (one test, every gate reported):
  T1  prefix hit (S = 2048 from a checkpoint block) equals a cold prefill of the same prompt with the same checkpoint split:
      first token and the first 64 tokens of the greedy stream through the lanes.
  T2  3 chunked calls (E = 2048 and 4096 intermediate, then final; another request's prefill between the calls) equal one
      unchunked call with the same checkpoint splits: first token and the first 64 tokens.
  T3  a decoding lane's stream is unchanged by chunks of another request between its passes (compared up to the pass where the
      other request joins the lanes; the rest is reported).
  T4  hybrid single: a lone request on the chain, a chunk of a second request moves it to its lane; its stream equals the
      same request's stream alone on the chain (the lane stream of T3 is reported next to it).
  T5  mean accepted drafts per pass over 128 tokens: hit vs cold (reported, not gated).

Run:  pytest models/demos/blackhole/qwen38_flash_next/tests/test_vllm_mtp_prefix_device.py -s
Env:  MESH_DEVICE=P150x8, HF_MODEL, TT_CACHE_PATH, TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0, QWEN38_MAMBA_SPEC=1,
      QWEN38_VLLM_MTP_K=3, QWEN38_MTP_HYBRID_SINGLE=1 (the test turns the hybrid off per gate)
"""

import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_batch_buckets_device import needle_prompts
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import MAMBA_STATE_FLOATS, QSA_SPEC_HEAD_SIZE
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm_mtp import PLACEHOLDER_TOKEN_ID, Qwen38MTPForCausalLM
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

ALLOCATED = 32768
MAX_SEQ_LEN = ALLOCATED - RESIDENT_CONTEXT_HEADROOM
LANES = 2
BLOCK = 1024
NUM_BLOCKS = 112
KV_BLOCKS = list(range(1, 72))
STATE_BLOCKS = list(range(72, 112))
REQUEST_KV_BLOCKS = 8
GROUPS = 3
STREAM_TOKENS = 128
GATE_TOKENS = 64
HIT_AT = 2048
HIT_LENGTH = 3000
CHUNK_LENGTH = 5000
DECODE_LENGTH = 1500
SOLO_TOKENS = 96  # the lane stream T3 and T4 compare (the pre-join part is gated)


def step(name: str) -> None:
    logger.info(f"PC3C step: {name}")


def first_diff(a: list[int], b: list[int], limit: int | None = None) -> int | None:
    n = min(len(a), len(b)) if limit is None else min(limit, len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if limit is None and len(a) != len(b):
        return n
    return None


# -- the plugin's view of the model ------------------------------------------------------------------------------------


def group_layers() -> list[list[int]]:
    """vLLM's Mamba groups: runs of 12 GDN layers in layer order."""

    gdn = [layer for layer in range(48) if layer % 4 != 3]
    return [gdn[g * 12 : (g + 1) * 12] for g in range(GROUPS)]


def kv_entries() -> list[tuple]:
    """The plugin's ``(shape, dtype, tensor_idx)`` per layer: slot j = QSA layer 4j+3 and the j-th layer of every group."""

    position = {layer: j for layers in group_layers() for j, layer in enumerate(layers)}
    out = []
    for layer in range(48):
        if layer % 4 == 3:
            out.append(((NUM_BLOCKS, 1, BLOCK, QSA_SPEC_HEAD_SIZE), torch.bfloat16, layer // 4))
        else:
            out.append((((MAMBA_STATE_FLOATS,),), (torch.float32,), position[layer]))
    return out


class Blocks:
    """vLLM's block pool as the test needs it: KV blocks and (separate) state blocks, freed when a request ends."""

    def __init__(self) -> None:
        self.kv = list(KV_BLOCKS)
        self.state = list(STATE_BLOCKS)

    def take_kv(self, count: int) -> list[int]:
        if len(self.kv) < count:
            raise RuntimeError("the test's KV block pool is empty")
        taken, self.kv = self.kv[:count], self.kv[count:]
        return taken

    def take_state(self) -> list[int]:
        if len(self.state) < GROUPS:
            raise RuntimeError("the test's state block pool is empty")
        taken, self.state = self.state[:GROUPS], self.state[GROUPS:]
        return taken

    def give(self, kv=(), state=()) -> None:
        self.kv = sorted(self.kv + [b for b in kv])
        self.state = sorted(self.state + [b for b in state])


@dataclass
class Req:
    name: str
    slot: int
    prompt: list[int]
    kv: list[int] = field(default_factory=list)
    own_kv: list[int] = field(
        default_factory=list
    )  # the blocks this request took itself (the prefix blocks are shared)
    windows: list[int] = field(default_factory=list)  # one running-state block per group
    checkpoints: dict[int, list[int]] = field(default_factory=dict)  # n -> one block per group


class Harness:
    def __init__(self, model: Qwen38MTPForCausalLM) -> None:
        self.model = model
        self.blocks = Blocks()
        self.lanes = model.lanes
        self.k = model.drafts
        self.width = model.paging.max_blocks
        self.layer_names = [tuple(f"model.layers.{layer}.linear_attn" for layer in layers) for layers in group_layers()]

    # requests

    def new_req(self, name: str, slot: int, prompt: list[int], shared_kv: list[int] = (), checkpoints=()) -> Req:
        own = self.blocks.take_kv(REQUEST_KV_BLOCKS - len(shared_kv))
        req = Req(name, slot, list(prompt), kv=list(shared_kv) + own, own_kv=own)
        req.windows = self.blocks.take_state()
        for n in checkpoints:
            req.checkpoints[n] = self.blocks.take_state()
        self.model.zero_kv_cache_blocks(
            kv_cache=None, block_ids=own
        )  # the plugin zeroes a fresh block before its first use
        return req

    def free(self, req: Req) -> None:
        self.blocks.give(
            kv=req.own_kv, state=[*req.windows, *[b for blocks in req.checkpoints.values() for b in blocks]]
        )

    # the prefill call

    def mamba_state(self, reqs: list[Req]):
        rows = len(reqs)
        width = max([1, *(len(r.checkpoints) for r in reqs)])
        any_checkpoint = any(r.checkpoints for r in reqs)
        groups = []
        for g in range(GROUPS):
            windows = torch.zeros(rows, 1 + self.k, dtype=torch.int32)
            blocks = torch.zeros(rows, width, dtype=torch.int32) if any_checkpoint else None
            for i, r in enumerate(reqs):
                windows[i, 0] = r.windows[g]
                for j, n in enumerate(sorted(r.checkpoints)):
                    blocks[i, j] = r.checkpoints[n][g]
            groups.append(
                SimpleNamespace(layer_names=self.layer_names[g], windows=windows, checkpoint_blocks=blocks, chain=None)
            )
        positions = None
        if any_checkpoint:
            positions = torch.zeros(rows, width, dtype=torch.int32)
            for i, r in enumerate(reqs):
                for j, n in enumerate(sorted(r.checkpoints)):
                    positions[i, j] = n
        return SimpleNamespace(groups=tuple(groups), accepted_counts=None, checkpoint_positions=positions)

    def prefill(self, items: list[tuple[Req, int, int, bool]]) -> list[int | None]:
        """One ``prefill_forward``: items ``(request, start, end, final)``; returns the first tokens (None: intermediate)."""

        reqs = [item[0] for item in items]
        width = max(end for _, _, end, _ in items)
        tokens = torch.zeros(len(items), width, dtype=torch.int32)
        table = torch.zeros(len(items), self.width, dtype=torch.int32)
        for i, (req, _start, end, _final) in enumerate(items):
            tokens[i, :end] = torch.tensor(req.prompt[:end], dtype=torch.int32)
            table[i, : len(req.kv)] = torch.tensor(req.kv, dtype=torch.int32)
        out, _ = self.model.prefill_forward(
            tokens=tokens,
            page_table=table,
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[end for _, _, end, _ in items],
            start_pos=[start for _, start, _, _ in items],
            empty_slots=[req.slot for req in reqs],
            mamba_state=self.mamba_state(reqs),
            intermediate_prefill_mask=[not final for _, _, _, final in items],
        )
        return [int(out[i, 0].argmax()) if final else None for i, (_, _, _, final) in enumerate(items)]

    def cold(self, req: Req, length: int | None = None) -> int:
        end = len(req.prompt) if length is None else length
        return self.prefill([(req, 0, end, True)])[0]

    def hit(self, req: Req, source: Req, start: int) -> int:
        """The plugin's prefix hit: the state block of the cached checkpoint is copied into the request's running-state block,
        then the chunk ``[start, len(prompt))`` runs."""

        self.model.copy_kv_cache_blocks(
            kv_cache=None, copies=[(source.checkpoints[start][g], req.windows[g]) for g in range(GROUPS)]
        )
        return self.prefill([(req, start, len(req.prompt), True)])[0]

    def release(self, req: Req) -> None:
        self.model.release_request(req.slot)

    # decode as the plugin's loop does

    def new_live(self, req: Req, first: int) -> "Live":
        return Live(req, first, len(req.prompt))

    def decode_pass(self, lives: list["Live"]) -> None:
        k, lanes = self.k, self.lanes
        tokens = torch.full((lanes, k + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        positions = torch.full((lanes, k + 1), -1, dtype=torch.int32)
        valid = torch.zeros(lanes, dtype=torch.int32)
        counts = torch.ones(lanes, dtype=torch.int32)
        table = torch.zeros(lanes, self.width, dtype=torch.int32)
        for lv in lives:
            s = lv.req.slot
            tokens[s, 0] = lv.last
            for i, draft in enumerate(lv.drafts):
                tokens[s, 1 + i] = draft
            if lv.fresh:
                positions[s, 0] = lv.pos
            else:
                positions[s] = torch.arange(lv.pos, lv.pos + k + 1, dtype=torch.int32)
            valid[s], counts[s] = lv.nvalid, lv.accepted_prev
            table[s, : len(lv.req.kv)] = torch.tensor(lv.req.kv, dtype=torch.int32)
        verified = self.model.decode_forward(
            tokens=tokens,
            start_pos=positions,
            spec_mode="argmax_ids",
            num_valid_drafts=valid,
            accepted_counts=counts,
            page_table=table,
        )
        committed_block = torch.zeros(lanes, k + 1, dtype=torch.int32)
        new_counts = torch.ones(lanes, dtype=torch.int32)
        walked = {}
        for lv in lives:
            s = lv.req.slot
            argmax = [int(t) for t in verified.argmax_ids[s].tolist()]
            committed = [argmax[0]]
            for i in range(lv.nvalid):  # the plugin's accept walk
                if int(tokens[s, 1 + i]) == argmax[i]:
                    committed.append(argmax[i + 1])
                else:
                    break
            walked[s] = committed
            committed_block[s, : len(committed)] = torch.tensor(committed, dtype=torch.int32)
            new_counts[s] = len(committed)
        offered = self.model.propose_draft_tokens(k, committed_block, positions, new_counts)
        for lv in lives:
            s = lv.req.slot
            committed = walked[s]
            if lv.nvalid == k:
                lv.accepts.append(len(committed) - 1)
            lv.tokens.extend(committed)
            lv.pos += len(committed)  # the next block starts at its first token: the last committed one
            lv.last = committed[-1]
            lv.accepted_prev = len(committed)
            count = int(offered.num_valid[s])
            lv.drafts = offered.draft_token_ids[s].tolist() if count == k else []
            lv.nvalid = len(lv.drafts)
            lv.fresh = False

    def decode_until(self, lives: list["Live"], count: int) -> None:
        while min(len(lv.tokens) for lv in lives) < count:
            self.decode_pass(lives)


class Live:
    """One request's plugin-side stream: ``tokens`` = the first token and every committed token; ``pos`` = the position of
    the next block's first token (``last``)."""

    def __init__(self, req: Req, first: int, length: int) -> None:
        self.req = req
        self.tokens = [first]
        self.last = first
        self.pos = length
        self.drafts: list[int] = []
        self.nvalid = 0
        self.accepted_prev = 1
        self.fresh = True
        self.accepts: list[int] = []

    def mean_accepted(self) -> float:
        return sum(self.accepts) / len(self.accepts) if self.accepts else float("nan")


def stream(h: Harness, req: Req, first: int, count: int = STREAM_TOKENS) -> Live:
    lv = h.new_live(req, first)
    h.decode_until([lv], count)
    return lv


# -- the gates ---------------------------------------------------------------------------------------------------------


class Report:
    def __init__(self) -> None:
        self.results: dict[str, tuple[bool, str]] = {}

    def gate(self, name: str, ok: bool, message: str) -> None:
        self.results[name] = (ok, message)
        logger.info(f"GATE {name} {'PASS' if ok else 'FAIL'}: {message}")


def t1_hit_vs_cold(h: Harness, prompts, report: Report) -> dict:
    t0 = time.time()
    a_prompt, other = prompts[0], prompts[1]
    b_prompt = a_prompt[:HIT_AT] + other[HIT_AT:HIT_LENGTH]
    assert len(b_prompt) == HIT_LENGTH
    step("T1 run A (cold, checkpoint at 2048)")
    a = h.new_req("A", 0, a_prompt, checkpoints=[HIT_AT])
    first_a = h.cold(a)
    h.release(a)
    step("T1 run B (hit from A's checkpoint)")
    b = h.new_req("B", 0, b_prompt, shared_kv=a.kv[:2])
    first_b = h.hit(b, a, HIT_AT)
    live_b = stream(h, b, first_b)
    h.release(b)
    step("T1 run C (cold, same checkpoint split)")
    c = h.new_req("C", 0, b_prompt, checkpoints=[HIT_AT])
    first_c = h.cold(c)
    live_c = stream(h, c, first_c)
    h.release(c)
    step("T1 run C0 (cold, no split; first token only)")
    c0 = h.new_req("C0", 0, b_prompt)
    first_c0 = h.cold(c0)
    h.release(c0)
    for req in (a, b, c, c0):
        h.free(req)
    diff = first_diff(live_b.tokens, live_c.tokens, GATE_TOKENS)
    diff_all = first_diff(live_b.tokens, live_c.tokens, STREAM_TOKENS)
    ok = first_b == first_c and diff is None
    report.gate(
        "T1",
        ok,
        f"first tokens A {first_a}, B(hit) {first_b}, C(cold+split) {first_c}, C0(cold, no split) {first_c0}; first differing "
        f"index of the {GATE_TOKENS}-token streams B/C {diff} (of {STREAM_TOKENS}: {diff_all}); {time.time() - t0:.1f} s",
    )
    return {"hit": live_b, "cold": live_c}


def t5_acceptance(live: dict, report: Report) -> None:
    hit, cold = live["hit"], live["cold"]
    report.gate(
        "T5",
        True,
        f"mean accepted drafts per pass over {STREAM_TOKENS} tokens (passes that carried K drafts): hit "
        f"{hit.mean_accepted():.3f} over {len(hit.accepts)} passes, cold {cold.mean_accepted():.3f} over {len(cold.accepts)} passes",
    )


def chunked(h: Harness, req: Req, between=None) -> int:
    """The prompt as 3 calls (E = 2048 and 4096 intermediate, then final); ``between(i)`` runs after call i."""

    ends = [2048, 4096, len(req.prompt)]
    start = 0
    first = None
    for i, end in enumerate(ends):
        final = end == len(req.prompt)
        out = h.prefill([(req, start, end, final)])[0]
        if final:
            first = out
        if between is not None and not final:
            between(i)
        start = end
    return first


def t2_chunked_vs_unchunked(h: Harness, prompts, filler, report: Report) -> Live:
    t0 = time.time()
    prompt = prompts[0]
    step("T2 run D (unchunked, checkpoints 2048 and 4096)")
    d = h.new_req("D", 0, prompt, checkpoints=[2048, 4096])
    first_d = h.cold(d)
    live_d = stream(h, d, first_d)
    h.release(d)
    step("T2 run E (3 chunked calls with another prefill between)")
    e = h.new_req("E", 0, prompt)
    x = h.new_req("X", 1, filler)

    def other_prefill(_i: int) -> None:
        h.cold(x)
        h.release(x)

    first_e = chunked(h, e, other_prefill)
    live_e = stream(h, e, first_e)
    h.release(e)
    for req in (d, e, x):
        h.free(req)
    diff = first_diff(live_d.tokens, live_e.tokens, GATE_TOKENS)
    diff_all = first_diff(live_d.tokens, live_e.tokens, STREAM_TOKENS)
    report.gate(
        "T2",
        first_d == first_e and diff is None,
        f"first tokens D(unchunked) {first_d}, E(chunked) {first_e}; first differing index of the {GATE_TOKENS}-token streams "
        f"{diff} (of {STREAM_TOKENS}: {diff_all}); accepted drafts per pass D {live_d.mean_accepted():.3f}, "
        f"E {live_e.mean_accepted():.3f}; {time.time() - t0:.1f} s",
    )
    return live_d


def t3_decode_between_chunks(h: Harness, prompts, long_prompt, live_d: Live, report: Report) -> list[int]:
    t0 = time.time()
    h.model.mtp_hybrid_single = False
    a_prompt = prompts[2]
    step("T3 solo stream of the decoding request")
    a_solo = h.new_req("A-solo", 0, a_prompt)
    solo = stream(h, a_solo, h.cold(a_solo), SOLO_TOKENS)
    h.release(a_solo)
    h.free(a_solo)
    step("T3 mixed run")
    a = h.new_req("A", 0, a_prompt)
    e = h.new_req("E", 1, long_prompt)
    live_a = h.new_live(a, h.cold(a))
    passes_before_join = 0

    def passes(count: int) -> None:
        nonlocal passes_before_join
        for _ in range(count):
            h.decode_pass([live_a])
            passes_before_join += 1

    passes(2)
    ends = [2048, 4096]
    start = 0
    for end in ends:
        h.prefill([(e, start, end, False)])
        passes(3)
        start = end
    pre_join = len(live_a.tokens)
    first_e = h.prefill([(e, start, len(long_prompt), True)])[0]
    live_e = h.new_live(e, first_e)
    both = [live_a, live_e]
    while len(live_a.tokens) < SOLO_TOKENS or len(live_e.tokens) < GATE_TOKENS:
        h.decode_pass(both)
    h.release(a)
    h.release(e)
    for req in (a, e):
        h.free(req)
    diff_pre = first_diff(live_a.tokens, solo.tokens, pre_join)
    diff_all = first_diff(live_a.tokens, solo.tokens, SOLO_TOKENS)
    diff_e = first_diff(live_e.tokens, live_d.tokens, GATE_TOKENS)
    report.gate(
        "T3",
        diff_pre is None,
        f"{pre_join} tokens of A decoded over {passes_before_join} passes while E's 2 chunks ran between them; first differing "
        f"index vs A alone {diff_pre} (up to the join), {diff_all} (up to {SOLO_TOKENS}, both lanes live after the join); "
        f"E's first token {first_e} vs D {live_d.tokens[0]}, E stream vs D first differing index {diff_e}; {time.time() - t0:.1f} s",
    )
    return solo.tokens


def t4_hybrid_move(h: Harness, prompts, long_prompt, lane_solo: list[int], report: Report) -> None:
    t0 = time.time()
    h.model.mtp_hybrid_single = True
    a_prompt = prompts[2]
    step("T4 solo stream of the lone request on the chain")
    a_solo = h.new_req("A-hybrid-solo", 0, a_prompt)
    first = h.cold(a_solo)
    hybrid_slot = h.model.mtp_hybrid_slot
    solo = stream(h, a_solo, first, SOLO_TOKENS)
    solo_on_chain = h.model.mtp_hybrid_slot == 0
    h.release(a_solo)
    h.free(a_solo)
    step("T4 hybrid request moved by a chunk of a second request")
    a = h.new_req("A", 0, a_prompt)
    e = h.new_req("E", 1, long_prompt)
    live_a = h.new_live(a, h.cold(a))
    on_chain = h.model.mtp_hybrid_slot == 0
    for _ in range(2):
        h.decode_pass([live_a])
    still_on_chain = h.model.mtp_hybrid_slot == 0
    h.prefill([(e, 0, 1024, False)])
    moved = h.model.mtp_hybrid_slot is None
    h.decode_until([live_a], SOLO_TOKENS)
    h.release(a)
    for req in (a, e):
        h.free(req)
    diff = first_diff(live_a.tokens, solo.tokens, SOLO_TOKENS)
    diff_lane = first_diff(live_a.tokens, lane_solo, SOLO_TOKENS)
    ok = diff is None and hybrid_slot == 0 and solo_on_chain and on_chain and still_on_chain and moved
    report.gate(
        "T4",
        ok,
        f"hybrid slot after the solo prefill {hybrid_slot}, solo stayed on the chain {solo_on_chain}; mixed run: on the chain "
        f"after prefill {on_chain}, after 2 passes {still_on_chain}, moved by the chunk {moved}; first differing index vs the "
        f"chain-only stream {diff}, vs the lanes-only stream of T3 {diff_lane} (of {SOLO_TOKENS}); {time.time() - t0:.1f} s",
    )


def run_gate(name: str, report: Report, model, fn, *args):
    try:
        return fn(*args)
    except BaseException as error:  # noqa: BLE001  (every gate is reported; the miss guard shows as its error text)
        text = "".join(traceback.format_exception(error))
        miss = "miss" in str(error).lower()
        report.gate(name, False, f"exception {type(error).__name__}: {error} (program-cache miss text: {miss})")
        logger.error(f"GATE {name} traceback:\n{text}")
        if getattr(model, "batch_failed", False):
            logger.error("the adapter marked its lane state as not trusted: the remaining gates are skipped")
            raise
        return None


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(5400)
def test_prefix_caching_and_chunked_prefill(mesh_device, monkeypatch):
    checkpoint: Path = checkpoint_directory()
    monkeypatch.setenv("QWEN38_MAMBA_SPEC", "1")
    if not os.environ.get("QWEN38_VLLM_MTP_K"):
        monkeypatch.setenv("QWEN38_VLLM_MTP_K", "3")
    if not os.environ.get("QWEN38_MTP_HYBRID_SINGLE"):
        monkeypatch.setenv("QWEN38_MTP_HYBRID_SINGLE", "1")
    mesh_device.enable_program_cache()

    step("prompts")
    hit_prompts, _, _ = needle_prompts(checkpoint, 3, HIT_LENGTH)
    long_prompts, _, _ = needle_prompts(checkpoint, 2, CHUNK_LENGTH)
    decode_prompts, _, _ = needle_prompts(checkpoint, 3, DECODE_LENGTH)
    assert [len(p) for p in hit_prompts] == [HIT_LENGTH] * 3
    assert [len(p) for p in long_prompts] == [CHUNK_LENGTH] * 2
    assert [len(p) for p in decode_prompts] == [DECODE_LENGTH] * 3

    step("initialize_vllm_model")
    model = Qwen38MTPForCausalLM.initialize_vllm_model(
        SimpleNamespace(_name_or_path=str(checkpoint)), mesh_device, max_batch_size=LANES, max_seq_len=MAX_SEQ_LEN
    )
    assert model.lanes_mode and model.lanes == LANES and model.drafts == 3 and model.allocated_context == ALLOCATED
    assert Qwen38MTPForCausalLM.model_capabilities["supports_prefix_caching"] is True
    assert Qwen38MTPForCausalLM.model_capabilities["supports_chunked_prefill"] is True
    report = Report()
    try:
        step("allocate_kv_cache_per_layer")
        model.kv_cache = model.allocate_kv_cache_per_layer(kv_entries())
        assert model.block_size == BLOCK and model.num_blocks == NUM_BLOCKS

        width = -(-MAX_SEQ_LEN // BLOCK)
        prefill_kwargs = dict(kv_cache=None, can_sample_on_device=False)
        decode_kwargs = dict(kv_cache=None, max_batch_size=LANES, num_blocks=width, can_sample_on_device=True)
        step("warmup_model_prefill eager")
        model.warmup_model_prefill(enable_trace=False, **prefill_kwargs)
        step("warmup_model_decode eager")
        model.warmup_model_decode(enable_trace=False, **decode_kwargs)
        if hasattr(model, "already_warmed_up_prefill"):
            model.already_warmed_up_prefill = False
        step("warmup_model_prefill traced")
        model.warmup_model_prefill(enable_trace=True, **prefill_kwargs)
        step("warmup_model_decode traced")
        t0 = time.time()
        model.warmup_model_decode(enable_trace=True, **decode_kwargs)
        step(f"warm-ups done ({time.time() - t0:.1f} s); the miss guard is closed from here")
        assert model.chain.misses_forbidden

        h = Harness(model)
        model.mtp_hybrid_single = False  # T1..T3 run every request on the lanes

        live = run_gate("T1", report, model, t1_hit_vs_cold, h, hit_prompts, report)
        if live is not None:
            t5_acceptance(live, report)
        live_d = run_gate("T2", report, model, t2_chunked_vs_unchunked, h, long_prompts, decode_prompts[1], report)
        lane_solo = None
        if live_d is not None:
            lane_solo = run_gate(
                "T3", report, model, t3_decode_between_chunks, h, decode_prompts, long_prompts[0], live_d, report
            )
        if lane_solo is not None:
            run_gate("T4", report, model, t4_hybrid_move, h, decode_prompts, long_prompts[0], lane_solo, report)
        misses = [name for name, (ok, text) in report.results.items() if not ok and "miss text: True" in text]
        logger.info(
            f"T6 miss guard: {'program-cache miss in ' + str(misses) if misses else 'no program-cache miss in any gate'}"
        )
    finally:
        step("release_persistent_capture")
        model.release_persistent_capture()
        step("release_persistent_capture done")
    for name, (ok, text) in sorted(report.results.items()):
        logger.info(f"RESULT {name} {'PASS' if ok else 'FAIL'}: {text}")
    failed = [name for name, (ok, _) in report.results.items() if not ok]
    assert not failed and {"T1", "T2", "T3", "T4", "T5"} <= set(report.results), f"gates failed or not run: {failed}"
