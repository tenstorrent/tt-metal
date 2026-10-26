# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P2a device test: the vLLM adapter's batch mode driven exactly as vllm-tt-plugin drives it, in-process.

``Qwen38ForCausalLM.initialize_vllm_model(max_batch_size=8, max_seq_len=32704)``, ``allocate_kv_cache`` with the plugin's
shape ``(num_blocks, 1, block_size, 128)`` (``num_blocks`` from ``get_max_tokens_all_users`` the way the worker derives it),
both warmups (eager, then traced), then a small host scheduler that plays the plugin's runner: vLLM-style block ids (block 0
never used, shuffled, blocks grow as a sequence grows, freed on finish), state slots chosen as ``_alloc_prefill_state_slots``
chooses them, ``prefill_forward`` with 1-3 rows and ``empty_slots``, ``decode_forward`` with the live rows packed first,
padding rows (token 0, ``start_pos`` -1, zero page row), the ``slot_remap`` the plugin derives (the finished requests' holes
are condensed, and every fifth step the live rows are also shuffled), ``note_state_slots_moved`` after an accepted decode,
``release_request`` when a request finishes.  16 requests of mixed prompt length and token budget arrive over time.

Every request's greedy tokens are compared with its B=1 reference.  The references come from the single-slot adapter
(``max_batch_size=1``) in a separate process (id ``ref_b1``, which writes a fixture JSON: two adapters do not share a
process's DRAM); the batch ids read it.  Gate: the first 8 tokens equal the reference and the pass key is in the output
of every request with at least 32 new tokens; the exact-match count and the first divergence are logged.

Ids: ``ref_b1`` (run first), ``contract_b8_bs64``, ``contract_b8_bs32``; ``m1_b8`` and ``m1_b32`` measure the free DRAM per
device after the batch build (context 32768, a small pool) for the token budget; ``cap_b32`` builds B=32 with the pool
``get_max_tokens_all_users`` gives (the DRAM budget's cap), warms up and serves a 30000-token prompt, an 8000-token one
and a short one (the cap's margin holds under the prefill's transients).

P3d-4 (``QWEN38_PREFILL_PACK=1``, optionally ``QWEN38_PACK_FORCE=1``): the same ids run with the pack on.  A row that the planner
put in a pack (recorded from ``decoder.admit_pack``) is gated on its FIRST token only: equal to the B=1 reference, or the
reference token is the pack's top-2 and the top-1 / top-2 logit margin is under ``PACK_MARGIN`` (a pack is the tolerance class of the
single path, not bitwise); its later tokens may diverge (reported).  Every other row keeps the exact-prefix gate.  Ids
``ref_pack32`` (B=1 references of the B=32 scenario into ``p3d4-pack32-refs.json``; knob off) and ``contract_b32_pack`` (B=32, 24 short
prompts and two long ones arrive in ONE prefill call, so S=128 packs of 16 and 8 users run and lanes 16..31 decode; pack on).
P3e: the pack is on by default with 8 or more lanes, so the batch ids run with it unless ``QWEN38_PREFILL_PACK=0`` (the gate
follows ``model.decoder.pack_enabled``); ``m1_*`` counts the pack's DRAM in ``lane_states_bytes`` when it is on (``pack_enabled``
in the report; run them with ``QWEN38_PREFILL_PACK=0`` to measure the lane states alone).

Run:  pytest models/demos/blackhole/qwen38_flash_next/tests/test_vllm_batch_contract_device.py -v -s -k ref_b1
Env:  QWEN38_HARDWARE_PROFILE=tt-quietbox-2, TT_METAL_TRACE_ALLOC_TRACKING=1, MODEL_WEIGHTS_DIR, QWEN38_CACHE_ROOT,
      TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0; ``QWEN38_VLLM_BATCH_REFS`` names the reference fixture (default
      ``<QWEN38_CACHE_ROOT>/demo-context/p2a-vllm-batch-refs.json``).
"""

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_flash_next.demo.long_prompt import build_prompt_ids, load_context_text
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    PROMPT_FILE,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import Qwen38ForCausalLM, greedy_batch_sampling_params
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

CONTEXT = 32768
MAX_SEQ_LEN = CONTEXT - RESIDENT_CONTEXT_HEADROOM
EXACT_PREFIX = 8
PASS_KEY_MIN_TOKENS = 32
MAX_PREFILL_ROWS = 3
FORCED_SHUFFLE_EVERY = 5  # decode steps between the extra row shuffles (on top of the condense a finish causes)
PACK_MARGIN = (
    0.5  # a packed row's first token may differ from the reference when the reference is its top-2 within this margin
)

# P3d-4 B=32 scenario: 24 short prompts (120..128 tokens, a distinct book excerpt and pass key each) and two long ones, all in one call
PACK32_LENS = [120 + (i % 9) for i in range(24)] + [700, 1500]
PACK32_NEWS = [32] * 26
PACK32_LANES = 32

# 16 requests: prompt length (the chat prefix and needle need >= 120), new tokens (16..64) and arrival iteration.
PROMPT_LENS = [130, 8000, 230, 1500, 640, 3100, 120, 5200, 400, 2300, 900, 7000, 180, 1200, 4100, 350]
MAX_NEWS = [48, 64, 16, 40, 32, 24, 64, 36, 20, 56, 32, 28, 44, 18, 33, 52]
ARRIVALS = [0, 0, 0, 1, 2, 4, 6, 6, 9, 12, 15, 15, 20, 25, 30, 40]


@dataclass
class Req:
    index: int
    prompt: list[int]
    max_new: int
    arrival: int
    tokens: list[int] = field(default_factory=list)
    blocks: list[int] = field(default_factory=list)
    slot: int | None = None
    admit_iteration: int | None = None
    finish_iteration: int | None = None
    prefill_s: float = 0.0

    @property
    def position(self) -> int:
        """The position the next decode step feeds: the prompt plus the tokens already fed (all but the last)."""

        return len(self.prompt) + len(self.tokens) - 1

    @property
    def done(self) -> bool:
        return len(self.tokens) >= self.max_new


def build_requests(checkpoint: Path):
    template = Qwen38OfficialChatTemplate(checkpoint)
    tokenizer = template.tokenizer
    config = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    context_text = load_context_text(config, Path(os.environ["QWEN38_CACHE_ROOT"]) / "demo-context")
    prompts = [
        build_prompt_ids(
            tokenizer,
            lambda content: template.render([{"role": "user", "content": content}], enable_thinking=False).text,
            config,
            context_text,
            length,
        )[0]
        for length in PROMPT_LENS
    ]
    assert [len(p) for p in prompts] == PROMPT_LENS
    assert len({len(p) % 4 for p in prompts}) > 1 and max(MAX_NEWS) <= 64 and min(MAX_NEWS) >= 16
    return tokenizer, config, prompts


def refs_path() -> Path:
    override = os.environ.get("QWEN38_VLLM_BATCH_REFS")
    if override:
        return Path(override)
    return Path(os.environ["QWEN38_CACHE_ROOT"]) / "demo-context" / "p2a-vllm-batch-refs.json"


def pack32_refs_path() -> Path:
    override = os.environ.get("QWEN38_PACK32_REFS")
    if override:
        return Path(override)
    return Path(os.environ["QWEN38_CACHE_ROOT"]) / "demo-context" / "p3d4-pack32-refs.json"


def build_pack32_requests(checkpoint: Path):
    from models.demos.blackhole.qwen38_flash_next.demo.long_prompt import build_user_prompt_ids, user_pass_key

    template = Qwen38OfficialChatTemplate(checkpoint)
    config = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    context_text = load_context_text(config, Path(os.environ["QWEN38_CACHE_ROOT"]) / "demo-context")
    render = lambda content: template.render(
        [{"role": "user", "content": content}], enable_thinking=False
    ).text  # noqa: E731
    prompts, keys = [], []
    for user, length in enumerate(PACK32_LENS):
        ids, _info = build_user_prompt_ids(template.tokenizer, render, config, context_text, length, user)
        assert len(ids) == length
        prompts.append(ids)
        keys.append(user_pass_key(config, user))
    return template.tokenizer, keys, prompts


class PackRecorder:
    """Wraps ``decoder.admit_pack``: keeps each packed user's first-token logits (top-2 and margin) by prompt."""

    def __init__(self, decoder) -> None:
        self.by_prompt: dict[tuple, dict] = {}
        self.calls: list[dict] = []
        self._admit = decoder.admit_pack
        decoder.admit_pack = self.admit

    def admit(self, lanes, prompts, rows_per_user, pages=None, users=None):
        logits = self._admit(lanes, prompts, rows_per_user, pages=pages, users=users)
        self.calls.append({"rows_per_user": rows_per_user, "users": len(lanes), "lengths": [len(p) for p in prompts]})
        for u, prompt in enumerate(prompts):
            row = logits[u].clone()
            row[TOKENIZER_SIZE:] = float("-inf")
            top = torch.topk(row, 2)
            self.by_prompt[tuple(prompt)] = {
                "top1": int(top.indices[0]),
                "top2": int(top.indices[1]),
                "margin": float(top.values[0] - top.values[1]),
            }
        return logits


def pack_first_token_gate(first: int, ref_first: int, info: dict) -> tuple[bool, float | None]:
    """A packed row's gate: equal, or the reference is the pack's top-2 with a margin under ``PACK_MARGIN``."""

    if first == ref_first:
        return True, info["margin"]
    return (ref_first == info["top2"] and info["margin"] < PACK_MARGIN), info["margin"]


def prompts_digest(prompts) -> str:
    return hashlib.sha256(json.dumps(prompts).encode()).hexdigest()


def initialize(mesh_device, checkpoint: Path, lanes: int):
    return Qwen38ForCausalLM.initialize_vllm_model(
        SimpleNamespace(_name_or_path=str(checkpoint)), mesh_device, max_batch_size=lanes, max_seq_len=MAX_SEQ_LEN
    )


def plugin_num_blocks(lanes: int, block_size: int) -> int:
    """The worker's block count: ``get_max_tokens_all_users`` plus its per-user output reservation, over the block size."""

    tokens = Qwen38ForCausalLM.get_max_tokens_all_users(
        model_name="", num_devices=4, tt_data_parallel=1, max_model_len=MAX_SEQ_LEN, max_num_seqs=lanes
    )
    tokens += max(block_size, 1) * lanes
    return -(-tokens // block_size)


def plugin_warmups(model, lanes: int, block_size: int, num_blocks: int) -> None:
    """The runner's order: eager prefill and decode, then the traced ones."""

    width = -(-MAX_SEQ_LEN // block_size)
    for trace in (False, True):
        model.warmup_model_prefill(kv_cache=None, can_sample_on_device=False, enable_trace=trace)
        model.warmup_model_decode(
            kv_cache=None, max_batch_size=lanes, num_blocks=width, can_sample_on_device=True, enable_trace=trace
        )


class PluginHost:
    """The plugin runner's side of the contract, over one adapter: block ids, state slots, packed rows, remaps."""

    def __init__(self, model, lanes: int, block_size: int, num_blocks: int, seed: int) -> None:
        self.model, self.lanes, self.block_size, self.num_blocks = model, lanes, block_size, num_blocks
        self.max_prefill_rows = MAX_PREFILL_ROWS
        self.width = -(-MAX_SEQ_LEN // block_size)
        self.rng = random.Random(seed)
        self.free_blocks = list(range(1, num_blocks))  # vLLM never hands out block 0 (its null block)
        self.rng.shuffle(self.free_blocks)
        self.running: list[Req] = []
        self.decode_steps = 0
        self.stats = {
            "remaps": 0,
            "shuffles": 0,
            "prefill_calls": 0,
            "decode_steps": 0,
            "max_running": 0,
            "blocks_peak": 0,
            "notes": 0,
        }
        self.decode_ms: list[float] = []
        self.live_per_step: list[int] = []

    # -- blocks and slots ------------------------------------------------------------------------

    def budget(self, request: Req) -> int:
        return -(-(len(request.prompt) + request.max_new + 1) // self.block_size)

    def alloc(self, count: int) -> list[int]:
        if count > len(self.free_blocks):
            raise RuntimeError(f"vLLM would preempt here: {count} blocks wanted, {len(self.free_blocks)} free")
        blocks = [self.free_blocks.pop() for _ in range(count)]
        in_use = self.num_blocks - 1 - len(self.free_blocks)
        self.stats["blocks_peak"] = max(self.stats["blocks_peak"], in_use)
        return blocks

    def row_of(self, blocks: list[int]) -> list[int]:
        return blocks + [0] * (self.width - len(blocks))

    def prefill_slots(self, batch: list[Req]) -> list[int]:
        """``TTModelRunner._alloc_prefill_state_slots``: the row's own index unless a running request holds it."""

        held = {r.slot for r in self.running}
        slots = []
        for row in range(len(batch)):
            slot = row if row not in held else next(s for s in range(self.lanes) if s not in held)
            held.add(slot)
            slots.append(slot)
        return slots

    # -- the two calls ------------------------------------------------------------------------------

    def prefill(self, batch: list[Req], iteration: int) -> None:
        slots = self.prefill_slots(batch)
        for request, slot in zip(batch, slots):
            request.slot = slot
            request.blocks = self.alloc(-(-len(request.prompt) // self.block_size))
            request.admit_iteration = iteration
        width = max(len(r.prompt) for r in batch)
        tokens = torch.zeros(len(batch), width, dtype=torch.int32)
        for i, request in enumerate(batch):
            tokens[i, : len(request.prompt)] = torch.tensor(request.prompt, dtype=torch.int32)
        started = time.perf_counter()
        out, rope_deltas = self.model.prefill_forward(
            tokens=tokens,
            page_table=torch.tensor([self.row_of(r.blocks) for r in batch], dtype=torch.int32),
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[len(r.prompt) for r in batch],
            start_pos=torch.zeros(len(batch), dtype=torch.int32),
            empty_slots=slots,
            sampling_params=greedy_batch_sampling_params(len(batch)),
        )
        seconds = time.perf_counter() - started
        assert out.shape == (len(batch), 1) and out.dtype == torch.int32 and rope_deltas.shape == (len(batch),)
        self.stats["prefill_calls"] += 1
        for i, request in enumerate(batch):
            request.tokens = [int(out[i, 0])]
            request.prefill_s = seconds / len(batch)
            self.running.append(request)
        self.stats["max_running"] = max(self.stats["max_running"], len(self.running))

    def decode(self, iteration: int, layout_changed: bool) -> list[Req]:
        """One decode step over the running requests; returns the requests that finished."""

        if self.decode_steps % FORCED_SHUFFLE_EVERY == FORCED_SHUFFLE_EVERY - 1 and len(self.running) >= 3:
            self.rng.shuffle(self.running)
            self.stats["shuffles"] += 1
            layout_changed = True
        self.decode_steps += 1
        rows = self.running
        for request in rows:  # the block the step writes into (vLLM allocates it before scheduling the token)
            while len(request.blocks) < request.position // self.block_size + 1:
                request.blocks += self.alloc(1)
        want = [r.slot for r in rows]
        remap = want + [s for s in range(self.lanes) if s not in want]
        moves = {remap[i]: i for i in range(self.lanes) if remap[i] != i}
        tokens = torch.zeros(self.lanes, 1, dtype=torch.int32)
        positions = torch.full((self.lanes,), -1, dtype=torch.int32)
        table = torch.zeros(self.lanes, self.width, dtype=torch.int32)
        for row, request in enumerate(rows):
            tokens[row, 0] = request.tokens[-1]
            positions[row] = request.position
            table[row] = torch.tensor(self.row_of(request.blocks), dtype=torch.int32)
        if moves:
            self.stats["remaps"] += 1
        started = time.perf_counter()
        out = self.model.decode_forward(
            tokens=tokens,
            start_pos=positions,
            page_table=table,
            kv_cache=None,
            enable_trace=True,
            read_from_device=False,
            slot_remap=torch.tensor(remap, dtype=torch.int32) if moves else None,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=layout_changed,
            reset_sampling_state=layout_changed,
            sampling_params=greedy_batch_sampling_params(self.lanes),
        )
        self.decode_ms.append(1000.0 * (time.perf_counter() - started))
        self.live_per_step.append(len(rows))
        assert out.shape == (self.lanes, 1) and out.dtype == torch.int32
        assert [int(t) for t in out[len(rows) :, 0]] == [0] * (self.lanes - len(rows))  # padding rows return token 0
        # the plugin commits the new slots after the accepted decode, then notes the move
        for row, request in enumerate(rows):
            request.slot = row
        if moves:
            self.model.note_state_slots_moved(moves)
            self.stats["notes"] += 1
        finished = []
        for row, request in enumerate(rows):
            request.tokens.append(int(out[row, 0]))
            if request.done:
                finished.append(request)
        self.stats["decode_steps"] += 1
        return finished

    def release(self, request: Req, iteration: int) -> None:
        self.model.release_request(request.slot)
        self.free_blocks.extend(request.blocks)
        self.rng.shuffle(self.free_blocks)
        request.blocks = []
        request.finish_iteration = iteration
        self.running.remove(request)

    # -- the loop ---------------------------------------------------------------------------------

    def run(self, requests: list[Req]) -> None:
        waiting = sorted(requests, key=lambda r: r.arrival)
        iteration, layout_changed = 0, True
        while waiting or self.running:
            reserved = sum(self.budget(r) - len(r.blocks) for r in self.running)
            batch: list[Req] = []
            while (
                waiting
                and waiting[0].arrival <= iteration
                and len(batch) < self.max_prefill_rows
                and len(self.running) + len(batch) < self.lanes
                and len(self.free_blocks) - reserved >= self.budget(waiting[0])
            ):
                batch.append(waiting.pop(0))
                reserved += self.budget(batch[-1])
            if batch:
                self.prefill(batch, iteration)
                for request in batch:
                    if request.done:
                        self.release(request, iteration)
                layout_changed = True
            elif self.running:
                finished = self.decode(iteration, layout_changed)
                layout_changed = False
                for request in finished:
                    self.release(request, iteration)
                    layout_changed = True
            elif waiting:
                iteration = max(iteration, waiting[0].arrival)
                continue
            iteration += 1


def single_slot_reference(model, prompt: list[int], max_new: int) -> list[int]:
    """B=1: greedy prefill token, then ``max_new - 1`` greedy decode steps (the P1d reference)."""

    params = greedy_batch_sampling_params(1)
    out, _ = model.prefill_forward(
        tokens=torch.tensor([prompt], dtype=torch.int32),
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[len(prompt)],
        start_pos=[0],
        empty_slots=[0],
        sampling_params=params,
    )
    tokens = [int(out.reshape(-1)[0])]
    for step in range(1, max_new):
        out = model.decode_forward(
            tokens=torch.tensor([[tokens[-1]]], dtype=torch.int32),
            start_pos=[len(prompt) + step - 1],
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            read_from_device=True,
            sampling_params=params,
        )
        tokens.append(int(out.reshape(-1)[0]))
    model.release_request(0)
    return tokens


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "kind, lanes, block_size",
    [
        pytest.param("ref", 1, 64, id="ref_b1", marks=pytest.mark.timeout(7200)),
        pytest.param("contract", 8, 64, id="contract_b8_bs64", marks=pytest.mark.timeout(10800)),
        pytest.param("contract", 8, 32, id="contract_b8_bs32", marks=pytest.mark.timeout(10800)),
        pytest.param("m1", 8, 64, id="m1_b8", marks=pytest.mark.timeout(5400)),
        pytest.param("m1", 32, 64, id="m1_b32", marks=pytest.mark.timeout(5400)),
        pytest.param("cap", 32, 64, id="cap_b32", marks=pytest.mark.timeout(7200)),
        pytest.param("ref32", 1, 64, id="ref_pack32", marks=pytest.mark.timeout(7200)),
        pytest.param("contract32", PACK32_LANES, 64, id="contract_b32_pack", marks=pytest.mark.timeout(10800)),
    ],
)
def test_vllm_batch_contract(mesh_device, kind, lanes, block_size, tmp_path, record_property):
    if not os.environ.get("MODEL_WEIGHTS_DIR"):
        pytest.fail("set MODEL_WEIGHTS_DIR to the checkpoint directory")
    checkpoint = Path(os.environ["MODEL_WEIGHTS_DIR"])
    if not os.environ.get("QWEN38_CACHE_ROOT"):
        pytest.fail("set QWEN38_CACHE_ROOT to the cache root (the book download goes to <root>/demo-context)")
    mesh_device.enable_program_cache()

    if kind == "m1":
        measure_dram(mesh_device, checkpoint, lanes, block_size, tmp_path, record_property)
        return

    if kind == "cap":
        serve_at_the_cap(mesh_device, checkpoint, lanes, block_size, tmp_path, record_property)
        return

    pack32 = kind in ("ref32", "contract32")
    if pack32:
        tokenizer, pass_keys, prompts = build_pack32_requests(checkpoint)
        news, arrivals, path = PACK32_NEWS, [0] * len(PACK32_LENS), pack32_refs_path()
    else:
        tokenizer, config, prompts = build_requests(checkpoint)
        news, arrivals, path = MAX_NEWS, ARRIVALS, refs_path()
        pass_keys = [config["pass_key"]] * len(prompts)
    digest = prompts_digest(prompts)

    if kind in ("ref", "ref32"):
        model = initialize(mesh_device, checkpoint, 1)
        try:
            model.allocate_kv_cache((1, 1, block_size, 128), None, 12)
            references = []
            for i, (prompt, max_new) in enumerate(zip(prompts, news)):
                started = time.perf_counter()
                references.append(single_slot_reference(model, prompt, max_new))
                logger.info(
                    f"reference {i} (P={len(prompt)}, n={max_new}) in {time.perf_counter() - started:.1f} s: "
                    f"{references[-1][:8]}"
                )
        finally:
            model.release_persistent_capture()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"prompts_sha256": digest, "tokens": references}) + "\n", encoding="utf-8")
        logger.info(f"references written to {path}")
        return

    if not path.is_file():
        pytest.fail(f"run the id {'ref_pack32' if pack32 else 'ref_b1'} first: it writes the B=1 references to {path}")
    fixture = json.loads(path.read_text(encoding="utf-8"))
    assert fixture["prompts_sha256"] == digest, "the reference fixture was made for other prompts: rerun ref_b1"
    references = fixture["tokens"]
    assert [len(r) for r in references] == news

    num_blocks = plugin_num_blocks(lanes, block_size)
    needed = max(-(-(len(p) + n + 1) // block_size) for p, n in zip(prompts, news))
    logger.info(
        f"B={lanes}, block size {block_size}: pool {num_blocks} blocks = {num_blocks * block_size} tokens "
        f"(the largest request needs {needed} blocks)"
    )
    assert num_blocks - 1 >= needed

    started = time.perf_counter()
    model = initialize(mesh_device, checkpoint, lanes)
    assert model.batch_mode and model.chain is None  # the deferred open
    construct_s = time.perf_counter() - started
    try:
        started = time.perf_counter()
        assert model.allocate_kv_cache((num_blocks, 1, block_size, 128), ttnn_bf16(), 12) is None
        allocate_s = time.perf_counter() - started
        started = time.perf_counter()
        plugin_warmups(model, lanes, block_size, num_blocks)
        warmup_s = time.perf_counter() - started
        logger.info(
            f"construct {construct_s:.1f} s, allocate_kv_cache (chain open + lane capture) {allocate_s:.1f} s, "
            f"warmups {warmup_s:.1f} s; DRAM {model.dram}"
        )
        assert model.decoder.positions == [None] * lanes and model.lane_of_slot == list(
            range(lanes)
        )  # warmup left nothing
        recorder = PackRecorder(model.decoder) if model.decoder.pack_enabled else None
        requests = [Req(i, prompts[i], news[i], arrivals[i]) for i in range(len(prompts))]
        host = PluginHost(model, lanes, block_size, num_blocks, seed=20261024 + block_size)
        if pack32:
            host.max_prefill_rows = lanes
        started = time.perf_counter()
        host.run(requests)
        wall_s = time.perf_counter() - started
        stats = list(model.decode_stats)
    finally:
        model.release_persistent_capture()

    results = []
    for request in requests:
        ref, out = references[request.index], request.tokens
        n = request.max_new
        gate = min(EXACT_PREFIX, n)
        diffs = [k for k in range(n) if out[k] != ref[k]]
        decoded = tokenizer.decode(out, skip_special_tokens=True)
        needs_key = n >= PASS_KEY_MIN_TOKENS
        found = pass_keys[request.index] in decoded
        info = None if recorder is None else recorder.by_prompt.get(tuple(request.prompt))
        if info is None:
            gate_ok, first_margin = out[:gate] == ref[:gate], None
        else:
            gate_ok, first_margin = pack_first_token_gate(out[0], ref[0], info)
        results.append(
            {
                "request": request.index,
                "prompt_len": len(request.prompt),
                "max_new_tokens": n,
                "arrival": request.arrival,
                "admit_iteration": request.admit_iteration,
                "finish_iteration": request.finish_iteration,
                "prefill_s": request.prefill_s,
                "exact_match": not diffs,
                "first_diff_index": diffs[0] if diffs else None,
                "gate_prefix_match": gate_ok,
                "packed": info is not None,
                "pack_first_token_margin": first_margin,
                "pack_first_token_equal": None if info is None else out[0] == ref[0],
                "pass_key_required": needs_key,
                "pass_key_found": found,
                "tokens": out,
                "reference_tokens": ref,
                "decoded": decoded,
            }
        )
        logger.info(
            f"req {request.index} P={len(request.prompt)} n={n} iterations {request.admit_iteration}.."
            f"{request.finish_iteration}: exact={'yes' if not diffs else 'no'} first_diff={results[-1]['first_diff_index']} "
            f"packed={info is not None} margin={first_margin} "
            f"gate={'ok' if results[-1]['gate_prefix_match'] else 'FAIL'} "
            f"pass_key={'found' if found else ('MISSING' if needs_key else 'n/a')}"
        )

    full_steps = [ms for ms, live in zip(host.decode_ms, host.live_per_step) if live == max(host.live_per_step)]
    step_s = [s["step_s"] for s in stats]
    overhead_s = [s["overhead_s"] for s in stats]
    summary = {
        "block_size": block_size,
        "num_blocks": num_blocks,
        "requests_exact": sum(r["exact_match"] for r in results),
        "packed_requests": sum(r["packed"] for r in results),
        "packed_exact": sum(r["exact_match"] for r in results if r["packed"]),
        "packed_first_token_misses": [
            {"request": r["request"], "margin": r["pack_first_token_margin"], "gate": r["gate_prefix_match"]}
            for r in results
            if r["packed"] and not r["pack_first_token_equal"]
        ],
        "first_divergence": {r["request"]: r["first_diff_index"] for r in results},
        "pack_calls": None if recorder is None else recorder.calls,
        "prefill_groups_last": [list(map(str, g)) for g in model.last_prefill_groups],
        "requests": len(results),
        "gate_ok": sum(r["gate_prefix_match"] for r in results),
        "pass_key_ok": sum(r["pass_key_found"] for r in results if r["pass_key_required"]),
        "pass_key_required": sum(r["pass_key_required"] for r in results),
        "decode_steps": host.stats["decode_steps"],
        "decode_call_ms_mean": statistics.mean(host.decode_ms),
        "decode_call_ms_peak_live_mean": statistics.mean(full_steps),
        "peak_live_rows": max(host.live_per_step),
        "replay_ms_mean": 1000.0 * statistics.mean(step_s),
        "adapter_overhead_ms_mean": 1000.0 * statistics.mean(overhead_s),
        "adapter_overhead_ms_max": 1000.0 * max(overhead_s),
        "prefill_s_mean": statistics.mean(r["prefill_s"] for r in results),
        "construct_s": construct_s,
        "allocate_s": allocate_s,
        "warmup_s": warmup_s,
        "wall_s": wall_s,
        "host": host.stats,
        "dram": model.dram,
    }
    logger.info(f"SUMMARY {json.dumps(summary)}")
    report = tmp_path / "vllm-batch-contract-report.json"
    report.write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n", encoding="utf-8")
    record_property("vllm_batch_contract_report", str(report))
    logger.info(f"report {report}")

    assert host.stats["remaps"] > 0 and host.stats["notes"] == host.stats["remaps"], "the run must exercise slot_remap"
    assert host.stats["max_running"] >= 4, "the arrivals must overlap"
    if pack32:
        assert host.stats["prefill_calls"] == 1 and host.stats["max_running"] >= 24
        assert recorder is not None and sum(c["users"] for c in recorder.calls) >= 16, recorder and recorder.calls
    for r in results:
        assert r["gate_prefix_match"], f"request {r['request']}: first tokens differ from the B=1 reference"
        if r["pass_key_required"]:
            assert r["pass_key_found"], f"request {r['request']}: pass key not in {r['decoded'][:300]!r}"


def ttnn_bf16():
    import ttnn

    return ttnn.bfloat16


def measure_dram(mesh_device, checkpoint: Path, lanes: int, block_size: int, tmp_path, record_property) -> None:
    """M1: free DRAM per device after the batch build at context 32768 with a small pool (the pool is the budget's unknown)."""

    num_blocks = 1 + 2 * (-(-MAX_SEQ_LEN // block_size))  # two full contexts and the scratch block
    model = initialize(mesh_device, checkpoint, lanes)
    try:
        model.allocate_kv_cache((num_blocks, 1, block_size, 128), ttnn_bf16(), 12)
        plugin_warmups(model, lanes, block_size, num_blocks)
        dram = {
            "lanes": lanes,
            "context": model.allocated_context,
            "block_size": block_size,
            "num_blocks": num_blocks,
            "pack_enabled": model.decoder.pack_enabled,
            "pool_bytes_per_device": num_blocks * block_size * 12 * 512 * 2,
            "chain_only": model.dram["chain_only"],
            "after_capture": model.dram["after_capture"],
        }
        dram["lane_states_bytes"] = (
            dram["chain_only"]["free_bytes"] - dram["after_capture"]["free_bytes"] - dram["pool_bytes_per_device"]
        )
    finally:
        model.release_persistent_capture()
    logger.info(f"M1 {json.dumps(dram)}")
    path = tmp_path / f"m1-b{lanes}.json"
    path.write_text(json.dumps(dram, indent=2) + "\n", encoding="utf-8")
    record_property("m1_report", str(path))


def serve_at_the_cap(mesh_device, checkpoint: Path, lanes: int, block_size: int, tmp_path, record_property) -> None:
    """The pool at the DRAM budget's cap, warmups, then a long prompt, a mid one and a short one through the host."""

    template = Qwen38OfficialChatTemplate(checkpoint)
    config = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    context_text = load_context_text(config, Path(os.environ["QWEN38_CACHE_ROOT"]) / "demo-context")
    lengths, news = [30000, 8000, 130], [8, 8, 8]
    prompts = [
        build_prompt_ids(
            template.tokenizer,
            lambda content: template.render([{"role": "user", "content": content}], enable_thinking=False).text,
            config,
            context_text,
            length,
        )[0]
        for length in lengths
    ]
    num_blocks = plugin_num_blocks(lanes, block_size)
    logger.info(f"B={lanes}: pool {num_blocks} blocks = {num_blocks * block_size} tokens at the budget's cap")
    model = initialize(mesh_device, checkpoint, lanes)
    try:
        model.allocate_kv_cache((num_blocks, 1, block_size, 128), ttnn_bf16(), 12)
        plugin_warmups(model, lanes, block_size, num_blocks)
        requests = [Req(i, prompts[i], news[i], arrival) for i, arrival in enumerate((0, 0, 1))]
        host = PluginHost(model, lanes, block_size, num_blocks, seed=7)
        host.run(requests)
        dram = model.dram
        free_after = model._dram_free(model.chain.mesh)
    finally:
        model.release_persistent_capture()
    logger.info(f"CAP {json.dumps({'num_blocks': num_blocks, 'dram': dram, 'free_after_serving': free_after})}")
    for request in requests:
        logger.info(f"req {request.index} P={len(request.prompt)}: {request.tokens}")
        assert len(request.tokens) == request.max_new
    assert host.stats["decode_steps"] > 0
