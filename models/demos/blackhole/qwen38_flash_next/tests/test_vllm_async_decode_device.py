# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test: the adapter's async decode (``QWEN38_ASYNC_DECODE=1``) driven as vllm-tt-plugin's async scheduler drives it.

Per step the plugin calls ``decode_forward(read_from_device=False, reload_inputs=False)`` (after the first step of a request
group: its ``tokens`` and ``start_pos`` are stale, here the last processed token and position, up to 1-2 behind), then
``read_decode_output(async_read=True)``; the NEXT ``decode_forward`` is issued BEFORE the previous step's output is processed
(``ttnn.event_synchronize`` on the returned events, then ``process_decode_output_host(is_tokens=True)``).  Greedy, one model
build (32 lanes: the decode buckets), ``QWEN38_PREFILL_PACK=0``.  Claims: every request's tokens are bitwise equal to the
blocking path's (``reload_inputs=True`` every step, ``model.async_decode`` False: today's code) for the same prompt:

(i)   one request on the hybrid chain route (``hybrid_single`` on), the steady chain step (HEAD, wait, PLE, TAIL, async read);
(ii)  three requests on lanes (bucket 4: one idle lane inside the bucket, so the token write stays), and four (bucket 4, all
      lanes live: no token write; the resolve epilogue's token row feeds the next replay);
(iii) request A finishes while B continues (A's next step is already in flight and its output discarded), A is released,
      B moves down a state slot (``slot_remap`` with ``reload_inputs=True``), a new request C is prefilled while a step of B
      is in flight (the prefill waits it), then B and C decode together.

It prints the mean ms per step (async vs blocking) for (i) and (ii).  Not run by CI: it needs the mesh.

Run:  QWEN38_ASYNC_DECODE=1 pytest models/demos/blackhole/qwen38_flash_next/tests/test_vllm_async_decode_device.py -v -s
Env:  MESH_DEVICE (the TP=8 mesh), HF_MODEL, TT_CACHE_PATH, as test_vllm_batch_contract_device.py.  The knob is read at
      import: set it before pytest starts.
"""

import os
import time
from collections import deque

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_hybrid_import_device import build_prompts
from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_batch_contract_device import (
    MAX_SEQ_LEN,
    initialize,
    plugin_num_blocks,
    plugin_warmups,
    ttnn_bf16,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import Qwen38AsyncStep, greedy_batch_sampling_params
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

LANES = 32  # decode buckets (1, 2, 4, 8, 16, 32) need 32 lanes
BLOCK = 64
STEPS = 48  # decode steps per request group in (i) and (ii)
LENGTHS = {"a": 128, "b": 200, "c": 300, "d": 160}
A_TOKENS, B_TOKENS, C_TOKENS = 12, 40, 24  # (iii): tokens each request produces (the prefill token included)


class Client:
    """The plugin's calls for hand-driven requests.  Requests are keyed by name; their blocks grow with their true
    position, whatever the (stale) position the call sends."""

    def __init__(self, model) -> None:
        self.model = model
        self.width = -(-MAX_SEQ_LEN // BLOCK)
        self.next_block = 1
        self.blocks: dict[str, list[int]] = {}

    def _grow(self, key: str, position: int) -> None:
        held = self.blocks.setdefault(key, [])
        while len(held) < position // BLOCK + 1:
            held.append(self.next_block)
            self.next_block += 1

    def _row(self, key: str) -> list[int]:
        return self.blocks[key] + [0] * (self.width - len(self.blocks[key]))

    def prefill(self, key: str, prompt: list[int], slot: int) -> int:
        self.blocks[key] = []
        self._grow(key, len(prompt) - 1)
        out, _ = self.model.prefill_forward(
            tokens=torch.tensor([prompt], dtype=torch.int32),
            page_table=torch.tensor([self._row(key)], dtype=torch.int32),
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[len(prompt)],
            start_pos=torch.zeros(1, dtype=torch.int32),
            empty_slots=[slot],
            sampling_params=greedy_batch_sampling_params(1),
        )
        assert out.shape == (1, 1) and out.dtype == torch.int32
        return int(out[0, 0])

    def call(self, rows: list[tuple[str, int, int, int]], *, reload_inputs: bool, remap: list[int] | None = None):
        """``rows``: per live row ``(key, sent token, sent position, true position)``; row i = state slot i unless
        ``remap`` says otherwise.  The raw ``decode_forward`` result."""

        tokens = torch.zeros(LANES, 1, dtype=torch.int32)
        positions = torch.full((LANES,), -1, dtype=torch.int32)
        table = torch.zeros(LANES, self.width, dtype=torch.int32)
        for row, (key, token, sent, true) in enumerate(rows):
            self._grow(key, true)
            tokens[row, 0], positions[row] = token, sent
            table[row] = torch.tensor(self._row(key), dtype=torch.int32)
        return self.model.decode_forward(
            tokens=tokens,
            start_pos=positions,
            page_table=table,
            kv_cache=None,
            enable_trace=True,
            read_from_device=False,
            slot_remap=remap,
            reload_inputs=reload_inputs,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
            sampling_params=greedy_batch_sampling_params(LANES),
        )

    def release(self, key: str, slot: int) -> None:
        self.model.release_request(slot)
        self.blocks.pop(key, None)


class Pipeline:
    """The plugin's async loop: every step's output is read with ``read_decode_output(async_read=True)`` and processed later."""

    def __init__(self, client: Client) -> None:
        self.client = client
        self.queue: deque = deque()

    def issue(self, rows, *, reload_inputs: bool, remap=None) -> None:
        out = self.client.call(rows, reload_inputs=reload_inputs, remap=remap)
        if not reload_inputs:
            assert isinstance(out, Qwen38AsyncStep), f"a steady step must return a handle, got {type(out).__name__}"
        host, events = self.client.model.read_decode_output(out, async_read=True)
        self.queue.append((host, events, len(rows)))

    def drain_one(self) -> list[int]:
        import ttnn

        host, events, count = self.queue.popleft()
        for event in events:
            ttnn.event_synchronize(event)
        out = self.client.model.process_decode_output_host(host, is_tokens=True)
        assert tuple(out.shape) == (LANES, 1) and out.dtype == torch.int32
        return [int(out[row, 0]) for row in range(count)]

    def drain(self) -> list[list[int]]:
        return [self.drain_one() for _ in range(len(self.queue))]


def run_blocking(
    client: Client, keys: list[str], prompts: list[list[int]], steps: int
) -> tuple[list[list[int]], float]:
    """Today's path: ``reload_inputs=True`` every step, the adapter's async decode off.  Returns the tokens per request
    (the prefill token and ``steps`` decode tokens) and the mean ms per step (after the first)."""

    model = client.model
    model.async_decode = False
    tokens = [[client.prefill(key, prompt, slot)] for slot, (key, prompt) in enumerate(zip(keys, prompts))]
    started = None
    for i in range(steps):
        if i == 1:
            started = time.perf_counter()
        rows = [
            (key, tokens[s][-1], len(prompt) + i, len(prompt) + i) for s, (key, prompt) in enumerate(zip(keys, prompts))
        ]
        out = client.call(rows, reload_inputs=True)
        for s in range(len(keys)):
            tokens[s].append(int(out[s, 0]))
    elapsed = time.perf_counter() - started
    for slot, key in enumerate(keys):
        client.release(key, slot)
    return tokens, 1000.0 * elapsed / (steps - 1)


def run_async(client: Client, keys: list[str], prompts: list[list[int]], steps: int) -> tuple[list[list[int]], float]:
    """The plugin's async loop over the same requests: the first step reloads (blocking), the rest are steady and
    overlapped: step i is issued before step i-1's output is processed."""

    model = client.model
    model.async_decode = True
    tokens = [[client.prefill(key, prompt, slot)] for slot, (key, prompt) in enumerate(zip(keys, prompts))]
    pipe = Pipeline(client)
    started = None
    for i in range(steps):
        true = [len(prompt) + i for prompt in prompts]
        if i == 0:
            rows = [(key, tokens[s][-1], true[s], true[s]) for s, key in enumerate(keys)]
            pipe.issue(rows, reload_inputs=True)
            for s, token in enumerate(pipe.drain_one()):
                tokens[s].append(token)
            started = time.perf_counter()
            continue
        # stale: the last token and position the host applied (the steps in flight are not in them)
        rows = [(key, tokens[s][-1], len(prompts[s]) + len(tokens[s]) - 1, true[s]) for s, key in enumerate(keys)]
        pipe.issue(rows, reload_inputs=False)
        if len(pipe.queue) > 1:  # step i is in flight: process step i-1 now
            for s, token in enumerate(pipe.drain_one()):
                tokens[s].append(token)
    for outputs in pipe.drain():
        for s, token in enumerate(outputs):
            tokens[s].append(token)
    elapsed = time.perf_counter() - started
    for slot, key in enumerate(keys):
        client.release(key, slot)
    assert model._inflight == [] or len(model._inflight) == 0  # the release waited the last step
    return tokens, 1000.0 * elapsed / (steps - 1)


def first_divergence(got: list[int], want: list[int]) -> int | None:
    return next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), None)


def run_finish_and_new(client: Client, prompts: dict[str, list[int]]) -> dict[str, list[int]]:
    """(iii): A (slot 0) and B (slot 1) decode steady; A finishes after ``A_TOKENS`` tokens while its next step is in
    flight; A is released, B moves down to state slot 0, a step of B is in flight when C is prefilled, then B and C decode.
    """

    model = client.model
    model.async_decode = True
    model.hybrid_single = False
    a, b, c = prompts["a"], prompts["b"], prompts["c"]
    tokens = {"a": [client.prefill("a", a, 0)], "b": [client.prefill("b", b, 1)]}
    pipe = Pipeline(client)
    plen = {"a": len(a), "b": len(b), "c": len(c)}

    def true(key: str, ahead: int = 0) -> int:
        return plen[key] + len(tokens[key]) - 1 + ahead

    pipe.issue(
        [("a", tokens["a"][-1], true("a"), true("a")), ("b", tokens["b"][-1], true("b"), true("b"))], reload_inputs=True
    )
    for key, token in zip("ab", pipe.drain_one()):
        tokens[key].append(token)
    # steady: both rows, until A's last token is processed; its next step is then already in flight
    issued = 0  # steps issued and not yet applied
    while len(tokens["a"]) < A_TOKENS:
        rows = [(k, tokens[k][-1], true(k), true(k, issued)) for k in "ab"]
        pipe.issue(rows, reload_inputs=False)
        issued += 1
        if len(pipe.queue) > 1:
            for key, token in zip("ab", pipe.drain_one()):
                tokens[key].append(token)
            issued -= 1
    # the step in flight (A's discarded one, B's real one) is processed: B's token counts, A's does not
    for outputs in pipe.drain():
        tokens["b"].append(outputs[1])
    # A finished: its state slot is released (the in-flight work is over); B's row moves to row 0 under a slot_remap
    client.release("a", 0)
    remap = [1, 0] + list(range(2, LANES))
    pipe.issue(
        [("b", tokens["b"][-1], true("b"), true("b"))], reload_inputs=True, remap=remap
    )  # a layout change: reload
    tokens["b"].append(pipe.drain_one()[0])
    model.note_state_slots_moved({1: 0, 0: 1})
    # B steady alone in slot 0, and C's prefill arrives while a step of B is in flight
    pipe.issue([("b", tokens["b"][-1], true("b"), true("b"))], reload_inputs=False)
    tokens["c"] = [client.prefill("c", c, 1)]  # waits the step in flight, then takes the free slot 1
    tokens["b"].append(pipe.drain_one()[0])
    # B and C together: C is new, so the first step reloads (the plugin's layout change), then steady
    rows = [("b", tokens["b"][-1], true("b"), true("b")), ("c", tokens["c"][-1], true("c"), true("c"))]
    pipe.issue(rows, reload_inputs=True)
    for key, token in zip("bc", pipe.drain_one()):
        tokens[key].append(token)
    issued = 0
    while len(tokens["c"]) < C_TOKENS or len(tokens["b"]) < B_TOKENS:
        rows = [(k, tokens[k][-1], true(k), true(k, issued)) for k in "bc"]
        pipe.issue(rows, reload_inputs=False)
        issued += 1
        if len(pipe.queue) > 1:
            for key, token in zip("bc", pipe.drain_one()):
                tokens[key].append(token)
            issued -= 1
    for outputs in pipe.drain():
        for key, token in zip("bc", outputs):
            tokens[key].append(token)
    client.release("b", 0)
    client.release("c", 1)
    return tokens


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(7200)
def test_async_decode(mesh_device, monkeypatch) -> None:
    if os.environ.get("QWEN38_ASYNC_DECODE") != "1":
        pytest.fail("set QWEN38_ASYNC_DECODE=1 before pytest starts (the knob and the capability are read at import)")
    try:
        checkpoint = checkpoint_directory()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    if not os.environ.get("TT_CACHE_PATH"):
        pytest.fail("set TT_CACHE_PATH to the cache root (the book download goes to <root>/demo-context)")
    monkeypatch.setenv("QWEN38_PREFILL_PACK", "0")  # the lone-row prefill is the exact single path in every run
    mesh_device.enable_program_cache()
    names = list(LENGTHS)
    prompts = dict(zip(names, build_prompts(checkpoint, [LENGTHS[name] for name in names])))

    model = initialize(mesh_device, checkpoint, LANES)
    try:
        assert model.model_capabilities["supports_async_decode"] is True and model.async_decode
        num_blocks = plugin_num_blocks(LANES, BLOCK)
        model.allocate_kv_cache((num_blocks, 1, BLOCK, 128), ttnn_bf16(), 12)
        plugin_warmups(model, LANES, BLOCK, num_blocks)
        client = Client(model)
        failures: list[str] = []

        def compare(label: str, got: list[list[int]], want: list[list[int]]) -> None:
            for index, (g, w) in enumerate(zip(got, want)):
                first = first_divergence(g, w)
                if first is not None or len(g) != len(w):
                    failures.append(
                        f"{label} request {index}: async tokens diverge from blocking at {first} (len {len(g)}/{len(w)})"
                    )

        # (i) one request on the hybrid chain route
        model.hybrid_single = True
        ref_a, blocking_ms = run_blocking(client, ["a"], [prompts["a"]], STEPS)
        assert model.hybrid_slot is None  # released
        got_a, async_ms = run_async(client, ["a"], [prompts["a"]], STEPS)
        compare("(i) hybrid chain", got_a, ref_a)
        print(f"(i) hybrid chain: blocking {blocking_ms:.2f} ms/step, async {async_ms:.2f} ms/step")

        # (ii) three requests on lanes (bucket 4, one idle lane in the bucket), then four (bucket 4, all live)
        model.hybrid_single = False
        for count in (3, 4):
            keys = names[:count]
            ref, blocking_ms = run_blocking(client, keys, [prompts[k] for k in keys], STEPS)
            got, async_ms = run_async(client, keys, [prompts[k] for k in keys], STEPS)
            compare(f"(ii) {count} lanes", got, ref)
            print(f"(ii) {count} lanes: blocking {blocking_ms:.2f} ms/step, async {async_ms:.2f} ms/step")

        # (iii) a request finishing with its next step in flight, a release, a slot move, a prefill with a step in flight
        lone = {}
        for key in ("a", "b", "c"):
            lone[key], _ms = run_blocking(client, [key], [prompts[key]], max(B_TOKENS, C_TOKENS, A_TOKENS) + 8)
        got = run_finish_and_new(client, prompts)
        for key in ("a", "b", "c"):
            want = lone[key][0]
            tokens = got[key]
            first = first_divergence(tokens, want)
            if first is not None or len(tokens) > len(want):
                failures.append(
                    f"(iii) request {key}: tokens diverge from blocking at {first} (len {len(tokens)}/{len(want)})"
                )
            logger.info(f"(iii) request {key}: {len(tokens)} tokens compared")
        assert not failures, "\n".join(failures)
    finally:
        model.release_persistent_capture()
