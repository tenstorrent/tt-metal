# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test: hybrid single mode (``QWEN38_HYBRID_SINGLE``) of the vLLM adapter's batch mode, driven as the plugin does.

A lone plain greedy request decodes on the B=1 chain; a second request's prefill demotes it into a lane
(``import_lane`` of the chain's state after k decode steps).  Claims, greedy, one model build for all cases:

(a) request A alone on the chain (never demoted) for k_max + N steps gives T_ref.  For each k: a fresh run, A on the chain
    for k steps, request B prefilled (demotes A), then A and B decode together for N steps.  A's tokens equal T_ref bitwise,
    B's tokens equal B's own lone chain reference.  The cases cover (P + k) mod 4 = 0..3 and (P + k) mod 64 = 0 (the next
    position needs a new block, block size 64).
(b) control, knob off: A through the old path (prefill, import, lane decode) equals T_ref (lane = chain).
(d) the real server's prefill passes ``sampling_params=None`` (device sampling is decode only) and returns the logits: per
    ISL, A alone on the chain from a logits prefill (the host argmax is the first token) then greedy decode equals T_ref.
(c) the lone run's chain decode time: ``hybrid chain decode X ms/step`` (mean of the last 40 steps; not asserted).

Run:  pytest models/demos/blackhole/qwen38_flash_next/tests/test_hybrid_import_device.py -v -s
Env:  MESH_DEVICE=P300x2 (or the TP=8 mesh), HF_MODEL, TT_CACHE_PATH, as test_vllm_batch_contract_device.py.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.chat import Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.long_prompt import build_prompt_ids, load_context_text
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    PROMPT_FILE,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_batch_contract_device import (
    MAX_SEQ_LEN,
    initialize,
    plugin_num_blocks,
    plugin_warmups,
    ttnn_bf16,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import greedy_batch_sampling_params
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

LANES = 8
BLOCK = 64
NEW_TOKENS = 48  # decode steps after the demote
B_LENGTH = 200  # the second request's prompt
TIMED_STEPS = 40
CASES = {  # ISL -> the chain steps k before the second request arrives
    128: [0, 1, 2, 3, 61, 64],  # P + k mod 4 = 0, 1, 2, 3, 1, 0; 128 + 64 = 192 = 3 * 64
    1000: [1, 3, 24, 61],  # 1000 + 24 = 1024 = 16 * 64
}
LONE_STEPS = max(max(ks) for ks in CASES.values()) + NEW_TOKENS


def build_prompts(checkpoint: Path, lengths: list[int]) -> list[list[int]]:
    template = Qwen38OfficialChatTemplate(checkpoint)
    config = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    context_text = load_context_text(config, Path(os.environ["TT_CACHE_PATH"]) / "demo-context")
    return [
        build_prompt_ids(
            template.tokenizer,
            lambda content: template.render([{"role": "user", "content": content}], enable_thinking=False).text,
            config,
            context_text,
            length,
        )[0]
        for length in lengths
    ]


class Client:
    """The plugin's calls for hand-driven requests: ids pack as rows in slot order, blocks grow with the position."""

    def __init__(self, model) -> None:
        self.model = model
        self.width = -(-MAX_SEQ_LEN // BLOCK)
        self.next_block = 1
        self.blocks: dict[int, list[int]] = {}  # slot -> blocks
        self.seconds: list[float] = []  # decode_forward wall seconds, in call order

    def _grow(self, slot: int, position: int) -> None:
        held = self.blocks.setdefault(slot, [])
        while len(held) < position // BLOCK + 1:
            held.append(self.next_block)
            self.next_block += 1

    def _row(self, slot: int) -> list[int]:
        return self.blocks[slot] + [0] * (self.width - len(self.blocks[slot]))

    def prefill(self, prompt: list[int], slot: int, *, logits: bool = False) -> int:
        """``logits``: ``sampling_params=None`` as the server passes it; the host argmax of the returned logits is the token."""

        self.blocks[slot] = []
        self._grow(slot, len(prompt) - 1)
        out, _ = self.model.prefill_forward(
            tokens=torch.tensor([prompt], dtype=torch.int32),
            page_table=torch.tensor([self._row(slot)], dtype=torch.int32),
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[len(prompt)],
            start_pos=torch.zeros(1, dtype=torch.int32),
            empty_slots=[slot],
            sampling_params=None if logits else greedy_batch_sampling_params(1),
        )
        if logits:
            assert out.dim() == 3 and out.shape[:2] == (1, 1)
            return int(torch.argmax(out[0, 0]))
        assert out.shape == (1, 1) and out.dtype == torch.int32
        return int(out[0, 0])

    def decode(self, rows: list[tuple[int, int, int]]) -> list[int]:
        """``rows``: ``(slot, token, position)`` per live row, in slot order (row i = slot i, no remap); the next tokens."""

        tokens = torch.zeros(LANES, 1, dtype=torch.int32)
        positions = torch.full((LANES,), -1, dtype=torch.int32)
        table = torch.zeros(LANES, self.width, dtype=torch.int32)
        for row, (slot, token, position) in enumerate(rows):
            assert slot == row
            self._grow(slot, position)
            tokens[row, 0], positions[row] = token, position
            table[row] = torch.tensor(self._row(slot), dtype=torch.int32)
        started = time.perf_counter()
        out = self.model.decode_forward(
            tokens=tokens,
            start_pos=positions,
            page_table=table,
            kv_cache=None,
            enable_trace=True,
            read_from_device=False,
            slot_remap=None,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
            sampling_params=greedy_batch_sampling_params(LANES),
        )
        self.seconds.append(time.perf_counter() - started)
        assert out.shape == (LANES, 1) and out.dtype == torch.int32
        return [int(out[row, 0]) for row in range(len(rows))]

    def release(self, slot: int) -> None:
        self.model.release_request(slot)
        self.blocks.pop(slot, None)


def run_alone(client: Client, prompt: list[int], steps: int, *, hybrid: bool, logits: bool = False) -> list[int]:
    """One request in slot 0: the prefill token and ``steps`` decode steps.  Hybrid: on the chain throughout; else the
    old path (imported into a lane at the prefill).  ``logits``: the prefill without sampling params (host argmax)."""

    model = client.model
    model.hybrid_single = hybrid
    tokens = [client.prefill(prompt, 0, logits=logits)]
    assert (model.hybrid_slot == 0) == hybrid and (model.decoder.positions[0] is None) == hybrid
    for i in range(steps):
        tokens.append(client.decode([(0, tokens[-1], len(prompt) + i)])[0])
        assert (model.decode_stats[-1]["bucket"] == 0) == hybrid  # bucket 0: no lane replay
    client.release(0)
    assert model.hybrid_slot is None and model.decoder.positions == [None] * LANES
    return tokens


def run_demoted(client: Client, a: list[int], b: list[int], k: int) -> tuple[list[int], list[int]]:
    """A on the chain for ``k`` steps, B's prefill demotes it, then A and B decode together for ``NEW_TOKENS`` steps."""

    model = client.model
    model.hybrid_single = True
    a_tokens = [client.prefill(a, 0)]
    for i in range(k):
        a_tokens.append(client.decode([(0, a_tokens[-1], len(a) + i)])[0])
    assert model.hybrid_slot == 0 and len(model.slot.committed) == len(a) + k
    b_tokens = [client.prefill(b, 1)]  # demotes A (slot 0) into a lane, then B takes the lane path
    assert model.hybrid_slot is None and model.decoder.positions[model.lane_of_slot[0]] == len(a) + k
    logger.info(f"k={k}: hybrid import {model.decoder.last_admit}")
    for i in range(NEW_TOKENS):
        next_a, next_b = client.decode([(0, a_tokens[-1], len(a) + k + i), (1, b_tokens[-1], len(b) + i)])
        a_tokens.append(next_a)
        b_tokens.append(next_b)
        assert model.decode_stats[-1]["bucket"] > 0 and model.decode_stats[-1]["rows"] == 2
    client.release(0)
    client.release(1)
    return a_tokens, b_tokens


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(7200)
def test_hybrid_import(mesh_device, monkeypatch) -> None:
    monkeypatch.setenv(
        "QWEN38_PREFILL_PACK", "0"
    )  # the pack diverges from the B=1 chain after token 1 (docs/PREFILL.md)
    try:
        checkpoint = checkpoint_directory()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    if not os.environ.get("TT_CACHE_PATH"):
        pytest.fail("set TT_CACHE_PATH to the cache root (the book download goes to <root>/demo-context)")
    mesh_device.enable_program_cache()
    lengths = [*CASES, B_LENGTH]
    prompts = dict(zip(lengths, build_prompts(checkpoint, lengths)))
    prompt_b = prompts[B_LENGTH]

    model = initialize(mesh_device, checkpoint, LANES)
    try:
        num_blocks = plugin_num_blocks(LANES, BLOCK)
        model.allocate_kv_cache((num_blocks, 1, BLOCK, 128), ttnn_bf16(), 12)
        plugin_warmups(model, LANES, BLOCK, num_blocks)
        client = Client(model)

        reference_b = run_alone(client, prompt_b, NEW_TOKENS, hybrid=True)
        failures = []
        for isl, ks in CASES.items():
            prompt_a = prompts[isl]
            client.seconds.clear()
            reference = run_alone(client, prompt_a, LONE_STEPS, hybrid=True)
            timed = client.seconds[-TIMED_STEPS:]
            print(f"hybrid chain decode {1000.0 * sum(timed) / len(timed):.2f} ms/step (ISL {isl})")

            control = run_alone(client, prompt_a, LONE_STEPS, hybrid=False)  # (b) lane = chain
            if control != reference:
                failures.append(
                    f"ISL {isl}: knob-off lane run diverges at {next(i for i, (x, y) in enumerate(zip(control, reference)) if x != y)}"
                )

            from_logits = run_alone(client, prompt_a, LONE_STEPS, hybrid=True, logits=True)  # (d)
            if from_logits != reference:
                failures.append(
                    f"ISL {isl}: logits-prefill hybrid run diverges at "
                    f"{next(i for i, (x, y) in enumerate(zip(from_logits, reference)) if x != y)}"
                )

            for k in ks:
                a_tokens, b_tokens = run_demoted(client, prompt_a, prompt_b, k)
                logger.info(f"ISL {isl} k={k} (P+k)%4={(isl + k) % 4} (P+k)%64={(isl + k) % 64}")
                if a_tokens != reference[: k + 1 + NEW_TOKENS]:
                    first = next(i for i, (x, y) in enumerate(zip(a_tokens, reference)) if x != y)
                    failures.append(f"ISL {isl} k={k}: A diverges from the chain reference at token {first}")
                if b_tokens != reference_b:
                    first = next(i for i, (x, y) in enumerate(zip(b_tokens, reference_b)) if x != y)
                    failures.append(f"ISL {isl} k={k}: B diverges from its lone reference at token {first}")
        assert not failures, "\n".join(failures)
    finally:
        model.release_persistent_capture()
