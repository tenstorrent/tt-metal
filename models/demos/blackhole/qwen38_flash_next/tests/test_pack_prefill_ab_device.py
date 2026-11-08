# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pack A/B: the vLLM adapter's prefill of 32 prompts in ONE ``prefill_forward`` call, 32 lanes, block size 128.

Run once per ``QWEN38_PREFILL_PACK`` value ("1", "0"), each in its own process (two adapters do not share a process's DRAM).
Writes JSON to ``PACK_AB_OUT`` (else ``tmp_path / "pack_ab.json"``).  ``PACK_AB_DECODE=1`` adds 16 greedy decode steps.

Copied from the existing tests (paths relative to this directory):
  - imports, ``@run_for_blackhole``, mesh_device / device_params indirect params: test_pack_landing_device.py:36-60 and
    :464-465; test_vllm_batch_contract_device.py:440-444 (``MESH_SHAPE``, ``DEVICE_PARAMS``, ``device_params`` fixture).
  - ``initialize``, ``plugin_num_blocks``, ``plugin_warmups`` are imported the way test_pack_landing_device.py:48-52 does.
  - setup order: test_vllm_batch_contract_device.py:472 (enable_program_cache), :517-531 (initialize, allocate_kv_cache,
    plugin_warmups), :534 (warmup leaves lanes empty), :537 (PackRecorder when pack_enabled).
  - ``PackRecorder``: test_vllm_batch_contract_device.py:166-187.
  - prompt builder: ``build_pack32_requests`` pattern, test_vllm_batch_contract_device.py:148-163.
  - prefill call: ``PluginHost.prefill``, test_vllm_batch_contract_device.py:283-310; block ids ``alloc``/``row_of``, :242-257.
  - release: ``PluginHost.release``, test_vllm_batch_contract_device.py:369-375 (``model.release_request(slot)``).
  - decode call: ``PluginHost.decode``, test_vllm_batch_contract_device.py:312-355.
  - device sync: ttnn.synchronize_device, test_pack_landing_device.py:255.

Run:  QWEN38_PREFILL_PACK=1 pytest models/demos/blackhole/qwen38_flash_next/tests/test_pack_prefill_ab_device.py -v -s
Env:  MESH_DEVICE=P150x8, HF_MODEL, TT_CACHE_PATH, TT_METAL_TRACE_ALLOC_TRACKING=1, TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0.
"""

import json
import os
import random
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.long_prompt import load_context_text
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
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import greedy_batch_sampling_params
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

LANES = 32
BLOCK = 128
LENGTHS = [120 + (i * 277) % 880 for i in range(32)]
REPS = 3
DECODE_STEPS = 16
SEED = 20261107


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


def build_prompts(checkpoint: Path):
    from models.demos.blackhole.qwen38_flash_next.demo.long_prompt import build_user_prompt_ids

    template = Qwen38OfficialChatTemplate(checkpoint)
    config = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    context_text = load_context_text(config, Path(os.environ["TT_CACHE_PATH"]) / "demo-context")
    render = lambda content: template.render(  # noqa: E731
        [{"role": "user", "content": content}], enable_thinking=False
    ).text
    prompts = []
    for user, length in enumerate(LENGTHS):
        ids, _info = build_user_prompt_ids(template.tokenizer, render, config, context_text, length, user)
        assert len(ids) == length
        prompts.append(ids)
    return prompts


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(3600)
def test_pack_prefill_ab(mesh_device, tmp_path):
    try:
        checkpoint = checkpoint_directory()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    if not os.environ.get("TT_CACHE_PATH"):
        pytest.fail("set TT_CACHE_PATH to the cache root (the book download goes to <root>/demo-context)")
    mesh_device.enable_program_cache()
    prompts = build_prompts(checkpoint)
    assert [len(p) for p in prompts] == LENGTHS

    num_blocks = plugin_num_blocks(LANES, BLOCK)
    width = -(-MAX_SEQ_LEN // BLOCK)
    need = [-(-len(p) // BLOCK) for p in prompts]
    assert num_blocks - 1 >= sum(need) + 2 * LANES  # prefill blocks plus room for the decode steps
    rng = random.Random(SEED)
    ids = list(range(1, num_blocks))  # vLLM never hands out block 0
    rng.shuffle(ids)
    blocks, at = [], 0
    for count in need:
        blocks.append(ids[at : at + count])
        at += count
    spare = ids[at:]  # one spare block per row for decode (a row may cross a block edge)
    assert len({b for row in blocks for b in row}) == sum(need)
    pages = torch.zeros(LANES, width, dtype=torch.int32)
    for row, row_blocks in enumerate(blocks):
        pages[row, : len(row_blocks)] = torch.tensor(row_blocks, dtype=torch.int32)
    tokens = torch.zeros(LANES, max(LENGTHS), dtype=torch.int32)
    for row, prompt in enumerate(prompts):
        tokens[row, : len(prompt)] = torch.tensor(prompt, dtype=torch.int32)

    model = initialize(mesh_device, checkpoint, LANES)
    assert model.batch_mode and model.chain is None  # the deferred open
    try:
        assert model.allocate_kv_cache((num_blocks, 1, BLOCK, 128), ttnn.bfloat16, 12) is None
        plugin_warmups(model, LANES, BLOCK, num_blocks)
        assert model.decoder.positions == [None] * LANES and model.lane_of_slot == list(range(LANES))
        knob = os.environ.get("QWEN38_PREFILL_PACK")
        if knob is not None:
            assert model.decoder.pack_enabled == (knob == "1")
        pack_enabled = bool(model.decoder.pack_enabled)
        recorder = PackRecorder(model.decoder) if pack_enabled else None

        def prefill():
            out, rope_deltas = model.prefill_forward(
                tokens=tokens,
                page_table=pages,
                kv_cache=None,
                enable_trace=True,
                prompt_lens=list(LENGTHS),
                start_pos=torch.zeros(LANES, dtype=torch.int32),
                empty_slots=list(range(LANES)),
                sampling_params=greedy_batch_sampling_params(LANES),
            )
            assert out.shape == (LANES, 1) and out.dtype == torch.int32 and rope_deltas.shape == (LANES,)
            return out

        def release_all():
            for slot in range(LANES):
                model.release_request(slot)

        prefill()  # warm call, untimed
        release_all()

        walls, firsts = [], []
        for _rep in range(REPS):
            started = time.perf_counter()
            out = prefill()
            ttnn.synchronize_device(mesh_device)
            walls.append(time.perf_counter() - started)
            firsts.append([int(t) for t in out[:, 0]])
            if _rep < REPS - 1:
                release_all()
        same = all(f == firsts[0] for f in firsts)
        groups = model.last_prefill_groups
        median = statistics.median(walls)

        margins = None
        if recorder is not None:
            margins = []
            for row, prompt in enumerate(prompts):
                info = recorder.by_prompt.get(tuple(prompt))
                margins.append(None if info is None else {"row": row, **info})

        decode_tokens = None
        if os.environ.get("PACK_AB_DECODE") == "1":
            decode_tokens = [[firsts[-1][row]] for row in range(LANES)]
            table = pages.clone()
            spare_of = {row: spare[row] for row in range(LANES)}
            used_spare = set()
            for step in range(DECODE_STEPS):
                positions = torch.tensor([len(p) + step for p in prompts], dtype=torch.int32)
                feed = torch.tensor([[decode_tokens[row][-1]] for row in range(LANES)], dtype=torch.int32)
                for row in range(LANES):
                    index = int(positions[row]) // BLOCK
                    if index >= need[row] and table[row, index] == 0:
                        table[row, index] = spare_of[row]
                        used_spare.add(row)
                out = model.decode_forward(
                    tokens=feed,
                    start_pos=positions,
                    page_table=table,
                    kv_cache=None,
                    enable_trace=True,
                    read_from_device=False,
                    slot_remap=None,
                    reload_inputs=True,
                    reload_page_table=step == 0 or bool(used_spare),
                    reload_sampling_params=step == 0,
                    reset_sampling_state=step == 0,
                    sampling_params=greedy_batch_sampling_params(LANES),
                )
                assert out.shape == (LANES, 1) and out.dtype == torch.int32
                for row in range(LANES):
                    decode_tokens[row].append(int(out[row, 0]))
            release_all()
        else:
            release_all()
    finally:
        model.release_persistent_capture()

    result = {
        "pack_enabled": pack_enabled,
        "mesh_device": os.environ.get("MESH_DEVICE"),
        "lengths": LENGTHS,
        "rep_wall_s": walls,
        "median_s": median,
        "first_tokens": firsts[-1],
        "reps_same_first_tokens": same,
        "last_prefill_groups": groups,
        "margins": margins,
        "decode_tokens": decode_tokens,
    }
    out_path = Path(os.environ.get("PACK_AB_OUT") or tmp_path / "pack_ab.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, default=str) + "\n", encoding="utf-8")
    logger.info(
        f"PACK_AB pack={int(pack_enabled)} median_s={median:.3f} reps={[round(w, 3) for w in walls]} groups={groups}"
    )
    logger.info(f"PACK_AB json {out_path}")
    assert same, f"the {REPS} reps gave different first tokens: {firsts}"
