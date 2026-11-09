# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""On-mode (QWEN38_MAMBA_SPEC=1: block 1024, checkpoint split) vs off-mode (block 128, no split) greedy streams of the MTP lanes
adapter on real text prompts (3 needle prompts of ~3k, ~6k, ~12k tokens and chat prompts P1..P3), 128 tokens each, one request
at a time.  Also the 12k prompt on-mode without a checkpoint.  Results go to ONOFF_RESULTS (default
$QWEN38_ONOFF_RESULTS or onoff_results.json in the cwd).  The off test runs first and writes its part; the on test adds its
part and prints the table.

Env: MESH_DEVICE=P150x8, HF_MODEL, TT_CACHE_PATH, TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0, QWEN38_VLLM_MTP_K=3
(the tests set QWEN38_MAMBA_SPEC themselves).
"""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.long_prompt import user_pass_key
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    build_chat_prompts,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_batch_buckets_device import needle_prompts
from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_mtp_prefix_device import (
    ALLOCATED,
    BLOCK,
    GROUPS,
    MAX_SEQ_LEN,
    NUM_BLOCKS,
    Harness,
    Req,
    first_diff,
    kv_entries,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm_mtp import Qwen38MTPForCausalLM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

LANES = 2
TOKENS = 128
OFF_BLOCK = 128
OFF_BLOCKS = 1026
NEEDLE_LENGTHS = (3000, 6000, 12000)
CHAT_NAMES = ("P1", "P2", "P3")
P1_KEY = "58213"
RESULTS = Path(os.environ.get("QWEN38_ONOFF_RESULTS", "onoff_results.json"))


def step(name: str) -> None:
    logger.info(f"ONOFF step: {name}")


def load_results() -> dict:
    try:
        return json.loads(RESULTS.read_text())
    except Exception:  # noqa: BLE001
        return {}


def save_results(update: dict) -> None:
    data = load_results()
    for key, value in update.items():
        data.setdefault(key, {}).update(value)
    RESULTS.write_text(json.dumps(data, indent=1))


def build_prompts(checkpoint):
    """[(name, ids, pass keys)]: 3 needle prompts and chat prompts P1..P3."""

    out = []
    for length in NEEDLE_LENGTHS:
        prompts, tokenizer, config = needle_prompts(checkpoint, 1, length)
        assert len(prompts[0]) == length, (length, len(prompts[0]))
        out.append((f"needle{length}", [int(t) for t in prompts[0]], [user_pass_key(config, 0)]))
    _, chat, _ = build_chat_prompts(checkpoint)
    by_name = dict(chat)
    for name in CHAT_NAMES:
        out.append((name, by_name[name], [P1_KEY] if name == "P1" else []))
    return out, tokenizer


def stop_ids(tokenizer) -> set[int]:
    ids = set()
    for text in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(text)
        if isinstance(tid, int) and tid >= 0:
            ids.add(tid)
    return ids


def cut(tokens: list[int], stops: set[int]) -> list[int]:
    out = []
    for t in tokens[:TOKENS]:
        out.append(int(t))
        if t in stops:
            break
    return out


def warm(model, lanes: int) -> None:
    width = -(-MAX_SEQ_LEN // model.block_size)
    prefill_kwargs = dict(kv_cache=None, can_sample_on_device=False)
    decode_kwargs = dict(kv_cache=None, max_batch_size=lanes, num_blocks=width, can_sample_on_device=True)
    step("warmup_model_prefill eager")
    model.warmup_model_prefill(enable_trace=False, **prefill_kwargs)
    step("warmup_model_decode eager")
    model.warmup_model_decode(enable_trace=False, **decode_kwargs)
    if hasattr(model, "already_warmed_up_prefill"):
        model.already_warmed_up_prefill = False
    step("warmup_model_prefill traced")
    model.warmup_model_prefill(enable_trace=True, **prefill_kwargs)
    step("warmup_model_decode traced")
    model.warmup_model_decode(enable_trace=True, **decode_kwargs)
    step("warm-ups done")


class OffHarness(Harness):
    """Off mode: block size 128, no mamba_state, no checkpoint."""

    def __init__(self, model) -> None:
        self.model = model
        self.lanes = model.lanes
        self.k = model.drafts
        self.width = model.paging.max_blocks
        self.free_kv = list(range(1, OFF_BLOCKS))

    def new_req(self, name, slot, prompt, **_):
        count = -(-(len(prompt) + TOKENS + 16) // OFF_BLOCK)
        own, self.free_kv = self.free_kv[:count], self.free_kv[count:]
        req = Req(name, slot, list(prompt), kv=list(own), own_kv=own)
        self.model.zero_kv_cache_blocks(kv_cache=None, block_ids=own)
        return req

    def free(self, req) -> None:
        self.free_kv = sorted(self.free_kv + req.own_kv)

    def prefill(self, items):
        reqs = [item[0] for item in items]
        width = max(end for _, _, end, _ in items)
        tokens = torch.zeros(len(items), width, dtype=torch.int32)
        table = torch.zeros(len(items), self.width, dtype=torch.int32)
        for i, (req, _s, end, _f) in enumerate(items):
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
        )
        return [int(out[i, 0].argmax()) for i in range(len(items))]


class OnHarness(Harness):
    def new_req(self, name, slot, prompt, shared_kv=(), checkpoints=()):
        count = -(-(len(prompt) + TOKENS + 16) // BLOCK) + 1
        own = self.blocks.take_kv(count)
        req = Req(name, slot, list(prompt), kv=list(own), own_kv=own)
        req.windows = self.blocks.take_state()
        for n in checkpoints:
            req.checkpoints[n] = self.blocks.take_state()
        self.model.zero_kv_cache_blocks(kv_cache=None, block_ids=own)
        return req


def generate(h, name, prompt, checkpoints) -> list[int]:
    req = h.new_req(name, 0, prompt, checkpoints=checkpoints)
    first = h.cold(req)
    lv = h.new_live(req, first)
    h.decode_until([lv], TOKENS)
    h.model.release_request(req.slot)
    h.free(req)
    return list(lv.tokens)


def last_full_block(length: int) -> list[int]:
    n = (length // BLOCK) * BLOCK
    if n == length:
        n -= BLOCK
    return [n] if n > 0 else []


def open_model(mesh_device, checkpoint, monkeypatch, on: bool):
    if on:
        monkeypatch.setenv("QWEN38_MAMBA_SPEC", "1")
    else:
        monkeypatch.delenv("QWEN38_MAMBA_SPEC", raising=False)
    if not os.environ.get("QWEN38_VLLM_MTP_K"):
        monkeypatch.setenv("QWEN38_VLLM_MTP_K", "3")
    monkeypatch.setenv("QWEN38_MTP_HYBRID_SINGLE", "0")
    mesh_device.enable_program_cache()
    model = Qwen38MTPForCausalLM.initialize_vllm_model(
        SimpleNamespace(_name_or_path=str(checkpoint)), mesh_device, max_batch_size=LANES, max_seq_len=MAX_SEQ_LEN
    )
    assert model.lanes_mode and model.lanes == LANES
    return model


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(5400)
def test_a_off_mode(mesh_device, monkeypatch):
    import ttnn

    checkpoint = checkpoint_directory()
    prompts, tokenizer = build_prompts(checkpoint)
    stops = stop_ids(tokenizer)
    model = open_model(mesh_device, checkpoint, monkeypatch, on=False)
    try:
        model.allocate_kv_cache((OFF_BLOCKS, 1, OFF_BLOCK, 128), ttnn.bfloat16, 12)
        assert model.block_size == OFF_BLOCK
        warm(model, LANES)
        if hasattr(model, "mtp_hybrid_single"):
            model.mtp_hybrid_single = False
        h = OffHarness(model)
        for name, ids, _keys in prompts:
            step(f"off {name} ({len(ids)} tokens)")
            t0 = time.time()
            tokens = cut(generate(h, name, ids, ()), stops)
            save_results({"off": {name: {"length": len(ids), "tokens": tokens}}})
            logger.info(f"ONOFF off {name}: {len(tokens)} tokens in {time.time() - t0:.1f} s")
    finally:
        model.release_persistent_capture()


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(5400)
def test_b_on_mode_and_report(mesh_device, monkeypatch):
    checkpoint = checkpoint_directory()
    prompts, tokenizer = build_prompts(checkpoint)
    stops = stop_ids(tokenizer)
    model = open_model(mesh_device, checkpoint, monkeypatch, on=True)
    try:
        model.kv_cache = model.allocate_kv_cache_per_layer(kv_entries())
        assert model.block_size == BLOCK and model.num_blocks == NUM_BLOCKS and ALLOCATED == 32768 and GROUPS == 3
        warm(model, LANES)
        model.mtp_hybrid_single = False
        h = OnHarness(model)
        for name, ids, _keys in prompts:
            step(f"on {name} ({len(ids)} tokens) checkpoint {last_full_block(len(ids))}")
            t0 = time.time()
            tokens = cut(generate(h, name, ids, last_full_block(len(ids))), stops)
            save_results(
                {"on": {name: {"length": len(ids), "checkpoint": last_full_block(len(ids)), "tokens": tokens}}}
            )
            logger.info(f"ONOFF on {name}: {len(tokens)} tokens in {time.time() - t0:.1f} s")
        name, ids, _keys = prompts[2]
        step(f"on {name} without checkpoint")
        tokens = cut(generate(h, name + "-nockpt", ids, ()), stops)
        save_results({"on_nockpt": {name: {"length": len(ids), "tokens": tokens}}})
    finally:
        model.release_persistent_capture()

    data = load_results()
    summary = {}
    lines = ["prompt | L | first diff | identical | key off | key on | on-nockpt vs off (12k)"]
    for name, ids, keys in prompts:
        off = data.get("off", {}).get(name, {}).get("tokens")
        on = data["on"][name]["tokens"]
        row = {"length": len(ids), "pass_keys": keys}
        text_on = tokenizer.decode(on)
        row["on_text"] = text_on[:300]
        row["key_on"] = any(k in tokenizer.decode(on) for k in keys) if keys else None
        if off is not None:
            text_off = tokenizer.decode(off)
            row["off_text"] = text_off[:300]
            row["key_off"] = any(k in text_off for k in keys) if keys else None
            row["first_diff"] = first_diff(off, on)
            n = row["first_diff"] if row["first_diff"] is not None else len(on)
            row["identical"] = n
            row["lengths"] = [len(off), len(on)]
        nock = data.get("on_nockpt", {}).get(name, {}).get("tokens")
        if nock is not None and off is not None:
            row["nockpt_first_diff_vs_off"] = first_diff(off, nock)
            row["nockpt_identical_vs_off"] = (
                row["nockpt_first_diff_vs_off"] if row["nockpt_first_diff_vs_off"] is not None else len(nock)
            )
            row["nockpt_first_diff_vs_on"] = first_diff(on, nock)
            row["nockpt_text"] = tokenizer.decode(nock)[:300]
        summary[name] = row
        lines.append(
            f"{name} | {len(ids)} | {row.get('first_diff')} | {row.get('identical')} | {row.get('key_off')} | {row['key_on']} | "
            f"{row.get('nockpt_first_diff_vs_off', '-')}"
        )
    save_results({"summary": summary})
    for line in lines:
        logger.info(f"ONOFF TABLE {line}")
    print("\n".join(lines))
