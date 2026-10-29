# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""W3 device test: the decode-width buckets of ``Qwen38PlainLaneDecoder`` (paged, page size 64).

a. ``test_bucket_view_matches_own_state``: an 8-bucket step over a bucket view of a 32-lane decoder against an 8-lane
   decoder of its own (ids ``own8-all8``, ``own8-holes``, ``view8-all8``, ``view8-holes``); each run saves the live lanes'
   tokens and fp32 logits per step to ``$QWEN38_BUCKET_TEST_OUT/<mode>_<live>.pt``.  Run it once per mode (the two
   decoders do not share a device session).
b. ``test_bucket_view_compare``: the saved own8 and view8 files must agree: equal tokens, bit-equal logits.
c. ``test_bucket_switch_keeps_users``: 8 users, steps 0-9 bucket 8, 10-19 forced to the full bucket, 20-29 bucket 8 again.
d. ``test_bucket_step_times``: median step ms of the decoder with buckets (1, 2, 4, 8, 16, 32) for 1..32 live users.

Run:  QWEN38_BUCKET_TEST_OUT=<dir> pytest models/demos/blackhole/qwen38_flash_next/tests/test_batch_buckets_device.py -v -s -k own8-all8
Env:  as demo/text_demo.py (HF_MODEL, TT_CACHE_PATH, TT_METAL_TRACE_ALLOC_TRACKING=1, MESH_DEVICE).
"""

import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.chat import Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.long_prompt import (
    build_prompt_ids,
    build_user_prompt_ids,
    load_context_text,
)
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    PROMPT_FILE,
    batch_pool_pages,
    device_params,
    greedy_sampling_params,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import Qwen38PlainLaneDecoder, chain_warm_hook
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import Qwen38ForCausalLM
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging

ALLOCATED_CONTEXT = 8192
PAGE_SIZE = 64
STEPS = 24  # test a
SWITCH_STEPS = 30  # test c
EXACT_PREFIX = 8
LIVE_LANES = {"all8": list(range(8)), "holes": [0, 5]}
OUT_VARIABLE = "QWEN38_BUCKET_TEST_OUT"


def out_dir() -> Path:
    value = os.environ.get(OUT_VARIABLE)
    if not value:
        pytest.skip(f"set {OUT_VARIABLE} to the directory the test writes its results to")
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def reference_decode(vm, prompt_ids: list[int], steps: int) -> list[int]:
    """B=1: greedy prefill token t0, then ``steps`` greedy decode steps; returns [t0..t_steps]."""

    out, _ = vm.prefill_forward(
        tokens=torch.tensor([prompt_ids], dtype=torch.int32),
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[len(prompt_ids)],
        start_pos=[0],
        empty_slots=[0],
        sampling_params=greedy_sampling_params(),
    )
    tokens = [int(out.reshape(-1)[0])]
    for step in range(1, steps + 1):
        out = vm.decode_forward(
            tokens=torch.tensor([[tokens[-1]]], dtype=torch.int32),
            start_pos=[len(prompt_ids) + step - 1],
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            read_from_device=True,
            sampling_params=greedy_sampling_params(),
        )
        tokens.append(int(out.reshape(-1)[0]))
    vm.release_request(0)
    return tokens


def needle_prompts(checkpoint, count: int, prompt_len: int | None):
    """``count`` prompts: the P1b needle prompts of length 300 + 181 * lane (``prompt_len`` None), else one length each
    with its own excerpt and pass key.  Returns (prompts, tokenizer, config)."""

    template = Qwen38OfficialChatTemplate(checkpoint)
    tokenizer = template.tokenizer
    config = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    context_text = load_context_text(config, Path(os.environ["TT_CACHE_PATH"]) / "demo-context")
    render = lambda content: template.render(
        [{"role": "user", "content": content}], enable_thinking=False
    ).text  # noqa: E731
    if prompt_len is None:
        prompts = [build_prompt_ids(tokenizer, render, config, context_text, 300 + 181 * u)[0] for u in range(count)]
    else:
        prompts = [
            build_user_prompt_ids(tokenizer, render, config, context_text, prompt_len, u)[0] for u in range(count)
        ]
    return prompts, tokenizer, config


@contextmanager
def open_decoder(mesh_device, lanes: int, prompts, gen: int, buckets=None):
    """The decoder (paged, 64-row pages) over a fresh adapter; yields (decoder, vm, checkpoint).  Released on exit."""

    checkpoint = checkpoint_directory()
    if not os.environ.get("TT_CACHE_PATH"):
        pytest.fail("set TT_CACHE_PATH to the cache root (the book download goes to <root>/demo-context)")
    max_blocks = -(-ALLOCATED_CONTEXT // PAGE_SIZE)
    paging = Qwen38LanePaging(
        page_size=PAGE_SIZE,
        num_pages=batch_pool_pages([len(p) for p in prompts], gen, PAGE_SIZE),
        max_blocks=-(-max_blocks // 8) * 8,
    )
    mesh_device.enable_program_cache()
    decoder = Qwen38PlainLaneDecoder(lanes, paging=paging, decode_steps=gen, buckets=buckets)
    decoder.warm_logits_gather = True  # the test reads the full lane logits under the closed miss guard
    with chain_warm_hook(decoder.prepare):
        vm = Qwen38ForCausalLM.initialize_vllm_model(
            SimpleNamespace(_name_or_path=str(checkpoint)),
            mesh_device,
            max_batch_size=1,
            max_seq_len=ALLOCATED_CONTEXT - RESIDENT_CONTEXT_HEADROOM,
        )
    try:
        decoder.capture(vm)
        logger.info(f"decoder {lanes} lanes buckets {decoder.buckets} captured: {decoder.timing}")
        yield decoder, vm, checkpoint
    finally:
        decoder.release()
        vm.release_persistent_capture()


def lock_step(decoder, current: list[int], live: list[int]) -> list[int]:
    """One step: idle lanes parked first (as test_lane_curve.py), idle lanes consume token 0; returns the next token
    list (live lanes' new ids, 0 for the others)."""

    idle = [u for u in range(decoder.lanes) if u not in live]
    if idle:
        decoder.park_idle(idle)
    out = decoder.step(current)
    return [out[u] if u in live else 0 for u in range(decoder.lanes)]


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("live", ["all8", "holes"])
@pytest.mark.parametrize("mode", ["own8", "view8"])
@pytest.mark.timeout(5400)
def test_bucket_view_matches_own_state(mesh_device, mode, live):
    out = out_dir()
    lanes_live = LIVE_LANES[live]
    checkpoint = checkpoint_directory()
    prompts, _tokenizer, _config = needle_prompts(checkpoint, 8, None)
    users = {u: prompts[u] for u in lanes_live}
    lanes, buckets = (8, None) if mode == "own8" else (32, (8, 32))
    with open_decoder(mesh_device, lanes, list(users.values()), STEPS + 1, buckets) as (decoder, _vm, _ckpt):
        current = [0] * lanes
        for u, prompt in users.items():
            current[u] = decoder.admit(u, prompt)
        saved = {"live": lanes_live, "tokens": [], "logits": [], "first": [current[u] for u in lanes_live]}
        for _ in range(STEPS):
            current = lock_step(decoder, current, lanes_live)
            saved["tokens"].append([current[u] for u in lanes_live])
            saved["logits"].append(decoder.read_full_logits()[lanes_live].to(torch.float32).clone())
        saved["logits"] = torch.stack(saved["logits"])
        torch.save(saved, out / f"{mode}_{live}.pt")
        logger.info(f"{mode}-{live}: bucket steps {decoder.bucket_steps}")
        if mode == "view8":
            assert decoder.bucket_steps[8] == STEPS and decoder.bucket_steps[32] == 0, decoder.bucket_steps


@pytest.mark.parametrize("live", ["all8", "holes"])
def test_bucket_view_compare(live):
    out = out_dir()
    own_path, view_path = out / f"own8_{live}.pt", out / f"view8_{live}.pt"
    if not own_path.exists() or not view_path.exists():
        pytest.skip(f"{own_path.name} and {view_path.name} are not both in {out}: run the own8 and view8 cases first")
    own, view = torch.load(own_path), torch.load(view_path)
    assert own["live"] == view["live"] and own["first"] == view["first"]
    for step, (a, b) in enumerate(zip(own["tokens"], view["tokens"])):
        assert a == b, f"step {step}: tokens differ own {a} view {b}"
    assert own["logits"].shape == view["logits"].shape
    if not torch.equal(own["logits"], view["logits"]):
        diff = (own["logits"] - view["logits"]).abs()
        per_step_lane = diff.reshape(diff.shape[0], diff.shape[1], -1).amax(-1)
        step, lane = [int(i) for i in (per_step_lane > 0).nonzero()[0]]
        pytest.fail(
            f"logits differ: first at step {step} lane {own['live'][lane]}, max abs diff there "
            f"{per_step_lane[step, lane]:.6g}, overall {diff.max():.6g}"
        )


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(5400)
def test_bucket_switch_keeps_users(mesh_device):
    checkpoint = checkpoint_directory()
    prompts, tokenizer, config = needle_prompts(checkpoint, 8, None)
    lanes_live = list(range(8))
    with open_decoder(mesh_device, 32, prompts, SWITCH_STEPS + 1, (8, 32)) as (decoder, vm, _ckpt):
        references = [reference_decode(vm, prompt, SWITCH_STEPS) for prompt in prompts]
        firsts = [decoder.admit(u, prompt) for u, prompt in enumerate(prompts)]
        assert firsts == [ref[0] for ref in references], f"admission t0 {firsts} vs reference"
        outputs = [[t] for t in firsts]
        current = firsts + [0] * 24
        finite = True
        for step in range(SWITCH_STEPS):
            decoder.force_bucket = 32 if 10 <= step < 20 else None
            current = lock_step(decoder, current, lanes_live)
            for u in lanes_live:
                outputs[u].append(current[u])
            logits = decoder.read_full_logits()
            assert logits.shape[0] == (32 if 10 <= step < 20 else 8), f"step {step}: logits rows {logits.shape[0]}"
            finite = finite and bool(torch.isfinite(logits[lanes_live].to(torch.float32)).all())
        steps_per_bucket = dict(decoder.bucket_steps)
        logger.info(f"bucket steps {steps_per_bucket}")
    assert steps_per_bucket[8] == 20 and steps_per_bucket[32] == 10, steps_per_bucket
    assert finite, "a logits row holds a non-finite value"
    for u in lanes_live:
        out, ref = outputs[u], references[u]
        decoded = tokenizer.decode(out, skip_special_tokens=True)
        diffs = [i for i in range(1, SWITCH_STEPS + 1) if out[i] != ref[i]]
        logger.info(f"lane {u}: first diff vs B=1 reference {diffs[0] if diffs else None}")
        assert out[0] == ref[0], f"lane {u}: first token"
        assert out[1 : EXACT_PREFIX + 1] == ref[1 : EXACT_PREFIX + 1], f"lane {u}: first {EXACT_PREFIX} tokens differ"
        assert config["pass_key"] in decoded, f"lane {u}: pass key not in {decoded[:300]!r}"


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(7200)
def test_bucket_step_times(mesh_device):
    out = out_dir()
    warm, timed = 16, 32
    gen = warm + timed + 8
    checkpoint = checkpoint_directory()
    prompts, _tokenizer, _config = needle_prompts(checkpoint, 32, 128)
    results = []
    with open_decoder(mesh_device, 32, prompts, gen, (1, 2, 4, 8, 16, 32)) as (decoder, _vm, _ckpt):
        for live_count in (1, 2, 4, 8, 16, 32):
            lanes_live = list(range(live_count))
            current = [0] * 32
            for u in lanes_live:
                current[u] = decoder.admit(u, prompts[u])
            step_ms, used = [], set()
            for _ in range(warm + timed):
                started = time.perf_counter()
                current = lock_step(decoder, current, lanes_live)
                step_ms.append(1000.0 * (time.perf_counter() - started))
                used.add(decoder.active.lanes)
            median = statistics.median(step_ms[warm:])
            results.append({"live": live_count, "bucket": sorted(used), "median_step_ms": median})
            logger.info(f"live {live_count}: bucket {sorted(used)} median step {median:.2f} ms")
            for u in lanes_live:
                decoder.release_lane(u)
            decoder.park_idle(range(32))
        report = {
            "schema": "qwen38-bucket-step-times/v1",
            "buckets": list(decoder.buckets),
            "warm_steps": warm,
            "timed_steps": timed,
            "rounds": results,
            "bucket_steps": decoder.bucket_steps,
            "timing": {k: v for k, v in decoder.timing.items() if k.startswith("bucket_") or k.endswith("_s")},
        }
    (out / "bucket_step_times.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for row in results:
        assert row["bucket"] == [next(r for r in (1, 2, 4, 8, 16, 32) if r >= row["live"])], row
