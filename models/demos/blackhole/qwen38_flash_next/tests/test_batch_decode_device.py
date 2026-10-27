# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P1b device test: the residue-free plain lane body at B = 2, 4, 8, 16, 32 users against the same prompts decoded alone
(B=1).

Each lane holds a War and Peace needle prompt of its own length (300 + 181 * lane: mixed residues mod 4, one trace for
every lane).  The B=1 reference is the adapter's greedy prefill plus 32 decode steps; the lanes are admitted from the
same B=1 prefill and replay 32 batched steps.  The first 8 tokens of every lane must equal the reference and the pass
key must appear in the lane output; the full 32-token equality is logged (near ties may differ).

Paged ids (``batch_N_pagedS``, page size S): the lanes' KV is a page pool with a shuffled page table (P1c-B); the same
checks.

Run:  pytest models/demos/blackhole/qwen38_flash_next/tests/test_batch_decode_device.py -v -s -k batch_2
Env:  as demo/text_demo.py (HF_MODEL, TT_CACHE_PATH, TT_METAL_TRACE_ALLOC_TRACKING=1, MESH_DEVICE).
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
from models.demos.blackhole.qwen38_flash_next.chat import Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.long_prompt import build_prompt_ids, load_context_text
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    PROMPT_FILE,
    device_params,
    greedy_sampling_params,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import (
    Qwen38PlainLaneDecoder,
    chain_warm_hook,
    lane_page_count,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import Qwen38ForCausalLM
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging

ALLOCATED_CONTEXT = 8192
STEPS = 32
EXACT_PREFIX = 8


def reference_decode(vm, prompt_ids: list[int]) -> tuple[list[int], float]:
    """B=1: greedy prefill token t0, then STEPS greedy decode steps; returns ([t0..tSTEPS], mean decode seconds)."""

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
    seconds = []
    for step in range(1, STEPS + 1):
        started = time.perf_counter()
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
        seconds.append(time.perf_counter() - started)
    vm.release_request(0)
    return tokens, sum(seconds) / len(seconds)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "lanes, page_size",
    [
        pytest.param(2, 0, id="batch_2", marks=pytest.mark.timeout(5400)),
        pytest.param(4, 0, id="batch_4", marks=pytest.mark.timeout(5400)),
        pytest.param(8, 0, id="batch_8", marks=pytest.mark.timeout(5400)),
        pytest.param(16, 0, id="batch_16", marks=pytest.mark.timeout(5400)),
        pytest.param(32, 0, id="batch_32", marks=pytest.mark.timeout(5400)),
        pytest.param(2, 64, id="batch_2_paged64", marks=pytest.mark.timeout(5400)),
        pytest.param(8, 64, id="batch_8_paged64", marks=pytest.mark.timeout(5400)),
        pytest.param(16, 64, id="batch_16_paged64", marks=pytest.mark.timeout(5400)),
        pytest.param(32, 64, id="batch_32_paged64", marks=pytest.mark.timeout(5400)),
        pytest.param(8, 32, id="batch_8_paged32", marks=pytest.mark.timeout(5400)),
    ],
)
def test_plain_lanes_match_single_user(mesh_device, lanes, page_size, tmp_path, record_property):
    try:
        checkpoint = checkpoint_directory()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    if not os.environ.get("TT_CACHE_PATH"):
        pytest.fail("set TT_CACHE_PATH to the cache root (the book download goes to <root>/demo-context)")
    max_seq_len = ALLOCATED_CONTEXT - RESIDENT_CONTEXT_HEADROOM

    template = Qwen38OfficialChatTemplate(checkpoint)
    tokenizer = template.tokenizer
    config = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    context_text = load_context_text(config, Path(os.environ["TT_CACHE_PATH"]) / "demo-context")
    lengths = [300 + 181 * i for i in range(lanes)]
    prompts = [
        build_prompt_ids(
            tokenizer,
            lambda content: template.render([{"role": "user", "content": content}], enable_thinking=False).text,
            config,
            context_text,
            length,
        )[0]
        for length in lengths
    ]
    assert len({len(p) % 4 for p in prompts}) > 1 or lanes < 2  # mixed residues

    paging = None
    if page_size:
        pages = [lane_page_count(len(p), STEPS, page_size) for p in prompts]  # P + 33 rows
        max_blocks = -(-ALLOCATED_CONTEXT // page_size)
        paging = Qwen38LanePaging(page_size=page_size, num_pages=1 + sum(pages) + 8, max_blocks=-(-max_blocks // 8) * 8)
        logger.info(
            f"paged KV: page_size {page_size}, {paging.num_pages} pages = {paging.pool_rows} pool rows vs "
            f"{lanes * ALLOCATED_CONTEXT} dense rows ({lanes} x {ALLOCATED_CONTEXT}); pages per lane {pages}"
        )

    mesh_device.enable_program_cache()
    started = time.perf_counter()
    decoder = Qwen38PlainLaneDecoder(lanes, paging=paging, decode_steps=STEPS)
    with chain_warm_hook(decoder.prepare):
        vm = Qwen38ForCausalLM.initialize_vllm_model(
            SimpleNamespace(_name_or_path=str(checkpoint)), mesh_device, max_batch_size=1, max_seq_len=max_seq_len
        )
    model_ready_s = time.perf_counter() - started
    logger.info(f"model ready in {model_ready_s:.1f} s (lane prepare {decoder.timing.get('allocate_s', 0):.1f} s)")

    try:
        references, ref_step_s = [], []
        for u, prompt in enumerate(prompts):
            tokens, mean_s = reference_decode(vm, prompt)
            references.append(tokens)
            ref_step_s.append(mean_s)
            logger.info(f"reference lane {u} (P={len(prompt)}): {tokens[:8]}...")

        started = time.perf_counter()
        decoder.capture(vm)
        build_s = time.perf_counter() - started
        logger.info(f"lane decoder captured in {build_s:.1f} s: {decoder.timing}")

        firsts = [decoder.admit(u, prompt) for u, prompt in enumerate(prompts)]
        admit_s = decoder.timing.get("admit_s", [])
        logger.info(f"admission per lane (after the B=1 prefill): mean {sum(admit_s) / len(admit_s):.3f} s, {admit_s}")
        if paging is not None:
            logger.info(f"page table rows: {decoder.page_rows}")
        assert firsts == [ref[0] for ref in references], f"admission t0 {firsts} vs reference"

        outputs = [[t] for t in firsts]
        step_s = []
        current = list(firsts)
        for _ in range(STEPS):
            started = time.perf_counter()
            current = decoder.step(current)
            step_s.append(time.perf_counter() - started)
            for u in range(lanes):
                outputs[u].append(current[u])
    finally:
        decoder.release()
        vm.release_persistent_capture()

    lane_ms = 1000.0 * sum(step_s) / len(step_s)
    ref_ms = 1000.0 * sum(ref_step_s) / len(ref_step_s)
    aggregate = lanes / (lane_ms / 1000.0)
    results = []
    for u in range(lanes):
        ref, out = references[u], outputs[u]
        diffs = [i for i in range(1, STEPS + 1) if out[i] != ref[i]]
        decoded = tokenizer.decode(out, skip_special_tokens=True)
        found = config["pass_key"] in decoded
        results.append(
            {
                "lane": u,
                "prompt_len": len(prompts[u]),
                "exact_match": not diffs,
                "first_diff_index": diffs[0] if diffs else None,
                "prefix8_match": out[1 : EXACT_PREFIX + 1] == ref[1 : EXACT_PREFIX + 1],
                "pass_key_found": found,
                "lane_tokens": out,
                "reference_tokens": ref,
                "decoded": decoded,
            }
        )
        logger.info(
            f"lane {u} P={len(prompts[u])}: exact32={'yes' if not diffs else 'no'} "
            f"first_diff={results[-1]['first_diff_index']} pass_key={'found' if found else 'MISSING'}"
        )
    logger.info(
        f"B={lanes}: lane step {lane_ms:.2f} ms | aggregate {aggregate:.1f} tok/s | B=1 reference {ref_ms:.2f} ms/tok | "
        f"exact lanes {sum(r['exact_match'] for r in results)}/{lanes} | model ready {model_ready_s:.1f} s | "
        f"warm {decoder.timing.get('warm_s', 0):.1f} s capture {decoder.timing.get('capture_s', 0):.1f} s"
    )
    report_path = tmp_path / "batch-decode-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "qwen38-plain-lanes-p1b/v1",
                "lanes": lanes,
                "page_size": page_size,
                "pool_rows": None if paging is None else paging.pool_rows,
                "dense_rows": lanes * ALLOCATED_CONTEXT,
                "page_rows": decoder.page_rows,
                "steps": STEPS,
                "model_ready_s": model_ready_s,
                "decoder_build_s": build_s,
                "decoder_timing": decoder.timing,
                "lane_step_ms": lane_ms,
                "lane_step_seconds": step_s,
                "aggregate_tok_s": aggregate,
                "reference_ms_per_token": ref_ms,
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    record_property("batch_decode_report", str(report_path))

    for r in results:
        assert r["pass_key_found"], f"lane {r['lane']}: pass key not in {r['decoded'][:300]!r}"
        assert r["prefix8_match"], f"lane {r['lane']}: first {EXACT_PREFIX} tokens differ from the B=1 reference"
