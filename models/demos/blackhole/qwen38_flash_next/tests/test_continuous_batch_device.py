# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P1d device test: continuous batching on the residue-free plain lane body (requests join and leave a running batch).

Requests of mixed prompt length, token budget and arrival step run through ``Qwen38ContinuousBatch``: a finished lane
is freed (paged: its pages go back to the pool, its table row is zeroed), a waiting request takes the lowest free lane,
pages are allocated on demand and the pool is sized so that freed pages are reused.  Idle lanes run at position 0 on the
scratch page.  Every request's tokens are compared with its B=1 reference (adapter greedy prefill + ``max_new - 1``
decode steps, computed before the lane capture).  Gate: the first ``min(8, n)`` tokens equal the reference and the
pass key is in the output of every request with ``max_new_tokens >= 32``; the full match and the first divergence are
logged (near ties may differ).

Ids: ``cont_8_paged64``, ``cont_32_paged64``, ``cont_8_paged64_32k``, ``cont_8`` (unpaged).

Run:  pytest models/demos/blackhole/qwen38_flash_next/tests/test_continuous_batch_device.py -v -s -k cont_8_paged64
Env:  as test_batch_decode_device.py.
"""

import json
import os
import random
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
    Qwen38ContinuousBatch,
    Qwen38PlainLaneDecoder,
    Qwen38Request,
    chain_warm_hook,
    lane_page_count,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import Qwen38ForCausalLM
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging

EXACT_PREFIX = 8
PASS_KEY_MIN_TOKENS = 32
MIN_PROMPT = 120

# token budgets, cycled: 1 (admit then release, no step), several >= 32
MAX_NEW_CYCLE = [48, 32, 1, 20, 40, 8, 33, 16, 44, 5, 36, 24, 2, 32, 12, 48, 1, 28, 34, 10]


def request_specs(kind: str, count: int, seed: int) -> list[dict]:
    """(prompt_len, max_new_tokens, arrival_step) per request, in submission order."""

    rng = random.Random(seed)
    if kind == "32k":
        lengths = [16000, 150, 30000, 700, 1900, 260, 1100, 480, 90 + MIN_PROMPT, 2400, 330, 820]
        news = [20, 32, 24, 16, 28, 1, 32, 18, 26, 30, 17, 22]
        arrivals = [0, 0, 0, 0, 0, 3, 6, 9, 14, 20, 26, 33]
        return [{"prompt_len": p, "max_new": n, "arrival": a} for p, n, a in zip(lengths, news, arrivals)]
    lengths = [rng.randint(MIN_PROMPT, 3000) for _ in range(count)]
    # a long prompt early and a short one right behind it: they meet in one lane when the long one finishes first
    lengths[0], lengths[1] = 2900, 130
    news = [MAX_NEW_CYCLE[i % len(MAX_NEW_CYCLE)] for i in range(count)]
    start = 5 if count <= 20 else 24
    arrivals = [0] * start + sorted(rng.randint(1, 70 if count <= 20 else 30) for _ in range(count - start))
    return [{"prompt_len": p, "max_new": n, "arrival": a} for p, n, a in zip(lengths, news, arrivals)]


def reference_tokens(vm, prompt_ids: list[int], max_new: int) -> tuple[list[int], float]:
    """B=1: greedy prefill token t0, then ``max_new - 1`` greedy decode steps; returns (tokens, mean decode s)."""

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
    for step in range(1, max_new):
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
    return tokens, (sum(seconds) / len(seconds) if seconds else 0.0)


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "lanes, page_size, context, kind, count, reserve_largest",
    [
        pytest.param(8, 64, 8192, "short", 20, 8, id="cont_8_paged64", marks=pytest.mark.timeout(7200)),
        pytest.param(32, 64, 8192, "short", 48, 32, id="cont_32_paged64", marks=pytest.mark.timeout(10800)),
        pytest.param(8, 64, 32768, "32k", 12, 8, id="cont_8_paged64_32k", marks=pytest.mark.timeout(14400)),
        pytest.param(8, 0, 8192, "short", 20, 0, id="cont_8", marks=pytest.mark.timeout(7200)),
    ],
)
def test_continuous_batch_matches_single_user(
    mesh_device, lanes, page_size, context, kind, count, reserve_largest, tmp_path, record_property
):
    try:
        checkpoint = checkpoint_directory()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    if not os.environ.get("TT_CACHE_PATH"):
        pytest.fail("set TT_CACHE_PATH to the cache root (the book download goes to <root>/demo-context)")
    max_seq_len = context - RESIDENT_CONTEXT_HEADROOM

    template = Qwen38OfficialChatTemplate(checkpoint)
    tokenizer = template.tokenizer
    config = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
    context_text = load_context_text(config, Path(os.environ["TT_CACHE_PATH"]) / "demo-context")
    specs = request_specs(kind, count, seed=20261024)
    prompts = [
        build_prompt_ids(
            tokenizer,
            lambda content: template.render([{"role": "user", "content": content}], enable_thinking=False).text,
            config,
            context_text,
            spec["prompt_len"],
        )[0]
        for spec in specs
    ]
    assert len({len(p) % 4 for p in prompts}) > 1  # mixed residues
    assert any(s["max_new"] == 1 for s in specs) and any(s["max_new"] >= PASS_KEY_MIN_TOKENS for s in specs)
    assert sum(1 for s in specs if s["arrival"] == 0) < lanes or kind == "32k"

    paging = None
    if page_size:
        budgets = sorted(
            (lane_page_count(len(p), s["max_new"], page_size) for p, s in zip(prompts, specs)), reverse=True
        )
        max_blocks = -(-context // page_size)
        paging = Qwen38LanePaging(
            page_size=page_size,
            num_pages=1 + sum(budgets[:reserve_largest]) + 4,
            max_blocks=-(-max_blocks // 8) * 8,
        )
        logger.info(
            f"paged KV: page_size {page_size}, {paging.num_pages} pages = {paging.pool_rows} pool rows vs "
            f"{lanes * context} dense rows ({lanes} x {context}); budgets (largest first) {budgets[:reserve_largest]}, "
            f"total budget of all requests {sum(budgets)}"
        )

    mesh_device.enable_program_cache()
    started = time.perf_counter()
    decoder = Qwen38PlainLaneDecoder(lanes, paging=paging)
    with chain_warm_hook(decoder.prepare):
        vm = Qwen38ForCausalLM.initialize_vllm_model(
            SimpleNamespace(_name_or_path=str(checkpoint)), mesh_device, max_batch_size=1, max_seq_len=max_seq_len
        )
    model_ready_s = time.perf_counter() - started
    logger.info(f"model ready in {model_ready_s:.1f} s (lane prepare {decoder.timing.get('allocate_s', 0):.1f} s)")

    try:
        references, ref_s = [], time.perf_counter()
        for i, (prompt, spec) in enumerate(zip(prompts, specs)):
            tokens, _ = reference_tokens(vm, prompt, spec["max_new"])
            references.append(tokens)
            logger.info(f"reference {i} (P={len(prompt)}, n={spec['max_new']}): {tokens[:8]}")
        logger.info(f"references took {time.perf_counter() - ref_s:.1f} s")

        started = time.perf_counter()
        decoder.capture(vm)
        build_s = time.perf_counter() - started
        logger.info(f"lane decoder captured in {build_s:.1f} s: {decoder.timing}")

        requests = [Qwen38Request(i, prompts[i], s["max_new"], s["arrival"]) for i, s in enumerate(specs)]
        batch = Qwen38ContinuousBatch(decoder, requests)
        metrics = batch.run()
    finally:
        decoder.release()
        vm.release_persistent_capture()

    lane_history: dict[int, list[int]] = {}
    for request_id, lane in batch.admission_order:
        lane_history.setdefault(lane, []).append(request_id)
    long_then_short = [
        (lane, a, b)
        for lane, ids in lane_history.items()
        for a, b in zip(ids, ids[1:])
        if len(prompts[a]) >= 2 * len(prompts[b])
    ]
    logger.info(f"lane history {lane_history}; long-then-short pairs in one lane: {long_then_short}")

    results = []
    for rec in metrics["requests"]:
        i = rec["request_id"]
        ref, out = references[i], rec["tokens"]
        n = specs[i]["max_new"]
        gate = min(EXACT_PREFIX, n)
        diffs = [k for k in range(n) if out[k] != ref[k]]
        decoded = tokenizer.decode(out, skip_special_tokens=True)
        needs_key = n >= PASS_KEY_MIN_TOKENS
        found = config["pass_key"] in decoded
        results.append(
            {
                "request": i,
                "lane": rec["lane"],
                "prompt_len": len(prompts[i]),
                "max_new_tokens": n,
                "arrival_step": specs[i]["arrival"],
                "admit_step": rec["admit_step"],
                "finish_step": rec["finish_step"],
                "prefill_s": rec["prefill_s"],
                "import_s": rec["import_s"],
                "pages_used": rec["pages_used"],
                "budget_pages": rec["budget_pages"],
                "exact_match": not diffs,
                "first_diff_index": diffs[0] if diffs else None,
                "gate_prefix_match": out[:gate] == ref[:gate],
                "pass_key_required": needs_key,
                "pass_key_found": found,
                "tokens": out,
                "reference_tokens": ref,
                "decoded": decoded,
            }
        )
        logger.info(
            f"req {i} lane {rec['lane']} P={len(prompts[i])} n={n} steps {rec['admit_step']}..{rec['finish_step']} "
            f"pages {rec['pages_used']}/{rec['budget_pages']}: exact={'yes' if not diffs else 'no'} "
            f"first_diff={results[-1]['first_diff_index']} gate={'ok' if results[-1]['gate_prefix_match'] else 'FAIL'} "
            f"pass_key={'found' if found else ('MISSING' if needs_key else 'n/a')}"
        )

    prefill = [r["prefill_s"] for r in results if r["prefill_s"] is not None]
    imports = [r["import_s"] for r in results if r["import_s"] is not None]
    summary = {
        "requests_exact": sum(r["exact_match"] for r in results),
        "requests": len(results),
        "gate_ok": sum(r["gate_prefix_match"] for r in results),
        "pass_key_ok": sum(r["pass_key_found"] for r in results if r["pass_key_required"]),
        "pass_key_required": sum(r["pass_key_required"] for r in results),
        "effective_tok_s": metrics["effective_tok_s"],
        "steady_step_ms": metrics["steady_step_ms"],
        "peak_active_lanes": metrics["peak_active_lanes"],
        "mean_prefill_s": sum(prefill) / max(len(prefill), 1),
        "mean_import_s": sum(imports) / max(len(imports), 1),
        "max_pages_in_use": metrics["max_pages_in_use"],
        "num_pages": None if paging is None else paging.num_pages,
        "page_reuses": metrics["page_reuses"],
        "wall_s": metrics["wall_s"],
        "generated_tokens": metrics["generated_tokens"],
        "steps": metrics["steps"],
        "long_then_short_pairs": long_then_short,
    }
    logger.info(f"SUMMARY {json.dumps(summary)}")
    report_path = tmp_path / "continuous-batch-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "qwen38-plain-lanes-p1d/v1",
                "lanes": lanes,
                "page_size": page_size,
                "context": context,
                "pool_rows": None if paging is None else paging.pool_rows,
                "model_ready_s": model_ready_s,
                "decoder_build_s": build_s,
                "decoder_timing": decoder.timing,
                "summary": summary,
                "metrics": {k: v for k, v in metrics.items() if k != "requests"},
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    record_property("continuous_batch_report", str(report_path))
    logger.info(f"report {report_path}")

    assert long_then_short, "the request set must put a long prompt before a short one in one lane"
    if paging is not None:
        assert metrics["page_reuses"] > 0, "the tight pool must reuse pages"
    for r in results:
        assert r["gate_prefix_match"], f"request {r['request']}: first tokens differ from the B=1 reference"
        if r["pass_key_required"]:
            assert r["pass_key_found"], f"request {r['request']}: pass key not in {r['decoded'][:300]!r}"
