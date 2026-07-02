# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_4b.tt.generator import build_generator
from models.autoports.qwen_qwen3_4b.tt.model import Qwen3FullModelConfig
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device
from models.common.readiness_check.schema import load_reference

MODEL_DIR = Path("models/autoports/qwen_qwen3_4b")
REFERENCE_PATH = MODEL_DIR / "doc/full_model/readiness_aime24_chat.refpt"


def _write_reduced_perf_outputs(result: dict, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    output_base.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = output_base.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ttft_ms",
                "decode_t/s/u",
                "prepare_decode_ms",
                "steady_decode_t/s/u",
                "e2e_t/s/u",
                "decode_tokens",
                "steady_decode_tokens",
                "trace_replays",
                "token_host_refreshes",
                "position_host_refreshes",
                "page_table_host_refreshes",
                "syncs",
                "readbacks",
            ],
        )
        writer.writeheader()
        counters = result["trace_counters"]
        writer.writerow(
            {
                "ttft_ms": result["ttft_ms"],
                "decode_t/s/u": result["decode_t/s/u"],
                "prepare_decode_ms": result.get("prepare_decode_ms"),
                "steady_decode_t/s/u": result.get("steady_decode_t/s/u"),
                "e2e_t/s/u": result["e2e_t/s/u"],
                "decode_tokens": result["decode_tokens"],
                "steady_decode_tokens": result.get("steady_decode_tokens"),
                "trace_replays": counters["trace_replays"],
                "token_host_refreshes": counters["token_host_refreshes"],
                "position_host_refreshes": counters["position_host_refreshes"],
                "page_table_host_refreshes": counters["page_table_host_refreshes"],
                "syncs": counters["syncs"],
                "readbacks": counters["readbacks"],
            }
        )


def _benchmark_reduced_token_out_with_steady_signpost(generator, prompt_token_ids: list[int], max_new_tokens: int):
    if max_new_tokens < 2:
        raise ValueError("reduced token-out profiler requires at least two generated tokens")
    generator.reset()
    prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
    start_s = time.perf_counter()
    logits = generator.prefill_forward(
        prompt,
        page_table=generator.page_table,
        kv_cache=generator.kv_cache,
        prompt_lens=[len(prompt_token_ids)],
        return_all_logits=False,
    )
    first_input_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
    first_s = time.perf_counter()
    generator.prepare_token_out_decode(
        first_input_token=first_input_token,
        start_pos=len(prompt_token_ids),
        prompt_len=len(prompt_token_ids),
        read_first_token=False,
    )
    steady_start_s = time.perf_counter()
    signpost("PERF_OPT_FULL_TOKEN_OUT_STEADY")
    for _ in range(2, max_new_tokens):
        generator.decode_next_token_on_device()
    ttnn.synchronize_device(generator.mesh_device)
    signpost("PERF_OPT_FULL_TOKEN_OUT_STEADY_END")
    end_s = time.perf_counter()
    decode_tokens = max_new_tokens - 1
    steady_tokens = max_new_tokens - 2
    return {
        "ttft_ms": (first_s - start_s) * 1000.0,
        "decode_t/s/u": decode_tokens / max(end_s - first_s, 1.0e-9),
        "prepare_decode_ms": (steady_start_s - first_s) * 1000.0,
        "steady_decode_t/s/u": steady_tokens / max(end_s - steady_start_s, 1.0e-9) if steady_tokens else 0.0,
        "steady_decode_tokens": steady_tokens,
        "e2e_t/s/u": max_new_tokens / max(end_s - start_s, 1.0e-9),
        "decode_tokens": decode_tokens,
        "trace_counters": dict(generator.model.trace_state.counters),
    }


@pytest.mark.timeout(600)
def test_reduced_full_model_token_out_perf_signposts():
    if os.environ.get("QWEN3_4B_FULL_MODEL_RUN_REDUCED_PERF") != "1":
        pytest.skip("set QWEN3_4B_FULL_MODEL_RUN_REDUCED_PERF=1 to run the reduced full-model profiler path")

    max_new_tokens = int(os.environ.get("QWEN3_4B_FULL_MODEL_REDUCED_NEW_TOKENS", "8"))
    output_base = Path(
        os.environ.get(
            "QWEN3_4B_FULL_MODEL_REDUCED_PERF_OUT",
            MODEL_DIR / "doc/optimized_full_model/reduced_full_model_token_out_perf",
        )
    )
    reference = load_reference(REFERENCE_PATH)
    prompt_token_ids = reference.entries[0].prompt_tokens.reshape(-1).tolist()
    mesh = open_readiness_mesh_device("P150_X4", "FABRIC_1D_RING")
    generator = None
    try:
        generator = build_generator(
            MODEL_DIR,
            mesh,
            model_config=Qwen3FullModelConfig(num_layers=1),
        )
        signpost("PERF_OPT_FULL_TOKEN_OUT")
        result = _benchmark_reduced_token_out_with_steady_signpost(generator, prompt_token_ids, max_new_tokens)
        signpost("PERF_OPT_FULL_TOKEN_OUT_END")
        _write_reduced_perf_outputs(result, output_base)
        assert result["trace_counters"]["readbacks"] == 0
        assert result["trace_counters"]["token_host_refreshes"] == 0
        assert result["trace_counters"]["page_table_host_refreshes"] == 0
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh, "FABRIC_1D_RING")
