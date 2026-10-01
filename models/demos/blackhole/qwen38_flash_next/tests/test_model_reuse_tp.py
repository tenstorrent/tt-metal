# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full-checkpoint trace reuse and fresh-request state at prefill bucket boundaries.

Run with the same held 1x4 mesh/cache environment as test_moe_tp.py. This measures
fixed-length greedy decoding with EOS ignored, separately from task accuracy.
"""

import json
import os
from types import SimpleNamespace

import pytest

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.chat import Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import DEVICE_PARAMS, MAX_SEQ_LEN, generate
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import checkpoint_root
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import Qwen38ForCausalLM, prefill_form
from models.demos.blackhole.qwen38_flash_next.tools.resident_decode import program_cache_count
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.perf.benchmarking_utils import BenchmarkProfiler

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1",
    reason="requires a held four-die Blackhole line and the complete checkpoint/cache",
)


@run_for_blackhole()
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_fresh_requests_reuse_traces_at_prefill_boundaries(mesh_device, tmp_path, record_property):
    checkpoint = checkpoint_root()
    tokenizer = Qwen38OfficialChatTemplate(checkpoint).tokenizer
    sentence = tokenizer.encode("The red fox walks past the river. Explain what happens next. ")
    other = tokenizer.encode("Write a short Python function that adds two integers. ")
    mesh_device.enable_program_cache()
    model = Qwen38ForCausalLM.initialize_vllm_model(
        SimpleNamespace(_name_or_path=str(checkpoint)), mesh_device, max_batch_size=1, max_seq_len=MAX_SEQ_LEN
    )
    profiler = BenchmarkProfiler()
    report = {"prefill_slab_rows": prefill_form()[1], "eos": "ignored", "concurrency": 1, "requests": []}

    def run(label, tokens, count):
        before = program_cache_count(mesh_device)
        generated, ttft, steps = generate(model, tokens, count, profiler)
        after = program_cache_count(mesh_device)
        row = {
            "label": label,
            "prompt_token_ids": tokens,
            "generated_token_ids": generated,
            "ttft_s": ttft,
            "decode_step_s": steps,
            "program_cache_before": before,
            "program_cache_after": after,
        }
        report["requests"].append(row)
        assert after == before, f"{label}: program cache grew from {before} to {after}"
        return generated

    try:
        prompt = (sentence * (128 // len(sentence) + 1))[:128]
        first = run("fixed128-run0", prompt, 50)
        assert run("fixed128-run1", prompt, 50) == first
        for size in (31, 32, 33, 127, 128, 129, 2047, 2048, 2049):
            prompt = (sentence * (size // len(sentence) + 1))[:size]
            first = run(f"boundary{size}-A0", prompt, 4)
            run(f"boundary{size}-B", other, 4)
            assert run(f"boundary{size}-A1", prompt, 4) == first, f"state leaked across boundary{size} A/B/A"
        report["pass"] = True
    finally:
        report_path = tmp_path / "model-reuse.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        record_property("model_reuse_report", str(report_path))
        model.release_persistent_capture()
