# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Order-reversed full-block timing check on shared, real D inputs.

Runs separately after image generation. No changes to the frozen image suite.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

from block_bench import BlockBench
from model_attention import FrontierAttention
from test_pipeline import device_params, test_stock_pipeline as run_pipeline


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["bh_lb"], indirect=True)
@pytest.mark.timeout(3600)
def test_order_reversed_blocks(mesh_device, model_location_generator, monkeypatch):
    import ttnn

    output = Path(os.environ["FLUX2_SWEEP_REPORT"])
    if output.exists():
        raise ValueError("Use a fresh sweep report")
    for key, value in dict(
        FLUX2_VARIANT="D",
        FLUX2_MODEL_REPAIR="main_fused",
        FLUX2_CONDITIONING="stock",
        FLUX2_BLOCK_BENCH="1",
        FLUX2_TRACED="0",
        FLUX2_STEPS="2",
        FLUX2_PROMPTS="1",
        FLUX2_SEEDS="1",
    ).items():
        monkeypatch.setenv(key, value)
    original_install = BlockBench.install
    original_benchmark = BlockBench.benchmark
    state = {}
    report = dict(
        status="running",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_inputs="D first denoising step, two-step prompt 0 seed 0 qualification",
        initial_dual0_warmup_replays=5000,
        per_block_warmup_replays=750,
        timed_replays=50,
        rows=[],
    )

    def install(bench, transformer):
        state["transformer"] = transformer
        original_install(bench, transformer)

    def benchmark(bench, device, **kwargs):
        transformer = state["transformer"]
        blocks = list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks)
        saved = [block.attn._attention_override for block in blocks]
        # Sustained device work before the first reported measurement. The
        # same captured inputs are reused across all numerical variants.
        original, args, call_kwargs = bench.captures["dual.0"]
        warmup_output = original(*args, **call_kwargs)
        trace = ttnn.begin_trace_capture(device, cq_id=0)
        warmup_output = original(*args, **call_kwargs)
        ttnn.end_trace_capture(device, trace, cq_id=0)
        try:
            for _ in range(report["initial_dual0_warmup_replays"]):
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
        finally:
            ttnn.release_trace(device, trace)
        del warmup_output
        forward = ("D", "stock", "C", "B", "A", "E", "F", "G")
        d_records = None
        try:
            for round_index, order in enumerate((forward, tuple(reversed(forward)))):
                for variant in order:
                    if variant == "stock":
                        for block in blocks:
                            if hasattr(block.attn, "_attention_override"):
                                del block.attn._attention_override
                    else:
                        FrontierAttention(variant).install(transformer)
                    # Diagnostic sweep records mismatches instead of concealing
                    # later choices behind the first failure. Image qualification
                    # keeps BlockBench's strict=True default unchanged.
                    records = original_benchmark(bench, device, warmup=750, iterations=50, strict=False)
                    if variant == "D":
                        d_records = records
                    report["rows"].extend(dict(round=round_index, variant=variant, **r) for r in records)
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    print("BLOCK_SWEEP_DONE", round_index, variant, flush=True)
        finally:
            for block, override in zip(blocks, saved, strict=True):
                block.attn._attention_override = override
        assert len(report["rows"]) == 96
        report["status"] = "completed"
        output.write_text(json.dumps(report, indent=2) + "\n")
        return d_records

    monkeypatch.setattr(BlockBench, "install", install)
    monkeypatch.setattr(BlockBench, "benchmark", benchmark)
    run_pipeline(mesh_device, model_location_generator, monkeypatch)
