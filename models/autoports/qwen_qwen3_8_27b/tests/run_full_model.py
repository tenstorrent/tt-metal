# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced or full real-weight probe; hardware runs must be serialized."""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator

p = argparse.ArgumentParser()
p.add_argument("--full", action="store_true")
p.add_argument("--sampling-strategy", choices=["split", "argmax"], default="split")
p.add_argument("--profile", action="store_true")
p.add_argument("--head-strategy", choices=["interleaved", "dram"], default="dram")
p.add_argument("--compare-head", action="store_true")
p.add_argument("--length", type=int, default=33)
p.add_argument("--generate", type=int, default=4)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.profile and a.full:
    p.error("Full-stack profiling is prohibited; use the reduced stack")
torch.set_num_threads(8)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
gen = None
try:
    begin = time.perf_counter()
    gen = build_generator(
        "models/autoports/qwen_qwen3_8_27b",
        mesh,
        layer_indices=None if a.full else [0, 3],
        sampling_strategy=a.sampling_strategy,
        head_strategy=a.head_strategy,
    )
    print("SETUP_SECONDS", time.perf_counter() - begin, flush=True)
    if a.profile:
        ttnn.ReadDeviceProfiler(mesh)
    prompt = gen.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain why the sky is blue."}], tokenize=False, add_generation_prompt=True
    )
    prompt = gen.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prompt = (prompt * ((a.length + len(prompt) - 1) // len(prompt)))[: a.length]
    print("PREFILL_BEGIN", flush=True)
    out = gen.generate(prompt, a.generate)
    print("GENERATED", out, gen.tokenizer.decode(out), flush=True)
    if a.profile:
        ttnn.ReadDeviceProfiler(mesh)
    print("SECOND_RUN", flush=True)
    again = gen.generate(prompt, a.generate)
    assert out == again, (out, again)
    benchmark = gen.last_perf
    comparison = None
    if a.compare_head:
        assert a.head_strategy == "dram"
        candidate = gen._host_logits(gen.logits)[0, 0, 0]
        gen._release_traces()
        gen.model.head_strategy = "interleaved"
        control_tokens = gen.generate(prompt, a.generate)
        control = gen._host_logits(gen.logits)[0, 0, 0]
        correlation = torch.corrcoef(torch.stack([candidate, control]))[0, 1].item()
        assert out == control_tokens and correlation >= 0.999, correlation
        comparison = dict(logit_pcc=correlation, greedy_equal=True)
    if a.profile:
        ttnn.ReadDeviceProfiler(mesh)
    if a.profile:
        import tracy

        tracy.signpost("PERF_PREFILL")
        gen.generate(prompt, 1)
        ttnn.synchronize_device(mesh)
        tracy.signpost("PERF_PREFILL_END")
        ttnn.ReadDeviceProfiler(mesh)
        tracy.signpost("PERF_MODEL")
        ttnn.execute_trace(mesh, gen.trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        tracy.signpost("PERF_MODEL_END")
        ttnn.ReadDeviceProfiler(mesh)
        tracy.signpost("PERF_SAMPLE")
        ttnn.execute_trace(mesh, gen.sample_trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        tracy.signpost("PERF_SAMPLE_END")
        ttnn.ReadDeviceProfiler(mesh)
        tracy.signpost("PERF_TOKEN_OUT")
        gen.decode_forward(page_table=gen.page_table, kv_cache=gen.cache)
        tracy.signpost("PERF_TOKEN_OUT_END")
        ttnn.ReadDeviceProfiler(mesh)
    report = dict(
        sampling_strategy=a.sampling_strategy,
        full=a.full,
        prompt=prompt,
        output=out,
        text=gen.tokenizer.decode(out),
        repeat_equal=True,
        perf=benchmark,
        head_strategy=a.head_strategy,
        head_comparison=comparison,
    )
    a.output.write_text(json.dumps(report, indent=2) + "\n")
finally:
    if gen is not None:
        gen.close()
    ttnn.close_mesh_device(mesh)
