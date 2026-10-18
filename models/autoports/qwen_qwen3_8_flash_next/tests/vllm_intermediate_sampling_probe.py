"""Prove discarded intermediate samples cannot perturb canonical final sampling.

Reduced real layers, identical physical prefill chunks, fixed request seeds.
The no-intermediate-sampling control suppresses only the unused host readback.
"""

import argparse
import json
import os
import traceback
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
if os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] != "1":
    raise RuntimeError("This proof requires allocation tracking")
if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
    raise RuntimeError("Program-cache allocations must remain tracked")

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.vllm_adapter_device_probe import (
    BATCH,
    BLOCK,
    CONTEXT,
    sampling_params,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM


def run(adapter, mode, intermediate, seed_offset):
    g = adapter.generator
    g.reset()
    adapter.slot_map = list(range(BATCH))
    pages = torch.zeros(BATCH, CONTEXT // BLOCK, dtype=torch.int32)
    for i in range(BATCH):
        pages[i, :8] = torch.arange(8) + i * 8
    prompts = torch.tensor(
        [
            g.tokenizer.encode(text * 40, add_special_tokens=False)[:131]
            for text in (
                "Leaves absorb sunlight and grow. ",
                "Ships navigate by stars at night. ",
                "Prime numbers have two positive factors. ",
            )
        ],
        dtype=torch.int32,
    )
    params = sampling_params(mode, list(range(BATCH)))
    params = replace(params, seed=[s + seed_offset for s in params.seed])
    before = dict(g.counters)
    original_read = g._read_logits
    # The control has no intermediate sampling. Returning a sentinel avoids
    # moving unused logits to the host; no request state or logits are changed.
    g._read_logits = lambda tensor: torch.zeros(BATCH, 1, 1)
    try:
        start = 0
        for end in (31, 64, 131):
            final = end == 131
            result, _ = adapter.prefill_forward(
                prompts,
                pages,
                adapter.kv_cache,
                [end] * BATCH,
                empty_slots=list(range(BATCH)),
                start_pos=[start] * BATCH,
                sampling_params=params if intermediate or final else None,
            )
            start = end
    finally:
        g._read_logits = original_read
    outputs = [result.reshape(BATCH).tolist()]
    for step in range(8):
        output = adapter.decode_forward(
            result.reshape(BATCH, 1),
            [131 + step] * BATCH,
            pages,
            adapter.kv_cache,
            sampling_params=params,
            reload_inputs=step == 0,
            reload_sampling_params=step == 0,
            reset_sampling_state=step == 0,
        )
        outputs.append(output.reshape(BATCH).tolist())
    delta = {k: g.counters[k] - before.get(k, 0) for k in g.counters}
    assert delta.get("logits_readbacks", 0) == 0
    return dict(
        mode=mode,
        intermediate_sampling=intermediate,
        seeds=params.seed,
        chunk_ends=[31, 64, 131],
        outputs=outputs,
        counter_delta=delta,
    )


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["QWEN_FLASH_SERVING_LAYERS"] = "0,1,3"
    os.environ["QWEN_FLASH_CHECKPOINT"] = args.checkpoint
    os.environ["QWEN_FLASH_ALLOW_HOST_SAMPLING"] = "1"
    os.environ["QWEN_FLASH_TRACE_EVENTS"] = str(args.output.with_suffix(".events.jsonl"))
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    report = dict(status="running", scope="Reduced real layers0,1,3; sampler state proof, not quality", phases=[])
    mesh, adapter = None, None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=160000000)
        cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
        adapter = QwenFlashNextForCausalLM.initialize_vllm_model(cfg, mesh, BATCH, CONTEXT)
        cache = adapter.allocate_kv_cache((CONTEXT // BLOCK + BATCH, 1, BLOCK, 256), torch.float8_e4m3fn, 1)
        adapter.warmup_model_prefill(cache, enable_trace=True, can_sample_on_device=True)
        baseline = dict(adapter.generator.traces)
        for mode, offset in (("greedy", 0), ("sample", 0), ("sample", 1000)):
            control = run(adapter, mode, False, offset)
            candidate = run(adapter, mode, True, offset)
            report["phases"].extend([control, candidate])
            assert control["outputs"] == candidate["outputs"], (mode, offset)
            assert adapter.generator.traces == baseline
        report.update(status="pass", trace_ids=baseline, counters=dict(adapter.generator.counters))
    except BaseException:
        report.update(status="failed", failure=traceback.format_exc())
        raise
    finally:
        if adapter is not None and adapter.generator is not None:
            adapter.release_persistent_capture()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        report["closed_mesh"] = mesh is not None
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("INTERMEDIATE_SAMPLING_PASS", args.output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args())
