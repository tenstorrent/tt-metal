"""Uninstrumented all-layer performance, long-context execution and paired text."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import ttnn

from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator
from models.autoports.qwen_qwen3_8_flash_next.tests.full_model_qualitative import run_tt
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_capacity_probe import view
from readiness_check.schema import load_reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default="full_model_final")
    ap.add_argument("--long-context", action="store_true")
    ap.add_argument("--wait-qualitative", action="store_true")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--prefill-trace-max", type=int, default=None)
    ap.add_argument("--trace-region-size", type=int, default=768000000)
    a = ap.parse_args()
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    root = Path("models/autoports/qwen_qwen3_8_flash_next")
    output = Path(a.output_dir) if a.output_dir else root / "doc/full_model"
    output.mkdir(parents=True, exist_ok=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=a.trace_region_size)
    gen = None
    result = dict(args=vars(a), layers=48, batch=1, context=262144, throw_exception_on_fallback=True)

    def save():
        (output / f"{a.tag}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")

    try:
        tick = time.perf_counter()
        gen = build_generator(root, mesh, checkpoint=a.checkpoint, prefill_trace_max=a.prefill_trace_max)
        gen.prepare()
        result["setup_seconds"] = time.perf_counter() - tick
        result["prepared_memory"] = view(mesh)
        result["trace_memory"] = view(mesh, ttnn.BufferType.TRACE)
        ids = dict(gen.traces)
        result["trace_ids"] = {k: str(v) for k, v in ids.items()}
        reset_times = []
        for _ in range(3):
            ttnn.synchronize_device(mesh)
            tick = time.perf_counter()
            gen.reset()
            ttnn.synchronize_device(mesh)
            reset_times.append((time.perf_counter() - tick) * 1000)
        result["reset_host_ms"] = reset_times
        entry = load_reference(root / "readiness_aime24_chat.refpt").entries[0]
        primary = (entry.prompt_tokens[0].tolist() * 2)[:128]
        gen.generate(primary, 128)
        records = []
        for _ in range(3):
            before = dict(gen.counters)
            tokens = gen.generate(primary, 128)
            records.append(
                dict(
                    perf=dict(gen.last_perf),
                    tokens=tokens,
                    counter_delta={k: gen.counters[k] - before.get(k, 0) for k in gen.counters},
                )
            )
        assert records[0]["tokens"] == records[1]["tokens"] == records[2]["tokens"]
        result["primary_128_128"] = records
        result["traced_logits_only"] = []
        for _ in range(3):
            gen.generate(primary, 1)
            ttnn.synchronize_device(mesh)
            tick = time.perf_counter()
            for _ in range(127):
                gen._replay("decode")
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - tick
            result["traced_logits_only"].append(
                dict(
                    seconds=elapsed,
                    tokens=127,
                    tps=127 / elapsed,
                    boundary="model to logits, fixed token input and device position advance; no sampling",
                )
            )
        save()
        if a.long_context:
            length = gen.model.context - 3
            tick = time.perf_counter()
            tokens = gen.generate([100 + i % 300 for i in range(length)], 4)
            position = ttnn.to_torch(ttnn.get_device_tensors(gen.tt_pos)[0])[:, 0].tolist()
            assert len(tokens) == 4 and position == [gen.model.context]
            assert ids == gen.traces and gen.counters["captures"] == len(ids)
            result["long_context"] = dict(
                prompt_len=length,
                tokens=tokens,
                final_position=position,
                seconds=time.perf_counter() - tick,
                perf=dict(gen.last_perf),
                scope="Full-stack execution/capacity with synthetic valid token IDs; no long-context language-quality claim",
            )
            print("FULL_CONTEXT_PASS", json.dumps(result["long_context"]), flush=True)
            save()
        controls = root / "doc/full_model/qualitative_hf.json"
        while not controls.exists() or len(json.loads(controls.read_text())["prompts"]) < 6:
            if not a.wait_qualitative:
                break
            print("WAITING_FOR_HF_QUALITATIVE", time.strftime("%H:%M:%S"), flush=True)
            time.sleep(30)
        if controls.exists() and len(json.loads(controls.read_text())["prompts"]) == 6:
            result["qualitative"] = run_tt(gen, root, output_dir=output)
        assert ids == gen.traces and gen.counters["captures"] == len(ids) and gen.counters["releases"] == 0
        result["final_memory"] = view(mesh)
        result["counters"] = dict(gen.counters)
        result["source_sha256"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "tt").glob("*.py")
        }
        save()
        print("FULL_MODEL_FINAL_PASS", flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
