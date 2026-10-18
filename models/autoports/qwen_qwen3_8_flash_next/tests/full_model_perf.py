"""Reduced real model profiling and split terminal timing (never all-layer Tracy)."""

import argparse
import json
import time
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default="reduced_perf")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--stage", default="full_model")
    ap.add_argument("--prefill-trace-max", type=int, default=512)
    ap.add_argument("--trace-region-size", type=int, default=768000000)
    a = ap.parse_args()
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=a.trace_region_size)
    gen = None
    try:
        gen = build_generator(
            "models/autoports/qwen_qwen3_8_flash_next",
            mesh,
            checkpoint=a.checkpoint,
            layer_indices=[0, 1, 3],
            prefill_trace_max=a.prefill_trace_max,
        )
        if a.profile:
            from tracy import signpost

            original = gen._prefill_chunk

            def draining_warmup(n, **kwargs):
                out = original(n, **kwargs)
                if not gen.prepared:
                    ttnn.synchronize_device(mesh)
                    ttnn.ReadDeviceProfiler(mesh)
                return out

            gen._prefill_chunk = draining_warmup
        else:
            signpost = lambda _: None
        gen.prepare()
        if a.profile:
            gen._prefill_chunk = original
            ttnn.ReadDeviceProfiler(mesh)
        rendered = gen.tokenizer.apply_chat_template(
            [dict(role="user", content="Explain how a rainbow forms.")], tokenize=False, add_generation_prompt=True
        )
        base = gen.tokenizer.encode(rendered, add_special_tokens=False)
        prompt = (base * (128 // len(base) + 1))[:128]
        gen.reset()
        signpost("PERF_PREFILL")
        gen.prefill_forward(
            torch.tensor([prompt]),
            page_table=gen.page_table,
            kv_cache=gen.kv_cache,
            prompt_lens=[128],
            read_tokens=False,
        )
        ttnn.synchronize_device(mesh)
        signpost("PERF_PREFILL_END")
        if a.profile:
            ttnn.ReadDeviceProfiler(mesh)
        report = dict(layers=[0, 1, 3], context=262144, prompt_len=128, profile=a.profile, trace_ms={})
        # Compare both common semantically greedy choices on identical logits.
        gen._replay("greedy")
        greedy = gen._read_tokens()
        gen._replay("argmax")
        argmax = gen._read_tokens()
        assert greedy == argmax, (greedy, argmax)
        report["greedy_argmax_equal"] = True
        if a.profile:
            ttnn.ReadDeviceProfiler(mesh)
        for mode in ("decode", "greedy", "argmax", "sample", "token_out"):
            reps = 1 if a.profile else 32
            samples = []
            report.setdefault("window_positions", {})[mode] = ttnn.to_torch(ttnn.get_device_tensors(gen.tt_pos)[0])[
                :, 0
            ].tolist()
            for _ in range(1 if a.profile else 3):
                ttnn.synchronize_device(mesh)
                signpost("PERF_" + mode.upper())
                tick = time.perf_counter()
                for _ in range(reps):
                    if mode == "token_out":
                        gen._replay("decode")
                        gen._replay("greedy")
                    else:
                        gen._replay(mode)
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - tick) * 1000 / reps)
                signpost("PERF_" + mode.upper() + "_END")
                if a.profile:
                    ttnn.ReadDeviceProfiler(mesh)
            report["trace_ms"][mode] = samples
        if not a.profile:
            gen.generate(prompt, 128)
            report["token_out"] = dict(gen.last_perf)
        report["counters"] = dict(gen.counters)
        Path(f"models/autoports/qwen_qwen3_8_flash_next/doc/{a.stage}/{a.tag}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print("REDUCED_PERF_PASS", json.dumps(report), flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
