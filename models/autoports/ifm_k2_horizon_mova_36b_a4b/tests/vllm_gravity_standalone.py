# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact-prompt standalone control for the mixed-traffic sampled anomaly."""

import json
import time
from pathlib import Path

import torch

import ttnn
from models.common.sampling.generator import SamplingParams

from ..tt.generator import build_generator
from ..tt.precision import load_precision


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "readiness_vllm"
    controls = json.loads((evidence / "gravity_server_controls.json").read_text())
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    generator = None
    rows = []
    try:
        generator = build_generator(
            root,
            mesh,
            max_seq_len=524288,
            max_batch_size=32,
            page_size=64,
            precision_config=load_precision(root / "doc/datatype_sweep/selected_precision_config.json"),
        )
        g = generator
        g.capture()
        # Match the serving B32 execution geometry without plugin orchestration.
        perf_prompt = json.loads((root / "doc/full_model/reference_metadata.json").read_text())["prompt_tokens"][:128]
        g.reset()
        first = g.prefill_forward(
            torch.tensor([perf_prompt]),
            page_table=g.page_table_host[:1],
            kv_cache=g.kv_cache,
            prompt_lens=[128],
            sampling_params=SamplingParams(0.0, 1, 1.0),
        )
        tokens = torch.zeros(32, 1, dtype=torch.int64)
        tokens[0] = first[0]
        positions = torch.full((32,), -1, dtype=torch.int32)
        positions[0] = 128
        g.decode_forward(
            tokens,
            positions,
            page_table=g.page_table_host,
            kv_cache=g.kv_cache,
            sampling_params=SamplingParams(0.0, 1, 1.0),
            reload_inputs=True,
            reload_sampling_params=True,
            reset_sampling_state=True,
        )
        ttnn.synchronize_device(mesh)
        before = g.counters.copy()
        started = time.monotonic()
        for _ in range(126):
            g.replay()
        ttnn.synchronize_device(mesh)
        elapsed = time.monotonic() - started
        perf = dict(
            prompt_tokens=128,
            output_tokens=128,
            active_requests=1,
            max_batch_size=32,
            measured_decode_steps=126,
            excluded_first_decode_steps=1,
            decode_ms=1000 * elapsed / 126,
            decode_tps=126 / elapsed,
            counters={k: g.counters[k] - before[k] for k in g.counters},
            method="Queued canonical split token-out; two boundary synchronizations; no per-token read",
        )
        (evidence / "standalone_b32_performance.json").write_text(json.dumps(perf, indent=2))
        print("B32_PERFORMANCE", perf, flush=True)
        for run in controls["runs"]:
            req = run["request"]
            prompt = torch.tensor([req["prompt"]])
            params = SamplingParams(
                temperature=[req["temperature"]] + [0.0] * 31,
                top_k=[req["top_k"]] + [1] * 31,
                top_p=1.0,
                presence_penalty=[req["presence_penalty"]] + [0.0] * 31,
                seed=[req["seed"]] + [None] * 31,
            )
            g.reset()
            started = time.monotonic()
            first = g.prefill_forward(
                prompt,
                page_table=g.page_table_host[:1],
                kv_cache=g.kv_cache,
                prompt_lens=[prompt.shape[1]],
                sampling_params=params,
            )
            result = [int(first[0, 0])]
            input_tokens = torch.zeros(32, 1, dtype=torch.int64)
            input_tokens[0] = first[0]
            prompt_history = torch.full((32, prompt.shape[1]), -1, dtype=torch.int64)
            prompt_history[0] = prompt[0]
            output_history = torch.full((32, 1), -1, dtype=torch.int64)
            output_history[0] = first[0]
            for step in range(1, 64):
                positions = torch.full((32,), -1, dtype=torch.int32)
                positions[0] = prompt.shape[1] + step - 1
                out = g.decode_forward(
                    input_tokens,
                    positions,
                    page_table=g.page_table_host,
                    kv_cache=g.kv_cache,
                    sampling_params=params,
                    reload_inputs=step == 1,
                    reload_sampling_params=step == 1,
                    reset_sampling_state=step == 1,
                    prompt_tokens=prompt_history,
                    output_tokens=output_history,
                )
                result.append(int(out[0, 0]))
            expected = run["response"]["choices"][0]["token_ids"]
            row = dict(
                temperature=req["temperature"],
                seed=req["seed"],
                prompt_token_ids=req["prompt"],
                token_ids=result,
                text=g.tokenizer.decode(result, skip_special_tokens=False),
                matches_serving=result == expected,
                seconds=time.monotonic() - started,
            )
            rows.append(row)
            (evidence / "gravity_standalone_control.json").write_text(
                json.dumps(
                    dict(
                        status="collected; quality inspection required",
                        comparison="Same canonical device sampler, fixed seeds, matched B32 decode geometry; standalone owns cache and has no adapter/plugin",
                        max_batch_size=32,
                        layers=48,
                        context=524288,
                        policy=g.model.precision,
                        page_size=64,
                        cache_owner="standalone generator",
                        prompt_format="Exact chat-rendered IDs from mixed_shapes gravity row",
                        runs=rows,
                    ),
                    indent=2,
                )
            )
            print(row["seed"], row["matches_serving"], repr(row["text"]), flush=True)
    finally:
        if generator is not None:
            generator.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
