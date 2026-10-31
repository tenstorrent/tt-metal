"""Compare buffered token history to streaming on the real two-layer path."""

import argparse
import json
import os
from pathlib import Path

import torch

import ttnn

from ..tt.generator import Gemma4Generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--head-mode", default="dram", choices=["dram", "interleaved"])
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = Path("models/autoports/google_gemma_4_31b_it/doc/optimized_full_model")
    prompt = json.loads((root.parent / "full_model/aime24_prompt.json").read_text())["tokens"]
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    try:
        gen = Gemma4Generator(mesh, layer_indices=[0, 5], batch_size=args.batch, head_mode=args.head_mode)
        gen.prepare_decode()
        cases = []
        for length, sampled in ((161, False), (129, True), (161, False)):
            prompts = [prompt[:length] if slot % 3 != 2 else None for slot in range(args.batch)]
            params = dict(temperature=0.7, top_k=16, top_p=0.9, seed=123) if sampled else None
            expected = gen.generate_batch(prompts, args.tokens, sampling_params=params, ignore_eos=True, buffered=False)
            before = dict(gen.counters)
            actual = gen.generate_buffered(prompts, args.tokens, sampling_params=params)
            assert expected == actual, (expected, actual)
            counts = {k: gen.counters[k] - v for k, v in before.items()}
            assert counts["readbacks"] == 1 and counts["token_refreshes"] == 0
            assert counts["captures"] == 0 and counts["position_refreshes"] == 1
            cases.append(
                dict(
                    length=length,
                    sampled=sampled,
                    exact=True,
                    counters=counts,
                    traces=[str(t) for t in (gen.model_trace, gen.sample_trace, gen.buffered_sample_trace)],
                )
            )
        (root / f"buffered_probe_b{args.batch}.json").write_text(
            json.dumps(
                dict(
                    scope="reduced_layers_0_5",
                    batch=args.batch,
                    generated_tokens=args.tokens,
                    head_mode=args.head_mode,
                    diagnostics={
                        k: os.environ.get(k, "0")
                        for k in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE")
                    },
                    cases=cases,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
