# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare standalone KV-clear elision with canonical physical pool reset."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import build_generator

ROOT = Path(__file__).parents[1]


def check_cache_reset_equivalence(generator):
    original = generator.clear_cache_on_generate
    rows = []
    try:
        # The long-to-short transition leaves old physical pages and a partial
        # page behind. New prompt token values also differ at every boundary.
        for length, token, top_k in ((4097, 42, 1), (33, 17, 1), (129, 91, 32), (1, 43, 1), (127, 71, 1)):
            outputs, timings, counters = {}, {}, {}
            for clear in (False, True):
                generator.clear_cache_on_generate = clear
                before = dict(generator.counters)
                outputs[str(clear)] = generator.generate(
                    [token] * length,
                    8,
                    top_k=top_k,
                    top_p=0.9 if top_k > 1 else 0.0,
                    seeds=1234,
                    stop_on_eos=False,
                )
                timings[str(clear)] = generator.last_ttft_seconds * 1000
                expected_positions = [length + 7] + [-1] * (generator.batch_size - 1)
                assert generator._host_positions == expected_positions
                device_positions = ttnn.to_torch(ttnn.get_device_tensors(generator.positions)[0]).flatten().tolist()
                assert device_positions == expected_positions
                counters[str(clear)] = {k: v - before.get(k, 0) for k, v in generator.counters.items()}
            assert outputs["False"] == outputs["True"], (length, outputs)
            assert counters["False"].get("cache_clears", 0) == 0
            assert counters["False"]["cache_clear_elisions"] == 1
            assert counters["True"]["cache_clears"] == 1
            rows.append(
                dict(
                    length=length,
                    prompt_token=token,
                    top_k=top_k,
                    seeds=1234,
                    tokens=outputs,
                    ttft_ms=timings,
                    counters=counters,
                    exact_match=True,
                )
            )
    finally:
        generator.clear_cache_on_generate = original
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    generator = None
    try:
        generator = build_generator(
            ROOT, mesh, batch_size=args.batch, **({} if args.full else {"layer_indices": [0, 3]})
        )
        rows = check_cache_reset_equivalence(generator)
        args.output.write_text(
            json.dumps(dict(batch=args.batch, layers=52 if args.full else 2, cases=rows), indent=2) + "\n"
        )
        print("CACHE_REUSE_PASS", flush=True)
    finally:
        if generator:
            generator.close()
        ttnn.close_mesh_device(mesh)
