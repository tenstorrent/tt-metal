# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare bucketed and fixed-capacity traced serving in a single model load."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric
from models.autoports.qwen_qwen3_8_27b.tt.generator_vllm import Qwen38ForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0,3", help="Comma-separated layers, or all")
    parser.add_argument("--occupancies", default="1,5,8,9,16,1")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    adapter = None
    try:
        root = Path(__file__).resolve().parents[1]
        generator = build_generator(
            root,
            mesh,
            layer_indices=None if args.layers == "all" else [int(i) for i in args.layers.split(",")],
            precision_config=root / "doc/datatype_sweep/selected_precision_config.json",
        )
        adapter = Qwen38ForCausalLM(generator, 16, 256)
        cache = adapter.allocate_kv_cache((128, 1, 32, 256), None, len(generator.model.layers))
        table = torch.arange(128, dtype=torch.int32).reshape(16, 8)
        params = SimpleNamespace(temperature=[0.0] * 16, top_k=[1] * 16, top_p=[0.0] * 16, seed=[17] * 16)
        for n in map(int, args.occupancies.split(",")):
            control = None
            for enabled in (False, True):
                generator._release_traces()
                generator.model.decode_buckets = enabled
                adapter._decode_bound = False
                prompts = (torch.arange(128).remainder(256) + 100).repeat(n, 1)
                prompts += torch.arange(n).reshape(-1, 1) * 13
                first, _ = adapter.prefill_forward(
                    prompts,
                    table[:n],
                    cache,
                    [128] * n,
                    sampling_params=SimpleNamespace(**{key: value[:n] for key, value in vars(params).items()}),
                    empty_slots=list(range(n)),
                )
                decoded = torch.zeros(16, 1, dtype=torch.int64)
                decoded[:n] = first[:n]
                outputs, timings = [], []
                for step in range(args.steps):
                    positions = torch.full((16,), -1, dtype=torch.int32)
                    positions[:n] = 128 + step
                    begin = time.perf_counter()
                    decoded = adapter.decode_forward(
                        decoded, positions, table, cache, sampling_params=params, reset_batch=step == 0
                    )
                    ttnn.synchronize_device(mesh)
                    timings.append(time.perf_counter() - begin)
                    outputs.append(decoded[:n].reshape(-1).tolist())
                if enabled:
                    assert outputs == control, f"Token mismatch for active batch {n}: {outputs} != {control}"
                else:
                    control = outputs
                print(
                    json.dumps(
                        dict(
                            active=n,
                            bucketed=enabled,
                            layers=args.layers,
                            warm_ms=sum(timings[1:]) / len(timings[1:]) * 1000,
                            tokens=outputs,
                        )
                    ),
                    flush=True,
                )
    finally:
        if adapter is not None:
            adapter.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
