# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Small wrapper/terminal/split-trace probe; no full-model quality claim."""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_27b.tt.model import Model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--all-layers", action="store_true")
    p.add_argument("--host-compatibility", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    start = time.perf_counter()
    try:
        model = Model(mesh, layer_indices=None if args.all_layers else [0, 3])
        gen = QwenGenerator(model, max_seq_len=1024)
        prompt = list(range(100, 133))
        if args.host_compatibility:
            diagnostic_lengths = sorted({len(prompt)} | ({gen.prefill_trace.length} if gen.prefill_trace else set()))
            gen.prefill_single.prepare(_diagnostic_lengths=diagnostic_lengths)
        gen.prefill_logits(prompt)
        output = gen.generate(prompt, 5)
        compatibility = None
        if args.host_compatibility:
            traces = dict(gen.traces)
            before = dict(gen.counters)
            host_output = gen.generate(
                prompt,
                5,
                host_sampling=True,
                temperature=0.7,
                top_k=50,
                host_sampler=lambda logits: int(logits.argmax()),
            )
            assert host_output == output
            assert gen.counters["full_logits_readbacks"] - before.get("full_logits_readbacks", 0) == 5
            assert gen.counters["sample_replays"] == before.get("sample_replays", 0)
            reads = gen.counters["full_logits_readbacks"]
            returned = gen.generate(prompt, 5)
            assert returned == output and gen.traces == traces
            assert gen.counters["full_logits_readbacks"] == reads
            compatibility = dict(
                passed=True,
                host_output=host_output,
                returned_device_output=returned,
                callback="Explicit greedy callback; device-ineligible top_k50 must not reject it",
                diagnostic_preparation_lengths=diagnostic_lengths,
            )
        result = dict(
            layers=model.layer_indices,
            output=output,
            counters=dict(gen.counters),
            timings=gen.last_timings,
            elapsed_seconds=time.perf_counter() - start,
            positions=ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).tolist(),
            host_compatibility=compatibility,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
