# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-shape terminal/trace probe; never full-model performance evidence."""
import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--prompt-length", type=int)
    parser.add_argument("--reserve-semaphores", action="store_true")
    parser.add_argument("--reserve-routers", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    start = time.perf_counter()
    try:
        gen = build_generator(None, mesh, max_seq_len=1024, layer_indices=None if args.all_layers else (0, 5))
        if args.reserve_semaphores:
            from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import _MeshCCLManager

            reservations = [_MeshCCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear) for _ in range(28)]
        if args.reserve_routers:
            router = gen.model.layers[0].layer.moe.router
            router_reservations = [
                ttnn.empty(
                    getattr(router, name).shape,
                    dtype=getattr(router, name).dtype,
                    layout=getattr(router, name).layout,
                    device=mesh,
                    memory_config=router.memory,
                )
                for _ in range(28)
                for name in ("bias", "indices", "output", "output_indices")
            ]
        prompt = gen.tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is two plus two?"}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        if args.prompt_length:
            prompt = (prompt * ((args.prompt_length + len(prompt) - 1) // len(prompt)))[: args.prompt_length]
        tokens = gen.generate(prompt, args.steps, stop_on_eos=False)
        report = dict(
            tokens=tokens,
            completion=gen.tokenizer.decode(tokens),
            prompt_tokens=len(prompt),
            elapsed_seconds=time.perf_counter() - start,
            metrics=gen.metrics,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
