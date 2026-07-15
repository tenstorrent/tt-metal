# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Two-kind traced construction smoke before expensive full-model datatype runs."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from models.autoports.google_gemma_4_31b.tt.generator import build_generator
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device
from models.common.readiness_check.schema import load_reference


def run(args: argparse.Namespace) -> None:
    reference = load_reference(args.reference.resolve())
    prompt = reference.entries[0].prompt_tokens.reshape(-1).tolist()
    forced = reference.entries[0].generated_tokens.reshape(-1).tolist()
    mesh = open_readiness_mesh_device("P150_X4", "FABRIC_1D")
    generator = None
    try:
        generator = build_generator(
            model_dir=args.model_dir.resolve(),
            mesh_device=mesh,
            precision_config_path=args.precision_config.resolve(),
            layer_indices=(0, 5),
            tensor_cache_path=args.tensor_cache,
        )
        callback_count = 0

        def next_input(step: int, predicted: int) -> int:
            nonlocal callback_count
            callback_count += 1
            return forced[step]

        generator.generate(
            prompt_token_ids=prompt,
            max_new_tokens=args.tokens,
            next_input=next_input,
            enable_trace=True,
        )
        counters = dict(generator.model.trace_state.counters)
        expected_replays = args.tokens - 1
        if callback_count != args.tokens or counters["model_trace_replays"] != expected_replays:
            raise RuntimeError(
                f"trace smoke mismatch callbacks={callback_count}, replays={counters['model_trace_replays']}, "
                f"expected={expected_replays}"
            )
        result = {
            "config_id": generator.model.config.precision_config_id,
            "status": "pass",
            "layers": [0, 5],
            "tokens": args.tokens,
            "runtime_policy_summary": generator.model.precision_runtime_summary(),
            "trace_counters": counters,
            "command": shlex.join(sys.argv),
            "hardware": "4x Blackhole P150b",
            "mesh": "MeshShape(1,4) TP4 FABRIC_1D",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"config_id": result["config_id"], "status": "pass", "trace_replays": expected_replays}))
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh, "FABRIC_1D")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--precision-config", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-cache", type=Path, default=Path("/tmp/gemma4_31b_datatype_sweep_cache"))
    parser.add_argument("--tokens", type=int, default=3)
    run(parser.parse_args())


if __name__ == "__main__":
    _main()
