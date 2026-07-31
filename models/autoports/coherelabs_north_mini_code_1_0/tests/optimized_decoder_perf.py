# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Candidate-aware warmed latency and Tracy harness for OptimizedDecoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tests.functional_decoder_perf import _decode, _prefill
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder import (
    REAL_REVISION,
    _synthetic_state,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import MODEL_ID, POLICIES, OptimizedDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0, choices=(0, 1, 4))
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--candidate", choices=sorted(POLICIES), default="default")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(MODEL_ID, revision=REAL_REVISION)
    state = _synthetic_state(config, args.layer, sparse_weights=args.layer != 0)
    max_cache_len = args.sequence if args.mode == "prefill" else 32
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh_device,
            batch=args.batch,
            max_cache_len=max_cache_len,
            candidate=args.candidate,
        )
        if args.mode == "prefill":
            result = _prefill(
                decoder,
                mesh_device,
                config,
                sequence=args.sequence,
                warmups=args.warmups,
                iterations=args.iterations,
            )
        else:
            result = _decode(
                decoder,
                mesh_device,
                config,
                warmups=args.warmups,
                iterations=args.iterations,
            )
    finally:
        ttnn.close_mesh_device(mesh_device)

    result.update(
        {
            "mode": args.mode,
            "batch": args.batch,
            "sequence": args.sequence,
            "layer": args.layer,
            "candidate": args.candidate,
            "model_revision": REAL_REVISION,
        }
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
