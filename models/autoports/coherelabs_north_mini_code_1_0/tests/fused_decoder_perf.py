# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Warmed Stage-02 fused-decoder latency and Tracy-signpost harness."""

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
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import MODEL_ID, FunctionalDecoder
from models.autoports.coherelabs_north_mini_code_1_0.tt.fused_decoder import FusedDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("functional", "fused"), required=True)
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0, choices=(0, 1, 4))
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--dense-gate-up-variant",
        choices=("decode_only", "packed_all", "packed_slice"),
        default="packed_slice",
    )
    parser.add_argument("--sparse-gate-up-variant", choices=("separate", "packed"), default="packed")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(MODEL_ID, revision=REAL_REVISION)
    state = _synthetic_state(config, args.layer)
    decoder_type = FunctionalDecoder if args.implementation == "functional" else FusedDecoder
    max_cache_len = args.sequence if args.mode == "prefill" else 32
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = decoder_type.from_state_dict(
            state,
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh_device,
            batch=args.batch,
            max_cache_len=max_cache_len,
            dense_gate_up_variant=args.dense_gate_up_variant,
            sparse_gate_up_variant=args.sparse_gate_up_variant,
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
        implementation=args.implementation,
        mode=args.mode,
        batch=args.batch,
        sequence=args.sequence,
        layer=args.layer,
        model_revision=REAL_REVISION,
        dense_gate_up_variant=args.dense_gate_up_variant,
        sparse_gate_up_variant=args.sparse_gate_up_variant,
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
