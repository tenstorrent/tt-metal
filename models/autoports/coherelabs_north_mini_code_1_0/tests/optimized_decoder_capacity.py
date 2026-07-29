# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Explicit long-context probes for the optimized decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tests.functional_decoder_capacity import (
    _decode_probe,
    _prefill_probe,
)
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder import (
    REAL_REVISION,
    _synthetic_state,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import MODEL_ID, OptimizedDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0, choices=(0, 1, 4))
    parser.add_argument("--warmed", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(MODEL_ID, revision=REAL_REVISION, local_files_only=True)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = OptimizedDecoder.from_state_dict(
            _synthetic_state(config, args.layer, sparse_weights=args.layer != 0),
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh_device,
            batch=args.batch,
            max_cache_len=args.context,
        )
        if args.mode == "decode":
            result = _decode_probe(decoder, mesh_device, config, args.context)
        else:
            result = _prefill_probe(decoder, mesh_device, config, args.context, warmed=args.warmed)
    finally:
        ttnn.close_mesh_device(mesh_device)

    result.update(
        {
            "mode": args.mode,
            "batch": args.batch,
            "context": args.context,
            "layer": args.layer,
            "model_revision": REAL_REVISION,
            "kv_cache_dtype": "bfloat16",
            "kv_cache_bytes": args.batch * args.context * 2 * 4 * 128 * 2,
        }
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
