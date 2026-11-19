# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Use the CI input distribution with real checkpoint weights and unchanged gates."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import test_synthetic_decoder as shared
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=(0, 3), default=0)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--sequences", type=int, nargs="+", default=list(range(1, 17)))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    weights = load_weights(args.snapshot, args.layer, "")
    with torch.device("meta"):
        reference = Qwen4ExpTextDecoderLayer(config, args.layer)
    reference.load_state_dict(weights, assign=True)
    reference.eval()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    report = dict(factory="OptimizedDecoder", layer=args.layer, checkpoint=str(args.snapshot), rows=[], passed=False)
    try:
        policy = json.loads(args.policy.read_text()) if args.policy else {}
        model = OptimizedDecoder.from_state_dict(
            weights, hf_config=config, layer_idx=args.layer, mesh_device=mesh, **policy
        )
        assert type(model) is OptimizedDecoder
        for sequence in args.sequences:
            row = dict(sequence=sequence, seed=1900 + 100 * args.layer + sequence, passed=False, properties={})
            try:
                shared.test_full_decoder_prefill_and_traced_decode(
                    (config, args.layer, reference, model),
                    mesh,
                    sequence,
                    lambda key, value: row["properties"].update({key: value}),
                )
                row["passed"] = True
            except AssertionError as error:
                row["error"] = str(error)
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(sequence, row["passed"], row.get("error", ""), flush=True)
        report["passed"] = all(row["passed"] for row in report["rows"])
    finally:
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
