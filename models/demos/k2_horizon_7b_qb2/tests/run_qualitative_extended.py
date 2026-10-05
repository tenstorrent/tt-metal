"""Longer shared-suite outputs, with independent pinned HF controls in progress."""

import argparse
import json
from pathlib import Path

import torch

import ttnn

from ..tt.generator import K2Generator

DOC = Path("models/demos/k2_horizon_7b_qb2/doc/full_model")


def run(gen, *, selected_ids=None, output_name="qualitative_extended.json"):
    controls = json.loads((DOC / "hf_qualitative.json").read_text())
    records = []
    for control in controls:
        if selected_ids is not None and control["id"] not in selected_ids:
            continue
        output = gen.generate(control["prompt_token_ids"], 512)
        # The Metal generator is fixed-step; external drivers own stopping.
        # Preserve raw output and a first-EOS view comparable to HF.generate.
        eos = gen.tokenizer.eos_token_id
        end = output.index(eos) + 1 if eos in output else len(output)
        records.append(
            {
                "id": control["id"],
                "prompt_token_ids": control["prompt_token_ids"],
                "tt_token_ids": output,
                "tt_text": gen.tokenizer.decode(output, skip_special_tokens=False),
                "first_eos_index": end - 1 if eos in output else None,
                "tt_text_through_eos": gen.tokenizer.decode(output[:end], skip_special_tokens=False),
                "hf_control_file": "hf_qualitative_extended.json",
                "perf": gen.last_perf.copy(),
            }
        )
        (DOC / output_name).write_text(json.dumps(records, indent=2) + "\n")
        print("EXTENDED", control["id"], records[-1]["tt_text_through_eos"], flush=True)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head-dtype", default="bfloat8_b")
    p.add_argument("--head-fidelity", default="HiFi2")
    p.add_argument("--head-split-size", type=int, default=16384)
    p.add_argument("--head-k", type=int, default=2)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--output-name", default="qualitative_extended.json")
    args = p.parse_args()
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    records = []
    try:
        gen = K2Generator(
            mesh,
            head_dtype=args.head_dtype,
            head_fidelity=args.head_fidelity,
            head_split_size=args.head_split_size,
            head_k=args.head_k,
        )
        run(gen, selected_ids=args.ids, output_name=args.output_name)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
