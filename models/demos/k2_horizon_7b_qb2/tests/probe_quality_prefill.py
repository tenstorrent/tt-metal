"""Capture real chat-prefill residuals to localize early quality divergence."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.generator import K2Generator
from ..tt.multichip_decoder import MultichipDecoder
from .probe_precision_policy import precision_loader
from .run_qualitative_extended import run

DOC = Path("models/demos/k2_horizon_7b_qb2/doc/full_model")
RAW = Path("bringup/artifacts/ifm_k2_full_model_raw_20260928")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", default=["shared_1", "shared_3", "shared_4"])
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    controls = [row for row in json.loads((DOC / "hf_qualitative.json").read_text()) if row["id"] in args.ids]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    originals, captured = [], {}
    try:
        if args.policy:
            with patch.object(
                MultichipDecoder, "from_state_dict", side_effect=precision_loader(json.loads(args.policy.read_text()))
            ):
                gen = K2Generator(mesh)
        else:
            gen = K2Generator(mesh)
        gen._ensure_owned_cache(1, 640)
        record = {
            "layers": 36,
            "runtime_policies": [asdict(layer.policy) for layer in gen.model.layers],
            "weight_allocations": [layer.weight_allocations for layer in gen.model.layers],
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((DOC.parents[1] / "tt").glob("*.py"))
            },
            "policy": str(args.policy) if args.policy else "production default",
            "records": [],
        }
        for index, layer in enumerate(gen.model.layers):
            original = layer.prefill_forward
            originals.append(original)

            def inspected(*pos, original=original, index=index, **kwargs):
                output = original(*pos, **kwargs)
                captured[current_id]["layer_outputs"][index] = ttnn.to_torch(
                    output, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)
                ).clone()
                return output

            layer.prefill_forward = inspected
        for control in controls:
            current_id = control["id"]
            ids = control["prompt_token_ids"]
            captured[current_id] = {"prompt_token_ids": ids, "layer_outputs": {}}
            gen.reset()
            logits = gen.prefill_forward(
                torch.tensor([ids]),
                page_table=gen.page_table,
                kv_cache=gen.kv_cache,
                prompt_lens=[len(ids)],
                sampling_mode="host",
            )
            captured[current_id]["final_logits"] = logits
            top = logits.flatten().float().topk(10)
            row = {
                "id": current_id,
                "prompt_token_ids": ids,
                "top10": [
                    {"id": int(i), "text": gen.tokenizer.decode([int(i)]), "logit": float(v)}
                    for i, v in zip(top.indices, top.values)
                ],
            }
            record["records"].append(row)
            print("QUALITY_PREFILL", json.dumps(row), flush=True)
        for layer, original in zip(gen.model.layers, originals):
            layer.prefill_forward = original
        raw = RAW / (args.output.stem + ".pt")
        torch.save(captured, raw)
        record["raw_capture"] = str(raw)
        if args.generate:
            quality = args.output.with_name(args.output.stem + "_quality.json")
            run(gen, selected_ids=args.ids, output_name=str(quality.relative_to(DOC)))
            record["quality_artifact"] = str(quality)
        record["complete"] = True
        args.output.write_text(json.dumps(record, indent=2) + "\n")
    finally:
        if gen is not None:
            for layer, original in zip(gen.model.layers, originals):
                layer.prefill_forward = original
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
