"""Diagnostic targeted precision policies on the real TP4 full-model path."""

import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.generator import K2Generator
from ..tt.multichip_decoder import MultichipDecoder
from ..tt.optimized_decoder import MatmulGeometry, PrecisionPolicy
from .probe_full_continuation import compare

RAW = Path("bringup/artifacts/ifm_k2_full_model_raw_20260928")


def precision_loader(configuration):
    original = MultichipDecoder.from_state_dict

    def controlled(*args, **kwargs):
        changes = dict(configuration.get("default", {}))
        changes.update(configuration.get("layers", {}).get(str(kwargs["layer_idx"]), {}))
        extras = {
            key: changes.pop(key)
            for key in [
                "decode_fp32",
                "prefill_m",
                "prefill_n",
                "fused_activation",
                "fused_mlp_activation",
                "down_block_w",
                "qkv_block_w",
            ]
            if key in changes
        }
        policy = PrecisionPolicy(
            qkv_geometry=MatmulGeometry(16, extras.get("qkv_block_w", 32), 1, extras.get("decode_fp32", False)),
            o_geometry=MatmulGeometry(32, 4, 2, extras.get("decode_fp32", False)),
            mlp_geometry=MatmulGeometry(32, 8, 3, extras.get("decode_fp32", False)),
            down_geometry=MatmulGeometry(32, extras.get("down_block_w", 12), 1, extras.get("decode_fp32", False)),
        )
        for alias, field in [
            ("fused_activation", "prefill_qkv_activation"),
            ("fused_mlp_activation", "prefill_mlp_activation"),
            ("prefill_m", "prefill_m_block"),
        ]:
            if alias in extras:
                if field in changes:
                    raise ValueError(f"Specify only one of {alias} and {field}")
                changes[field] = extras[alias]
        if "fused_activation" in extras:
            changes.setdefault("prefill_mlp_activation", extras["fused_activation"])
        kwargs.pop("policy", None)
        layer = original(*args, **kwargs, policy=replace(policy, **changes))
        if extras.get("prefill_n", 16) != 16:
            # Historical N8 diagnostic; current candidates keep inherited N16.
            layer._prefill_matmul_config = lambda: ttnn.MinimalMatmulConfig(
                M_block_size=layer.policy.prefill_m_block,
                K_block_size=8,
                N_block_size=extras["prefill_n"],
                subblock_h=2,
                subblock_w=2 if layer.policy.prefill_fp32 else 4,
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
            )
        return layer

    return controlled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--quality", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(8)
    configuration = json.loads(args.policy.read_text())
    hf_path = RAW / "hf_continuation_l36.pt"
    if not hf_path.exists():
        hf_path = RAW / "hf_continuation_l3.pt"
    hf = torch.load(hf_path, map_location="cpu", weights_only=True)
    prompt = hf["prompt_tokens"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    records, phase, originals = {}, "whole", []
    result = {"policy": configuration, "layers": args.layers, "prompt_token_ids": prompt[0].tolist(), "checks": []}
    root = Path("models/autoports/ifm_k2_horizon_7b")
    result["provenance"] = {
        "command": shlex.join([sys.executable, "-m", __spec__.name, *sys.argv[1:]]),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "hf_reference": str(hf_path),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                root / "tt/optimized_decoder.py",
                root / "tt/multichip_decoder.py",
                root / "tt/model.py",
                root / "tt/generator.py",
            ]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with patch.object(MultichipDecoder, "from_state_dict", side_effect=precision_loader(configuration)):
            gen = K2Generator(mesh, override_num_layers=args.layers)
        result["runtime_policies"] = [asdict(layer.policy) for layer in gen.model.layers]
        result["weight_allocations"] = [layer.weight_allocations for layer in gen.model.layers]
        gen._ensure_owned_cache(1, 512)
        for index, layer in enumerate(gen.model.layers):
            original = layer.prefill_forward
            originals.append(original)

            def inspected(*pos, original=original, index=index, **kwargs):
                output = original(*pos, **kwargs)
                records.setdefault(phase, {})[index] = ttnn.to_torch(
                    output, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)
                ).clone()
                return output

            layer.prefill_forward = inspected
        gen.reset()
        whole = gen.prefill_forward(
            prompt, page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[257], sampling_mode="host"
        )
        all_logits = {"whole": whole}
        for cut in [31, 32]:
            gen.reset()
            phase = f"prefix{cut}"
            gen.prefill_forward(prompt[:, :cut], page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[cut])
            phase = f"split{cut}"
            split = gen.prefill_forward(
                prompt[:, cut:],
                page_table=gen.page_table,
                kv_cache=gen.kv_cache,
                prompt_lens=[257 - cut],
                start_pos=[cut],
                sampling_mode="host",
            )
            all_logits[f"split{cut}"] = split
            check = {
                "cut": cut,
                "whole_split_logits": compare(whole, split),
                "whole_token": int(whole.flatten().argmax()),
                "split_token": int(split.flatten().argmax()),
                "layers": [],
            }
            for index in range(args.layers):
                w, s = records["whole"][index], records[phase][index]
                row = {"index": index, "whole_split_last": compare(w[:, :, -1:], s[:, :, -1:])}
                if index in hf["layer_outputs"]:
                    h = hf["layer_outputs"][index]
                    both = torch.cat([records[f"prefix{cut}"][index], s], dim=2)
                    row.update(
                        whole_hf_last=compare(h[:, -1:], w[:, :, -1:]),
                        split_hf_last=compare(h[:, -1:], s[:, :, -1:]),
                        whole_hf_all=compare(h, w),
                        split_hf_all=compare(h, both),
                    )
                check["layers"].append(row)
            result["checks"].append(check)
            if args.layers == 36:
                reference = json.loads(
                    Path("models/autoports/ifm_k2_horizon_7b/doc/full_model/stress_hf_topk.json").read_text()
                )["records"][0]
                assert reference["prompt_token_ids"] == result["prompt_token_ids"]
                ranks = {row["id"]: row["rank"] for row in reference["hf_top100"]}
                check["hf_reference_top1"] = reference["hf_top1"]["id"]
                check["whole_hf_rank"] = ranks.get(check["whole_token"])
                check["split_hf_rank"] = ranks.get(check["split_token"])
                check["passes_continuation_pcc"] = check["whole_split_logits"]["pcc"] >= 0.995
                if "final_logits" in hf:
                    check["whole_hf_logits"] = compare(hf["final_logits"], whole)
                    check["split_hf_logits"] = compare(hf["final_logits"], split)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("PRECISION_CHECK", json.dumps({k: v for k, v in check.items() if k != "layers"}), flush=True)
        if args.dump:
            torch.save(
                {
                    "records": records,
                    "whole_logits": whole,
                    "last_split_logits": split,
                    "all_logits": all_logits,
                    "policy": configuration,
                },
                RAW / (args.output.stem + ".pt"),
            )
        for layer, original in zip(gen.model.layers, originals):
            layer.prefill_forward = original
        if args.quality:
            from .run_qualitative_extended import run

            quality_path = args.output.with_name(args.output.stem + "_quality.json")
            run(
                gen,
                selected_ids=args.quality,
                output_name=str(quality_path.relative_to(Path("models/autoports/ifm_k2_horizon_7b/doc/full_model"))),
            )
            result["quality_artifact"] = str(quality_path)
        result["complete"] = True
        result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    except Exception as error:
        result["error"] = str(error).split("backtrace:")[0][:2000]
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        raise
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
