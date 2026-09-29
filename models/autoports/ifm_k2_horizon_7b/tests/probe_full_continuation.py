"""Localize whole-prefill versus unaligned-continuation differences per layer."""

import argparse
import inspect
import json
import textwrap
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check import run_prefill_check

import ttnn

from ..tt.generator import K2Generator
from ..tt.multichip_decoder import MultichipDecoder
from ..tt.optimized_decoder import MatmulGeometry, PrecisionPolicy


def compare(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return {
        "pcc": torch.corrcoef(torch.stack([a, b]))[0, 1].item(),
        "relative_l2": float((a - b).norm() / a.norm()),
        "finite": bool(torch.isfinite(b).all()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--split", type=int, default=31)
    p.add_argument("--aligned-control", action="store_true")
    p.add_argument("--readiness", action="store_true")
    p.add_argument("--tensor-dump", action="store_true")
    p.add_argument("--normal-text", action="store_true")
    p.add_argument(
        "--decoder-control",
        choices=["baseline", "math_hifi4", "weights_bfp8", "cache_bf16", "fold_fp32"],
        default="baseline",
    )
    p.add_argument("--prefill-n", type=int, choices=[8, 16], default=16)
    p.add_argument("--output")
    args = p.parse_args()
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    records = {"full": {}, "prefix": {}, "split": {}}
    phase = "full"
    try:
        original_loader = MultichipDecoder.from_state_dict
        if args.decoder_control == "fold_fp32":
            # Diagnostic only: preserve folded topology, precision policy and
            # layouts while eliminating the intermediate BF16 product rounding.
            method = MultichipDecoder.from_state_dict.__func__
            source = textwrap.dedent(inspect.getsource(method))
            old = "return (weight(name).float().T * gamma[:, None]).bfloat16()"
            assert source.count(old) == 1
            source = source.replace(old, "return weight(name).float().T * gamma[:, None]")
            namespace = dict(method.__globals__)
            exec(compile(source, "<diagnostic_fp32_fold>", "exec"), namespace)
            original_loader = namespace["from_state_dict"].__get__(None, MultichipDecoder)

        def controlled(*pos, **kw):
            policy = PrecisionPolicy(
                qkv_geometry=MatmulGeometry(16, 32, 1, False),
                o_geometry=MatmulGeometry(32, 4, 2, False),
                mlp_geometry=MatmulGeometry(32, 8, 3, False),
                down_geometry=MatmulGeometry(32, 12, 1, False),
            )
            changes = {
                "baseline": {},
                "math_hifi4": dict(attention_fidelity="HiFi4", mlp_fidelity="HiFi4", down_fidelity="HiFi4"),
                "weights_bfp8": dict(attention="bfloat8_b", mlp="bfloat8_b", down="bfloat8_b"),
                "cache_bf16": dict(kv="bfloat16"),
                "fold_fp32": {},
            }[args.decoder_control]
            layer = original_loader(*pos, **kw, policy=replace(policy, **changes))
            if args.decoder_control == "math_hifi4":
                layer.attention_compute = layer.compute
            if args.prefill_n == 8:
                layer._prefill_matmul_config = lambda: ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=2,
                    subblock_w=4,
                    compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                )
            return layer

        with patch.object(MultichipDecoder, "from_state_dict", side_effect=controlled):
            gen = K2Generator(mesh, override_num_layers=args.layers)
        suffix = ("_normal" if args.normal_text else "") + (
            "_" + args.decoder_control if args.decoder_control != "baseline" else ""
        )
        suffix += "_n8" if args.prefill_n == 8 else ""
        gen._ensure_owned_cache(1, 4356)
        for index, layer in enumerate(gen.model.layers):
            original = layer.prefill_forward

            def inspected(*pos, original=original, index=index, **kw):
                output = original(*pos, **kw)
                records[phase][index] = ttnn.to_torch(
                    output, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)
                ).clone()
                return output

            layer.prefill_forward = inspected
        base = gen.tokenizer.encode("A careful scientist checks the evidence before drawing a conclusion. ")
        ids = (
            gen.tokenizer.encode("A careful scientist checks the evidence before drawing a conclusion. " * 30)
            if args.normal_text
            else base * 30
        )
        prompt = torch.tensor([ids[:257]])
        if args.normal_text:
            assert gen.tokenizer.bos_token_id not in prompt[0, 1:]
        gen.reset()
        full = gen.prefill_forward(
            prompt, page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[257], sampling_mode="host"
        )
        results, logits_by_split = [], {}
        for cut in [args.split, 32] if args.aligned_control else [args.split]:
            gen.reset()
            phase = "prefix"
            gen.prefill_forward(prompt[:, :cut], page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[cut])
            phase = "split"
            split = gen.prefill_forward(
                prompt[:, cut:],
                page_table=gen.page_table,
                kv_cache=gen.kv_cache,
                prompt_lens=[257 - cut],
                start_pos=[cut],
                sampling_mode="host",
            )
            logits_by_split[cut] = split
            results.append(
                {
                    "split_position": cut,
                    "logits": compare(full, split),
                    "full_greedy": int(full.flatten().argmax()),
                    "split_greedy": int(split.flatten().argmax()),
                    "layer_outputs": [
                        compare(records["full"][i][:, :, -1:, :], records["split"][i][:, :, -1:, :])
                        for i in range(args.layers)
                    ],
                }
            )
            results[-1]["prefix_outputs"] = [
                compare(records["full"][i][:, :, :cut, :], records["prefix"][i]) for i in range(args.layers)
            ]
            results[-1]["suffix_outputs"] = [
                compare(records["full"][i][:, :, cut:, :], records["split"][i]) for i in range(args.layers)
            ]
            if args.tensor_dump:
                dump = Path(
                    f"bringup/artifacts/ifm_k2_full_model_raw_20260928/continuation_l{args.layers}_s{cut}{suffix}.pt"
                )
                dump.parent.mkdir(parents=True, exist_ok=True)
                torch.save(records, dump)
        result = {
            "layers": args.layers,
            "normal_text": args.normal_text,
            "decoder_control": args.decoder_control,
            "prefill_n": args.prefill_n,
            "prompt_tokens": prompt.tolist(),
            "splits": results,
            "head_dtype": gen.model.head_dtype,
            "head_fidelity": gen.model.head_fidelity,
        }
        if args.aligned_control:
            result["unaligned_vs_aligned_logits"] = compare(logits_by_split[args.split], logits_by_split[32])
        if args.readiness:
            assert args.layers == 36
            model_dir = Path("models/autoports/ifm_k2_horizon_7b")

            def chunked_logits(ids):
                gen._ensure_owned_cache(1, len(ids))
                gen.reset()
                rows = []
                for start, end in [(0, args.split), (args.split, len(ids))]:
                    rows.append(
                        gen.prefill_forward(
                            torch.tensor([ids[start:end]]),
                            page_table=gen.page_table,
                            kv_cache=gen.kv_cache,
                            prompt_lens=[end - start],
                            start_pos=[start],
                            return_all_logits=True,
                        )
                    )
                return torch.cat(rows, dim=1)

            with (
                patch.object(gen, "prefill_logits", side_effect=chunked_logits),
                patch.object(run_prefill_check, "_import_build_generator", return_value=lambda **kw: gen),
            ):
                result["chunked_hf_topk"] = run_prefill_check.run_prefill_check(
                    model_dir=model_dir, reference_path=model_dir / "readiness_aime24_chat.refpt", mesh_device=mesh
                )
        path = (
            Path(args.output)
            if args.output
            else Path(
                f"models/autoports/ifm_k2_horizon_7b/doc/full_model/continuation_l{args.layers}_s{args.split}{suffix}.json"
            )
        )
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
