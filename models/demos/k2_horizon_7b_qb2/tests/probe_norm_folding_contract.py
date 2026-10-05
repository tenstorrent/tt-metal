"""Fixed layer-1 residual: folded QKV versus native weighted group RMSNorm.

Diagnostic only. Both projections retain BF4 weights, BFP8 AGMM inputs,
BF16 outputs, and the decoder's existing math fidelity and program config.
This script opens TT devices when explicitly executed; importing it does not
run the probe. It does not perform attention, cache updates, or generation.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.multichip_decoder import MultichipDecoder
from .analyze_continuation_cpu import fingerprint, load_layer

RAW = Path("bringup/artifacts/ifm_k2_full_model_raw_20260928")
DOC = Path("models/demos/k2_horizon_7b_qb2/doc/full_model")


def metric(reference, actual):
    a, b = reference.flatten().double(), actual.flatten().double()
    return {
        "pcc": float(torch.corrcoef(torch.stack([a, b]))[0, 1]),
        "relative_l2": float((a - b).norm() / a.norm().clamp_min(1e-30)),
        "max_abs": float((a - b).abs().max()),
        "finite": bool(torch.isfinite(b).all()),
    }


def selected_metrics(reference, actual, rows):
    return {
        "selected_rows": metric(reference, actual),
        "per_row": {str(row): metric(reference[i], actual[i]) for i, row in enumerate(rows)},
    }


def unpack_qkv(ranks):
    # Each rank stores eight Q heads, two K heads, and two V heads, in that order.
    pieces = [rank.split((1024, 256, 256), dim=-1) for rank in ranks]
    return {role: torch.cat([part[i] for part in pieces], dim=-1) for i, role in enumerate("qkv")}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tt-artifact", type=Path, default=RAW / "continuation_l3_s31.pt")
    parser.add_argument("--hf-artifact", type=Path, default=RAW / "hf_continuation_l3.pt")
    parser.add_argument("--input-variant", choices=("full", "split"), default="full")
    parser.add_argument("--output", type=Path, default=DOC / "norm_folding_fixed_input.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    tt_saved = torch.load(args.tt_artifact, map_location="cpu", weights_only=True)
    hf_saved = torch.load(args.hf_artifact, map_location="cpu", weights_only=True)
    hidden = (
        tt_saved["full"][0]
        if args.input_variant == "full"
        else torch.cat([tt_saved["prefix"][0], tt_saved["split"][0]], dim=2)
    )
    hidden = hidden.to(torch.bfloat16)
    assert tuple(hidden.shape[:2]) == (1, 1) and hidden.shape[-1] == 4096
    logical = hidden.shape[2]
    physical = (logical + 31) // 32 * 32
    assert physical >= 256, "This control holds the BFP8 AGMM path fixed"
    tokens = hf_saved["prompt_tokens"].reshape(-1)
    assert tokens.numel() == logical
    bos_rows = torch.where(tokens == 0)[0].tolist()
    rows = sorted(set(bos_rows + [logical - 1]))

    config, hf_layer, _ = load_layer(1, "sdpa")
    state = {f"model.layers.1.{name}": value for name, value in hf_layer.state_dict().items()}
    gamma = hf_layer.input_layernorm.weight.detach()
    normalized_hf = hf_layer.input_layernorm(hidden.squeeze(0))
    reference = {
        role: getattr(hf_layer.self_attn, f"{role}_proj")(normalized_hf).squeeze(0)[rows].double() for role in "qkv"
    }
    grouped = hidden.float().reshape(1, logical, 4, 1024)
    unweighted_hf = (
        (grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + config.rms_norm_eps))
        .reshape(logical, 4096)
        .bfloat16()[rows]
        .double()
    )
    weighted_hf = normalized_hf.squeeze(0)[rows].double()

    original_weights = {
        role: getattr(hf_layer.self_attn, f"{role}_proj").weight.detach().T.contiguous() for role in "qkv"
    }
    original_packed = torch.cat(
        [torch.cat([original_weights[role].chunk(4, dim=-1)[rank] for role in "qkv"], dim=-1) for rank in range(4)],
        dim=-1,
    ).contiguous()
    result = {
        "layer_index": 1,
        "input": f"saved TT {args.input_variant} layer 0 residual; no upstream TT recomputation",
        "tt_artifact": str(args.tt_artifact),
        "tt_artifact_sha256": fingerprint(args.tt_artifact),
        "hf_artifact": str(args.hf_artifact),
        "rows": rows,
        "bos_rows": bos_rows,
        "ordinary_last_row": logical - 1,
        "logical_rows": logical,
        "physical_rows": physical,
        "reference": "Pinned HF weighted group RMSNorm and BF16 Q/K/V projections on the identical TT residual",
        "purpose": "Diagnostic of affine-folding order; weighted RMSNorm is not an accepted production fallback",
        "variants": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        layer = MultichipDecoder.from_state_dict(state, hf_config=config, layer_idx=1, mesh_device=mesh)
        result["policy"] = asdict(layer.policy)

        def upload(value, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )

        def read_local(value):
            return [
                ttnn.to_torch(part).reshape(-1, part.shape[-1])[rows].double()
                for part in ttnn.get_device_tensors(value)
            ]

        x = upload(torch.nn.functional.pad(hidden, (0, 0, 0, physical - logical)))
        gamma_tt = upload(gamma.reshape(1, 1, 1, -1))
        original_weight_tt = upload(original_packed, dtype=layer.wqkv.dtype)
        unweighted_tt = layer._norm(x)
        weighted_tt = ttnn.rms_norm(
            x,
            epsilon=layer.eps,
            weight=gamma_tt,
            compute_kernel_config=layer.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        assert unweighted_tt.shape[-1] == weighted_tt.shape[-1] == 1024
        assert original_weight_tt.dtype == layer.wqkv.dtype == ttnn.bfloat4_b
        outputs = {}
        for name, normalized, weight, weighted in (
            ("folded", unweighted_tt, layer.wqkv, False),
            ("weighted_rms_unfolded", weighted_tt, original_weight_tt, True),
        ):
            # Retain the actual BFP8 source, rather than inspecting AGMM scratch:
            # the scratch's self-rank shard is not part of its public contract.
            cast = ttnn.typecast
            quantized = cast(normalized, ttnn.bfloat8_b)

            def retained_cast(value, dtype, *pos, **kw):
                if value is normalized and dtype == ttnn.bfloat8_b:
                    return quantized
                return cast(value, dtype, *pos, **kw)

            with patch.object(ttnn, "typecast", side_effect=retained_cast):
                output = layer._fused_prefill_projection(normalized, weight)
            actual = unpack_qkv(read_local(output))
            outputs[name] = actual
            norm_host = torch.cat(read_local(normalized), dim=-1)
            quantized_host = torch.cat(read_local(quantized), dim=-1)
            weight_parts = ttnn.get_device_tensors(weight)
            exact_native = unpack_qkv(
                [quantized_host @ ttnn.to_torch(part).reshape(4096, 1536).double() for part in weight_parts]
            )
            # The effective folded activation is expressed in HF's weighted
            # coordinates for comparison only; this multiply is not executed on TT.
            affine = 1.0 if weighted else gamma.double()
            result["variants"][name] = {
                "weight_dtype": str(weight.dtype),
                "input_dtype": str(quantized.dtype),
                "output_dtype": str(output.dtype),
                "native_norm_vs_cpu_same_formula": selected_metrics(
                    weighted_hf if weighted else unweighted_hf, norm_host, rows
                ),
                "effective_norm_vs_hf_weighted": selected_metrics(weighted_hf, norm_host * affine, rows),
                "effective_bfp8_norm_vs_hf_weighted": selected_metrics(weighted_hf, quantized_host * affine, rows),
                "qkv_vs_hf": {role: selected_metrics(reference[role], actual[role], rows) for role in "qkv"},
                "qkv_vs_exact_native_operands": {
                    role: selected_metrics(exact_native[role], actual[role], rows) for role in "qkv"
                },
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "variant": name,
                        "qkv_vs_hf": {
                            role: result["variants"][name]["qkv_vs_hf"][role]["selected_rows"] for role in "qkv"
                        },
                    },
                    indent=2,
                ),
                flush=True,
            )
        result["weighted_vs_folded"] = {
            role: selected_metrics(outputs["folded"][role], outputs["weighted_rms_unfolded"][role], rows)
            for role in "qkv"
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {args.output}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
