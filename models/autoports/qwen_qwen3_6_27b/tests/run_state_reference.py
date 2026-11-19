# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic HF/TT persistent GDN state comparison on captured real activations.

Supports QWEN_DECODER=functional/fused/optimized and normal candidate/policy vars.
No perf claims; does not replace run_decoder's functional-equivalence gates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.evidence import decoder_class, decoder_kwargs, provenance
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import MODEL, REVISION, load_layer


def stats(reference, actual):
    a, b = reference.double().flatten(), actual.double().flatten()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"finite": False, "pcc": None}
    variance_a, variance_b = (a - a.mean()).square().sum(), (b - b.mean()).square().sum()
    correlation = ((a - a.mean()) * (b - b.mean())).sum() / (variance_a * variance_b).sqrt()
    norm = a.square().sum().clamp_min(1e-30)
    return {"finite": True, "pcc": float(correlation) if torch.isfinite(correlation) else None,
            "relative_rms": float(((a - b).square().sum() / norm).sqrt()),
            "max_abs": float((a - b).abs().max()), "reference_rms": float(a.square().mean().sqrt()),
            "actual_rms": float(b.square().mean().sqrt()), "actual_to_reference_gain": float((a * b).sum() / norm)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--activation-file", type=Path, required=True, help="Existing x/d/d2 real-text activation .pt")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dump-intermediates", type=Path)
    ap.add_argument("--user", type=int, action="append", help="Dump selected users; defaults to user0")
    args = ap.parse_args()
    prov = provenance()
    torch.set_num_threads(8)
    fixture = torch.load(args.activation_file, map_location="cpu", weights_only=True)
    x, d, d2 = (fixture[key] for key in ("x", "d", "d2"))
    batch, length, width = x.shape
    assert width == 5120 and d.shape == d2.shape == (batch, 1, width)
    users = sorted(set(args.user or [0]))
    assert all(0 <= user < batch for user in users)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True).text_config
    config._attn_implementation = "sdpa"
    weights = load_layer(args.layer)
    ref = Qwen3_5DecoderLayer(config, args.layer).eval().to(torch.bfloat16)
    ref.load_state_dict(weights)
    assert hasattr(ref, "linear_attn"), "GDN state diagnostic only"
    rotary = Qwen3_5TextRotaryEmbedding(config)
    cache = DynamicCache(config=config)
    inputs = [("prefill", x, torch.arange(length).expand(batch, -1)),
              ("decode1", d, torch.full((batch, 1), length)),
              ("decode2", d2, torch.full((batch, 1), length + 1))]
    expected = {}
    with torch.no_grad():
        for name, token, positions in inputs:
            out = ref(token, position_embeddings=rotary(token, positions), past_key_values=cache)
            hf_state = cache.layers[args.layer]
            expected[name] = {"output": out.clone(), "state": hf_state.recurrent_states.clone(),
                              "conv": hf_state.conv_states.transpose(1, 2)[:, -3:].clone()}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        layer = decoder_class().from_state_dict(weights, hf_config=config, layer_idx=args.layer, mesh_device=mesh, **decoder_kwargs())
        state = layer.create_recurrent_state(batch)
        read = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).clone()
        results = []
        for name, token, positions in inputs:
            forward = layer.prefill_forward if name == "prefill" else layer.decode_forward
            output = read(forward(layer._upload(token), state=state))
            recurrent, conv = read(state.recurrent), read(state.conv)
            hf = expected[name]
            row = {"boundary": name, "output": stats(hf["output"], output), "recurrent": stats(hf["state"], recurrent),
                   "conv": stats(hf["conv"], conv), "hf_state_dtype": str(hf["state"].dtype),
                   "tt_state_dtype": str(recurrent.dtype),
                   "per_user": [{"user": b, "output": stats(hf["output"][b], output[b]),
                                 "state": stats(hf["state"][b], recurrent[b]),
                                 "heads": [stats(hf["state"][b, h], recurrent[b, h]) for h in range(48)]}
                                for b in range(batch)]}
            if args.dump_intermediates:
                args.dump_intermediates.mkdir(parents=True, exist_ok=True)
                path = args.dump_intermediates / f"layer{args.layer}_{name}.pt"
                torch.save({"provenance": prov, "users": users, "input": token[users],
                            "hf": {key: value[users] for key, value in hf.items()},
                            "tt": {"output": output[users], "state": recurrent[users], "conv": conv[users]}}, path)
                row["tensor_artifact"] = str(path)
            results.append(row)
            print(name, "output", row["output"], "recurrent", row["recurrent"], flush=True)
        result = {"provenance": prov, "diagnostic_only_no_perf_claim": True, "layer": args.layer, "batch": batch,
                  "length": length, "activation_file": str(args.activation_file),
                  "activation_sha256": hashlib.sha256(args.activation_file.read_bytes()).hexdigest(), "boundaries": results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        # No new state acceptance reference is declared here. Existing stage state gates still apply.
        assert all(row["output"]["finite"] and row["output"]["pcc"] is not None and row["output"]["pcc"] >= .995 for row in results)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
