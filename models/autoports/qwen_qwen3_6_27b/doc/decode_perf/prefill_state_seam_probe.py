"""Is the linear-attention state after a batch-32 prefill the same in both
conv arms?  Dumps conv and recurrent state in one common layout so the fused
and composite arms can be compared bit for bit.
"""
import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator

BATCH = 32


def conv_common(layer):
    """Return the conv state as [B, K, C] whichever layout it is stored in."""
    shards = [ttnn.to_torch(s).clone().float() for s in ttnn.get_device_tensors(layer.caches["conv"])]
    out = []
    for s in shards:
        if s.dim() == 3:  # fused window [B, K, C]
            out.append(s)
        else:  # composite [1, B, C, K]
            out.append(s[0].permute(0, 2, 1).contiguous())
    return out


def rec_common(layer):
    return [ttnn.to_torch(s).clone().float() for s in ttnn.get_device_tensors(layer.caches["recurrent"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--baseline", type=Path)
    args = ap.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    gen = None
    try:
        gen = build_generator(
            model_dir=Path("models/autoports/qwen_qwen3_6_27b"),
            mesh_device=mesh,
            max_context=512,
            batch=BATCH,
            num_layers=args.num_layers,
        )
        rendered = gen.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt.read_text().strip()}], tokenize=False, add_generation_prompt=True
        )
        ids = gen.tokenizer.encode(rendered, add_special_tokens=False)
        tokens = torch.stack([torch.tensor(ids[b : b + args.prompt_tokens], dtype=torch.long) for b in range(BATCH)])
        gen.reset()
        gen.prefill_forward(
            tokens,
            page_table=gen._page_table,
            kv_cache=gen.kv_cache,
            prompt_lens=[args.prompt_tokens] * BATCH,
        )
        ttnn.synchronize_device(mesh)
        state = {}
        for i, layer in enumerate(gen.model.layers):
            if layer.layer_kind != "linear_attention":
                continue
            state[f"conv{i}"] = torch.stack(conv_common(layer))
            state[f"rec{i}"] = torch.stack(rec_common(layer))
        cfg = gen.model.precision_config.config_id
    finally:
        if gen is not None:
            gen.reset()
        ttnn.close_mesh_device(mesh)

    torch.save({"state": state, "config": cfg}, args.output)
    print(json.dumps({"config": cfg, "tensors": sorted(state)}, indent=2), flush=True)

    if args.baseline:
        base = torch.load(args.baseline, weights_only=False)
        report = {"this": cfg, "baseline": base["config"], "per_tensor": []}
        for key in sorted(state):
            a, b = state[key], base["state"][key]
            if a.shape != b.shape:
                report["per_tensor"].append({"tensor": key, "shape_mismatch": [list(a.shape), list(b.shape)]})
                continue
            diff = (a - b).abs()
            # per fixed slot, so a single bad row is visible
            slot_axis = 1 if key.startswith("conv") else 1
            worst_slot = int(diff.amax(dim=tuple(d for d in range(diff.dim()) if d != slot_axis)).argmax())
            report["per_tensor"].append(
                {
                    "tensor": key,
                    "max_abs_diff": float(diff.max()),
                    "mean_abs_diff": float(diff.mean()),
                    "worst_slot": worst_slot,
                    "exact": bool(torch.equal(a, b)),
                }
            )
        report["all_exact"] = all(t.get("exact") for t in report["per_tensor"])
        report["worst_max_abs_diff"] = max(t.get("max_abs_diff", 0.0) for t in report["per_tensor"])
        print(json.dumps(report, indent=2), flush=True)
        args.output.with_suffix(".cmp.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
