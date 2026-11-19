# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real target shape/weight component diagnostics against the installed HF code."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextGatedDeltaNet,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextSparseMoeBlock,
)


def load_weights(snapshot, layer, component):
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    layer_prefix = f"model.language_model.layers.{layer}."
    prefix = layer_prefix + component
    state = {}
    for filename in sorted({v for k, v in index.items() if k.startswith(prefix)}):
        with safe_open(snapshot / filename, framework="pt") as stream:
            for key in stream.keys():
                if key.startswith(prefix):
                    state[key.removeprefix(layer_prefix)] = stream.get_tensor(key)
    return state


def pcc(expected, actual):
    return torch.corrcoef(torch.stack((expected.float().flatten(), actual.float().flatten())))[0, 1].item()


def main():
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import (
        GatedDeltaNet,
        GatedResidual,
        SparseMoe,
        WeightLoader,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--component", choices=("gdn", "moe", "residual"), required=True)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--expert-dtype", choices=("bfloat16", "bfloat8_b", "bfloat4_b"), default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    torch.manual_seed(19)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    prefix = {"gdn": "linear_attn.", "moe": "mlp.", "residual": "attn_hyper_connection."}[args.component]
    state = load_weights(args.snapshot, 0, prefix)
    with torch.device("meta"):
        reference = {
            "gdn": lambda: Qwen4ExpTextGatedDeltaNet(config, 0),
            "moe": lambda: Qwen4ExpTextSparseMoeBlock(config),
            "residual": lambda: Qwen4ExpTextGatedResidual(config),
        }[args.component]()
    reference.load_state_dict({k.removeprefix(prefix): v for k, v in state.items()}, assign=True)
    reference.eval()
    width = config.hidden_size * (config.hc_count if args.component == "residual" else 1)
    x = (torch.randn(1, args.seq_len, width) * 0.02).bfloat16()
    cache = DynamicCache(config=config)
    with torch.no_grad():
        expected = reference(x, cache_params=cache) if args.component == "gdn" else reference(x)
    print("REFERENCE_OK", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    report = {"component": args.component, "seq_len": args.seq_len, "weights": "real", "layer_idx": 0}
    report["expert_dtype"] = args.expert_dtype if args.component == "moe" else None
    try:
        loader = WeightLoader(state, mesh)
        module = {
            "gdn": lambda: GatedDeltaNet(loader, config),
            "moe": lambda: SparseMoe(loader, config, expert_dtype=getattr(ttnn, args.expert_dtype)),
            "residual": lambda: GatedResidual(loader, prefix, config),
        }[args.component]()
        inputs = ttnn.from_torch(x.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        if args.component == "gdn":
            conv_state = ttnn.zeros((1, 1, 3, 10240), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            recurrent_state = ttnn.zeros((1, 48, 128, 128), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
            result = module(inputs, conv_state=conv_state, recurrent_state=recurrent_state)
        elif args.component == "residual":
            result, injection = module.read(inputs)
            report["injection_pcc"] = pcc(
                expected[2], ttnn.to_torch(injection).permute(0, 2, 1, 3).reshape(expected[2].shape)
            )
            expected = expected[0]
        else:
            result = module(inputs)
        actual = ttnn.to_torch(result).reshape(expected.shape)
        report["pcc"] = pcc(expected, actual)
        report["max_abs_error"] = (expected.float() - actual.float()).abs().max().item()
        report["finite"] = bool(torch.isfinite(actual).all())
    finally:
        ttnn.close_mesh_device(mesh)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["finite"] or report["pcc"] < 0.995:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
