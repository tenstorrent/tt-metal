# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate HF BF16-logit rounding from upstream mixing and TT router precision."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(932)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    weights = load_weights(args.snapshot, 0, "")
    with torch.device("meta"):
        reference = Qwen4ExpTextDecoderLayer(config, 0)
    reference.load_state_dict(weights, assign=True)
    reference.eval()
    hidden = (torch.randn(1, 65, 10240) * 0.02).bfloat16()[:, :32].contiguous()
    captured = {}

    def capture(_module, inputs, output):
        captured["input"] = inputs[0].clone()
        captured["logits"], captured["scores"], captured["indices"] = [value.clone() for value in output]

    reference.mlp.gate.register_forward_hook(capture)
    with torch.inference_mode():
        reference(hidden, position_embeddings=(None, None), past_key_values=DynamicCache(config=config))
    report = {"seed": 932, "seq_len": 32, "layer": 0, "policies": []}
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import (
        GatedDeltaNet,
        GatedResidual,
        WeightLoader,
        linear,
    )

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        loader = WeightLoader(weights, mesh)
        attn_mix = GatedResidual(loader, "attn_hyper_connection.", config)
        attention = GatedDeltaNet(loader, config)
        mlp_mix = GatedResidual(loader, "mlp_hyper_connection.", config)
        router = loader.matrix("mlp.gate.weight", ttnn.float32)
        x = loader.tensor(hidden.unsqueeze(0))
        mixed, injection = attn_mix.read(x)
        conv = ttnn.zeros((1, 1, 3, 10240), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        state = ttnn.zeros((1, 48, 128, 128), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        block = attention(mixed, conv_state=conv, recurrent_state=state)
        updated = attn_mix.write(x, block, injection)
        actual_input, _ = mlp_mix.read(updated)
        exact_input = loader.tensor(captured["input"].reshape(1, 1, 32, 2560))
        reference_sets = captured["indices"].sort(dim=-1).values
        for name, value in (("exact_hf_input", exact_input), ("tt_upstream_input", actual_input)):
            logits = linear(ttnn.typecast(value, ttnn.float32), router, loader.compute)
            for rounding in (False, True):
                rounded = ttnn.typecast(ttnn.typecast(logits, ttnn.bfloat16), ttnn.float32) if rounding else logits
                top, indices = ttnn.topk(rounded, k=10, dim=-1, sorted=True)
                actual_indices = ttnn.to_torch(indices).reshape(32, 10).long()
                actual_logits = ttnn.to_torch(rounded).reshape(32, 512)
                mismatches = (actual_indices.sort(dim=-1).values != reference_sets).any(dim=-1)
                rows = torch.where(mismatches)[0].tolist()
                report["policies"].append(
                    {
                        "input": name,
                        "round_logits_to_bf16": rounding,
                        "mismatched_rows": rows,
                        "maximum_logit_error": (actual_logits - captured["logits"].float()).abs().max().item(),
                        "details": [
                            {
                                "row": row,
                                "hf_indices": captured["indices"][row].tolist(),
                                "tt_indices": actual_indices[row].tolist(),
                                "hf_top12_logits": captured["logits"][row].float().topk(12).values.tolist(),
                                "tt_top12_logits": actual_logits[row].topk(12).values.tolist(),
                            }
                            for row in rows
                        ],
                    }
                )
    finally:
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "details"} for row in report["policies"]]), flush=True)


if __name__ == "__main__":
    main()
