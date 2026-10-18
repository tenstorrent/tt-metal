# SPDX-License-Identifier: Apache-2.0
"""Localize dedicated-expert work admission, independently of residual dilution."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.fusion_candidates import DedicatedMoE
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tests.test_functional_decoder import CHECKPOINT
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedMoE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-weights", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    tag = "moe_capacity_bf16" if args.bf16_weights else "moe_capacity_bfp8"
    dest = root / "doc/fused_decoder" / tag
    dest.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
    state = CheckpointState(CHECKPOINT, "model.language_model.layers.0.")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        reference = FusedMoE.from_state_dict(state, hf_config=cfg, mesh_device=mesh)
        if args.bf16_weights:
            DedicatedMoE.dedicated_weight_dtype = ttnn.bfloat16
        candidate = DedicatedMoE.from_state_dict(state, hf_config=cfg, mesh_device=mesh)
        candidate.retain_intermediates = True
        torch.manual_seed(2026)
        x = torch.randn(1, 1, 32, cfg.hidden_size).bfloat16()
        xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        expected = ttnn.to_torch(reference._group(xd))
        artifacts = dict(x=x, expected=expected)
        results = {}
        for pad_capacity in (False, True):
            candidate.pad_capacity = pad_capacity
            result = ttnn.to_torch(candidate._group(xd))
            routing = ttnn.to_torch(candidate.intermediates["routing"])
            active = torch.where(routing.sum((0, 1, 2)) > 0)[0]
            expert = int(active[active > 0][0])
            down = ttnn.to_torch(candidate.intermediates["down"][:, expert : expert + 1])
            # Same uploaded expert weights in a normal FP32-accumulating TTNN control.
            gate = ttnn.linear(xd, candidate.expert_gates[expert], compute_kernel_config=reference.compute)
            up = ttnn.linear(xd, candidate.expert_ups[expert], compute_kernel_config=reference.compute)
            hidden = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            down_control = ttnn.to_torch(
                ttnn.linear(hidden, candidate.expert_downs[expert], compute_kernel_config=reference.compute)
            )
            key = "padded" if pad_capacity else "original"
            results[key] = dict(
                final=pcc_metrics(expected, result),
                expert=expert,
                active_experts=active.tolist(),
                expert_nonzero=int(torch.count_nonzero(down)),
                expert_metrics=pcc_metrics(down_control, down) if torch.count_nonzero(down) else None,
                per_token_min_pcc=min(pcc_metrics(expected[..., i, :], result[..., i, :])["pcc"] for i in range(32)),
            )
            artifacts[key] = dict(
                result=result, routing=routing, selected_expert=expert, down=down, down_control=down_control
            )
            print(json.dumps({key: results[key]}), flush=True)
        saved = Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/fusion_outputs") / (
            tag + ".pt"
        )
        torch.save(artifacts, saved)
        results["artifacts"] = str(saved)
        results["source_hashes"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), root / "tests/fusion_candidates.py")
        }
        (dest / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
