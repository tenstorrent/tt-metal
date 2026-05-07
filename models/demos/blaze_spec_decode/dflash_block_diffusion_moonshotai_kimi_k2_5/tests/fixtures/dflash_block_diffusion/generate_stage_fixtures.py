#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


DEFAULT_GOLDEN_PATH = Path(
    "/Users/jrock/repos/flows/spec-decode-paper-reproduction/"
    "dflash_block_diffusion_agent_23/golden_dflash_block_diffusion_agent_23"
)


def _tensor(value: torch.Tensor) -> list:
    return value.detach().cpu().float().tolist()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    return weight.float() * x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)


def build_fixtures(golden_path: Path, output_dir: Path) -> None:
    sys.path.insert(0, str(golden_path))
    from golden_dflash.config import DFlashConfig
    from golden_dflash.draft_model import KimiDFlashDraftModel

    torch.manual_seed(28028)
    config = DFlashConfig.tiny(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        block_size=4,
        target_layer_ids=[0, 1],
    )
    model = KimiDFlashDraftModel(config).eval()

    target_len = 3
    position_ids = torch.arange(target_len + config.block_size, dtype=torch.long).unsqueeze(0)
    target_hidden = torch.randn(1, target_len, len(config.target_layer_ids) * config.hidden_size)
    noise_embedding = torch.randn(1, config.block_size, config.hidden_size)
    base_hidden = torch.randn(1, config.hidden_size)
    base_norm_weight = torch.randn(config.hidden_size).abs() + 0.1
    target_lm_head_weight = torch.randn(config.vocab_size, config.hidden_size)

    base_normed = _rms_norm(base_hidden, base_norm_weight, config.rms_norm_eps)
    base_logits = base_normed @ target_lm_head_weight.T
    base_token_id = int(torch.argmax(base_logits, dim=-1)[0].item())

    target_context = model.hidden_norm(model.fc(target_hidden))
    cos, sin = model.rotary_emb(position_ids, noise_embedding.dtype, noise_embedding.device)

    hidden_by_stage = [noise_embedding]
    for layer in model.layers:
        hidden_by_stage.append(layer(hidden_by_stage[-1], target_context, (cos, sin)))

    final_hidden = model.norm(hidden_by_stage[-1])
    draft_logits = final_hidden[:, 1 - config.block_size :, :] @ target_lm_head_weight.T
    draft_token_ids = torch.argmax(draft_logits, dim=-1)

    common = {
        "schema_version": 1,
        "approach": "dflash_block_diffusion",
        "target_model": "moonshotai/Kimi-K2.5",
        "draft_model": "z-lab/Kimi-K2.5-DFlash",
        "golden_reference": "golden_dflash.KimiDFlashDraftModel",
        "seed": 28028,
        "config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "block_size": config.block_size,
            "max_position_embeddings": config.max_position_embeddings,
            "rope_theta": config.rope_theta,
            "rms_norm_eps": config.rms_norm_eps,
            "target_layer_ids": config.target_layer_ids,
            "mask_token_id": config.mask_token_id,
        },
    }

    _write_json(
        output_dir / "stage_pre_decoder_fused.json",
        {
            **common,
            "stage": "pre_decoder_fused",
            "mapping": (
                "base-model norm/lmhead/sampling plus DFlash target-hidden projection, "
                "noise block, and RoPE prep"
            ),
            "inputs": {
                "base_hidden": _tensor(base_hidden),
                "target_hidden": _tensor(target_hidden),
                "noise_embedding": _tensor(noise_embedding),
                "position_ids": position_ids.tolist(),
            },
            "weights": {
                "base_norm_weight": _tensor(base_norm_weight),
                "target_lm_head_weight": _tensor(target_lm_head_weight),
                "fc_weight": _tensor(model.fc.weight),
                "hidden_norm_weight": _tensor(model.hidden_norm.weight),
            },
            "expected": {
                "base_logits": _tensor(base_logits),
                "base_token_id": base_token_id,
                "target_context": _tensor(target_context),
                "position_cos": _tensor(cos),
                "position_sin": _tensor(sin),
                "decoder_input": _tensor(noise_embedding),
            },
        },
    )

    for layer_idx, layer in enumerate(model.layers):
        _write_json(
            output_dir / f"stage_decoder_layer_{layer_idx}.json",
            {
                **common,
                "stage": f"decoder_layer_{layer_idx}",
                "mapping": "one DFlash drafter decoder layer stage",
                "inputs": {
                    "hidden_states": _tensor(hidden_by_stage[layer_idx]),
                    "target_context": _tensor(target_context),
                    "position_cos": _tensor(cos),
                    "position_sin": _tensor(sin),
                },
                "weights": {
                    "input_layernorm_weight": _tensor(layer.input_layernorm.weight),
                    "q_proj_weight": _tensor(layer.self_attn.q_proj.weight),
                    "k_proj_weight": _tensor(layer.self_attn.k_proj.weight),
                    "v_proj_weight": _tensor(layer.self_attn.v_proj.weight),
                    "o_proj_weight": _tensor(layer.self_attn.o_proj.weight),
                    "q_norm_weight": _tensor(layer.self_attn.q_norm.weight),
                    "k_norm_weight": _tensor(layer.self_attn.k_norm.weight),
                    "post_attention_layernorm_weight": _tensor(layer.post_attention_layernorm.weight),
                    "gate_proj_weight": _tensor(layer.mlp.gate_proj.weight),
                    "up_proj_weight": _tensor(layer.mlp.up_proj.weight),
                    "down_proj_weight": _tensor(layer.mlp.down_proj.weight),
                },
                "expected": {
                    "hidden_states": _tensor(hidden_by_stage[layer_idx + 1]),
                },
            },
        )

    _write_json(
        output_dir / "stage_post_decoder_fused.json",
        {
            **common,
            "stage": "post_decoder_fused",
            "mapping": "DFlash final norm, drafter lmhead/sampling, and host packet construction",
            "inputs": {
                "hidden_states": _tensor(hidden_by_stage[-1]),
                "anchor_position": 2,
            },
            "weights": {
                "final_norm_weight": _tensor(model.norm.weight),
                "target_lm_head_weight": _tensor(target_lm_head_weight),
            },
            "expected": {
                "final_hidden": _tensor(final_hidden),
                "draft_logits": _tensor(draft_logits),
                "draft_token_ids": draft_token_ids.tolist(),
                "host_packet": {
                    "type": "DRAFT_BLOCK_PROPOSAL",
                    "anchor_position": 2,
                    "token_ids": draft_token_ids.tolist()[0],
                    "positions": [3, 4, 5],
                },
            },
        },
    )

    _write_json(
        output_dir / "stage_combined_drafter.json",
        {
            **common,
            "stage": "combined_drafter",
            "mapping": "full DFlash drafter pipeline: pre-decoder, decoder layers, and post-decoder packet emit",
            "stage_fixture_paths": [
                "stage_pre_decoder_fused.json",
                "stage_decoder_layer_0.json",
                "stage_decoder_layer_1.json",
                "stage_post_decoder_fused.json",
            ],
            "device_size_choice": {
                "smallest_viable": "one_blackhole_galaxy",
                "stage_slots": 4,
                "stage_shape": "4x2",
                "mapped_stage_count": 4,
                "fits_one_galaxy": True,
                "reason": "pre-decoder fused + two drafter decoder layers + post-decoder fused equals four stages",
            },
            "expected": {
                "final_hidden": _tensor(final_hidden),
                "draft_logits": _tensor(draft_logits),
                "draft_token_ids": draft_token_ids.tolist(),
                "host_packet": {
                    "type": "DRAFT_BLOCK_PROPOSAL",
                    "anchor_position": 2,
                    "token_ids": draft_token_ids.tolist()[0],
                    "positions": [3, 4, 5],
                },
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tiny DFlash block-diffusion stage fixtures.")
    parser.add_argument("--golden-path", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    build_fixtures(args.golden_path, args.output_dir)


if __name__ == "__main__":
    main()
