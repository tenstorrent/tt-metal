# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.demo.demo import _default_mesh_shape, validate_model_path
from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer


def build_reference_tt(
    model_path: str,
    cache_dir: str,
    prompt: str,
    max_new_tokens: int,
    out_path: str,
) -> None:
    model_path = str(model_path)
    cache_dir = str(cache_dir)
    out_path = Path(out_path)

    validate_model_path(
        model_path,
        require_safetensors=True,
        require_tokenizer=True,
    )

    mesh_shape = _default_mesh_shape()
    logger.info("Setting fabric config to FABRIC_1D for reference generation")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    logger.info(f"Opening mesh device with shape {mesh_shape}")
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    tokenizer = load_tokenizer(model_path)

    gen = None
    gen2 = None

    try:
        gen = DeepseekGenerator(
            mesh_device=mesh_device,
            model_path=Path(model_path),
            cache_dir=Path(cache_dir),
            tokenizer=tokenizer,
            random_weights=False,
        )

        logger.info(f"[refgen] Running canonical TT generation for prompt: {prompt!r}")
        generations, _stats = gen.generate(
            prompts=[prompt],
            max_new_tokens=max_new_tokens,
            teacher_forcing=None,
            early_print_first_user=False,
            repeat_batches=1,
            validate_against_ref=False,
            reference_texts=None,
        )

        # 'generations[0]' is the decode-phase tokens (no prompt)
        continuation_tokens = generations[0]

        # Recreate the encoded prompt exactly as used in generate()
        encoded_prompt = gen._encode_prompt(prompt)  # same internal helper
        reference_tokens = torch.tensor(
            encoded_prompt + continuation_tokens,
            dtype=torch.long,
        ).unsqueeze(
            0
        )  # [1, T]
        T = reference_tokens.shape[1]
        logger.info(f"[refgen] Reference sequence length T={T}")

        # We don't reuse this instance for logits to avoid any cache side effects
        gen.cleanup_all()
        gen = None

        gen2 = DeepseekGenerator(
            mesh_device=mesh_device,
            model_path=Path(model_path),
            cache_dir=Path(cache_dir),
            tokenizer=tokenizer,
            random_weights=False,
        )

        # Prepare prefill config (same as demo)
        gen2._prepare_run_configs("prefill")

        # Use full reference sequence as input; RowPipelinedModel.forward_prefill
        # returns logits for all positions: [1, 1, T, V]
        tokens_1d = reference_tokens[0]  # [T]
        logits = gen2._prefill(tokens_1d, user_id=0)  # [1, 1, T, V]
        logits = logits[0, 0]  # [T, V]

        # Top-5 over vocab for each position
        top5_tokens = torch.topk(logits, k=5, dim=-1).indices.to(torch.long)  # [T, 5]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "reference_tokens": reference_tokens.to(torch.long),  # [1, T]
                "top5_tokens": top5_tokens,  # [T, 5]
            },
            out_path,
        )
        logger.info(f"[refgen] Saved reference file to {out_path}")

    finally:
        # Cleanup generators if they exist
        try:
            if gen is not None:
                gen.cleanup_all()
        except Exception as e:
            logger.warning(f"Cleanup of first generator failed: {e}")
        try:
            if gen2 is not None:
                gen2.cleanup_all()
        except Exception as e:
            logger.warning(f"Cleanup of second generator failed: {e}")

        # Close mesh device(s)
        try:
            for submesh in mesh_device.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(mesh_device)
        except Exception as e:
            logger.warning(f"Failed to close mesh device cleanly: {e}")

        # Reset fabric config
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser("Generate DeepSeek-V3 reference .refpt using TT demo pipeline")
    ap.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Same --model-path you use for demo.py (HF DeepSeek-V3 directory)",
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        required=True,
        help="Cache dir used by the existing DeepSeek-V3 demo",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for .refpt, e.g. models/demos/deepseek_v3/reference/usb.refpt",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Max decode tokens for the canonical reference run",
    )
    ap.add_argument(
        "prompt",
        type=str,
        help="Prompt text (exactly what you would pass to demo.py)",
    )
    args = ap.parse_args()

    build_reference_tt(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
