# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

from loguru import logger
from transformers import AutoTokenizer

import ttnn
from conftest import bh_2d_mesh_device_context
from models.demos.deepseek_v3_b1.demo.model_pipeline import ModelPipeline
from models.demos.deepseek_v3_b1.demo.pipeline import create_fabric_router_config

DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-V3"


def _fabric_config_for_num_procs(num_procs: int):
    """Infer fabric config from process count: 4 → FABRIC_2D, 16 → FABRIC_2D_TORUS_Y."""
    if num_procs == 4:
        return ttnn.FabricConfig.FABRIC_2D
    if num_procs == 16:
        return ttnn.FabricConfig.FABRIC_2D_TORUS_Y
    if num_procs == 64:
        return ttnn.FabricConfig.FABRIC_2D_TORUS_Y
    raise ValueError(f"Unsupported num_procs for fabric config: {num_procs} (expected 4, 16, or 64)")


@contextlib.contextmanager
def open_mesh_device():
    """Open mesh device using bh_2d_mesh_device_context (pod pipeline settings)."""
    if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
        os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = "30000"
    num_procs = int(ttnn.distributed_context_get_size())
    device_params = {
        "fabric_config": _fabric_config_for_num_procs(num_procs),
        "fabric_router_config": create_fabric_router_config(15232),
        "trace_region_size": 573440,
    }
    logger.info("Opening mesh device...")
    with bh_2d_mesh_device_context(device_params) as mesh_device:
        logger.info(
            "Mesh device opened (id={}, shape={})",
            mesh_device.get_system_mesh_id(),
            mesh_device.shape,
        )
        yield mesh_device


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("DeepSeek-V3-B1 Demo on TT-NN (pod pipeline)")
    parser.add_argument("--prompt", type=str, default="Hello, world!", help="Prompt text (for future real decode loop)")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Number of pipeline token iterations",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="HF tokenizer id or local tokenizer path",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        required=True,
        help="Path to the weight cache directory (required for --weights real)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        choices=("synthetic", "real"),
        default="synthetic",
        help="Use synthetic or real (cached) weights (default: synthetic)",
    )
    parser.add_argument(
        "--fp32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FP32 destination accumulator for LMHead sampling",
    )
    parser.add_argument(
        "--persistent-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use persistent mode for LMHead sampling kernel",
    )
    parser.add_argument(
        "--dense-layer-id-override",
        type=int,
        default=None,
        metavar="ID",
        help="Force all dense stages to use this layer id (e.g. 0); default: use 0,1,2",
    )
    parser.add_argument(
        "--moe-layer-id-override",
        type=int,
        default=None,
        metavar="ID",
        help="Force all MoE stages to use this layer id (e.g. 3); default: use stage-dependent layer ids",
    )
    return parser


def load_tokenizer(tokenizer_name_or_path: str):
    return AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)


def run_demo(
    *,
    prompt: str,
    max_new_tokens: int,
    tokenizer_name_or_path: str,
    cache_path: Path,
    use_real_weights: bool = False,
    lm_head_fp32_dest_acc_en: bool = True,
    lm_head_persistent_mode: bool = True,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
) -> None:
    """Run the pod pipeline. Requires 4, 16, or 64 distributed processes."""
    iterations = max_new_tokens
    logger.info(f"Starting DeepSeek V3 B1 demo (iterations={iterations})")

    with open_mesh_device() as mesh_device:
        # Initialize model pipeline
        model_pipeline = ModelPipeline(
            mesh_device=mesh_device,
            cache_path=cache_path,
            use_real_weights=use_real_weights,
            lm_head_fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode=lm_head_persistent_mode,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
        )

        my_mesh_id = mesh_device.get_system_mesh_id()
        if my_mesh_id == 0:
            tokenizer = load_tokenizer(tokenizer_name_or_path)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            logger.debug(f"Encoded prompt: {prompt_ids}")
            if not prompt_ids:
                prompt_ids = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0]

            generated = model_pipeline.run_inference(
                prompt_token_ids=prompt_ids,
                max_new_tokens=iterations,
                on_token=lambda tid: logger.info("Generated token: {}", tid),
                eos_token_id=tokenizer.eos_token_id,
            )
            logger.info("Generated {} tokens total", len(generated))
            logger.info("Output: {}", tokenizer.decode(generated, skip_special_tokens=True))

        model_pipeline.pipeline.barrier()
    logger.info("Pod pipeline complete")


def main(argv: list[str] | None = None) -> int:
    ttnn.init_distributed_context()
    parser = create_parser()
    args = parser.parse_args(argv)

    run_demo(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        tokenizer_name_or_path=args.tokenizer,
        cache_path=args.cache_path,
        use_real_weights=(args.weights == "real"),
        lm_head_fp32_dest_acc_en=args.fp32,
        lm_head_persistent_mode=args.persistent_mode,
        dense_layer_id_override=args.dense_layer_id_override,
        moe_layer_id_override=args.moe_layer_id_override,
    )
    print(file=sys.stdout, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
