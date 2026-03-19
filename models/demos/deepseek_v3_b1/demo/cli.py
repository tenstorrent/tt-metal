# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Literal

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
        "worker_l1_size": 1431568,
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
        default=None,
        help="Path to the weight cache directory (required for --weights real)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local HuggingFace model dir with model.safetensors.index.json (required for --weights state_dict)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        choices=("synthetic", "real", "state_dict"),
        default="real",
        help="synthetic: random prepare path; real: load tensorbin cache; state_dict: HF safetensors + prepare path",
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
    weights_mode: Literal["synthetic", "real", "state_dict"] = "real",
    cache_path: Path | None = None,
    model_path: Path | None = None,
    lm_head_fp32_dest_acc_en: bool = True,
    lm_head_persistent_mode: bool = True,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
) -> None:
    """Run the pod pipeline. Requires 4, 16, or 64 distributed processes."""
    iterations = max_new_tokens
    logger.info(f"Starting DeepSeek V3 B1 demo (iterations={iterations})")

    with open_mesh_device() as mesh_device:
        model_pipeline = ModelPipeline(
            mesh_device=mesh_device,
            weights_mode=weights_mode,
            cache_path=cache_path,
            model_path=model_path,
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

            logger.info("Running inference on prompt with {} tokens", len(prompt_ids))
            generated_tokens = model_pipeline.run_inference(
                prompt_token_ids=prompt_ids,
                max_new_tokens=iterations,
                eos_token_id=tokenizer.eos_token_id,
                return_generated_tokens=True,
            )
            assert generated_tokens is not None
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            logger.info("Output ({} tokens): {}", len(generated_tokens), generated_text)

        model_pipeline.barrier()
    logger.info("Pod pipeline complete")


def main(argv: list[str] | None = None) -> int:
    ttnn.init_distributed_context()
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.weights == "real" and args.cache_path is None:
        parser.error("--cache-path is required when --weights real")
    if args.weights == "state_dict":
        if args.model_path is None:
            parser.error("--model-path is required when --weights state_dict")
        index_path = args.model_path / "model.safetensors.index.json"
        if not index_path.is_file():
            parser.error(f"--model-path must contain model.safetensors.index.json (missing {index_path})")

    run_demo(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        tokenizer_name_or_path=args.tokenizer,
        weights_mode=args.weights,
        cache_path=args.cache_path,
        model_path=args.model_path,
        lm_head_fp32_dest_acc_en=args.fp32,
        lm_head_persistent_mode=args.persistent_mode,
        dense_layer_id_override=args.dense_layer_id_override,
        moe_layer_id_override=args.moe_layer_id_override,
    )
    print(file=sys.stdout, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
