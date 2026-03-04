# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import TextIO

import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from conftest import bh_2d_mesh_device_context
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.pipeline import (
    create_fabric_router_config,
    create_pipeline_configuration_from_num_procs,
)
from models.demos.deepseek_v3_b1.demo.weight_provider import (
    CacheWeightProvider,
    SyntheticWeightProvider,
    WeightProvider,
)
from models.demos.deepseek_v3_b1.model import TOKEN_ID_BYTES, DeepSeekV3, page_size_bytes, to_padded_input

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
    output_stream: TextIO,
) -> None:
    """Run the pod pipeline. Requires 4 or 16 distributed processes."""
    iterations = max_new_tokens
    logger.info(
        "Starting DeepSeek V3 B1 demo pod pipeline (iterations={}, weights={}, lm_head_fp32={}, lm_head_persistent_mode={})",
        iterations,
        "real" if use_real_weights else "synthetic",
        lm_head_fp32_dest_acc_en,
        lm_head_persistent_mode,
    )
    if not is_slow_dispatch():
        raise RuntimeError(
            "DeepSeek V3 B1 demo requires slow dispatch mode. Set TT_METAL_SLOW_DISPATCH_MODE=1 and rerun."
        )

    with open_mesh_device() as mesh_device:
        num_procs = int(ttnn.distributed_context_get_size())
        if num_procs not in (4, 16, 64):
            raise RuntimeError(f"Pod pipeline requires 4 or 16 distributed processes; got {num_procs}")
        ttnn.enable_asynchronous_slow_dispatch(mesh_device)

        # Each host loads/creates only the weights for its stage via the provider.
        provider: WeightProvider = CacheWeightProvider(cache_path) if use_real_weights else SyntheticWeightProvider()
        config = create_pipeline_configuration_from_num_procs(
            num_procs,
            provider,
            lm_head_fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode=lm_head_persistent_mode,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
        )
        assert (
            config.num_stages == num_procs
        ), f"Pipeline configuration has {config.num_stages} stages but {num_procs} processes"

        logger.info(f"Building pipeline")
        pipeline = config.build_pipeline(mesh_device)

        logger.info(f"Setting up and running pipeline")
        pipeline.setup_and_run()

        if pipeline.my_mesh_id == 0:
            # Prefill + decode pattern per model.py: prefill(prompt) -> sample y0; decode_step(y_t) -> sample y_{t+1}.
            tokenizer = load_tokenizer(tokenizer_name_or_path)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            logger.debug(f"Encoded prompt: {prompt_ids}")
            if not prompt_ids:
                prompt_ids = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0]
            page_size_datums = page_size_bytes(1) // TOKEN_ID_BYTES
            prompt_token_tensors = [
                to_padded_input(
                    torch.tensor([[tid]], dtype=torch.int32),
                    batch_size=1,
                    page_size_datums=page_size_datums,
                )
                for tid in prompt_ids
            ]
            model = DeepSeekV3(
                write_fn=pipeline.write_token,
                read_fn=pipeline.read_output,
                batch_size=1,
            )
            # Prefill: send prompt tokens; discard outputs for i < S-1; use last output to sample y0.
            logger.debug(f"Prefilling...")
            last_output = model.prefill(prompt_token_tensors)
            next_token_id = int(ttnn.to_torch(last_output).to(torch.int32)[0, 0].item())
            generated = [next_token_id]
            logger.info(
                "Prefill done ({} prompt tokens); sampled y0: {}",
                len(prompt_ids),
                next_token_id,
            )
            # Generation loop: feed y[t], get output, sample y[t+1].
            for step in range(iterations - 1):
                output = model.decode_step(
                    torch.tensor([[next_token_id]], dtype=torch.int32),
                )
                next_token_id = int(ttnn.to_torch(output).to(torch.int32)[0, 0].item())
                generated.append(next_token_id)
                logger.info("Decode step {} output token: {}", step + 1, next_token_id)
            logger.info("Generated {} tokens total", len(generated))

        pipeline.barrier()
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
        output_stream=sys.stdout,
    )
    print(file=sys.stdout, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
