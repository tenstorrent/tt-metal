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

from loguru import logger
from transformers import AutoTokenizer

import ttnn
from conftest import bh_2d_mesh_device_context
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.runner import GenerationResult, run_generation
from models.demos.deepseek_v3_b1.demo.runtime import TokenCodec, create_model
from models.demos.deepseek_v3_b1.prepare_weights import (
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    load_dense_decoder_layer,
    load_embedding_weights,
    load_lm_head_weights,
    load_moe_decoder_layer,
    load_moe_routed_experts,
)

DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-V3"
FIRST_K_DENSE_REPLACE = 3

EXPECTED_PIPELINE_STAGE_MESH_SHAPE = (4, 2)


@contextlib.contextmanager
def open_mesh_device():
    """Open mesh device using bh_2d_mesh_device_context."""
    if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
        os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = "30000"
    device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}
    logger.info("Opening mesh device...")
    with bh_2d_mesh_device_context(device_params) as mesh_device:
        mesh_shape = (mesh_device.shape[0], mesh_device.shape[1])
        assert (
            mesh_shape == EXPECTED_PIPELINE_STAGE_MESH_SHAPE
        ), f"Demo requires a {EXPECTED_PIPELINE_STAGE_MESH_SHAPE[0]}x{EXPECTED_PIPELINE_STAGE_MESH_SHAPE[1]} mesh; got {mesh_shape[0]}x{mesh_shape[1]}"
        logger.info(f"Mesh device opened (id={mesh_device.get_system_mesh_id()}, shape={mesh_device.shape})")
        yield mesh_device


def decoder_layer_id_from_mesh_id(mesh_id: int) -> int:
    """
    Layer ID is the index of the layer in the original model (i.e. layer 0-3 are dense layers, layer 4-60 are MoE layers)
    The mesh ID is the pipeline stage index (0 is embedding, 1-3 are dense layers, 4-61 are MoE, and 62 is LM head + sampling)
    """
    assert mesh_id > 0 and mesh_id <= 61, f"Cannot get layer ID from mesh ID: {mesh_id}"
    return mesh_id - 1


SYSTEM_MESH_ID_EMBEDDING = 0
SYSTEM_MESH_ID_LM_HEAD = 62


def load_weights_from_cache(
    cache_path: Path,
    mesh_device: ttnn.MeshDevice,
    layer_offset: int,
) -> (
    DeepSeekV3EmbeddingLayerWeights | DeepSeekV3DenseLayerWeights | DeepSeekV3MoELayerWeights | DeepSeekV3LMHeadWeights
):
    """Load weights from cache (embedding, decoder layer, or lm_head)."""
    mesh_id = mesh_device.get_system_mesh_id() + layer_offset
    assert (
        mesh_id >= SYSTEM_MESH_ID_EMBEDDING and mesh_id <= SYSTEM_MESH_ID_LM_HEAD
    ), f"Mesh ID must be between {SYSTEM_MESH_ID_EMBEDDING} and {SYSTEM_MESH_ID_LM_HEAD} but got {mesh_id}"
    if mesh_id == SYSTEM_MESH_ID_EMBEDDING:
        logger.info("Loading embedding weights from cache")
        return load_embedding_weights(cache_path, mesh_device)
    elif mesh_id == SYSTEM_MESH_ID_LM_HEAD:
        logger.info("Loading LM head weights from cache")
        return load_lm_head_weights(cache_path, mesh_device)
    else:
        layer_id = decoder_layer_id_from_mesh_id(mesh_id)
        is_moe = layer_id >= FIRST_K_DENSE_REPLACE
        logger.info(f"Loading {'moe' if is_moe else 'dense'} layer weights from cache")
        if is_moe:
            with ttnn.device.setup_fast_dispatch(mesh_device):
                preloaded_experts = load_moe_routed_experts(cache_path, mesh_device, layer_id)
            return load_moe_decoder_layer(cache_path, mesh_device, layer_id, preloaded_routed_experts=preloaded_experts)
        return load_dense_decoder_layer(cache_path, mesh_device, layer_id)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("DeepSeek-V3-B1 Demo on TT-NN")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Number of decode steps to run")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="HF tokenizer id or local tokenizer path",
    )
    parser.add_argument(
        "--loopback-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use HostInterface loopback path (recommended for current B1 bring-up)",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        required=True,
        help="Path to the weight cache directory (contains layer_NNN/ subdirs)",
    )
    parser.add_argument(
        "--layer-id-offset",
        type=int,
        default=0,
        help="Layer ID offset (default 0)",
    )
    return parser


def load_tokenizer(tokenizer_name_or_path: str):
    return AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)


def run_demo(
    *,
    prompt: str,
    max_new_tokens: int,
    tokenizer_name_or_path: str,
    loopback_mode: bool,
    cache_path: Path,
    layer_id_offset: int = 0,
    output_stream: TextIO,
) -> GenerationResult:
    logger.info(
        "Starting DeepSeek V3 B1 demo (max_new_tokens={}, loopback_mode={}, layer_id_offset={})",
        max_new_tokens,
        loopback_mode,
        layer_id_offset,
    )
    if not is_slow_dispatch():
        raise RuntimeError(
            "DeepSeek V3 B1 demo requires slow dispatch mode. Set TT_METAL_SLOW_DISPATCH_MODE=1 and rerun."
        )

    logger.info("Loading tokenizer: {}", tokenizer_name_or_path)
    tokenizer = load_tokenizer(tokenizer_name_or_path)
    token_codec = TokenCodec(batch_size=1)
    is_first_decode_chunk = True

    def write_text(text: str) -> None:
        nonlocal is_first_decode_chunk
        if is_first_decode_chunk:
            if prompt:
                output_stream.write(prompt)
            is_first_decode_chunk = False
        output_stream.write(text)
        output_stream.flush()

    with open_mesh_device() as mesh_device:
        weights = load_weights_from_cache(cache_path, mesh_device, layer_id_offset)
        logger.info("Weights loaded!")
        return None

        logger.info("Creating DeepSeekV3 model")
        model = create_model(mesh_device=mesh_device, batch_size=1, loopback_mode=loopback_mode)
        logger.info("Running prefill + decode")
        result = run_generation(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            make_input_tensor=token_codec.make_input,
            extract_token_id=token_codec.extract_token_id,
            write_text=write_text,
        )
        logger.info(
            "Generation complete (prompt_tokens={}, generated_tokens={})",
            len(result.prompt_token_ids),
            len(result.generated_token_ids),
        )
        return result


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    run_demo(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        tokenizer_name_or_path=args.tokenizer,
        loopback_mode=args.loopback_mode,
        cache_path=args.cache_path,
        layer_id_offset=args.layer_id_offset,
        output_stream=sys.stdout,
    )
    print(file=sys.stdout, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
