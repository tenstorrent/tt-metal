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
    DeepSeekV3Weights,
    MoERoutedExpertWeights,
    load_layer,
    load_moe_routed_experts_from_cache,
)

DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-V3"
FIRST_K_DENSE_REPLACE = 3
# Each layer is loaded onto a separate (4, 2) submesh; full mesh is (4, 2*num_layers).
SUBMESH_SHAPE = (4, 2)


@contextlib.contextmanager
def enable_fast_dispatch_mode(device):
    """Stub: will switch device to fast dispatch mode. Currently a no-op."""
    yield


@contextlib.contextmanager
def open_mesh_device_with_submeshes(num_layers: int):
    """Open mesh device using bh_2d_mesh_device_context and create (4, 2) submeshes.

    Yields (mesh_device, submeshes). Ensures at least num_layers submeshes exist.
    Same setup as generate_cache.py.
    """
    if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
        os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = "30000"
        logger.info("Set TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=30000 (fabric init may be slow)")
    device_params = {"fabric_config": ttnn.FabricConfig.FABRIC_2D}
    logger.info("Opening mesh device (via bh_2d_mesh_device_context)...")
    with bh_2d_mesh_device_context(device_params) as mesh_device:
        submeshes = mesh_device.create_submeshes(ttnn.MeshShape(*SUBMESH_SHAPE))
        if len(submeshes) < num_layers:
            raise RuntimeError(
                f"Mesh has {len(submeshes)} (4x2) submeshes but num_layers={num_layers}; "
                f"need at least {num_layers}*8 devices (e.g. 8 for 1 layer, 32 for 4 layers)"
            )
        logger.info("Mesh has {} (4x2) submeshes", len(submeshes))
        yield mesh_device, submeshes


def load_weights_from_cache(
    cache_path: Path,
    mesh_device: ttnn.MeshDevice,
    submeshes: list,
    num_layers: int,
) -> DeepSeekV3Weights:
    """Load weights from cache: routed experts in fast-dispatch phase, then all layers on their submesh."""
    # Phase 1: Fast dispatch -- load routed experts for MoE layers (each on its submesh)
    preloaded_experts: dict[int, MoERoutedExpertWeights] = {}
    with enable_fast_dispatch_mode(mesh_device):
        for layer_idx in range(FIRST_K_DENSE_REPLACE, num_layers):
            preloaded_experts[layer_idx] = load_moe_routed_experts_from_cache(
                cache_path, submeshes[layer_idx], layer_idx
            )

    # Phase 2: Slow dispatch -- load each layer onto its (4, 2) submesh
    layers = []
    for layer_idx in range(num_layers):
        layer = load_layer(
            cache_path,
            submeshes[layer_idx],
            layer_idx,
            preloaded_routed_experts=preloaded_experts.get(layer_idx),
        )
        layers.append(layer)
    weights = DeepSeekV3Weights(layers=layers)
    logger.info("Loaded {} layers from cache (one per submesh)", len(weights.layers))
    return weights


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
        "--num-layers",
        type=int,
        default=1,
        help="Number of decoder layers to load from cache",
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
    num_layers: int,
    output_stream: TextIO,
) -> GenerationResult:
    logger.info(
        "Starting DeepSeek V3 B1 demo (max_new_tokens={}, loopback_mode={}, num_layers={})",
        max_new_tokens,
        loopback_mode,
        num_layers,
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

    with open_mesh_device_with_submeshes(num_layers) as (mesh_device, submeshes):
        weights = load_weights_from_cache(cache_path, mesh_device, submeshes, num_layers)
        breakpoint()

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
        num_layers=args.num_layers,
        output_stream=sys.stdout,
    )
    print(file=sys.stdout, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
