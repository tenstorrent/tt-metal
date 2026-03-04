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
    WeightProvider,
    create_fabric_router_config,
    create_pipeline_configuration_from_num_procs,
)
from models.demos.deepseek_v3_b1.demo.stage import TOKEN_PAGE_SIZE_BYTES
from models.demos.deepseek_v3_b1.prepare_weights import (
    NUM_ROUTED_EXPERTS,
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    load_dense_decoder_layer,
    load_embedding_weights,
    load_lm_head_weights,
    load_moe_decoder_layer,
    load_moe_routed_experts,
    prepare_dense_layer_weights,
    prepare_embedding_weights,
    prepare_lm_head_weights,
    prepare_moe_layer_weights,
)

DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-V3"
FIRST_K_DENSE_REPLACE = 3


def _layer_key(layer_id: int, suffix: str) -> str:
    """State dict key under model.layers.{layer_id}."""
    return f"model.layers.{layer_id}.{suffix}"


def _build_synthetic_moe_state_dict(
    layer_id: int,
    *,
    num_routed_experts: int = NUM_ROUTED_EXPERTS,
) -> dict[str, torch.Tensor]:
    """Build a synthetic MoE layer state dict with HF tensor shapes (randn for weights, ones for norms)."""
    state_dict: dict[str, torch.Tensor] = {}
    dtype = torch.bfloat16

    # Attention weights (HF shapes)
    state_dict[_layer_key(layer_id, "self_attn.q_a_proj.weight")] = torch.randn(1536, 7168, dtype=dtype)
    state_dict[_layer_key(layer_id, "self_attn.q_b_proj.weight")] = torch.randn(24576, 1536, dtype=dtype)
    state_dict[_layer_key(layer_id, "self_attn.kv_a_proj_with_mqa.weight")] = torch.randn(576, 7168, dtype=dtype)
    state_dict[_layer_key(layer_id, "self_attn.kv_b_proj.weight")] = torch.randn(32768, 512, dtype=dtype)
    state_dict[_layer_key(layer_id, "self_attn.o_proj.weight")] = torch.randn(7168, 16384, dtype=dtype)

    # Norms (ones per plan)
    state_dict[_layer_key(layer_id, "input_layernorm.weight")] = torch.ones(7168, dtype=dtype)
    state_dict[_layer_key(layer_id, "self_attn.q_a_layernorm.weight")] = torch.ones(1536, dtype=dtype)
    state_dict[_layer_key(layer_id, "self_attn.kv_a_layernorm.weight")] = torch.ones(512, dtype=dtype)
    state_dict[_layer_key(layer_id, "post_attention_layernorm.weight")] = torch.ones(7168, dtype=dtype)

    # MoE gate
    state_dict[_layer_key(layer_id, "mlp.gate.weight")] = torch.randn(256, 7168, dtype=dtype)
    state_dict[_layer_key(layer_id, "mlp.gate.e_score_correction_bias")] = torch.randn(256, dtype=dtype)

    # Shared experts
    state_dict[_layer_key(layer_id, "mlp.shared_experts.gate_proj.weight")] = torch.randn(2048, 7168, dtype=dtype)
    state_dict[_layer_key(layer_id, "mlp.shared_experts.up_proj.weight")] = torch.randn(2048, 7168, dtype=dtype)
    state_dict[_layer_key(layer_id, "mlp.shared_experts.down_proj.weight")] = torch.randn(7168, 2048, dtype=dtype)

    # Routed experts
    for e in range(num_routed_experts):
        state_dict[_layer_key(layer_id, f"mlp.experts.{e}.gate_proj.weight")] = torch.randn(2048, 7168, dtype=dtype)
        state_dict[_layer_key(layer_id, f"mlp.experts.{e}.up_proj.weight")] = torch.randn(2048, 7168, dtype=dtype)
        state_dict[_layer_key(layer_id, f"mlp.experts.{e}.down_proj.weight")] = torch.randn(7168, 2048, dtype=dtype)

    return state_dict


def _build_synthetic_dense_state_dict(layer_id: int) -> dict[str, torch.Tensor]:
    """Build a synthetic dense layer state dict with HF tensor shapes (randn for weights, ones for norms)."""
    state_dict: dict[str, torch.Tensor] = {}
    dtype = torch.bfloat16

    # Attention weights (HF shapes)
    state_dict[_layer_key(layer_id, "self_attn.q_a_proj.weight")] = torch.randn(1536, 7168, dtype=dtype)
    state_dict[_layer_key(layer_id, "self_attn.q_b_proj.weight")] = torch.randn(24576, 1536, dtype=dtype)
    state_dict[_layer_key(layer_id, "self_attn.kv_a_proj_with_mqa.weight")] = torch.randn(576, 7168, dtype=dtype)
    state_dict[_layer_key(layer_id, "self_attn.kv_b_proj.weight")] = torch.randn(32768, 512, dtype=dtype)
    state_dict[_layer_key(layer_id, "self_attn.o_proj.weight")] = torch.randn(7168, 16384, dtype=dtype)

    # Norms (ones per plan)
    state_dict[_layer_key(layer_id, "input_layernorm.weight")] = torch.ones(7168, dtype=dtype)
    state_dict[_layer_key(layer_id, "self_attn.q_a_layernorm.weight")] = torch.ones(1536, dtype=dtype)
    state_dict[_layer_key(layer_id, "self_attn.kv_a_layernorm.weight")] = torch.ones(512, dtype=dtype)
    state_dict[_layer_key(layer_id, "post_attention_layernorm.weight")] = torch.ones(7168, dtype=dtype)

    # Single MLP (used for both shared and routed in dense)
    state_dict[_layer_key(layer_id, "mlp.gate_proj.weight")] = torch.randn(2048, 7168, dtype=dtype)
    state_dict[_layer_key(layer_id, "mlp.up_proj.weight")] = torch.randn(2048, 7168, dtype=dtype)
    state_dict[_layer_key(layer_id, "mlp.down_proj.weight")] = torch.randn(7168, 2048, dtype=dtype)

    return state_dict


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


def decoder_layer_id_from_mesh_id(mesh_id: int) -> int:
    """
    Layer ID is the index of the layer in the original model (i.e. layer 0-3 are dense layers, layer 4-60 are MoE layers)
    The mesh ID is the pipeline stage index (0 is embedding, 1-3 are dense layers, 4-61 are MoE, and 62 is LM head + sampling)
    """
    assert mesh_id > 0 and mesh_id <= 61, f"Cannot get layer ID from mesh ID: {mesh_id}"
    return mesh_id - 1


SYSTEM_MESH_ID_EMBEDDING = 0
SYSTEM_MESH_ID_LM_HEAD = 62


class CacheWeightProvider:
    """Load embedding and LM head weights from cache; each host loads only what its stage needs."""

    def __init__(self, cache_path: Path) -> None:
        assert cache_path.exists(), f"Cache path does not exist: {cache_path}"
        assert cache_path.is_dir(), f"Cache path is not a directory: {cache_path}"
        self._path = cache_path

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        return load_embedding_weights(self._path, device)

    def load_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        return load_lm_head_weights(self._path, device)

    def load_moe_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3MoELayerWeights:
        with ttnn.device.setup_fast_dispatch(device):
            preloaded_experts = load_moe_routed_experts(self._path, device, layer_id)
        return load_moe_decoder_layer(self._path, device, layer_id, preloaded_routed_experts=preloaded_experts)

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        return load_dense_decoder_layer(self._path, device, layer_id)


class SyntheticWeightProvider:
    """Create deterministic synthetic embedding and LM head weights in place (no cache)."""

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        emb_w = torch.zeros((129280, 7168), dtype=torch.bfloat16)
        emb_w[torch.arange(129280), torch.arange(129280, dtype=torch.int64) % 7168] = 1
        return prepare_embedding_weights({"model.embed_tokens.weight": emb_w}, device, move_to_device=True)

    def load_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        lm_w = torch.full((129280, 7168), -1.0, dtype=torch.bfloat16)
        lm_w[torch.arange(7168, dtype=torch.int64) % 16160, torch.arange(7168)] = 1
        return prepare_lm_head_weights(
            {
                "lm_head.weight": lm_w,
                "model.norm.weight": torch.ones(7168, dtype=torch.bfloat16),
            },
            device,
            move_to_device=True,
        )

    def load_moe_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3MoELayerWeights:
        from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights

        sd = _build_synthetic_moe_state_dict(layer_id, num_routed_experts=NUM_ROUTED_EXPERTS)
        bdw = BlitzDecodeWeights(device)
        return prepare_moe_layer_weights(bdw, sd, layer_id, num_routed_experts=NUM_ROUTED_EXPERTS)

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights

        sd = _build_synthetic_dense_state_dict(layer_id)
        bdw = BlitzDecodeWeights(device)
        return prepare_dense_layer_weights(bdw, sd, layer_id, move_to_device=True)


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
    parser = argparse.ArgumentParser("DeepSeek-V3-B1 Demo on TT-NN (pod pipeline)")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text (for future real decode loop)")
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
        help="Path to the weight cache directory (for future real weights)",
    )
    parser.add_argument(
        "--layer-id-offset",
        type=int,
        default=0,
        help="Layer ID offset (for future multi-pod offset)",
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
    return parser


def load_tokenizer(tokenizer_name_or_path: str):
    return AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)


def run_demo(
    *,
    prompt: str,
    max_new_tokens: int,
    tokenizer_name_or_path: str,
    cache_path: Path,
    layer_id_offset: int = 0,
    fp32: bool = True,
    persistent_mode: bool = True,
    output_stream: TextIO,
) -> None:
    """Run the pod pipeline (synthetic weights). Requires 4 or 16 distributed processes."""
    iterations = max_new_tokens
    logger.info(
        "Starting DeepSeek V3 B1 demo pod pipeline (iterations={}, layer_id_offset={}, fp32={}, persistent_mode={})",
        iterations,
        layer_id_offset,
        fp32,
        persistent_mode,
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
        provider: WeightProvider = SyntheticWeightProvider()
        config = create_pipeline_configuration_from_num_procs(
            num_procs,
            provider,
            fp32_dest_acc_en=fp32,
            persistent_mode=persistent_mode,
        )

        logger.info(f"Building pipeline")
        pipeline = config.build_pipeline(mesh_device)

        logger.info(f"Setting up and running pipeline")
        pipeline.setup_and_run()

        if pipeline.my_mesh_id == 0:
            for iteration in range(iterations):
                logger.info(f"Writing token for iteration {iteration}")
                torch_token = torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32)
                torch_token[0, 0] = iteration
                token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                output_tensor = ttnn.from_torch(
                    torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                pipeline.write_token(token_tensor)
                pipeline.read_output(output_tensor)
                got = ttnn.to_torch(output_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
                logger.info("Iteration {} output token: {}", iteration, got.item())

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
        layer_id_offset=args.layer_id_offset,
        fp32=args.fp32,
        persistent_mode=args.persistent_mode,
        output_stream=sys.stdout,
    )
    print(file=sys.stdout, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
