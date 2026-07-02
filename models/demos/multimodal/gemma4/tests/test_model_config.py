# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test Gemma 4 model config initialization on TT device."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from loguru import logger

import ttnn


def test_model_config():
    logger.info("Opening mesh device...")
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    logger.info(f"Mesh device opened: {mesh_device.shape}")

    try:
        from models.demos.multimodal.gemma4.tt.model_config import ModelArgs

        logger.info("Creating Gemma 4 ModelArgs...")
        model_args = ModelArgs(
            mesh_device=mesh_device,
            instruct=True,
            max_batch_size=1,
            max_seq_len=2048,  # Start small for testing
        )

        logger.info(f"Model: {model_args.model_name}")
        logger.info(f"Layers: {model_args.n_layers}")
        logger.info(f"Hidden dim: {model_args.dim}")
        logger.info(f"Heads: {model_args.n_heads}")
        logger.info(f"KV heads (sliding): {model_args.n_kv_heads}")
        logger.info(f"Head dim (sliding): {model_args.head_dim}")
        logger.info(f"Global head dim: {model_args.global_head_dim}")
        logger.info(f"Global KV heads: {model_args.num_global_kv_heads}")
        logger.info(f"Vocab size: {model_args.vocab_size}")
        logger.info(f"FFN dim: {model_args.hidden_dim}")
        logger.info(f"Embed scale: {model_args.embed_scale}")
        logger.info(f"Sliding window: {model_args.sliding_window}")

        # Test per-layer config
        for layer_idx in [0, 5, 11, 59]:
            is_sliding = model_args.is_layer_sliding(layer_idx)
            hd = model_args.get_layer_head_dim(layer_idx)
            nkv = model_args.get_layer_n_kv_heads(layer_idx)
            has_v = model_args.get_layer_has_v_proj(layer_idx)
            qkv = model_args.get_layer_qkv_size(layer_idx)
            logger.info(
                f"Layer {layer_idx}: sliding={is_sliding} head_dim={hd} n_kv_heads={nkv} has_v_proj={has_v} qkv_size={qkv}"
            )

        # Verify assertions
        assert model_args.n_layers == 60
        assert model_args.dim == 5376
        assert model_args.n_heads == 32
        assert model_args.head_dim == 256
        assert model_args.global_head_dim == 512
        assert model_args.num_global_kv_heads == 4
        assert model_args.is_layer_sliding(0) == True
        assert model_args.is_layer_sliding(5) == False
        assert model_args.get_layer_head_dim(0) == 256
        assert model_args.get_layer_head_dim(5) == 512
        assert model_args.get_layer_n_kv_heads(0) == 16
        assert model_args.get_layer_n_kv_heads(5) == 4
        assert model_args.get_layer_has_v_proj(0) == True
        assert model_args.get_layer_has_v_proj(5) == False

        logger.info("All assertions passed!")
        logger.info("ModelArgs creation SUCCESS")

    finally:
        ttnn.close_mesh_device(mesh_device)
        logger.info("Mesh device closed")


if __name__ == "__main__":
    test_model_config()
