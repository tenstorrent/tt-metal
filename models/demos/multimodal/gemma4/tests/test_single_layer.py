# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test loading a single Gemma 4 decoder layer on TT device."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from loguru import logger

import ttnn


def test_single_layer(layer_idx=0):
    """Test loading a single decoder layer (sliding or full attention)."""
    logger.info(f"Testing layer {layer_idx}")
    logger.info("Opening mesh device...")
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))

    try:
        from models.demos.multimodal.gemma4.tt.model_config import ModelArgs

        logger.info("Creating ModelArgs...")
        model_args = ModelArgs(
            mesh_device=mesh_device,
            instruct=True,
            max_batch_size=1,
            max_seq_len=2048,
        )

        is_sliding = model_args.is_layer_sliding(layer_idx)
        logger.info(f"Layer {layer_idx} is_sliding={is_sliding}")
        logger.info(f"  head_dim={model_args.get_layer_head_dim(layer_idx)}")
        logger.info(f"  n_kv_heads={model_args.get_layer_n_kv_heads(layer_idx)}")
        logger.info(f"  has_v_proj={model_args.get_layer_has_v_proj(layer_idx)}")

        # Load only the needed layer weights
        logger.info("Loading state dict (this loads full model - takes time)...")
        t0 = time.time()
        state_dict = model_args.load_state_dict()
        t1 = time.time()
        logger.info(f"State dict loaded in {t1-t0:.1f}s, {len(state_dict)} keys")

        # Verify keys for this layer
        layer_keys = [k for k in state_dict if f"layers.{layer_idx}." in k]
        logger.info(f"Layer {layer_idx} keys ({len(layer_keys)}):")
        for k in sorted(layer_keys):
            logger.info(f"  {k}: {state_dict[k].shape}")

        # Try creating the decoder block
        from models.demos.multimodal.gemma4.tt.gemma4_decoder import Gemma4TransformerBlock
        from models.tt_transformers.tt.ccl import TT_CCL
        from models.tt_transformers.tt.rope import RotarySetup

        tt_ccl = TT_CCL(mesh_device)

        # Setup RoPE
        rope_setup = RotarySetup(
            device=mesh_device,
            batch_size=model_args.max_batch_size,
            head_dim=model_args.head_dim,
            max_seq_len=model_args.max_seq_len,
            rope_theta=10000.0,
            use_qk_fused=False,
        )
        trans_mats_dict = rope_setup.get_both_trans_mats()

        logger.info(f"Creating Gemma4TransformerBlock for layer {layer_idx}...")
        t0 = time.time()
        decoder = Gemma4TransformerBlock(
            args=model_args,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
            layer_num=layer_idx,
            transformation_mats=trans_mats_dict,
        )
        t1 = time.time()
        logger.info(f"Decoder block created in {t1-t0:.1f}s")
        logger.info(f"  attention.is_sliding = {decoder.attention.is_sliding}")
        logger.info(f"  attention.head_dim = {decoder.attention.head_dim}")
        logger.info(f"  attention.n_kv_heads = {decoder.attention.n_kv_heads}")
        logger.info(f"  layer_scalar_value = {decoder.layer_scalar_value}")

        logger.info(f"SUCCESS: Layer {layer_idx} loaded on device!")

    except Exception as e:
        logger.error(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
    finally:
        ttnn.close_mesh_device(mesh_device)
        logger.info("Mesh device closed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0, help="Layer index to test")
    args = parser.parse_args()
    test_single_layer(args.layer)
