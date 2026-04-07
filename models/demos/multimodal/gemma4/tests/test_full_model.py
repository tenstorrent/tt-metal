# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test loading the full Gemma 4 31B IT model on TT device and running a simple decode step."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import torch
import ttnn
from loguru import logger


def test_full_model():
    logger.info("Opening mesh device with fabric...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))

    try:
        from models.demos.multimodal.gemma4.tt.model_config import ModelArgs
        from models.demos.multimodal.gemma4.tt.gemma4_model import Gemma4Transformer

        # For initial bring-up, test with limited layers (skip full attention layers for now)
        # Full attention layers have head_dim=512 which exceeds RoPE kernel limit of 256
        n_layers_override = int(os.environ.get("GEMMA4_N_LAYERS", "0"))

        logger.info("Creating ModelArgs...")
        model_args = ModelArgs(
            mesh_device=mesh_device,
            instruct=True,
            max_batch_size=1,
            max_seq_len=2048,
        )

        if n_layers_override > 0:
            logger.info(f"Overriding n_layers from {model_args.n_layers} to {n_layers_override}")
            model_args.n_layers = n_layers_override
        logger.info(f"Model: {model_args.model_name}, {model_args.n_layers} layers")

        logger.info("Loading state dict...")
        t0 = time.time()
        state_dict = model_args.load_state_dict()
        t1 = time.time()
        logger.info(f"State dict loaded in {t1-t0:.1f}s ({len(state_dict)} keys)")

        logger.info("Creating Gemma4Transformer (loading all 60 layers)...")
        t0 = time.time()
        model = Gemma4Transformer(
            args=model_args,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(ttnn.bfloat16),
        )
        t1 = time.time()
        logger.info(f"Model loaded in {t1-t0:.1f}s")

        # Test a simple decode step
        logger.info("Testing decode forward pass...")
        tokenizer = model_args.create_tokenizer()
        prompt = "Hello, I am"
        tokens = tokenizer.encode(prompt, return_tensors="pt")
        logger.info(f"Prompt: '{prompt}' -> {tokens.shape[1]} tokens")

        # Pad tokens to 256 (use minimal_matmul path which handles DRAM-sharded weights better)
        seq_len = tokens.shape[1]
        padded_len = max(256, ((seq_len + 127) // 128) * 128)
        padded_tokens = torch.nn.functional.pad(tokens, (0, padded_len - seq_len), value=0)
        logger.info(f"Padded from {seq_len} to {padded_len} tokens")

        # Prepare prefill inputs
        last_token_idx = seq_len - 1
        tt_inputs, rot_mats_global, rot_mats_local, _, _ = model.prepare_inputs_prefill(
            padded_tokens,
            start_pos=0,
            last_token_idx=last_token_idx,
        )

        # Run prefill
        logger.info("Running prefill...")
        t0 = time.time()
        get_last = (last_token_idx // 32) * 32
        tt_out = model.ttnn_prefill_forward(
            tt_inputs,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            get_last_token=get_last,
        )
        t1 = time.time()
        logger.info(f"Prefill done in {t1-t0:.1f}s")

        # Get output logits - move to host first
        tt_out_host = ttnn.from_device(tt_out)
        logits = model.process_output_prefill(tt_out_host, last_token_idx % 32)
        next_token = torch.argmax(logits).item()
        next_text = tokenizer.decode([next_token])
        logger.info(f"Next token: {next_token} = '{next_text}'")
        logger.info("FULL MODEL TEST SUCCESS!")

    except Exception as e:
        logger.error(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ttnn.close_mesh_device(mesh_device)
        logger.info("Mesh device closed")


if __name__ == "__main__":
    test_full_model()
