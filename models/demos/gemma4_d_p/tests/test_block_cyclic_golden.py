# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""256K block-cyclic prefill with rewinds, checked against the Gutenberg GPU KV trace."""

import json
import os
import random
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open
from transformers import AutoTokenizer

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.tt.common import create_tt_model
from models.demos.gemma4_d_p.tt.model import _cp_chunk_major_row_order
from models.demos.gemma4_d_p.tt.prefill_metadata import chunk_positions


def prefill_chunk(model, token_ids, actual_start, actual_end, pad_token_id=0):
    """Pack and prefill a request; return synchronized forward latency in milliseconds."""
    mesh_config = model.mesh_config
    positions = chunk_positions(actual_start, model.prefill_chunk_size, mesh_config.cp_degree).flatten()
    valid = positions < actual_end
    tokens = torch.full((1, model.prefill_chunk_size), pad_token_id, dtype=torch.int32)
    tokens[0, valid] = torch.tensor(token_ids, dtype=torch.int32)[positions[valid]]
    device_tokens = ttnn.from_torch(
        tokens,
        device=mesh_config.device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_config.shard_mapper(mesh_dims=(1, None)),
    )
    ttnn.synchronize_device(mesh_config.device)
    start = time.perf_counter()
    hidden_states = model.transform_and_embed_prefill_inputs_device(device_tokens)
    output = model(hidden_states, actual_start=actual_start, actual_end=actual_end)
    ttnn.synchronize_device(mesh_config.device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    output.deallocate(True)
    device_tokens.deallocate(True)
    return elapsed_ms


@pytest.mark.timeout(900)
@torch.no_grad()
def test_block_cyclic_prefill_256k():
    golden_dir = Path("/mnt/models/huggingface/gpu_traces/gemma4_d_p/gutenberg-135")
    metadata = json.loads((golden_dir / "metadata.json").read_text())
    model_path = os.getenv("HF_MODEL", metadata["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    context_len, chunk_size = 262144, 8192
    token_ids = tokenizer.encode((golden_dir / "input.txt").read_text())[:context_len]
    assert len(token_ids) == context_len
    assert token_ids == metadata["token_ids"], "Input tokenization differs from the GPU golden"

    router_config = ttnn.FabricRouterConfig()
    router_config.max_packet_payload_size_bytes = 8192
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, router_config=router_config)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), l1_small_size=16384)
    try:
        mesh_config = MeshConfig(mesh_device)
        _, model, caches, _ = create_tt_model(
            mesh_config, prefill_chunk_size=chunk_size, max_seq_len=context_len, model_path=model_path
        )
        rng = random.Random(42)
        actual_end = 0
        while actual_end < context_len:
            # Rewind up to 2K tokens, then extend the populated prefix.
            actual_start = max(0, actual_end - rng.randint(0, 2048)) // 32 * 32
            actual_end = min(context_len, actual_start + rng.randint(chunk_size // 2, chunk_size))
            elapsed_ms = prefill_chunk(model, token_ids, actual_start, actual_end, tokenizer.pad_token_id)
            logger.info("Prefill [{}, {}): synchronized forward {:.3f} ms", actual_start, actual_end, elapsed_ms)

        # Restore absolute token order in the final-layer KV cache.
        row_order = _cp_chunk_major_row_order(context_len, mesh_config.cp_degree, chunk_size).argsort()
        composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_config.mesh_shape, dims=(2, 1))
        layer_idx = len(caches) - 1
        actual = ttnn.to_torch(caches[layer_idx].kv, mesh_composer=composer)[0, :, row_order, :]
        with safe_open(golden_dir / f"kv_cache/layer_{layer_idx}.safetensors", framework="pt") as golden:
            expected = torch.stack([golden.get_tensor(f"{h:02d}_global_h{h}") for h in range(actual.shape[0])])
        assert torch.isfinite(actual).all(), "Non-finite values in the final-layer KV cache"
        passed, pcc = comp_pcc(expected, actual, pcc=0.98)
        logger.info("Final-layer KV cache vs GPU: PCC {:.6f}", pcc)
        assert passed, f"Final-layer KV cache PCC {pcc:.6f} < 0.98"
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
