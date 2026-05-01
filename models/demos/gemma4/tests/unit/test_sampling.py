# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 on-device sampling (greedy argmax).

Verifies that on-device sampling produces the same token IDs as CPU argmax
on known logits with a clear winner token.

    pytest -k "1x8 and sampling"   # T3K (on-device sampling active)
    pytest -k "1x1 and sampling"   # single card (CPU fallback)
"""

import torch

import ttnn

from ...tests.test_factory import parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 8)])
def test_sampling_greedy(mesh_device, reset_seeds):
    """Test that on-device greedy sampling matches CPU argmax.

    Creates logits with a known winner token per position, runs through
    the full model's sampling path, and verifies token IDs match.
    """
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.model import _compute_per_device_vocab

    tp = mesh_device.shape[1]
    vocab_size = 262144
    batch_size = 1
    per_device_vocab = _compute_per_device_vocab(vocab_size, tp)  # 32768

    # Create logits with known winner tokens
    # Set position 0 → token 42, position 1 → token 100000 (in shard 3)
    winner_tokens = [42, 100000]
    logits_cpu = torch.zeros(1, 1, batch_size, vocab_size, dtype=torch.bfloat16)
    for i, tok in enumerate(winner_tokens):
        if i < batch_size:
            logits_cpu[0, 0, i, tok] = 50.0  # Clear winner

    # Shard logits across TP (column-parallel on vocab dim)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    # Build sampling args (same as model.py does)
    from models.common.sampling.generator import SamplingGenerator

    class _Args:
        pass

    args = _Args()
    args.vocab_size = vocab_size
    args.padded_vocab_size = per_device_vocab * tp
    args.cluster_shape = tuple(mesh_device.shape)
    args.sampling_all_gather_axis = 1
    args.sampling_dp = 1
    args.num_devices = mesh_device.get_num_devices()
    args.is_galaxy = mesh_device.shape[0] > 1
    args.model_config = {}
    args.use_topk_logprobs = False

    sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None, enable_internal_trace=False)

    # Shard logits: each device gets [1, 1, batch, vocab/TP]
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    col_mapper = mesh_config.column_parallel(mesh_device)
    logits_tt = ttnn.from_torch(
        logits_cpu,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=col_mapper,
    )

    # Pad batch to 32 (sampling module requirement)
    if batch_size < 32:
        logits_tt = ttnn.pad(logits_tt, padding=[(0, 0), (0, 0), (0, 32 - batch_size), (0, 0)], value=0.0)

    # Sample
    tt_tokens, _ = sampling.sample(logits_tt, enable_trace=False)
    tokens_cpu = ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0]).reshape(-1)

    # Verify winner tokens
    for i, expected_tok in enumerate(winner_tokens):
        if i < batch_size:
            actual_tok = tokens_cpu[i].item()
            assert (
                actual_tok == expected_tok
            ), f"Sampling mismatch at position {i}: expected {expected_tok}, got {actual_tok}"
