# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.6 on-device sampling integration regressions.

Run:
  MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
    pytest -svq models/demos/blackhole/qwen36/tests/test_sampling.py
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.tt_transformers.tt.generator import Generator


def _make_hot_logits(args, mesh_device, num_hot=20):
    """Build fixed, vocab-sharded logits with equiprobable hot tokens."""
    num_devices = mesh_device.get_num_devices()
    per_device_vocab = args.padded_vocab_size // num_devices
    hot_tokens = []
    for index in range(num_hot):
        shard = index % num_devices
        token = shard * per_device_vocab + 128 + index
        assert token < args.vocab_size
        hot_tokens.append(token)

    sampling_batch = max(32, args.max_batch_size)
    logits = torch.zeros(1, 1, sampling_batch, args.padded_vocab_size)
    logits[:, :, :, hot_tokens] = 10.0
    logits[:, :, :, args.vocab_size :] = -float("inf")
    tt_logits = ttnn.from_torch(
        logits,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 3),
            mesh_shape=args.cluster_shape,
        ),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    return tt_logits, set(hot_tokens)


def _extract_user_zero_token(tt_tokens):
    device_tensor = ttnn.get_device_tensors(tt_tokens)[0]
    return int(ttnn.to_torch(device_tensor).reshape(-1)[0].item())


@torch.no_grad()
@parametrize_mesh_tp()
def test_decode_only_unseeded_sampling_initializes_rng(mesh_device, reset_seeds, ensure_gc):
    """Host prefill followed by device decode must not reuse seed zero.

    This reproduces the vLLM ``sample_on_device_mode=decode_only`` lifecycle:
    device prefill sampling state is deliberately skipped, then the first
    device decode arrives with ``reset_sampling_state=True`` and no request
    seed.
    Sampling traces are disabled so this test isolates seed initialization
    from the separate sampling-trace correctness issue.
    """
    if mesh_device.get_num_devices() == 1:
        pytest.skip("Qwen3.6-27B sampling is the TP path; run with MESH_DEVICE=P150x4 or P150x8")
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=128)
    args.sampling_dp = 1

    sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None)
    model = SimpleNamespace(sampling=sampling, sampling_dp=1)
    generator = Generator([model], [args], mesh_device)

    tt_logits, hot_tokens = _make_hot_logits(args, mesh_device)
    sampling_batch = sampling.tt_sampling.max_batch_size
    params = format_sampling_params(
        SamplingParams(
            temperature=1.0,
            top_k=20,
            top_p=0.95,
            seed=None,
        ),
        sampling_batch,
    )
    positions = torch.full((sampling_batch,), -1, dtype=torch.int32)
    positions[0] = 0

    sampled_tokens = []
    for step in range(100):
        outputs = generator.sample_decode_on_device(
            [tt_logits],
            sampling_params=params,
            start_pos=[positions],
            enable_trace=False,
            reload_sampling_params=step == 0,
            reset_sampling_state=step == 0,
        )
        tt_tokens, _ = outputs[0]
        sampled_tokens.append(_extract_user_zero_token(tt_tokens))
        ttnn.deallocate(tt_tokens)
        positions[0] += 1

    sampled_set = set(sampled_tokens)
    assert sampled_set <= hot_tokens
    assert len(sampled_set) >= 2, (
        "Unseeded decode-only sampling reused one random draw for all 100 "
        f"steps: token counts={torch.tensor(sampled_tokens).unique(return_counts=True)}"
    )
