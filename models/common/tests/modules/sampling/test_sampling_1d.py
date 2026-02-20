# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Sampling1D module."""

import pytest
import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig, _resolve_sampling1d_config


# todo)) this is not the TTTv2 way! look at mlp_1d.py for the correct way.
def _get_tt_ccl_if_multi_device(mesh_device):
    """Get TT_CCL for multi-device meshes, None for single device."""
    if mesh_device.get_num_devices() <= 1:
        return None
    from models.common.modules.tt_ccl import get_tt_ccl

    return get_tt_ccl(mesh_device)


# ==============================================================================
# Unit tests: Config (no device)
# ==============================================================================


class TestConfigUnit:
    def test_config_defaults(self):
        cfg = Sampling1DConfig(vocab_size=1024)
        assert cfg.max_batch_size == 32
        assert cfg.max_top_k == 32
        assert cfg.allow_force_argmax is False
        assert cfg.num_gather_links == 1
        assert cfg.mesh_device is None
        assert cfg.index_offsets is None
        assert cfg.seeds is None

    def test_config_custom(self):
        cfg = Sampling1DConfig(vocab_size=128256, max_top_k=64, allow_force_argmax=True)
        assert cfg.vocab_size == 128256
        assert cfg.max_top_k == 64
        assert cfg.allow_force_argmax is True

    def test_config_not_resolved_without_device(self):
        cfg = Sampling1DConfig(vocab_size=1024)
        assert not cfg.is_resolved()


# ==============================================================================
# Device tests
# ==============================================================================


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
class TestSampling1DDevice:
    @pytest.mark.parametrize("vocab_size", [1024])
    def test_resolve_config(self, ttnn_mesh_device, vocab_size):
        cfg = Sampling1DConfig(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        resolved = _resolve_sampling1d_config(cfg)
        assert resolved.is_resolved()
        assert resolved.start_core is not None
        assert resolved.sampling_memory_config is not None
        assert resolved.index_offsets is not None
        assert resolved.seeds is not None

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_load_device_buffers(self, ttnn_mesh_device, vocab_size):
        sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        sampler.load_device_buffers()
        assert sampler._device_buffers_loaded
        assert isinstance(sampler._index_offsets, ttnn.Tensor)
        assert isinstance(sampler._local_indices, ttnn.Tensor)
        assert isinstance(sampler._seeds, ttnn.Tensor)
        assert isinstance(sampler._user_ids, ttnn.Tensor)

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_force_argmax(self, ttnn_mesh_device, vocab_size):
        """allow_force_argmax=True, k/p/temp=None → matches torch.argmax."""
        sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device, allow_force_argmax=True)
        sampler.load_device_buffers()
        B = sampler.config.max_batch_size

        torch.manual_seed(42)
        logits_host = torch.randn(1, 1, B, vocab_size, dtype=torch.bfloat16)

        logits_tt = _make_logits_tt(logits_host, ttnn_mesh_device)

        tokens_tt, log_probs = sampler.decode_forward(logits_tt)
        tokens_host = to_torch_auto_compose(tokens_tt)

        expected_argmax = logits_host.float().argmax(dim=-1)
        tokens_flat = tokens_host.flatten()[:B]
        expected_flat = expected_argmax.flatten()[:B]

        assert torch.equal(
            tokens_flat, expected_flat
        ), f"Argmax mismatch: got {tokens_flat[:5]} vs expected {expected_flat[:5]}"

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_error_on_partial_params(self, ttnn_mesh_device, vocab_size):
        """k provided but not p/temp → ValueError."""
        sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        logits_host = torch.randn(1, 1, 32, vocab_size, dtype=torch.bfloat16)
        logits_tt = _make_logits_tt(logits_host, ttnn_mesh_device)
        k_tt = ttnn.from_torch(torch.ones(32), device=ttnn_mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        with pytest.raises(ValueError, match="k, p, temp must all be provided"):
            sampler.decode_forward(logits_tt, k=k_tt)

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_from_model_args(self, ttnn_mesh_device, vocab_size):
        """from_model_args backward compat factory."""

        class MockArgs:
            padded_vocab_size = vocab_size
            sub_core_grids = None
            sub_core_grid_topk = None
            start_core = ttnn.CoreCoord(0, 0)
            max_top_k = 32

        sampler = Sampling1D.from_model_args(ttnn_mesh_device, None, MockArgs())
        assert sampler.config.vocab_size == vocab_size
        assert sampler.config.mesh_device is ttnn_mesh_device

    def test_rejects_galaxy(self, ttnn_mesh_device):
        """from_model_args should reject 2D (Galaxy) topologies."""

        class FakeMesh:
            shape = (2, 4)

            def get_num_devices(self):
                return 8

        class MockArgs:
            padded_vocab_size = 1024
            sub_core_grids = None
            sub_core_grid_topk = None
            start_core = ttnn.CoreCoord(0, 0)
            max_top_k = 32

        with pytest.raises(ValueError, match="1D mesh topologies"):
            Sampling1D.from_model_args(FakeMesh(), None, MockArgs())


# ==============================================================================
# VS Reference tests — sampling correctness against torch golden
# ==============================================================================


def _make_logits_tt(logits_host, ttnn_mesh_device, *, shard_vocab=False):
    """Create logits on device. shard_vocab=True shards the last dim across devices (for top-k path)."""
    cluster_shape = tuple(ttnn_mesh_device.shape)
    if not shard_vocab or max(cluster_shape) == 1:
        shard_dims = (None, None)
    elif cluster_shape[-1] >= cluster_shape[-2]:
        shard_dims = (None, -1)
    else:
        shard_dims = (-1, None)
    return ttnn.from_torch(
        logits_host,
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=shard_dims, mesh_shape=cluster_shape),
    )


def _make_sampling_params(ttnn_mesh_device, B, *, k_val=1, p_val=0.0, temp_val=1.0, cluster_shape=(1, 1)):
    """Helper: create k/p/temp device tensors for Sampling1D.decode_forward()."""
    k = ttnn.from_torch(
        torch.full((B,), k_val, dtype=torch.int32),
        device=ttnn_mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=cluster_shape),
    )
    p = ttnn.from_torch(
        torch.full((B,), p_val),
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=cluster_shape),
    )
    temp = ttnn.from_torch(
        torch.full((B,), temp_val),
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=cluster_shape),
    )
    return k, p, temp


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
@pytest.mark.parametrize(
    "vocab_size",
    [
        pytest.param(1024, id="v1024"),
        pytest.param(32000, id="v32000"),
    ],
)
def test_sampling1d_topk1_vs_argmax(ttnn_mesh_device, vocab_size):
    """
    Top-k=1, p=0.0, temp=1.0 should produce the same result as torch.argmax.

    This is the primary correctness test: with k=1 the sampling degenerates to argmax,
    giving us an exact reference to compare against.
    """
    torch.manual_seed(42)
    B = 32

    tt_ccl = _get_tt_ccl_if_multi_device(ttnn_mesh_device)
    sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device, tt_ccl=tt_ccl)

    logits_host = torch.randn(1, 1, B, vocab_size, dtype=torch.bfloat16)

    logits_tt = _make_logits_tt(logits_host, ttnn_mesh_device, shard_vocab=True)

    cluster_shape = tuple(ttnn_mesh_device.shape)
    k, p, temp = _make_sampling_params(
        ttnn_mesh_device, B, k_val=1, p_val=0.0, temp_val=1.0, cluster_shape=cluster_shape
    )

    tokens_tt, _ = sampler.decode_forward(logits_tt, k=k, p=p, temp=temp)
    tokens_host = to_torch_auto_compose(tokens_tt).flatten()[:B]

    expected = logits_host.float().argmax(dim=-1).flatten()[:B]

    # bfloat16 tie-breaking: when two logits are nearly equal, device top-k may pick
    # a different token than float32 argmax. Allow ≤15% mismatch rate.
    num_mismatches = (tokens_host != expected).sum().item()
    max_mismatches = max(2, int(B * 0.15))
    assert num_mismatches <= max_mismatches, (
        f"top-k=1 vs argmax: {num_mismatches}/{B} mismatches (max {max_mismatches} allowed)\n"
        f"  got:      {tokens_host[:8]}\n  expected: {expected[:8]}"
    )


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
def test_sampling1d_argmax_vs_reference(ttnn_mesh_device):
    """
    Force-argmax path (k/p/temp=None, allow_force_argmax=True) vs torch.argmax.

    Tests the all-gather-free argmax path on single device.
    """
    torch.manual_seed(99)
    B = 32
    vocab_size = 1024

    sampler = Sampling1D(
        vocab_size=vocab_size,
        mesh_device=ttnn_mesh_device,
        allow_force_argmax=True,
    )

    logits_host = torch.randn(1, 1, B, vocab_size, dtype=torch.bfloat16)
    logits_tt = _make_logits_tt(logits_host, ttnn_mesh_device)

    tokens_tt, _ = sampler.decode_forward(logits_tt)
    tokens_host = to_torch_auto_compose(tokens_tt).flatten()[:B]

    expected = logits_host.float().argmax(dim=-1).flatten()[:B]

    assert torch.equal(
        tokens_host, expected
    ), f"argmax path mismatch:\n  got:      {tokens_host[:8]}\n  expected: {expected[:8]}"


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
def test_sampling1d_topk32_in_range(ttnn_mesh_device):
    """
    Top-k=32, p=1.0 → sampled token must be within the top-32 set for every batch element.

    This is a statistical correctness test: we don't know which token will be sampled
    (it's stochastic), but it MUST be one of the top-32 tokens by logit value.
    """
    torch.manual_seed(77)
    B = 32
    vocab_size = 1024

    tt_ccl = _get_tt_ccl_if_multi_device(ttnn_mesh_device)
    sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device, tt_ccl=tt_ccl)

    logits_host = torch.randn(1, 1, B, vocab_size, dtype=torch.bfloat16)
    logits_tt = _make_logits_tt(logits_host, ttnn_mesh_device, shard_vocab=True)

    cluster_shape = tuple(ttnn_mesh_device.shape)
    k, p, temp = _make_sampling_params(
        ttnn_mesh_device, B, k_val=32, p_val=1.0, temp_val=1.0, cluster_shape=cluster_shape
    )

    tokens_tt, _ = sampler.decode_forward(logits_tt, k=k, p=p, temp=temp)
    tokens_host = to_torch_auto_compose(tokens_tt).flatten()[:B]

    # Compute the top-32 token set per batch element
    _, top32_indices = logits_host.float().squeeze().topk(32, dim=-1)  # [B, 32]

    for b in range(B):
        sampled_token = tokens_host[b].item()
        top32_set = set(top32_indices[b].tolist())
        assert sampled_token in top32_set, f"Batch {b}: sampled token {sampled_token} not in top-32 set"


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
def test_sampling1d_deterministic_with_same_seed(ttnn_mesh_device):
    """
    Two decode_forward calls with the same seed tensor should produce the same tokens.
    """
    torch.manual_seed(42)
    B = 32
    vocab_size = 1024

    tt_ccl = _get_tt_ccl_if_multi_device(ttnn_mesh_device)
    sampler = Sampling1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device, tt_ccl=tt_ccl)

    logits_host = torch.randn(1, 1, B, vocab_size, dtype=torch.bfloat16)
    cluster_shape = tuple(ttnn_mesh_device.shape)
    k, p, temp = _make_sampling_params(
        ttnn_mesh_device, B, k_val=32, p_val=0.9, temp_val=0.8, cluster_shape=cluster_shape
    )

    # Use explicit seed tensor
    seed_tensor = ttnn.from_torch(
        torch.arange(B, dtype=torch.int64).to(torch.int32),
        device=ttnn_mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # First call
    logits_tt1 = _make_logits_tt(logits_host, ttnn_mesh_device, shard_vocab=True)
    tokens1, _ = sampler.decode_forward(logits_tt1, k=k, p=p, temp=temp, seeds=seed_tensor)
    tokens1_host = to_torch_auto_compose(tokens1).flatten()[:B]

    # Second call with same seed
    logits_tt2 = _make_logits_tt(logits_host, ttnn_mesh_device, shard_vocab=True)
    tokens2, _ = sampler.decode_forward(logits_tt2, k=k, p=p, temp=temp, seeds=seed_tensor)
    tokens2_host = to_torch_auto_compose(tokens2).flatten()[:B]

    assert torch.equal(
        tokens1_host, tokens2_host
    ), f"Same seed produced different tokens:\n  call1: {tokens1_host[:8]}\n  call2: {tokens2_host[:8]}"
