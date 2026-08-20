# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Determinism and repeated-run stability of the TTNN decoder layer.

Bit-exactness is asserted, not PCC. Identical inputs through identical kernels
must give identical bits; anything less means the result depends on something
not in the inputs -- uninitialised memory, a race between cores, or a reduction
whose order varies run to run. Those defects are intermittent by nature, so a
tolerance-based check would hide exactly the cases worth catching.

``test_repeated_decode_steps_are_stable`` runs a long decode rollout instead.
It is the counterpart test: nothing there is compared against a reference, it
just has to keep producing finite, non-degenerate activations for 64 steps.
Cache-indexing and accumulation faults tend to show up as slow drift rather
than a hard failure, and a handful of steps will not surface them.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ..tt.functional_decoder import (
    DecoderLayerConfig,
    build_expert_sparsity,
    build_rope_cache,
    create_kv_cache,
    decoder_layer_decode,
    decoder_layer_prefill,
    upload_layer_weights,
)
from ..tt.weight_mapping import convert_layer_weights
from .reference import build_reference_layer, layer_state_dict

LAYER_IDX = 0
MAX_SEQ = 256
BLOCK_SIZE = 32


@pytest.fixture(scope="module")
def reference():
    return build_reference_layer(LAYER_IDX)


@pytest.fixture(scope="module")
def torch_weights(reference):
    _, hf_config = reference
    return convert_layer_weights(layer_state_dict(LAYER_IDX), hf_config)


def _hidden(hf_config, seq_len, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.float32) * 0.02


def _to_device(t, mesh_device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _build(mesh_device, hf_config, torch_weights):
    config = DecoderLayerConfig.from_hf(hf_config)
    weights = upload_layer_weights(torch_weights, mesh_device, config)
    cos_cache, sin_cache = build_rope_cache(hf_config, MAX_SEQ, mesh_device)
    sparsity = build_expert_sparsity(mesh_device, config.moe.num_experts)
    return config, weights, cos_cache, sin_cache, sparsity


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_prefill_is_deterministic(mesh_device, reference, torch_weights):
    _, hf_config = reference
    config, weights, cos_cache, sin_cache, sparsity = _build(mesh_device, hf_config, torch_weights)
    hidden = _hidden(hf_config, 128)

    outs = []
    for _ in range(3):
        tt_in = _to_device(hidden.unsqueeze(0), mesh_device)
        out = decoder_layer_prefill(tt_in, weights, config, cos_cache, sin_cache, sparsity)
        outs.append(ttnn.to_torch(out).clone())

    assert torch.equal(outs[0], outs[1]), "prefill run 1 != run 2 (bitwise)"
    assert torch.equal(outs[0], outs[2]), "prefill run 1 != run 3 (bitwise)"
    logger.info("prefill: 3 runs bit-identical")


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_decode_is_deterministic(mesh_device, reference, torch_weights):
    """Two independent prefill+decode sequences must agree bitwise.

    Each repetition allocates a fresh paged cache, so this also checks that no
    state leaks between runs through the cache or the page table.
    """
    _, hf_config = reference
    config, weights, cos_cache, sin_cache, sparsity = _build(mesh_device, hf_config, torch_weights)
    prompt_len = 32
    hidden_full = _hidden(hf_config, prompt_len + 1)

    outs = []
    for _ in range(2):
        kv_cache = create_kv_cache(
            mesh_device, config.attention, max_batch=1, max_seq_len=MAX_SEQ, block_size=BLOCK_SIZE
        )
        decoder_layer_prefill(
            _to_device(hidden_full[:, :prompt_len, :].unsqueeze(0), mesh_device),
            weights,
            config,
            cos_cache,
            sin_cache,
            sparsity,
            kv_cache=kv_cache,
        )
        current_pos = ttnn.from_torch(
            torch.tensor([prompt_len], dtype=torch.int32), dtype=ttnn.int32, device=mesh_device
        )
        tt_in = _to_device(hidden_full[:, prompt_len, :].reshape(1, 1, 1, hf_config.hidden_size), mesh_device)
        out = decoder_layer_decode(
            tt_in, weights, config, cos_cache, sin_cache, kv_cache, current_pos, token_index=prompt_len
        )
        outs.append(ttnn.to_torch(out).clone())

    assert torch.equal(outs[0], outs[1]), "decode from two fresh caches differs bitwise"
    logger.info("decode: 2 independent prefill+decode sequences bit-identical")


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_repeated_decode_steps_are_stable(mesh_device, reference, torch_weights):
    """A 64-step rollout must stay finite and non-degenerate.

    Long rollouts are where cache-indexing and accumulation faults show up as
    drift rather than an exception, so this watches the activation scale across
    every step instead of only checking the last one.
    """
    _, hf_config = reference
    config, weights, cos_cache, sin_cache, sparsity = _build(mesh_device, hf_config, torch_weights)
    prompt_len, steps = 32, 64
    hidden_full = _hidden(hf_config, prompt_len + steps)

    kv_cache = create_kv_cache(mesh_device, config.attention, max_batch=1, max_seq_len=MAX_SEQ, block_size=BLOCK_SIZE)
    decoder_layer_prefill(
        _to_device(hidden_full[:, :prompt_len, :].unsqueeze(0), mesh_device),
        weights,
        config,
        cos_cache,
        sin_cache,
        sparsity,
        kv_cache=kv_cache,
    )

    stds = []
    for step in range(steps):
        pos = prompt_len + step
        current_pos = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, device=mesh_device)
        tt_in = _to_device(hidden_full[:, pos, :].reshape(1, 1, 1, hf_config.hidden_size), mesh_device)
        out = decoder_layer_decode(tt_in, weights, config, cos_cache, sin_cache, kv_cache, current_pos, token_index=pos)
        t = ttnn.to_torch(out).float()
        assert torch.isfinite(t).all(), f"decode step {step} (pos {pos}) produced non-finite values"
        stds.append(float(t.std()))

    lo, hi = min(stds), max(stds)
    logger.info(f"64-step rollout: activation std min={lo:.5f} max={hi:.5f} ratio={hi / lo:.3f}")
    assert lo > 1e-6, f"activations collapsed to zero during the rollout (min std {lo:.3e})"
    assert hi / lo < 5.0, f"activation scale drifted {hi / lo:.1f}x across 64 steps -- suspect cache indexing"
