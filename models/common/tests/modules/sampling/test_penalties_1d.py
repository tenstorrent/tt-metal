# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Penalties1D module."""

import pytest
import torch

import ttnn
from models.common.modules.sampling.penalties_1d import (
    Penalties1D,
    Penalties1DConfig,
    PenaltyAccumulator,
    PenaltyParams,
    _resolve_penalties1d_config,
)
from models.common.utility_functions import comp_pcc

# ==============================================================================
# Reference implementation (pure torch)
# ==============================================================================


def reference_apply_penalties(logits, prompt_mask, output_mask, output_counts, presence, frequency, repetition):
    """Pure-torch reference for penalty math, following the OpenAI API spec.

    Algorithm source: vLLM's ``apply_penalties`` in ``vllm/model_executor/layers/utils.py``
    (https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/utils.py).

    - Presence: subtract flat penalty for each token that appeared in output
    - Frequency: subtract penalty proportional to token occurrence count
    - Repetition: sign-dependent scaling for tokens in prompt OR output
      (positive logits divided by penalty, negative logits multiplied by penalty)
    """
    logits = logits.clone().float()
    output_mask_f = output_mask.float()
    output_counts_f = output_counts.float()

    # Presence: logits -= output_mask * presence  (vLLM: presence_penalties * output_mask)
    logits -= output_mask_f * presence

    # Frequency: logits -= output_counts * frequency  (vLLM: frequency_penalties * output_bin_counts)
    logits -= output_counts_f * frequency

    # Repetition: sign-dependent scaling  (vLLM: apply_repetition_penalties on combined prompt+output mask)
    combined = ((prompt_mask + output_mask) > 0).float()
    inv_rep = 1.0 / repetition
    # If logit > 0: multiply by 1/rep (shrink toward 0). If logit <= 0: multiply by rep (push away from 0).
    scale = torch.where(
        logits > 0,
        torch.where(combined.bool(), inv_rep, torch.ones_like(logits)),
        torch.where(combined.bool(), repetition, torch.ones_like(logits)),
    )
    logits *= scale
    return logits


# ==============================================================================
# Unit tests: Config and dataclasses (no device)
# ==============================================================================


class TestConfigUnit:
    def test_config_defaults(self):
        cfg = Penalties1DConfig(vocab_size=1024)
        assert cfg.max_batch_size == 32
        assert cfg.mesh_device is None
        assert cfg.sub_core_grids is None
        assert cfg.prompt_mask is None

    def test_config_not_resolved_without_mesh_device(self):
        cfg = Penalties1DConfig(vocab_size=1024)
        assert not cfg.is_resolved()

    def test_penalty_params_fields(self):
        fields = PenaltyParams.__dataclass_fields__
        assert set(fields.keys()) == {
            "prompt_mask",
            "presence_penalties",
            "frequency_penalties",
            "repetition_penalties",
            "inverse_repetition_penalties",
        }

    def test_penalty_accumulator_fields(self):
        fields = PenaltyAccumulator.__dataclass_fields__
        assert set(fields.keys()) == {"output_mask", "output_counts", "output_counts_gathered"}


# ==============================================================================
# Device tests: Config resolution and Penalties1D
# ==============================================================================


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
class TestPenalties1DDevice:
    @pytest.mark.parametrize("vocab_size", [1024])
    def test_resolve_config(self, ttnn_mesh_device, vocab_size):
        cfg = Penalties1DConfig(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        resolved = _resolve_penalties1d_config(cfg)
        assert resolved.is_resolved()
        assert resolved.mesh_device is ttnn_mesh_device

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_load_device_buffers(self, ttnn_mesh_device, vocab_size):
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        assert pen._device_buffers_loaded
        assert isinstance(pen._decode_src, ttnn.Tensor)
        assert isinstance(pen._zeros, ttnn.Tensor)

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_decode_forward_none_passthrough(self, ttnn_mesh_device, vocab_size):
        """forward() with None params/accum returns logits unchanged."""
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        logits_host = torch.randn(32, vocab_size, dtype=torch.bfloat16)
        logits_tt = ttnn.from_torch(logits_host, device=ttnn_mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        result = pen.forward(logits_tt, params=None, accum=None)
        assert result is logits_tt

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_from_model_args(self, ttnn_mesh_device, vocab_size):
        """from_model_args backward compat factory."""

        class MockArgs:
            padded_vocab_size = vocab_size
            sub_core_grids = None

        pen = Penalties1D.from_model_args(ttnn_mesh_device, MockArgs())
        assert pen.config.vocab_size == vocab_size
        assert pen.config.mesh_device is ttnn_mesh_device

    def test_rejects_galaxy(self, ttnn_mesh_device):
        """from_model_args should reject 2D (Galaxy) topologies."""

        class FakeMesh:
            shape = (2, 4)

        class MockArgs:
            padded_vocab_size = 1024
            sub_core_grids = None

        with pytest.raises(ValueError, match="1D mesh topologies"):
            Penalties1D.from_model_args(FakeMesh(), MockArgs())


# ==============================================================================
# VS Reference tests — full penalty pipeline compared against pure-torch golden
# ==============================================================================


def _make_penalty_tensors_on_device(
    ttnn_mesh_device,
    B,
    *,
    prompt_mask_host,
    output_mask_host,
    output_counts_host,
    presence_val,
    frequency_val,
    repetition_val,
):
    """Helper: build PenaltyParams + PenaltyAccumulator on device from host tensors."""
    params = PenaltyParams(
        prompt_mask=ttnn.from_torch(
            prompt_mask_host,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        presence_penalties=ttnn.from_torch(
            torch.full((B, 1), presence_val),
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        ),
        frequency_penalties=ttnn.from_torch(
            torch.full((B, 1), frequency_val),
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        ),
        repetition_penalties=ttnn.from_torch(
            torch.full((B, 1), repetition_val),
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        ),
        inverse_repetition_penalties=ttnn.from_torch(
            torch.full((B, 1), 1.0 / repetition_val),
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        ),
    )
    accum = PenaltyAccumulator(
        output_mask=ttnn.from_torch(
            output_mask_host,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        output_counts=ttnn.from_torch(
            output_counts_host,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        output_counts_gathered=ttnn.from_torch(
            output_counts_host,
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )
    return params, accum


def _readback_logits(result_tt, ttnn_mesh_device, B, vocab_size):
    """Helper: read logits back from device to host torch tensor."""
    result_host = ttnn.to_torch(
        result_tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            ttnn_mesh_device, dims=(0, 1), mesh_shape=tuple(ttnn_mesh_device.shape)
        ),
    )
    return result_host[:B, :vocab_size].float()


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
@pytest.mark.parametrize(
    "vocab_size,presence,frequency,repetition,pcc",
    [
        pytest.param(1024, 0.0, 0.0, 1.0, 0.999, id="no-penalties"),
        pytest.param(1024, 0.6, 0.0, 1.0, 0.98, id="presence-only"),
        pytest.param(1024, 0.0, 0.4, 1.0, 0.98, id="frequency-only"),
        pytest.param(1024, 0.0, 0.0, 1.3, 0.97, id="repetition-only"),
        pytest.param(1024, 0.6, 0.4, 1.3, 0.95, id="all-penalties"),
    ],
)
def test_penalties1d_vs_reference(
    ttnn_mesh_device,
    vocab_size,
    presence,
    frequency,
    repetition,
    pcc,
):
    """
    Test Penalties1D.decode_forward matches the pure-torch reference_apply_penalties.

    Parametrized across penalty combinations to verify each penalty type independently
    and in combination. PCC thresholds account for bfloat16 precision.
    """
    torch.manual_seed(42)
    B = 32

    pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)

    # Build host tensors with realistic patterns
    logits_host = torch.randn(B, vocab_size, dtype=torch.bfloat16)

    # prompt_mask: first 50 tokens were in the prompt
    prompt_mask_host = torch.zeros(B, vocab_size, dtype=torch.int32)
    prompt_mask_host[:, :50] = 1

    # output_mask: tokens 40-60 appeared in output (overlaps with prompt)
    output_mask_host = torch.zeros(B, vocab_size, dtype=torch.int32)
    output_mask_host[:, 40:60] = 1

    # output_counts: tokens 40-60 appeared 1-3 times
    output_counts_host = torch.zeros(B, vocab_size, dtype=torch.int32)
    output_counts_host[:, 40:50] = 2
    output_counts_host[:, 50:60] = 1

    # --- Reference (pure torch) ---
    expected = reference_apply_penalties(
        logits_host,
        prompt_mask_host,
        output_mask_host,
        output_counts_host,
        presence,
        frequency,
        repetition,
    )

    # --- TT device ---
    logits_tt = ttnn.from_torch(
        logits_host,
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    params, accum = _make_penalty_tensors_on_device(
        ttnn_mesh_device,
        B,
        prompt_mask_host=prompt_mask_host,
        output_mask_host=output_mask_host,
        output_counts_host=output_counts_host,
        presence_val=presence,
        frequency_val=frequency,
        repetition_val=repetition,
    )

    result_tt = pen.decode_forward(logits_tt, params, accum)
    result_host = _readback_logits(result_tt, ttnn_mesh_device, B, vocab_size)

    passing, pcc_msg = comp_pcc(expected, result_host, pcc=pcc)
    assert passing, f"Penalties1D vs reference failed: {pcc_msg} (threshold={pcc})"


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
def test_penalties1d_changes_argmax(ttnn_mesh_device):
    """
    Heavy repetition penalty should change which token has the highest logit.

    Setup: token 0 has the highest logit AND appears in prompt+output.
    With repetition=5.0, the penalty should push token 0's logit down far enough
    that a different token becomes the argmax.
    """
    torch.manual_seed(123)
    B = 32
    vocab_size = 1024

    pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)

    # Token 0 is the clear winner in raw logits
    logits_host = torch.randn(B, vocab_size, dtype=torch.bfloat16)
    logits_host[:, 0] = 10.0  # make token 0 dominant

    # Token 0 appears in prompt and output
    prompt_mask_host = torch.zeros(B, vocab_size, dtype=torch.int32)
    prompt_mask_host[:, 0] = 1
    output_mask_host = torch.zeros(B, vocab_size, dtype=torch.int32)
    output_mask_host[:, 0] = 1
    output_counts_host = torch.zeros(B, vocab_size, dtype=torch.int32)
    output_counts_host[:, 0] = 3

    # Original argmax should be token 0
    assert logits_host[0].argmax().item() == 0

    logits_tt = ttnn.from_torch(
        logits_host,
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    params, accum = _make_penalty_tensors_on_device(
        ttnn_mesh_device,
        B,
        prompt_mask_host=prompt_mask_host,
        output_mask_host=output_mask_host,
        output_counts_host=output_counts_host,
        presence_val=2.0,
        frequency_val=2.0,
        repetition_val=5.0,
    )

    result_tt = pen.decode_forward(logits_tt, params, accum)
    result_host = _readback_logits(result_tt, ttnn_mesh_device, B, vocab_size)

    # After heavy penalties, token 0 should no longer be argmax
    new_argmax = result_host[0].argmax().item()
    assert new_argmax != 0, f"Expected penalty to change argmax from 0, but it's still {new_argmax}"
