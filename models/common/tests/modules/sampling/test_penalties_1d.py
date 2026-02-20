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
    _materialize,
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


# ==============================================================================
# Helper: build topology-correct PenaltyParams + PenaltyAccumulator from config
# ==============================================================================


def _make_proper_params_accum(pen: Penalties1D):
    """Build PenaltyParams + PenaltyAccumulator from the module's resolved config.

    Uses the config's LazyBuffer mesh_mappers to guarantee the correct dtype/layout/
    sharding for whatever device topology is active. This is required for methods like
    prefill_forward and update_output_tokens that call _token_bin_counts_and_mask,
    which expects properly sharded output tensors.
    """
    params = PenaltyParams(
        prompt_mask=_materialize(pen.config.prompt_mask),
        presence_penalties=_materialize(pen.config.presence_penalties),
        frequency_penalties=_materialize(pen.config.frequency_penalties),
        repetition_penalties=_materialize(pen.config.repetition_penalties),
        inverse_repetition_penalties=_materialize(pen.config.inverse_repetition_penalties),
    )
    accum = PenaltyAccumulator(
        output_mask=_materialize(pen.config.output_mask),
        output_counts=_materialize(pen.config.output_counts),
        output_counts_gathered=_materialize(pen.config.output_counts_gathered),
    )
    return params, accum


# ==============================================================================
# Additional unit tests (no device)
# ==============================================================================


class TestConfigUnitMore:
    def test_buf_resolved_with_lazy_buffer(self):
        """_buf_resolved calls buf.is_resolved() for a LazyBuffer (line 97)."""
        from models.common.modules.lazy_buffer import LazyBuffer

        lb = LazyBuffer(source=torch.zeros(1), device=None)
        assert not Penalties1DConfig._buf_resolved(lb)  # is_resolved() → False (device=None)

    def test_buf_resolved_returns_false_for_none(self):
        """_buf_resolved returns False for None (baseline, complements line 95/97 tests)."""
        assert not Penalties1DConfig._buf_resolved(None)


# ==============================================================================
# Additional device tests: coverage for previously untested methods
# ==============================================================================


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 2), (1, 8)], ids=["1x1", "1x2", "1x8"], indirect=True)
class TestPenalties1DDeviceExtra:
    """Coverage for methods not exercised by the reference tests."""

    # ------------------------------------------------------------------
    # _buf_resolved ttnn.Tensor path (line 95)
    # ------------------------------------------------------------------

    def test_buf_resolved_with_tt_tensor(self, ttnn_mesh_device):
        """_buf_resolved returns True for a real ttnn.Tensor (line 95)."""
        tt = ttnn.from_torch(
            torch.zeros(1, 1, dtype=torch.int32),
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
        )
        assert Penalties1DConfig._buf_resolved(tt)

    # ------------------------------------------------------------------
    # from_config (lines 141-145)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_from_config(self, ttnn_mesh_device, vocab_size):
        """from_config power-path classmethod (lines 141-145)."""
        cfg = Penalties1DConfig(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen = Penalties1D.from_config(cfg)
        assert pen.config.vocab_size == vocab_size
        assert pen.config.mesh_device is ttnn_mesh_device
        assert not pen._device_buffers_loaded

    # ------------------------------------------------------------------
    # load_device_buffers idempotent guard (line 166)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_load_device_buffers_idempotent(self, ttnn_mesh_device, vocab_size):
        """Second call to load_device_buffers returns early without re-allocating (line 166)."""
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        decode_src_first = pen._decode_src
        pen.load_device_buffers()  # hits early-return at line 166
        assert pen._decode_src is decode_src_first

    # ------------------------------------------------------------------
    # prefill_forward (lines 198-222) + _token_bin_counts_and_mask counts=None path (line 421)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_prefill_forward(self, ttnn_mesh_device, vocab_size):
        """prefill_forward scatters prompt tokens into prompt_mask and returns logits unchanged."""
        B = 32
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        params, accum = _make_proper_params_accum(pen)

        logits_tt = ttnn.from_torch(
            torch.randn(B, vocab_size, dtype=torch.bfloat16),
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        prompt_tokens = torch.randint(0, vocab_size, (B, 10))
        result = pen.prefill_forward(logits_tt, params, accum, prompt_tokens)
        assert result is logits_tt  # returns logits unchanged

    # ------------------------------------------------------------------
    # forward() prefill dispatch (lines 278-280)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_forward_dispatches_to_prefill(self, ttnn_mesh_device, vocab_size):
        """forward() with prompt_tokens kwarg routes to prefill_forward (lines 278-279)."""
        B = 32
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        params, accum = _make_proper_params_accum(pen)

        logits_tt = ttnn.from_torch(
            torch.randn(B, vocab_size, dtype=torch.bfloat16),
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        prompt_tokens = torch.randint(0, vocab_size, (B, 5))
        result = pen.forward(logits_tt, params=params, accum=accum, prompt_tokens=prompt_tokens)
        assert result is logits_tt

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_forward_dispatches_to_decode(self, ttnn_mesh_device, vocab_size):
        """forward() without prompt_tokens routes to decode_forward (line 280).

        Uses unsharded (replicated) tensors — same topology as test_penalties1d_vs_reference —
        so the broadcast between logits [B, V] and penalty masks [B, V] is valid on all mesh
        shapes. The goal here is line 280 coverage, not correctness (covered elsewhere).
        """
        B = 32
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)

        zeros_BV = torch.zeros(B, vocab_size, dtype=torch.int32)
        params, accum = _make_penalty_tensors_on_device(
            ttnn_mesh_device,
            B,
            prompt_mask_host=zeros_BV,
            output_mask_host=zeros_BV,
            output_counts_host=zeros_BV,
            presence_val=0.0,
            frequency_val=0.0,
            repetition_val=1.0,
        )
        logits_tt = ttnn.from_torch(
            torch.randn(B, vocab_size, dtype=torch.bfloat16),
            device=ttnn_mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        result = pen.forward(logits_tt, params=params, accum=accum)
        assert result is not None

    # ------------------------------------------------------------------
    # update_output_tokens: standard decode path (lines 290-294)
    # and _token_bin_counts_and_mask counts-not-None path (line 419)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_update_output_tokens_standard(self, ttnn_mesh_device, vocab_size):
        """update_output_tokens with standard decode-shape [1,1,1,B] (lines 290-294)."""
        B = 32
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        _, accum = _make_proper_params_accum(pen)

        # Standard sampling output: shape[-1]=B=32, shape[-2]=1 → if-branch
        tokens_tt = ttnn.from_torch(
            torch.randint(0, vocab_size, (1, 1, 1, B), dtype=torch.int32),
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        pen.update_output_tokens(accum, tokens_tt)

    # ------------------------------------------------------------------
    # update_output_tokens: multi-token else branch (lines 296-303)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_update_output_tokens_multi_token(self, ttnn_mesh_device, vocab_size):
        """update_output_tokens with multi-token [B,S] shape triggers else-branch (lines 296-303)."""
        B = 32
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        _, accum = _make_proper_params_accum(pen)

        # shape[-1]=4 != B=32 → else-branch; src = ones(B, 4) created inline
        tokens_tt = ttnn.from_torch(
            torch.randint(0, vocab_size, (B, 4), dtype=torch.int32),
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        pen.update_output_tokens(accum, tokens_tt)

    # ------------------------------------------------------------------
    # reset_output_tokens: tokens=None (lines 317-324) and with tokens (lines 326-346)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_reset_output_tokens_no_tokens(self, ttnn_mesh_device, vocab_size):
        """reset_output_tokens(tokens=None) zeros the accum buffers (lines 317-324)."""
        B = 32
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        _, accum = _make_proper_params_accum(pen)
        pen.reset_output_tokens(accum, tokens=None)

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_reset_output_tokens_with_tokens(self, ttnn_mesh_device, vocab_size):
        """reset_output_tokens(tokens=...) zeros then re-initializes from tokens (lines 326-346)."""
        B = 32
        pen = Penalties1D(vocab_size=vocab_size, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        _, accum = _make_proper_params_accum(pen)
        tokens = torch.randint(0, vocab_size, (B, 5))
        pen.reset_output_tokens(accum, tokens=tokens)

    # ------------------------------------------------------------------
    # _pad_batch_to_max: pad (lines 399-401), truncate (402-403), ValueError (396-397)
    # ------------------------------------------------------------------

    def test_pad_batch_to_max_pads_small_batch(self, ttnn_mesh_device):
        """_pad_batch_to_max pads when B < max_batch_size (lines 399-401)."""
        pen = Penalties1D(vocab_size=1024, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        small = torch.randint(0, 100, (4, 10))
        padded = pen._pad_batch_to_max(small, pad_value=-1)
        assert padded.shape[0] == pen.config.max_batch_size
        assert (padded[4:] == -1).all()

    def test_pad_batch_to_max_truncates_large_batch(self, ttnn_mesh_device):
        """_pad_batch_to_max truncates when B > max_batch_size (lines 402-403)."""
        pen = Penalties1D(vocab_size=1024, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        large = torch.randint(0, 100, (64, 10))
        truncated = pen._pad_batch_to_max(large, pad_value=-1)
        assert truncated.shape[0] == pen.config.max_batch_size

    def test_pad_batch_to_max_raises_on_non_2d(self, ttnn_mesh_device):
        """_pad_batch_to_max raises ValueError for non-2D input (lines 396-397)."""
        pen = Penalties1D(vocab_size=1024, mesh_device=ttnn_mesh_device)
        pen.load_device_buffers()
        with pytest.raises(ValueError, match="Expected 2D"):
            pen._pad_batch_to_max(torch.zeros(10), pad_value=-1)

    # ------------------------------------------------------------------
    # _resolve_buf: ttnn.Tensor passthrough (lines 474-475)
    # and LazyBuffer resolve (line 476)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_resolve_buf_tensor_passthrough(self, ttnn_mesh_device, vocab_size):
        """Pre-existing ttnn.Tensor passes through _resolve_buf unchanged (lines 474-475)."""
        B = 32
        pre_tensor = ttnn.from_torch(
            torch.zeros(B, vocab_size, dtype=torch.int32),
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
        )
        cfg = Penalties1DConfig(vocab_size=vocab_size, mesh_device=ttnn_mesh_device, prompt_mask=pre_tensor)
        resolved = _resolve_penalties1d_config(cfg)
        assert resolved.prompt_mask is pre_tensor

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_resolve_buf_lazy_buffer_passthrough(self, ttnn_mesh_device, vocab_size):
        """Pre-existing LazyBuffer with device=None gets device filled in (line 476)."""
        from models.common.modules.lazy_buffer import LazyBuffer

        B = 32
        partial_lb = LazyBuffer(
            source=torch.zeros(B, vocab_size, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cfg = Penalties1DConfig(vocab_size=vocab_size, mesh_device=ttnn_mesh_device, prompt_mask=partial_lb)
        resolved = _resolve_penalties1d_config(cfg)
        assert isinstance(resolved.prompt_mask, LazyBuffer)
        assert resolved.prompt_mask.device is ttnn_mesh_device

    # ------------------------------------------------------------------
    # _materialize: ttnn.Tensor passthrough (line 552)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("vocab_size", [1024])
    def test_materialize_tensor_passthrough(self, ttnn_mesh_device, vocab_size):
        """Pre-existing ttnn.Tensor as decode_src passes through _materialize (line 552)."""
        B = 32
        pre_src = ttnn.from_torch(
            torch.ones(B, 1, dtype=torch.int32),
            device=ttnn_mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cfg = Penalties1DConfig(vocab_size=vocab_size, mesh_device=ttnn_mesh_device, decode_src=pre_src)
        pen = Penalties1D.from_config(cfg)
        pen.load_device_buffers()
        assert pen._decode_src is pre_src
