# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Qwen3.5-9B RoPE (``tt/rope.py``).

This validates the rope module on its own — both the host-side frequency
generator (``compute_rope_freqs``) and the on-device setup/lookup
(``Qwen36RoPESetup``) against the analytic torch reference.

Qwen3.5 uses *partial* rotary embeddings: only ``rope_head_dim`` (64) of the
256-wide head is rotated, and the cos/sin table is GPT-NeoX style — the first
and second halves are duplicated (``cat([angles, angles])``), NOT interleaved
like the Meta/Llama format.

Requires a Blackhole P150 device.
Run: pytest models/demos/blackhole/qwen36/tests/unit/test_rope.py -v
"""
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc
from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup, compute_rope_freqs

pytestmark = run_for_blackhole()

# Qwen3.5-9B partial-rotary constants (see tt/rope.py and tt/model_config.py).
# A small max_seq_len keeps the precomputed table cheap — the code path that
# builds and slices it is identical regardless of the table length.
ROPE_HEAD_DIM = 64
ROPE_THETA = 10_000_000.0
MAX_SEQ_LEN = 8192
PCC_THRESHOLD = 0.99
MAX_ABS_DIFF = 0.05  # bf16 cos/sin round-trip slack


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="module")
def rope_setup(device):
    """Build the on-device rope setup once with the real Qwen3.5-9B constants."""
    args = SimpleNamespace(
        rope_head_dim=ROPE_HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
        rope_theta=ROPE_THETA,
    )
    return Qwen36RoPESetup(device, args), args


class TestComputeRopeFreqs:
    """Analytic properties of the host-side cos/sin generator (no device)."""

    def test_shapes(self):
        cos, sin = compute_rope_freqs(ROPE_HEAD_DIM, MAX_SEQ_LEN, theta=ROPE_THETA)
        assert cos.shape == (MAX_SEQ_LEN, ROPE_HEAD_DIM)
        assert sin.shape == (MAX_SEQ_LEN, ROPE_HEAD_DIM)

    def test_position_zero_is_identity(self):
        # At position 0 every angle is 0 -> cos == 1, sin == 0 (no rotation).
        cos, sin = compute_rope_freqs(ROPE_HEAD_DIM, MAX_SEQ_LEN, theta=ROPE_THETA)
        assert torch.allclose(cos[0], torch.ones(ROPE_HEAD_DIM))
        assert torch.allclose(sin[0], torch.zeros(ROPE_HEAD_DIM))

    def test_neox_half_duplication(self):
        # GPT-NeoX layout: table = cat([angles, angles]) so the first and second
        # halves are identical. A regression to interleaved (Meta) layout breaks this.
        cos, sin = compute_rope_freqs(ROPE_HEAD_DIM, MAX_SEQ_LEN, theta=ROPE_THETA)
        half = ROPE_HEAD_DIM // 2
        assert torch.allclose(cos[:, :half], cos[:, half:])
        assert torch.allclose(sin[:, :half], sin[:, half:])

    def test_unit_circle(self):
        # cos^2 + sin^2 == 1 for every position/frequency.
        cos, sin = compute_rope_freqs(ROPE_HEAD_DIM, MAX_SEQ_LEN, theta=ROPE_THETA)
        ones = torch.ones_like(cos)
        assert torch.allclose(cos**2 + sin**2, ones, atol=1e-4)

    def test_theta_controls_frequency(self):
        # A larger theta -> lower frequencies -> slower rotation, so at a fixed
        # (late) position the angle is smaller and cos stays closer to 1. The
        # effect lives at the lowest-frequency component (the last unique index);
        # component 0 has exponent 0 -> freq 1.0 regardless of theta, so it can't
        # be used to probe this. Guards the theta wiring.
        seq_len, late_pos = 4096, 4000
        last_freq = ROPE_HEAD_DIM // 2 - 1
        cos_lo, _ = compute_rope_freqs(ROPE_HEAD_DIM, seq_len, theta=10_000.0)
        cos_hi, _ = compute_rope_freqs(ROPE_HEAD_DIM, seq_len, theta=10_000_000.0)
        assert cos_hi[late_pos, last_freq] > cos_lo[late_pos, last_freq]


class TestRoPEDeviceTable:
    """The precomputed on-device cos/sin table must match the host reference."""

    def test_device_table_matches_reference(self, rope_setup):
        setup, args = rope_setup
        cos_ref, sin_ref = compute_rope_freqs(args.rope_head_dim, args.max_seq_len, theta=args.rope_theta)

        cos_dev = ttnn.to_torch(setup.cos_device).reshape(-1, args.rope_head_dim)[: args.max_seq_len]
        sin_dev = ttnn.to_torch(setup.sin_device).reshape(-1, args.rope_head_dim)[: args.max_seq_len]

        cos_pcc = compute_pcc(cos_ref, cos_dev)
        sin_pcc = compute_pcc(sin_ref, sin_dev)
        cos_err = (cos_ref - cos_dev.float()).abs().max().item()
        sin_err = (sin_ref - sin_dev.float()).abs().max().item()
        logger.info(f"Device table cos PCC={cos_pcc:.6f} (max abs err {cos_err:.4f})")
        logger.info(f"Device table sin PCC={sin_pcc:.6f} (max abs err {sin_err:.4f})")

        assert cos_pcc > PCC_THRESHOLD, f"cos table PCC too low: {cos_pcc}"
        assert sin_pcc > PCC_THRESHOLD, f"sin table PCC too low: {sin_pcc}"
        assert cos_err < MAX_ABS_DIFF and sin_err < MAX_ABS_DIFF


class TestGetRotMats:
    """The lookup path used by attention: decode (single pos) and prefill (range)."""

    @pytest.mark.parametrize("pos", [0, 1, 100, 1000, 4096])
    def test_decode_fast_path(self, rope_setup, pos):
        # T == 1 and B == 1 hits the fast path that slices the device table.
        setup, args = rope_setup
        cos_ref, sin_ref = compute_rope_freqs(args.rope_head_dim, args.max_seq_len, theta=args.rope_theta)

        cos_tt, sin_tt = setup.get_rot_mats(torch.tensor([[pos]]))
        cos = ttnn.to_torch(cos_tt).reshape(-1, args.rope_head_dim)[0]
        sin = ttnn.to_torch(sin_tt).reshape(-1, args.rope_head_dim)[0]

        cos_err = (cos_ref[pos] - cos.float()).abs().max().item()
        sin_err = (sin_ref[pos] - sin.float()).abs().max().item()
        logger.info(f"decode pos={pos}: cos max abs err {cos_err:.4f}, sin max abs err {sin_err:.4f}")
        assert cos_err < MAX_ABS_DIFF, f"cos mismatch at pos {pos}: {cos_err}"
        assert sin_err < MAX_ABS_DIFF, f"sin mismatch at pos {pos}: {sin_err}"

    @pytest.mark.parametrize("seq_len", [4, 128])
    def test_prefill_range(self, rope_setup, seq_len):
        # T > 1 hits the general path that builds a fresh tensor from cos_cpu.
        setup, args = rope_setup
        cos_ref, sin_ref = compute_rope_freqs(args.rope_head_dim, args.max_seq_len, theta=args.rope_theta)

        pos_ids = torch.arange(seq_len).unsqueeze(0)
        cos_tt, sin_tt = setup.get_rot_mats(pos_ids)
        cos = ttnn.to_torch(cos_tt).reshape(seq_len, args.rope_head_dim)
        sin = ttnn.to_torch(sin_tt).reshape(seq_len, args.rope_head_dim)

        cos_pcc = compute_pcc(cos_ref[:seq_len], cos)
        sin_pcc = compute_pcc(sin_ref[:seq_len], sin)
        logger.info(f"prefill T={seq_len}: cos PCC={cos_pcc:.6f}, sin PCC={sin_pcc:.6f}")
        assert cos_pcc > PCC_THRESHOLD, f"cos PCC too low for T={seq_len}: {cos_pcc}"
        assert sin_pcc > PCC_THRESHOLD, f"sin PCC too low for T={seq_len}: {sin_pcc}"

    @pytest.mark.parametrize("pos", [0, 1, 500, 2047])
    def test_get_cos_sin_host(self, rope_setup, pos):
        # The host-DMA helper used to refresh the traced decode buffers must return
        # the same row, on host, shaped [1, 1, head_dim].
        setup, args = rope_setup
        cos_ref, sin_ref = compute_rope_freqs(args.rope_head_dim, args.max_seq_len, theta=args.rope_theta)

        cos_host, sin_host = setup.get_cos_sin_host(pos)
        cos = ttnn.to_torch(cos_host).reshape(-1, args.rope_head_dim)[0]
        sin = ttnn.to_torch(sin_host).reshape(-1, args.rope_head_dim)[0]

        assert (cos_ref[pos] - cos.float()).abs().max().item() < MAX_ABS_DIFF
        assert (sin_ref[pos] - sin.float()).abs().max().item() < MAX_ABS_DIFF
