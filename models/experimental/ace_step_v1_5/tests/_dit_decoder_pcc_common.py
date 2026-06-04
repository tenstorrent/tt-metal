# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and PCC helpers for ACE-Step v1.5 device tests."""

from __future__ import annotations

import torch

from models.common.utility_functions import comp_pcc
from models.experimental.ace_step_v1_5.torch_ref.dit_decoder_core import make_tiny_state_dict
from models.experimental.ace_step_v1_5.ttnn_impl.dit_decoder_core import AceStepDecoderConfigTTNN
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC_THRESHOLD = 0.99


def assert_pcc_print(label: str, expected, actual, *, pcc: float = PCC_THRESHOLD) -> float:
    """Print PCC score, then assert ``comp_pcc`` passes at ``pcc`` (default 0.99)."""
    import ttnn

    def _to_torch(t):
        if isinstance(t, ttnn.Tensor):
            t = ttnn.to_torch(t)
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        return t

    exp = _to_torch(expected)
    act = _to_torch(actual)
    passed, score = comp_pcc(exp, act, pcc=pcc)
    print(
        f"[ace_step_v1_5][PCC] {label}: {float(score):.6f} (threshold={pcc}, ok={passed})",
        flush=True,
    )
    assert_with_pcc(expected, actual, pcc=pcc)
    return float(score)


def tiny_dit_decoder_fixture(
    *,
    seq_len: int = 32,
    enc_len: int = 16,
    n_heads: int = 4,
    head_dim: int = 32,
    cond_dim: int = 32,
    intermediate: int = 256,
    num_layers: int = 1,
    n_kv_heads: int | None = None,
):
    """
    Tiny config/state for ``dit_decoder_core`` PCC tests.

    ``head_dim=32`` keeps SDPA tile-aligned (same constraint as ``test_pcc_dit_decoder_core``).
    """
    n_kv = int(n_kv_heads) if n_kv_heads is not None else int(n_heads)
    d_model = int(n_heads * head_dim)
    cfg = AceStepDecoderConfigTTNN(
        hidden_size=d_model,
        num_hidden_layers=num_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        sliding_window=None,
        max_position_embeddings=max(seq_len, enc_len, 512),
    )
    sd = make_tiny_state_dict(
        d_model=d_model,
        n_heads=n_heads,
        head_dim=head_dim,
        cond_dim=cond_dim,
        intermediate=intermediate,
        num_layers=num_layers,
        n_kv_heads=n_kv,
    )
    return cfg, sd, d_model, seq_len, enc_len


def modulation_chunks_torch(
    state_dict: dict,
    *,
    layer_idx: int,
    timestep_proj_b6d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate) like ``TtAceStepDiTLayer``."""
    sst = torch.from_numpy(state_dict[f"layers.{layer_idx}.scale_shift_table"]).to(torch.bfloat16)
    sst = sst + timestep_proj_b6d
    chunks = [sst[:, i : i + 1, :].unsqueeze(2) for i in range(6)]
    return chunks[0], chunks[1], chunks[2], chunks[3], chunks[4], chunks[5]
