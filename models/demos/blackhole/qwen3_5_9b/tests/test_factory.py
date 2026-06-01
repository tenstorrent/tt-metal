# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared builders for qwen3_5_9b unit tests."""

from models.demos.blackhole.qwen3_5_9b.tt.common import create_tt_model


def build_model(mesh_device, n_layers=None, max_seq_len=2048):
    """Return (args, model, state_dict) for tests; honors HF_MODEL."""
    return create_tt_model(mesh_device, max_seq_len=max_seq_len, n_layers=n_layers)
