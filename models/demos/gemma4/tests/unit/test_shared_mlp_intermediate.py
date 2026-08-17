# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from models.demos.gemma4.tt.shared_mlp import resolve_shared_mlp_intermediate_size


def test_resolve_intermediate_from_weights():
    state = {"gate_proj.weight": torch.zeros(12288, 1536)}
    cfg = SimpleNamespace(
        intermediate_size=6144, use_double_wide_mlp=True, num_kv_shared_layers=20, num_hidden_layers=35
    )
    assert resolve_shared_mlp_intermediate_size(cfg, state, layer_idx=15) == 12288


def test_resolve_double_wide_from_layer_idx():
    cfg = SimpleNamespace(
        intermediate_size=6144, use_double_wide_mlp=True, num_kv_shared_layers=20, num_hidden_layers=35
    )
    assert resolve_shared_mlp_intermediate_size(cfg, None, layer_idx=14) == 6144
    assert resolve_shared_mlp_intermediate_size(cfg, None, layer_idx=15) == 12288


def test_resolve_no_double_wide():
    cfg = SimpleNamespace(
        intermediate_size=10240, use_double_wide_mlp=False, num_kv_shared_layers=18, num_hidden_layers=42
    )
    assert resolve_shared_mlp_intermediate_size(cfg, None, layer_idx=24) == 10240
