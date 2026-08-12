# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import torch

from models.demos.blackhole.qwen36.utils.substate import has_substate, substate


def test_substate_strips_prefix_and_filters():
    state = {
        "mlp.gate_proj.weight": torch.zeros(2),
        "mlp.up_proj.weight": torch.zeros(2),
        "self_attn.q_proj.weight": torch.zeros(2),
    }
    sub = substate(state, "mlp")
    assert set(sub.keys()) == {"gate_proj.weight", "up_proj.weight"}
    assert "self_attn.q_proj.weight" not in sub


def test_has_substate():
    state = {"linear_attn.qkv_proj.weight": torch.zeros(1)}
    assert has_substate(state, "linear_attn") is True
    assert has_substate(state, "mlp") is False
