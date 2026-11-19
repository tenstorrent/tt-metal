# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: which MoE a model builds, and the mesh it then opens with (tt/moe_impl.py)."""

import pytest

from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt import moe_impl


@pytest.fixture(autouse=True)
def _no_env_choice(monkeypatch):
    monkeypatch.delenv("MOE_IMPL", raising=False)
    monkeypatch.delenv("K3_MOE_IMPL", raising=False)


@pytest.mark.parametrize("cfg", [KimiK3Config, KimiK31Config, "KimiK3Config", "GLM53Config"])
def test_block_is_default(cfg):
    assert moe_impl.moe_impl_of(cfg) == moe_impl.BLOCK


@pytest.mark.parametrize("cfg", [None, "DeepSeekV3Config"])
def test_other_models_stay_on_ttmoe(cfg):
    assert moe_impl.moe_impl_of(cfg) == moe_impl.TTMOE


def test_k3_opt_out_reaches_k31(monkeypatch):
    monkeypatch.setenv("K3_MOE_IMPL", "ttmoe")
    assert moe_impl.moe_impl_of(KimiK31Config) == moe_impl.TTMOE
    assert moe_impl.moe_impl_of("GLM53Config") == moe_impl.BLOCK


def test_caller_choice_wins(monkeypatch):
    monkeypatch.setenv("MOE_IMPL", "block")
    assert moe_impl.moe_impl_of(KimiK31Config, "ttmoe") == moe_impl.TTMOE


def test_k31_sizes():
    # Words slab inherited from K3 (1712 B); L1_SMALL: K3.1's own 4880 B + slab + 64 B alignment.
    assert moe_impl.moe_block_words_slab(KimiK31Config) == 1712
    assert moe_impl.model_l1_small_with_block(KimiK31Config, 2048) == 4880 + 1712 + 64
    assert moe_impl.model_l1_small_with_block(KimiK3Config, 1536) == 3328 + 1712 + 64


def test_k31_mesh_params():
    # One latent combine row: 2 * 3584 + 32 = 7200 B, 64-B aligned to 7232.
    payload, l1 = moe_impl.moe_block_mesh_params(KimiK31Config, 7168, 2048)
    assert (payload, l1) == (7232, 6656)
    assert moe_impl.moe_block_mesh_params(KimiK31Config, 7168, 2048, moe_impl="ttmoe") == (7168, 2048)
