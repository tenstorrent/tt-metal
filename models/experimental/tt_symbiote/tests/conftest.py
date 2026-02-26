# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from models.experimental.tt_symbiote.modules.moe import Glm4MoeConfig


@pytest.fixture
def default_glm_config():
    """Default GLM configuration for testing."""
    return Glm4MoeConfig(
        hidden_size=2048,
        intermediate_size=10240,
        moe_intermediate_size=1536,
        num_local_experts=64,
        num_experts_per_tok=4,
        n_shared_experts=1,
        routed_scaling_factor=1.8,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
    )
