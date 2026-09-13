# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Host-only sizing checks; runnable with pytest --noconftest without TTNN."""

import sys
from types import SimpleNamespace

import pytest

from models.demos.common.prefill.fabric import (
    FABRIC_MAX_PAYLOAD_SIZE_BYTES,
    MOE_FABRIC_DTYPE_SIZE_BYTES,
    MOE_FABRIC_HEADER_SIZE_BYTES,
    create_fabric_router_config,
    moe_fabric_payload_size,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_2_config import DeepseekV32Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.reference.gpt_oss_20b_config import GptOss20BConfig
from models.demos.deepseek_v3_d_p.reference.gpt_oss_120b_config import GptOss120BConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.minimax_m2_7_config import MiniMaxM27Config
from models.demos.deepseek_v3_d_p.reference.minimax_m3_config import MiniMaxM3Config
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import MistralSmall4Config

MODEL_CONFIGS = [
    DeepseekV32Config,
    DeepSeekV3Config,
    DeepSeekV4FlashConfig,
    DeepSeekV4ProConfig,
    GLM51Config,
    GLM52Config,
    GptOss20BConfig,
    GptOss120BConfig,
    KimiK26Config,
    KimiK27Config,
    KimiK3Config,
    MiniMaxM27Config,
    MiniMaxM3Config,
    MistralSmall4Config,
]


@pytest.mark.parametrize("model_config", MODEL_CONFIGS, ids=lambda config: config.__name__)
def test_model_payload_fits_one_bf16_embedding_row_and_header(model_config):
    expected = model_config.EMB_SIZE * MOE_FABRIC_DTYPE_SIZE_BYTES + MOE_FABRIC_HEADER_SIZE_BYTES
    assert model_config.FABRIC_PAYLOAD_SIZE == expected


@pytest.mark.parametrize("model_config", MODEL_CONFIGS, ids=lambda config: config.__name__)
def test_payload_helper_uses_shared_transport_constants(model_config):
    assert moe_fabric_payload_size(model_config.EMB_SIZE) == model_config.FABRIC_PAYLOAD_SIZE


@pytest.mark.parametrize("model_config", MODEL_CONFIGS, ids=lambda config: config.__name__)
@pytest.mark.parametrize("arch", FABRIC_MAX_PAYLOAD_SIZE_BYTES)
def test_router_config_applies_architecture_limit(monkeypatch, arch, model_config):
    # Exercise the entry point shared by the runner and operator fixtures without opening hardware.
    monkeypatch.setitem(
        sys.modules,
        "ttnn",
        SimpleNamespace(
            get_arch_name=lambda: arch,
            _ttnn=SimpleNamespace(fabric=SimpleNamespace(FabricRouterConfig=SimpleNamespace)),
        ),
    )
    config = create_fabric_router_config(model_config.FABRIC_PAYLOAD_SIZE)
    assert config.max_packet_payload_size_bytes == min(
        model_config.FABRIC_PAYLOAD_SIZE, FABRIC_MAX_PAYLOAD_SIZE_BYTES[arch]
    )
