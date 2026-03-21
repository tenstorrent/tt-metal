# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""test_moe.py — M4 single-layer MoE accuracy test for Kimi K2.5.

Validates that DSV3's ``MoE`` module produces output matching the HF reference
``DeepseekV3MoE`` when configured with Kimi K2.5 parameters.

Target: PCC ≥ 0.98 (hidden-dim) for both decode and prefill modes.

Kimi-specific differences tested here:
  - 384 routed experts (vs 256 in DSV3)
  - n_group=1 flat routing
  - routed_scaling_factor=2.827
  - moe_intermediate_size=2048
  - Weights loaded via KimiLazyStateDict (INT4 → BF16 dequant) when real

Milestone: M4 (Single-Layer Accuracy)
Reference: models/demos/deepseek_v3/tests/test_moe.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.model.row_batched_model import get_fabric_config
from models.demos.deepseek_v3.tt.moe import MoE
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    get_test_weight_config,
    run_module_forward,
)
from models.demos.kimi_k25.utils.config_adapter import KimiK25Config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PCC_REQUIRED = 0.98

# Layer 3 is a MoE layer (first_k_dense_replace=1 → layers 1..60 are MoE).
# In the Kimi checkpoint the path is: language_model.model.layers.3.mlp
# KimiLazyStateDict strips 'language_model.' → DSV3 convention: model.layers.3.mlp
_MOE_LAYER_PATH = "model.layers.3.mlp"

_MAX_SEQ_LEN_ENV = os.getenv("DEEPSEEK_MAX_SEQ_LEN_OVERRIDE")
_PREFILL_SEQ_LEN = int(_MAX_SEQ_LEN_ENV) if _MAX_SEQ_LEN_ENV else 128


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_model(hf_config: KimiK25Config):
    """DSV3 DeepseekV3MoE reference model configured with Kimi K2.5 params."""
    try:
        from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE
    except ImportError:
        pytest.skip("DSV3 reference model not available")
    torch.use_deterministic_algorithms(True)
    # Disable shared experts for isolated MoE test (mirrors DSV3 test pattern)
    cfg = KimiK25Config.from_fixture()
    cfg.n_shared_experts = None  # type: ignore[assignment]
    return DeepseekV3MoE(cfg).eval()


def _get_test_state_dict(
    reference_model,
    weight_type: str,
    checkpoint_state_dict,
    hf_config: KimiK25Config,
) -> dict:
    """Prepare state dict for test: random (no weights needed) or real."""
    if weight_type == "random":
        return {name: tensor.detach().clone() for name, tensor in reference_model.state_dict().items()}

    # Real weights from KimiLazyStateDict
    assert weight_type == "real"
    assert checkpoint_state_dict is not None

    # KimiLazyStateDict strips 'language_model.' prefix, so DSV3 convention applies:
    # sub_state_dict(state_dict, "model.layers.3.mlp.") → routed experts
    moe_state_dict = {
        name: tensor
        for name, tensor in sub_state_dict(checkpoint_state_dict, f"{_MOE_LAYER_PATH}.").items()
        if not name.startswith("shared_experts.")
    }
    if not moe_state_dict:
        pytest.skip(f"Checkpoint does not contain routed MoE weights under '{_MOE_LAYER_PATH}'")
    reference_model.load_state_dict(moe_state_dict)
    return moe_state_dict


# ---------------------------------------------------------------------------
# Hardware tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": get_fabric_config()}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode,num_tokens",
    [
        ("decode", 128),
        ("prefill", _PREFILL_SEQ_LEN),
    ],
)
@pytest.mark.parametrize("topk_fallback", [True])
@pytest.mark.parametrize("weight_type", ["random"], ids=["random_weights"])
def test_forward_pass(
    device_params,
    mode,
    num_tokens,
    set_deterministic_env,
    reference_model,
    hf_config: KimiK25Config,
    request,
    cache_path,
    mesh_device,
    ccl,
    topk_fallback,
    weight_type,
    force_recalculate_weight_config,
):
    """Single-layer Kimi MoE forward pass PCC ≥ 0.98 vs reference.

    Exercises:
    - 384 routed experts on the mesh
    - n_group=1 flat expert routing
    - routed_scaling_factor=2.827 (applied by DSV3 MoEGate)
    - moe_intermediate_size=2048 (Kimi FFN hidden dim)
    """
    checkpoint_state_dict = request.getfixturevalue("state_dict") if weight_type == "real" else None
    state_dict = _get_test_state_dict(reference_model, weight_type, checkpoint_state_dict, hf_config)

    torch_input = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)

    # Reference forward
    reference_model.eval()
    reference_model.to(torch.bfloat16)
    with torch.no_grad():
        reference_output = reference_model(torch_input)

    weight_config = get_test_weight_config(
        MoE,
        hf_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=force_recalculate_weight_config,
        test_name="test_kimi_moe",
        real_weights=(weight_type == "real"),
        layer_id=_MOE_LAYER_PATH,
    )

    model_config = get_model_config(
        MoE,
        mode,
        hf_config,
        mesh_device,
        device_params["fabric_config"],
        topk_fallback=topk_fallback,
    )

    model_state = MoE.create_state(hf_config, mesh_device, ccl)
    model_shared_state = MoE.create_shared_state(hf_config, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_output = run_module_forward(MoE, mode, tt_input, run_config, handle_tensor_parallel=True)

    # Validate output memory config
    assert tt_output.memory_config() == run_config["output_memory_config"], (
        f"MoE output memory config mismatch: "
        f"expected {run_config['output_memory_config']}, "
        f"got {tt_output.memory_config()}"
    )

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    logger.info(
        f"Kimi MoE test: mode={mode}, num_tokens={num_tokens}, "
        f"experts={hf_config.n_routed_experts}, n_group={hf_config.n_group}"
    )
    assert_hidden_dim_pcc(tt_output_torch, reference_output.unsqueeze(0), pcc_required=_PCC_REQUIRED)
    logger.info(f"[PASS] Kimi MoE single-layer PCC ≥ {_PCC_REQUIRED} [{mode}, tokens={num_tokens}]")


# ---------------------------------------------------------------------------
# CPU-only sanity tests (no hardware required)
# ---------------------------------------------------------------------------


class TestKimiMoEConfigSanity:
    """Fast sanity checks — run on any machine, no hardware needed."""

    def test_moe_intermediate_size(self, hf_config: KimiK25Config):
        """Kimi MoE FFN hidden=2048 (DSV3 is also 2048 — same, no change)."""
        assert hf_config.moe_intermediate_size == 2048

    def test_n_routed_experts_384(self, hf_config: KimiK25Config):
        """384 routed experts (key Kimi delta from DSV3's 256)."""
        assert hf_config.n_routed_experts == 384

    def test_expert_sharding_tg(self, hf_config: KimiK25Config):
        """TG (32 devices): 384 / 32 = 12 experts/device (even division)."""
        assert hf_config.n_routed_experts % 32 == 0
        assert hf_config.experts_per_device_tg == 12

    def test_reference_model_forward(self, reference_model, hf_config: KimiK25Config):
        """Reference DeepseekV3MoE runs forward without error on CPU."""
        x = torch.randn(1, 4, hf_config.hidden_size, dtype=torch.bfloat16)
        with torch.no_grad():
            out = reference_model(x)
        assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
        assert not torch.isnan(out).any(), "NaN in reference MoE output"
        logger.info(f"[PASS] Reference Kimi MoE forward: " f"input {x.shape} → output {out.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
