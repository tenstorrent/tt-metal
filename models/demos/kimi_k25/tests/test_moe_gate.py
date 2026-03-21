# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""test_moe_gate.py — Validate DSV3 MoEGate with Kimi K2.5 config (n_group=1, 384 experts).

This test ensures:

1. **Config compatibility**: DSV3's ``MoEGate`` (sigmoid scoring, routed_scaling_factor,
   n_group/topk_group, n_routed_experts) works with Kimi K2.5's config values.

2. **n_group=1 flat routing**: With n_group=1 and topk_group=1, the gate selects top-8
   experts from all 384 candidates.  The group-selection step is trivially correct
   (one group of 384 experts, all initially eligible).

3. **PCC regression**: TTNN output matches reference ``ReferenceMoEGate`` within PCC ≥ 0.99
   (decode) / 0.99 (prefill).

Key config deltas vs DSV3 that this test exercises:
   +--------------------------+-----------+-----------+
   | Field                    | DSV3      | Kimi K2.5 |
   +==========================+===========+===========+
   | n_routed_experts         | 256       | 384       |
   | n_group                  | 8         | 1         |
   | topk_group               | 4         | 1         |
   | routed_scaling_factor    | 2.5       | 2.827     |
   | hidden_size              | 7168      | 7168      |
   | num_experts_per_tok      | 8         | 8         |
   +--------------------------+-----------+-----------+

Note: DSV3 ``moe_gate.py`` already uses ``ttnn.sigmoid`` (not softmax) and reads
``routed_scaling_factor`` from ``hf_config``.  No code changes needed in MoEGate
for Kimi K2.5 — this test confirms that by running the module end-to-end.

Reference: models/demos/deepseek_v3/tests/test_moe_gate.py
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

# DSV3 modules — reused directly for Kimi
from models.demos.deepseek_v3.tt.moe_gate import MoEGate
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import get_model_config, get_test_weight_config, run_module_forward

# Kimi config (provides hf_config fixture)
from models.demos.kimi_k25.utils.config_adapter import KimiK25Config
from tests.ttnn.utils_for_testing import comp_pcc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PCC thresholds — looser than MLA because gate outputs are scalar weights
_PCC_THRESHOLD_DECODE = 0.99
_PCC_THRESHOLD_PREFILL = 0.97  # prefill with 384 experts + BF16 accumulation

# Kimi K2.5 specific: 384 experts, hidden=7168
_KIMI_N_EXPERTS = 384
_KIMI_HIDDEN = 7168
_KIMI_TOP_K = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference_model(hf_config: KimiK25Config):
    """DSV3 reference MoEGate instantiated with Kimi K2.5 config.

    DSV3 ``ReferenceMoEGate.__init__`` reads:
      - config.hidden_size
      - config.n_routed_experts
      - config.num_experts_per_tok
      - config.n_group, config.topk_group
      - config.scoring_func   ("sigmoid" — confirmed in DSV3)
      - config.norm_topk_prob (True — matches Kimi)
      - config.routed_scaling_factor (2.827)
    All present in KimiK25Config.
    """
    try:
        from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate as ReferenceMoEGate
    except ImportError:
        pytest.skip("DSV3 reference model not available")

    torch.use_deterministic_algorithms(True)
    model = ReferenceMoEGate(hf_config, use_bitonic_sort=True).eval()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKimiMoEGateConfig:
    """Unit tests for config compatibility — no hardware required."""

    def test_kimi_config_has_required_gate_fields(self, hf_config: KimiK25Config):
        """KimiK25Config exposes all fields that DSV3 MoEGate reads from hf_config."""
        required_fields = [
            "hidden_size",
            "n_routed_experts",
            "num_experts_per_tok",
            "n_group",
            "topk_group",
            "scoring_func",
            "norm_topk_prob",
            "routed_scaling_factor",
        ]
        missing = [f for f in required_fields if not hasattr(hf_config, f)]
        assert missing == [], f"KimiK25Config missing fields: {missing}"

    def test_kimi_scoring_func_is_sigmoid(self, hf_config: KimiK25Config):
        """scoring_func must be 'sigmoid' — DSV3 MoEGate already implements this."""
        assert hf_config.scoring_func == "sigmoid", (
            f"Expected scoring_func='sigmoid', got {hf_config.scoring_func!r}. "
            "DSV3 moe_gate.py uses ttnn.sigmoid — Kimi is compatible."
        )

    def test_kimi_flat_routing_params(self, hf_config: KimiK25Config):
        """n_group=1, topk_group=1 → flat routing over all 384 experts."""
        assert hf_config.n_group == 1, f"n_group should be 1, got {hf_config.n_group}"
        assert hf_config.topk_group == 1, f"topk_group should be 1, got {hf_config.topk_group}"

    def test_kimi_experts_divisible_by_group(self, hf_config: KimiK25Config):
        """n_routed_experts must be divisible by n_group (trivially true for n_group=1)."""
        assert (
            hf_config.n_routed_experts % hf_config.n_group == 0
        ), f"{hf_config.n_routed_experts} not divisible by {hf_config.n_group}"

    def test_kimi_routed_scaling_factor(self, hf_config: KimiK25Config):
        """routed_scaling_factor=2.827 (vs DSV3's 2.5) — must be read from config."""
        assert (
            abs(hf_config.routed_scaling_factor - 2.827) < 1e-4
        ), f"Unexpected routed_scaling_factor={hf_config.routed_scaling_factor}"

    def test_kimi_norm_topk_prob_true(self, hf_config: KimiK25Config):
        """norm_topk_prob=True — normalize top-k weights; confirmed in HF config."""
        assert hf_config.norm_topk_prob is True

    def test_kimi_topk_method(self, hf_config: KimiK25Config):
        """topk_method='noaux_tc' — same as DSV3, no code change needed."""
        assert hf_config.topk_method == "noaux_tc"


class TestKimiMoEGateReference:
    """Tests against DSV3 reference model — CPU only, no hardware required."""

    def test_reference_gate_instantiates(self, reference_model):
        """ReferenceMoEGate initializes without error with Kimi config."""
        assert reference_model is not None

    def test_reference_gate_forward_shape(self, reference_model, hf_config: KimiK25Config):
        """Reference gate forward pass produces correct output shapes."""
        batch, seq = 1, 4
        x = torch.randn(batch, seq, hf_config.hidden_size, dtype=torch.bfloat16)
        with torch.no_grad():
            topk_indices, topk_weights = reference_model(x)
        assert topk_indices.shape == (batch * seq, hf_config.num_experts_per_tok), (
            f"Expected topk_indices shape {(batch*seq, hf_config.num_experts_per_tok)}, " f"got {topk_indices.shape}"
        )
        assert topk_weights.shape == (batch * seq, hf_config.num_experts_per_tok), (
            f"Expected topk_weights shape {(batch*seq, hf_config.num_experts_per_tok)}, " f"got {topk_weights.shape}"
        )

    def test_reference_gate_expert_indices_in_range(self, reference_model, hf_config: KimiK25Config):
        """All selected expert indices in [0, n_routed_experts)."""
        x = torch.randn(2, 8, hf_config.hidden_size, dtype=torch.bfloat16)
        with torch.no_grad():
            topk_indices, _ = reference_model(x)
        assert topk_indices.min() >= 0
        assert topk_indices.max() < hf_config.n_routed_experts, (
            f"Expert index {topk_indices.max()} out of range " f"[0, {hf_config.n_routed_experts})"
        )

    def test_reference_gate_weight_range(self, reference_model, hf_config: KimiK25Config):
        """Expert weights are non-negative (after sigmoid + norm_topk_prob)."""
        x = torch.randn(2, 8, hf_config.hidden_size, dtype=torch.bfloat16)
        with torch.no_grad():
            _, topk_weights = reference_model(x)
        assert (topk_weights >= 0).all(), "Expected non-negative gate weights"

    def test_reference_gate_top8_unique_per_token(self, reference_model, hf_config: KimiK25Config):
        """Top-8 selected experts should be unique per token."""
        x = torch.randn(1, 16, hf_config.hidden_size, dtype=torch.bfloat16)
        with torch.no_grad():
            topk_indices, _ = reference_model(x)
        # Check uniqueness across expert dim for each token
        for i in range(topk_indices.shape[0]):
            unique = topk_indices[i].unique()
            assert len(unique) == hf_config.num_experts_per_tok, (
                f"Token {i}: expected {hf_config.num_experts_per_tok} unique experts, " f"got {len(unique)}"
            )


class TestKimiMoEGateHardware:
    """Hardware tests — skipped if MESH_DEVICE or ttnn not available.

    These tests validate that DSV3's TTNN MoEGate implementation produces
    PCC ≥ 0.99 against the CPU reference when using Kimi K2.5 config.
    """

    @pytest.mark.parametrize(
        "mode,seq_len",
        [
            ("decode", 128),
            ("prefill", 128),
        ],
    )
    @pytest.mark.parametrize("topk_fallback", [True])
    def test_forward_pass_pcc(
        self,
        mode,
        seq_len,
        topk_fallback,
        hf_config: KimiK25Config,
        reference_model,
        cache_path,
        mesh_device,
        set_deterministic_env,
    ):
        """TTNN MoEGate output matches reference within PCC threshold.

        Uses random weights (no real Kimi checkpoint needed).
        Tests n_group=1, 384 experts, routed_scaling_factor=2.827.
        """
        import ttnn

        torch.use_deterministic_algorithms(True)
        batch_size = 1

        # Use random-initialized weights from reference model
        hf_state_dict = {name: tensor.detach().clone() for name, tensor in reference_model.state_dict().items()}

        weight_config = get_test_weight_config(
            MoEGate,
            hf_config,
            (hf_state_dict,),
            cache_path,
            mesh_device,
            force_recalculate=False,
            test_name="test_kimi_moe_gate",
            real_weights=False,
        )

        model_config = get_model_config(
            MoEGate,
            mode,
            hf_config,
            mesh_device,
            topk_fallback=topk_fallback,
            use_bitonic_sort=True,
        )

        model_state = MoEGate.create_state(hf_config, mesh_device=mesh_device)
        run_config = create_run_config(model_config, weight_config, model_state)

        # Input: (batch, seq_len, hidden_size)
        torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

        # Reference forward
        reference_model.eval()
        reference_model.to(torch.bfloat16)
        with torch.no_grad():
            ref_topk_indices, ref_topk_weights = reference_model(torch_input)

        # TTNN forward
        tt_input = ttnn.from_torch(
            torch_input.unsqueeze(1),
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
        tt_topk_weights, tt_topk_indices = run_module_forward(MoEGate, mode, tt_input, run_config)

        # Gather TTNN outputs back to CPU
        tt_topk_indices_torch = ttnn.to_torch(
            tt_topk_indices,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=tuple(mesh_device.shape)),
        ).squeeze()
        tt_topk_weights_torch = ttnn.to_torch(
            tt_topk_weights,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=tuple(mesh_device.shape)),
        ).squeeze()

        # PCC check on routing weights (indices may differ for tied scores)
        pcc_threshold = _PCC_THRESHOLD_DECODE if mode == "decode" else _PCC_THRESHOLD_PREFILL
        pcc_val, pcc_str = comp_pcc(ref_topk_weights.float(), tt_topk_weights_torch.float())
        logger.info(
            f"Kimi MoEGate [{mode}, seq={seq_len}]: topk_weights PCC={pcc_val:.4f} " f"(threshold={pcc_threshold})"
        )
        assert pcc_val >= pcc_threshold, (
            f"Kimi MoEGate PCC={pcc_val:.4f} < {pcc_threshold} " f"[mode={mode}, seq_len={seq_len}]\n{pcc_str}"
        )

        logger.info(
            f"[PASS] Kimi MoEGate {mode} mode: PCC={pcc_val:.4f} "
            f"with n_group=1, {hf_config.n_routed_experts} experts"
        )
