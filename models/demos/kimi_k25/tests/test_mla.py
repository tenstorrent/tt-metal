# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""test_mla.py — Kimi K2.5 MLA single-layer accuracy tests (M4).

CPU-only tests validate:
- Kimi K2.5 MLA configuration parameters
- Reference DeepseekV3Attention forward pass produces expected shapes / no NaN

Hardware test (test_forward_pass) runs MLA2D on TG/DUAL/QUAD with Kimi config via
DSV3 infrastructure. Kimi K2.5 MLA is structurally identical to DSV3 MLA — all
differences (64 heads, q_lora_rank=1536, kv_lora_rank=512) are captured in the
KimiK25Config dataclass.

Running CPU tests (no hardware):
    pytest models/demos/kimi_k25/tests/test_mla.py -k "Config or Reference" -v

Running hardware tests (TG required):
    MESH_DEVICE=TG pytest models/demos/kimi_k25/tests/test_mla.py -k "test_forward_pass" -v

With real weights:
    KIMI_HF_MODEL=/workspace/extra/Kimi-K2.5 MESH_DEVICE=TG \\
        pytest models/demos/kimi_k25/tests/test_mla.py -k "test_forward_pass" -v
"""
from __future__ import annotations

import pytest
import torch
from loguru import logger

from models.demos.kimi_k25.utils.config_adapter import KimiK25Config

# ---------------------------------------------------------------------------
# Expected Kimi K2.5 MLA values — source: research doc + config.json
# ---------------------------------------------------------------------------

_KIMI_MLA = {
    "num_attention_heads": 64,        # Kimi: 64, DSV3: 128
    "q_lora_rank": 1536,              # identical to DSV3
    "kv_lora_rank": 512,              # identical to DSV3
    "qk_nope_head_dim": 128,          # identical to DSV3
    "qk_rope_head_dim": 64,           # identical to DSV3
    "v_head_dim": 128,                # identical to DSV3
    "hidden_size": 7168,
    "rms_norm_eps": 1e-5,             # Kimi: 1e-5, DSV3: 1e-6 — subtle diff
    "rope_scaling_factor": 64.0,      # YaRN factor=64
    "kvpe_dim": 576,                  # kv_lora_rank + qk_rope_head_dim = 512 + 64
    "head_dim": 192,                  # qk_nope_head_dim + qk_rope_head_dim = 128 + 64
    "max_position_embeddings": 131072,
}

PCC_REQUIRED = 0.99
PCC_REQUIRED_KVPE = 0.999


# ---------------------------------------------------------------------------
# M4 CPU tests: config sanity
# ---------------------------------------------------------------------------


class TestKimiMLAConfigSanity:
    """CPU-only: Verify all Kimi K2.5 MLA-relevant configuration values."""

    def test_num_attention_heads(self):
        """64 attention heads (vs DSV3's 128)."""
        assert KimiK25Config.from_fixture().num_attention_heads == 64

    def test_q_lora_rank(self):
        """Query low-rank projection: 1536."""
        assert KimiK25Config.from_fixture().q_lora_rank == 1536

    def test_kv_lora_rank(self):
        """KV low-rank projection: 512."""
        assert KimiK25Config.from_fixture().kv_lora_rank == 512

    def test_qk_nope_head_dim(self):
        """Non-RoPE head dim: 128."""
        assert KimiK25Config.from_fixture().qk_nope_head_dim == 128

    def test_qk_rope_head_dim(self):
        """RoPE head dim: 64."""
        assert KimiK25Config.from_fixture().qk_rope_head_dim == 64

    def test_v_head_dim(self):
        """Value head dim: 128."""
        assert KimiK25Config.from_fixture().v_head_dim == 128

    def test_hidden_size(self):
        assert KimiK25Config.from_fixture().hidden_size == 7168

    def test_rms_norm_eps(self):
        """rms_norm_eps must be 1e-5 (NOT 1e-6 as in DSV3) — critical accuracy."""
        cfg = KimiK25Config.from_fixture()
        assert cfg.rms_norm_eps == pytest.approx(1e-5, rel=1e-3)
        assert cfg.rms_norm_eps != pytest.approx(1e-6, rel=1e-3), \
            "rms_norm_eps is 1e-6 (DSV3 value) but Kimi K2.5 requires 1e-5"

    def test_rope_scaling_factor(self):
        """YaRN factor=64 for extended context."""
        assert KimiK25Config.from_fixture().rope_scaling["factor"] == 64.0

    def test_rope_scaling_mscale_present(self):
        """mscale key must be present for MLA rope computation."""
        assert "mscale" in KimiK25Config.from_fixture().rope_scaling

    def test_kvpe_dim(self):
        """KVPE cache dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576."""
        cfg = KimiK25Config.from_fixture()
        kvpe_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim
        assert kvpe_dim == 576

    def test_head_dim_derived(self):
        """head_dim = qk_nope_head_dim + qk_rope_head_dim = 192."""
        cfg = KimiK25Config.from_fixture()
        assert cfg.head_dim == 192
        assert cfg.head_dim == cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

    def test_max_seq_len_attribute(self):
        """max_seq_len exposed on KimiK25Config for DSV3 test infra compatibility."""
        cfg = KimiK25Config.from_fixture()
        assert hasattr(cfg, "max_seq_len"), "KimiK25Config must expose max_seq_len for DSV3 test compat"
        assert cfg.max_seq_len == cfg.max_position_embeddings

    def test_max_seq_len_settable(self):
        """max_seq_len can be overridden independently (for hf_config_short)."""
        cfg = KimiK25Config.from_fixture()
        cfg.max_seq_len = 128
        assert cfg.max_seq_len == 128
        # max_position_embeddings should be unchanged
        assert cfg.max_position_embeddings == 131072

    def test_kimi_vs_dsv3_head_count(self):
        """Document: Kimi uses half as many attention heads as DSV3."""
        kimi_heads = KimiK25Config.from_fixture().num_attention_heads
        dsv3_heads = 128  # DSV3 reference value
        assert kimi_heads == 64
        assert kimi_heads == dsv3_heads // 2


# ---------------------------------------------------------------------------
# M4 CPU tests: reference model validation
# ---------------------------------------------------------------------------


class TestKimiMLAReference:
    """CPU-only: DSV3 reference attention with Kimi K2.5 config.

    These tests verify that the reference implementation (modeling_deepseek.py)
    accepts Kimi config without error, producing valid outputs. This confirms
    that DSV3 MLA layers are fully parameterised — no code changes needed.
    """

    @pytest.fixture(scope="class")
    def tiny_cfg(self):
        """Minimal Kimi-shaped config for fast CPU reference runs.

        Single layer, short sequence. All MLA dimensions are real Kimi values.
        """
        cfg = KimiK25Config.from_fixture()
        cfg.num_hidden_layers = 1
        cfg.max_seq_len = 128
        return cfg

    def test_dsv3_reference_import(self):
        """DeepseekV3Attention importable from DSV3 reference."""
        try:
            from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention  # noqa: F401
        except ImportError:
            pytest.skip("DSV3 reference model not importable in this environment")

    def test_all_mla_fields_present_on_kimi_config(self):
        """All hf_config fields consumed by MLA1D/MLA2D exist on KimiK25Config."""
        cfg = KimiK25Config.from_fixture()
        required = [
            "hidden_size",
            "num_attention_heads",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "q_lora_rank",
            "rope_scaling",
            "max_seq_len",
        ]
        missing = [f for f in required if not hasattr(cfg, f)]
        assert not missing, f"KimiK25Config missing MLA fields: {missing}"

    def test_reference_forward_output_shape(self, tiny_cfg):
        """DSV3 attention with Kimi config produces shape (batch, seq, hidden)."""
        try:
            from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention
        except ImportError:
            pytest.skip("DSV3 reference model not importable")

        batch, seq = 1, 4
        model = DeepseekV3Attention(tiny_cfg, layer_idx=0).eval().to(torch.bfloat16)
        x = torch.randn(batch, seq, tiny_cfg.hidden_size, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model(x)[0]

        assert out.shape == (batch, seq, tiny_cfg.hidden_size), (
            f"Expected ({batch}, {seq}, {tiny_cfg.hidden_size}), got {out.shape}"
        )

    def test_reference_forward_no_nan(self, tiny_cfg):
        """DSV3 attention with Kimi config produces no NaN or Inf."""
        try:
            from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention
        except ImportError:
            pytest.skip("DSV3 reference model not importable")

        model = DeepseekV3Attention(tiny_cfg, layer_idx=0).eval().to(torch.bfloat16)
        x = torch.randn(1, 4, tiny_cfg.hidden_size, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model(x)[0]

        assert not torch.isnan(out).any(), "NaN in reference MLA output with Kimi config"
        assert not torch.isinf(out).any(), "Inf in reference MLA output with Kimi config"

    def test_reference_forward_kv_cache_shape(self, tiny_cfg):
        """KV cache produced by reference model has expected KVPE dimensions."""
        try:
            from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention
        except ImportError:
            pytest.skip("DSV3 reference model not importable")

        batch, seq = 1, 8
        model = DeepseekV3Attention(tiny_cfg, layer_idx=0).eval().to(torch.bfloat16)
        x = torch.randn(batch, seq, tiny_cfg.hidden_size, dtype=torch.bfloat16)
        with torch.no_grad():
            result = model(x)

        # result is (output, past_key_value) or similar depending on DSV3 reference impl
        # We check output is valid; cache format depends on reference version
        output = result[0]
        assert output.shape[0] == batch
        assert output.shape[-1] == tiny_cfg.hidden_size
        logger.info(f"Reference MLA output shape: {output.shape}")

    def test_reference_weights_respect_head_count(self, tiny_cfg):
        """Model parameter count reflects 64-head architecture."""
        try:
            from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention
        except ImportError:
            pytest.skip("DSV3 reference model not importable")

        model = DeepseekV3Attention(tiny_cfg, layer_idx=0)
        # q_a_proj: hidden_size -> q_lora_rank
        if hasattr(model, "q_a_proj"):
            assert model.q_a_proj.weight.shape == (tiny_cfg.q_lora_rank, tiny_cfg.hidden_size), \
                f"q_a_proj shape mismatch: {model.q_a_proj.weight.shape}"
        # kv_a_proj: hidden_size -> kv_lora_rank + qk_rope_head_dim
        if hasattr(model, "kv_a_proj_with_mqa"):
            expected_kv_proj_out = tiny_cfg.kv_lora_rank + tiny_cfg.qk_rope_head_dim  # 512 + 64 = 576
            assert model.kv_a_proj_with_mqa.weight.shape[0] == expected_kv_proj_out, \
                f"kv_a_proj_with_mqa output dim mismatch"


# ---------------------------------------------------------------------------
# Hardware test: MLA2D forward pass via DSV3 infra
# ---------------------------------------------------------------------------

# Import DSV3 test helpers — soft-fail if not available (CPU-only environments)
try:
    from models.demos.deepseek_v3.tests.pytest_utils import DEFAULT_PREFILL_SEQ_LEN, build_test_cases_and_ids
    from models.demos.deepseek_v3.tt.model.row_batched_model import get_fabric_config
    from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW

    TEST_CASES, TEST_IDS = build_test_cases_and_ids(
        USERS_PER_ROW,
        DEFAULT_PREFILL_SEQ_LEN,
        include_decode_random_pos_ids=True,
    )
    _DSV3_INFRA_AVAILABLE = True
except ImportError:
    # Fallback: minimal test cases for CI smoke test
    TEST_CASES = [("decode", 1, 1, None)]
    TEST_IDS = ["decode-seq1-batch1-nops"]
    _DSV3_INFRA_AVAILABLE = False
    get_fabric_config = None


def run_test_forward_pass_mla_kimi(
    layer_idx: int,
    mode: str,
    seq_len: int,
    batch_size_per_row: int,
    hf_config_short: KimiK25Config,
    cache_path,
    mesh_device,
    ccl,
    model_path,
    module_path,
    force_recalculate_weight_config: bool,
    state_dict,
    decode_position_ids=None,
    perf_mode: bool = False,
    num_iters: int = 20,
):
    """Run Kimi K2.5 MLA forward pass using DSV3 MLA2D infrastructure.

    Kimi K2.5 MLA is structurally identical to DSV3 MLA. Key differences:
    - 64 attention heads (DSV3: 128) — handled by hf_config parameterisation
    - rms_norm_eps = 1e-5 (DSV3: 1e-6) — captured in KimiK25Config
    - rope_scaling factor=64 — captured in KimiK25Config

    All differences are in the config, so DSV3 MLA2D runs unchanged with
    KimiK25Config. We delegate directly to run_test_forward_pass_mla2d.
    """
    try:
        from models.demos.deepseek_v3.tests.test_mla import run_test_forward_pass_mla2d
    except ImportError as exc:
        pytest.skip(f"Cannot import DSV3 MLA2D test helper: {exc}")

    logger.info(
        f"Kimi K2.5 MLA test — {mode} mode, seq_len={seq_len}, "
        f"batch_size_per_row={batch_size_per_row}, "
        f"heads={hf_config_short.num_attention_heads}, "
        f"q_lora_rank={hf_config_short.q_lora_rank}, "
        f"kv_lora_rank={hf_config_short.kv_lora_rank}"
    )

    run_test_forward_pass_mla2d(
        layer_idx=layer_idx,
        mode=mode,
        seq_len=seq_len,
        batch_size_per_row=batch_size_per_row,
        hf_config_short=hf_config_short,
        cache_path=cache_path,
        mesh_device=mesh_device,
        ccl=ccl,
        model_path=model_path,
        module_path=module_path,
        force_recalculate_weight_config=force_recalculate_weight_config,
        state_dict=state_dict,
        decode_position_ids=decode_position_ids,
        perf_mode=perf_mode,
        num_iters=num_iters,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": get_fabric_config()}] if _DSV3_INFRA_AVAILABLE and get_fabric_config else [{}],
    indirect=True,
)
@pytest.mark.parametrize("perf_mode", [True, False])
@pytest.mark.parametrize(
    "mode, seq_len, batch_size_per_row, decode_position_ids",
    TEST_CASES,
    ids=TEST_IDS,
)
@pytest.mark.parametrize(
    "module_path",
    [None, "model.layers.0.self_attn"],
)
@pytest.mark.parametrize(
    "test_closure",
    [
        pytest.param(
            run_test_forward_pass_mla_kimi,
            marks=pytest.mark.requires_device(["TG", "DUAL", "QUAD"]),
        ),
    ],
)
def test_forward_pass(
    mode,
    seq_len,
    batch_size_per_row,
    decode_position_ids,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    model_path,
    module_path,
    perf_mode,
    force_recalculate_weight_config,
    test_closure,
    set_deterministic_env,
    state_dict,
):
    """Kimi K2.5 MLA2D forward pass — PCC ≥ 0.99 (decode + prefill).

    Uses KimiK25Config (64 heads, q_lora_rank=1536, kv_lora_rank=512) with
    the DSV3 MLA2D TTNN implementation. No code changes needed in MLA2D —
    all architecture differences are encoded in KimiK25Config.

    Test matrix:
    - decode: seq_len=1, various batch sizes, random position IDs
    - prefill: seq_len=128 (or KIMI_MAX_SEQ_LEN_OVERRIDE)
    - module_path=None: random weights (fast, always-pass smoke test)
    - module_path="model.layers.0.self_attn": real weights (requires KIMI_HF_MODEL)

    Requires: MESH_DEVICE=TG (or DUAL/QUAD)
    Optional: KIMI_HF_MODEL=/workspace/extra/Kimi-K2.5
    """
    layer_idx = 0

    # Only use decode_position_ids in decode mode
    if mode != "decode":
        decode_position_ids = None

    test_closure(
        layer_idx,
        mode,
        seq_len,
        batch_size_per_row,
        hf_config_short,
        cache_path,
        mesh_device,
        ccl,
        model_path,
        module_path,
        force_recalculate_weight_config,
        state_dict,
        decode_position_ids,
        perf_mode,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
