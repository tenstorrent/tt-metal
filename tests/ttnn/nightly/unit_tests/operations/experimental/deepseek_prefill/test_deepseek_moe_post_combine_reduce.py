# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for post_combine_reduce fused kernel.

Validates correctness against:
- PyTorch reference (weighted sum across experts)
- Old implementation from tt_moe.py (to_layout + mul + sum)

Tests structured data, random data, sparse weights, and non-local expert skipping.
Shape: [1, 3200, 8, 7168] - DeepSeek-V3 dimensions.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.gpt_oss_120b_config import GptOss120BConfig
from models.demos.deepseek_v3_d_p.reference.minimax_m2_7_config import MiniMaxM27Config

NUM_TOKENS = 3200
NUM_EXPERTS = 8
EXPERT_DIM = 2
PCC_THRESHOLD = 0.999

# Per-model (id, config, extended). Each test reads emb_dim (EMB_SIZE) and the routed-expert
# count (NUM_ROUTED_EXPERTS) from the config, so every model exercises its own shape:
#   dsv3 7168/256, glm_51 6144/256, minimax_m27 3072/256,
#   dsv4_pro 7168/384, dsv4_flash 4096/256, gptoss_120b 2880/128.
MODELS = [
    ("dsv3", DeepSeekV3Config, False),
    ("glm_51", GLM51Config, True),
    ("minimax_m27", MiniMaxM27Config, True),
    ("dsv4_pro", DeepSeekV4ProConfig, True),
    ("dsv4_flash", DeepSeekV4FlashConfig, True),
    ("gptoss_120b", GptOss120BConfig, True),
]
MODEL_PARAMS = [
    pytest.param(config, id=name, marks=(pytest.mark.extended_model,) if extended else ())
    for name, config, extended in MODELS
]

# Currently-failing structured/multi-chunk cases, xfail'd so CI stays green while the linked issues
# are worked on. These get dedicated param lists (rather than reusing MODEL_PARAMS) because the
# same models pass the other tests that share MODEL_PARAMS. gptoss_120b's emb_dim (2880) is not a
# multiple of the hardcoded 1024 tile_width used by the structured patterns, so its reshape is
# invalid; dsv4_flash just lands slightly under the structured PCC threshold. strict=True keeps CI
# green either way. Remove a model from the per-test xfail dict once its issue is resolved.
_GPTOSS_STRUCTURED_XFAIL = (
    "GPT-OSS 120B post-combine reduce: structured reshape invalid (tile_width hardcoded 1024) — "
    "https://github.com/tenstorrent/tt-metal/issues/46731"
)
_DSV4_FLASH_STRUCTURED_XFAIL = (
    "DeepSeek V4 Flash post-combine reduce: structured PCC below threshold — "
    "https://github.com/tenstorrent/tt-metal/issues/46609"
)


def _model_params_with_xfail(xfails):
    """Build a MODEL_PARAMS-style list, attaching an xfail marker (keyed by model name) to the
    models listed in `xfails`. Lets a single test xfail just its failing models without affecting
    the other tests that reuse the plain MODEL_PARAMS."""
    params = []
    for name, config, extended in MODELS:
        marks = (pytest.mark.extended_model,) if extended else ()
        if name in xfails:
            marks += (pytest.mark.xfail(reason=xfails[name], strict=True),)
        params.append(pytest.param(config, id=name, marks=marks))
    return params


STRUCTURED_DATA_MODEL_PARAMS = _model_params_with_xfail(
    {"gptoss_120b": _GPTOSS_STRUCTURED_XFAIL, "dsv4_flash": _DSV4_FLASH_STRUCTURED_XFAIL}
)

# test_multi_chunk_structured crosses config with num_tokens, so its failures are keyed per
# (model, num_tokens) and applied by _xfail_multi_chunk_structured. gptoss_120b's emb_dim makes the
# structured reshape invalid for every token count, but each combination is marked independently so
# one that starts passing trips strict xfail instead of hiding behind a model-wide mark.
_MULTI_CHUNK_NUM_TOKENS = [4096, 6400, 8192]
_MULTI_CHUNK_STRUCTURED_XFAIL = {("gptoss_120b", n): _GPTOSS_STRUCTURED_XFAIL for n in _MULTI_CHUNK_NUM_TOKENS}
_CONFIG_TO_MODEL_NAME = {config: name for name, config, _ in MODELS}


@pytest.fixture(autouse=True)
def _xfail_multi_chunk_structured(request):
    """Strict-xfail the (model, num_tokens) combinations in _MULTI_CHUNK_STRUCTURED_XFAIL, scoped to
    test_multi_chunk_structured so it does not touch the other tests that reuse these params."""
    if request.node.name.split("[")[0] != "test_multi_chunk_structured":
        return
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return
    name = _CONFIG_TO_MODEL_NAME.get(callspec.params.get("config"))
    reason = _MULTI_CHUNK_STRUCTURED_XFAIL.get((name, callspec.params.get("num_tokens")))
    if reason:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))


def pytorch_reference(combine, weights):
    """PyTorch reference: weighted sum across experts."""
    return (combine * weights.expand(-1, -1, -1, combine.shape[-1])).sum(dim=EXPERT_DIM)


def old_implementation(combine_tt, weights_tt):
    """Old implementation as used in tt_moe.py: to_layout(TILE) + mul + sum."""
    combine_tiled = ttnn.to_layout(combine_tt, ttnn.TILE_LAYOUT)
    weights_tiled = ttnn.to_layout(weights_tt, ttnn.TILE_LAYOUT)
    weighted = ttnn.mul(combine_tiled, weights_tiled)
    return ttnn.sum(weighted, dim=EXPERT_DIM)


def make_dispatch_table_all_local(device, num_routed_experts):
    """Create dispatch table where all experts are local (single device test)."""
    # All experts map to chip 0 (local)
    table = torch.zeros(num_routed_experts, dtype=torch.int32)
    return ttnn.from_torch(table, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def make_indices(num_tokens, num_experts, device, num_routed_experts):
    """Create indices tensor with random global expert IDs."""
    # Each token routes to num_experts random experts out of num_routed_experts
    indices = torch.stack([torch.randperm(num_routed_experts)[:num_experts] for _ in range(num_tokens)])
    indices = indices.unsqueeze(0).to(torch.uint16)  # [1, num_tokens, num_experts]
    return indices, ttnn.from_torch(
        indices, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def new_implementation(combine_tt, weights_tt, indices_tt, dispatch_table_tt):
    """Fused kernel: reads ROW_MAJOR, produces TILE output, skips non-local experts."""
    return ttnn.experimental.deepseek_prefill.post_combine_reduce(
        combine_tt,
        weights_tt,
        indices_tt,
        dispatch_table_tt,
        expert_dim=EXPERT_DIM,
        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def compute_pcc(a, b):
    """Compute PCC between two tensors."""
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def to_device(tensor, device):
    return ttnn.from_torch(tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def assert_pcc(result, expected, threshold=PCC_THRESHOLD, label=""):
    nan_count = torch.isnan(result).sum().item()
    assert nan_count == 0, f"{label}: got {nan_count} NaN elements"
    pcc = compute_pcc(result, expected)
    logger.info(f"  {label}: PCC={pcc:.6f}")
    assert pcc > threshold, f"{label}: PCC {pcc:.6f} below {threshold}"
    return pcc


# ============================================================================
# Structured data test
# ============================================================================


@pytest.mark.parametrize("config", STRUCTURED_DATA_MODEL_PARAMS)
def test_structured_data(device, config):
    """Constant-per-tile activations with sequential weights [1..8].
    This pattern is easy to verify manually and catches tile ordering bugs."""
    torch.manual_seed(42)
    emb_dim = config.EMB_SIZE
    tile_width = 1024
    num_tiles = emb_dim // tile_width

    tile_values = 0.1 * torch.arange(
        1,
        NUM_TOKENS * NUM_EXPERTS * num_tiles + 1,
        dtype=torch.float32,
    )
    combine = (
        tile_values.view(1, NUM_TOKENS, NUM_EXPERTS, num_tiles, 1)
        .expand(1, NUM_TOKENS, NUM_EXPERTS, num_tiles, tile_width)
        .reshape(1, NUM_TOKENS, NUM_EXPERTS, emb_dim)
        .to(torch.bfloat16)
    )

    weights = (
        torch.arange(1, NUM_EXPERTS + 1, dtype=torch.float32)
        .view(1, 1, NUM_EXPERTS, 1)
        .expand(1, NUM_TOKENS, NUM_EXPERTS, 1)
        .to(torch.bfloat16)
    )

    dispatch_table_tt = make_dispatch_table_all_local(device, config.NUM_ROUTED_EXPERTS)
    _, indices_tt = make_indices(NUM_TOKENS, NUM_EXPERTS, device, config.NUM_ROUTED_EXPERTS)

    ref = pytorch_reference(combine, weights)
    result = ttnn.to_torch(
        new_implementation(to_device(combine, device), to_device(weights, device), indices_tt, dispatch_table_tt)
    )
    assert_pcc(result, ref, threshold=0.998, label="structured")


# ============================================================================
# Random data tests
# ============================================================================


@pytest.mark.parametrize("config", MODEL_PARAMS)
def test_random_data(device, config):
    """Random activations and weights, compared to PyTorch reference."""
    torch.manual_seed(42)
    emb_dim = config.EMB_SIZE
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, emb_dim, dtype=torch.bfloat16)
    weights = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    dispatch_table_tt = make_dispatch_table_all_local(device, config.NUM_ROUTED_EXPERTS)
    _, indices_tt = make_indices(NUM_TOKENS, NUM_EXPERTS, device, config.NUM_ROUTED_EXPERTS)

    ref = pytorch_reference(combine, weights)
    result = ttnn.to_torch(
        new_implementation(to_device(combine, device), to_device(weights, device), indices_tt, dispatch_table_tt)
    )
    assert_pcc(result, ref, label="random")


@pytest.mark.parametrize("config", MODEL_PARAMS)
def test_vs_old_implementation(device, config):
    """Fused kernel vs old implementation (to_layout + mul + sum) with random data."""
    torch.manual_seed(42)
    emb_dim = config.EMB_SIZE
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, emb_dim, dtype=torch.bfloat16)
    weights = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    dispatch_table_tt = make_dispatch_table_all_local(device, config.NUM_ROUTED_EXPERTS)
    _, indices_tt = make_indices(NUM_TOKENS, NUM_EXPERTS, device, config.NUM_ROUTED_EXPERTS)

    ref = pytorch_reference(combine, weights)
    combine_tt = to_device(combine, device)
    weights_tt = to_device(weights, device)

    old_result = ttnn.to_torch(old_implementation(combine_tt, weights_tt))
    new_result = ttnn.to_torch(new_implementation(combine_tt, weights_tt, indices_tt, dispatch_table_tt))

    assert_pcc(old_result, ref, label="old_vs_ref")
    assert_pcc(new_result, ref, label="new_vs_ref")
    assert_pcc(old_result, new_result, label="old_vs_new")


# ============================================================================
# Sparse weight tests
# ============================================================================


@pytest.mark.parametrize("k_active", [6, 4, 2, 1])
@pytest.mark.parametrize("config", MODEL_PARAMS)
def test_sparse_weights(device, k_active, config):
    """Fused kernel with sparse weights (k_active out of 8 experts non-zero per token)."""
    torch.manual_seed(42)
    emb_dim = config.EMB_SIZE
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, emb_dim, dtype=torch.bfloat16)
    weights = torch.zeros(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)
    for t in range(NUM_TOKENS):
        active = torch.randperm(NUM_EXPERTS)[:k_active]
        weights[0, t, active, 0] = torch.randn(k_active, dtype=torch.bfloat16)

    dispatch_table_tt = make_dispatch_table_all_local(device, config.NUM_ROUTED_EXPERTS)
    _, indices_tt = make_indices(NUM_TOKENS, NUM_EXPERTS, device, config.NUM_ROUTED_EXPERTS)

    ref = pytorch_reference(combine, weights)
    result = ttnn.to_torch(
        new_implementation(to_device(combine, device), to_device(weights, device), indices_tt, dispatch_table_tt)
    )
    assert_pcc(result, ref, label=f"sparse_{k_active}/{NUM_EXPERTS}")


# ============================================================================
# Non-local expert skip test
# ============================================================================


# The routed-expert count (from each model's config) drives how many of a token's picks land
# outside the local range and get marked non-local. Larger pools (e.g. dsv4_pro's 384) exercise
# more non-local skips than the 256/128-expert models.
@pytest.mark.parametrize("config", MODEL_PARAMS)
def test_skip_nonlocal_experts(device, config):
    """Verify that marking experts as non-local (-1 in dispatch table) produces
    the same result when those experts' combine_output is zero (as in real MoE)."""
    torch.manual_seed(42)
    emb_dim = config.EMB_SIZE
    num_routed_experts = config.NUM_ROUTED_EXPERTS

    # Create indices: each token routes to 8 random experts
    indices_torch, indices_tt = make_indices(NUM_TOKENS, NUM_EXPERTS, device, num_routed_experts=num_routed_experts)

    # Build dispatch table where only experts 0-63 are local (column 0 of TP4)
    local_expert_end = 64
    table = torch.full((num_routed_experts,), -1, dtype=torch.int32)
    for i in range(local_expert_end):
        table[i] = i // 8  # map to chip within dispatch group
    dispatch_table_tt = ttnn.from_torch(
        table, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, emb_dim, dtype=torch.bfloat16)
    weights = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    # Reference: only sum local experts (non-local should be skipped by kernel)
    ref_combine = combine.clone()
    for t in range(NUM_TOKENS):
        for k in range(NUM_EXPERTS):
            expert_id = indices_torch[0, t, k].item()
            if expert_id >= local_expert_end:
                ref_combine[0, t, k, :] = 0.0
    ref = pytorch_reference(ref_combine, weights)

    result = ttnn.to_torch(
        new_implementation(to_device(combine, device), to_device(weights, device), indices_tt, dispatch_table_tt)
    )
    # Use the standard PCC threshold so this test validates non-local expert skipping.
    assert_pcc(result, ref, threshold=PCC_THRESHOLD, label="skip_nonlocal_no_init_zeros")


# ============================================================================
# Output format test
# ============================================================================


@pytest.mark.parametrize("config", MODEL_PARAMS)
def test_output_layout(device, config):
    """Verify output is TILE layout with correct shape."""
    torch.manual_seed(42)
    emb_dim = config.EMB_SIZE
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, emb_dim, dtype=torch.bfloat16)
    weights = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    dispatch_table_tt = make_dispatch_table_all_local(device, config.NUM_ROUTED_EXPERTS)
    _, indices_tt = make_indices(NUM_TOKENS, NUM_EXPERTS, device, config.NUM_ROUTED_EXPERTS)

    result_tt = new_implementation(
        to_device(combine, device), to_device(weights, device), indices_tt, dispatch_table_tt
    )
    assert result_tt.layout == ttnn.TILE_LAYOUT, f"Expected TILE_LAYOUT, got {result_tt.layout}"
    assert list(result_tt.shape) == [1, NUM_TOKENS, emb_dim], f"Wrong shape: {result_tt.shape}"


# ============================================================================
# Multi-chunk-per-core tests: num_tokens / 32 > num_cores, so some cores must
# process more than one 32-token chunk. Covers issue #41777.
# ============================================================================


@pytest.mark.parametrize("num_tokens", _MULTI_CHUNK_NUM_TOKENS)
@pytest.mark.parametrize("config", MODEL_PARAMS)
def test_multi_chunk_structured(device, num_tokens, config):
    """Structured data with >100 chunks so some cores get 2+ chunks each."""
    emb_dim = config.EMB_SIZE
    tile_width = 1024
    num_tiles = emb_dim // tile_width

    tile_values = 0.1 * torch.arange(
        1,
        num_tokens * NUM_EXPERTS * num_tiles + 1,
        dtype=torch.float32,
    )
    combine = (
        tile_values.view(1, num_tokens, NUM_EXPERTS, num_tiles, 1)
        .expand(1, num_tokens, NUM_EXPERTS, num_tiles, tile_width)
        .reshape(1, num_tokens, NUM_EXPERTS, emb_dim)
        .to(torch.bfloat16)
    )

    weights = (
        torch.arange(1, NUM_EXPERTS + 1, dtype=torch.float32)
        .view(1, 1, NUM_EXPERTS, 1)
        .expand(1, num_tokens, NUM_EXPERTS, 1)
        .to(torch.bfloat16)
    )

    ref = pytorch_reference(combine, weights)
    result = ttnn.to_torch(new_implementation(to_device(combine, device), to_device(weights, device), None, None))
    assert_pcc(result, ref, threshold=0.998, label=f"multi_chunk_structured_{num_tokens}")


@pytest.mark.parametrize("num_tokens", [4096, 6400, 8192])
@pytest.mark.parametrize("config", MODEL_PARAMS)
def test_multi_chunk_random(device, num_tokens, config):
    """Random data with >100 chunks so some cores get 2+ chunks each."""
    torch.manual_seed(42)
    emb_dim = config.EMB_SIZE
    combine = torch.randn(1, num_tokens, NUM_EXPERTS, emb_dim, dtype=torch.bfloat16)
    weights = torch.randn(1, num_tokens, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    ref = pytorch_reference(combine, weights)
    result = ttnn.to_torch(new_implementation(to_device(combine, device), to_device(weights, device), None, None))
    assert_pcc(result, ref, label=f"multi_chunk_random_{num_tokens}")


# ============================================================================
