# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Minimal single-device, single-expert test for TtRoutedExpert profiling.

The simplest scenario: 1 chip, 1 expert, minimal dimensions.
"""

import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.gpt_oss_120b_config import GptOss120BConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.minimax_m2_7_config import MiniMaxM27Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc


SINGLE_CHIP_MESH_PARAMS = [
    pytest.param(
        1,
        {"fabric_config": ttnn.FabricConfig.DISABLED},
        id="single-chip",
    ),
]

# Token-count sweep for the single-expert profiling test, applied per model with that model's
# (emb_dim, hidden_dim). The (num_tokens, id) pairs are model-independent.
#
# NOTE on sub-tile counts: TILE_LAYOUT pads the input to a 32-row tile, so any
# num_tokens <= 32 maps to a single M tile-row and produces identical device work
# (the runtime count only bounds the chunk loop, min 1 chunk). The perf-meaningful
# granularity is therefore multiples of 32; the sub-32 entries are kept so the
# collapse is visible in a sweep.
_TOKEN_SWEEP = [
    (2, "2"),
    (4, "4"),
    (8, "8"),
    (16, "16"),
    (32, "32"),
    (64, "64"),
    (128, "128"),
    (256, "256"),
    (512, "512"),
    (1024, "1k"),
    (2048, "2k"),
    (4096, "4k"),
    (8192, "8k"),
    (16384, "16k"),
    (25600, "25k"),
]


def run_single_routed_expert(
    mesh_device,
    device_params,
    num_tokens: int,
    emb_dim: int,
    hidden_dim: int,
):
    """
    Simplest scenario: 1 chip, 1 expert. Shared body for the per-model entrypoints below — they
    differ only on the (emb_dim, hidden_dim) shape axis.

    Perfect for profiling the core FFN computation without any mesh complexity.
    """
    experts_per_chip = 1

    signpost(f"SingleRoutedExpert {num_tokens=} {emb_dim=} {hidden_dim=}")

    logger.debug(f"Testing single routed expert: {num_tokens=}, {emb_dim=}, {hidden_dim=}")
    logger.debug(f"Mesh: {mesh_device.shape}, num_devices={mesh_device.get_num_devices()}")

    # Create random weights
    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }

    # Create torch reference
    torch_expert = TorchExpert(emb_dim, hidden_dim, weights)

    # 2D input (num_tokens, emb_dim) — the single expert's dispatch buffer.
    torch_input = torch.randn(num_tokens, emb_dim, dtype=torch.float32)
    logger.debug(f"Input shape: {torch_input.shape}")

    # Run torch reference
    logger.debug("Running torch reference...")
    with torch.no_grad():
        torch_output = torch_expert(torch_input)
    logger.debug(f"Torch output shape: {torch_output.shape}")

    # Create TTNN input: 2D (num_tokens, emb_dim), replicated across the 1-device mesh.
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )
    logger.debug(f"TTNN input shape: {tt_input.shape}")

    # Single-expert auxiliaries (1D, length 1, UINT32 ROW_MAJOR DRAM):
    #   - global_expert_idx_table[0] = 0   (local 0 -> global 0)
    #   - expert_token_counts[0]     = num_tokens
    #   - expert_region_offsets[0]   = 0   (expert's slice starts at row 0)
    def _make_idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )

    global_expert_idx_tt = _make_idx_tensor([0])
    expert_token_counts_tt = _make_idx_tensor([num_tokens])
    expert_region_offsets_tt = _make_idx_tensor([0])

    # Create TtRoutedExpert
    logger.debug("Creating TtRoutedExpert...")
    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=num_tokens,
        torch_weights=[weights],  # List with single expert weights
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=ttnn.RoutedExpertActivation.Silu,
    )

    # Run TTNN forward. ROUTED_EXPERT_PERF_ITERS > 1 repeats the op back-to-back
    # (warm, cached program) so a tracy profiling run captures multiple device
    # invocations to median over; the first (cold) iteration is dropped in
    # analysis. The signpost above brackets the whole loop.
    perf_iters = max(1, int(os.environ.get("ROUTED_EXPERT_PERF_ITERS", "1")))
    logger.debug(f"Running TTNN forward ({perf_iters} iters)...")
    for _ in range(perf_iters):
        tt_output = tt_expert(tt_input, expert_token_counts_tt, expert_region_offsets_tt)
    ttnn.synchronize_device(mesh_device)
    logger.debug(f"TTNN output shape: {tt_output.shape}")

    # RE_SKIP_MATMUL / RE_SKIP_OUTPUT_WRITE strip compute/output work to isolate
    # DRAM I/O for perf investigation; the numerical result is intentionally
    # garbage, so skip the correctness checks below.
    skip_correctness = (
        os.environ.get("RE_SKIP_MATMUL") not in (None, "", "0")
        or os.environ.get("RE_SKIP_OUTPUT_WRITE") not in (None, "", "0")
        or os.environ.get("RE_SKIP_WEIGHT_READ") not in (None, "", "0")
    )
    if skip_correctness:
        logger.warning("RE_SKIP_* set: skipping PCC/NaN checks (perf-only run)")
        return

    # Convert back to torch for comparison. For a 1-device replicated tensor,
    # ConcatMeshToTensor(dim=0) with 1 slice is a no-op that returns the tensor.
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    logger.debug(f"TTNN output (torch) shape: {tt_output_torch.shape}")

    # Compare PCC
    _, pcc = comp_pcc(torch_output, tt_output_torch)
    logger.debug(f"PCC: {pcc:.6f}")

    # Validate
    pcc_threshold = 0.97
    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold}"
    assert not torch.isnan(tt_output_torch).any(), "Output contains NaN"
    assert not torch.isinf(tt_output_torch).any(), "Output contains Inf"

    logger.debug("Test PASSED!")


# Per-model dims as (id_prefix, config, extended_model), each run at its own (emb_dim,
# MOE_INTERMEDIATE_SIZE). DeepSeek V3 is the baseline and runs by default; every other model is
# gated behind @pytest.mark.extended_model.
SINGLE_EXPERT_MODELS = [
    ("dsv3", DeepSeekV3Config, False),
    ("minimax_m27", MiniMaxM27Config, True),
    ("glm_51", GLM51Config, True),
    ("dsv4_pro", DeepSeekV4ProConfig, True),
    ("dsv4_flash", DeepSeekV4FlashConfig, True),
    ("gptoss_120b", GptOss120BConfig, True),
    ("kimi_k26", KimiK26Config, True),
]


# Registry of currently-failing single-routed-expert cases, keyed by the exact "{model}-{tag}"
# param id -> xfail reason (with tracking issue), so CI stays green while linked issues are worked
# on. _TOKEN_SWEEP_XFAIL is applied (strict) only on blackhole by _xfail_blackhole_token_sweep;
# _FAKED_XFAIL by single_routed_expert_faked_params. Both are empty: the factory's adaptive L1 guard
# (shrinks per-core CBs to fit L1 and picks an in0_block_w_gu that divides K_gate_tiles) resolved the
# prior dsv4_pro L1 and gptoss_120b K_gate failures. Add an entry when a new blackhole-specific
# failure appears (these pass on other arches); delete it once resolved.
_TOKEN_SWEEP_XFAIL = {}
_FAKED_XFAIL = {}


def single_routed_expert_token_sweep_params():
    """Build the per-model (num_tokens, emb_dim, hidden_dim) parametrization over _TOKEN_SWEEP.
    Non-baseline models carry the extended_model marker so they stay gated as before; the
    _TOKEN_SWEEP_XFAIL cases are xfail'd per-arch by _xfail_blackhole_token_sweep, not here."""
    params = []
    for name, config, extended in SINGLE_EXPERT_MODELS:
        for num_tokens, tag in _TOKEN_SWEEP:
            test_id = f"{name}-{tag}"
            marks = (pytest.mark.extended_model,) if extended else ()
            params.append(
                pytest.param(num_tokens, config.EMB_SIZE, config.MOE_INTERMEDIATE_SIZE, marks=marks, id=test_id)
            )
    return params


@pytest.fixture(autouse=True)
def _xfail_blackhole_token_sweep(request, silicon_arch_name):
    """Strict-xfail the _TOKEN_SWEEP_XFAIL cases only on blackhole: the K_gate / L1 issues are
    blackhole-specific and these cases pass on other arches, where an unconditional strict xfail
    would turn CI red on XPASS. Keyed by the full param id so each (model, token-count) is marked
    independently."""
    if silicon_arch_name != "blackhole" or request.node.name.split("[")[0] != "test_single_routed_expert_models":
        return
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return
    for param_id, reason in _TOKEN_SWEEP_XFAIL.items():
        if callspec.id.endswith(param_id):
            request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
            break


@pytest.mark.parametrize("num_tokens, emb_dim, hidden_dim", single_routed_expert_token_sweep_params())
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_single_routed_expert_models(mesh_device, device_params, num_tokens: int, emb_dim: int, hidden_dim: int):
    run_single_routed_expert(mesh_device, device_params, num_tokens, emb_dim, hidden_dim)


# (allocated_tokens, active_tokens, id) sweep for the count-aware sparsity test, applied per
# model with that model's (emb_dim, hidden_dim). The alloc/active pairs are model-independent.
# Count-sparsity sweep: a FIXED 5120-token dispatch buffer with a varying number of
# active (real) tokens; the rest is zero padding the kernel must skip via the
# device-side count. This mirrors the real MoE dispatch layout (buffer sized for the
# max, sparsely filled). Tags are fixed-width (aNNNNN) so pytest -k selects each
# uniquely (no substring collisions). 25k is intentionally omitted.
_FAKED_SWEEP = [
    (5120, 1, "a00001"),
    (5120, 2, "a00002"),
    (5120, 4, "a00004"),
    (5120, 8, "a00008"),
    (5120, 16, "a00016"),
    (5120, 32, "a00032"),
    (5120, 64, "a00064"),
    (5120, 128, "a00128"),
    (5120, 256, "a00256"),
    (5120, 512, "a00512"),
    (5120, 1024, "a01024"),
    (5120, 2048, "a02048"),
    (5120, 4096, "a04096"),
    (5120, 5120, "a05120"),
]


def run_single_routed_expert_faked_token_count(
    mesh_device,
    device_params,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
):
    """
    Verifies the unified kernel honors expert_token_counts and skips work on
    inactive padding rows. Shared body for the per-model entrypoints below — they differ only on
    the (emb_dim, hidden_dim) shape axis.

    Dispatch buffer sized for ``allocated_tokens`` but only the first
    ``active_tokens`` rows hold real data; the rest is zero padding. The
    kernel must (a) produce correct output on the active slice and (b) not
    do matmuls on the inactive padding rows (device-side count sparsity).
    """
    experts_per_chip = 1

    signpost(f"SingleRoutedExpertFaked {allocated_tokens=} {active_tokens=} {emb_dim=} {hidden_dim=}")

    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }
    torch_expert = TorchExpert(emb_dim, hidden_dim, weights)

    torch_active = torch.randn(active_tokens, emb_dim, dtype=torch.float32)
    torch_input = torch.zeros(allocated_tokens, emb_dim, dtype=torch.float32)
    torch_input[:active_tokens] = torch_active

    with torch.no_grad():
        torch_output_active = torch_expert(torch_active)

    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )

    def _make_idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )

    global_expert_idx_tt = _make_idx_tensor([0])
    expert_token_counts_tt = _make_idx_tensor([active_tokens])
    expert_region_offsets_tt = _make_idx_tensor([0])

    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=allocated_tokens,
        torch_weights=[weights],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=ttnn.RoutedExpertActivation.Silu,
    )

    perf_iters = max(1, int(os.environ.get("ROUTED_EXPERT_PERF_ITERS", "1")))
    for _ in range(perf_iters):
        tt_output = tt_expert(tt_input, expert_token_counts_tt, expert_region_offsets_tt)
    ttnn.synchronize_device(mesh_device)

    # RE_SKIP_* strip compute/output/read work for perf investigation; the result is
    # intentionally garbage, so skip correctness checks on those runs.
    skip_correctness = (
        os.environ.get("RE_SKIP_MATMUL") not in (None, "", "0")
        or os.environ.get("RE_SKIP_OUTPUT_WRITE") not in (None, "", "0")
        or os.environ.get("RE_SKIP_WEIGHT_READ") not in (None, "", "0")
    )
    if skip_correctness:
        logger.warning("RE_SKIP_* set: skipping PCC/NaN checks (perf-only run)")
        return

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    tt_output_active = tt_output_torch[:active_tokens]

    _, pcc = comp_pcc(torch_output_active, tt_output_active)
    logger.debug(f"PCC over active slice ({active_tokens} rows): {pcc:.6f}")

    assert pcc >= 0.97, f"PCC {pcc:.6f} below threshold 0.97"
    assert not torch.isnan(tt_output_active).any(), "Active output contains NaN"
    assert not torch.isinf(tt_output_active).any(), "Active output contains Inf"


def single_routed_expert_faked_params():
    """Build the per-model (allocated_tokens, active_tokens, emb_dim, hidden_dim) parametrization
    over _FAKED_SWEEP. Reuses SINGLE_EXPERT_MODELS, so non-baseline models stay gated behind the
    extended_model marker exactly as the separate tests were."""
    params = []
    for name, config, extended in SINGLE_EXPERT_MODELS:
        for alloc, active, tag in _FAKED_SWEEP:
            test_id = f"{name}-{tag}"
            marks = (pytest.mark.extended_model,) if extended else ()
            if test_id in _FAKED_XFAIL:
                marks += (pytest.mark.xfail(reason=_FAKED_XFAIL[test_id], strict=True),)
            params.append(
                pytest.param(
                    alloc,
                    active,
                    config.EMB_SIZE,
                    config.MOE_INTERMEDIATE_SIZE,
                    marks=marks,
                    id=test_id,
                )
            )
    return params


@pytest.mark.parametrize("allocated_tokens, active_tokens, emb_dim, hidden_dim", single_routed_expert_faked_params())
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
def test_single_routed_expert_faked_token_count_models(
    mesh_device, device_params, allocated_tokens: int, active_tokens: int, emb_dim: int, hidden_dim: int
):
    run_single_routed_expert_faked_token_count(
        mesh_device, device_params, allocated_tokens, active_tokens, emb_dim, hidden_dim
    )
