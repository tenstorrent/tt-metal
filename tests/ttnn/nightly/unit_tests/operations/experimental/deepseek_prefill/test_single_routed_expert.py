# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Minimal single-device, single-expert test for TtRoutedExpert profiling.

The simplest scenario: 1 chip, 1 expert, minimal dimensions.
"""

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
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs

SINGLE_CHIP_MESH_PARAMS = [
    pytest.param(
        1,
        {"fabric_config": ttnn.FabricConfig.DISABLED},
        id="single-chip",
    ),
]


def run_single_routed_expert(
    device,
    allocated_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    active_tokens: int = None,
    x_row_major: bool = False,
    weights_dram_sharded: bool = False,
    implementation: str = "unified",
):
    """
    Simplest scenario: 1 chip, 1 expert. Shared body for the per-model entrypoints below — they
    differ only on the (emb_dim, hidden_dim) shape axis and the x input layout.

    The single expert's dispatch buffer is sized for ``allocated_tokens`` but only the first
    ``active_tokens`` rows hold real data; the rest is zero padding. When ``active_tokens`` is
    None it defaults to ``allocated_tokens`` (a fully-active buffer), which is the plain
    profiling case. With ``active_tokens < allocated_tokens`` this exercises device-side
    count-aware sparsity: the kernel must (a) produce correct output on the active slice and
    (b) not do matmuls on the inactive padding rows. The ROW_MAJOR variant additionally drives
    the reader's clamp-read of the inactive rows past the runtime count.
    """
    if active_tokens is None:
        active_tokens = allocated_tokens
    if implementation not in ("unified", "moe_fused"):
        raise ValueError(f"Unknown routed-expert implementation: {implementation}")
    experts_per_chip = 1

    signpost(f"SingleRoutedExpert {implementation=} {allocated_tokens=} " f"{active_tokens=} {emb_dim=} {hidden_dim=}")

    logger.debug(f"Testing single routed expert: {allocated_tokens=}, {active_tokens=}, {emb_dim=}, {hidden_dim=}")
    logger.debug(f"Mesh: {device.shape}, num_devices={device.get_num_devices()}")

    # Create random weights
    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }

    # Create torch reference
    torch_expert = TorchExpert(emb_dim, hidden_dim, weights)

    # 2D input (allocated_tokens, emb_dim) — the single expert's dispatch buffer. The first
    # active_tokens rows hold real data; the rest is zero padding (a no-op when active==allocated).
    torch_active = torch.randn(active_tokens, emb_dim, dtype=torch.float32)
    torch_input = torch.zeros(allocated_tokens, emb_dim, dtype=torch.float32)
    torch_input[:active_tokens] = torch_active
    logger.debug(f"Input shape: {torch_input.shape}")

    # Run torch reference over the active slice only.
    logger.debug("Running torch reference...")
    with torch.no_grad():
        torch_output_active = torch_expert(torch_active)
    logger.debug(f"Torch output shape: {torch_output_active.shape}")

    # Create TTNN input: 2D (allocated_tokens, emb_dim), replicated across the 1-device mesh.
    # The composite op branches on x layout: ROW_MAJOR is bf16 (tilized and bf8-packed
    # inside the op), TILE is consumed directly as bf8. Pair the dtype with the layout so
    # each variation drives its real device path.
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        layout=ttnn.ROW_MAJOR_LAYOUT if x_row_major else ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16 if x_row_major else ttnn.bfloat8_b,
    )
    logger.debug(f"TTNN input shape: {tt_input.shape}")

    # Single-expert auxiliaries (1D, length 1, UINT32 ROW_MAJOR DRAM):
    #   - global_expert_idx_table[0] = 0             (local 0 -> global 0)
    #   - expert_token_counts[0]     = active_tokens (runtime count; drives count sparsity)
    #   - expert_region_offsets[0]   = 0             (expert's slice starts at row 0)
    def _make_idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            dtype=ttnn.uint32,
        )

    global_expert_idx_tt = _make_idx_tensor([0])
    expert_token_counts_tt = _make_idx_tensor([active_tokens])
    expert_region_offsets_tt = _make_idx_tensor([0])

    # Create TtRoutedExpert
    logger.debug("Creating TtRoutedExpert...")
    tt_expert = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=allocated_tokens,
        torch_weights=[weights],  # List with single expert weights
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        # Each implementation partitions W_down differently. Build the unified
        # placement here only for unified; the moe_fused placement is applied below.
        weights_dram_sharded=weights_dram_sharded and implementation == "unified",
        activation=ttnn.RoutedExpertActivation.Silu,
    )

    if weights_dram_sharded and implementation == "moe_fused":
        gate_up_memory_config, down_memory_config = weight_memory_configs(
            device,
            emb_dim,
            hidden_dim,
            core_grid=(11, 8),
        )
        tt_expert.gate_projs = [ttnn.to_memory_config(tt_expert.gate_projs[0], gate_up_memory_config)]
        tt_expert.up_projs = [ttnn.to_memory_config(tt_expert.up_projs[0], gate_up_memory_config)]
        tt_expert.down_projs = [ttnn.to_memory_config(tt_expert.down_projs[0], down_memory_config)]

    # Run TTNN forward
    logger.debug("Running TTNN forward...")
    if implementation == "unified":
        tt_output = tt_expert(tt_input, expert_token_counts_tt, expert_region_offsets_tt)
    else:
        signpost(header="MoeFusedSwiGlu")
        tt_output = ttnn.empty(
            tt_input.shape,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_output = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            tt_input,
            tt_expert.gate_projs[0],
            tt_expert.up_projs[0],
            tt_expert.down_projs[0],
            expert_token_counts_tt,
            global_expert_idx_tt,
            0,
            input_m_tiles=allocated_tokens // ttnn.TILE_SIZE,
            core_grid=ttnn.CoreCoord(11, 8),
            output=tt_output,
            expert_region_offsets=expert_region_offsets_tt,
            read_x_at_offset=True,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
                dst_full_sync_en=False,
            ),
        )
    logger.debug(f"TTNN output shape: {tt_output.shape}")

    # Convert back to torch for comparison. For a 1-device replicated tensor,
    # ConcatMeshToTensor(dim=0) with 1 slice is a no-op that returns the tensor.
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0),
    )
    logger.debug(f"TTNN output (torch) shape: {tt_output_torch.shape}")
    tt_output_active = tt_output_torch[:active_tokens]

    # Compare PCC over the active slice.
    _, pcc = comp_pcc(torch_output_active, tt_output_active)
    logger.debug(f"PCC over active slice ({active_tokens} rows): {pcc:.6f}")

    # Validate
    pcc_threshold = 0.97
    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold}"
    assert not torch.isnan(tt_output_active).any(), "Active output contains NaN"
    assert not torch.isinf(tt_output_active).any(), "Active output contains Inf"

    logger.debug("Test PASSED!")


MULTI_EXPERT_PARITY_SCENARIOS = [
    pytest.param(
        [0, 1, 2, 3],
        [128, 256, 64, 160],
        [0, 256, 512, 768],
        1024,
        id="packed-aligned",
    ),
    pytest.param(
        [2, 0, 3, 1],
        [32, 255, 0, 160],
        [64, 384, 704, 1024],
        1344,
        id="permuted-gapped-zero-tail",
    ),
    pytest.param(
        [7, 0, 11, 3, 9, 1, 5, 10, 2, 8, 4, 6],
        [1, 7, 17, 31, 33, 47, 63, 65, 95, 97, 127, 255],
        [expert * 256 for expert in range(12)],
        12 * 256,
        id="twelve-experts-all-ragged",
    ),
]


@pytest.mark.parametrize("input_format", ["bfp8_tile", "bf16_rm"])
@pytest.mark.parametrize(
    "global_ids, counts_by_local, offsets_by_local, total_rows",
    MULTI_EXPERT_PARITY_SCENARIOS,
)
@pytest.mark.skipif(not is_blackhole(), reason="shared-region fused routed expert is Blackhole-only")
def test_multi_routed_expert_silu_shared_regions(
    device,
    input_format,
    global_ids,
    counts_by_local,
    offsets_by_local,
    total_rows,
):
    """Compare unified and moe_fused over the same multi-expert dispatch buffer.

    This is the bias-free, standard-SiLU counterpart of
    ``test_gptoss_bias_multi_expert``. It exercises the fused extract/insert
    addressing itself: non-identity local-to-global mappings, nonzero and gapped
    offsets, a zero-token expert, and many non-tile-aligned counts. The 12-expert
    case spans 1..255 tokens and deliberately samples both sides of several tile
    boundaries. No standalone extract or insert op runs.

    Both implementations receive independently allocated copies of the same
    quantized input and share the same converted BFP4 weight tensors. Their
    active output regions must each match Torch and match one another.
    """
    torch.manual_seed(123)
    # DeepSeek-style target shape: K=7168 input features, N=2048 FFN features.
    emb_dim, hidden_dim = 7168, 2048
    max_tokens = 256
    num_experts = len(global_ids)
    assert len(counts_by_local) == len(offsets_by_local) == num_experts

    torch_weights = []
    torch_experts = []
    torch_inputs = []
    torch_buffer = torch.randn(total_rows, emb_dim, dtype=torch.float32)
    for local_expert, (count, offset) in enumerate(zip(counts_by_local, offsets_by_local)):
        weights = {
            "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
        }
        active = torch.randn(count, emb_dim, dtype=torch.float32)
        torch_buffer[offset : offset + count] = active
        # Hostile region padding catches a reader that ignores the device-side count.
        torch_buffer[offset + count : offset + max_tokens] = 100.0
        torch_weights.append(weights)
        torch_experts.append(TorchExpert(emb_dim, hidden_dim, weights))
        torch_inputs.append(active)

    with torch.no_grad():
        expected = [
            torch_experts[local_expert](torch_inputs[local_expert]) if count else None
            for local_expert, count in enumerate(counts_by_local)
        ]

    def u32(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            dtype=ttnn.uint32,
        )

    counts_by_global = [0] * num_experts
    offsets_by_global = [0] * num_experts
    for local_expert, global_expert in enumerate(global_ids):
        counts_by_global[global_expert] = counts_by_local[local_expert]
        offsets_by_global[global_expert] = offsets_by_local[local_expert]

    global_expert_idx = u32(global_ids)
    expert_counts = u32(counts_by_global)
    expert_offsets = u32(offsets_by_global)
    input_dtype, input_layout = {
        "bfp8_tile": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        "bf16_rm": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    }[input_format]

    def make_input():
        return ttnn.from_torch(
            torch_buffer,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            layout=input_layout,
            device=device,
            dtype=input_dtype,
        )

    compose = ttnn.ConcatMeshToTensor(device, dim=0)
    tt_unified_input = make_input()
    tt_fused_input = make_input()
    unified_input_before = ttnn.to_torch(tt_unified_input, mesh_composer=compose).clone()
    fused_input_before = ttnn.to_torch(tt_fused_input, mesh_composer=compose).clone()

    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )

    tt_expert = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=num_experts,
        global_expert_idx_table=global_expert_idx,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=max_tokens,
        torch_weights=torch_weights,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        compute_kernel_config=compute_config,
        activation=ttnn.RoutedExpertActivation.Silu,
    )
    assert tt_expert.gate_biases is None and tt_expert.up_biases is None and tt_expert.down_biases is None

    tt_unified_output = tt_expert(tt_unified_input, expert_counts, expert_offsets)
    tt_fused_output = ttnn.from_torch(
        torch.full_like(torch_buffer, -7.5),
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat8_b,
    )
    for local_expert in range(num_experts):
        tt_fused_output = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            tt_fused_input,
            tt_expert.gate_projs[local_expert],
            tt_expert.up_projs[local_expert],
            tt_expert.down_projs[local_expert],
            expert_counts,
            global_expert_idx,
            local_expert,
            input_m_tiles=max_tokens // ttnn.TILE_SIZE,
            core_grid=ttnn.CoreCoord(11, 8),
            output=tt_fused_output,
            expert_region_offsets=expert_offsets,
            read_x_at_offset=True,
            compute_kernel_config=compute_config,
        )

    unified_output = ttnn.to_torch(tt_unified_output, mesh_composer=compose)
    fused_output = ttnn.to_torch(tt_fused_output, mesh_composer=compose)
    written_tiles = torch.zeros(total_rows, dtype=torch.bool)
    for local_expert, (count, start) in enumerate(zip(counts_by_local, offsets_by_local)):
        if count == 0:
            continue
        unified_region = unified_output[start : start + count]
        fused_region = fused_output[start : start + count]
        unified_passing, unified_pcc = comp_pcc(expected[local_expert], unified_region, 0.97)
        fused_passing, fused_pcc = comp_pcc(expected[local_expert], fused_region, 0.97)
        parity_passing, parity_pcc = comp_pcc(unified_region, fused_region, 0.999)
        logger.info(
            f"{input_format}: local={local_expert} global={global_ids[local_expert]} "
            f"offset={start} count={count} unified_pcc={unified_pcc} "
            f"fused_pcc={fused_pcc} parity_pcc={parity_pcc}"
        )
        assert unified_passing, f"unified expert {local_expert} PCC below threshold: {unified_pcc}"
        assert fused_passing, f"moe_fused expert {local_expert} PCC below threshold: {fused_pcc}"
        assert parity_passing, f"expert {local_expert} unified/moe_fused parity PCC too low: {parity_pcc}"
        assert torch.isfinite(unified_region).all(), f"unified expert {local_expert} output is not finite"
        assert torch.isfinite(fused_region).all(), f"moe_fused expert {local_expert} output is not finite"
        written_rows = ((count + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        written_tiles[start : start + written_rows] = True

    # moe_fused uses a separate sentinel-filled destination and must never mutate x.
    fused_input_after = ttnn.to_torch(tt_fused_input, mesh_composer=compose)
    assert torch.equal(fused_input_after, fused_input_before), "moe_fused mutated its shared input buffer"
    assert torch.equal(
        fused_output[~written_tiles], torch.full_like(fused_output[~written_tiles], -7.5)
    ), "moe_fused wrote outside the active experts' tile prefixes"

    # unified intentionally aliases TILE input, while its BF16 ROW_MAJOR path allocates
    # a fresh BFP8 TILE output. Check the observable untouched-buffer contract of each.
    if input_format == "bfp8_tile":
        assert torch.equal(
            unified_output[~written_tiles], unified_input_before[~written_tiles]
        ), "unified wrote outside the active experts' tile prefixes"
    else:
        unified_input_after = ttnn.to_torch(tt_unified_input, mesh_composer=compose)
        assert torch.equal(unified_input_after, unified_input_before), "unified mutated its BF16 ROW_MAJOR input"


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


# Registry of currently-failing single-routed-expert cases -> xfail reason (with tracking issue), so
# CI stays green while linked issues are worked on. Applied strict, only on blackhole, by
# _xfail_blackhole (these cases pass on other arches, where an unconditional strict xfail would turn
# CI red on XPASS). Each key is a space-separated set of id tokens that must ALL appear in the param
# id, so a case can be scoped by any combination of layout ("x_tile"/"x_rm") and model/isl id.
# Empty: the program factory now snaps in0_block_w_gu to a divisor of K_gate_tiles on every path
# (not just when the L1 guard fires), so the prior gptoss_120b TILE-layout K_gate failure is fixed.
_XFAIL = {}


@pytest.fixture(autouse=True)
def _xfail_blackhole(request, silicon_arch_name):
    """Strict-xfail the _XFAIL cases only on blackhole: the K_gate / L1 issues are blackhole-specific
    and these cases pass on other arches, where an unconditional strict xfail would turn CI red on
    XPASS. A case matches when every whitespace-separated token of the key appears in the param id."""
    if silicon_arch_name != "blackhole":
        return
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        return
    for key, reason in _XFAIL.items():
        if all(token in callspec.id for token in key.split()):
            request.applymarker(pytest.mark.xfail(reason=reason, strict=True))
            break


# All active-token sweeps run against a fixed 5K allocated buffer: 5120 dispatch rows with only the
# first `active_tokens` holding real data; the rest is zero padding.
_ISL_ALLOCATED_TOKENS = 5120

# Functional sweep: a few active-token counts across every model
_ISL_FUNCTIONAL_SWEEP = [251, 768, 3001]

# Exhaustive sweep: the full range from empty to fully-packed
# 64 is the smallest non-empty point: 2 tile-rows, so only 2 of the 8 M-rows carry real
# tokens and the op is almost entirely the fixed weight read — the regime where the DRAM
# work dominates and layout changes show up most clearly.
_ISL_EXHAUSTIVE_SWEEP = [0, 64, 128, 256, 512, 1024, 2048, 4096, 5120]
_ISL_EXHAUSTIVE_MODELS = ("kimi_k26", "glm_51", "k7168_n3072")

# Additional benchmark shapes that do not have a model configuration class.
_ISL_EXHAUSTIVE_EXTRA_SHAPES = {
    "k7168_n3072": (7168, 3072),
}


def _isl_params(active_sweep, only_models=None):
    """Build the per-model (allocated_tokens, active_tokens, emb_dim, hidden_dim) parametrization over
    `active_sweep`, all against the fixed _ISL_ALLOCATED_TOKENS buffer. Reuses SINGLE_EXPERT_MODELS so
    non-baseline models stay gated behind the extended_model marker; `only_models` restricts to a
    subset of model names."""
    params = []
    for name, config, extended in SINGLE_EXPERT_MODELS:
        if only_models is not None and name not in only_models:
            continue
        for active in active_sweep:
            marks = (pytest.mark.extended_model,) if extended else ()
            params.append(
                pytest.param(
                    _ISL_ALLOCATED_TOKENS,
                    active,
                    config.EMB_SIZE,
                    config.MOE_INTERMEDIATE_SIZE,
                    marks=marks,
                    id=f"{name}-isl-{active}",
                )
            )
    return params


def _isl_exhaustive_params():
    """Build the exhaustive shape/layout axis.

    Model-config shapes exercise both activation layouts. Extra benchmark shapes
    can be limited to the production ROW_MAJOR BF16 input path when their tiled
    activation variant does not fit in L1.
    """
    params = []
    standard_models = {name: (config, extended) for name, config, extended in SINGLE_EXPERT_MODELS}
    for name in _ISL_EXHAUSTIVE_MODELS:
        if name in standard_models:
            config, extended = standard_models[name]
            emb_dim, hidden_dim = config.EMB_SIZE, config.MOE_INTERMEDIATE_SIZE
            x_layouts = ((True, "x_rm"), (False, "x_tile"))
        else:
            emb_dim, hidden_dim = _ISL_EXHAUSTIVE_EXTRA_SHAPES[name]
            extended = True
            x_layouts = ((True, "x_rm"),)

        for x_row_major, x_layout_id in x_layouts:
            for active in _ISL_EXHAUSTIVE_SWEEP:
                marks = (pytest.mark.extended_model,) if extended else ()
                params.append(
                    pytest.param(
                        _ISL_ALLOCATED_TOKENS,
                        active,
                        emb_dim,
                        hidden_dim,
                        x_row_major,
                        marks=marks,
                        id=f"{x_layout_id}-{name}-isl-{active}",
                    )
                )
    return params


@pytest.mark.parametrize("allocated_tokens, active_tokens, emb_dim, hidden_dim", _isl_params(_ISL_FUNCTIONAL_SWEEP))
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
def test_single_routed_expert_functional(
    device,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    x_row_major: bool,
):
    run_single_routed_expert(
        device,
        allocated_tokens,
        emb_dim,
        hidden_dim,
        active_tokens=active_tokens,
        x_row_major=x_row_major,
    )


@pytest.mark.parametrize(
    "allocated_tokens, active_tokens, emb_dim, hidden_dim, x_row_major",
    _isl_exhaustive_params(),
)
# DRAM ND-sharded weights let the FFN reader fetch a whole K-row weight slice in one
# NoC request instead of one per tile, with the shard->bank round-robin rotating banks
# across K-rows. Both layouts are swept so the interleaved path stays covered.
@pytest.mark.parametrize("weights_dram_sharded", [False, True], ids=["w_interleaved", "w_ndshard"])
@pytest.mark.parametrize("implementation", ["unified", "moe_fused"], ids=["unified", "moe_fused"])
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
def test_single_routed_expert_isl_sweep(
    device,
    allocated_tokens: int,
    active_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    x_row_major: bool,
    weights_dram_sharded: bool,
    implementation: str,
):
    run_single_routed_expert(
        device,
        allocated_tokens,
        emb_dim,
        hidden_dim,
        active_tokens=active_tokens,
        x_row_major=x_row_major,
        weights_dram_sharded=weights_dram_sharded,
        implementation=implementation,
    )
