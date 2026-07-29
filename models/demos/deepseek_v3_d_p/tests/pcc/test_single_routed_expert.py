# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Minimal single-device, single-expert test for TtRoutedExpert profiling.

The simplest scenario: 1 chip, 1 expert, minimal dimensions.
"""

import os
import time

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


def run_single_routed_expert(
    device,
    allocated_tokens: int,
    emb_dim: int,
    hidden_dim: int,
    active_tokens: int = None,
    x_row_major: bool = False,
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
    experts_per_chip = 1

    signpost(f"SingleRoutedExpert {allocated_tokens=} {active_tokens=} {emb_dim=} {hidden_dim=}")

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
        activation=ttnn.RoutedExpertActivation.Silu,
    )

    # Run TTNN forward
    logger.debug("Running TTNN forward...")
    tt_output = tt_expert(tt_input, expert_token_counts_tt, expert_region_offsets_tt)
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

# Exhaustive sweep: the full range from empty to fully-packed. 126 is the token
# count of the expert that wedged device 10 in the 8x4 Kimi run (see the hang
# replay at the bottom of this file): count_tiles=4 like 128, but NOT a multiple
# of TILE_HEIGHT, so it also drives the partial-tail-tile read path (30 valid
# rows in the last tile-row) that every other value here misses at this size.
_ISL_EXHAUSTIVE_SWEEP = [0, 126, 128, 256, 512, 1024, 2048, 4096, 5120]
_ISL_EXHAUSTIVE_MODELS = ("kimi_k26", "glm_51")


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
    "allocated_tokens, active_tokens, emb_dim, hidden_dim",
    _isl_params(_ISL_EXHAUSTIVE_SWEEP, only_models=_ISL_EXHAUSTIVE_MODELS),
)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
def test_single_routed_expert_isl_sweep(
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


# ---------------------------------------------------------------------------
# Multi-expert hang replay
# ---------------------------------------------------------------------------
# Single-chip replay of a Tensix lockup seen on a 32-chip Kimi K2.7 prefill run
# (mesh-8x4, L61, chunks20, iters20). Metal device 10 wedged in
# UnifiedRoutedExpertFfnDeviceOperation op 161106 while its 7 column peers had
# already moved on to CombineDeviceOperation, stalling the whole mesh behind it.
#
# All 88 cores on that device sat at an IDENTICAL PC:
#   UNPACK (trisc0) load_replay_buf   <- matmul_block_init, fused_swiglu.cpp:390
#   PACK   (trisc2) cfg_reg_rmw_tensix <- pack_reconfig_data_format, :391
#   MATH   (trisc1) no credible frame (halted mid-Tensix-instruction)
#   reader (ncrisc) cb_reserve_back, reader.cpp:427
#   writer (brisc)  cb_wait_front(cb_out), writer.cpp:257
# i.e. the Tensix backend stopped retiring, not a CB/NoC deadlock: both UNPACK
# and PACK are blocked pushing into a full instruction FIFO behind a MATH thread
# that never completes.
#
# The 12 counts below were read off the still-hung device from the reader's L1
# scratch CBs (c_10 counts / c_11 local->global idx) with
# tools/triage/dump_expert_counts.py. Op 161106 is local expert 6 (the composite
# emits one op per local expert in order, unified_routed_expert_ffn.cpp:147),
# i.e. global expert 318 with 126 tokens.
#
# Note this is NOT a shape-dependent failure: every one of the 12 counts is
# <= 6 tile-rows, so adaptive_chunk.hpp gives per_core_M = 1 and 1 chunk for ALL
# of them (kGridY=8, per_core_M_max=8, max_chunk=64). Locals 0 and 5 have the
# same count_tiles=4 as local 6 and both completed. The geometry is identical
# across all 12 ops, so the trigger is a race, and reproducing it needs the 12
# ops dispatched back-to-back many times over -- not one op in isolation.
_HANG_REPLAY_GLOBAL_EXPERT_IDS = list(range(312, 324))
_HANG_REPLAY_TOKEN_COUNTS = [118, 170, 7, 5, 87, 120, 126, 51, 135, 33, 79, 6]
# max_dispatched_tokens_per_expert on the failing run (x was [40960, 7168]).
_HANG_REPLAY_ALLOCATED_TOKENS = 40960
_HANG_REPLAY_NUM_ROUTED_EXPERTS = KimiK26Config.NUM_ROUTED_EXPERTS  # 384
_HANG_REPLAY_EXPERTS_PER_CHIP = len(_HANG_REPLAY_TOKEN_COUNTS)  # 12
#
# Scale. ONE iteration of the loop below == ONE unified_routed_expert_moe call
# (tt_routed_expert.py:500 -- the fused Blackhole path dispatches all 12 local
# experts per call, no per-expert Python loop), so iterations here are directly
# comparable to mesh-level op calls in the real run:
#
#   MoE layers          61 - 1 dense        = 60      (NUM_DENSE_LAYERS = 1)
#   per iteration       20 chunks x 60      = 1,200 calls
#   per run             20 iters x 1,200    = 24,000 calls
#   10 runs                                 = 240,000 calls
#
#   mesh-level op calls                      240,000
#   per-device expert FFNs (x12 local)       2.88M per device
#   across the 32-device mesh                92.2M expert executions
#
# This test replays ONE device's 12 experts, so matching the mesh-wide 92.2M
# expert executions on a single chip means ~7.68M iterations. Measured throughput
# is ~13,000 iterations/min (interleave on), i.e.
#
#   240,000 iterations   (per-device figure)      ~0.3 h
#   8,333,333 iterations (100M expert executions) ~10.7 h
#   100,000,000 iterations (literal)              ~128 h / 5.3 days
#
# Pick the scale by EXPERT EXECUTIONS (the unit the table above uses) --
# iterations are derived, so the count stays right if experts_per_chip changes:
#   HANG_REPLAY_EXPERT_EXECUTIONS=100000000 pytest ... -k hang_replay
# or set the raw iteration count directly (takes effect only if the expert-
# execution target is unset):
#   HANG_REPLAY_ITERATIONS=100000 pytest ... -k hang_replay
# Default stays 100 iterations so a nightly pass is cheap.
_HANG_REPLAY_TARGET_EXPERT_EXECUTIONS = int(os.environ.get("HANG_REPLAY_EXPERT_EXECUTIONS", "0"))
if _HANG_REPLAY_TARGET_EXPERT_EXECUTIONS > 0:
    _HANG_REPLAY_ITERATIONS = -(-_HANG_REPLAY_TARGET_EXPERT_EXECUTIONS // _HANG_REPLAY_EXPERTS_PER_CHIP)  # ceil-div
else:
    _HANG_REPLAY_ITERATIONS = int(os.environ.get("HANG_REPLAY_ITERATIONS", "100"))
# Dispatch an unrelated program between expert batches. The real run interleaves
# dispatch/combine/reduce ops between each layer's 12 experts, churning per-core
# Tensix state (replay buffer, packer/unpacker config) and the launch-message /
# kernel-config rings; a bare back-to-back loop re-runs the same 12 cached
# programs from the same kernel_config_base every pass and never produces that
# interleaving. Since the hung UNPACK sat in load_replay_buf -- per-core state
# that survives program launches -- this is the more likely trigger of the two.
#   HANG_REPLAY_INTERLEAVE=1 pytest ... -k hang_replay
_HANG_REPLAY_INTERLEAVE = os.environ.get("HANG_REPLAY_INTERLEAVE", "0") != "0"
_TILE_HEIGHT = 32


def _tile_aligned_region_offsets(counts: list[int]) -> tuple[list[int], int]:
    """Exclusive prefix sum of TILE-aligned counts, matching what offset_cumsum
    produces for one chip's expert group (the reader/writer turn start[global_id]
    into a tile offset via `start_value / TILE_HEIGHT`, so starts must be
    tile-aligned)."""
    offsets = []
    acc = 0
    for count in counts:
        offsets.append(acc)
        acc += ((count + _TILE_HEIGHT - 1) // _TILE_HEIGHT) * _TILE_HEIGHT
    return offsets, acc


# Derived from the iteration count, because a fixed timeout silently turns a long
# soak into a spurious failure. Budget: ~600 s of setup (36 bfloat4 weight
# conversions + JIT-building 12 expert programs) plus iterations at a deliberately
# pessimistic 100/s (measured ~216/s with interleave on, so >2x headroom). A soak
# at 100M expert executions therefore gets ~24 h, not pytest.ini's 300 s default.
_HANG_REPLAY_TIMEOUT_S = 600 + _HANG_REPLAY_ITERATIONS // 100


@pytest.mark.extended_model
# Deliberately NOT @pytest.mark.slow: tests/ttnn/conftest.py auto-skips that marker
# unless --runslow is passed, and this regression should run when selected.
@pytest.mark.timeout(_HANG_REPLAY_TIMEOUT_S)
@pytest.mark.skipif(not is_blackhole(), reason="device-side count-aware sparsity is Blackhole-only")
def test_routed_expert_hang_replay_kimi_dev10(device):
    """Replays metal device 10's 12-expert dispatch from the hung 8x4 Kimi K2.7 run,
    100 times, on one chip. Each iteration dispatches all 12 UnifiedRoutedExpertFfn
    ops back-to-back exactly as the composite does, so the 1200 op launches give the
    race the repetition the single-expert cases cannot. A hang fails the test by
    wedging the device (triage it with tools/triage/dump_expert_counts.py and
    --run=dump_callstacks before killing the process)."""
    emb_dim = KimiK26Config.EMB_SIZE  # 7168
    hidden_dim = KimiK26Config.MOE_INTERMEDIATE_SIZE  # 2048
    experts_per_chip = len(_HANG_REPLAY_TOKEN_COUNTS)  # 12
    counts = _HANG_REPLAY_TOKEN_COUNTS
    region_offsets, region_span = _tile_aligned_region_offsets(counts)

    logger.info(
        f"Hang replay: {experts_per_chip} experts, globals "
        f"{_HANG_REPLAY_GLOBAL_EXPERT_IDS[0]}-{_HANG_REPLAY_GLOBAL_EXPERT_IDS[-1]}, "
        f"{emb_dim=}, {hidden_dim=}, allocated={_HANG_REPLAY_ALLOCATED_TOKENS}, "
        f"{_HANG_REPLAY_ITERATIONS} iterations"
    )
    for local, (global_id, count, offset) in enumerate(zip(_HANG_REPLAY_GLOBAL_EXPERT_IDS, counts, region_offsets)):
        count_tiles = (count + _TILE_HEIGHT - 1) // _TILE_HEIGHT
        logger.info(
            f"  local {local:2d} -> global {global_id}  tokens={count:4d}  "
            f"count_tiles={count_tiles}  region_offset={offset}"
        )
    assert (
        region_span <= _HANG_REPLAY_ALLOCATED_TOKENS
    ), f"expert regions span {region_span} rows, exceeding the {_HANG_REPLAY_ALLOCATED_TOKENS}-row buffer"

    # Per-expert random weights in HF (out_features, in_features) layout, indexed
    # by LOCAL expert: on a 1-device mesh gather_weights_for_mesh_distribution maps
    # local L -> torch_weights[L]. The global ids only drive the kernel's
    # counts[idx[local]] / start[idx[local]] lookups.
    torch.manual_seed(42)
    weights = [
        {
            "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
        }
        for _ in range(experts_per_chip)
    ]

    # Shared dispatch buffer: each expert's `count` real rows start at its
    # tile-aligned region offset, everything else is zero padding. bfloat16 to keep
    # the 40960-row host buffer at ~590 MB.
    torch_input = torch.zeros(_HANG_REPLAY_ALLOCATED_TOKENS, emb_dim, dtype=torch.bfloat16)
    torch_expert_inputs = []
    for count, offset in zip(counts, region_offsets):
        expert_input = torch.randn(count, emb_dim, dtype=torch.bfloat16)
        torch_input[offset : offset + count] = expert_input
        torch_expert_inputs.append(expert_input)

    # Torch reference per expert, over its active rows only.
    logger.info("Computing torch reference for all experts...")
    torch_references = []
    with torch.no_grad():
        for local in range(experts_per_chip):
            torch_expert = TorchExpert(emb_dim, hidden_dim, weights[local])
            torch_references.append(torch_expert(torch_expert_inputs[local].to(torch.float32)))

    # ROW_MAJOR bf16 x -- the layout the failing run used (tilized and bf8-packed
    # inside the op), which is also what drives the reader's clamp-read of the
    # padding rows past each runtime count.
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )

    def _make_idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            dtype=ttnn.uint32,
        )

    # counts / region offsets are indexed by GLOBAL expert id, so both vectors are
    # full length (NUM_ROUTED_EXPERTS) with this chip's group filled in -- the same
    # shapes the failing op reported ([1, 384] UINT32).
    counts_vec = [0] * _HANG_REPLAY_NUM_ROUTED_EXPERTS
    offsets_vec = [0] * _HANG_REPLAY_NUM_ROUTED_EXPERTS
    for global_id, count, offset in zip(_HANG_REPLAY_GLOBAL_EXPERT_IDS, counts, region_offsets):
        counts_vec[global_id] = count
        offsets_vec[global_id] = offset

    global_expert_idx_tt = _make_idx_tensor(_HANG_REPLAY_GLOBAL_EXPERT_IDS)
    expert_token_counts_tt = _make_idx_tensor(counts_vec)
    expert_region_offsets_tt = _make_idx_tensor(offsets_vec)

    tt_expert = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=_HANG_REPLAY_ALLOCATED_TOKENS,
        torch_weights=weights,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=ttnn.RoutedExpertActivation.Silu,
    )

    pcc_threshold = 0.97
    # PCC on the first and last iteration only: reading back the [40960, 7168]
    # output every pass would dominate the runtime (~2.4 s each), and the point of
    # the loop is repetition, not N identical correctness checks.
    validated_iterations = {0, _HANG_REPLAY_ITERATIONS - 1}
    # Keep the progress log to ~100 lines regardless of the iteration count.
    log_every = max(1, _HANG_REPLAY_ITERATIONS // 100)

    # Interleave source: a tile-layout slice of the dispatch buffer, wide enough to
    # span many cores so the churn lands on the same grid the expert ops use.
    interleave_src = None
    if _HANG_REPLAY_INTERLEAVE:
        interleave_src = ttnn.from_torch(
            torch_input[:4096],
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.bfloat16,
        )
        logger.info("Interleave ON: dispatching a typecast between expert batches")

    # A signpost per iteration is fine at 100 but not at millions -- tracy would
    # accumulate one zone name per iteration and the string formatting alone becomes
    # measurable. Above a few thousand iterations, signpost on the progress cadence.
    signpost_every = 1 if _HANG_REPLAY_ITERATIONS <= 1000 else log_every
    started_at = time.monotonic()

    for iteration in range(_HANG_REPLAY_ITERATIONS):
        if iteration % signpost_every == 0:
            signpost(f"HangReplay iter {iteration + 1}/{_HANG_REPLAY_ITERATIONS}")
        tt_output = tt_expert(tt_input, expert_token_counts_tt, expert_region_offsets_tt)
        ttnn.synchronize_device(device)

        if interleave_src is not None:
            # A different program on the same cores between expert batches: drives
            # unpack/pack data-format reconfig and moves the kernel-config ring on,
            # so the next expert batch does not start from byte-identical per-core
            # Tensix state.
            interleaved = ttnn.typecast(interleave_src, ttnn.bfloat8_b)
            ttnn.synchronize_device(device)
            ttnn.deallocate(interleaved)

        if iteration in validated_iterations:
            tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
            for local in range(experts_per_chip):
                count = counts[local]
                offset = region_offsets[local]
                actual = tt_output_torch[offset : offset + count]
                _, pcc = comp_pcc(torch_references[local], actual)
                logger.info(
                    f"iter {iteration}: local {local:2d} "
                    f"(global {_HANG_REPLAY_GLOBAL_EXPERT_IDS[local]}, {count} tokens) PCC={pcc:.6f}"
                )
                assert pcc >= pcc_threshold, (
                    f"iter {iteration}, local expert {local} "
                    f"(global {_HANG_REPLAY_GLOBAL_EXPERT_IDS[local]}): PCC {pcc:.6f} below {pcc_threshold}"
                )
                assert not torch.isnan(actual).any(), f"iter {iteration}, local expert {local}: output contains NaN"
                assert not torch.isinf(actual).any(), f"iter {iteration}, local expert {local}: output contains Inf"
            del tt_output_torch

        ttnn.deallocate(tt_output)
        if (iteration + 1) % log_every == 0:
            done = iteration + 1
            elapsed = time.monotonic() - started_at
            rate = done / elapsed if elapsed > 0 else 0.0
            eta_h = (_HANG_REPLAY_ITERATIONS - done) / rate / 3600 if rate > 0 else float("nan")
            logger.info(
                f"Completed {done}/{_HANG_REPLAY_ITERATIONS} iterations "
                f"({done * experts_per_chip} expert executions) "
                f"elapsed={elapsed / 3600:.2f}h rate={rate:.0f} it/s eta={eta_h:.2f}h"
            )

    if interleave_src is not None:
        ttnn.deallocate(interleave_src)
    total_elapsed = time.monotonic() - started_at
    logger.info(
        f"Hang replay PASSED: {_HANG_REPLAY_ITERATIONS} iterations "
        f"({_HANG_REPLAY_ITERATIONS * experts_per_chip} expert executions, "
        f"interleave={'on' if _HANG_REPLAY_INTERLEAVE else 'off'}) "
        f"in {total_elapsed / 3600:.2f}h, no hang"
    )
