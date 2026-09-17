# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Grades the hybrid op's carried unified half against the op it was carried from.

hybrid_routed_expert_ffn holds a copy of unified_routed_expert_ffn ported from the
CachedProgram model to ProgramDescriptor. The port rewrote every CB, semaphore, kernel and
runtime-arg call site, so the failure it can produce is a shifted runtime arg -- which a PCC
bar against a torch reference can absorb. The bar here is bit-identity against the original
op: same kernels, same args, same result, or the port moved something.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill import ci_pruning

# The hybrid op is not wired into any model yet; nothing in CI should collect it.
pytestmark = pytest.mark.uncollect_if(pred=ci_pruning.no_production_counterpart)

_ALLOCATED_TOKENS = 5120

# Per-expert active counts. 0 covers the skip path, and the rest straddle the block sizes the
# kernels pick so the two halves cannot agree merely by both running one uniform shape.
_ACTIVE_COUNTS = [0, 251, 1024, 3001]

# The split is inclusive on both sides -- the fused half owns [0, threshold], the unified half
# [threshold+1, inf) -- so the only counts that can make BOTH halves claim an expert, or neither,
# sit exactly on the seam. _SEAM_THRESHOLD is chosen so these three land on it.
# Both halves' kernels live in one binary per RISC-V, and all five plus the runtime args have to
# fit ONE core's kernel-config region. At the stock worker_l1_size that region is 70656 B and the
# union program needs 85712. The region is `l1_unreserved_base - KERNEL_CONFIG`, and the base is
# `1572864 - worker_l1_size`, so lowering worker_l1_size grows it -- at the cost of the L1 buffer
# pool the shared arena comes from. 1444864 puts the base at 128000: region 87040 (needs 85712),
# arena 1395712 (needs 1389120). Both fit, neither by much.
_UNION_WORKER_L1_SIZE = 1_444_864

_SEAM_THRESHOLD = 1024
_SEAM_COUNTS = [_SEAM_THRESHOLD - 1, _SEAM_THRESHOLD, _SEAM_THRESHOLD + 1, 3001]


def _idx_tensor(device, values):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.uint32,
    )


def run_port_equivalence(
    device,
    emb_dim: int,
    hidden_dim: int,
    x_row_major: bool,
    activation,
    with_bias: bool,
):
    experts_per_chip = len(_ACTIVE_COUNTS)
    torch.manual_seed(42)

    weights = [
        {
            "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
        }
        for _ in range(experts_per_chip)
    ]
    biases = (
        [
            {
                "gate_proj_bias": torch.randn(hidden_dim, dtype=torch.float32) * 0.05,
                "up_proj_bias": torch.randn(hidden_dim, dtype=torch.float32) * 0.05,
                "down_proj_bias": torch.randn(emb_dim, dtype=torch.float32) * 0.05,
            }
            for _ in range(experts_per_chip)
        ]
        if with_bias
        else None
    )

    # Every expert owns a contiguous _ALLOCATED_TOKENS-row region of one shared buffer, the
    # layout the dispatch op produces. Only the first count rows of each region hold data.
    total_rows = _ALLOCATED_TOKENS * experts_per_chip

    def make_x():
        torch_input = torch.zeros(total_rows, emb_dim, dtype=torch.float32)
        for e, count in enumerate(_ACTIVE_COUNTS):
            base = e * _ALLOCATED_TOKENS
            torch_input[base : base + count] = torch.randn(count, emb_dim, dtype=torch.float32)
        return ttnn.from_torch(
            torch_input,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            layout=ttnn.ROW_MAJOR_LAYOUT if x_row_major else ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.bfloat16 if x_row_major else ttnn.bfloat8_b,
        )

    idx_tt = _idx_tensor(device, list(range(experts_per_chip)))
    counts_tt = _idx_tensor(device, _ACTIVE_COUNTS)
    offsets_tt = _idx_tensor(device, [e * _ALLOCATED_TOKENS for e in range(experts_per_chip)])

    tt_expert = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=_ALLOCATED_TOKENS,
        torch_weights=weights,
        torch_biases=biases,
        activation=activation,
    )

    def run(op, tt_input, **extra):
        # A TILE input is written in place and handed straight back, so each op needs its own
        # copy or the second one grades against the first one's output.
        x = tt_input if x_row_major else ttnn.clone(tt_input)
        out = op(
            x,
            offsets_tt,
            counts_tt,
            idx_tt,
            tt_expert.gate_projs,
            tt_expert.up_projs,
            tt_expert.down_projs,
            max_dispatched_tokens_per_expert=_ALLOCATED_TOKENS,
            compute_kernel_config=tt_expert.compute_kernel_config,
            activation=activation,
            gate_biases=tt_expert.gate_biases,
            up_biases=tt_expert.up_biases,
            down_biases=tt_expert.down_biases,
            **extra,
        )
        return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))

    # Twice, each dispatch on a freshly allocated x holding different data. The second pass is
    # the one that matters for the port: it hits the program cache, where the framework patches
    # the descriptor's buffer bindings in place instead of rebuilding it, so a binding the port
    # left as a raw address would go on pointing at the first pass's buffer.
    for attempt in range(2):
        tt_input = make_x()
        reference = run(ttnn.experimental.deepseek_prefill.unified_routed_expert_moe, tt_input)
        # Threshold 0 leaves every expert on the unified half and runs no fused pass, so this is
        # the carried copy on its own -- the configuration the port has to match bit-for-bit.
        ported = run(ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe, tt_input, hybrid_token_threshold=0)

        # Only the rows the ops actually write are defined; the padding rows of a freshly
        # allocated output are whatever the allocator left there.
        for e, count in enumerate(_ACTIVE_COUNTS):
            base = e * _ALLOCATED_TOKENS
            mismatches = int((reference[base : base + count] != ported[base : base + count]).sum())
            logger.debug(f"pass {attempt}, expert {e} ({count} rows): {mismatches} mismatching elements")
            assert mismatches == 0, (
                f"pass {attempt}, expert {e} ({count} active rows): "
                f"{mismatches} elements differ from the original op"
            )


@pytest.mark.skipif(not is_blackhole(), reason="the unified routed expert is Blackhole-only")
@pytest.mark.parametrize(
    "emb_dim, hidden_dim",
    [
        pytest.param(DeepSeekV3Config.EMB_SIZE, DeepSeekV3Config.MOE_INTERMEDIATE_SIZE, id="dsv3"),
        pytest.param(GLM51Config.EMB_SIZE, GLM51Config.MOE_INTERMEDIATE_SIZE, id="glm_51"),
    ],
)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
def test_hybrid_unified_half_matches_original(device, emb_dim: int, hidden_dim: int, x_row_major: bool):
    run_port_equivalence(
        device,
        emb_dim,
        hidden_dim,
        x_row_major=x_row_major,
        activation=ttnn.RoutedExpertActivation.Silu,
        with_bias=False,
    )


@pytest.mark.skipif(not is_blackhole(), reason="the unified routed expert is Blackhole-only")
@pytest.mark.parametrize(
    "activation, with_bias",
    [
        pytest.param(ttnn.RoutedExpertActivation.SwiGluOai, True, id="swigluoai_bias"),
        pytest.param(ttnn.RoutedExpertActivation.ClampedSiluGlu, False, id="clamped_silu_glu"),
    ],
)
def test_hybrid_unified_half_matches_original_variants(device, activation, with_bias: bool):
    # The bias and non-SiLU epilogues take their own runtime args and their own CBs, which the
    # SiLU case never binds.
    run_port_equivalence(
        device,
        DeepSeekV3Config.EMB_SIZE,
        DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
        x_row_major=True,
        activation=activation,
        with_bias=with_bias,
    )


def run_merged_vs_unified(
    device,
    emb_dim: int,
    hidden_dim: int,
    x_row_major: bool,
    activation,
    with_bias: bool,
    threshold: int = 0,
    active_counts=None,
):
    """The merged op against the two ops it replaces, per expert.

    Each expert is graded against the half that SHOULD own it -- the fused op below the threshold,
    the unified op above it -- and required to match that half more closely than the other. The
    second half of that is what catches a seam bug: both halves claiming one expert is invisible
    to a single reference, because pass B runs last and overwrites with exactly the unified value.

    The bar is PCC, not equality: the merged compute binary runs with bfp8_pack_precise set, which
    the fused half requires and the unified op alone does not use.

    Everything runs twice on freshly allocated inputs, because the second pass is the one that
    exercises the program cache, where buffer bindings are patched in place rather than rebuilt.
    """
    active_counts = list(_ACTIVE_COUNTS if active_counts is None else active_counts)
    experts_per_chip = len(active_counts)
    torch.manual_seed(42)

    weights = [
        {
            "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
        }
        for _ in range(experts_per_chip)
    ]
    biases = (
        [
            {
                "gate_proj_bias": torch.randn(hidden_dim, dtype=torch.float32) * 0.05,
                "up_proj_bias": torch.randn(hidden_dim, dtype=torch.float32) * 0.05,
                "down_proj_bias": torch.randn(emb_dim, dtype=torch.float32) * 0.05,
            }
            for _ in range(experts_per_chip)
        ]
        if with_bias
        else None
    )

    total_rows = _ALLOCATED_TOKENS * experts_per_chip
    torch_input = torch.zeros(total_rows, emb_dim, dtype=torch.float32)
    for e, count in enumerate(active_counts):
        base = e * _ALLOCATED_TOKENS
        torch_input[base : base + count] = torch.randn(count, emb_dim, dtype=torch.float32)

    idx_tt = _idx_tensor(device, list(range(experts_per_chip)))
    counts_tt = _idx_tensor(device, active_counts)
    offsets_tt = _idx_tensor(device, [e * _ALLOCATED_TOKENS for e in range(experts_per_chip)])

    tt_expert = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=_ALLOCATED_TOKENS,
        torch_weights=weights,
        torch_biases=biases,
        activation=activation,
    )

    common = dict(
        max_dispatched_tokens_per_expert=_ALLOCATED_TOKENS,
        compute_kernel_config=tt_expert.compute_kernel_config,
        activation=activation,
        gate_biases=tt_expert.gate_biases,
        up_biases=tt_expert.up_biases,
        down_biases=tt_expert.down_biases,
    )

    def to_torch(out):
        return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))

    def fresh_x():
        # A new allocation each pass, so pass 1 hits the program cache with buffers at different
        # addresses than pass 0 built the descriptor from.
        return ttnn.from_torch(
            torch_input,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            layout=ttnn.ROW_MAJOR_LAYOUT if x_row_major else ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.bfloat16 if x_row_major else ttnn.bfloat8_b,
        )

    def unified_reference(x_src):
        x = x_src if x_row_major else ttnn.clone(x_src)
        return to_torch(
            ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
                x,
                offsets_tt,
                counts_tt,
                idx_tt,
                tt_expert.gate_projs,
                tt_expert.up_projs,
                tt_expert.down_projs,
                **common,
            )
        )

    def fused_reference(x_src):
        # The fused op refuses to write into its own activations, so it always needs its own
        # output. Same band the merged op gives its pass A.
        #
        # ZEROED, not empty: the fused op writes only the experts inside its band, and a fresh
        # allocation lands on DRAM the merged output just used -- so an unwritten region reads
        # back as the merged result and compares equal to it, which would make this reference
        # agree with everything and the owner check below vacuous.
        out = ttnn.zeros(
            x_src.shape,
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            x_src,
            tt_expert.gate_projs,
            tt_expert.up_projs,
            tt_expert.down_projs,
            counts_tt,
            idx_tt,
            input_m_tiles=_ALLOCATED_TOKENS // 32,
            core_grid=ttnn.UNIFIED_ROUTED_EXPERT_CORE_GRID,
            activation=activation,
            output=out,
            expert_region_offsets=offsets_tt,
            read_x_at_offset=True,
            max_active_tokens=threshold,
            gate_biases=tt_expert.gate_biases,
            up_biases=tt_expert.up_biases,
            down_biases=tt_expert.down_biases,
        )
        return to_torch(out)

    def merged(x_src):
        x = x_src if x_row_major else ttnn.clone(x_src)
        return to_torch(
            ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
                x,
                offsets_tt,
                counts_tt,
                idx_tt,
                tt_expert.gate_projs,
                tt_expert.up_projs,
                tt_expert.down_projs,
                hybrid_token_threshold=threshold,
                **common,
            )
        )

    # Twice, on freshly allocated inputs. Pass 1 is the one that matters: it hits the program
    # cache, where the framework patches the descriptor's buffer bindings in place. The merged op
    # carries more of that state than either half alone -- every unified binding is re-indexed by
    # the fused block length, and every circular buffer points into an arena tensor reallocated on
    # each call.
    for attempt in range(2):
        x_src = fresh_x()
        unified_ref = unified_reference(x_src)
        merged_out = merged(x_src)
        fused_ref = fused_reference(x_src) if threshold > 0 else None

        bad = []
        for e, count in enumerate(active_counts):
            if count == 0:
                continue
            base = e * _ALLOCATED_TOKENS
            got = merged_out[base : base + count]
            _, pcc_unified = comp_pcc(unified_ref[base : base + count], got)

            # Which half SHOULD own this expert. The bands are inclusive on both sides.
            owner_is_fused = threshold > 0 and count <= threshold
            if fused_ref is None:
                logger.info(f"pass {attempt}, expert {e} ({count} rows): pcc_unified={pcc_unified}")
                if pcc_unified < 0.999:
                    bad.append(f"expert {e} ({count} rows): pcc {pcc_unified} vs the unified op")
                continue

            _, pcc_fused = comp_pcc(fused_ref[base : base + count], got)
            logger.info(
                f"pass {attempt}, expert {e} ({count} rows): owner={'fused' if owner_is_fused else 'unified'}, "
                f"pcc_fused={pcc_fused}, pcc_unified={pcc_unified}"
            )
            # Matching the owning half is correctness. Matching it BETTER than the other half is
            # the part that catches a seam bug: if both halves claimed the expert, pass B runs
            # last and the result would look like the unified reference no matter which half was
            # supposed to own it -- which the owner check alone cannot see.
            owner_pcc, other_pcc = (pcc_fused, pcc_unified) if owner_is_fused else (pcc_unified, pcc_fused)
            if owner_pcc < 0.999:
                bad.append(f"expert {e} ({count} rows): pcc {owner_pcc} against its own half")
            if owner_pcc < other_pcc:
                bad.append(
                    f"expert {e} ({count} rows): closer to the other half "
                    f"(fused={pcc_fused}, unified={pcc_unified}) -- claimed by the wrong half, or by both"
                )
            if torch.isnan(got).any():
                bad.append(f"expert {e} ({count} rows): NaN")
        assert not bad, f"pass {attempt}: " + "; ".join(bad)


@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.parametrize("device_params", [{"worker_l1_size": _UNION_WORKER_L1_SIZE}], indirect=True)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
# threshold 100 is below every non-zero count, so the fused half is configured and placed but
# claims no expert. It separates "the fused half wrote something wrong" from "the unified half
# behaves differently once the output no longer aliases x", which only a live fused band forces.
@pytest.mark.parametrize("threshold", [512, 100], ids=["split", "fused_empty_band"])
def test_merged_op_both_passes(device, x_row_major: bool, threshold: int):
    """Both halves live in one program, on disjoint columns of the same 88-core rectangle.

    _ACTIVE_COUNTS straddles the threshold deliberately (251 below it, 1024 and 3001 above), so
    each half owns some experts and both must work for the output to be complete. The reference
    is the unified op run over every expert on its own.

    The bar is PCC, not equality: the fused and unified implementations do not agree bit-for-bit
    on the same expert, and here the low-count expert is computed by a different one than in the
    reference.
    """
    run_merged_vs_unified(
        device,
        DeepSeekV3Config.EMB_SIZE,
        DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
        x_row_major=x_row_major,
        activation=ttnn.RoutedExpertActivation.Silu,
        with_bias=False,
        threshold=threshold,
    )


@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.parametrize("device_params", [{"worker_l1_size": _UNION_WORKER_L1_SIZE}], indirect=True)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
def test_merged_op_both_passes_with_bias(device, x_row_major: bool):
    """Both halves live WITH biases -- the path the bias compile-time args actually reach.

    Worth its own case because the bias CB ids are the one place the unified half reads the
    compile-time array at an index derived from an accessor offset rather than a literal. That
    index is already absolute, so it must bypass the rebasing shim; getting it wrong is invisible
    until a merged build puts the unified half at a non-zero base, which only happens here.
    """
    run_merged_vs_unified(
        device,
        DeepSeekV3Config.EMB_SIZE,
        DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
        x_row_major=x_row_major,
        activation=ttnn.RoutedExpertActivation.SwiGluOai,
        with_bias=True,
        threshold=512,
    )


@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.parametrize("device_params", [{"worker_l1_size": _UNION_WORKER_L1_SIZE}], indirect=True)
@pytest.mark.parametrize("x_row_major", [True, False], ids=["x_rm", "x_tile"])
def test_merged_op_band_seam(device, x_row_major: bool):
    """Counts sitting exactly on the band boundary.

    The fused half owns [0, threshold] and the unified half [threshold+1, inf), both inclusive on
    their own side, so threshold and threshold+1 are the only counts where an off-by-one makes
    both halves claim an expert or neither does. _SEAM_COUNTS puts an expert on each side of the
    seam and one exactly on it.
    """
    run_merged_vs_unified(
        device,
        DeepSeekV3Config.EMB_SIZE,
        DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
        x_row_major=x_row_major,
        activation=ttnn.RoutedExpertActivation.Silu,
        with_bias=False,
        threshold=_SEAM_THRESHOLD,
        active_counts=_SEAM_COUNTS,
    )


# The barrier is bespoke and grid-wide, and this op family has produced an 88-core wedge at
# roughly 0.4% per dispatch before -- a handful of launches would miss that most of the time.
# Opt-in rather than always-on because a soak costs minutes, not seconds.
_SOAK_DISPATCHES = int(os.environ.get("TT_HYBRID_RE_SOAK", "0"))


@pytest.mark.skipif(not is_blackhole(), reason="the routed expert is Blackhole-only")
@pytest.mark.skipif(_SOAK_DISPATCHES <= 0, reason="set TT_HYBRID_RE_SOAK=<n> to soak the pass barrier")
@pytest.mark.parametrize("device_params", [{"worker_l1_size": _UNION_WORKER_L1_SIZE}], indirect=True)
def test_merged_op_barrier_soak(device):
    """Repeat the split configuration to expose a low-probability barrier race.

    A wedge shows up as a dispatch that never returns, so the assertion is reaching the end; the
    per-iteration value check guards against a release that lets pass B start early and read
    half-written L1 rather than hanging outright.
    """
    for i in range(_SOAK_DISPATCHES):
        run_merged_vs_unified(
            device,
            DeepSeekV3Config.EMB_SIZE,
            DeepSeekV3Config.MOE_INTERMEDIATE_SIZE,
            x_row_major=True,
            activation=ttnn.RoutedExpertActivation.Silu,
            with_bias=False,
            threshold=512,
        )
        if (i + 1) % 10 == 0:
            logger.info(f"barrier soak: {i + 1}/{_SOAK_DISPATCHES} iterations")
