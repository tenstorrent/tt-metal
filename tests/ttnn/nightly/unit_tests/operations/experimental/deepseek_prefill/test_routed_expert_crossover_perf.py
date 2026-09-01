# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device perf gate for the routed-expert crossover: at each ISL, the better of the two ops that can
serve a routed expert, measured with the real-time program profiler.

The load is the single-expert load of test_single_routed_expert_perf.py and
test_moe_fused_swiglu_perf.py, unchanged: one expert, `active_tokens` live rows against the same
_ISL_ALLOCATED_TOKENS region. Both ops run it, each alone in its own profiler window, and the case
gates whichever came back faster.

That is the number the hybrid split exists to deliver, and it is the one thing neither per-op gate
can see. They each hold their own op to its own baseline; only the minimum of the two moves when the
crossover moves. A model's ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD is a claim about where that crossover
sits, and this file measures both sides of it at every sweep point -- a threshold that drifts off the
crossover shows up here as the losing op winning, logged per case.

Neither run is a hybrid: threshold None leaves the composite as the only dispatch, and a threshold at
the region capacity trips TtRoutedExpert's fused_only path, which drops the composite entirely. So
each window holds exactly one op and neither number carries the other's skip launch. The cost of the
split itself -- the two-op forward, its shared output buffer, the skipped band's empty launch -- is
therefore NOT gated here; test_routed_expert_hybrid.py grades that path for correctness only.

Needs a host-IOMMU runner, hence requires_host_iommu: on Blackhole the profiler's D2H socket uses
64-bit PCIe addressing, which requires IOMMU with no hugepage fallback (realtime_profiler_manager.cpp).
"""

import statistics
from typing import Optional

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_p150
from tests.ttnn.profiling.realtime_profiler_utils import (
    profile_realtime_program_merged,
    require_realtime_profiler,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    _ISL_ALLOCATED_TOKENS,
    _ISL_EXHAUSTIVE_MODELS,
    _ISL_EXHAUSTIVE_SWEEP,
    SINGLE_EXPERT_MODELS,
)

# RT records carry kernel sources, not an OP CODE, so identify each op by its kernel directory.
_OP_KERNEL_DIRS = {
    "composite": "/unified_routed_expert_ffn/",
    "fused": "/moe_fused_swiglu/",
}

# Median of this many dispatches. A single dispatch throws occasional flyers, so per-case margins
# turn into whack-a-mole; the median collapses them.
_ITERS = 3

# Log every program in each window -- use when recalibrating.
_VERBOSE = False

_MARGIN = 0.03
# active=0 is launch overhead only; margin=1.0 zeroes the floor and leaves only a ceiling.
_CEILING_ONLY = 1.0
# Below its knee an op skips most chunks, fixed overhead dominates and the median keeps a long right
# tail. The gated number is a minimum over both ops, so it inherits the LATER of the two knees the
# per-op gates calibrated, which is 512 on every shape here.
_LOW_ISL_MARGIN = 0.08
_KNEE_TOKENS = 512

MAX_TOKENS = _ISL_ALLOCATED_TOKENS
WEIGHT_SCALE = 0.02

# Best-of-both device duration in ns per (model, active): median of 3 dispatches on a BH p150b.
# Recalibrate on the perf runner (DDR-speed dependent): each case logs an "RT-CAL" line in this
# dict's format, so one run regenerates the table. A case with no entry measures, logs its line and
# skips rather than asserting -- an empty slot must not report green.
#
# Keyed without the winning op on purpose: at a crossover point the two ops are within noise of each
# other, so the winner flips between runs while the minimum does not. The winner is logged per case
# instead, against the one the model's threshold picks.
_EXPECTED_NS: dict[tuple[str, int], int] = {
    ("kimi_k26", 0): 3_936,
    ("kimi_k26", 128): 96_516,
    ("kimi_k26", 256): 120_004,
    ("kimi_k26", 512): 199_965,
    ("kimi_k26", 1024): 346_680,
    ("kimi_k26", 2048): 638_824,
    ("kimi_k26", 4096): 1_229_701,
    ("kimi_k26", 5120): 1_523_330,
    ("glm_51", 0): 3_732,
    ("glm_51", 128): 85_255,
    ("glm_51", 256): 107_508,
    ("glm_51", 512): 181_859,
    ("glm_51", 1024): 318_813,
    ("glm_51", 2048): 568_589,
    ("glm_51", 4096): 1_111_346,
    ("glm_51", 5120): 1_420_327,
}


def _threshold_of(config) -> Optional[int]:
    """The hybrid split the model ships, read as tt_prefill_block reads it. A config without one
    runs single-op, which TtRoutedExpert spells `None`."""
    return getattr(config, "ROUTED_EXPERT_HYBRID_TOKEN_THRESHOLD", None)


def _predicted_winner(active: int, threshold: Optional[int]) -> Optional[str]:
    """The op the shipped threshold routes this count to, or None where the model ships no split.
    Measuring both ops stays worthwhile there -- it is the data that would justify enabling one --
    but there is no claim to check the winner against."""
    if threshold is None:
        return None
    return "fused" if active <= threshold else "composite"


def _margin_for(active: int) -> float:
    return _CEILING_ONLY if active == 0 else _LOW_ISL_MARGIN if active <= _KNEE_TOKENS else _MARGIN


def _perf_params():
    """Baseline and margin per (model, active) over the exhaustive ISL sweep, dims and threshold from
    SINGLE_EXPERT_MODELS. No extended_model mark: the markers below already scope where these run."""
    params = []
    for name, config, _extended in SINGLE_EXPERT_MODELS:
        if name not in _ISL_EXHAUSTIVE_MODELS:
            continue
        threshold = _threshold_of(config)
        for active in _ISL_EXHAUSTIVE_SWEEP:
            params.append(
                pytest.param(
                    name,
                    active,
                    threshold,
                    config.EMB_SIZE,
                    config.MOE_INTERMEDIATE_SIZE,
                    _EXPECTED_NS.get((name, active)),
                    _margin_for(active),
                    # "-perf" keeps ids collision-free under -k: "512-perf" is not in "5120-perf".
                    id=f"{name}-isl-{active}-perf",
                )
            )
    return params


def _build(device, emb_dim: int, hidden_dim: int, active_tokens: int, activation):
    """Module and forward for one case, built OUTSIDE the measured callable so a profiled window
    carries forwards rather than weight uploads.

    One expert over a MAX_TOKENS region with `active_tokens` live rows: the dispatch buffer
    run_single_routed_expert builds, at the same seed, so the per-op baselines at this ISL are the
    same measurement with the module taken out.

    Constructed at the fused-only threshold, the value that has to pass the constructor's checks;
    the caller retargets `hybrid_token_threshold` to pick the other op. One module serves both
    measurements, so the two ops are compared on identical weights and identical input."""
    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * WEIGHT_SCALE,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * WEIGHT_SCALE,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * WEIGHT_SCALE,
    }

    torch_input = torch.zeros(MAX_TOKENS, emb_dim, dtype=torch.float32)
    torch_input[:active_tokens] = torch.randn(active_tokens, emb_dim, dtype=torch.float32)

    def idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.uint32
        )

    tt_expert = TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=1,
        global_expert_idx_table=idx_tensor([0]),
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=MAX_TOKENS,
        torch_weights=[weights],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=activation,
        hybrid_token_threshold=MAX_TOKENS,
    )
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        # x_rm: the Blackhole fused-tilize production fast path, as both per-op gates measure.
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )
    counts_tt = idx_tensor([active_tokens])
    offsets_tt = idx_tensor([0])
    # Construction dispatches device work of its own. Drain it here or its tail lands inside the
    # profiler window, where it can overlap the first measured forward.
    ttnn.synchronize_device(device)
    return tt_expert, lambda: tt_expert(tt_input, counts_tt, offsets_tt)


def _measure(device, run_fn, kernel_path: str) -> float:
    """Median device duration of the one program matching `kernel_path`, over _ITERS forwards in a
    single profiler window. One match per forward, or the window is not what the caller thinks it
    is and the dump says why."""

    def run_all():
        for _ in range(_ITERS):
            run_fn()

    _, per_program = profile_realtime_program_merged(device, run_all)

    def dump(log):
        for seq, (runtime_id, entry) in enumerate(per_program.items()):  # arrival = dispatch order
            log(
                f"  [{seq}] runtime_id={runtime_id} duration_ns={entry['duration_ns']:.0f} "
                f"kernels={sorted({source.rsplit('/', 1)[-1] for source in entry['kernel_sources']})}"
            )

    if _VERBOSE:
        dump(logger.info)

    matched = [
        entry["duration_ns"]
        for entry in per_program.values()
        if any(kernel_path in source.replace("\\", "/") for source in entry["kernel_sources"])
    ]
    if len(matched) != _ITERS:
        if not _VERBOSE:
            dump(logger.error)
        raise AssertionError(f"expected {_ITERS} programs matching {kernel_path}, got {len(matched)}")
    return statistics.median(matched)


def _gate_best_of_both(
    device,
    emb_dim: int,
    hidden_dim: int,
    active_tokens: int,
    threshold: Optional[int],
    expected_ns: Optional[int],
    margin: float,
    label: str,
    activation=ttnn.RoutedExpertActivation.Silu,
) -> None:
    """Measure both ops standalone at this shape and count, gate the faster one, and report the
    winner against the one the model's threshold predicts."""
    tt_expert, forward = _build(device, emb_dim, hidden_dim, active_tokens, activation)

    durations = {}
    for op, op_threshold in (("fused", MAX_TOKENS), ("composite", None)):
        tt_expert.hybrid_token_threshold = op_threshold
        durations[op] = _measure(device, forward, _OP_KERNEL_DIRS[op])

    winner = min(durations, key=durations.get)
    best = durations[winner]
    predicted = _predicted_winner(active_tokens, threshold)
    logger.info(
        f"RT {label}: composite={durations['composite']:_.0f} fused={durations['fused']:_.0f} ns"
        f" -> {winner} by {abs(durations['composite'] - durations['fused']):_.0f} ns"
    )
    # A bare mismatch is not reportable: at a crossover point the two ops tie to within a fraction
    # of a percent and the winner flips run to run. Only a gap the case's own band would resolve
    # says the shipped crossover has actually moved.
    gap = abs(durations["composite"] - durations["fused"]) / best
    if predicted is not None and winner != predicted and gap > _MARGIN:
        logger.warning(
            f"{label}: threshold {threshold} routes this count to {predicted}, but {winner} measured "
            f"{gap:.1%} faster -- the shipped crossover no longer matches the hardware"
        )

    if expected_ns is None:
        logger.warning(f"RT-CAL {label}: {round(best):_},  # {winner}, NO BASELINE, not gated")
        pytest.skip(f"no baseline for {label}; copy the RT-CAL line into the expected table")

    lower, upper = expected_ns * (1 - margin), expected_ns * (1 + margin)
    assert lower <= best <= upper, (
        f"{label} best-of-both device time {best:.0f} ns ({winner}) outside band "
        f"[{lower:.0f}, {upper:.0f}] ns (expected {expected_ns} ns, margin +/- {margin * 100:.0f}%)"
    )


# TtRoutedExpert takes a mesh_device, so this uses the mesh fixture where the per-op files take
# `device`; the profiler helpers only synchronize on it.
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_name, active_tokens, threshold, emb_dim, hidden_dim, expected_ns, margin", _perf_params()
)
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="the fused routed-expert path is Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="perf baselines are P150-specific; skip on any other board")
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_routed_expert_crossover_perf(
    mesh_device,
    model_name: str,
    active_tokens: int,
    threshold: Optional[int],
    emb_dim: int,
    hidden_dim: int,
    expected_ns: Optional[int],
    margin: float,
):
    require_realtime_profiler("routed expert crossover perf checks")
    _gate_best_of_both(
        mesh_device,
        emb_dim,
        hidden_dim,
        active_tokens,
        threshold,
        expected_ns,
        margin,
        label=f'("{model_name}", {active_tokens})',
    )
