# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Device performance test for the unified routed expert FFN op.

Perf counterpart to ``test_single_routed_expert_isl_sweep``: for kimi and glm
across the exhaustive ISL (active-token) sweep against the fixed 5K allocated
buffer, spawn one worker per (model, active) that runs ``run_single_routed_expert``
on device under Tracy, and assert the ``UnifiedRoutedExpertFfnDeviceOperation``
device time against a per-case baseline. One op per worker, so a regression
localizes to the FFN kernel.

The ROW_MAJOR x layout (``x_rm``) is measured — the Blackhole fused-tilize
production fast path (x tilized + bf8-packed inside the op, fresh output). BOTH
weight memory layouts are covered, DRAM-interleaved and DRAM ND-sharded, each with
its own baselines, so a regression in either is caught and the gap between them
stays visible (see ``_WEIGHTS_IDS``).

Baselines in ``_EXPECTED_NS`` were MEASURED LOCALLY on a BH board on 2026-07-29
and must be RECALIBRATED on the perf CI runner: device times are DDR-speed
dependent, so the canonical baselines have to come from the CI runner the check
actually runs on (mirrors the dated recalibration comments in the sibling
``test_moe_perf`` / ``test_dispatch_combine_perf``). One sample per case; see
``_MARGIN`` for the observed run-to-run spread.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_per_op
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_p150
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    _ISL_EXHAUSTIVE_MODELS,
    _ISL_EXHAUSTIVE_SWEEP,
)

# Device-op code emitted by the routed expert FFN; the harness sums the rows whose
# OP CODE contains this substring, so incidental setup ops are excluded.
_OP_CODE = "UnifiedRoutedExpertFfnDeviceOperation"

# Worker that runs the op on device (its signposts + the op launch land in the
# Tracy CSV the harness reads back).
_WORKER = (
    "tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/"
    "test_single_routed_expert.py::test_single_routed_expert_isl_sweep"
)
_LAYOUT_ID = "x_rm"  # Blackhole fused-tilize production fast path (ROW_MAJOR bf16 input)

# Weight memory layouts the worker sweeps, both measured here. The ``-k`` filter MUST pin
# one of them: without it a single perf invocation runs both cases, the ops CSV holds two
# FFN rows, and the harness sums them into a ~2x "measured" value.
#
#   w_interleaved — DRAM-interleaved weights (page = tile). One NoC request per tile.
#   w_ndshard     — DRAM ND-sharded on a [TILE, per_core_N*TILE] shard, so a core's whole
#                   K-row weight slice is ONE request and consecutive K-rows rotate DRAM
#                   banks. Worth 1.19-1.48x at isl <= 512, where the fixed weight read is
#                   the critical path, and neutral (within +-2%) from isl 1024 up, where
#                   the x read and the matmuls dominate instead.
_WEIGHTS_IDS = ("w_interleaved", "w_ndshard")

# Per-(weights layout, model, active) UnifiedRoutedExpertFfnDeviceOperation device time in
# ns, measured on a Blackhole P150 (2026-07-29, card 0, single sample per case).
# Recalibrate on the perf CI runner — device times are HW/DDR-speed dependent.
_EXPECTED_NS: dict[tuple[str, str, int], int] = {
    # ---- DRAM-interleaved weights ----
    ("w_interleaved", "kimi_k26", 0): 3_952,
    ("w_interleaved", "kimi_k26", 128): 173_988,
    ("w_interleaved", "kimi_k26", 256): 190_135,
    ("w_interleaved", "kimi_k26", 512): 244_350,
    ("w_interleaved", "kimi_k26", 1024): 309_656,
    ("w_interleaved", "kimi_k26", 2048): 592_184,
    ("w_interleaved", "kimi_k26", 4096): 1_186_882,
    ("w_interleaved", "kimi_k26", 5120): 1_471_016,
    ("w_interleaved", "glm_51", 0): 3_866,
    ("w_interleaved", "glm_51", 128): 146_539,
    ("w_interleaved", "glm_51", 256): 166_129,
    ("w_interleaved", "glm_51", 512): 223_597,
    ("w_interleaved", "glm_51", 1024): 269_973,
    ("w_interleaved", "glm_51", 2048): 518_296,
    ("w_interleaved", "glm_51", 4096): 1_026_701,
    ("w_interleaved", "glm_51", 5120): 1_289_506,
    # ---- DRAM ND-sharded weights ----
    ("w_ndshard", "kimi_k26", 0): 3_978,
    ("w_ndshard", "kimi_k26", 128): 139_511,
    ("w_ndshard", "kimi_k26", 256): 157_155,
    ("w_ndshard", "kimi_k26", 512): 180_430,
    ("w_ndshard", "kimi_k26", 1024): 308_223,
    ("w_ndshard", "kimi_k26", 2048): 590_855,
    ("w_ndshard", "kimi_k26", 4096): 1_167_613,
    ("w_ndshard", "kimi_k26", 5120): 1_477_976,
    ("w_ndshard", "glm_51", 0): 3_945,
    ("w_ndshard", "glm_51", 128): 122_241,
    ("w_ndshard", "glm_51", 256): 136_586,
    ("w_ndshard", "glm_51", 512): 157_119,
    ("w_ndshard", "glm_51", 1024): 269_410,
    ("w_ndshard", "glm_51", 2048): 517_196,
    ("w_ndshard", "glm_51", 4096): 1_027_824,
    ("w_ndshard", "glm_51", 5120): 1_288_875,
}

# 8% rather than the usual 3%: repeated runs of the SAME build on the same card show the
# long-ISL cases stable to <1% but the short-ISL ones spanning ~7% — e.g. w_ndshard kimi
# isl-256 measured 151.6 and 162.7 us, and w_interleaved glm isl-512 measured 214.9 and
# 232.3 us. Those cases are dominated by fixed per-K-block sync latency rather than by
# streaming work, so they pick up jitter that a single-sample baseline cannot represent.
# Cases with more than one observation are centred on their min/max midpoint. Tightening
# this needs multi-iteration averaging in the harness, not a narrower band.
_MARGIN = 0.08


def _k_filter(model: str, active: int, weights: str) -> str:
    """Pin exactly one ``test_single_routed_expert_isl_sweep`` case: model + isl + x
    layout + weight layout. Disambiguate substring collisions in the pytest ``-k`` match
    — e.g. ``isl-512`` is a substring of ``isl-5120`` — by excluding any other sweep value
    whose id contains this one."""
    parts = [f"{model}-isl-{active}", _LAYOUT_ID, weights]
    parts += [
        f"not isl-{other}" for other in _ISL_EXHAUSTIVE_SWEEP if other != active and f"isl-{active}" in f"isl-{other}"
    ]
    return " and ".join(parts)


def _perf_params():
    params = []
    for weights in _WEIGHTS_IDS:
        for model in _ISL_EXHAUSTIVE_MODELS:
            for active in _ISL_EXHAUSTIVE_SWEEP:
                command = f"pytest {_WORKER} -k '{_k_filter(model, active, weights)}'"
                params.append(
                    pytest.param(
                        command,
                        {_OP_CODE: _EXPECTED_NS[(weights, model, active)]},
                        f"single_routed_expert_{model}_isl{active}_{_LAYOUT_ID}_{weights}",
                        id=f"{weights}-{model}-isl-{active}",
                    )
                )
    return params


@pytest.mark.parametrize("command, expected_per_op, model_name", _perf_params())
@pytest.mark.models_device_performance_bare_metal
# Gate to P150 via tt-smi board telemetry (SMBus). This also skips Wormhole and any
# other board. Do NOT use ttnn.cluster.get_cluster_type() here: it opens and locks the
# chip, and since skipif is evaluated at collection time in the parent process, the
# spawned Tracy worker then deadlocks on CHIP_IN_USE. is_p150() reads tt-smi only, so
# it takes no device lock.
@pytest.mark.skipif(not is_p150(), reason="perf baselines are P150-specific; skip on any other board")
def test_single_routed_expert_perf(command, expected_per_op, model_name):
    run_model_device_perf_test_per_op(
        command=command,
        expected_per_op=expected_per_op,
        subdir="prefill_single_routed_expert",
        model_name=model_name,
        margin=_MARGIN,
        comments=model_name,
    )
