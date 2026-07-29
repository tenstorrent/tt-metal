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
production fast path (x tilized + bf8-packed inside the op, fresh output).

Baselines in ``_EXPECTED_NS`` were MEASURED LOCALLY on a BH board on 2026-07-29
and must be RECALIBRATED on the perf CI runner: device times are DDR-speed
dependent, so the canonical baselines have to come from the CI runner the check
actually runs on (mirrors the dated recalibration comments in the sibling
``test_moe_perf`` / ``test_dispatch_combine_perf``). Each value is the midpoint of
a 3-sample min/max on one board; see ``_MARGIN`` for the observed spread.
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

# Per-(model, active) UnifiedRoutedExpertFfnDeviceOperation device time in ns, measured
# on a Blackhole P150 (2026-07-27). Recalibrate on the perf CI runner (device times are HW-dependent).
_EXPECTED_NS: dict[tuple[str, int], int] = {
    ("kimi_k26", 0): 3_907,
    ("kimi_k26", 128): 179_491,
    ("kimi_k26", 256): 194_029,
    ("kimi_k26", 512): 254_381,
    ("kimi_k26", 1024): 328_781,
    ("kimi_k26", 2048): 591_428,
    ("kimi_k26", 4096): 1_173_901,
    ("kimi_k26", 5120): 1_493_475,
    ("glm_51", 0): 3_923,
    ("glm_51", 128): 154_624,
    ("glm_51", 256): 171_652,
    ("glm_51", 512): 226_592,
    ("glm_51", 1024): 287_046,
    ("glm_51", 2048): 517_995,
    ("glm_51", 4096): 1_025_398,
    ("glm_51", 5120): 1_305_806,
}

# 5% rather than 3%: re-measuring every case three times on a P150 showed most are
# stable to <1%, but the short-ISL cases are not — kimi isl-256 spans 7.5%
# (186.8/188.2/201.3 us) and glm isl-512 3.6%. Those runs are dominated by fixed
# per-K-block sync latency rather than by streaming work, so they pick up
# run-to-run jitter. Each baseline is the midpoint of its 3-sample min/max, and
# +-5% covers the observed spread; tightening this needs multi-iteration averaging
# in the harness, not a narrower band.
_MARGIN = 0.05


def _k_filter(model: str, active: int) -> str:
    """Pin exactly one ``test_single_routed_expert_isl_sweep`` case: model + isl +
    layout. Disambiguate substring collisions in the pytest ``-k`` match — e.g.
    ``isl-512`` is a substring of ``isl-5120`` — by excluding any other sweep value
    whose id contains this one."""
    parts = [f"{model}-isl-{active}", _LAYOUT_ID]
    parts += [
        f"not isl-{other}" for other in _ISL_EXHAUSTIVE_SWEEP if other != active and f"isl-{active}" in f"isl-{other}"
    ]
    return " and ".join(parts)


def _perf_params():
    params = []
    for model in _ISL_EXHAUSTIVE_MODELS:
        for active in _ISL_EXHAUSTIVE_SWEEP:
            command = f"pytest {_WORKER} -k '{_k_filter(model, active)}'"
            params.append(
                pytest.param(
                    command,
                    {_OP_CODE: _EXPECTED_NS[(model, active)]},
                    f"single_routed_expert_{model}_isl{active}_{_LAYOUT_ID}",
                    id=f"{model}-isl-{active}",
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
