# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device kernel time for the unoptimized port, via the repo's device-perf harness.

This is the third of the three baseline numbers. Subtracting it from the steady-state host
latency in test_nomic_perf.py gives the dispatch gap, which is what decides whether trace is
worth doing before or after the expert work.

EXPECTED_SEQUENCES_PER_S is the measured baseline, not a target. It exists so a later change
that silently regresses the kernel total fails here; raise it whenever an optimization lands.

Run this separately from anything using TT_METAL_WATCHER: the profiler and Watcher contend for
the same debug resources.
"""

import pytest
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.tests.perf.perf_common import HEADLINE_SHAPE
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf

# Measured on a p300c at origin/main 40653194d78, bfloat16 weights and activations,
# HiFi4 with fp32 destination accumulation, DRAM-interleaved, no trace, one command queue.
EXPECTED_SEQUENCES_PER_S = 66.7

# Wide enough to absorb run-to-run kernel variation, tight enough to catch a real regression.
MARGIN = 0.03


@run_for_blackhole()
@pytest.mark.models_device_performance_bare_metal
def test_device_perf_nomic_embed_text_v2_moe():
    batch, seqlen = HEADLINE_SHAPE
    subdir = "nomic_embed_text_v2_moe"
    command = (
        "pytest models/experimental/nomic_embed_text_v2_moe/tests/perf/test_nomic_perf.py" "::test_device_perf_target"
    )
    columns = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    key = "AVG DEVICE KERNEL SAMPLES/S"

    # batch rather than batch * seqlen, so the figure is sequences per second and comparable
    # with the encoder demos in models/demos.
    results = run_device_perf(command, subdir, 1, columns, batch, has_signposts=True)
    logger.info(f"B={batch} S={seqlen} T={batch * seqlen}: {results[key]:.1f} sequences/s")

    if EXPECTED_SEQUENCES_PER_S is None:
        pytest.skip(f"no baseline recorded yet; measured {results[key]:.1f} sequences/s, set EXPECTED_SEQUENCES_PER_S")

    expected = check_device_perf(results, MARGIN, {key: EXPECTED_SEQUENCES_PER_S}, assert_on_fail=True)
    prep_device_perf_report(
        model_name=f"nomic_embed_text_v2_moe_b{batch}_s{seqlen}",
        batch_size=batch,
        post_processed_results=results,
        expected_results=expected,
        comments="baseline",
    )
