# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reproducer for the intermittent down-phase deadlock in unified_routed_expert_ffn.

OPT-IN ONLY. This test is expected to WEDGE THE DEVICE when it succeeds, so it is skipped
unless RE_DEADLOCK_REPRO is set. Never enable it in CI.

    RE_DEADLOCK_REPRO=1 REPRO_ROUNDS=40 pytest <this file> -q

What triggers it (all three seem to be needed -- 40 bare back-to-back iterations of the
hanging cell never reproduced it):
  * the realtime profiler active,
  * several ISLs swept inside ONE process, and
  * minimax_m3 6144x3072 / SwiGluOai / DRAM-interleaved weights.

Rate is roughly 0.4% per cell (3 hangs observed in ~800 cells), and every hang so far has
landed on the largest ISLs -- 5120 (5 chunks) and 4096 (4 chunks) -- never below 2048.

At the hang all 88 worker cores are parked: the in1 senders on the down-phase receiver
credit wait, the receivers on in1_valid/act_valid, and compute on its in0/in1 CB waits, so
the credits were sent and then lost. Splitting in1_ready into separate gate/up and down
semaphores was tried and did NOT help (identical stack, unchanged rate), which rules out
the cross-phase credit theft that reader.cpp's own comment warns about. in1_valid is still
shared between the two phases and is ordered only WITHIN a chunk by the mcast_done barrier,
which is the next thing to look at.

Triage from a live hang: ~/.tt-buddy/triage/2026-09-12T152500-postsplit/
"""

import os

import pytest
import ttnn
from loguru import logger

from models.common.utility_functions import is_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_p150
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler
from tests.ttnn.nightly.unit_tests.operations.experimental.deepseek_prefill.test_single_routed_expert import (
    run_single_routed_expert,
)

_RE_DIR = "/unified_routed_expert_ffn/"
_ISL = [0, 64, 128, 256, 512, 1024, 2048, 4096, 5120]
_ROUNDS = int(os.environ.get("REPRO_ROUNDS", "40"))
_ALLOC, _EMB, _HID = 5120, 6144, 3072


@pytest.mark.skipif(
    not os.environ.get("RE_DEADLOCK_REPRO"),
    reason="hangs the device on success; opt in with RE_DEADLOCK_REPRO=1",
)
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="the composite FFN path is Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="observed on P150")
@pytest.mark.timeout(0)
@skip_with_llk_assert("the hang is a race; LLK asserts perturb the timing that produces it")
@skip_with_watcher("watcher perturbs the timing that produces the hang")
def test_re_down_deadlock_repro(device):
    require_realtime_profiler("down-phase deadlock reproducer")
    for r in range(_ROUNDS):
        for active in _ISL:
            logger.info(f"REPRO round {r + 1}/{_ROUNDS} isl-{active}")

            def run():
                for _ in range(3):
                    run_single_routed_expert(
                        device,
                        _ALLOC,
                        _EMB,
                        _HID,
                        active_tokens=active,
                        x_row_major=True,
                        weights_dram_sharded=False,
                        activation=ttnn.RoutedExpertActivation.SwiGluOai,
                    )

            _, per_program = profile_realtime_program_merged(device, run)
            matched = [
                e["duration_ns"]
                for e in per_program.values()
                if any(_RE_DIR in s.replace("\\", "/") for s in e["kernel_sources"])
            ]
            logger.info(f"REPRO   ok isl-{active}: {min(matched) / 1000:.1f} us")
    logger.info(f"REPRO survived {_ROUNDS * len(_ISL)} cells")
