# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity, PerfRunType
from helpers.param_config import DEST_SYNC_TILE_LIMITS, parametrize
from helpers.perf.core import PerfConfig
from test_hadamard import FORMATS, FP32_FORMATS, _config, _signs

pytestmark = blackhole_only


@pytest.mark.perf
@parametrize(
    dest_sync=[DestSync.Half, DestSync.Full],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    normalize=[False, True],
)
def test_perf_hadamard_h128(perf_report, dest_sync, dest_acc, normalize):
    if dest_acc == DestAccumulation.Yes and normalize:
        pytest.skip("normalize is unsupported with dest accumulation")

    capacity = DEST_SYNC_TILE_LIMITS[dest_sync] // (
        2 if dest_acc == DestAccumulation.Yes else 1
    )
    formats = FP32_FORMATS if dest_acc == DestAccumulation.Yes else FORMATS
    inputs = _signs(seed=1, count=capacity)
    test_cfg = _config(
        inputs,
        normalize=normalize,
        fidelity=MathFidelity.HiFi4,
        dest_sync=dest_sync,
        dest_acc=dest_acc,
        formats=formats,
    )
    configuration = PerfConfig(
        test_cfg.test_name,
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=list(test_cfg.templates),
        runtimes=list(test_cfg.runtimes),
        variant_stimuli=test_cfg.variant_stimuli,
        dest_acc=test_cfg.dest_acc,
        unpack_to_dest=test_cfg.unpack_to_dest,
    )
    configuration.run(perf_report)
