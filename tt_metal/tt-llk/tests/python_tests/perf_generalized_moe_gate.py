# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_quasar
from helpers.llk_params import ApproximationMode, DestSync, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.test_variant_parameters import GENERALIZED_MOE_GATE
from test_generalized_moe_gate import (
    EPS,
    FORMATS,
    MODE_GATE,
    SCALE,
    _bits,
    _config,
    _gate_stimuli,
    _gate_tiles,
    _ids_face,
)

pytestmark = skip_for_quasar


@pytest.mark.perf
@parametrize(
    dest_sync=[DestSync.Half, DestSync.Full],
    approx=[ApproximationMode.No, ApproximationMode.Yes],
    softmax=[False, True],
)
def test_perf_generalized_moe_gate(perf_report, dest_sync, approx, softmax):
    payload, bias, _keys = _gate_stimuli(seed=8)
    ids = _ids_face()
    test_cfg = _config(
        GENERALIZED_MOE_GATE(
            mode=MODE_GATE,
            topk=8,
            softmax=softmax,
            eps=_bits(EPS),
            scale=_bits(SCALE),
        ),
        _gate_tiles(payload, ids),
        src_b=bias,
        dest_sync=dest_sync,
        approx=approx,
    )
    configuration = PerfConfig(
        test_cfg.test_name,
        FORMATS,
        run_types=[PerfRunType.L1_TO_L1],
        templates=list(test_cfg.templates),
        runtimes=list(test_cfg.runtimes),
        variant_stimuli=test_cfg.variant_stimuli,
        dest_acc=test_cfg.dest_acc,
        unpack_to_dest=test_cfg.unpack_to_dest,
    )
    configuration.run(perf_report)
