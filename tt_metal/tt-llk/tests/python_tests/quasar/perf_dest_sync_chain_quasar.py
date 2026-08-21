# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    DestSyncScheme,
    PerfRunType,
    SyncChainOp,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    DEST_SYNC,
    DEST_SYNC_SCHEME,
    SYNC_CHAIN,
)

CHAINS = [
    "FPU,PACK",
    "UNPACK,PACK",
    "SFPU,PACK",
    "FPU,SFPU,PACK",
    "SFPU,FPU,PACK",
    "UNPACK,SFPU,PACK",
    "UNPACK,FPU,PACK",
    "FPU,SFPU,PACK,FPU,SFPU,PACK",
    "UNPACK,SFPU,PACK,UNPACK,SFPU,PACK",
    "FPU,SFPU,PACK,UNPACK,SFPU,PACK",
    "FPU,SFPU,PACK,SFPU,FPU,PACK",
]

COMPUTE_OPS = (SyncChainOp.UNPACK, SyncChainOp.FPU, SyncChainOp.SFPU)

INPUT_DIMENSIONS = [32, 32]
NUM_FACES_PER_TILE = 4


def parse_chain(chain: str) -> tuple:
    ops = tuple(SyncChainOp[name] for name in chain.split(","))
    iterations = []
    current = []
    for op in ops:
        if op is SyncChainOp.PACK:
            assert current, f"L1 iteration must have at least one compute op: {chain}"
            iterations.append(tuple(current))
            current = []
        else:
            assert op in COMPUTE_OPS, f"Unknown compute op in chain: {chain}"
            current.append(op)
    assert not current, f"Every L1 iteration must end with PACK: {chain}"

    for iteration in iterations:
        assert len(set(iteration)) == len(
            iteration
        ), f"Each unit may appear at most once per L1 iteration: {chain}"
        assert (
            SyncChainOp.UNPACK not in iteration[1:]
        ), f"UNPACK must be the first op of its L1 iteration: {chain}"

    if len(set(iterations)) > 1:
        assert all(
            len(iteration) <= 2 for iteration in iterations
        ), f"Non-uniform chains support at most two compute ops per iteration: {chain}"

    return ops


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    chain=CHAINS,
    scheme=[DestSyncScheme.Semaphore, DestSyncScheme.Dvalid],
    dest_sync=[DestSync.Half, DestSync.Full],
)
def test_perf_dest_sync_chain_quasar(perf_report, chain, scheme, dest_sync):
    ops = parse_chain(chain)
    formats = input_output_formats([DataFormat.Float16_b], same=True)[0]

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=INPUT_DIMENSIONS,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=INPUT_DIMENSIONS,
    )

    configuration = create_test_or_perf_config(
        is_perf=True,
        run_types=[PerfRunType.L1_TO_L1],
        test_config_kwargs={
            "test_name": "sources/quasar/dest_sync_chain_quasar_test.cpp",
            "formats": formats,
            "templates": [
                SYNC_CHAIN(ops),
                DEST_SYNC_SCHEME(scheme),
                DEST_SYNC(dest_sync),
            ],
            "runtimes": [],
            "variant_stimuli": StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=tile_cnt_A,
                tile_count_B=tile_cnt_A,
                tile_count_res=tile_cnt_A,
                num_faces=NUM_FACES_PER_TILE,
            ),
            "unpack_to_dest": False,
            "dest_acc": DestAccumulation.No,
        },
    )
    configuration.run(perf_report)
