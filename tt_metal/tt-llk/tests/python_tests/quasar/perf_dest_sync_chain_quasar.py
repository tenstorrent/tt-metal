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
    LOOP_FACTOR,
    NUM_FACES,
    SYNC_CHAIN,
    TEST_FACE_DIMS,
    TILE_COUNT,
)

CHAINS = [
    "FPU,PACK",
    "UNPACK,PACK",
    "FPU,SFPU,PACK",
    "UNPACK,SFPU,PACK",
    "FPU,SFPU,PACK,FPU,SFPU,PACK",
    "UNPACK,SFPU,PACK,UNPACK,SFPU,PACK",
    # Mixed producers need a drain before the per-iteration dvalid chain
    # reconfiguration; re-enable once the kernel adds one.
    # "FPU,SFPU,PACK,UNPACK,SFPU,PACK",
]

PRODUCER_OPS = (SyncChainOp.FPU, SyncChainOp.UNPACK)

INPUT_DIMENSIONS = [32, 32]
NUM_FACES_PER_TILE = 4


def parse_chain(chain: str) -> tuple:
    ops = tuple(SyncChainOp[name] for name in chain.split(","))
    expect_producer = True
    for op in ops:
        if expect_producer:
            assert (
                op in PRODUCER_OPS
            ), f"L1 iteration must start with FPU or UNPACK: {chain}"
            expect_producer = False
        elif op is SyncChainOp.PACK:
            expect_producer = True
        else:
            assert (
                op is SyncChainOp.SFPU
            ), f"Only SFPU ops allowed between producer and PACK: {chain}"
    assert expect_producer, f"Every L1 iteration must end with PACK: {chain}"
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
            "runtimes": [
                TILE_COUNT(tile_cnt_A),
                NUM_FACES(NUM_FACES_PER_TILE),
                TEST_FACE_DIMS(),
                LOOP_FACTOR(1),
            ],
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
