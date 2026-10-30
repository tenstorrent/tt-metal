# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Traced op time of sparse_ring_sdpa with forced Q pass counts, to test the pacing model. Dev branch only.

Model (md_files/sparse_ring_sdpa_review_2026-10-30.md): the first pass is paced by the all-gather when a pass's
compute is shorter than the gather, and every later pass runs after it with no overlap, so each extra pass adds about
one pass of compute. Per shape: the dense op and the gather alone (the perf test's baselines), then the sparse op at
the factory's pass count (q_blocks_per_pass None) and at forced smaller passes. The factory logs the split it chose.

    pytest dev/passes_case.py -s
"""

import statistics

import pytest
import ttnn
from loguru import logger

from tests.nightly.blackhole.sdpa.test_sparse_ring_sdpa import BLOCK, make_mask
from tests.nightly.blackhole.sdpa.test_sparse_ring_sdpa_perf import (  # noqa: F401 (runtime is a fixture)
    WORKING_POINTS,
    Bench,
    runtime,
    traced_ms,
)

# rows -> forced q_blocks_per_pass values (None = the factory's choice): 5 s 10 blocks/core (1 pass) -> 5 (2 passes);
# 10 s 19 (2 passes of 10) -> 7 (3 passes); 15 s 28 (2 passes of 14) -> 10 (3 passes).
FORCED = {4736: [None, 5], 9184: [None, 7], 13664: [None, 10]}


@pytest.fixture(scope="module")
def bench(runtime):
    b = Bench(runtime)
    yield b
    b._free()


@pytest.mark.parametrize("rows", list(FORCED), ids=[WORKING_POINTS[r] for r in FORCED])
def test_passes(bench, rows):
    dense, gather = bench.baseline(rows)
    case = bench.shape(rows)
    mask = make_mask(case.geo, case.S, case.H, density=0.2, seed=rows)
    t_mask = case._upload(mask, case.dims(0, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    del mask
    rt = case.rt
    q, k, v = bench.tensors
    pk, pv = bench.pooled
    config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=rt.sdpa_compute_grid, q_chunk_size=BLOCK, k_chunk_size=256, exp_approx_mode=False
    )
    tag = WORKING_POINTS[rows]
    logger.info(
        f"[passes {tag}] dense {statistics.median(dense):.3f} ms, gather {statistics.median(gather):.3f} ms "
        f"(all dense {' '.join(f'{t:.3f}' for t in dense)})"
    )
    try:
        for qbpp in FORCED[rows]:
            # The pass split is not in the program hash: a cached program keeps the split of the call that built it.
            rt.mesh_device.clear_program_cache()
            fn = lambda: ttnn.transformer.sparse_ring_sdpa(
                q,
                k,
                v,
                t_mask,
                pk,
                pv,
                persistent_output_buffer_k=case.buf_k,
                persistent_output_buffer_v=case.buf_v,
                logical_n=case.geo.logical_n,
                program_config=config,
                compute_kernel_config=bench.ckc,
                dim=2,
                multi_device_global_semaphore=rt.ccl_semaphore_handles,
                num_links=rt.num_links,
                cluster_axis=rt.sp_axis,
                mesh_device=rt.mesh_device,
                topology=rt.topology,
                subdevice_id=rt.worker_sub_device_id,
                ccl_core_grid_offset=(rt.ccl_column, 0),
                use_column_major_ccl=True,
                block_size=BLOCK,
                q_blocks_per_pass=qbpp,
            )
            times = traced_ms(rt.mesh_device, fn)
            logger.info(
                f"[passes {tag}] q_blocks_per_pass {qbpp}: {statistics.median(times):.3f} ms "
                f"(all {' '.join(f'{t:.3f}' for t in times)})"
            )
    finally:
        ttnn.deallocate(t_mask)
        rt.mesh_device.clear_program_cache()
