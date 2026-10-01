# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Repro for PR #58242: rotating sender + Counter + handshake=False loses a data_ready increment.

handshake=False is the suspected-broken configuration and is expected to report lost increments.
handshake=True is the control: the same kernel and topology must never fail.
"""

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.kernel_lib.mcast_test_utils import KERNEL_DIR, make_cb

ST_WORDS = 16
MAGIC = 0xC0FFEE00
(ST_MAGIC, ST_CORE, ST_FAILED, ST_FAIL_ROUND, ST_OBSERVED, ST_EXPECTED, ST_FINAL, ST_SENT, ST_DONE) = range(9)


def _run_once(device, *, n_cores, handshake, rounds, max_delay, timeout_polls, noc):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_cores - 1, 0))])
    mc = ttnn.Mcast(
        device,
        ttnn.McastConfig(
            noc=ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0,
            handshake=handshake,
            data_ready=ttnn.McastDataReady.Counter,
        ),
        grid,
        grid.num_cores(),
        # Every core is a sender, in line order: a rotating channel with span == n_cores.
        ttnn.McastExplicitSenderConfig([[ttnn.CoreCoord(x, 0) for x in range(n_cores)]]),
    )

    dummy = ttnn.from_torch(
        torch.zeros([1, 1, 1, ST_WORDS], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, n_cores, ST_WORDS]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    cb = 0
    cbs = [make_cb(cb, grid, pages=1, page_bytes=ST_WORDS * 4, dtype=ttnn.uint32)]
    ct = [cb, rounds, max_delay, timeout_polls]
    ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    for x in range(n_cores):
        rt[x][0] = [out.buffer_address(), x]
    k = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/repro_rotating_counter_race.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0
        ),
    )
    pd = ttnn.ProgramDescriptor(cbs=cbs)
    mc.attach(pd, "mcast", [k], 0)
    pd.kernels = [k]
    result = ttnn.generic_op([dummy, out], pd)
    status = ttnn.to_torch(result).reshape(n_cores, ST_WORDS).to(torch.int64) & 0xFFFFFFFF
    for row in status:
        assert int(row[ST_MAGIC]) == MAGIC, f"core status not written: {row.tolist()}"
    return status


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("max_delay", [0, 32, 128, 512])
@pytest.mark.parametrize("n_cores", [2, 3, 4, 8])
@pytest.mark.parametrize("handshake", [False, True], ids=["no_handshake", "handshake_control"])
def test_rotating_counter_lost_increment(device, handshake, n_cores, max_delay, noc):
    size = device.compute_with_storage_grid_size()
    if n_cores > size.x:
        pytest.skip("requires a wider worker grid")
    rounds = 200_000
    timeout_polls = 50_000_000
    launches = 5

    failures = []
    for launch in range(launches):
        status = _run_once(
            device,
            n_cores=n_cores,
            handshake=handshake,
            rounds=rounds,
            max_delay=max_delay,
            timeout_polls=timeout_polls,
            noc=noc,
        )
        failed = status[status[:, ST_FAILED] != 0]
        if len(failed):
            # The core that failed in the earliest round lost the increment. The others then
            # time out waiting for it to send.
            first = failed[torch.argmin(failed[:, ST_FAIL_ROUND])]
            failures.append((launch, first.tolist(), failed.shape[0]))
            logger.warning(
                f"launch {launch}: core {int(first[ST_CORE])} round {int(first[ST_FAIL_ROUND])}: counter "
                f"{int(first[ST_OBSERVED])} < expected {int(first[ST_EXPECTED])} "
                f"(final {int(first[ST_FINAL])}, sent {int(first[ST_SENT])}); {failed.shape[0]}/{n_cores} cores timed out"
            )
        else:
            for row in status:
                assert int(row[ST_DONE]) == rounds
                assert int(row[ST_FINAL]) == rounds, f"core {int(row[ST_CORE])} final counter {int(row[ST_FINAL])}"

    logger.info(
        f"handshake={handshake} n_cores={n_cores} max_delay={max_delay} noc={noc}: "
        f"{len(failures)}/{launches} launches lost an increment"
    )
    assert not failures, f"lost data_ready increment(s): {failures}"
