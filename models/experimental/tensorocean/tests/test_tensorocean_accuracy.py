# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Accuracy against the float64 reference (LANL's algorithm), random inputs from the reference's make_inputs."""
import pytest
from loguru import logger

from models.experimental.tensorocean.tests.common import PCC_MIN, RMS_REL_MAX, VERSIONS, check


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("version", ["optimized", "baseline"])
@pytest.mark.parametrize("n, levels", [(8, 3), (30, 5), (100, 100)])
@pytest.mark.parametrize("seed", [0, 1])
def test_tensorocean_accuracy(device, version, n, levels, seed):
    if version == "baseline" and n == 100 and seed == 1:
        pytest.skip("baseline at 100 x 100 is slow; seed 0 covers it")
    m, _ = check(VERSIONS[version], device, n, levels, seed)
    for part, r in zip(("even", "odd"), m):
        logger.info(f"{version} N={n} L={levels} seed={seed} {part}: pcc {r['pcc']:.12f} rms_rel {r['rms_rel']:.2e}")
        assert r["finite"]
        assert r["pcc"] >= PCC_MIN
        assert r["rms_rel"] <= RMS_REL_MAX
