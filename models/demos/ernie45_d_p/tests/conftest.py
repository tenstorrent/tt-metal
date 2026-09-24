# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import functools
import os

import pytest
import torch

from models.demos.ernie45_d_p.bringup import metrics
from models.demos.ernie45_d_p.reference.ernie_ref import ErnieConfig, WeightLoader, load_layer, pcc, resolve_model_path
from models.demos.ernie45_d_p.tt.common import DEVICE_PARAMS, Golden

MESH_1X4 = [pytest.param((1, 4), id="1x4")]


def mesh_1x4(fn):
    """Parametrize a test with the 1x4 Blackhole mesh + ring fabric."""
    fn = pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)(fn)
    return pytest.mark.parametrize("mesh_device", MESH_1X4, indirect=True)(fn)


@functools.lru_cache(maxsize=1)
def _loader():
    path = resolve_model_path()
    return WeightLoader(path), ErnieConfig.from_json(os.path.join(path, "config.json"))


@pytest.fixture(scope="session")
def cfg():
    return _loader()[1]


@pytest.fixture(scope="session")
def loader():
    return _loader()[0]


@pytest.fixture(scope="session")
def layer_weights():
    @functools.lru_cache(maxsize=2)
    def get(i):
        ld, c = _loader()
        return load_layer(ld, c, i, dtype=torch.float32)

    return get


@pytest.fixture(scope="session")
def golden_2k():
    return Golden(4096, 2048)


class PccRecorder:
    """Records a PCC metric for the active bring-up task and collects failures for one final assert."""

    def __init__(self, task):
        self.task = task
        self.fails = []

    def __call__(self, name, got, want, thr):
        p = pcc(got.float(), want.float())
        metrics.record(self.task, name, p)
        ok = p >= thr
        print(f"{'ok  ' if ok else 'FAIL'} {name}: pcc={p:.6f} (>= {thr})")
        if not ok:
            self.fails.append(f"{name}={p:.6f}<{thr}")
        return p

    def check(self):
        assert not self.fails, "PCC below threshold: " + ", ".join(self.fails)


@pytest.fixture
def record(request):
    default = getattr(request.module, "TASK", "adhoc")
    r = PccRecorder(os.environ.get("ERNIE_BRINGUP_TASK", default))
    yield r
