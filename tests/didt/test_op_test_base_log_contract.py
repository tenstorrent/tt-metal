# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""op_test_base's Chip map, ND summary and iteration lines, written by the real logging code and read back by the
didt_runner parsers. If either side changes its wording, these fail. No device: ttnn, tt_umd and the op are faked.
The log uses loguru's default format, as a pytest run on a galaxy prints it; a run that configures another format
is not covered."""

import sys
from types import SimpleNamespace

import pytest
import torch
from loguru import logger
from loguru._defaults import LOGURU_FORMAT

import tt_py_test_utils_common.op_test_base as otb
from tests.didt.runner.checks import loop_seconds
from tests.didt.runner.chips import Chip, by_dev
from tests.didt.runner.devices import nd_report_devices
from tests.didt.test_op_test_base_min_loop import FAKE_TTNN, REFERENCE_RUNS, FakeOp
from tt_py_test_utils_common import nd_core_map

LOOP = 4
BLOCK = (32, 32)
GRID = (2, 2)
# device id -> tray, asic, /dev/tenstorrent/N, bdf, as the cluster descriptor gives them on a galaxy
CLUSTER = {
    3: (1, 4, 3, "0000:04:00.0"),
    16: (4, 1, 24, "0000:81:00.0"),
    19: (4, 4, 27, "0000:84:00.0"),
}
BOARD = 0x47330000000
# loop iteration -> (device, row, col) of each wrong output element; block (32, 32) so col 40 is core (1,0)
WRONG = {1: [(16, 5, 40)], 2: [(19, 40, 3), (19, 40, 40)], 3: [(16, 5, 40)]}
CHIPS = by_dev([Chip(d, bdf, tray, asic, f"0x{d:016x}", kmd) for d, (tray, asic, kmd, bdf) in CLUSTER.items()])


class ClusterDescriptor:
    @staticmethod
    def create_from_yaml_content(text):
        return ClusterDescriptor()

    def get_all_chips(self):
        return list(CLUSTER)

    def get_chips_with_mmio(self):
        return {d: c[2] for d, c in CLUSTER.items()}

    def get_chip_pci_bdfs(self):
        return {d: c[3] for d, c in CLUSTER.items()}

    def get_tray_id(self, chip):
        return CLUSTER[chip][0]

    def get_asic_location(self, chip):
        return CLUSTER[chip][1]

    def get_board_id_for_chip(self, chip):
        return BOARD


class Out:
    def __init__(self, iteration):
        self.tensors = [torch.zeros(64, 64) for _ in CLUSTER]
        for dev, row, col in WRONG.get(iteration, []):
            self.tensors[list(CLUSTER).index(dev)][row, col] = 1.0

    def cpu(self):
        return self.tensors

    def deallocate(self, force):
        pass


class NDOp(FakeOp):
    log_nd_core_summary = otb.OpTestBase.log_nd_core_summary

    def __init__(self):
        super().__init__(loop_count=LOOP, interval=1, op_seconds=0)
        self.device_ids = list(CLUSTER)
        self.nd_core_hits, self.nd_core_structures, self.nd_core_translated = {}, {}, {}
        self._chip_locations = None
        self.nd_core_block, self.nd_core_grid_size = BLOCK, GRID

    def run_device_operation(self):
        self.ops += 1
        return Out(self.ops - REFERENCE_RUNS - 1)

    def get_device(self, device_idx):
        return SimpleNamespace(worker_core_from_logical_core=lambda c: SimpleNamespace(x=c.x + 1, y=c.y + 2))


@pytest.fixture(scope="module")
def log_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("log")
    (tmp / "cluster.yaml").write_text("")
    fake_ttnn = SimpleNamespace(
        **vars(FAKE_TTNN),
        CoreCoord=lambda x, y: SimpleNamespace(x=x, y=y),
        cluster=SimpleNamespace(serialize_cluster_descriptor=lambda: str(tmp / "cluster.yaml")),
    )
    path = tmp / "log"
    sink = logger.add(path, format=LOGURU_FORMAT, colorize=False, level="DEBUG")
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(otb, "ttnn", fake_ttnn)
            mp.setitem(sys.modules, "tt_umd", SimpleNamespace(ClusterDescriptor=ClusterDescriptor))
            mp.delenv("DIDT_MIN_LOOP_SECONDS", raising=False)
            with pytest.raises(AssertionError):  # allow-pytest.raises: module fixture, root conftest not loaded
                NDOp().run_op_test()
    finally:
        logger.remove(sink)
    return path


def test_nd_lines_parse_with_runner(log_path):
    devices, warnings = nd_report_devices(log_path.read_text(), CHIPS)
    assert warnings == []
    d16, d19 = devices
    assert {k: d16[k] for k in ("dev", "bdf", "tray", "asic", "asic_id", "kmd")} == CHIPS[16].as_dict()
    assert (d16["failed_checks"], d16["checks"], d16["pattern"], d16["hung"]) == (2, LOOP, "single core (1,0)", False)
    (core,) = d16["cores"]
    assert (core["x"], core["y"], core["translated"], core["failed_checks"]) == (1, 0, [2, 2], 2)
    assert core["signature"] and core["signature"] == core["signature"].strip()
    assert d19["dev"] == 19 and (d19["failed_checks"], d19["checks"]) == (1, LOOP)
    assert d19["pattern"] == nd_core_map.classify({(0, 1), (1, 1)}, GRID)
    assert [(c["x"], c["y"], c["translated"]) for c in d19["cores"]] == [(0, 1, [1, 3]), (1, 1, [2, 3])]


def test_chip_map_bdf_reaches_runner(log_path):
    # No chip snapshot: the identity comes from the Chip map line alone.
    devices, _ = nd_report_devices(log_path.read_text(), {})
    assert [{k: d[k] for k in ("dev", "bdf", "tray", "asic")} for d in devices] == [
        {"dev": 16, "bdf": "0000:81:00.0", "tray": 4, "asic": 1},
        {"dev": 19, "bdf": "0000:84:00.0", "tray": 4, "asic": 4},
    ]


def test_iteration_lines_parse_with_runner(log_path):
    seconds, iterations = loop_seconds(log_path)
    assert iterations == LOOP and 0 <= seconds < 60
