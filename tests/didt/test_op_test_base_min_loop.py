# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""OpTestBase.run_op_test iteration count under DIDT_MIN_LOOP_SECONDS. No device: ttnn and the op are faked."""

import time
from types import SimpleNamespace

import torch

import tt_py_test_utils_common.op_test_base as otb

ENV = "DIDT_MIN_LOOP_SECONDS"


class Out:
    def cpu(self):
        return [torch.zeros(2)]

    def deallocate(self, force):
        pass


class FakeOp(otb.OpTestBase):
    def __init__(self, loop_count, interval, op_seconds):
        self.device_ids = [16]
        self.activation = otb.OpParameter((2,), None, None, SimpleNamespace(buffer_type="DRAM"))
        self.arguments = []
        self.loop_count = loop_count
        self.determinism_check_enabled = True
        self.determinism_check_interval = interval
        self.op_seconds, self.ops, self.checks = op_seconds, 0, None

    def generate_tt_activations_from_torch(self, t):
        return t

    def convert_activations_to_memory_config(self, a):
        return a

    def run_device_operation(self):
        self.ops += 1
        time.sleep(self.op_seconds)
        return Out()

    def get_device(self, device_idx):
        return None

    def log_nd_core_summary(self, num_checks, nd_checks_per_device):
        self.checks = num_checks


FAKE_TTNN = SimpleNamespace(
    to_torch=lambda t: t,
    get_device_tensors=lambda ts: ts,
    device=SimpleNamespace(synchronize_device=lambda d: None),
    BufferType=SimpleNamespace(L1="L1"),
)
REFERENCE_RUNS = 10  # one reference run per activation tensor before the loop


def test_unset_runs_exactly_loop_count(monkeypatch):
    monkeypatch.setattr(otb, "ttnn", FAKE_TTNN)
    monkeypatch.delenv(ENV, raising=False)
    t = FakeOp(loop_count=7, interval=3, op_seconds=0)
    t.run_op_test()
    assert t.ops - REFERENCE_RUNS == 7 and t.checks == 3


def test_min_loop_seconds_extends_the_loop(monkeypatch):
    monkeypatch.setattr(otb, "ttnn", FAKE_TTNN)
    monkeypatch.setenv(ENV, "0.5")
    t = FakeOp(loop_count=2, interval=2, op_seconds=0.05)
    start = time.time()
    t.run_op_test()
    iterations = t.ops - REFERENCE_RUNS
    assert iterations >= 9 and time.time() - start >= 0.5
    # checks keep the same interval over the longer loop
    assert t.checks == -(-iterations // 2)


def test_min_loop_seconds_never_shortens_the_loop(monkeypatch):
    monkeypatch.setattr(otb, "ttnn", FAKE_TTNN)
    monkeypatch.setenv(ENV, "0.01")
    t = FakeOp(loop_count=5, interval=1, op_seconds=0.01)
    t.run_op_test()
    assert t.ops - REFERENCE_RUNS == 5 and t.checks == 5
