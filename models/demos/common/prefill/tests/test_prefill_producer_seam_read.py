# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-aware KV reads and row masking of the producer's validation on a stage that spans two hosts."""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.common.prefill.runners import prefill_producer
from models.demos.common.prefill.runners.stage_layout import host_name_of_tag

LOCAL = host_name_of_tag(1)
REMOTE = host_name_of_tag(2)


class _FakeTable:
    """Chunk ``pos`` lives on device group ``pos``; each group is one fabric node on ``hosts[group]``."""

    def __init__(self, hosts):
        self.hosts = hosts

    def lookup(self, layer, pos, slot_id, config_id):
        return SimpleNamespace(device_group_index=pos, noc_addr=0x100 * pos, size_bytes=4)

    def get_device_group(self, index):
        return SimpleNamespace(fabric_node_ids=[SimpleNamespace(mesh_id=0, chip_id=index)])

    def get_host(self, node):
        return self.hosts[int(node.chip_id)]


@pytest.fixture
def reads(monkeypatch):
    seen = []

    def read(unique_ids, addrs, sizes):
        seen.append(list(unique_ids))
        return b"".join(bytes([uid]) * size for uid, size in zip(unique_ids, sizes))

    monkeypatch.setattr(ttnn.experimental.disaggregation, "read_dram_umd_batch", read)
    return seen


def test_read_chunks_reads_local_chips_and_skips_the_other_hosts(reads):
    table = _FakeTable([LOCAL, REMOTE, LOCAL])
    device_map = {(0, 0): 10, (0, 1): 11, (0, 2): 12}
    chunks = prefill_producer._read_chunks(table, device_map, 0, 0, 0, [0, 1, 2], local_host=LOCAL)
    assert [None if c is None else bytes(c) for c in chunks] == [bytes([10]) * 4, None, bytes([12]) * 4]
    assert reads == [[10, 12]]


def test_read_chunks_fails_on_a_local_chip_missing_from_the_device_map(reads, expect_error):
    table = _FakeTable([LOCAL, LOCAL])
    with expect_error(KeyError, "no fabric node"):
        prefill_producer._read_chunks(table, {(0, 0): 10}, 0, 0, 0, [0, 1], local_host=LOCAL)


def test_read_chunks_propagates_a_failing_local_read(monkeypatch, expect_error):
    def read(unique_ids, addrs, sizes):
        raise RuntimeError("UMD read failed")

    monkeypatch.setattr(ttnn.experimental.disaggregation, "read_dram_umd_batch", read)
    with expect_error(RuntimeError, "UMD read failed"):
        prefill_producer._read_chunks(_FakeTable([LOCAL]), {(0, 0): 10}, 0, 0, 0, [0], local_host=LOCAL)


def test_visible_rows_drop_the_other_hosts_rows(monkeypatch):
    monkeypatch.setattr(prefill_producer, "_MULTI_HOST_STAGES", True)
    rows = torch.tensor([[1.0, 2.0], [float("nan")] * 2, [3.0, 4.0]])
    golden = torch.arange(6.0).reshape(3, 2)
    kept, kept_golden = prefill_producer._visible_rows(rows, golden, "test")
    assert kept.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert kept_golden.tolist() == [[0.0, 1.0], [4.0, 5.0]]


def test_visible_rows_raise_when_this_host_scored_nothing(monkeypatch, expect_error):
    monkeypatch.setattr(prefill_producer, "_MULTI_HOST_STAGES", True)
    with expect_error(RuntimeError, "nothing to score"):
        prefill_producer._visible_rows(torch.full((2, 2), float("nan")), torch.zeros(2, 2), "test")


def test_visible_rows_keep_device_nans_on_a_single_host_stage(monkeypatch):
    monkeypatch.setattr(prefill_producer, "_MULTI_HOST_STAGES", False)
    rows = torch.tensor([[1.0, 2.0], [float("nan")] * 2])
    kept, _ = prefill_producer._visible_rows(rows, torch.zeros(2, 2), "test")
    assert not torch.isfinite(kept).all()
