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


def test_visible_rows_drop_the_other_hosts_rows():
    rows = torch.tensor([[1.0, 2.0], [float("nan")] * 2, [3.0, 4.0]])
    available = torch.tensor([True, False, True])
    golden = torch.arange(6.0).reshape(3, 2)
    kept, kept_golden = prefill_producer._visible_rows(rows, available, golden, "test")
    assert kept.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert kept_golden.tolist() == [[0.0, 1.0], [4.0, 5.0]]


def test_visible_rows_raise_when_this_host_scored_nothing(expect_error):
    with expect_error(RuntimeError, "nothing to score"):
        prefill_producer._visible_rows(
            torch.full((2, 2), float("nan")), torch.zeros(2, dtype=torch.bool), torch.zeros(2, 2), "test"
        )


def test_visible_rows_reject_a_nan_in_a_row_that_was_read(expect_error):
    rows = torch.tensor([[1.0, 2.0], [float("nan"), 1.0], [float("nan")] * 2])
    available = torch.tensor([True, True, False])
    with expect_error(RuntimeError, "1 of the 2 rows read from the device are NaN"):
        prefill_producer._visible_rows(rows, available, torch.zeros(3, 2), "test")


def test_read_kv_slice_masked_marks_the_other_hosts_rows_unavailable(monkeypatch):
    from models.demos.minimax_m3.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK as BANK

    monkeypatch.setattr(prefill_producer, "_local_host_name", lambda multi_host: LOCAL)
    monkeypatch.setattr(prefill_producer, "_read_chunks", lambda *args, **kwargs: [b"a", None, b"c"])
    decode = lambda raw, head_dim: torch.full((BANK, head_dim), float("nan") if raw == b"c" else 1.0)
    rows, available = prefill_producer._read_kv_slice_masked(None, None, 0, 0, 0, 3 * BANK, 2, decode)
    assert rows.shape == (3 * BANK, 2)
    assert available.tolist() == [True] * BANK + [False] * BANK + [True] * BANK
    # A NaN decoded from a chunk that was read stays available, so the mask check rejects it.
    assert torch.isnan(rows[available]).any()
