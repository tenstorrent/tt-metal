# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device tests of the multi-user pack on shared page pools: the flat pack page-table row and the landing without the
K/V and compressed families (fakes)."""

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import pack_landing
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging

PAGING = Qwen38LanePaging(128, 64, 264)


def test_pack_page_table_puts_each_user_on_its_slab_pages():
    host = model_module.pack_page_table_host(PAGING, [[5, 6], [7], [], [9, 10]], 256)
    assert host.dtype == torch.int32 and tuple(host.shape) == (1, 1, 32, 264)
    assert all(torch.equal(host[0, 0, 0], host[0, 0, r]) for r in range(32))
    row = host[0, 0, 0].tolist()
    assert row[:8] == [5, 6, 7, 0, 0, 0, 9, 10]  # users own pages [2u, 2u + 2)
    assert row[8:] == [0] * (264 - 8)


def test_pack_page_table_with_one_block_per_row_of_the_bucket():
    host = model_module.pack_page_table_host(PAGING, [[3], [4]], 128)
    assert host[0, 0, 0, :4].tolist() == [3, 4, 0, 0]


@pytest.mark.parametrize(
    "paging, users, rows, message",
    [
        (None, [[1]], 128, "not paged"),
        (PAGING, [[1]], 64, "multiple of the block size"),
        (Qwen38LanePaging(256, 64, 264), [[1]], 128, "multiple of the block size"),
        (PAGING, [[1, 2, 3]], 256, "pages per user"),
        (PAGING, [[0]], 128, "block ids"),
        (PAGING, [[64]], 128, "block ids"),
        (Qwen38LanePaging(128, 64, 8), [[1]], 128, "page table holds"),
    ],
)
def test_pack_page_table_validation(paging, users, rows, message, expect_error):
    with expect_error(ValueError, message):
        model_module.pack_page_table_host(paging, users, rows)


def test_pack_page_table_refuses_more_users_than_the_bucket_holds(expect_error):
    with expect_error(ValueError, "users"):
        model_module.pack_page_table_host(PAGING, [[1]] * 3, 1024)


def test_write_pack_page_table_uploads_in_place(monkeypatch):
    seen = {}
    monkeypatch.setattr(model_module.ttnn, "from_torch", lambda host, **kw: ("host", host), raising=False)
    monkeypatch.setattr(
        model_module.ttnn,
        "copy_host_to_device_tensor",
        lambda host, dev: seen.update(host=host, dev=dev),
        raising=False,
    )
    monkeypatch.setattr(model_module, "replicate_tensor_2d_mesh_mapper", lambda mesh: None)
    fake = type("M", (), {"mesh_device": None, "_validate_generic_state": lambda self, state: None})()
    state = type("S", (), {"paging": PAGING, "page_table": "TABLE"})()
    model_module.Qwen38TTNNTextModel.write_pack_page_table(fake, state, [[5], [6]], 128)
    assert seen["dev"] == "TABLE" and seen["host"][1][0, 0, 3, :3].tolist() == [5, 6, 0]


class _Layout:
    def __init__(self, compressed):
        self.lanes = 8
        self.recurrent = ()
        self.conv = ()
        self.compressed = compressed


def test_the_lander_skips_kv_and_compressed_with_shared_pools(monkeypatch):
    calls = []
    monkeypatch.setattr(pack_landing, "kv_page_scatter", lambda *a, **k: calls.append("kv"))
    monkeypatch.setattr(pack_landing.ttnn, "slice", lambda *a, **k: calls.append("slice") or "t", raising=False)
    monkeypatch.setattr(pack_landing.ttnn, "matmul", lambda *a, **k: calls.append("matmul") or "t", raising=False)
    monkeypatch.setattr(pack_landing.ttnn, "concat", lambda *a, **k: calls.append("concat") or "t", raising=False)
    monkeypatch.setattr(
        pack_landing.ttnn, "fill_cache", lambda view, *a, **k: calls.append("fill") or view, raising=False
    )
    monkeypatch.setattr(pack_landing, "_landed", lambda *a, **k: None)
    monkeypatch.setattr(pack_landing, "_deallocate", lambda *a, **k: None)
    monkeypatch.setattr(pack_landing.pack_module, "gdn_state_slot", lambda bucket, user: (0, 0))
    for name in ("staging_select_image", "raw_ring_select_image", "ple_history_select_image"):
        monkeypatch.setattr(pack_landing.pack_module, name, lambda *a, **k: None)
    attention = type(
        "A", (), {"packed_kv_cache": "kv", "compressed_index_cache": "c", "kept_kv": "k", "kept_raw": "r"}
    )()
    layer = type("L", (), {"attention": attention})()
    lander = object.__new__(pack_landing.Qwen38TTNNPackLander)
    lander.shared_pools = True
    lander.pager = type("P", (), {"views": {"staging": ["s"], "ring": ["r"], "ple": []}})()
    lander.gdn_layers = []
    lander.qsa_layers = [0]
    lander.chain_state = type("C", (), {"layers": [layer]})()
    lander.pack_state = type(
        "K", (), {"chunk_state": type("CS", (), {"layers": [layer]})(), "ple_snapshots": {0: "snap"}}
    )()
    lander.lane_state = type("LS", (), {"layers": [layer], "page_table": "T"})()
    lander._write = lambda *a: None
    lander.staging_select = lander.raw_select = lander.ple_select = None
    lander.compute_config = None
    monkeypatch.setattr(pack_landing.ttnn, "permute", lambda *a, **k: "t", raising=False)
    bucket = type("B", (), {"rows_per_user": 128})()
    lander._land_user(0, 0, 5, bucket)
    assert "kv" not in calls and "concat" not in calls and "slice" not in calls
    assert calls.count("fill") == 2 and calls.count("matmul") == 3  # staging and ring selects, the PLE window
