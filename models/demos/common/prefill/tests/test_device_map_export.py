# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Device-free checks of the file-export device map: ranks sharing a host (one file per node) each
write a sidecar, and the merged per-host file holds every rank's chips."""

from models.demos.common.prefill.runners.migration import (
    device_map_rank_sidecar_path,
    merge_device_map_rank_sidecars,
    write_device_map_rank_sidecar,
)


def _tray(mesh_id, first_umd):
    return [(mesh_id, chip, first_umd + chip) for chip in range(8)]


def test_four_ranks_on_one_host_merge(tmp_path):
    path = str(tmp_path / "device_map.node-a.txt")
    for rank in range(4, 8):
        write_device_map_rank_sidecar(_tray(rank, rank * 8), path, rank)
    for rank in range(4, 8):
        assert merge_device_map_rank_sidecars(path, range(4, 8), rank) == 32

    lines = (tmp_path / "device_map.node-a.txt").read_text().splitlines()
    assert lines == [f"{m} {c} {u}" for m in range(4, 8) for (_, c, u) in _tray(m, m * 8)]


def test_stale_merged_map_removed_and_foreign_sidecar_ignored(tmp_path):
    path = str(tmp_path / "device_map.node-a.txt")
    (tmp_path / "device_map.node-a.txt").write_text("9 0 999\n")
    (tmp_path / "device_map.node-a.txt.r3").write_text("3 0 333\n")  # not a rank on this host now

    write_device_map_rank_sidecar(_tray(0, 0), path, 0)
    assert not (tmp_path / "device_map.node-a.txt").exists()

    assert merge_device_map_rank_sidecars(path, [0], 0) == 8
    assert "333" not in (tmp_path / "device_map.node-a.txt").read_text()


def test_conflicting_sidecars_rejected(tmp_path, expect_error):
    path = str(tmp_path / "map.txt")
    write_device_map_rank_sidecar([(0, 0, 1)], path, 0)
    write_device_map_rank_sidecar([(0, 0, 2)], path, 1)
    with expect_error(RuntimeError, "disagree"):
        merge_device_map_rank_sidecars(path, [0, 1], 0)


def test_sidecar_path():
    assert device_map_rank_sidecar_path("/kv-handoff/device_map.n.txt", 5) == "/kv-handoff/device_map.n.txt.r5"
