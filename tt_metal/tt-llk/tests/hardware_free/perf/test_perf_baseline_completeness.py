# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the partial-baseline guard (lives in tt_metal/tt-llk/perf)."""

import json
import pathlib
import sys

_PERF = pathlib.Path(__file__).parents[3] / "perf"
sys.path.insert(0, str(_PERF))
from baseline_completeness import (
    expected_shards,
    found_shards,
    main,
    short_arches,
)

_MATRIX = [
    {"name": f"llk_perf_merge_gate_{arch} group {n}/3", "split_group": n}
    for arch in ("wormhole", "blackhole")
    for n in (1, 2, 3)
]


def _run(tmp_path, arrived, pipeline="merge_baseline"):
    """Lay out one run's artifacts and merged files, then apply the guard."""
    in_dir = tmp_path / "perf-data"
    out_dir = tmp_path / "generated"
    in_dir.mkdir()
    out_dir.mkdir()
    for arch, splits in arrived.items():
        for n in splits:
            (in_dir / f"perf-data-{arch}-{n}").mkdir()
        (out_dir / f"llk_perf_merge_baseline-20261102-4147-{arch}.parquet").touch()
    main(
        [
            "--matrix",
            json.dumps(_MATRIX),
            "--in-dir",
            str(in_dir),
            "--out-dir",
            str(out_dir),
            "--pipeline",
            pipeline,
        ]
    )
    return sorted(p.name for p in out_dir.iterdir())


def test_the_matrix_says_which_shards_each_arch_owes():
    assert expected_shards(_MATRIX) == {
        "wormhole": {"1", "2", "3"},
        "blackhole": {"1", "2", "3"},
    }


def test_an_artifact_suffix_does_not_hide_a_shard(tmp_path):
    (tmp_path / "perf-data-blackhole-2-head").mkdir()
    (tmp_path / "perf-data-blackhole-3").mkdir()
    (tmp_path / "llk-shadow-blackhole-1").mkdir()
    assert found_shards(str(tmp_path)) == {"blackhole": {"2", "3"}}


def test_a_complete_run_is_left_alone():
    found = {"wormhole": {"1", "2", "3"}, "blackhole": {"1", "2", "3"}}
    assert short_arches(expected_shards(_MATRIX), found) == {}


def test_a_lost_shard_drops_only_that_arch(tmp_path):
    kept = _run(
        tmp_path,
        {"wormhole": (1, 2, 3), "blackhole": (1, 2)},
    )
    assert kept == ["llk_perf_merge_baseline-20261102-4147-wormhole.parquet"]


def test_a_complete_run_keeps_both_arches(tmp_path):
    kept = _run(tmp_path, {"wormhole": (1, 2, 3), "blackhole": (1, 2, 3)})
    assert len(kept) == 2


def test_another_pipeline_keeps_its_partial_night(tmp_path):
    kept = _run(tmp_path, {"wormhole": (1, 2, 3), "blackhole": (1,)}, pipeline="pr")
    assert len(kept) == 2


def test_a_matrix_that_does_not_parse_drops_nothing(tmp_path, capsys):
    out_dir = tmp_path / "generated"
    out_dir.mkdir()
    (out_dir / "llk_perf_merge_baseline-1-blackhole.parquet").touch()
    main(
        [
            "--matrix",
            "{not json",
            "--in-dir",
            str(tmp_path / "absent"),
            "--out-dir",
            str(out_dir),
            "--pipeline",
            "merge_baseline",
        ]
    )
    assert list(out_dir.iterdir())
    assert "did not parse" in capsys.readouterr().out
