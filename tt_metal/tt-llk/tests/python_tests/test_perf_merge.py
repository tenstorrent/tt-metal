# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests for merging a run's shards into one file per arch.

The warehouse wants one file per run; CI produces ten. These cover the two
columns the merge has to unify, the two it must refuse to paper over, and the
shape of the run_id it mints.
"""

import pandas as pd
import pyarrow.parquet as pq
from helpers.perf.merge import (
    attempt_of,
    group_by_arch,
    main,
    merge_run,
    merged_run_id,
    workflow_run_id_of,
)
from helpers.perf.parquet import write_run_batch


def _shard(root, *, run_id, arch, shard, timestamp, commit_sha="deadbeef", rows=2):
    """Write one shard's Parquet the way a nightly job does."""
    tag = f"{run_id.split('-', 1)[0]}-{arch}-{shard}"
    path = root / tag / f"{tag}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_run_batch(
        {
            f"perf_{shard}": pd.DataFrame(
                {
                    "marker": ["INIT", "KERNEL"][:rows],
                    "tile_cnt": [4, 4][:rows],
                    "mean(MATH_ISOLATE)": [10.0, 20.0][:rows],
                }
            )
        },
        str(path),
        commit_sha=commit_sha,
        arch=arch,
        run_id=run_id,
        timestamp=timestamp,
        pipeline="nightly",
    )
    return path


def _night(root, run="42", *, commit_sha="deadbeef"):
    """Five shards on each of two arches, with staggered write times."""
    for arch in ("wormhole", "blackhole"):
        for shard in range(5):
            _shard(
                root,
                run_id=run,
                arch=arch,
                shard=shard,
                timestamp=f"2026-09-01T0{3 + shard}:10:00+00:00",
                commit_sha=commit_sha,
            )


def test_merge_produces_one_file_per_arch(tmp_path):
    _night(tmp_path / "runs")

    merged = merge_run(
        list((tmp_path / "runs").glob("**/*.parquet")), str(tmp_path / "out")
    )

    assert [(m["arch"], m["shards"], m["rows"]) for m in merged] == [
        ("blackhole", 5, 10),
        ("wormhole", 5, 10),
    ]
    assert sorted(p.name for p in (tmp_path / "out").glob("*.parquet")) == [
        "llk_perf_nightly-20260901-42-blackhole.parquet",
        "llk_perf_nightly-20260901-42-wormhole.parquet",
    ]


def test_merge_unifies_run_id_and_takes_the_earliest_timestamp(tmp_path):
    # Both vary per shard, and the loader fails a file whose run columns are
    # not constant. RUN_TS is "when the run executed", so earliest wins.
    _night(tmp_path / "runs")

    merge_run(list((tmp_path / "runs").glob("**/*.parquet")), str(tmp_path / "out"))

    table = pq.read_table(
        tmp_path / "out" / "llk_perf_nightly-20260901-42-wormhole.parquet"
    )
    assert set(table.column("run_id").to_pylist()) == {"nightly-20260901-42-wormhole"}
    assert set(table.column("timestamp").to_pylist()) == {"2026-09-01T03:10:00+00:00"}
    assert set(table.column("arch").to_pylist()) == {"wormhole"}


def test_merge_keeps_every_row(tmp_path):
    _night(tmp_path / "runs")
    shards = list((tmp_path / "runs").glob("**/*.parquet"))
    before = sum(pq.read_metadata(p).num_rows for p in shards)

    merged = merge_run(shards, str(tmp_path / "out"))

    assert sum(m["rows"] for m in merged) == before


def test_merge_refuses_shards_from_two_commits(tmp_path):
    # Two workflows' artefacts in one directory is a mistake worth failing on:
    # merging them would publish one run_id over two different SHAs.
    runs = tmp_path / "runs"
    _shard(
        runs,
        run_id="42",
        arch="wormhole",
        shard=0,
        timestamp="2026-09-01T03:00:00+00:00",
    )
    _shard(
        runs,
        run_id="43",
        arch="wormhole",
        shard=1,
        timestamp="2026-09-01T04:00:00+00:00",
        commit_sha="cafebabe",
    )

    try:
        merge_run(list(runs.glob("**/*.parquet")), str(tmp_path / "out"))
    except ValueError as e:
        assert "commit_sha is not constant" in str(e)
    else:
        raise AssertionError("expected a ValueError")


def test_group_by_arch_reads_the_rows_not_the_file_name(tmp_path):
    # File naming has been three different things across the archive; the arch
    # column is the same in every file ever written.
    runs = tmp_path / "runs"
    path = _shard(
        runs,
        run_id="42",
        arch="blackhole",
        shard=0,
        timestamp="2026-09-01T03:00:00+00:00",
    )
    renamed = runs / "totally-unrelated-name.parquet"
    path.rename(renamed)

    assert list(group_by_arch([str(renamed)])) == ["blackhole"]


def test_run_id_carries_the_workflow_run_so_a_dispatch_cannot_replace_a_night():
    # Same date, same arch, different workflow run -> different id. Without the
    # run id, a manual dispatch would replay over that night's scheduled run.
    scheduled = merged_run_id(
        pipeline="nightly",
        timestamp="2026-09-02T03:09:31+00:00",
        workflow_run_id="33585970698",
        arch="wormhole",
    )
    dispatched = merged_run_id(
        pipeline="nightly",
        timestamp="2026-09-02T11:37:38+00:00",
        workflow_run_id="33619869335",
        arch="wormhole",
    )

    assert scheduled == "nightly-20260902-33585970698-wormhole"
    assert scheduled != dispatched


def test_run_id_keeps_a_re_run_attempt():
    assert (
        merged_run_id(
            pipeline="nightly",
            timestamp="2026-09-01T03:00:00+00:00",
            workflow_run_id="42",
            arch="wormhole",
            attempt="2",
        )
        == "nightly-20260901-42-wormhole-2"
    )


def test_attempt_and_workflow_id_come_from_the_shard_run_id():
    assert workflow_run_id_of("33465181016-wormhole-4") == "33465181016"
    assert workflow_run_id_of("33465181016") == "33465181016"
    # A shard index is a trailing number; only the prefix rule can tell them apart.
    assert attempt_of("33465181016-wormhole-4", "33465181016-wormhole-4") == ""
    assert attempt_of("33465181016-2", "33465181016-blackhole-7") == "2"
    assert attempt_of("42-wormhole-0-2", "42-wormhole-0") == "2"


def test_cli_merges_and_reports(tmp_path, capsys):
    _night(tmp_path / "runs")

    code = main(
        ["--in-dir", str(tmp_path / "runs"), "--out-dir", str(tmp_path / "out")]
    )

    assert code == 0
    assert "10 shard file(s) -> 2 run file(s)" in capsys.readouterr().out


def test_cli_fails_on_an_empty_input_dir(tmp_path):
    assert main(["--in-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")]) == 1
