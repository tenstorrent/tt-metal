# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the run-attempt log archive unpacker.

The helper is standard-library-only by design, so these run without the rest of the
data-collection dependency set.

The cases that matter are the ones where a wrong answer is silent: an entry attributed to
the wrong job produces a plausible, non-empty <job_id>.log, which the caller's "is the
file missing?" fallback check cannot detect. Those are covered explicitly.
"""

import io
import json
import zipfile

import pytest

from infra.data_collection.github import extract_job_logs_from_archive as extractor


def _archive(entries: dict) -> zipfile.ZipFile:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


def _jobs(*pairs) -> list:
    return [{"id": job_id, "name": name} for job_id, name in pairs]


def test_maps_entries_to_job_ids_on_the_exact_archive_name():
    archive = _archive({"0_build.txt": "log body"})
    assert extractor.resolve_entries(archive, _jobs((11, "build"))) == {"0_build.txt": 11}


def test_slash_in_a_job_name_is_a_underscore_in_the_archive():
    # "<caller job> / <called job>" is the shape every reusable-workflow job takes, and
    # the "/" is the one character GitHub rewrites.
    archive = _archive({"7_ttnn-merge-gate-tests _ fetch-ttsim (libttsim_bh.so).txt": "body"})
    jobs = _jobs((22, "ttnn-merge-gate-tests / fetch-ttsim (libttsim_bh.so)"))
    assert extractor.resolve_entries(archive, jobs) == {
        "7_ttnn-merge-gate-tests _ fetch-ttsim (libttsim_bh.so).txt": 22
    }


def test_emoji_and_punctuation_survive_byte_for_byte():
    # `unzip -l` renders these as "?", but the central directory holds them verbatim, so
    # the exact-name tier must match without any normalization.
    name = "build-sweeps _ build (Ubuntu 22.04, Release) _ 🛠️ Build Release"
    archive = _archive({f"50_{name}.txt": "body"})
    assert extractor.resolve_entries(archive, _jobs((33, name))) == {f"50_{name}.txt": 33}


def test_normalized_form_is_a_second_tier_when_the_exact_name_does_not_match():
    archive = _archive({"3_Build   Release.txt": "body"})
    assert extractor.resolve_entries(archive, _jobs((44, "Build Release"))) == {"3_Build   Release.txt": 44}


def test_two_jobs_sharing_a_name_are_left_unmapped():
    # Rather than handing the entry to whichever job the API happened to list first.
    archive = _archive({"0_matrix leg.txt": "body"})
    assert extractor.resolve_entries(archive, _jobs((1, "matrix leg"), (2, "matrix leg"))) == {}


def test_distinct_names_colliding_only_under_normalization_are_left_unmapped():
    # "a/b" -> "a_b" and "a-b" both normalize to "ab". The exact tier still resolves
    # "a_b", but "a-b" is ambiguous at the normalized tier and must not be guessed.
    archive = _archive({"0_a_b.txt": "first", "1_a-b.txt": "second"})
    resolved = extractor.resolve_entries(archive, _jobs((1, "a/b"), (2, "a-b")))
    assert resolved == {"0_a_b.txt": 1, "1_a-b.txt": 2}


def test_one_job_claimed_by_several_entries_is_left_unmapped():
    archive = _archive({"0_Build Release.txt": "first", "1_build release.txt": "second"})
    assert extractor.resolve_entries(archive, _jobs((5, "Build Release"))) == {}


def test_per_step_entries_and_unrelated_files_are_ignored():
    archive = _archive(
        {
            "0_build.txt": "job log",
            "build/1_Set up job.txt": "step log",
            "build/2_🛠️ Compile.txt": "step log",
            "notes.md": "not a job log",
        }
    )
    assert extractor.resolve_entries(archive, _jobs((9, "build"))) == {"0_build.txt": 9}


def test_entry_with_no_matching_job_is_left_unmapped():
    archive = _archive({"0_build.txt": "body", "1_ghost.txt": "body"})
    assert extractor.resolve_entries(archive, _jobs((9, "build"))) == {"0_build.txt": 9}


def test_extract_writes_job_id_named_files_with_the_entry_contents(tmp_path):
    archive_path = tmp_path / "logs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("0_build.txt", "first body")
        archive.writestr("1_test _ run.txt", "second body")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    written = extractor.extract(archive_path, _jobs((11, "build"), (22, "test / run")), logs_dir)

    assert written == 2
    assert (logs_dir / "11.log").read_text() == "first body"
    assert (logs_dir / "22.log").read_text() == "second body"
    # Nothing named after an archive entry reaches disk.
    assert sorted(p.name for p in logs_dir.iterdir()) == ["11.log", "22.log"]


def test_ambiguous_job_gets_no_file_so_the_caller_falls_back(tmp_path):
    archive_path = tmp_path / "logs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("0_matrix leg.txt", "body")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    assert extractor.extract(archive_path, _jobs((1, "matrix leg"), (2, "matrix leg")), logs_dir) == 0
    assert list(logs_dir.iterdir()) == []


def test_malformed_archive_is_reported_rather_than_raising(tmp_path, capsys):
    archive_path = tmp_path / "logs.zip"
    archive_path.write_bytes(b"this is not a zip")
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(json.dumps({"jobs": _jobs((1, "build"))}))
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    exit_code = _run_main(tmp_path, archive_path, jobs_path, logs_dir)

    assert exit_code == 1
    assert "could not unpack" in capsys.readouterr().err


def test_archive_with_no_recognizable_job_logs_signals_fallback(tmp_path, capsys):
    archive_path = tmp_path / "logs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("build/1_Set up job.txt", "step log only")
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(json.dumps({"jobs": _jobs((1, "build"))}))
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    exit_code = _run_main(tmp_path, archive_path, jobs_path, logs_dir)

    assert exit_code == 1
    assert "no recognizable job logs" in capsys.readouterr().err


def _run_main(tmp_path, archive_path, jobs_path, logs_dir) -> int:
    import sys

    argv = sys.argv
    sys.argv = [
        "extract_job_logs_from_archive.py",
        "--archive",
        str(archive_path),
        "--jobs-json",
        str(jobs_path),
        "--logs-dir",
        str(logs_dir),
    ]
    try:
        return extractor.main()
    finally:
        sys.argv = argv


if __name__ == "__main__":
    pytest.main([__file__])
