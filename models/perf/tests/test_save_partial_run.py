# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pickle
from datetime import datetime, timezone

import models.perf.benchmarking_utils as benchmarking_utils
from infra.data_collection.pydantic_models import BenchmarkMeasurement, PartialBenchmarkRun


def _profiler_at(start: datetime):
    profiler = benchmarking_utils.BenchmarkProfiler()
    profiler.start("run")
    profiler.end("run")
    profiler.start_times[(0, "run")] = start
    profiler.end_times[(0, "run")] = start
    return profiler


def test_second_save_same_second_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmarking_utils, "IS_CI_ENV", True)
    monkeypatch.setattr(benchmarking_utils, "PartialBenchmarkRun", PartialBenchmarkRun, raising=False)
    monkeypatch.setattr(benchmarking_utils, "BenchmarkMeasurement", BenchmarkMeasurement, raising=False)

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts = start.strftime("%Y-%m-%dT%H:%M:%S%z")

    def save(model_name: str):
        benchmark_data = benchmarking_utils.BenchmarkData()
        benchmark_data.output_folder = str(tmp_path)
        benchmark_data.save_partial_run_json(
            _profiler_at(start),
            run_type="end_to_end_perf",
            ml_model_name=model_name,
        )

    save("first-model")
    save("second-model")

    unsuffixed = tmp_path / f"partial_run_{ts}.pkl"
    suffixed = tmp_path / f"partial_run_{ts}_2.pkl"
    assert set(tmp_path.glob("partial_run_*.pkl")) == {unsuffixed, suffixed}

    with unsuffixed.open("rb") as handle:
        first = pickle.load(handle)
    with suffixed.open("rb") as handle:
        second = pickle.load(handle)
    assert first.ml_model_name == "first-model"
    assert second.ml_model_name == "second-model"
    assert first.run_start_ts == second.run_start_ts


def test_save_does_not_overwrite_preexisting_files(tmp_path, monkeypatch):
    """Emulate a concurrent writer that has already reserved the unsuffixed name and the
    first numeric suffix. The atomic O_CREAT | O_EXCL create must skip those files and
    land on the next free suffix instead of silently clobbering an existing partial."""
    monkeypatch.setattr(benchmarking_utils, "IS_CI_ENV", True)
    monkeypatch.setattr(benchmarking_utils, "PartialBenchmarkRun", PartialBenchmarkRun, raising=False)
    monkeypatch.setattr(benchmarking_utils, "BenchmarkMeasurement", BenchmarkMeasurement, raising=False)

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts = start.strftime("%Y-%m-%dT%H:%M:%S%z")

    # Pre-fill the files a competing rank would have created with sentinel bytes.
    preexisting = {
        tmp_path / f"partial_run_{ts}.pkl": b"existing-unsuffixed",
        tmp_path / f"partial_run_{ts}_2.pkl": b"existing-suffix-2",
    }
    for path, payload in preexisting.items():
        path.write_bytes(payload)

    benchmark_data = benchmarking_utils.BenchmarkData()
    benchmark_data.output_folder = str(tmp_path)
    benchmark_data.save_partial_run_json(
        _profiler_at(start),
        run_type="end_to_end_perf",
        ml_model_name="new-model",
    )

    # Pre-existing files must be byte-for-byte untouched.
    for path, payload in preexisting.items():
        assert path.read_bytes() == payload

    # Our write must have gone to the next free suffix and be a valid pickle.
    new_file = tmp_path / f"partial_run_{ts}_3.pkl"
    assert set(tmp_path.glob("partial_run_*.pkl")) == set(preexisting) | {new_file}
    with new_file.open("rb") as handle:
        saved = pickle.load(handle)
    assert saved.ml_model_name == "new-model"
