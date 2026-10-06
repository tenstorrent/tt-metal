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
