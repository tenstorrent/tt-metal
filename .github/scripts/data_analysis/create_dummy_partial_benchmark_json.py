import os
from pathlib import Path

from infra.data_collection.pydantic_models import BenchmarkMeasurement, PartialBenchmarkRun

if __name__ == "__main__":
    measurements = [
        BenchmarkMeasurement(
            step_start_ts="2024-10-09T20:45:49+0000",
            step_end_ts="2024-10-09T20:45:52+0000",
            iteration=0,
            step_name="inference_decode",
            name="tokens/s",
            value=3051.942319201063,
        )
    ]
    run_start_ts = "2024-10-09T20:44:58+0000"
    partial_benchmark_run = PartialBenchmarkRun(
        run_start_ts=run_start_ts,
        run_end_ts="2024-10-09T20:45:52+0000",
        run_type="demo_perf_8chip",
        ml_model_name="tiiuae/falcon-7b-instruct",
        measurements=measurements,
    )

    json_data = partial_benchmark_run.model_dump_json()

    current_data_analysis_path = Path(__file__)
    benchmark_data_dir = current_data_analysis_path.parent.parent.parent.parent / "generated/benchmark_data"
    assert benchmark_data_dir.exists()
    assert benchmark_data_dir.is_dir()

    output_path = os.path.join(benchmark_data_dir, f"partial_run_{run_start_ts}.json")
    with open(output_path, "w") as f:
        f.write(json_data)
