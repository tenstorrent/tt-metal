"""Evaluate a model on all tasks in a benchmark. See models/README.md for usage."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_MODELS_DIR = Path(__file__).parent
_DEFAULT_BENCHMARK = (
    "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/fev_bench/tasks_mini.yaml"
)


def list_available_models() -> list[str]:
    return sorted(d.name for d in _MODELS_DIR.iterdir() if (d / "model.py").exists())


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on a benchmark.")
    parser.add_argument("-m", "--model", required=True, help="Model name (must match a folder in models/)")
    parser.add_argument("-b", "--benchmark", default=_DEFAULT_BENCHMARK, help="Path or URL to benchmark YAML")
    parser.add_argument("-n", "--name", default=None, help="Display name for results (defaults to --model)")
    parser.add_argument("-k", "--model-kwargs", default="{}", help="JSON dict of kwargs passed to model constructor")
    parser.add_argument("-t", "--num-tasks", type=int, default=None, help="Limit number of tasks (for testing)")
    parser.add_argument("-d", "--deps-installed", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = _MODELS_DIR / args.model

    if not (model_dir / "model.py").exists():
        sys.exit(f"Model '{args.model}' not found. Available: {list_available_models()}")

    # Re-exec in ephemeral env with model deps if needed
    requirements_path = model_dir / "requirements.txt"
    if not args.deps_installed and requirements_path.exists():
        cmd = ["uv", "run", f"--with-requirements={requirements_path}", sys.argv[0]] + sys.argv[1:] + ["-d"]
        sys.exit(subprocess.run(cmd).returncode)

    import pandas as pd
    from tqdm import tqdm

    import fev

    spec = importlib.util.spec_from_file_location(f"models.{args.model}", model_dir / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_cls = fev.ForecastingModel.get_model_cls(args.model)
    model = model_cls(**json.loads(args.model_kwargs))
    display_name = args.name or args.model

    benchmark = fev.Benchmark.from_yaml(args.benchmark)
    tasks = benchmark.tasks[: args.num_tasks]

    extra_info = {"model_class": args.model, "model_kwargs": args.model_kwargs}
    summaries = []
    for task in tqdm(tasks):
        tqdm.write(f"Evaluating {task.task_name}")
        predictions = model.fit_predict(task)
        summary = task.evaluation_summary(
            predictions,
            model_name=display_name,
            training_time_s=model.training_time,
            inference_time_s=model.inference_time,
            trained_on_this_dataset=task.dataset_config in model.trained_on_datasets,
            extra_info=extra_info,
        )
        summaries.append(summary)

    df = pd.DataFrame(summaries)
    print(df[["model_name", "task_name", "test_error", "inference_time_s"]].to_string())
    output_path = f"{display_name}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
