# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import gc
from dataclasses import dataclass
from pathlib import Path

import torch
import ttnn

from models.experimental.mole.demo.core import (
    TrainingConfig,
    add_dataset_arguments,
    add_model_arguments,
    add_training_arguments,
    build_reference_expert,
    build_reference_mole,
    build_ttnn_expert,
    build_ttnn_mole,
    model_config_from_args,
    open_ttnn_device,
    predict_expert_from_torch,
    resolve_dataset_config,
    set_random_seed,
    training_config_from_args,
    train_model_on_dataloader,
    unpack_batch,
)
from models.experimental.mole.reference.config import MoLEConfig, replace_num_experts
from models.experimental.mole.utils.datasets import (
    create_real_dataset_loaders,
    RegressionMetricTotals,
    finalize_regression_metric_totals,
    update_regression_metric_totals,
)


DEFAULT_DATASET_COMPARE_EVAL_BATCHES = 64


@dataclass(frozen=True)
class ComparisonOptions:
    csv_output_path: str | None = None
    max_eval_batches: int | None = None
    ttnn_device: object | None = None
    verbose: bool = False
    seed: int = 0
    dataset_path: str | None = None


def _evaluate_ttnn_model_on_dataloader(
    device,
    model,
    data_loader,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    totals = RegressionMetricTotals()
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs, targets, input_marks, _ = unpack_batch(batch)
            predictions = predict_expert_from_torch(
                model=model,
                device=device,
                torch_input=inputs,
                torch_input_mark=input_marks,
            )
            update_regression_metric_totals(totals, predictions, targets)

    if totals.numel == 0:
        raise ValueError("TTNN evaluation produced no batches; increase max_eval_batches or verify the dataset split")

    return finalize_regression_metric_totals(totals)


def _run_dataset_comparison(
    config: MoLEConfig,
    training: TrainingConfig,
    *,
    dataset_name: str,
    dataset_path: str | None,
    max_eval_batches: int | None,
    ttnn_device=None,
    verbose: bool,
) -> dict[str, dict[str, float]]:
    loaders, input_dim, resolved_freq = create_real_dataset_loaders(
        dataset_name,
        dataset_path,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        batch_size=training.batch_size,
        eval_batch_size=training.eval_batch_size,
        freq=config.freq,
    )
    dataset_config = resolve_dataset_config(config, input_dim=input_dim, freq=resolved_freq)
    baseline_config = replace_num_experts(dataset_config, num_experts=1)

    if verbose:
        print(
            f"[compare] dataset={dataset_name} input_dim={input_dim} steps={training.steps} eval_batches={max_eval_batches if max_eval_batches is not None else 'full'}",
            flush=True,
        )
    owns_device = ttnn_device is None
    device = ttnn_device if ttnn_device is not None else open_ttnn_device()
    try:
        if verbose:
            print(
                f"[compare] training PyTorch reference {dataset_config.base_model_type} baseline, then evaluating with TTNN inference",
                flush=True,
            )
        baseline_result = train_model_on_dataloader(
            build_reference_expert(baseline_config),
            loaders,
            training,
            max_eval_batches=max_eval_batches,
            return_summary=True,
        )
        tt_baseline = build_ttnn_expert(device, baseline_config, baseline_result["trained_model"])
        baseline_metrics = _evaluate_ttnn_model_on_dataloader(
            device,
            tt_baseline,
            loaders["test"],
            max_batches=max_eval_batches,
        )
        del baseline_result
        del tt_baseline
        gc.collect()

        if verbose:
            print("[compare] training PyTorch reference mole, then evaluating with TTNN inference", flush=True)
        mole_result = train_model_on_dataloader(
            build_reference_mole(dataset_config),
            loaders,
            training,
            max_eval_batches=max_eval_batches,
            return_summary=True,
        )
        tt_mole = build_ttnn_mole(device, dataset_config, mole_result["trained_model"])
        mole_metrics = _evaluate_ttnn_model_on_dataloader(
            device,
            tt_mole,
            loaders["test"],
            max_batches=max_eval_batches,
        )
        del mole_result
        del tt_mole
        gc.collect()
    finally:
        if owns_device and device is not None:
            ttnn.close_device(device)
    return {
        dataset_config.base_model_type: baseline_metrics,
        "mole": mole_metrics,
    }


def _write_results_csv(output_path: str | None, results: dict[str, dict[str, float | str]]):
    if output_path is None:
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["model", "mse", "mae", "source"])
        writer.writeheader()
        for name, metrics in results.items():
            writer.writerow(
                {
                    "model": name,
                    "mse": metrics["mse"],
                    "mae": metrics["mae"],
                    "source": metrics.get("source", "measured"),
                }
            )
    return str(path)


def _summarize_results(measured_results: dict[str, dict[str, float]]) -> dict[str, object]:
    merged_results: dict[str, dict[str, float | str]] = {
        name: {**metrics, "source": "measured"} for name, metrics in measured_results.items()
    }

    return {"results": merged_results}


def run_reference_comparison(
    model_config: MoLEConfig,
    training_config: TrainingConfig,
    *,
    dataset_name: str,
    options: ComparisonOptions | None = None,
) -> dict[str, object]:
    compare_options = options or ComparisonOptions()

    set_random_seed(compare_options.seed)

    max_eval_batches = compare_options.max_eval_batches
    if max_eval_batches is None:
        max_eval_batches = DEFAULT_DATASET_COMPARE_EVAL_BATCHES
    measured_results = _run_dataset_comparison(
        model_config,
        training_config,
        dataset_name=dataset_name,
        dataset_path=compare_options.dataset_path,
        max_eval_batches=max_eval_batches,
        ttnn_device=compare_options.ttnn_device,
        verbose=compare_options.verbose,
    )

    summary = _summarize_results(measured_results)
    summary["metadata"] = {
        "dataset_name": dataset_name,
        "seq_len": model_config.seq_len,
        "pred_len": model_config.pred_len,
        "baseline_num_experts": 1,
        "num_experts": model_config.num_experts,
        "batch_size": training_config.batch_size,
        "eval_batch_size": training_config.eval_batch_size,
        "training_steps": training_config.steps,
        "compared_eval_batches": max_eval_batches,
        "training_backend": "pytorch_reference",
        "inference_backend": "ttnn",
        "seed": compare_options.seed,
    }
    csv_path = _write_results_csv(compare_options.csv_output_path, summary["results"])
    summary["csv_output_path"] = csv_path
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PyTorch reference models, then compare their TTNN inference metrics against single-model baselines."
    )
    add_dataset_arguments(
        parser,
        dataset_help="Dataset name, e.g. weather, ETTh1, ETTh2, ETTm1, ETTm2",
        dataset_path_help="Path to a TSLib-style CSV dataset file; when omitted for supported datasets, the file is downloaded to cache",
    )
    parser.add_argument(
        "--csv-output", type=str, default=None, help="Optional CSV output path for the comparison table"
    )
    add_model_arguments(parser)
    add_training_arguments(parser, default_batch_size=16, default_eval_batch_size=32, default_steps=80)
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Maximum number of test batches to score per model for dataset-backed comparisons; defaults to a bounded value for real datasets",
    )
    args = parser.parse_args()
    options = ComparisonOptions(
        csv_output_path=args.csv_output,
        max_eval_batches=args.max_eval_batches,
        verbose=True,
        seed=args.seed,
        dataset_path=args.dataset_path,
    )
    comparison = run_reference_comparison(
        model_config=model_config_from_args(args),
        training_config=training_config_from_args(args),
        dataset_name=args.dataset_name,
        options=options,
    )
    for name, metrics in comparison["results"].items():
        print(f"{name}: mse={metrics['mse']:.6f} mae={metrics['mae']:.6f}")
    print(f"csv_output_path: {comparison['csv_output_path']}")
