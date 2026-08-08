# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import time
from typing import Any

import torch
import ttnn

from models.experimental.mole.demo.run import (
    CHECKPOINT_BASE_DIR,
    build_ttnn_mole_from_checkpoint,
    close_ttnn_device,
    config_from_checkpoint_resolution,
    create_local_dataset_loaders,
    load_checkpoint_index,
    load_reference_checkpoint,
    open_ttnn_device,
    resolve_mole_checkpoint,
    set_random_seed,
    to_torch_with_cached_host,
    unpack_batch,
    upload_mole_inputs,
)
from models.experimental.mole.reference.mole import MixtureOfLinearExperts


MILLISECONDS_PER_SECOND = 1000.0
BATCH_SIZE = 1
WARMUP_ITERATIONS = 3
MEASURE_ITERATIONS = 20
SEED = 0


def _measure_ttnn(
    *,
    model: Any,
    device: Any,
    torch_input: torch.Tensor,
    torch_input_mark: torch.Tensor,
    warmup_iterations: int,
    measure_iterations: int,
) -> tuple[torch.Tensor, float, float]:
    tt_input, tt_marks = upload_mole_inputs(
        model=model,
        device=device,
        torch_input=torch_input,
        torch_input_mark=torch_input_mark,
    )

    def infer_once() -> ttnn.Tensor:
        return model.forward_prediction(tt_input, tt_marks)

    for _ in range(warmup_iterations):
        infer_once()
    ttnn.synchronize_device(device)

    output = None
    start = time.perf_counter()
    for _ in range(measure_iterations):
        output = infer_once()
    ttnn.synchronize_device(device)
    elapsed_s = time.perf_counter() - start

    prediction = to_torch_with_cached_host(model=model, device_tensor=output, cache_name="demo_prediction").squeeze(0)
    latency_ms = (elapsed_s / measure_iterations) * MILLISECONDS_PER_SECOND
    sequences_per_second = (int(torch_input.shape[0]) * measure_iterations) / elapsed_s
    return prediction, latency_ms, sequences_per_second


def _measure_reference(
    *,
    model: MixtureOfLinearExperts,
    torch_input: torch.Tensor,
    torch_input_mark: torch.Tensor,
    warmup_iterations: int,
    measure_iterations: int,
) -> tuple[torch.Tensor, float, float]:
    with torch.no_grad():
        for _ in range(warmup_iterations):
            model(torch_input, torch_input_mark)

        prediction = None
        start = time.perf_counter()
        for _ in range(measure_iterations):
            prediction, _ = model(torch_input, torch_input_mark)
        elapsed_s = time.perf_counter() - start

    latency_ms = (elapsed_s / measure_iterations) * MILLISECONDS_PER_SECOND
    sequences_per_second = (int(torch_input.shape[0]) * measure_iterations) / elapsed_s
    return prediction, latency_ms, sequences_per_second


def _errors(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    diff = prediction.to(dtype=torch.float32) - target.to(dtype=torch.float32)
    return float(torch.mean(diff * diff).item()), float(torch.mean(torch.abs(diff)).item())


def _format_row(values: list[object], widths: list[int]) -> str:
    return "  ".join(str(value).rjust(width) for value, width in zip(values, widths))


def _print_rows(rows: list[list[object]]) -> None:
    headers = [
        "checkpoint",
        "ch",
        "TTNN ms",
        "TTNN seq/s",
        "Ref ms",
        "Ref seq/s",
        "TTNN MSE",
        "Ref MSE",
        "TTNN MAE",
        "Ref MAE",
    ]
    widths = [max(len(str(row[index])) for row in [headers, *rows]) for index in range(len(headers))]
    print(_format_row(headers, widths))
    print(_format_row(["-" * width for width in widths], widths))
    for row in rows:
        print(_format_row(row, widths))


def run_demo(batch_size: int = BATCH_SIZE) -> list[dict[str, object]]:
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    set_random_seed(SEED)
    entries = load_checkpoint_index(assets_root=CHECKPOINT_BASE_DIR)
    rows: list[dict[str, object]] = []
    table_rows: list[list[object]] = []
    device = open_ttnn_device()
    try:
        for index, entry in enumerate(entries, start=1):
            resolution = resolve_mole_checkpoint(
                dataset=entry.dataset,
                base_model_type=entry.base_model,
                num_experts=entry.experts,
                assets_root=CHECKPOINT_BASE_DIR,
            )
            config = config_from_checkpoint_resolution(
                resolution,
                base_model_type=entry.base_model,
                num_experts=entry.experts,
            )
            loaders, _, _ = create_local_dataset_loaders(
                resolution.dataset_csv_path,
                seq_len=entry.seq_len,
                pred_len=entry.pred_len,
                eval_batch_size=batch_size,
                freq=resolution.freq,
            )
            torch_input, torch_target, torch_input_mark, _ = unpack_batch(next(iter(loaders["test"])))
            if torch_input_mark is None:
                raise ValueError(f"{entry.dataset}/{entry.base_model}/e{entry.experts} did not provide time marks")

            reference_model = MixtureOfLinearExperts(config).eval()
            load_reference_checkpoint(reference_model, resolution.checkpoint_path)
            reference_prediction, reference_ms, reference_seq_s = _measure_reference(
                model=reference_model,
                torch_input=torch_input,
                torch_input_mark=torch_input_mark,
                warmup_iterations=WARMUP_ITERATIONS,
                measure_iterations=MEASURE_ITERATIONS,
            )
            reference_mse, reference_mae = _errors(reference_prediction, torch_target)

            ttnn_model = build_ttnn_mole_from_checkpoint(device, config, resolution.checkpoint_path)
            ttnn_prediction, ttnn_ms, ttnn_seq_s = _measure_ttnn(
                model=ttnn_model,
                device=device,
                torch_input=torch_input,
                torch_input_mark=torch_input_mark,
                warmup_iterations=WARMUP_ITERATIONS,
                measure_iterations=MEASURE_ITERATIONS,
            )
            ttnn_mse, ttnn_mae = _errors(ttnn_prediction, torch_target)

            row = {
                "checkpoint": f"{entry.dataset}/{entry.base_model}/e{entry.experts}",
                "channels": entry.enc_in,
                "ttnn_latency_ms": ttnn_ms,
                "ttnn_sequences_per_second": ttnn_seq_s,
                "reference_latency_ms": reference_ms,
                "reference_sequences_per_second": reference_seq_s,
                "ttnn_mse": ttnn_mse,
                "reference_mse": reference_mse,
                "ttnn_mae": ttnn_mae,
                "reference_mae": reference_mae,
            }
            rows.append(row)

            table_row: list[object] = [
                row["checkpoint"],
                entry.enc_in,
                f"{ttnn_ms:.3f}",
                f"{ttnn_seq_s:.1f}",
                f"{reference_ms:.3f}",
                f"{reference_seq_s:.1f}",
                f"{ttnn_mse:.4f}",
                f"{reference_mse:.4f}",
                f"{ttnn_mae:.4f}",
                f"{reference_mae:.4f}",
            ]
            table_rows.append(table_row)
            print(f"[{index:02d}/{len(entries):02d}] {row['checkpoint']}", flush=True)

            del ttnn_model, reference_model
    finally:
        close_ttnn_device(device)

    _print_rows(table_rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every MoLE checkpoint on TTNN and PyTorch reference.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    run_demo(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
