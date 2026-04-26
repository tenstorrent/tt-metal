# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from models.demos.wormhole.patchtst.config import PatchTSTDemoConfig, merge_demo_config
from models.demos.wormhole.patchtst.demo.data_utils import ARCHIVE_DATASET_FILES, FORECAST_DATASET_FILES
from models.demos.wormhole.patchtst.demo.runner import run_patchtst, run_streaming_forecast
from models.demos.wormhole.patchtst.reference.finetune import run_reference_finetune


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PatchTST demo runner")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run PatchTST on one workload.")
    run_dataset_choices = list((*FORECAST_DATASET_FILES.keys(), *ARCHIVE_DATASET_FILES.keys()))
    run_parser.add_argument(
        "--task",
        choices=["forecast", "regression", "pretraining", "classification", "multi_task"],
        default="forecast",
    )
    run_parser.add_argument("--channel-mode", choices=["independent", "attention"], default="independent")
    run_parser.add_argument("--share-embedding", choices=["true", "false"], default="true")
    run_parser.add_argument("--dataset", choices=run_dataset_choices, default=run_dataset_choices[0])
    run_parser.add_argument("--dataset-root", type=Path, default=Path("data/patchtst"))
    run_parser.add_argument("--trace", action="store_true")
    run_parser.add_argument("--batch-size", type=int, default=1)
    run_parser.add_argument("--max-windows", type=int, default=64)
    run_parser.add_argument("--context-length", type=int, default=512)

    streaming_parser = subparsers.add_parser(
        "streaming",
        help="Run stateful online forecasting with either cached or legacy full-rerun streaming semantics.",
    )
    streaming_dataset_choices = ["etth1", "weather", "traffic", "electricity", "exchange_rate"]
    streaming_parser.add_argument(
        "--dataset",
        choices=streaming_dataset_choices,
        default=streaming_dataset_choices[0],
    )
    streaming_parser.add_argument("--dataset-root", type=Path, default=Path("data/patchtst"))
    streaming_parser.add_argument("--trace", action="store_true")
    streaming_parser.add_argument("--streaming-mode", choices=["cached", "full-rerun"], default="cached")
    streaming_parser.add_argument("--stream-steps", type=int, default=4)

    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Generate a dataset-matched reference checkpoint for forecast/classification/regression.",
    )
    finetune_dataset_choices = [
        "etth1",
        "weather",
        "traffic",
        "electricity",
        "exchange_rate",
        "heartbeat_cls",
        "flood_modeling1_reg",
    ]
    finetune_parser.add_argument("--task", choices=["forecast", "classification", "regression"], default="forecast")
    finetune_parser.add_argument("--channel-mode", choices=["independent", "attention"], default="independent")
    finetune_parser.add_argument(
        "--dataset",
        choices=finetune_dataset_choices,
        default=finetune_dataset_choices[0],
    )
    finetune_parser.add_argument("--dataset-root", type=Path, default=Path("data/patchtst"))
    finetune_parser.add_argument("--steps", type=int, default=20)
    finetune_parser.add_argument("--batch-size", type=int, default=4)
    finetune_parser.add_argument("--context-length", type=int, default=512)
    finetune_parser.add_argument("--prediction-length", type=int, default=96)
    finetune_parser.add_argument("--learning-rate", type=float, default=1e-4)
    finetune_parser.add_argument("--seed", type=int, default=1337)
    finetune_parser.add_argument("--checkpoint-id-override", type=str, default=None)
    finetune_parser.add_argument("--checkpoint-revision-override", type=str, default=None)
    finetune_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/demos/wormhole/patchtst/artifacts/finetune/reference_ckpt"),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    cli_args = sys.argv[1:]
    if not cli_args:
        cli_args = ["run"]
    elif cli_args[0].startswith("-") and cli_args[0] not in {"-h", "--help"}:
        cli_args = ["run", *cli_args]
    args = parser.parse_args(cli_args)

    if args.command == "run":
        base = PatchTSTDemoConfig(task=args.task)
        cfg = merge_demo_config(
            base,
            task=args.task,
            channel_mode=args.channel_mode,
            share_embedding=(args.share_embedding == "true"),
            dataset=args.dataset,
            dataset_root=args.dataset_root,
            use_trace=args.trace,
            batch_size=args.batch_size,
            max_windows=args.max_windows,
            context_length=args.context_length,
        )
        prediction = run_patchtst(cfg)
        print(prediction)
        return

    if args.command == "streaming":
        cfg = merge_demo_config(
            PatchTSTDemoConfig(task="forecast"),
            task="forecast",
            dataset=args.dataset,
            dataset_root=args.dataset_root,
            use_trace=args.trace,
        )
        prediction = run_streaming_forecast(
            config=cfg,
            stream_steps=args.stream_steps,
            streaming_mode=args.streaming_mode,
        )
        print(prediction)
        return

    if args.command == "finetune":
        cfg = merge_demo_config(
            PatchTSTDemoConfig(task=args.task),
            task=args.task,
            channel_mode=args.channel_mode,
            dataset=args.dataset,
            dataset_root=args.dataset_root,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            checkpoint_id_override=args.checkpoint_id_override,
            checkpoint_revision_override=args.checkpoint_revision_override,
        )
        result = run_reference_finetune(
            config=cfg,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        print(result)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
