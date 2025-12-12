#!/usr/bin/env python
import os

import numpy as np
import pandas as pd
import torch

from transformers import (
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index


def main():
    # --------------------------------------------------------
    # 0. Basic setup
    # --------------------------------------------------------
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters (kept small to be CPU-friendly)
    context_length = 512
    forecast_horizon = 96
    batch_size = 32              # reduce if RAM is tight
    num_workers = 4              # adjust to your CPU
    num_train_epochs = 4         # keep small for quick training

    dataset = "ETTh2"
    dataset_path = (
        f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{dataset}.csv"
    )
    timestamp_column = "date"
    id_columns = []
    forecast_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    # --------------------------------------------------------
    # 1. Load raw data and create splits (same as blog)
    # --------------------------------------------------------
    print("Loading ETTh2 data...")
    data = pd.read_csv(dataset_path, parse_dates=[timestamp_column])

    # same index logic as the blog
    train_start_index = None  # beginning of dataset
    train_end_index = 12 * 30 * 24

    valid_start_index = 12 * 30 * 24 - context_length
    valid_end_index = 12 * 30 * 24 + 4 * 30 * 24

    test_start_index = 12 * 30 * 24 + 4 * 30 * 24 - context_length
    test_end_index = 12 * 30 * 24 + 8 * 30 * 24

    train_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=train_start_index,
        end_index=train_end_index,
    )
    valid_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=valid_start_index,
        end_index=valid_end_index,
    )
    test_data = select_by_index(
        data,
        id_columns=id_columns,
        start_index=test_start_index,
        end_index=test_end_index,
    )

    print(f"Train samples: {len(train_data)}, "
          f"Valid samples: {len(valid_data)}, "
          f"Test samples: {len(test_data)}")

    # --------------------------------------------------------
    # 2. TimeSeriesPreprocessor and ForecastDFDataset
    # --------------------------------------------------------
    print("Fitting TimeSeriesPreprocessor...")
    time_series_processor = TimeSeriesPreprocessor(
        context_length=context_length,
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        input_columns=forecast_columns,
        output_columns=forecast_columns,
        scaling=True,
    )
    time_series_processor.train(train_data)

    print("Building PyTorch datasets...")
    train_dataset = ForecastDFDataset(
        time_series_processor.preprocess(train_data),
        id_columns=id_columns,
        target_columns=forecast_columns,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )
    valid_dataset = ForecastDFDataset(
        time_series_processor.preprocess(valid_data),
        id_columns=id_columns,
        target_columns=forecast_columns,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )
    test_dataset = ForecastDFDataset(
        time_series_processor.preprocess(test_data),
        id_columns=id_columns,
        target_columns=forecast_columns,
        context_length=context_length,
        prediction_length=forecast_horizon,
    )

    print(
        f"Train windows: {len(train_dataset)}, "
        f"Valid windows: {len(valid_dataset)}, "
        f"Test windows: {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # 3. Configure PatchTSMixer (small model for CPU)
    # --------------------------------------------------------
    patch_length = 8  # should divide context_length
    config = PatchTSMixerConfig(
        context_length=context_length,
        prediction_length=forecast_horizon,
        patch_length=patch_length,
        num_input_channels=len(forecast_columns),
        patch_stride=patch_length,
        d_model=16,          # small hidden dim
        num_layers=4,        # fewer layers than blog (8) to speed up
        expansion_factor=2,
        dropout=0.1,
        head_dropout=0.1,
        mode="common_channel",  # channel-independent mode (simpler)
        scaling="std",
    )
    model = PatchTSMixerForPrediction(config).to(device)
    print(model)

    # --------------------------------------------------------
    # 4. TrainingArguments + Trainer
    # --------------------------------------------------------
    output_dir = "./checkpoint/patchtsmixer/etth2/simple_run/"

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=1e-3,
        num_train_epochs=num_train_epochs,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        report_to="none",      # turn off TB for simplicity
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=4,
        early_stopping_threshold=0.001,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback],
    )

    # --------------------------------------------------------
    # 5. Train + evaluate
    # --------------------------------------------------------
    print("\n=== Training PatchTSMixer on ETTh2 (small run) ===")
    trainer.train()

    print("\n=== Evaluating on test set ===")
    results = trainer.evaluate(test_dataset)
    print("Test results:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # --------------------------------------------------------
    # 6. Save model and preprocessor
    # --------------------------------------------------------
    save_model_dir = "patchtsmixer/etth2/simple_model/"
    save_preproc_dir = "patchtsmixer/etth2/simple_preprocessor/"

    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_preproc_dir, exist_ok=True)

    print(f"\nSaving model to: {save_model_dir}")
    trainer.save_model(save_model_dir)

    print(f"Saving preprocessor to: {save_preproc_dir}")
    time_series_processor.save_pretrained(save_preproc_dir)


if __name__ == "__main__":
    main()
