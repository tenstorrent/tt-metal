import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_patchtsmixer import PatchTSMixerModelForForecasting
from torch.utils.data import DataLoader
from transformers import set_seed
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index


def build_etth2_datasets(
    context_length: int,
    prediction_length: int,
    num_workers: int = 4,
):
    """
    Build ETTh2 train/valid/test ForecastDFDataset splits,
    same index logic as the IBM blog.
    """
    dataset = "ETTh2"
    dataset_path = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{dataset}.csv"
    timestamp_column = "date"
    id_columns = []

    data = pd.read_csv(dataset_path, parse_dates=[timestamp_column])

    # columns to forecast
    forecast_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

    # same split logic as the blog
    train_start_index = None  # beginning
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

    # Preprocess: scaling + window extraction
    time_series_processor = TimeSeriesPreprocessor(
        context_length=context_length,
        timestamp_column=timestamp_column,
        id_columns=id_columns,
        input_columns=forecast_columns,
        output_columns=forecast_columns,
        scaling=True,
    )
    time_series_processor.train(train_data)

    train_dataset = ForecastDFDataset(
        time_series_processor.preprocess(train_data),
        id_columns=id_columns,
        target_columns=forecast_columns,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    valid_dataset = ForecastDFDataset(
        time_series_processor.preprocess(valid_data),
        id_columns=id_columns,
        target_columns=forecast_columns,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    test_dataset = ForecastDFDataset(
        time_series_processor.preprocess(test_data),
        id_columns=id_columns,
        target_columns=forecast_columns,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    return (
        train_dataset,
        valid_dataset,
        test_dataset,
        time_series_processor,
        forecast_columns,
    )


def train_or_eval_epoch(model, loader, optimizer=None, device="cpu"):
    """
    One epoch over the loader.
    If optimizer is None -> eval mode.
    """
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        past_values = batch["past_values"].to(device)  # (B, L, C)
        future_values = batch["future_values"].to(device)  # (B, H, C)

        preds = model(past_values)  # (B, H, C)

        loss = F.mse_loss(preds, future_values)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = past_values.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--prediction_length", type=int, default=96)
    parser.add_argument("--patch_length", type=int, default=8)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mode", type=str, default="common_channel", choices=["common_channel", "mix_channel"])
    parser.add_argument("--use_gated_attn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="simple_patchtsmixer_etth2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Config ===")
    print(args)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("=== Building ETTh2 datasets ===")
    (
        train_dataset,
        valid_dataset,
        test_dataset,
        time_series_processor,
        forecast_columns,
    ) = build_etth2_datasets(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        num_workers=args.num_workers,
    )

    num_channels = len(forecast_columns)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("=== Building model ===")
    model = PatchTSMixerModelForForecasting(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        num_channels=num_channels,
        d_model=args.d_model,
        num_layers=args.num_layers,
        mode=args.mode,
        expansion=2,
        dropout=args.dropout,
        use_gated_attn=args.use_gated_attn,
        head_dropout=args.head_dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_path = os.path.join(args.output_dir, "best_model.pt")

    print("=== Training ===")
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_or_eval_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_loss = train_or_eval_epoch(model, valid_loader, optimizer=None, device=device)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved to {best_path}")

    print("=== Evaluating on test set (best model) ===")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss = train_or_eval_epoch(model, test_loader, optimizer=None, device=device)
    print(f"Test MSE: {test_loss:.6f}")

    # Save model + preprocessor for later reference / TTNN comparison
    model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Final model weights saved to {model_path}")

    preproc_dir = os.path.join(args.output_dir, "preprocessor")
    os.makedirs(preproc_dir, exist_ok=True)
    time_series_processor.save_pretrained(preproc_dir)
    print(f"Preprocessor saved to {preproc_dir}")


if __name__ == "__main__":
    main()
