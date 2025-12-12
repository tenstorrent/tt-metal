import os
import numpy as np
import torch
import pandas as pd

from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction, set_seed
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

set_seed(42)

dataset = "ETTh2"
dataset_path = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{dataset}.csv"
timestamp_column = "date"
id_columns = []
context_length = 512
forecast_horizon = 96

forecast_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

# FIX: use the variable, not the literal string
data = pd.read_csv(dataset_path, parse_dates=[timestamp_column])

train_start_index = None
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

time_series_processor = TimeSeriesPreprocessor(
    context_length=context_length,
    timestamp_column=timestamp_column,
    id_columns=id_columns,
    input_columns=forecast_columns,
    output_columns=forecast_columns,
    scaling=True,
)
time_series_processor.train(train_data)

# Optional but recommended: pass timestamp_column
test_dataset = ForecastDFDataset(
    time_series_processor.preprocess(test_data),
    id_columns=id_columns,
    target_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model.
model_dir = "patchtsmixer/etth2/simple_model/"

model = PatchTSMixerForPrediction.from_pretrained(model_dir).to(device)
model.eval()

# Take one batch from test_dataset
batch_size = 4  # something small
samples = [test_dataset[i] for i in range(batch_size)]

past_values = torch.stack([s["past_values"] for s in samples], dim=0)    # (B, L, C)
future_values = torch.stack([s["future_values"] for s in samples], dim=0)  # (B, H, C)

past_values = past_values.to(device)
future_values = future_values.to(device)

# Run the model
with torch.no_grad():
    outputs = model(past_values=past_values, future_values=future_values)
    preds = outputs.prediction_outputs.detach().cpu()  # (B, H, C)

# Save reference tensors
os.makedirs("patchtsmixer_ref", exist_ok=True)

np.save("patchtsmixer_ref/ref_input_past_values.npy", past_values.cpu().numpy())
np.save("patchtsmixer_ref/ref_future_values.npy", future_values.cpu().numpy())
np.save("patchtsmixer_ref/ref_predictions.npy", preds.numpy())

print("Saved reference tensors in patchtsmixer_ref/")
print("  - ref_input_past_values.npy")
print("  - ref_future_values.npy")
print("  - ref_predictions.npy")
