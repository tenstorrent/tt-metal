# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Save reference tensors from HuggingFace TimeSeriesTransformer.
These are used to validate the TTNN port component by component.

Data source: real Tourism Monthly batch from hf-internal-testing/tourism-monthly-batch
Model: huggingface/time-series-transformer-tourism-monthly (pinned revision)
"""

import torch
import json
from pathlib import Path
from transformers import TimeSeriesTransformerForPrediction
from huggingface_hub import hf_hub_download

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID          = "huggingface/time-series-transformer-tourism-monthly"
MODEL_REVISION    = "2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35"

DATASET_REPO      = "hf-internal-testing/tourism-monthly-batch"
DATASET_FILE      = "train-batch.pt"
DATASET_REVISION  = "81c7ee3cf3317e51beb97327df55926cd5bbfadb"  # pinned commit hash

SAVE_DIR = Path(__file__).resolve().parent.parent / "reference"


def main():
    # Move directory creation inside main — avoids import-time side effects
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading HuggingFace reference model...")
    model = TimeSeriesTransformerForPrediction.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    model.eval()
    cfg = model.config

    print(f"  MODEL_ID:                {MODEL_ID}")
    print(f"  MODEL_REVISION:          {MODEL_REVISION}")
    print(f"  context_length:          {cfg.context_length}")
    print(f"  prediction_length:       {cfg.prediction_length}")
    print(f"  num_time_features:       {cfg.num_time_features}")
    print(f"  lags_sequence:           {cfg.lags_sequence}")
    print(f"  d_model:                 {cfg.d_model}")
    print(f"  encoder_layers:          {cfg.encoder_layers}")
    print(f"  decoder_layers:          {cfg.decoder_layers}")
    print(f"  encoder_attention_heads: {cfg.encoder_attention_heads}")
    print(f"  input_size:              {cfg.input_size}")
    print(f"  num_static_real:         {cfg.num_static_real_features}")
    print(f"  num_static_categorical:  {cfg.num_static_categorical_features}")
    print(f"  cardinality:             {cfg.cardinality}")
    print(f"  embedding_dimension:     {cfg.embedding_dimension}")

    # ── Load real tourism batch ───────────────────────────────────────────────
    print("\nDownloading real tourism batch...")
    file = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    batch = torch.load(file)

    past_values                 = batch["past_values"]
    past_time_features          = batch["past_time_features"]
    past_observed_mask          = batch["past_observed_mask"]
    static_categorical_features = batch["static_categorical_features"]
    static_real_features        = batch["static_real_features"]
    future_time_features        = batch["future_time_features"]

    # ── Fix shapes ────────────────────────────────────────────────────────────
    if past_values.dim() == 2:
        past_values = past_values.unsqueeze(-1)
    if past_observed_mask.dim() == 2:
        past_observed_mask = past_observed_mask.unsqueeze(-1)

    # ── Validate past_len against actual tensor ───────────────────────────────
    expected_past_len = cfg.context_length + max(cfg.lags_sequence)
    actual_past_len   = past_values.shape[1]
    assert actual_past_len == expected_past_len, (
        f"past_len mismatch: config expects {expected_past_len}, "
        f"actual tensor is {actual_past_len}"
    )
    past_len = actual_past_len
    B        = past_values.shape[0]

    print(f"  batch_size:             {B}")
    print(f"  past_len:               {past_len}")
    print(f"  past_values shape:      {past_values.shape}")
    print(f"  past_observed_mask:     {past_observed_mask.shape}")
    print(f"  future_time_features:   {future_time_features.shape}")

    # ── Save inputs ───────────────────────────────────────────────────────────
    torch.save(past_values,                 SAVE_DIR / "input_past_values.pt")
    torch.save(past_time_features,          SAVE_DIR / "input_past_time_features.pt")
    torch.save(future_time_features,        SAVE_DIR / "input_future_time_features.pt")
    torch.save(past_observed_mask,          SAVE_DIR / "input_past_observed_mask.pt")
    torch.save(static_categorical_features, SAVE_DIR / "input_static_categorical.pt")
    torch.save(static_real_features,        SAVE_DIR / "input_static_real.pt")
    print("\nInputs saved.")

    # ── Register hooks ────────────────────────────────────────────────────────
    captured = {}

    def make_hook(name):
        def fn(module, input, output):
            if isinstance(output, torch.Tensor):
                captured[name] = output.detach()
            elif isinstance(output, tuple):
                captured[name] = output[0].detach()
            else:
                captured[name] = output.last_hidden_state.detach()
        return fn

    # Iterate actual layer list — avoids config/module count mismatch
    model.model.encoder.register_forward_hook(make_hook("encoder_output"))
    for i, layer in enumerate(model.model.encoder.layers):
        layer.self_attn.register_forward_hook(make_hook(f"encoder_layer{i}_attn"))
        layer.fc1.register_forward_hook(make_hook(f"encoder_layer{i}_fc1"))
        layer.fc2.register_forward_hook(make_hook(f"encoder_layer{i}_fc2"))

    for i, layer in enumerate(model.model.decoder.layers):
        layer.self_attn.register_forward_hook(make_hook(f"decoder_layer{i}_self_attn"))
        layer.encoder_attn.register_forward_hook(make_hook(f"decoder_layer{i}_cross_attn"))
        layer.fc1.register_forward_hook(make_hook(f"decoder_layer{i}_fc1"))
        layer.fc2.register_forward_hook(make_hook(f"decoder_layer{i}_fc2"))

    # ── Forward pass ──────────────────────────────────────────────────────────
    print("\nRunning forward pass...")
    with torch.no_grad():
        out = model(
            past_values=past_values,
            past_time_features=past_time_features,
            future_time_features=future_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
        )
    print("Forward pass complete.")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving outputs:")

    torch.save(out.encoder_last_hidden_state, SAVE_DIR / "encoder_last_hidden_state.pt")
    torch.save(out.loc,                       SAVE_DIR / "loc.pt")
    torch.save(out.scale,                     SAVE_DIR / "scale.pt")
    torch.save(out.static_features,           SAVE_DIR / "static_features.pt")
    print(f"  encoder_last_hidden_state: {out.encoder_last_hidden_state.shape}")
    print(f"  loc:                       {out.loc.shape}")
    print(f"  scale:                     {out.scale.shape}")
    print(f"  static_features:           {out.static_features.shape}")

    for name, tensor in captured.items():
        torch.save(tensor, SAVE_DIR / f"{name}.pt")
        print(f"  {name}: {tensor.shape}")

    # ── Save config ───────────────────────────────────────────────────────────
    config_dict = {
        "model_id":                        MODEL_ID,
        "model_revision":                  MODEL_REVISION,
        "dataset_repo":                    DATASET_REPO,
        "dataset_file":                    DATASET_FILE,
        "dataset_revision":                DATASET_REVISION,
        "context_length":                  cfg.context_length,
        "prediction_length":               cfg.prediction_length,
        "num_time_features":               cfg.num_time_features,
        "lags_sequence":                   cfg.lags_sequence,
        "d_model":                         cfg.d_model,
        "encoder_layers":                  cfg.encoder_layers,
        "decoder_layers":                  cfg.decoder_layers,
        "encoder_attention_heads":         cfg.encoder_attention_heads,
        "decoder_attention_heads":         cfg.decoder_attention_heads,
        "encoder_ffn_dim":                 cfg.encoder_ffn_dim,
        "decoder_ffn_dim":                 cfg.decoder_ffn_dim,
        "distribution_output":             cfg.distribution_output,
        "input_size":                      cfg.input_size,
        "num_static_real_features":        cfg.num_static_real_features,
        "num_static_categorical_features": cfg.num_static_categorical_features,
        "cardinality":                     cfg.cardinality,
        "embedding_dimension":             cfg.embedding_dimension,
        "batch_size":                      B,
        "past_len":                        past_len,
    }
    with open(SAVE_DIR / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nAll reference tensors saved to {SAVE_DIR}/")
    print("Use these to validate each TTNN component against PCC per-layer and 5% tolerance end-to-end.")


if __name__ == "__main__":
    main()