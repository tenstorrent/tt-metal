# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Save reference tensors from HuggingFace TimeSeriesTransformer.
These are used to validate the TTNN port component by component.

Data source: real Tourism Monthly batch from hf-internal-testing/tourism-monthly-batch
Model: huggingface/time-series-transformer-tourism-monthly (pinned revision)
"""

import sys
import torch
import json
import transformers
import huggingface_hub
from pathlib import Path
from transformers import TimeSeriesTransformerForPrediction
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file as save_safetensors

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID          = "huggingface/time-series-transformer-tourism-monthly"
MODEL_REVISION    = "2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35"

DATASET_REPO      = "hf-internal-testing/tourism-monthly-batch"
DATASET_FILE      = "train-batch.pt"
DATASET_REVISION  = "81c7ee3cf3317e51beb97327df55926cd5bbfadb"

SAVE_DIR = Path(__file__).resolve().parent.parent / "reference"


def _remove_stale_reference_tensors(save_dir: Path) -> None:
    """Delete known script outputs to avoid mixing artifacts across runs."""
    for name in (
        "inputs.safetensors",
        "outputs.safetensors",
        "intermediates.safetensors",
        "config_runtime.json",
    ):
        target = save_dir / name
        if target.exists():
            target.unlink()


def _write_or_assert_config(save_dir: Path, config_dict: dict) -> None:
    """Write config.json if it doesn't exist; assert it matches if it does."""
    config_path = (save_dir / "config.json").resolve()
    expected_parent = save_dir.resolve()

    # Use relative_to for reliable path containment check (avoids prefix collisions)
    try:
        config_path.relative_to(expected_parent)
    except ValueError:
        raise ValueError(f"Unsafe path detected: {config_path} is outside {expected_parent}")

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing != config_dict:
            raise RuntimeError(
                "Generated config does not match committed reference/config.json. "
                "If this is intentional, delete reference/config.json and re-run."
            )
        print("  config.json unchanged — matches committed provenance.")
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        print("  config.json written.")


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    _remove_stale_reference_tensors(SAVE_DIR)

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
    # Require weights_only=True for safe deserialization — fail fast otherwise
    try:
        batch = torch.load(file, map_location="cpu", weights_only=True)
    except TypeError as e:
        raise RuntimeError(
            f"torch {torch.__version__} does not support weights_only=True. "
            "Please upgrade to torch>=2.1.0 for this demo's safe deserialization path."
        ) from e

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
    if actual_past_len != expected_past_len:
        raise ValueError(
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

    # ── Save inputs (safetensors format for safe cross-machine sharing) ───────
    save_safetensors(
        {
            "input_past_values":                 past_values.contiguous(),
            "input_past_time_features":          past_time_features.contiguous(),
            "input_future_time_features":        future_time_features.contiguous(),
            "input_past_observed_mask":          past_observed_mask.contiguous(),
            "input_static_categorical_features": static_categorical_features.contiguous(),
            "input_static_real_features":        static_real_features.contiguous(),
        },
        str(SAVE_DIR / "inputs.safetensors"),
    )
    print("\nInputs saved.")

    # ── Register hooks ────────────────────────────────────────────────────────
    captured = {}

    def make_hook(name):
        # 'inputs' avoids shadowing Python's built-in input()
        def fn(module, inputs, output):
            if isinstance(output, torch.Tensor):
                captured[name] = output.detach().cpu()
            elif isinstance(output, tuple):
                captured[name] = output[0].detach().cpu()
            else:
                captured[name] = output.last_hidden_state.detach().cpu()
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

    # ── Save outputs (safetensors format) ─────────────────────────────────────
    print("\nSaving outputs:")

    main_outputs = {
        "encoder_last_hidden_state": out.encoder_last_hidden_state.cpu().contiguous(),
        "loc":                       out.loc.cpu().contiguous(),
        "scale":                     out.scale.cpu().contiguous(),
        "static_features":           out.static_features.cpu().contiguous(),
    }
    print(f"  encoder_last_hidden_state: {out.encoder_last_hidden_state.shape}")
    print(f"  loc:                       {out.loc.shape}")
    print(f"  scale:                     {out.scale.shape}")
    print(f"  static_features:           {out.static_features.shape}")

    captured_contiguous = {}
    for name in sorted(captured.keys()):
        tensor = captured[name]
        captured_contiguous[name] = tensor.contiguous()
        print(f"  {name}: {tensor.shape}")

    save_safetensors(main_outputs,        str(SAVE_DIR / "outputs.safetensors"))
    save_safetensors(captured_contiguous, str(SAVE_DIR / "intermediates.safetensors"))

    # ── Write or assert static provenance config ──────────────────────────────
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
        "past_len":                        past_len,
    }
    _write_or_assert_config(SAVE_DIR, config_dict)

    # ── Save runtime environment (gitignored, not committed) ──────────────────
    config_runtime = {
        "batch_size":              B,
        "python_version":          sys.version,
        "torch_version":           torch.__version__,
        "transformers_version":    transformers.__version__,
        "huggingface_hub_version": huggingface_hub.__version__,
    }
    with open(SAVE_DIR / "config_runtime.json", "w", encoding="utf-8") as f:
        json.dump(config_runtime, f, indent=2)

    print(f"\nAll reference tensors saved to {SAVE_DIR}/")
    print("Static provenance:   reference/config.json (committed)")
    print("Runtime environment: reference/config_runtime.json (gitignored)")
    print("Use these to validate each TTNN component against PCC per-layer and 5% tolerance end-to-end.")


if __name__ == "__main__":
    main()