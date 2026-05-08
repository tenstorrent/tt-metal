"""
Save reference tensors from HuggingFace TimeSeriesTransformer.
These are used to validate the TTNN port component by component.
"""

import torch
import os
import json
from transformers import TimeSeriesTransformerForPrediction

SAVE_DIR = "models/demos/time_series_transformer/reference_tensors"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading HuggingFace reference model...")
model = TimeSeriesTransformerForPrediction.from_pretrained(
    "huggingface/time-series-transformer-tourism-monthly"
)
model.eval()
cfg = model.config

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

# ── Build inputs ──────────────────────────────────────────────────────────────
torch.manual_seed(42)
B        = 2
past_len = model.model._past_length  # 61 = context_length + max(lags)

print(f"\n  past_len: {past_len}")
print(f"  batch_size: {B}")

past_values                 = torch.randn(B, past_len, cfg.input_size)
past_time_features          = torch.randn(B, past_len, cfg.num_time_features)
future_time_features        = torch.randn(B, cfg.prediction_length, cfg.num_time_features)
past_observed_mask          = torch.ones(B,  past_len, cfg.input_size)
static_categorical_features = torch.zeros(B, 1, dtype=torch.long)
static_real_features        = torch.zeros(B, cfg.num_static_real_features)

# Save inputs
torch.save(past_values,                 f"{SAVE_DIR}/input_past_values.pt")
torch.save(past_time_features,          f"{SAVE_DIR}/input_past_time_features.pt")
torch.save(future_time_features,        f"{SAVE_DIR}/input_future_time_features.pt")
torch.save(past_observed_mask,          f"{SAVE_DIR}/input_past_observed_mask.pt")
torch.save(static_categorical_features, f"{SAVE_DIR}/input_static_categorical.pt")
torch.save(static_real_features,        f"{SAVE_DIR}/input_static_real.pt")
print("\nInputs saved.")

# ── Register hooks ────────────────────────────────────────────────────────────
captured = {}

def make_hook(name):
    def fn(module, input, output):
        if isinstance(output, torch.Tensor):
            captured[name] = output.detach()
        elif isinstance(output, tuple):
            captured[name] = output[0].detach()
        else:
            # BaseModelOutput or similar — extract last_hidden_state
            captured[name] = output.last_hidden_state.detach()
    return fn

model.model.encoder.register_forward_hook(make_hook("encoder_output"))
model.model.encoder.layers[0].self_attn.register_forward_hook(make_hook("encoder_layer0_attn"))
model.model.encoder.layers[0].fc1.register_forward_hook(make_hook("encoder_layer0_fc1"))
model.model.encoder.layers[0].fc2.register_forward_hook(make_hook("encoder_layer0_fc2"))
model.model.encoder.layers[1].self_attn.register_forward_hook(make_hook("encoder_layer1_attn"))
model.model.decoder.layers[0].self_attn.register_forward_hook(make_hook("decoder_layer0_self_attn"))
model.model.decoder.layers[0].encoder_attn.register_forward_hook(make_hook("decoder_layer0_cross_attn"))
model.model.decoder.layers[0].fc1.register_forward_hook(make_hook("decoder_layer0_fc1"))
model.model.decoder.layers[0].fc2.register_forward_hook(make_hook("decoder_layer0_fc2"))

# ── Forward pass ──────────────────────────────────────────────────────────────
print("Running forward pass...")
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

# ── Save outputs ──────────────────────────────────────────────────────────────
print("\nSaving outputs:")

# Main model outputs
torch.save(out.encoder_last_hidden_state, f"{SAVE_DIR}/encoder_last_hidden_state.pt")
torch.save(out.loc,                       f"{SAVE_DIR}/loc.pt")
torch.save(out.scale,                     f"{SAVE_DIR}/scale.pt")
torch.save(out.static_features,           f"{SAVE_DIR}/static_features.pt")
print(f"  encoder_last_hidden_state: {out.encoder_last_hidden_state.shape}")
print(f"  loc:                       {out.loc.shape}")
print(f"  scale:                     {out.scale.shape}")
print(f"  static_features:           {out.static_features.shape}")

# Intermediate captured outputs
for name, tensor in captured.items():
    torch.save(tensor, f"{SAVE_DIR}/{name}.pt")
    print(f"  {name}: {tensor.shape}")

# ── Save config ───────────────────────────────────────────────────────────────
config_dict = {
    "context_length":              cfg.context_length,
    "prediction_length":           cfg.prediction_length,
    "num_time_features":           cfg.num_time_features,
    "lags_sequence":               cfg.lags_sequence,
    "d_model":                     cfg.d_model,
    "encoder_layers":              cfg.encoder_layers,
    "decoder_layers":              cfg.decoder_layers,
    "encoder_attention_heads":     cfg.encoder_attention_heads,
    "decoder_attention_heads":     cfg.decoder_attention_heads,
    "encoder_ffn_dim":             cfg.encoder_ffn_dim,
    "decoder_ffn_dim":             cfg.decoder_ffn_dim,
    "distribution_output":         cfg.distribution_output,
    "input_size":                  cfg.input_size,
    "num_static_real_features":    cfg.num_static_real_features,
    "num_static_categorical_features": cfg.num_static_categorical_features,
    "cardinality":                 cfg.cardinality,
    "embedding_dimension":         cfg.embedding_dimension,
    "batch_size":                  B,
    "past_len":                    past_len,
}
with open(f"{SAVE_DIR}/config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

print(f"\nAll reference tensors saved to {SAVE_DIR}/")
print("Use these to validate each TTNN component against the 5% tolerance requirement.")