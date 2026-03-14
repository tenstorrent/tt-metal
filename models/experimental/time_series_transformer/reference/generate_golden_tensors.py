import torch
import os
from pathlib import Path

from transformers import TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig

MODEL_NAME = "huggingface/time-series-transformer-tourism-monthly"
GOLDEN_DIR = Path(__file__).parent.parent / "golden_tensors"
GOLDEN_DIR.mkdir(exist_ok=True)

def register_hooks(model):
    intermediates = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                intermediates[name] = output.detach().cpu()
            elif isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        intermediates[f"{name}_out{i}"] = o.detach().cpu()
            elif hasattr(output, "last_hidden_state"):
                intermediates[f"{name}_last_hidden_state"] = output.last_hidden_state.detach().cpu()
        return hook

    if hasattr(model.model, "scaler"):
        model.model.scaler.register_forward_hook(make_hook("scaler"))

    for i, layer in enumerate(model.model.encoder.layers):
        layer.register_forward_hook(make_hook(f"encoder_layer_{i}"))

    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(make_hook(f"decoder_layer_{i}"))

    model.model.encoder.register_forward_hook(make_hook("encoder_final"))
    model.model.decoder.register_forward_hook(make_hook("decoder_final"))

    if hasattr(model.model, "value_embedding"):
        model.model.value_embedding.register_forward_hook(make_hook("value_embedding"))

    if hasattr(model, "parameter_projection"):
        model.parameter_projection.register_forward_hook(make_hook("parameter_projection"))

    return intermediates


def create_dummy_input(config: TimeSeriesTransformerConfig):
    batch_size = 2
    context_length = config.context_length
    prediction_length = config.prediction_length
    num_time_features = config.num_time_features
    input_size = config.input_size

    max_lag = max(config.lags_sequence)
    past_length = context_length + max_lag 

    print(f"Creating inputs with:")
    print(f"  batch_size={batch_size}")
    print(f"  context_length={context_length}")
    print(f"  prediction_length={prediction_length}")
    print(f"  num_time_features={num_time_features}")
    print(f"  input_size={input_size}")
    print(f"  max_lag={max_lag}")
    print(f"  past_length (context + max_lag)={past_length}")

    inputs = {}

    if input_size == 1:
        inputs["past_values"] = torch.randn(batch_size, past_length)
    else:
        inputs["past_values"] = torch.randn(batch_size, past_length, input_size)

    inputs["past_time_features"] = torch.randn(batch_size, past_length, num_time_features)

    if input_size == 1:
        inputs["past_observed_mask"] = torch.ones(batch_size, past_length)
    else:
        inputs["past_observed_mask"] = torch.ones(batch_size, past_length, input_size)

    inputs["future_time_features"] = torch.randn(batch_size, prediction_length, num_time_features)

    if config.num_static_categorical_features > 0:
        inputs["static_categorical_features"] = torch.zeros(
            batch_size, config.num_static_categorical_features
        ).long()

    if config.num_static_real_features > 0:
        inputs["static_real_features"] = torch.randn(
            batch_size, config.num_static_real_features
        )

    return inputs


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_NAME)
    model.eval()

    config = model.config

    intermediates = register_hooks(model)

    print("Creating dummy inputs...")
    inputs = create_dummy_input(config)

    torch.save(inputs, GOLDEN_DIR / "inputs.pt")
    print(f"Saved inputs to {GOLDEN_DIR / 'inputs.pt'}")

    print("\nRunning forward pass (generate mode)...")
    with torch.no_grad():
        outputs = model.generate(**inputs)

    print(f"Output type: {type(outputs)}")

    if hasattr(outputs, "sequences"):
        torch.save(outputs.sequences.cpu(), GOLDEN_DIR / "generated_sequences.pt")
        print(f"  generated_sequences shape: {outputs.sequences.shape}")

    print("\nRunning forward pass (teacher-forcing mode)...")
    fwd_inputs = dict(inputs)
    if config.input_size == 1:
        fwd_inputs["future_values"] = torch.randn(2, config.prediction_length)
    else:
        fwd_inputs["future_values"] = torch.randn(2, config.prediction_length, config.input_size)

    with torch.no_grad():
        fwd_outputs = model(**fwd_inputs)

    fwd_saved = {}
    if fwd_outputs.loss is not None:
        fwd_saved["loss"] = fwd_outputs.loss.cpu()
        print(f"  loss: {fwd_outputs.loss.item():.6f}")
    if hasattr(fwd_outputs, "params") and fwd_outputs.params is not None:
        for i, p in enumerate(fwd_outputs.params):
            if isinstance(p, torch.Tensor):
                fwd_saved[f"dist_param_{i}"] = p.detach().cpu()
                print(f"  dist_param_{i} shape: {p.shape}")

    torch.save(fwd_saved, GOLDEN_DIR / "forward_outputs.pt")

    print(f"\nSaving {len(intermediates)} intermediate tensors...")
    for name, tensor in intermediates.items():
        filepath = GOLDEN_DIR / f"{name}.pt"
        torch.save(tensor, filepath)
        print(f"  {name}: {tensor.shape}")

    torch.save(config.to_dict(), GOLDEN_DIR / "model_config_dict.pt")

    print(f"\nDone! All golden tensors saved to {GOLDEN_DIR}/")
    print(f"Total files: {len(list(GOLDEN_DIR.glob('*.pt')))}")


if __name__ == "__main__":
    main()