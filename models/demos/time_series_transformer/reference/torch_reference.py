# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace reference harness: model loading, deterministic inputs, golden capture."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
DEFAULT_BATCH = 4
DEFAULT_SEED = 0


@lru_cache(maxsize=2)
def load_hf_model(model_id: str = MODEL_ID):
    """Load and freeze the reference ``TimeSeriesTransformerForPrediction``."""
    from transformers import TimeSeriesTransformerForPrediction

    model = TimeSeriesTransformerForPrediction.from_pretrained(model_id)
    model.eval()
    return model


def past_length(hf_config) -> int:
    return int(hf_config.context_length) + int(max(hf_config.lags_sequence))


def make_inputs(hf_config, *, batch: int = DEFAULT_BATCH, seed: int = DEFAULT_SEED) -> dict[str, torch.Tensor]:
    """Deterministic, dimensionally valid inputs covering every optional feature."""
    generator = torch.Generator().manual_seed(seed)
    length = past_length(hf_config)
    channels = int(getattr(hf_config, "input_size", 1) or 1)
    # HuggingFace drops the channel axis entirely when input_size == 1.
    value_shape = (batch, length) if channels == 1 else (batch, length, channels)
    inputs = {
        "past_values": torch.rand(*value_shape, generator=generator) * 100 + 50,
        "past_time_features": torch.randn(batch, length, hf_config.num_time_features, generator=generator),
        "past_observed_mask": torch.ones(*value_shape),
        "future_time_features": torch.randn(
            batch, hf_config.prediction_length, hf_config.num_time_features, generator=generator
        ),
    }
    if hf_config.num_static_categorical_features:
        inputs["static_categorical_features"] = torch.randint(
            0, int(hf_config.cardinality[0]), (batch, hf_config.num_static_categorical_features), generator=generator
        )
    if hf_config.num_static_real_features:
        inputs["static_real_features"] = torch.randn(batch, hf_config.num_static_real_features, generator=generator)
    return inputs


def make_future_values(hf_config, *, batch: int = DEFAULT_BATCH, seed: int = DEFAULT_SEED) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 1)
    return torch.rand(batch, hf_config.prediction_length, generator=generator) * 100 + 50


def embedder_weights(model) -> list[torch.Tensor]:
    """Static categorical embedding tables, in column order."""
    state = model.state_dict()
    weights = []
    index = 0
    while f"model.embedder.embedders.{index}.weight" in state:
        weights.append(state[f"model.embedder.embedders.{index}.weight"].detach().float())
        index += 1
    return weights


@dataclass
class Goldens:
    """Per-stage reference tensors for PCC comparison."""

    inputs: dict[str, torch.Tensor]
    future_values: torch.Tensor
    transformer_inputs: torch.Tensor
    loc: torch.Tensor
    scale: torch.Tensor
    static_feat: torch.Tensor
    encoder_embedding: torch.Tensor
    decoder_embedding: torch.Tensor
    encoder_hidden: list[torch.Tensor]
    decoder_hidden: list[torch.Tensor]
    encoder_last_hidden_state: torch.Tensor
    decoder_last_hidden_state: torch.Tensor
    distribution_params: list[torch.Tensor]
    distribution_mean: torch.Tensor
    nll: torch.Tensor
    generated: torch.Tensor
    context_length: int

    @property
    def encoder_input(self) -> torch.Tensor:
        return self.transformer_inputs[:, : self.context_length, ...]

    @property
    def decoder_input(self) -> torch.Tensor:
        return self.transformer_inputs[:, self.context_length :, ...]


@lru_cache(maxsize=2)
def capture_goldens(
    model_id: str = MODEL_ID,
    batch: int = DEFAULT_BATCH,
    seed: int = DEFAULT_SEED,
) -> Goldens:
    """Run the reference model once and collect every intermediate the TTNN tests need."""
    model = load_hf_model(model_id)
    config = model.config
    inputs = make_inputs(config, batch=batch, seed=seed)
    future_values = make_future_values(config, batch=batch, seed=seed)

    with torch.no_grad():
        transformer_inputs, loc, scale, static_feat = model.model.create_network_inputs(
            future_values=future_values, **inputs
        )
        context = int(config.context_length)
        encoder_embedding = model.model.encoder.value_embedding(transformer_inputs[:, :context, ...])
        decoder_embedding = model.model.decoder.value_embedding(transformer_inputs[:, context:, ...])

        outputs = model.model(
            future_values=future_values,
            output_hidden_states=True,
            output_attentions=True,
            **inputs,
        )
        params = list(model.parameter_projection(outputs.last_hidden_state))
        distribution = model.output_distribution(params, loc=loc, scale=scale)

        torch.manual_seed(seed)
        generated = model.generate(**inputs).sequences

    goldens = Goldens(
        inputs=inputs,
        future_values=future_values,
        transformer_inputs=transformer_inputs,
        loc=loc,
        scale=scale,
        static_feat=static_feat,
        encoder_embedding=encoder_embedding,
        decoder_embedding=decoder_embedding,
        encoder_hidden=list(outputs.encoder_hidden_states),
        decoder_hidden=list(outputs.decoder_hidden_states),
        encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        decoder_last_hidden_state=outputs.last_hidden_state,
        distribution_params=params,
        distribution_mean=distribution.mean,
        nll=-distribution.log_prob(future_values),
        generated=generated,
        context_length=int(config.context_length),
    )
    return goldens


def build_reference_model(distribution_output: str, *, seed: int = 0, input_size: int = 1, model_id: str = MODEL_ID):
    """A randomly-initialised HF model with a chosen distribution head.

    The published checkpoint only carries a Student's t head, so parity for the Normal and
    Negative Binomial heads has to be established against a reference built for the purpose.
    Geometry is copied from the real checkpoint so the shapes stay representative.
    """
    from transformers import TimeSeriesTransformerConfig as HFConfig
    from transformers import TimeSeriesTransformerForPrediction

    template = load_hf_model(model_id).config
    config = HFConfig(
        prediction_length=int(template.prediction_length),
        context_length=int(template.context_length),
        lags_sequence=list(template.lags_sequence),
        num_time_features=int(template.num_time_features),
        num_static_categorical_features=int(template.num_static_categorical_features),
        cardinality=list(template.cardinality),
        embedding_dimension=list(template.embedding_dimension),
        num_static_real_features=int(template.num_static_real_features),
        d_model=int(template.d_model),
        encoder_layers=int(template.encoder_layers),
        decoder_layers=int(template.decoder_layers),
        encoder_attention_heads=int(template.encoder_attention_heads),
        decoder_attention_heads=int(template.decoder_attention_heads),
        encoder_ffn_dim=int(template.encoder_ffn_dim),
        decoder_ffn_dim=int(template.decoder_ffn_dim),
        distribution_output=distribution_output,
        input_size=input_size,
        scaling=template.scaling,
        dropout=0.0,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    torch.manual_seed(seed)
    return TimeSeriesTransformerForPrediction(config).eval()


def generate_mean_reference(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Reference autoregressive rollout that takes the distribution mean at each step.

    HuggingFace's ``generate`` only samples, so comparing our deterministic mean-mode output
    against it would measure Monte Carlo noise rather than model error. This mirrors the same
    loop with ``distr.mean`` substituted for ``distr.sample()``.
    """
    config = model.config
    with torch.no_grad():
        transformer_inputs, loc, scale, static_feat = model.model.create_network_inputs(**inputs)
        encoder_hidden = model.model.encoder(
            inputs_embeds=transformer_inputs[:, : config.context_length, ...]
        ).last_hidden_state

        future_time_features = inputs["future_time_features"]
        expanded_static = static_feat.unsqueeze(1).expand(-1, future_time_features.shape[1], -1)
        features = torch.cat((expanded_static, future_time_features), dim=-1)

        running = (inputs["past_values"] - loc) / scale
        collected = []
        for step in range(config.prediction_length):
            lagged = model.model.get_lagged_subsequences(sequence=running, subsequences_length=1 + step, shift=1)
            reshaped = lagged.reshape(lagged.shape[0], lagged.shape[1], -1)
            decoder_input = torch.cat((reshaped, features[:, : step + 1]), dim=-1)
            decoder_hidden = model.model.decoder(
                inputs_embeds=decoder_input, encoder_hidden_states=encoder_hidden
            ).last_hidden_state

            params = model.parameter_projection(decoder_hidden[:, -1:])
            next_value = model.output_distribution(params, loc=loc, scale=scale).mean
            while next_value.dim() < running.dim():
                next_value = next_value.unsqueeze(1)
            running = torch.cat((running, (next_value - loc) / scale), dim=1)
            collected.append(next_value)

    return torch.cat(collected, dim=1)


# -- metrics ---------------------------------------------------------------


def compute_pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    """Pearson correlation over flattened tensors; 1.0 for exact matches."""
    a = expected.detach().float().flatten()
    b = actual.detach().float().flatten()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if torch.allclose(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.norm() * b.norm()
    if denominator == 0:
        return 0.0
    return float((a @ b) / denominator)


def compute_metrics(expected: torch.Tensor, actual: torch.Tensor) -> tuple[float, float, float]:
    """Return ``(mse, mae, pcc)``."""
    expected = expected.detach().float()
    actual = actual.detach().float()
    mse = float(torch.mean((expected - actual) ** 2))
    mae = float(torch.mean(torch.abs(expected - actual)))
    return mse, mae, compute_pcc(expected, actual)


def crps(samples: torch.Tensor, target: torch.Tensor, *, max_pairs: int = 64) -> float:
    """Sample-based CRPS estimator ``E|X - y| - 0.5 E|X - X'|``.

    ``samples`` is ``(batch, num_samples, prediction_length)``. The pairwise term is capped
    at ``max_pairs`` samples so the O(n^2) part stays bounded for 100+ sample runs.
    """
    samples = samples.detach().float()
    target = target.detach().float().unsqueeze(1)
    term_observation = torch.mean(torch.abs(samples - target))

    capped = samples[:, :max_pairs, :]
    diffs = torch.abs(capped.unsqueeze(1) - capped.unsqueeze(2))
    term_spread = torch.mean(diffs)
    return float(term_observation - 0.5 * term_spread)


def relative_error(reference: float, candidate: float) -> float:
    """Absolute relative difference, guarded against a zero reference."""
    return abs(candidate - reference) / max(abs(reference), 1e-8)


__all__ = [
    "DEFAULT_BATCH",
    "DEFAULT_SEED",
    "Goldens",
    "MODEL_ID",
    "build_reference_model",
    "capture_goldens",
    "compute_metrics",
    "compute_pcc",
    "crps",
    "embedder_weights",
    "generate_mean_reference",
    "load_hf_model",
    "make_future_values",
    "make_inputs",
    "past_length",
    "relative_error",
]
