# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partialmethod
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open

import ttnn

from .config import TILE_SIZE
from .hf_common import (
    HfEmbedding,
    ParameterProjection,
    torch_mean_scaler,
    torch_negative_binomial_domain_map,
    torch_normal_domain_map,
    torch_student_t_domain_map,
)
from .ops import make_causal_mask


def substate(state: Mapping[str, torch.Tensor], key: str) -> dict[str, torch.Tensor]:
    prefix = f"{key}."
    plen = len(prefix)
    return {k[plen:]: v for k, v in state.items() if k.startswith(prefix)}


def pick_prefixed(
    state: Mapping[str, torch.Tensor],
    *,
    prefix: str = "",
    allowed_roots: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    base = substate(state, prefix) if prefix else dict(state)
    return {k: v for k, v in base.items() if k.split(".", 1)[0] in allowed_roots}


def resolve_checkpoint_dir(ckpt_dir_or_repo: str) -> Path:
    path = Path(ckpt_dir_or_repo)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    resolved = snapshot_download(
        ckpt_dir_or_repo,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "model.safetensors.index.json"],
    )
    return Path(resolved)


def load_hf_checkpoint_state(
    ckpt_dir_or_repo: str,
    *,
    key_prefixes: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    ckpt_dir = resolve_checkpoint_dir(ckpt_dir_or_repo)
    prefixes = tuple(key_prefixes or ())

    index_path = ckpt_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map: dict[str, str] = index_data["weight_map"]
        file_to_keys: dict[str, list[str]] = {}
        for key, filename in weight_map.items():
            if prefixes and not key.startswith(prefixes):
                continue
            file_to_keys.setdefault(filename, []).append(key)
        state: dict[str, torch.Tensor] = {}
        for filename, keys in file_to_keys.items():
            with safe_open(str(ckpt_dir / filename), framework="pt", device="cpu") as handle:
                for key in keys:
                    state[key] = handle.get_tensor(key)
        return state

    single_file = ckpt_dir / "model.safetensors"
    if not single_file.is_file():
        raise FileNotFoundError(f"Missing checkpoint in {ckpt_dir}: expected model.safetensors(.index.json)")
    with safe_open(str(single_file), framework="pt", device="cpu") as handle:
        keys = [k for k in handle.keys() if not prefixes or k.startswith(prefixes)]
        return {k: handle.get_tensor(k) for k in keys}


@dataclass(frozen=True)
class PrefixedModuleLoadSpec:
    """Descriptor for loading a scoped state slice into one module-owned loader."""

    module: object
    prefix: str
    load_method: str
    allowed_roots: Optional[tuple[str, ...]] = None


def _prefix_key(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _slice_state(
    state: dict[str, torch.Tensor],
    *,
    prefix: str,
    allowed_roots: Optional[tuple[str, ...]],
) -> dict[str, torch.Tensor]:
    if allowed_roots is None:
        return substate(state, prefix) if prefix else dict(state)
    return pick_prefixed(state, prefix=prefix, allowed_roots=allowed_roots)


def apply_prefixed_module_loads(
    state: dict[str, torch.Tensor],
    specs: list[PrefixedModuleLoadSpec],
    *,
    strict: bool,
    initial_used_keys: Optional[set[str]] = None,
) -> dict[str, list[str]]:
    """
    Apply module-owned loaders over scoped state slices and merge load results.

    Each module receives only its prefixed slice and owns its own key mapping logic.
    """
    used: set[str] = set() if initial_used_keys is None else set(initial_used_keys)
    missing: list[str] = []
    unexpected: list[str] = []

    for spec in specs:
        module_state = _slice_state(state, prefix=spec.prefix, allowed_roots=spec.allowed_roots)
        for key in module_state:
            used.add(_prefix_key(spec.prefix, key))

        loader = getattr(spec.module, spec.load_method)
        result = loader(module_state, strict=False)
        missing.extend(_prefix_key(spec.prefix, key) for key in result.get("missing_keys", []))
        unexpected.extend(_prefix_key(spec.prefix, key) for key in result.get("unexpected_keys", []))

    unexpected = sorted(set(unexpected) | (set(state.keys()) - used))
    if strict and missing:
        raise ValueError(f"State dict mismatch. Missing: {missing}.")
    return {"missing_keys": missing, "unexpected_keys": unexpected}


class InformerHfRuntimeMixin:
    """HuggingFace-compatible runtime, loading, preprocessing, and generation utilities."""

    def _ensure_hf_runtime(self) -> None:
        cfg = self.config
        if not cfg.hf_compat:
            raise ValueError("HuggingFace helpers require config.hf_compat=True.")
        if self.hf_encoder_embedding is not None:
            return
        if cfg.feature_size is None:
            raise ValueError("feature_size must be set for HF compatibility.")

        max_len = int(cfg.context_length or cfg.seq_len) + int(cfg.prediction_length or cfg.pred_len)
        memory_config = ttnn.L1_MEMORY_CONFIG if cfg.use_l1 else None
        compute_dtype = ttnn.float32 if cfg.hf_compat else self.dtype
        weight_dtype = compute_dtype if cfg.hf_compat else self.dtype

        self.hf_encoder_embedding = HfEmbedding(
            cfg.feature_size,
            cfg.d_model,
            self.rng,
            device=self.device,
            dtype=self.dtype,
            weight_dtype=weight_dtype,
            compute_dtype=compute_dtype,
            max_len=max_len,
            dropout=cfg.dropout,
            memory_config=memory_config,
        )
        self.hf_decoder_embedding = HfEmbedding(
            cfg.feature_size,
            cfg.d_model,
            self.rng,
            device=self.device,
            dtype=self.dtype,
            weight_dtype=weight_dtype,
            compute_dtype=compute_dtype,
            max_len=max_len,
            dropout=cfg.dropout,
            memory_config=memory_config,
        )

        self.hf_embedder_weights = []
        if cfg.num_static_categorical_features and cfg.cardinality and cfg.embedding_dimension:
            for card, dim in zip(cfg.cardinality, cfg.embedding_dimension):
                weight = torch.randn((card, dim), generator=self.rng, dtype=torch.float32) * 0.02
                self.hf_embedder_weights.append(weight)

        out_dim = int(cfg.input_size or 1)
        dist_name = cfg.distribution_output.lower()
        if dist_name == "student_t":
            num_params = 3
        elif dist_name in ("normal", "negative_binomial"):
            num_params = 2
        else:
            raise ValueError(f"Unsupported distribution_output: {cfg.distribution_output}")

        self.hf_parameter_projection = ParameterProjection(
            cfg.d_model,
            out_dim,
            num_params,
            self.rng,
            device=self.device,
            dtype=self.dtype,
            compute_dtype=compute_dtype,
            weight_dtype=weight_dtype,
            memory_config=memory_config,
        )

    def _load_static_embedder_weights(self, state: dict[str, torch.Tensor]) -> set[str]:
        used: set[str] = set()
        if not self.config.num_static_categorical_features:
            return used

        for index in range(self.config.num_static_categorical_features):
            key = f"model.embedder.embedders.{index}.weight"
            weight = state.get(key)
            if weight is None:
                continue
            used.add(key)
            if index >= len(self.hf_embedder_weights):
                self.hf_embedder_weights.append(weight.detach().float())
            else:
                self.hf_embedder_weights[index] = weight.detach().float()
        return used

    def _hf_module_load_specs(self) -> list[PrefixedModuleLoadSpec]:
        assert self.hf_encoder_embedding is not None
        assert self.hf_decoder_embedding is not None
        assert self.hf_parameter_projection is not None

        specs: list[PrefixedModuleLoadSpec] = [
            PrefixedModuleLoadSpec(
                module=self.hf_encoder_embedding,
                prefix="model.encoder",
                allowed_roots=("value_embedding", "embed_positions", "layernorm_embedding"),
                load_method="load_hf_state_dict",
            ),
            PrefixedModuleLoadSpec(
                module=self.hf_decoder_embedding,
                prefix="model.decoder",
                allowed_roots=("value_embedding", "embed_positions", "layernorm_embedding"),
                load_method="load_hf_state_dict",
            ),
        ]

        for index, layer in enumerate(self.encoder.layers):
            layer_prefix = f"model.encoder.layers.{index}"
            specs.extend(
                [
                    PrefixedModuleLoadSpec(
                        module=layer.attn,
                        prefix=f"{layer_prefix}.self_attn",
                        allowed_roots=("q_proj", "k_proj", "v_proj", "out_proj"),
                        load_method="load_hf_state_dict",
                    ),
                    PrefixedModuleLoadSpec(
                        module=layer.ffn,
                        prefix=layer_prefix,
                        allowed_roots=("fc1", "fc2"),
                        load_method="load_hf_state_dict",
                    ),
                    PrefixedModuleLoadSpec(
                        module=layer.norm1,
                        prefix=f"{layer_prefix}.self_attn_layer_norm",
                        allowed_roots=("weight", "bias"),
                        load_method="load_hf_state_dict",
                    ),
                    PrefixedModuleLoadSpec(
                        module=layer.norm2,
                        prefix=f"{layer_prefix}.final_layer_norm",
                        allowed_roots=("weight", "bias"),
                        load_method="load_hf_state_dict",
                    ),
                ]
            )

        if self.encoder.conv_layers is not None:
            for index, conv in enumerate(self.encoder.conv_layers):
                if conv is None:
                    continue
                specs.append(
                    PrefixedModuleLoadSpec(
                        module=conv,
                        prefix=f"model.encoder.conv_layers.{index}",
                        allowed_roots=("downConv", "norm"),
                        load_method="load_hf_state_dict",
                    )
                )

        for index, layer in enumerate(self.decoder.layers):
            layer_prefix = f"model.decoder.layers.{index}"
            specs.extend(
                [
                    PrefixedModuleLoadSpec(
                        module=layer.self_attn,
                        prefix=f"{layer_prefix}.self_attn",
                        allowed_roots=("q_proj", "k_proj", "v_proj", "out_proj"),
                        load_method="load_hf_state_dict",
                    ),
                    PrefixedModuleLoadSpec(
                        module=layer.cross_attn,
                        prefix=f"{layer_prefix}.encoder_attn",
                        allowed_roots=("q_proj", "k_proj", "v_proj", "out_proj"),
                        load_method="load_hf_state_dict",
                    ),
                    PrefixedModuleLoadSpec(
                        module=layer.ffn,
                        prefix=layer_prefix,
                        allowed_roots=("fc1", "fc2"),
                        load_method="load_hf_state_dict",
                    ),
                    PrefixedModuleLoadSpec(
                        module=layer.norm1,
                        prefix=f"{layer_prefix}.self_attn_layer_norm",
                        allowed_roots=("weight", "bias"),
                        load_method="load_hf_state_dict",
                    ),
                    PrefixedModuleLoadSpec(
                        module=layer.norm2,
                        prefix=f"{layer_prefix}.encoder_attn_layer_norm",
                        allowed_roots=("weight", "bias"),
                        load_method="load_hf_state_dict",
                    ),
                    PrefixedModuleLoadSpec(
                        module=layer.norm3,
                        prefix=f"{layer_prefix}.final_layer_norm",
                        allowed_roots=("weight", "bias"),
                        load_method="load_hf_state_dict",
                    ),
                ]
            )

        specs.append(
            PrefixedModuleLoadSpec(
                module=self.hf_parameter_projection,
                prefix="parameter_projection",
                allowed_roots=("proj",),
                load_method="load_hf_state_dict",
            )
        )
        return specs

    def load_hf_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
        self._ensure_hf_runtime()
        used = self._load_static_embedder_weights(state)
        return apply_prefixed_module_loads(
            state,
            self._hf_module_load_specs(),
            strict=strict,
            initial_used_keys=used,
        )

    def load_hf_checkpoint(
        self,
        ckpt_dir_or_repo: str,
        *,
        strict: bool = True,
        key_prefixes: tuple[str, ...] | None = None,
    ) -> dict[str, list[str]]:
        state = load_hf_checkpoint_state(ckpt_dir_or_repo, key_prefixes=key_prefixes)
        return self.load_hf_state_dict(state, strict=strict)

    @staticmethod
    def maybe_to_torch(x: torch.Tensor | ttnn.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        if isinstance(x, ttnn.Tensor):
            return ttnn.to_torch(x)
        return x

    def hf_embed_static(self, static_categorical_features: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if static_categorical_features is None or not self.hf_embedder_weights:
            return None
        if static_categorical_features.dim() == 1:
            static_categorical_features = static_categorical_features.unsqueeze(-1)
        embeds = []
        for idx, weight in enumerate(self.hf_embedder_weights):
            feats = static_categorical_features[:, idx]
            embeds.append(torch.nn.functional.embedding(feats, weight))
        return torch.cat(embeds, dim=-1)

    def hf_past_length(self) -> int:
        if not self.config.lags_sequence:
            return int(self.config.context_length or self.config.seq_len)
        return int(self.config.context_length or self.config.seq_len) + int(max(self.config.lags_sequence))

    def hf_get_lagged_subsequences(
        self, sequence: torch.Tensor, *, subsequences_length: int, shift: int = 0
    ) -> torch.Tensor:
        sequence_length = sequence.shape[1]
        indices = [lag - shift for lag in self.config.lags_sequence]
        if max(indices) + subsequences_length > sequence_length:
            raise ValueError(
                f"lags cannot go further than history length, found lag {max(indices)} while history length is only {sequence_length}"
            )
        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    def hf_prepare_inputs(
        self,
        past_values: torch.Tensor | ttnn.Tensor,
        past_time_features: torch.Tensor | ttnn.Tensor,
        *,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build HF-compatible transformer inputs plus normalization statistics."""
        self._ensure_hf_runtime()
        (
            past_values,
            past_time_features,
            future_values,
            future_time_features,
            static_categorical_features,
            static_real_features,
            past_observed_mask,
        ) = (
            self.maybe_to_torch(past_values),
            self.maybe_to_torch(past_time_features),
            self.maybe_to_torch(future_values),
            self.maybe_to_torch(future_time_features),
            self.maybe_to_torch(static_categorical_features),
            self.maybe_to_torch(static_real_features),
            self.maybe_to_torch(past_observed_mask),
        )
        assert past_values is not None
        assert past_time_features is not None

        if past_values.dim() == 2:
            past_values = past_values.unsqueeze(-1)
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)
        elif past_observed_mask.dim() == 2:
            past_observed_mask = past_observed_mask.unsqueeze(-1)

        past_length = self.hf_past_length()
        context_length = int(self.config.context_length or self.config.seq_len)
        time_feat_start = past_length - context_length
        time_feat = past_time_features[:, time_feat_start:, ...]
        if future_values is not None and future_time_features is not None:
            time_feat = torch.cat((time_feat, future_time_features), dim=1)

        context = past_values[:, -context_length:]
        observed_context = past_observed_mask[:, -context_length:]

        if self.config.scaling != "mean":
            raise ValueError(f"Unsupported scaler: {self.config.scaling}")

        _, loc, scale = torch_mean_scaler(
            context,
            observed_context,
            minimum_scale=self.config.minimum_scale,
            default_scale=self.config.default_scale,
        )

        if future_values is not None:
            inputs = (torch.cat((past_values, future_values), dim=1) - loc) / scale
        else:
            inputs = (past_values - loc) / scale

        if loc.ndim == 3:
            squeezed_loc = loc.squeeze(1)
            squeezed_scale = scale.squeeze(1)
        else:
            squeezed_loc = loc
            squeezed_scale = scale
        log_abs_loc = squeezed_loc.abs().log1p()
        log_scale = squeezed_scale.log()
        static_feat = torch.cat((log_abs_loc, log_scale), dim=1)

        if static_real_features is not None:
            static_feat = torch.cat((static_real_features, static_feat), dim=1)
        embed_cat = self.hf_embed_static(static_categorical_features)
        if embed_cat is not None:
            static_feat = torch.cat((embed_cat, static_feat), dim=1)

        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, time_feat.shape[1], -1)
        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        subseq_length = context_length + int(self.config.prediction_length or self.config.pred_len)
        if future_values is None:
            subseq_length = context_length
        lagged_sequence = self.hf_get_lagged_subsequences(sequence=inputs, subsequences_length=subseq_length)
        lags_shape = lagged_sequence.shape
        reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)
        if reshaped_lagged_sequence.shape[1] != time_feat.shape[1]:
            raise ValueError(
                f"input length {reshaped_lagged_sequence.shape[1]} and time feature lengths {time_feat.shape[1]} does not match"
            )
        transformer_inputs = torch.cat((reshaped_lagged_sequence, features), dim=-1)
        return transformer_inputs, loc, scale, static_feat

    def hf_encode(self, transformer_inputs: torch.Tensor) -> tuple[ttnn.Tensor, int]:
        self._ensure_hf_runtime()
        assert self.hf_encoder_embedding is not None
        enc_input = ttnn.from_torch(transformer_inputs, device=self.device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        enc_embed = self.hf_encoder_embedding(enc_input, position_offset=0)
        return self.encoder(enc_embed, None)

    def hf_decode(self, decoder_inputs: torch.Tensor, enc_out: ttnn.Tensor, enc_valid_len: int) -> ttnn.Tensor:
        self._ensure_hf_runtime()
        assert self.hf_decoder_embedding is not None
        dec_input = ttnn.from_torch(decoder_inputs, device=self.device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        dec_embed = self.hf_decoder_embedding(dec_input, position_offset=0)
        dec_len = dec_embed.shape[1]
        pad_len = int(math.ceil(dec_len / TILE_SIZE)) * TILE_SIZE
        self_mask = make_causal_mask(
            pad_len,
            batch=dec_embed.shape[0],
            heads=1,
            device=self.device,
            dtype=self.dtype,
            mask_value=self.config.attn_mask_value,
        )
        return self.decoder(dec_embed, enc_out, self_mask, None, enc_valid_len)

    def build_output_distribution(self, params_torch: list[torch.Tensor]) -> torch.distributions.Distribution:
        dist_name = self.config.distribution_output.lower()
        if dist_name == "student_t":
            if len(params_torch) < 3:
                raise ValueError("StudentT output expects 3 parameters (df, loc, scale).")
            df, loc_param, scale_param = torch_student_t_domain_map(
                params_torch[0],
                params_torch[1],
                params_torch[2],
            )
            return torch.distributions.StudentT(df, loc_param, scale_param)
        if dist_name == "normal":
            if len(params_torch) < 2:
                raise ValueError("Normal output expects 2 parameters (loc, scale).")
            loc_param, scale_param = torch_normal_domain_map(
                params_torch[0],
                params_torch[1],
                minimum_scale=self.config.minimum_scale,
            )
            return torch.distributions.Normal(loc_param, scale_param)
        if dist_name == "negative_binomial":
            if len(params_torch) < 2:
                raise ValueError("Negative binomial output expects 2 parameters (mu, alpha).")
            total_count, logits = torch_negative_binomial_domain_map(
                params_torch[0],
                params_torch[1],
                minimum_scale=self.config.minimum_scale,
            )
            return torch.distributions.NegativeBinomial(total_count=total_count, logits=logits)
        raise ValueError(f"Unsupported distribution_output: {self.config.distribution_output}")

    def hf_generate_with_mode(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        *,
        past_observed_mask: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        num_parallel_samples: int = 1,
        mode: str,
    ) -> torch.Tensor:
        """Run autoregressive HF-style generation using either sampling or mean decoding."""
        self._ensure_hf_runtime()
        assert self.hf_parameter_projection is not None
        out_dim = int(self.hf_parameter_projection.out_dim)
        if past_values.dim() == 2:
            past_values = past_values.unsqueeze(-1)

        transformer_inputs, loc, scale, static_feat = self.hf_prepare_inputs(
            past_values=past_values,
            past_time_features=past_time_features,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            past_observed_mask=past_observed_mask,
            future_values=None,
            future_time_features=future_time_features,
        )
        context_length = int(self.config.context_length or self.config.seq_len)
        enc_input = transformer_inputs[:, :context_length, ...]
        enc_out, enc_valid_len = self.hf_encode(enc_input)

        if context_length >= transformer_inputs.shape[1]:
            zero_dec = torch.zeros(
                (transformer_inputs.shape[0], 1, transformer_inputs.shape[2]),
                dtype=transformer_inputs.dtype,
                device=transformer_inputs.device,
            )
            _ = self.hf_decode(zero_dec, enc_out, enc_valid_len)

        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_past_values = ((past_values - loc) / scale).repeat_interleave(repeats=num_parallel_samples, dim=0)

        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, future_time_features.shape[1], -1)
        features = torch.cat((expanded_static_feat, future_time_features), dim=-1)
        repeated_features = features.repeat_interleave(repeats=num_parallel_samples, dim=0)
        enc_out = ttnn.repeat_interleave(enc_out, repeats=num_parallel_samples, dim=0)

        future_samples = []
        pred_len = int(self.config.prediction_length or self.config.pred_len)
        for k in range(pred_len):
            lagged_sequence = self.hf_get_lagged_subsequences(
                sequence=repeated_past_values,
                subsequences_length=1 + k,
                shift=1,
            )
            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)
            decoder_input = torch.cat((reshaped_lagged_sequence, repeated_features[:, : k + 1]), dim=-1)

            dec_out = self.hf_decode(decoder_input, enc_out, enc_valid_len)
            dec_last_hidden = dec_out[:, -1:, :]
            params = self.hf_parameter_projection(dec_last_hidden)
            params_cat = params[0] if len(params) == 1 else ttnn.concat(params, dim=-1)
            params_cat_torch = ttnn.to_torch(params_cat).float()
            params_torch = list(torch.split(params_cat_torch, out_dim, dim=-1))
            distr = self.build_output_distribution(params_torch)

            if mode == "sample":
                next_sample = distr.sample()
            elif mode == "mean":
                next_sample = distr.mean
            else:
                raise ValueError(f"Unsupported generation mode: {mode}")

            while next_sample.dim() < repeated_loc.dim():
                next_sample = next_sample.unsqueeze(1)
            next_sample = repeated_loc + repeated_scale * next_sample
            repeated_past_values = torch.cat(
                (repeated_past_values, (next_sample - repeated_loc) / repeated_scale),
                dim=1,
            )
            future_samples.append(next_sample)
        concat_future_samples = torch.cat(future_samples, dim=1)
        batch = past_values.shape[0]
        return concat_future_samples.view(batch, num_parallel_samples, pred_len, -1)

    hf_generate = partialmethod(hf_generate_with_mode, mode="sample")
    hf_generate_mean = partialmethod(hf_generate_with_mode, mode="mean")
