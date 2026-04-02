# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import replace

import torch

import ttnn

from .config import TILE_SIZE, InformerConfig, create_sharded_memory_config, get_shard_shape, get_ttnn_dtype
from .embeddings import InformerEmbedding
from .hf_runtime import InformerHfRuntimeMixin
from .ops import linear, make_causal_mask, precompute_trace_constants
from .state_io import export_ttnn_model_state, load_ttnn_model_state
from .transformer import Decoder, Encoder


def enforce_runtime_fallback_policy() -> None:
    """Enforce strict TTNN execution without silent fallback kernels."""
    ttnn.CONFIG.throw_exception_on_fallback = True


def enable_program_cache_if_configured(device, *, use_program_cache: bool) -> None:
    """Apply program-cache policy from config in one explicit runtime helper."""
    if use_program_cache:
        device.enable_program_cache()


def build_teacher_runtime_config(config: InformerConfig) -> InformerConfig | None:
    """
    Build eager teacher-forcing config when the main runtime is sharded.

    Teacher-forcing runs are stateful and easier to reason about with eager/non-sharded execution.
    """
    if not config.use_sharded or config.hf_compat:
        return None
    return replace(
        config,
        use_sharded=False,
        use_trace=False,
        use_program_cache=False,
    )


def should_use_trace(config: InformerConfig, *, future_values_provided: bool) -> bool:
    """
    Decide whether trace replay is valid for this call.

    Trace execution is restricted to non-teacher-forcing paths with moderate decoder/hidden sizes.
    """
    decode_len = int(config.label_len + config.pred_len)
    trace_supported = decode_len <= 256 and int(config.d_model) <= 96
    return config.use_trace and trace_supported and not future_values_provided


class InformerModel(InformerHfRuntimeMixin):
    """Informer time-series forecasting model implemented with TTNN APIs."""

    def __init__(self, config: InformerConfig, *, device, seed: int = 0, create_teacher_runtime: bool = True):
        enforce_runtime_fallback_policy()

        self.config = config
        self.device = device
        self.dtype = get_ttnn_dtype(config.dtype)
        self.seed = seed
        self.rng = torch.Generator().manual_seed(seed)
        self.embedding = InformerEmbedding(config, self.rng, device=device)
        self.encoder = Encoder(config, self.rng, device=device)
        self.decoder = Decoder(config, self.rng, device=device)
        self.proj_w_torch = torch.randn((config.c_out, config.d_model), generator=self.rng, dtype=torch.float32) * 0.02
        self.proj_b_torch = torch.zeros((config.c_out,), dtype=torch.float32)
        self.proj_w = ttnn.from_torch(self.proj_w_torch, device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        self.proj_b = ttnn.from_torch(self.proj_b_torch, device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        self.output_memory_config = ttnn.L1_MEMORY_CONFIG if config.use_l1 else None

        self.last_output: ttnn.Tensor | None = None
        self.trace_id: int | None = None
        self.trace_inputs: dict[str, ttnn.Tensor] | None = None
        self.traced_batch_size: int | None = None
        self.trace_output: ttnn.Tensor | None = None

        self.hf_encoder_embedding = None
        self.hf_decoder_embedding = None
        self.hf_parameter_projection = None
        self.hf_embedder_weights: list[torch.Tensor] = []
        self.teacher_forcing_runtime: InformerModel | None = None

        enable_program_cache_if_configured(self.device, use_program_cache=config.use_program_cache)

        teacher_cfg = build_teacher_runtime_config(config) if create_teacher_runtime else None
        if teacher_cfg is not None:
            self.teacher_forcing_runtime = InformerModel(
                teacher_cfg,
                device=device,
                seed=seed,
                create_teacher_runtime=False,
            )
            self.teacher_forcing_runtime.load_state_dict(self.state_dict(), strict=True)

    def _enable_program_cache(self) -> None:
        self.device.enable_program_cache()

    def _synchronize_device(self) -> None:
        ttnn.synchronize_device(self.device)

    def _iter_attention_modules(self):
        for layer in self.encoder.layers:
            yield layer.attn
        for layer in self.decoder.layers:
            yield layer.self_attn
            yield layer.cross_attn

    def _prewarm_attention_output_module(self, attn, dummy: ttnn.Tensor) -> None:
        memcfg = attn.memory_config
        core_grid = None
        dummy_in = dummy
        if attn.use_sharded:
            memcfg = create_sharded_memory_config(
                get_shard_shape(dummy),
                device=self.device,
                strategy=attn.shard_strategy,
            )
            core_grid = attn.core_grid
            dummy_in = ttnn.to_memory_config(dummy, memcfg)
        _ = linear(dummy_in, attn.o_weight, attn.o_bias, dtype=self.dtype, memory_config=memcfg, core_grid=core_grid)

    def prewarm_attention_output(self, batch_size: int) -> None:
        if not self.config.use_program_cache:
            return
        lengths = {int(self.config.seq_len), int(self.config.label_len + self.config.pred_len)}
        base_memcfg = ttnn.L1_MEMORY_CONFIG if self.config.use_l1 else ttnn.DRAM_MEMORY_CONFIG
        for length in lengths:
            dummy = ttnn.zeros(
                (batch_size, length, self.config.d_model),
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=base_memcfg,
            )
            for attn in self._iter_attention_modules():
                self._prewarm_attention_output_module(attn, dummy)

    def to_device(self, x: torch.Tensor | ttnn.Tensor) -> ttnn.Tensor:
        if isinstance(x, ttnn.Tensor):
            return x
        return ttnn.from_torch(x, device=self.device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return export_ttnn_model_state(self)

    def load_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
        result = load_ttnn_model_state(self, state, strict=strict)
        if self.teacher_forcing_runtime is not None:
            self.teacher_forcing_runtime.load_state_dict(state, strict=strict)
        return result

    def capture_trace(
        self,
        batch_size: int,
        past_values: ttnn.Tensor,
        past_time: ttnn.Tensor,
        future_time: ttnn.Tensor,
    ) -> int:
        """Capture a reusable TTNN execution trace for fixed batch/shape inputs."""
        precompute_trace_constants(self.config, device=self.device)

        future_values = ttnn.from_torch(
            torch.zeros(batch_size, self.config.pred_len, self.config.dec_in, dtype=torch.float32),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.trace_inputs = {
            "past_values": past_values,
            "past_time": past_time,
            "future_time": future_time,
            "future_values": future_values,
        }
        self.traced_batch_size = batch_size

        # Run multiple warmups before trace capture so all kernels/programs are compiled.
        _ = self.forward_ttnn(past_values, past_time, future_time, future_values)
        _ = self.forward_ttnn(past_values, past_time, future_time, future_values)
        self.prewarm_attention_output(batch_size)
        _ = self.forward_ttnn(past_values, past_time, future_time, future_values)
        self._synchronize_device()

        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            _ = self.forward_ttnn(past_values, past_time, future_time, future_values)
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        except RuntimeError:
            # Failure during capture must close and release trace state before re-raising.
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            ttnn.release_trace(self.device, trace_id)
            raise
        self._synchronize_device()

        self.trace_output = self.last_output
        return trace_id

    def copy_trace_input(self, name: str, value: torch.Tensor | ttnn.Tensor) -> None:
        if self.trace_inputs is None:
            raise RuntimeError("Trace inputs are not initialized.")
        target = self.trace_inputs[name]
        if isinstance(value, ttnn.Tensor):
            ttnn.copy(value, target)
            return
        host = ttnn.from_torch(value, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, target)

    def replay_trace(
        self,
        past_values: torch.Tensor | ttnn.Tensor,
        past_time: torch.Tensor | ttnn.Tensor,
        future_time: torch.Tensor | ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Replay a captured trace after copying current call inputs into trace buffers."""
        if self.trace_id is None or self.trace_output is None:
            raise RuntimeError("Trace has not been captured.")
        self.copy_trace_input("past_values", past_values)
        self.copy_trace_input("past_time", past_time)
        self.copy_trace_input("future_time", future_time)
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
        return self.trace_output

    def release_trace(self) -> None:
        if self.trace_id is None:
            return
        ttnn.release_trace(self.device, self.trace_id)
        self.trace_id = None
        self.trace_inputs = None
        self.traced_batch_size = None
        self.trace_output = None

    def _build_decoder_inputs(
        self,
        past_values: ttnn.Tensor,
        past_time_features: ttnn.Tensor,
        future_time_features: ttnn.Tensor,
        *,
        future_values: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        cfg = self.config
        label_len = cfg.label_len
        dec_known = past_values[:, -label_len:, :]
        if future_values is None:
            future_pad = ttnn.zeros(
                (past_values.shape[0], cfg.pred_len, cfg.dec_in),
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
        else:
            future_pad = future_values[:, : cfg.pred_len, :]
        dec_values = ttnn.concat([dec_known, future_pad], dim=1)
        dec_time = ttnn.concat(
            [past_time_features[:, -label_len:, :], future_time_features[:, : cfg.pred_len, :]],
            dim=1,
        )
        return dec_values, dec_time

    def _project_prediction(self, decoder_output: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        return linear(
            decoder_output[:, -cfg.pred_len :, :],
            self.proj_w,
            self.proj_b,
            dtype=self.dtype,
            memory_config=self.output_memory_config,
        )

    def _embed_encoder_decoder(
        self,
        past_values: ttnn.Tensor,
        past_time_features: ttnn.Tensor,
        future_time_features: ttnn.Tensor,
        *,
        future_values: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, int, ttnn.Tensor]:
        enc_embed = self.embedding.encoder(past_values, past_time_features)
        enc_out, enc_valid_len = self.encoder(enc_embed, None)
        dec_values, dec_time = self._build_decoder_inputs(
            past_values,
            past_time_features,
            future_time_features,
            future_values=future_values,
        )
        dec_embed = self.embedding.decoder(dec_values, dec_time)
        return enc_out, enc_valid_len, dec_embed

    def forward_ttnn(
        self,
        past_values: ttnn.Tensor,
        past_time_features: ttnn.Tensor,
        future_time_features: ttnn.Tensor,
        future_values: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Execute one TTNN forward pass and return predicted values for `pred_len`."""
        cfg = self.config
        past_values = ttnn.to_layout(past_values, ttnn.TILE_LAYOUT)
        past_time_features = ttnn.to_layout(past_time_features, ttnn.TILE_LAYOUT)
        future_time_features = ttnn.to_layout(future_time_features, ttnn.TILE_LAYOUT)
        if future_values is not None:
            future_values = ttnn.to_layout(future_values, ttnn.TILE_LAYOUT)

        enc_out, enc_valid_len, dec_embed = self._embed_encoder_decoder(
            past_values,
            past_time_features,
            future_time_features,
            future_values=future_values,
        )

        dec_len = dec_embed.shape[1]
        pad_len = int(math.ceil(dec_len / TILE_SIZE)) * TILE_SIZE
        self_mask = make_causal_mask(
            pad_len,
            batch=past_values.shape[0],
            heads=1,
            device=self.device,
            dtype=self.dtype,
            mask_value=cfg.attn_mask_value,
        )
        dec_out = self.decoder(dec_embed, enc_out, self_mask, None, enc_valid_len)

        output = self._project_prediction(dec_out)
        self.last_output = output
        return output

    def _forward_teacher_forcing(
        self,
        past_values: ttnn.Tensor,
        past_time_features: ttnn.Tensor,
        future_time_features: ttnn.Tensor,
        future_values: ttnn.Tensor,
    ) -> ttnn.Tensor:
        if self.teacher_forcing_runtime is None:
            return self.forward_ttnn(past_values, past_time_features, future_time_features, future_values)
        runtime = self.teacher_forcing_runtime
        return runtime.forward_ttnn(
            runtime.to_device(past_values),
            runtime.to_device(past_time_features),
            runtime.to_device(future_time_features),
            runtime.to_device(future_values),
        )

    def __call__(
        self,
        past_values: torch.Tensor | ttnn.Tensor,
        past_time_features: torch.Tensor | ttnn.Tensor,
        future_time_features: torch.Tensor | ttnn.Tensor,
        future_values: torch.Tensor | ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Dispatch between trace replay, teacher-forcing, and eager forward execution."""
        cfg = self.config
        if cfg.hf_compat:
            raise ValueError("Use hf_generate/hf_generate_mean for HuggingFace-compatible inference.")
        use_trace = should_use_trace(cfg, future_values_provided=future_values is not None)
        batch_size = int(past_values.shape[0])

        if use_trace:
            if self.trace_id is not None and self.traced_batch_size != batch_size:
                self.release_trace()
            if self.trace_id is not None and self.traced_batch_size == batch_size:
                return self.replay_trace(past_values, past_time_features, future_time_features)

            past_values_ttnn = self.to_device(past_values)
            past_time_ttnn = self.to_device(past_time_features)
            future_time_ttnn = self.to_device(future_time_features)
            self.trace_id = self.capture_trace(batch_size, past_values_ttnn, past_time_ttnn, future_time_ttnn)
            return self.replay_trace(past_values, past_time_features, future_time_features)

        past_values_ttnn = self.to_device(past_values)
        past_time_ttnn = self.to_device(past_time_features)
        future_time_ttnn = self.to_device(future_time_features)
        future_values_ttnn = self.to_device(future_values) if future_values is not None else None
        if future_values_ttnn is None:
            return self.forward_ttnn(past_values_ttnn, past_time_ttnn, future_time_ttnn, None)
        return self._forward_teacher_forcing(
            past_values_ttnn,
            past_time_ttnn,
            future_time_ttnn,
            future_values_ttnn,
        )

    def stream_forecast(
        self,
        past_values: torch.Tensor | ttnn.Tensor,
        past_time_features: torch.Tensor | ttnn.Tensor,
        future_time_features: torch.Tensor | ttnn.Tensor,
        *,
        future_values: torch.Tensor | ttnn.Tensor | None = None,
        chunk_size: int = 32,
    ) -> ttnn.Tensor:
        cfg = self.config
        if cfg.hf_compat:
            raise ValueError("Use hf_generate/hf_generate_mean for HuggingFace-compatible inference.")
        past_values = self.to_device(past_values)
        past_time_features = self.to_device(past_time_features)
        future_time_features = self.to_device(future_time_features)
        future_values = self.to_device(future_values) if future_values is not None else None

        enc_out, enc_valid_len, dec_embed = self._embed_encoder_decoder(
            past_values,
            past_time_features,
            future_time_features,
            future_values=future_values,
        )

        dec_out, _ = self.decoder.forward_streaming(
            dec_embed,
            enc_out,
            enc_valid_len,
            chunk_size=chunk_size,
        )

        return self._project_prediction(dec_out)


def create_informer(
    config: InformerConfig,
    *,
    device,
    seed: int = 0,
) -> InformerModel:
    return InformerModel(config, device=device, seed=seed)
