# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Source lineage: HuggingFace PatchTST and PatchTST paper implementation details
# - https://huggingface.co/docs/transformers/en/model_doc/patchtst
# - https://github.com/huggingface/transformers/tree/main/src/transformers/models/patchtst
# - https://arxiv.org/abs/2211.14730

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

import ttnn
from models.demos.wormhole.patchtst.config import PatchTSTDemoConfig
from models.demos.wormhole.patchtst.reference.hf_reference import ReferenceArtifacts
from models.demos.wormhole.patchtst.tt.common import MEMORY_CONFIG_BY_TIER, TT_DTYPE, PatchTSTRuntimePolicy
from models.demos.wormhole.patchtst.tt.embeddings import PatchTSTEmbedder, build_embedder
from models.demos.wormhole.patchtst.tt.encoder import build_encoder_layers, encode_hidden_state
from models.demos.wormhole.patchtst.tt.heads import (
    ClassificationHead,
    ForecastHead,
    PretrainingHead,
    RegressionHead,
    build_classification_head,
    build_forecast_head,
    build_pretraining_head,
    build_regression_head,
)
from models.demos.wormhole.patchtst.tt.patching import apply_mask, patchify, patchify_tt
from models.demos.wormhole.patchtst.tt.positional_encoding import (
    PatchTSTPositionalEncoding,
    build_positional_encoding,
    build_sincos_position_encoding,
)


def _to_torch_recursive(value: Any) -> Any:
    if isinstance(value, ttnn.Tensor):
        return ttnn.to_torch(ttnn.from_device(value))
    if isinstance(value, dict):
        return {k: _to_torch_recursive(v) for k, v in value.items()}
    return value


def _release_tt_value(value: Any) -> None:
    if isinstance(value, ttnn.Tensor):
        ttnn.deallocate(value)
        return
    if isinstance(value, dict):
        for item in value.values():
            _release_tt_value(item)


@dataclass
class PreparedEncoderInput:
    hidden_state: ttnn.Tensor
    mask: torch.Tensor | None
    patch_input: torch.Tensor | None
    loc: torch.Tensor
    scale: torch.Tensor

    def release(self) -> None:
        ttnn.deallocate(self.hidden_state)


@dataclass
class PatchTSTTTNNOutput:
    prediction: Any
    mask: torch.Tensor | None = None
    patch_input: torch.Tensor | None = None
    loc: torch.Tensor | None = None
    scale: torch.Tensor | None = None

    def release(self) -> None:
        _release_tt_value(self.prediction)


class PatchTSTTTNNModel:
    def __init__(
        self,
        demo_config: PatchTSTDemoConfig,
        reference: ReferenceArtifacts,
        device,
        runtime_policy: PatchTSTRuntimePolicy,
        classification_reference: ReferenceArtifacts | None = None,
    ):
        self.cfg = demo_config
        self.reference = reference
        self.device = device
        self.runtime = runtime_policy
        self.classification_reference = classification_reference

        self.task = self.reference.task
        self.ref_model = self.reference.model
        self.ref_core = self.reference.model.model
        self.ref_encoder = self.ref_core.encoder
        self.io_dtype = TT_DTYPE
        self.activation_memory_config = MEMORY_CONFIG_BY_TIER[self.runtime.activation_memory_tier]
        self.embedder: PatchTSTEmbedder = build_embedder(
            self.reference, device=device, dtype=TT_DTYPE, memory_tier=self.runtime.weight_memory_tier
        )
        self.positional: PatchTSTPositionalEncoding = build_positional_encoding(self.reference)
        self.encoder_layers, self.pre_norm = build_encoder_layers(
            self.reference, device=device, dtype=TT_DTYPE, memory_tier=self.runtime.weight_memory_tier
        )
        self.forecast_head: ForecastHead | None = build_forecast_head(
            self.reference, device=device, dtype=TT_DTYPE, memory_tier=self.runtime.weight_memory_tier
        )
        self.regression_head: RegressionHead | None = build_regression_head(
            self.reference, device=device, dtype=TT_DTYPE, memory_tier=self.runtime.weight_memory_tier
        )
        self.classification_head: ClassificationHead | None = build_classification_head(
            self.reference, device=device, dtype=TT_DTYPE, memory_tier=self.runtime.weight_memory_tier
        )
        self.pretraining_head: PretrainingHead | None = build_pretraining_head(
            self.reference, device=device, dtype=TT_DTYPE, memory_tier=self.runtime.weight_memory_tier
        )
        self.multi_task_classification_head: ClassificationHead | None = None
        self.classification_config = None
        if self.classification_reference is not None:
            self._validate_multi_task_compatibility()
            self.multi_task_classification_head = build_classification_head(
                self.classification_reference,
                device=device,
                dtype=TT_DTYPE,
                memory_tier=self.runtime.weight_memory_tier,
            )
            self.classification_config = self.classification_reference.config

    def _validate_multi_task_compatibility(self) -> None:
        if self.classification_reference is None:
            return
        forecast_cfg = self.reference.config
        classification_cfg = self.classification_reference.config
        for field_name in (
            "context_length",
            "patch_length",
            "patch_stride",
            "num_input_channels",
            "d_model",
            "ffn_dim",
            "num_hidden_layers",
            "num_attention_heads",
            "share_embedding",
            "use_cls_token",
            "pooling_type",
            "channel_attention",
        ):
            forecast_value = getattr(forecast_cfg, field_name)
            classification_value = getattr(classification_cfg, field_name)
            if forecast_value != classification_value:
                raise ValueError(
                    "Public multi_task runs require forecast and classification checkpoints with identical "
                    f"encoder geometry, but field {field_name!r} differs: "
                    f"forecast={forecast_value!r}, classification={classification_value!r}"
                )
        forecast_state = self.reference.model.state_dict()
        classification_state = self.classification_reference.model.state_dict()
        for key, tensor in forecast_state.items():
            if (
                not key.startswith("model.encoder.")
                and not key.startswith("model.patchifier.")
                and not key.startswith("model.scaler.")
            ):
                continue
            if key not in classification_state:
                raise ValueError(f"Classification checkpoint is missing shared encoder key: {key}")
            if not torch.equal(tensor.detach().cpu(), classification_state[key].detach().cpu()):
                raise ValueError(
                    "Public multi_task requires the classification checkpoint to reuse the forecast encoder exactly, "
                    f"but parameter {key!r} does not match."
                )

    def close(self) -> None:
        self.embedder.release()
        for layer in self.encoder_layers:
            layer.release()
        if self.forecast_head is not None:
            self.forecast_head.release()
        if self.regression_head is not None:
            self.regression_head.release()
        if self.classification_head is not None:
            self.classification_head.release()
        if self.pretraining_head is not None:
            self.pretraining_head.release()
        if self.multi_task_classification_head is not None:
            self.multi_task_classification_head.release()

    def prepare_hidden_input(
        self,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor | None = None,
    ) -> PreparedEncoderInput:
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        with torch.no_grad():
            scaled_past_values, loc, scale = self.ref_core.scaler(past_values, past_observed_mask)
            use_device_patching = bool(self.runtime.use_device_patching and not self.ref_core.do_mask_input)
            if use_device_patching:
                scaled_tt = ttnn.from_torch(
                    scaled_past_values,
                    dtype=self.io_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=self.activation_memory_config,
                )
                patch_input_tt = patchify_tt(
                    scaled_tt,
                    context_length=int(self.cfg.context_length),
                    patch_length=self.cfg.patch_length,
                    patch_stride=self.cfg.patch_stride,
                )
                ttnn.deallocate(scaled_tt)
                patch_input = None
            else:
                patch_input_tt = None
                patch_input = patchify(
                    scaled_past_values,
                    context_length=int(self.cfg.context_length),
                    patch_length=self.cfg.patch_length,
                    patch_stride=self.cfg.patch_stride,
                )

            if self.ref_core.do_mask_input:
                # Pretraining checkpoints expect host-side masking semantics before TTNN embedding.
                if patch_input is None:
                    patch_input = ttnn.to_torch(ttnn.from_device(patch_input_tt))
                    ttnn.deallocate(patch_input_tt)
                    patch_input_tt = None
                masked, mask = apply_mask(
                    patch_input,
                    mask_type=self.ref_core.masking.mask_type,
                    random_mask_ratio=float(self.ref_core.masking.random_mask_ratio),
                    num_forecast_mask_patches=self.ref_core.masking.num_forecast_mask_patches,
                    mask_value=float(self.ref_core.masking.mask_value),
                    seed=int(self.cfg.masking_seed),
                )
            else:
                masked, mask = (patch_input_tt if patch_input_tt is not None else patch_input), None

            checkpoint_share_embedding = bool(self.ref_encoder.embedder.share_embedding)
            if self.cfg.share_embedding and not checkpoint_share_embedding:
                raise ValueError(
                    "share_embedding=True requested but checkpoint embedder is configured as non-shared. "
                    "Set --share-embedding false for this checkpoint."
                )

            embedded = self.embedder(masked, bool(self.cfg.share_embedding), self.device, self.runtime, TT_DTYPE)
            hidden = self.positional(embedded, self.device, self.runtime)
            if hidden is not embedded:
                ttnn.deallocate(embedded)
            return PreparedEncoderInput(
                hidden_state=hidden,
                mask=mask,
                patch_input=patch_input,
                loc=loc,
                scale=scale,
            )

    def prepare_hidden_input_host(
        self,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor | None = None,
        *,
        loc: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        if loc is None or scale is None:
            scaled_past_values, loc, scale = self.ref_core.scaler(past_values, past_observed_mask)
        else:
            scaled_past_values = (past_values - loc) / scale

        patch_input = patchify(
            scaled_past_values,
            context_length=int(self.cfg.context_length),
            patch_length=self.cfg.patch_length,
            patch_stride=self.cfg.patch_stride,
        )
        if self.ref_core.do_mask_input:
            raise ValueError("Cached streaming host prepare is only supported for non-masked forecast inference.")
        if not bool(self.cfg.share_embedding):
            raise ValueError("Cached streaming host prepare currently supports only share_embedding=True.")

        projection_module = self.ref_encoder.embedder.input_embedding
        projection_weight = projection_module.weight.detach()
        projection_bias = projection_module.bias.detach() if projection_module.bias is not None else None
        patch_input = patch_input.to(torch.float32)
        embedded = F.linear(
            patch_input,
            projection_weight.to(torch.float32),
            projection_bias.to(torch.float32) if projection_bias is not None else None,
        )

        checkpoint_pos = self.positional.position_enc.detach().to(torch.float32)
        expected_tokens = int(embedded.shape[2]) + (1 if bool(self.reference.config.use_cls_token) else 0)
        if int(checkpoint_pos.shape[0]) != expected_tokens:
            if str(self.reference.config.positional_encoding_type) != "sincos":
                raise ValueError(
                    "Cached host prepare requires positional token count to match the checkpoint table or use sincos "
                    f"position encoding. runtime_tokens={expected_tokens}, checkpoint_tokens={checkpoint_pos.shape[0]}"
                )
            checkpoint_pos = build_sincos_position_encoding(
                num_positions=expected_tokens,
                d_model=int(checkpoint_pos.shape[-1]),
                device=checkpoint_pos.device,
            )

        if bool(self.reference.config.use_cls_token):
            cls_token = self.positional.cls_token
            if cls_token is None:
                raise ValueError("use_cls_token=True but cls token parameters are missing.")
            cls_token = cls_token.detach().to(torch.float32)
            patch_pos = checkpoint_pos[1:, :].reshape(1, 1, int(embedded.shape[2]), -1)
            patch_hidden = embedded + patch_pos
            cls = (
                (cls_token + checkpoint_pos[:1, :])
                .reshape(1, 1, 1, -1)
                .expand(int(embedded.shape[0]), int(embedded.shape[1]), -1, -1)
            )
            hidden = torch.cat([cls, patch_hidden], dim=2)
        else:
            pos = checkpoint_pos.reshape(1, 1, int(embedded.shape[2]), -1)
            hidden = embedded + pos
        return hidden.contiguous(), loc.contiguous(), scale.contiguous()

    def forward_from_hidden_tt(
        self,
        prepared: PreparedEncoderInput,
        task: str | None = None,
    ) -> PatchTSTTTNNOutput:
        encoded_hidden = encode_hidden_state(
            prepared.hidden_state,
            self.encoder_layers,
            self.pre_norm,
            self.cfg.channel_mode == "attention",
            self.runtime,
            self.device,
            TT_DTYPE,
        )
        prediction = self.forward_heads_tt(encoded_hidden, task or self.task)
        ttnn.deallocate(encoded_hidden)
        return PatchTSTTTNNOutput(prediction, prepared.mask, prepared.patch_input, prepared.loc, prepared.scale)

    def _forward_classification_head_multi_task(self, hidden_state: ttnn.Tensor):
        if self.multi_task_classification_head is None or self.classification_config is None:
            raise ValueError("Multi-task path requires a classification checkpoint/weights.")
        if int(hidden_state.shape[1]) != int(self.classification_config.num_input_channels):
            raise ValueError(
                "Public multi_task runs do not allow channel coercion between the shared forecast encoder and the "
                "classification head. Generate a task-matched checkpoint pair instead. "
                f"hidden_channels={hidden_state.shape[1]}, classification_channels={self.classification_config.num_input_channels}"
            )
        return self.multi_task_classification_head(hidden_state, self.runtime, TT_DTYPE)

    def forward_heads_tt(self, hidden_state: ttnn.Tensor, task: str):
        if task == "forecast":
            if self.forecast_head is None:
                raise ValueError("Forecast head parameters are missing.")
            return self.forecast_head(hidden_state, self.runtime, TT_DTYPE)
        if task == "regression":
            if self.regression_head is None:
                raise ValueError("Regression head parameters are missing.")
            return self.regression_head(hidden_state, self.runtime, TT_DTYPE)
        if task == "pretraining":
            if self.pretraining_head is None:
                raise ValueError("Pretraining head parameters are missing.")
            return self.pretraining_head(hidden_state, self.runtime, TT_DTYPE)
        if task == "classification":
            if self.classification_head is None:
                raise ValueError("Classification head parameters are missing.")
            return self.classification_head(hidden_state, self.runtime, TT_DTYPE)
        if task == "multi_task":
            return {
                "forecast": self.forward_heads_tt(hidden_state, "forecast"),
                "classification": self._forward_classification_head_multi_task(hidden_state),
            }
        raise ValueError(f"Unsupported task: {task}")

    def forward_tt(
        self,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor | None = None,
        task: str | None = None,
    ) -> PatchTSTTTNNOutput:
        prepared = self.prepare_hidden_input(past_values=past_values, past_observed_mask=past_observed_mask)
        try:
            return self.forward_from_hidden_tt(prepared=prepared, task=task)
        finally:
            prepared.release()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor | None = None,
        task: str | None = None,
    ) -> PatchTSTTTNNOutput:
        encoded = self.forward_tt(past_values=past_values, past_observed_mask=past_observed_mask, task=task)
        encoded.prediction = _to_torch_recursive(encoded.prediction)
        effective_task = task or self.task
        if effective_task == "forecast" and isinstance(encoded.prediction, torch.Tensor):
            encoded.prediction = encoded.prediction * encoded.scale + encoded.loc
        if effective_task == "multi_task" and isinstance(encoded.prediction, dict):
            encoded.prediction["forecast"] = encoded.prediction["forecast"] * encoded.scale + encoded.loc
        return encoded
