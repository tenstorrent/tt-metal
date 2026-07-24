# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Source lineage: HuggingFace PatchTST and PatchTST paper implementation details
# - https://huggingface.co/docs/transformers/en/model_doc/patchtst
# - https://github.com/huggingface/transformers/tree/main/src/transformers/models/patchtst
# - https://arxiv.org/abs/2211.14730

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.demos.wormhole.patchtst.reference.hf_reference import ReferenceArtifacts
from models.demos.wormhole.patchtst.tt.common import (
    MEMORY_CONFIG_BY_TIER,
    PatchTSTRuntimePolicy,
    TTLinear,
    build_linear_from_state,
)


def _linear_on_last_dim(x: ttnn.Tensor, linear: TTLinear, mem_cfg, dtype: ttnn.DataType) -> ttnn.Tensor:
    in_dim = int(x.shape[-1])
    rank = len(x.shape)
    original_shape = tuple(int(x.shape[idx]) for idx in range(rank - 1))
    flattened = 1
    for dim in original_shape:
        flattened *= dim
    flat = ttnn.reshape(x, (flattened, 1, in_dim))
    out_tt = ttnn.linear(flat, linear.weight, bias=linear.bias, memory_config=mem_cfg, dtype=dtype)
    out_dim = int(out_tt.shape[-1])
    return ttnn.reshape(out_tt, (*original_shape, out_dim))


def _pool_embedding(
    embedding: ttnn.Tensor,
    use_cls_token: bool,
    pooling_type: str | None,
    mem_cfg,
) -> ttnn.Tensor:
    if use_cls_token:
        batch_size = int(embedding.shape[0])
        num_channels = int(embedding.shape[1])
        hidden_size = int(embedding.shape[3])
        cls = ttnn.slice(embedding, (0, 0, 0, 0), (batch_size, num_channels, 1, hidden_size))
        return ttnn.reshape(cls, (batch_size, num_channels, hidden_size))
    if pooling_type == "mean":
        return ttnn.mean(embedding, dim=2, keepdim=False, memory_config=mem_cfg)
    if pooling_type == "max":
        max_values = ttnn.max(embedding, dim=2, keepdim=False, memory_config=mem_cfg)
        return max_values[0] if isinstance(max_values, tuple) else max_values
    # For forecast heads when pooling is disabled, keep patch dimension.
    return embedding


@dataclass
class ForecastHead:
    share_projection: bool
    shared_projection: TTLinear | None
    channel_projections: list[TTLinear]
    use_cls_token: bool
    pooling_type: str | None
    num_input_channels: int

    def __call__(
        self, embedding: ttnn.Tensor, runtime: PatchTSTRuntimePolicy, dtype: ttnn.DataType = ttnn.bfloat16
    ) -> ttnn.Tensor:
        mem_cfg = MEMORY_CONFIG_BY_TIER[runtime.activation_memory_tier]
        pooled = _pool_embedding(embedding, self.use_cls_token, self.pooling_type, mem_cfg)
        if self.share_projection:
            if len(pooled.shape) == 4:
                pooled = ttnn.reshape(
                    pooled, (int(pooled.shape[0]), int(pooled.shape[1]), int(pooled.shape[2]) * int(pooled.shape[3]))
                )
            if self.shared_projection is None:
                raise ValueError("share_projection=True requires a shared forecast projection.")
            return ttnn.permute(_linear_on_last_dim(pooled, self.shared_projection, mem_cfg, dtype), (0, 2, 1))
        outputs = []
        for channel_idx in range(self.num_input_channels):
            channel_input = ttnn.reshape(
                ttnn.slice(pooled, (0, channel_idx, 0), (int(pooled.shape[0]), channel_idx + 1, int(pooled.shape[2]))),
                (int(pooled.shape[0]), int(pooled.shape[2])),
            )
            outputs.append(_linear_on_last_dim(channel_input, self.channel_projections[channel_idx], mem_cfg, dtype))
        stacked = ttnn.reshape(outputs[0], (int(outputs[0].shape[0]), 1, int(outputs[0].shape[1])))
        for channel_output in outputs[1:]:
            combined = ttnn.concat(
                [
                    stacked,
                    ttnn.reshape(channel_output, (int(channel_output.shape[0]), 1, int(channel_output.shape[1]))),
                ],
                dim=1,
            )
            ttnn.deallocate(stacked)
            stacked = combined
        return ttnn.permute(stacked, (0, 2, 1))

    def release(self) -> None:
        if self.shared_projection is not None:
            self.shared_projection.release()
        for projection in self.channel_projections:
            projection.release()


@dataclass
class RegressionHead:
    projection: TTLinear | None
    distribution_projections: tuple[TTLinear, TTLinear] | None
    use_cls_token: bool
    pooling_type: str | None
    distribution_output: str | None

    def __call__(self, embedding: ttnn.Tensor, runtime: PatchTSTRuntimePolicy, dtype: ttnn.DataType = ttnn.bfloat16):
        mem_cfg = MEMORY_CONFIG_BY_TIER[runtime.activation_memory_tier]
        pooled = _pool_embedding(embedding, self.use_cls_token, self.pooling_type, mem_cfg)
        flat = ttnn.reshape(pooled, (int(pooled.shape[0]), int(pooled.shape[1]) * int(pooled.shape[2])))
        if self.distribution_output:
            if self.distribution_projections is None:
                raise ValueError("distribution regression requires distribution projections.")
            return tuple(_linear_on_last_dim(flat, proj, mem_cfg, dtype) for proj in self.distribution_projections)
        if self.projection is None:
            raise ValueError("regression projection is missing.")
        return _linear_on_last_dim(flat, self.projection, mem_cfg, dtype)

    def release(self) -> None:
        if self.projection is not None:
            self.projection.release()
        if self.distribution_projections is not None:
            self.distribution_projections[0].release()
            self.distribution_projections[1].release()


@dataclass
class PretrainingHead:
    projection: TTLinear
    use_cls_token: bool

    def __call__(
        self, embedding: ttnn.Tensor, runtime: PatchTSTRuntimePolicy, dtype: ttnn.DataType = ttnn.bfloat16
    ) -> ttnn.Tensor:
        out = _linear_on_last_dim(
            embedding, self.projection, MEMORY_CONFIG_BY_TIER[runtime.activation_memory_tier], dtype
        )
        if self.use_cls_token:
            out = ttnn.slice(
                out, (0, 0, 1, 0), (int(out.shape[0]), int(out.shape[1]), int(out.shape[2]), int(out.shape[3]))
            )
        return out

    def release(self) -> None:
        self.projection.release()


@dataclass
class ClassificationHead:
    projection: TTLinear
    use_cls_token: bool
    pooling_type: str | None

    def __call__(
        self, embedding: ttnn.Tensor, runtime: PatchTSTRuntimePolicy, dtype: ttnn.DataType = ttnn.bfloat16
    ) -> ttnn.Tensor:
        mem_cfg = MEMORY_CONFIG_BY_TIER[runtime.activation_memory_tier]
        pooled = _pool_embedding(embedding, self.use_cls_token, self.pooling_type, mem_cfg)
        flat = ttnn.reshape(pooled, (int(pooled.shape[0]), int(pooled.shape[1]) * int(pooled.shape[2])))
        return _linear_on_last_dim(flat, self.projection, mem_cfg, dtype)

    def release(self) -> None:
        self.projection.release()


def build_forecast_head(
    reference: ReferenceArtifacts, *, device, dtype: ttnn.DataType = ttnn.bfloat16, memory_tier: str = "dram"
) -> ForecastHead | None:
    if reference.task != "forecast":
        return None
    state = reference.model.state_dict()
    memory_config = MEMORY_CONFIG_BY_TIER[memory_tier]
    if bool(reference.config.share_projection):
        return ForecastHead(
            share_projection=True,
            shared_projection=build_linear_from_state(
                state, "head.projection", device=device, dtype=dtype, memory_config=memory_config
            ),
            channel_projections=[],
            use_cls_token=bool(reference.config.use_cls_token),
            pooling_type=reference.config.pooling_type,
            num_input_channels=int(reference.config.num_input_channels),
        )
    return ForecastHead(
        share_projection=False,
        shared_projection=None,
        channel_projections=[
            build_linear_from_state(
                state, f"head.projections.{idx}", device=device, dtype=dtype, memory_config=memory_config
            )
            for idx in range(int(reference.config.num_input_channels))
        ],
        use_cls_token=bool(reference.config.use_cls_token),
        pooling_type=reference.config.pooling_type,
        num_input_channels=int(reference.config.num_input_channels),
    )


def build_regression_head(
    reference: ReferenceArtifacts, *, device, dtype: ttnn.DataType = ttnn.bfloat16, memory_tier: str = "dram"
) -> RegressionHead | None:
    if reference.task != "regression":
        return None
    state = reference.model.state_dict()
    memory_config = MEMORY_CONFIG_BY_TIER[memory_tier]
    if getattr(reference.model, "distribution_output", None):
        return RegressionHead(
            projection=None,
            distribution_projections=(
                build_linear_from_state(
                    state, "head.projection.proj.0", device=device, dtype=dtype, memory_config=memory_config
                ),
                build_linear_from_state(
                    state, "head.projection.proj.1", device=device, dtype=dtype, memory_config=memory_config
                ),
            ),
            use_cls_token=bool(reference.config.use_cls_token),
            pooling_type=reference.config.pooling_type,
            distribution_output=reference.model.distribution_output,
        )
    return RegressionHead(
        projection=build_linear_from_state(
            state, "head.projection", device=device, dtype=dtype, memory_config=memory_config
        ),
        distribution_projections=None,
        use_cls_token=bool(reference.config.use_cls_token),
        pooling_type=reference.config.pooling_type,
        distribution_output=None,
    )


def build_classification_head(
    reference: ReferenceArtifacts, *, device, dtype: ttnn.DataType = ttnn.bfloat16, memory_tier: str = "dram"
) -> ClassificationHead | None:
    if reference.task != "classification":
        return None
    return ClassificationHead(
        projection=build_linear_from_state(
            reference.model.state_dict(),
            "head.linear",
            device=device,
            dtype=dtype,
            memory_config=MEMORY_CONFIG_BY_TIER[memory_tier],
        ),
        use_cls_token=bool(reference.config.use_cls_token),
        pooling_type=reference.config.pooling_type,
    )


def build_pretraining_head(
    reference: ReferenceArtifacts, *, device, dtype: ttnn.DataType = ttnn.bfloat16, memory_tier: str = "dram"
) -> PretrainingHead | None:
    if reference.task != "pretraining":
        return None
    return PretrainingHead(
        projection=build_linear_from_state(
            reference.model.state_dict(),
            "head.linear",
            device=device,
            dtype=dtype,
            memory_config=MEMORY_CONFIG_BY_TIER[memory_tier],
        ),
        use_cls_token=bool(reference.config.use_cls_token),
    )
