# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch
import ttnn

from models.experimental.mole.reference.config import MoLEConfig, replace_num_experts
from models.experimental.mole.reference.mole import MixtureOfLinearExperts
from models.experimental.mole.tt.common import (
    LOW_FIDELITY_LINEAR_COMPUTE_KERNEL_CONFIG,
    TtRuntimeOptions,
    apply_linear,
    apply_two_layer_mlp,
    default_activation_memory_config,
    extract_initial_marks,
    moving_average_projection_matrix,
    permute_temporal_router_mlp_to_channel_major,
    register_trace_release_hook,
    reduce_weighted_heads_batch_major,
    temporal_gating,
    upload_linear,
    upload_timeseries_and_marks_to_device,
    validate_time_marks,
    validate_timeseries_input,
)
from models.experimental.mole.tt.rlinear import TtRLinearExpert
from models.experimental.mole.tt.rmlp import TtRMLPExpert


@dataclass
class _PackedDLinearParams:
    moving_average: Any
    seasonal: Any
    trend: Any


@dataclass
class _PackedRLinearParams:
    projection: Any
    scale: Any
    constant: Any


@dataclass
class _PackedRmlpParams:
    temporal_1: Any
    temporal_2: Any
    projection: Any
    affine_weight: Any
    affine_bias: Any
    output_bias: Any
    output_denom: Any


@dataclass
class _MoLEPredictionTraceState:
    trace_id: int
    prediction: ttnn.Tensor
    input_ids: tuple[int, int]


@dataclass
class _MoLEForwardTraceState:
    trace_id: int
    output: tuple[ttnn.Tensor, ttnn.Tensor]
    input_ids: tuple[int, int]


class TtMoLE:
    def __init__(
        self,
        config: MoLEConfig,
        *,
        reference_model: MixtureOfLinearExperts,
        device,
        runtime_options: TtRuntimeOptions | None = None,
    ):
        options = runtime_options or TtRuntimeOptions()
        self.config = config
        self.device = device
        self.parameter_memory_config = options.memory_config
        self.activation_memory_config = (
            default_activation_memory_config()
            if options.activation_memory_config is None
            else options.activation_memory_config
        )
        self.memory_config = self.activation_memory_config
        self.dtype = options.dtype
        expert_runtime_options = TtRuntimeOptions(
            memory_config=self.parameter_memory_config,
            activation_memory_config=self.activation_memory_config,
            dtype=self.dtype,
        )
        single_expert_config = replace_num_experts(config, num_experts=1)
        self._packed_dlinear_parameters = None
        self._packed_rlinear_parameters = None
        self._packed_rmlp_parameters = None
<<<<<<< HEAD
        self.experts = self._build_individual_experts_if_needed(
            reference_model, single_expert_config, expert_runtime_options
        )
        self._reference_experts = reference_model.experts
        self._reference_router = reference_model.router
        self._tt_router_parameters = None
        self._prediction_trace_state = None
        self._forward_trace_state = None
        self._trace_capture_enabled = True
        register_trace_release_hook(device=self.device, hook=self._release_traces)

    def _uses_packed_expert_path(self) -> bool:
        c = self.config
        return c.base_model_type == "dlinear" or (
            c.base_model_type == "rlinear" and not c.individual
        ) or (c.base_model_type == "rmlp" and not c.individual)

    def _build_individual_experts_if_needed(
        self,
        reference_model: MixtureOfLinearExperts,
        single_expert_config: MoLEConfig,
        expert_runtime_options: TtRuntimeOptions,
    ):
        if self._uses_packed_expert_path():
            return None
        base = self.config.base_model_type
        if base == "rlinear":
            expert_class = TtRLinearExpert
        elif base == "rmlp":
            expert_class = TtRMLPExpert
        else:
            raise ValueError(f"unsupported base_model_type: {base}")
        return [
            expert_class(
                single_expert_config,
                reference_model=reference_expert,
                runtime_options=expert_runtime_options,
            )
            for reference_expert in reference_model.experts
        ]

    def _release_trace_state(self, state: _MoLEPredictionTraceState | _MoLEForwardTraceState | None) -> None:
        if state is None:
            return
        with contextlib.suppress(Exception):
            ttnn.release_trace(self.device, state.trace_id)
=======
        self.experts = None
        if config.base_model_type == "dlinear":
            pass
        elif config.base_model_type == "rlinear" and not config.individual:
            pass
        elif config.base_model_type == "rmlp" and not config.individual:
            pass
        elif config.base_model_type == "rlinear":
            expert_class = TtRLinearExpert
        elif config.base_model_type == "rmlp":
            expert_class = TtRMLPExpert
        else:
            raise ValueError(f"unsupported base_model_type: {config.base_model_type}")

        if config.base_model_type not in {"dlinear"} and not (
            (config.base_model_type == "rlinear" and not config.individual)
            or (config.base_model_type == "rmlp" and not config.individual)
        ):
            self.experts = [
                expert_class(
                    single_expert_config,
                    reference_model=reference_expert,
                    runtime_options=expert_runtime_options,
                )
                for reference_expert in reference_model.experts
            ]
        self._reference_experts = reference_model.experts
        self._reference_router = reference_model.router
        self._tt_router_parameters = None
        self._prediction_trace_state = None
        self._forward_trace_state = None
        self._trace_capture_enabled = True
        register_trace_release_hook(device=self.device, hook=self._release_traces)

    def _release_trace_state(self, state) -> None:
        if state is None:
            return
        try:
            ttnn.release_trace(self.device, state["trace_id"])
        except Exception:
            pass
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f

    def _release_prediction_trace(self) -> None:
        self._release_trace_state(self._prediction_trace_state)
        self._prediction_trace_state = None

    def _release_forward_trace(self) -> None:
        self._release_trace_state(self._forward_trace_state)
        self._forward_trace_state = None

    def _release_traces(self) -> None:
        self._release_prediction_trace()
        self._release_forward_trace()

    def __del__(self):
        self._release_traces()

<<<<<<< HEAD
    def _expected_time_features(self) -> int:
        return 4 if self.config.freq.lower().endswith("h") else 5

=======
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
    def _ensure_router_parameters(self) -> None:
        if self._tt_router_parameters is not None:
            return
        router_parameters = permute_temporal_router_mlp_to_channel_major(
            self._reference_router,
            channels=self.config.input_dim,
            num_predictions=self.config.num_experts,
        )
        up = lambda weight, bias: upload_linear(
            weight,
            bias,
            device=self.device,
            dtype=self.dtype,
            memory_config=self.parameter_memory_config,
        )
        self._tt_router_parameters = {
            "linear_1": up(router_parameters["linear_1_weight"], router_parameters["linear_1_bias"]),
            "linear_2": up(router_parameters["linear_2_weight"], router_parameters["linear_2_bias"]),
        }

    def _ensure_packed_dlinear_parameters(self) -> None:
        if self._packed_dlinear_parameters is not None:
            return

        seasonal_weight = torch.cat(
            [expert.seasonal_projection.weight.detach() for expert in self._reference_experts], dim=0
        )
        seasonal_bias = torch.cat(
            [expert.seasonal_projection.bias.detach() for expert in self._reference_experts], dim=0
        )
        trend_weight = torch.cat([expert.trend_projection.weight.detach() for expert in self._reference_experts], dim=0)
        trend_bias = torch.cat([expert.trend_projection.bias.detach() for expert in self._reference_experts], dim=0)
        moving_average_weight = moving_average_projection_matrix(
            seq_len=self.config.seq_len,
            kernel_size=self.config.moving_average_kernel_size,
        )
        moving_average_bias = seasonal_bias.new_zeros(self.config.seq_len)
<<<<<<< HEAD
        self._packed_dlinear_parameters = _PackedDLinearParams(
            moving_average=upload_linear(
=======
        self._packed_dlinear_parameters = {
            "moving_average": upload_linear(
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
                moving_average_weight,
                moving_average_bias,
                device=self.device,
                dtype=self.dtype,
                memory_config=self.parameter_memory_config,
            ),
<<<<<<< HEAD
            seasonal=upload_linear(
=======
            "seasonal": upload_linear(
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
                seasonal_weight,
                seasonal_bias,
                device=self.device,
                dtype=self.dtype,
                memory_config=self.parameter_memory_config,
            ),
<<<<<<< HEAD
            trend=upload_linear(
=======
            "trend": upload_linear(
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
                trend_weight,
                trend_bias,
                device=self.device,
                dtype=self.dtype,
                memory_config=self.parameter_memory_config,
            ),
<<<<<<< HEAD
        )
=======
        }
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f

    def _compute_dlinear_expert_projection(
        self,
        input_tensor: ttnn.Tensor,
    ) -> ttnn.Tensor:
        self._ensure_packed_dlinear_parameters()
        mc = self.activation_memory_config
        batch_size = input_tensor.shape[1]

        input_channels_first = ttnn.permute(input_tensor, (0, 1, 3, 2))
<<<<<<< HEAD
        p = self._packed_dlinear_parameters
        trend = apply_linear(input_channels_first, p.moving_average, memory_config=mc)
        seasonal = ttnn.subtract(input_channels_first, trend, memory_config=mc)
        combined = ttnn.add(
            apply_linear(seasonal, p.seasonal, memory_config=mc),
            apply_linear(trend, p.trend, memory_config=mc),
=======
        trend = apply_linear(input_channels_first, self._packed_dlinear_parameters["moving_average"], memory_config=mc)
        seasonal = ttnn.subtract(input_channels_first, trend, memory_config=mc)
        combined = ttnn.add(
            apply_linear(seasonal, self._packed_dlinear_parameters["seasonal"], memory_config=mc),
            apply_linear(trend, self._packed_dlinear_parameters["trend"], memory_config=mc),
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
            memory_config=mc,
        )
        combined = ttnn.reshape(
            combined,
            (1, batch_size, self.config.input_dim, self.config.num_experts, self.config.pred_len),
        )
        combined = ttnn.permute(combined, (0, 1, 2, 4, 3))
        return ttnn.reshape(
            combined,
            (1, batch_size * self.config.input_dim, self.config.pred_len, self.config.num_experts),
        )

    def _ensure_packed_rlinear_parameters(self) -> None:
        if self._packed_rlinear_parameters is not None:
            return

        projection_weight = torch.cat([expert.projection.weight.detach() for expert in self._reference_experts], dim=0)
        zero_bias = torch.zeros(
            projection_weight.shape[0], dtype=projection_weight.dtype, device=projection_weight.device
        )

        scales = []
        constants = []
        for expert in self._reference_experts:
            projection_bias = expert.projection.bias.detach()
<<<<<<< HEAD
            affine_weight = expert.rev.affine_weight.detach()
            affine_bias = expert.rev.affine_bias.detach()
            denom = affine_weight + (self.config.revin_eps**2)
            scale = affine_weight / denom
            row_sum = expert.projection.weight.detach().sum(dim=1, keepdim=True)
            constant = (
                row_sum * affine_bias.unsqueeze(0) + projection_bias.unsqueeze(1) - affine_bias.unsqueeze(0)
            ) / denom.unsqueeze(0)
=======
            if expert.rev is None:
                scale = torch.ones(self.config.input_dim, dtype=projection_bias.dtype, device=projection_bias.device)
                constant = projection_bias.unsqueeze(1).repeat(1, self.config.input_dim)
            else:
                affine_weight = expert.rev.affine_weight.detach()
                affine_bias = expert.rev.affine_bias.detach()
                denom = affine_weight + (self.config.revin_eps**2)
                scale = affine_weight / denom
                row_sum = expert.projection.weight.detach().sum(dim=1, keepdim=True)
                constant = (
                    row_sum * affine_bias.unsqueeze(0) + projection_bias.unsqueeze(1) - affine_bias.unsqueeze(0)
                ) / denom.unsqueeze(0)
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
            scales.append(scale)
            constants.append(constant)

        packed_scale = torch.stack(scales, dim=1).reshape(1, 1, self.config.input_dim * self.config.num_experts, 1)
        packed_constant = torch.stack([constant.transpose(0, 1) for constant in constants], dim=1).reshape(
            1, 1, self.config.input_dim * self.config.num_experts, self.config.pred_len
        )

<<<<<<< HEAD
        self._packed_rlinear_parameters = _PackedRLinearParams(
            projection=upload_linear(
=======
        self._packed_rlinear_parameters = {
            "projection": upload_linear(
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
                projection_weight,
                zero_bias,
                device=self.device,
                dtype=self.dtype,
                memory_config=self.parameter_memory_config,
            ),
<<<<<<< HEAD
            scale=ttnn.from_torch(
=======
            "scale": ttnn.from_torch(
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
                packed_scale,
                device=self.device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.parameter_memory_config,
            ),
<<<<<<< HEAD
            constant=ttnn.from_torch(
=======
            "constant": ttnn.from_torch(
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
                packed_constant,
                device=self.device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.parameter_memory_config,
            ),
<<<<<<< HEAD
        )
=======
        }
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f

    def _compute_rlinear_expert_projection(
        self,
        input_tensor: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None, ttnn.Tensor | None]:
        self._ensure_packed_rlinear_parameters()
        mc = self.activation_memory_config
        batch_size = input_tensor.shape[1]

<<<<<<< HEAD
        mean = ttnn.sum(input_tensor, dim=2, keepdim=True, scalar=1.0 / self.config.seq_len, memory_config=mc)
        centered = ttnn.subtract(input_tensor, mean, memory_config=mc)
        variance = ttnn.multiply(centered, centered, memory_config=mc)
        variance = ttnn.sum(variance, dim=2, keepdim=True, scalar=1.0 / self.config.seq_len, memory_config=mc)
        stdev = ttnn.sqrt(ttnn.add(variance, self.config.revin_eps, memory_config=mc))
        normalized = ttnn.div(centered, stdev, memory_config=mc)

        pr = self._packed_rlinear_parameters
        projected = apply_linear(
            ttnn.permute(normalized, (0, 1, 3, 2)),
            pr.projection,
=======
        mean = None
        stdev = None
        normalized = input_tensor
        if self._reference_experts[0].rev is not None:
            mean = ttnn.sum(input_tensor, dim=2, keepdim=True, scalar=1.0 / self.config.seq_len, memory_config=mc)
            centered = ttnn.subtract(input_tensor, mean, memory_config=mc)
            variance = ttnn.multiply(centered, centered, memory_config=mc)
            variance = ttnn.sum(variance, dim=2, keepdim=True, scalar=1.0 / self.config.seq_len, memory_config=mc)
            stdev = ttnn.sqrt(ttnn.add(variance, self.config.revin_eps, memory_config=mc))
            normalized = ttnn.div(centered, stdev, memory_config=mc)

        projected = apply_linear(
            ttnn.permute(normalized, (0, 1, 3, 2)),
            self._packed_rlinear_parameters["projection"],
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
            memory_config=mc,
            compute_kernel_config=LOW_FIDELITY_LINEAR_COMPUTE_KERNEL_CONFIG,
        )
        projected = ttnn.reshape(
            projected,
            (1, batch_size, self.config.input_dim * self.config.num_experts, self.config.pred_len),
        )
<<<<<<< HEAD
        projected = ttnn.multiply(projected, pr.scale, memory_config=mc)
        projected = ttnn.add(projected, pr.constant, memory_config=mc)
=======
        projected = ttnn.multiply(projected, self._packed_rlinear_parameters["scale"], memory_config=mc)
        projected = ttnn.add(projected, self._packed_rlinear_parameters["constant"], memory_config=mc)
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
        projected = ttnn.reshape(
            projected,
            (1, batch_size * self.config.input_dim, self.config.num_experts, self.config.pred_len),
        )
        projected = ttnn.permute(projected, (0, 1, 3, 2))
        return projected, mean, stdev

    def _ensure_packed_rmlp_parameters(self) -> None:
        if self._packed_rmlp_parameters is not None:
            return
        packed_parameter_memory_config = ttnn.DRAM_MEMORY_CONFIG

        def block_diagonal_weights(modules) -> torch.Tensor:
            return torch.block_diag(*[module.weight.detach() for module in modules])

        def concatenated_biases(modules) -> torch.Tensor:
            return torch.cat([module.bias.detach() for module in modules], dim=0)

        temporal_1_modules = [expert.temporal[0] for expert in self._reference_experts]
        temporal_2_modules = [expert.temporal[2] for expert in self._reference_experts]
        projection_modules = [expert.projection for expert in self._reference_experts]

        temporal_1_weight = block_diagonal_weights(temporal_1_modules)
        temporal_1_bias = concatenated_biases(temporal_1_modules)
        temporal_2_weight = block_diagonal_weights(temporal_2_modules)
        temporal_2_bias = concatenated_biases(temporal_2_modules)
        projection_weight = block_diagonal_weights(projection_modules)
        projection_bias = concatenated_biases(projection_modules)

        affine_weight_blocks = []
        affine_bias_blocks = []
        output_bias_blocks = []
        output_denom_blocks = []
        seq_len = self.config.seq_len
        pred_len = self.config.pred_len
        for expert in self._reference_experts:
<<<<<<< HEAD
            affine_weight = expert.rev.affine_weight.detach()
            affine_bias = expert.rev.affine_bias.detach()
            output_bias = affine_bias
            output_denom = affine_weight + (self.config.revin_eps**2)
=======
            if expert.rev is None:
                affine_weight = torch.ones(
                    self.config.input_dim, dtype=projection_bias.dtype, device=projection_bias.device
                )
                affine_bias = torch.zeros(
                    self.config.input_dim, dtype=projection_bias.dtype, device=projection_bias.device
                )
                output_bias = torch.zeros(
                    self.config.input_dim, dtype=projection_bias.dtype, device=projection_bias.device
                )
                output_denom = torch.ones(
                    self.config.input_dim, dtype=projection_bias.dtype, device=projection_bias.device
                )
            else:
                affine_weight = expert.rev.affine_weight.detach()
                affine_bias = expert.rev.affine_bias.detach()
                output_bias = affine_bias
                output_denom = affine_weight + (self.config.revin_eps**2)
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f

            affine_weight_blocks.append(affine_weight.unsqueeze(1).repeat(1, seq_len))
            affine_bias_blocks.append(affine_bias.unsqueeze(1).repeat(1, seq_len))
            output_bias_blocks.append(output_bias.unsqueeze(1).repeat(1, pred_len))
            output_denom_blocks.append(output_denom.unsqueeze(1).repeat(1, pred_len))

        packed_affine_weight = torch.stack(affine_weight_blocks, dim=1).reshape(
            1, 1, self.config.input_dim, self.config.num_experts * seq_len
        )
        packed_affine_bias = torch.stack(affine_bias_blocks, dim=1).reshape(
            1, 1, self.config.input_dim, self.config.num_experts * seq_len
        )
        packed_output_bias = torch.stack(output_bias_blocks, dim=1).reshape(
            1, 1, self.config.input_dim, self.config.num_experts * pred_len
        )
        packed_output_denom = torch.stack(output_denom_blocks, dim=1).reshape(
            1, 1, self.config.input_dim, self.config.num_experts * pred_len
        )

<<<<<<< HEAD
        self._packed_rmlp_parameters = _PackedRmlpParams(
            temporal_1=upload_linear(
=======
        self._packed_rmlp_parameters = {
            "temporal_1": upload_linear(
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
                temporal_1_weight,
                temporal_1_bias,
                device=self.device,
                dtype=self.dtype,
                memory_config=packed_parameter_memory_config,
            ),
<<<<<<< HEAD
            temporal_2=upload_linear(
=======
            "temporal_2": upload_linear(
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
                temporal_2_weight,
                temporal_2_bias,
                device=self.device,
                dtype=self.dtype,
                memory_config=packed_parameter_memory_config,
            ),
<<<<<<< HEAD
            projection=upload_linear(
=======
            "projection": upload_linear(
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
                projection_weight,
                projection_bias,
                device=self.device,
                dtype=self.dtype,
                memory_config=packed_parameter_memory_config,
            ),
<<<<<<< HEAD
            affine_weight=ttnn.from_torch(
=======
            "affine_weight": ttnn.from_torch(
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
                packed_affine_weight,
                device=self.device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=packed_parameter_memory_config,
            ),
<<<<<<< HEAD
            affine_bias=ttnn.from_torch(
=======
            "affine_bias": ttnn.from_torch(
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
                packed_affine_bias,
                device=self.device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=packed_parameter_memory_config,
            ),
<<<<<<< HEAD
            output_bias=ttnn.from_torch(
=======
            "output_bias": ttnn.from_torch(
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
                packed_output_bias,
                device=self.device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=packed_parameter_memory_config,
            ),
<<<<<<< HEAD
            output_denom=ttnn.from_torch(
=======
            "output_denom": ttnn.from_torch(
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
                packed_output_denom,
                device=self.device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=packed_parameter_memory_config,
            ),
<<<<<<< HEAD
        )
=======
        }
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f

    def _compute_rmlp_expert_projection(
        self,
        input_tensor: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None, ttnn.Tensor | None]:
        self._ensure_packed_rmlp_parameters()
        mc = self.activation_memory_config
        batch_size = input_tensor.shape[1]

        normalized, mean, stdev = self._compute_shared_normalized_input(input_tensor)
        packed_input = ttnn.permute(normalized, (0, 1, 3, 2))
        packed_input = ttnn.concat([packed_input] * self.config.num_experts, dim=3, memory_config=mc)
<<<<<<< HEAD
        pm = self._packed_rmlp_parameters
        packed_input = ttnn.multiply(packed_input, pm.affine_weight, memory_config=mc)
        packed_input = ttnn.add(packed_input, pm.affine_bias, memory_config=mc)

        temporal_hidden = apply_linear(
            packed_input,
            pm.temporal_1,
=======
        packed_input = ttnn.multiply(packed_input, self._packed_rmlp_parameters["affine_weight"], memory_config=mc)
        packed_input = ttnn.add(packed_input, self._packed_rmlp_parameters["affine_bias"], memory_config=mc)

        temporal_hidden = apply_linear(
            packed_input,
            self._packed_rmlp_parameters["temporal_1"],
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
            memory_config=mc,
        )
        temporal_hidden = ttnn.relu(temporal_hidden)
        temporal_hidden = apply_linear(
            temporal_hidden,
<<<<<<< HEAD
            pm.temporal_2,
=======
            self._packed_rmlp_parameters["temporal_2"],
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
            memory_config=mc,
        )
        temporal_hidden = ttnn.add(temporal_hidden, packed_input, memory_config=mc)

        projected = apply_linear(
            temporal_hidden,
<<<<<<< HEAD
            pm.projection,
            memory_config=mc,
        )
        projected = ttnn.subtract(projected, pm.output_bias, memory_config=mc)
        projected = ttnn.div(projected, pm.output_denom, memory_config=mc)
=======
            self._packed_rmlp_parameters["projection"],
            memory_config=mc,
        )
        projected = ttnn.subtract(projected, self._packed_rmlp_parameters["output_bias"], memory_config=mc)
        projected = ttnn.div(projected, self._packed_rmlp_parameters["output_denom"], memory_config=mc)
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
        projected = ttnn.reshape(
            projected,
            (1, batch_size, self.config.input_dim, self.config.num_experts, self.config.pred_len),
        )
        projected = ttnn.permute(projected, (0, 1, 2, 4, 3))
        projected = ttnn.reshape(
            projected,
            (1, batch_size * self.config.input_dim, self.config.pred_len, self.config.num_experts),
        )
        return projected, mean, stdev

    def _compute_shared_normalized_input(
        self,
        input_tensor: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None, ttnn.Tensor | None]:
<<<<<<< HEAD
        mc = self.activation_memory_config
        mean = ttnn.sum(input_tensor, dim=2, keepdim=True, scalar=1.0 / self.config.seq_len, memory_config=mc)
        centered = ttnn.subtract(input_tensor, mean, memory_config=mc)
        variance = ttnn.multiply(centered, centered, memory_config=mc)
        variance = ttnn.sum(variance, dim=2, keepdim=True, scalar=1.0 / self.config.seq_len, memory_config=mc)
        stdev = ttnn.sqrt(ttnn.add(variance, self.config.revin_eps, memory_config=mc))
        normalized = ttnn.div(centered, stdev, memory_config=mc)
        return normalized, mean, stdev

    def _reduce_prediction_heads(
        self, projected: ttnn.Tensor, gating_flat: ttnn.Tensor, batch_size: int
    ) -> ttnn.Tensor:
        return reduce_weighted_heads_batch_major(
            projected,
            gating_flat,
            batch_size=batch_size,
            channels=self.config.input_dim,
            pred_len=self.config.pred_len,
            num_predictions=self.config.num_experts,
            memory_config=self.activation_memory_config,
        )

    def _denormalize_prediction_if_needed(
        self,
        prediction: ttnn.Tensor,
        mean: ttnn.Tensor | None,
        stdev: ttnn.Tensor | None,
    ) -> ttnn.Tensor:
        if stdev is None or mean is None:
            return prediction
        mc = self.activation_memory_config
        prediction = ttnn.multiply(prediction, stdev, memory_config=mc)
        return ttnn.add(prediction, mean, memory_config=mc)

    def _predict_packed_dlinear(self, input_tensor: ttnn.Tensor, gating_flat: ttnn.Tensor, batch_size: int) -> ttnn.Tensor:
        projected = self._compute_dlinear_expert_projection(input_tensor)
        return self._reduce_prediction_heads(projected, gating_flat, batch_size)

    def _predict_packed_rlinear(self, input_tensor: ttnn.Tensor, gating_flat: ttnn.Tensor, batch_size: int) -> ttnn.Tensor:
        projected, mean, stdev = self._compute_rlinear_expert_projection(input_tensor)
        prediction = self._reduce_prediction_heads(projected, gating_flat, batch_size)
        return self._denormalize_prediction_if_needed(prediction, mean, stdev)

    def _predict_packed_rmlp(self, input_tensor: ttnn.Tensor, gating_flat: ttnn.Tensor, batch_size: int) -> ttnn.Tensor:
        projected, mean, stdev = self._compute_rmlp_expert_projection(input_tensor)
        prediction = self._reduce_prediction_heads(projected, gating_flat, batch_size)
        return self._denormalize_prediction_if_needed(prediction, mean, stdev)

    def _predict_individual_experts(
        self,
        input_tensor: ttnn.Tensor,
        input_marks: ttnn.Tensor,
        gating_weights: ttnn.Tensor,
        batch_size: int,
    ) -> ttnn.Tensor:
        shared_normalized = None
        mean = None
        stdev = None
        if self.config.base_model_type == "rmlp":
            shared_normalized, mean, stdev = self._compute_shared_normalized_input(input_tensor)

        prediction = None
        experts = self.experts
        if experts is None:
            raise RuntimeError("individual expert path requires non-empty expert modules")
        for expert_index, expert in enumerate(experts):
            if shared_normalized is None:
                expert_prediction = expert.forward(input_tensor, input_marks)
            else:
                expert._ensure_parameters(input_tensor.device())
                expert_prediction = expert._forward_normalized_prediction(shared_normalized)
            expert_weight = ttnn.slice(
                gating_weights,
                (0, 0, expert_index, 0),
                (1, batch_size, expert_index + 1, self.config.input_dim),
            )
            weighted_prediction = ttnn.multiply(
                expert_prediction,
                expert_weight,
                memory_config=self.activation_memory_config,
            )
            prediction = (
                weighted_prediction
                if prediction is None
                else ttnn.add(prediction, weighted_prediction, memory_config=self.activation_memory_config)
            )
        if prediction is None:
            raise RuntimeError("expected at least one expert prediction")
        if shared_normalized is not None and mean is not None and stdev is not None:
            prediction = ttnn.multiply(prediction, stdev, memory_config=self.activation_memory_config)
            prediction = ttnn.add(prediction, mean, memory_config=self.activation_memory_config)
        return prediction

    def _prediction_and_gating_weights(
        self,
        input_tensor: ttnn.Tensor,
        input_marks: ttnn.Tensor,
=======
        mean = None
        stdev = None
        normalized = input_tensor
        if self._reference_experts[0].rev is not None:
            mc = self.activation_memory_config
            mean = ttnn.sum(input_tensor, dim=2, keepdim=True, scalar=1.0 / self.config.seq_len, memory_config=mc)
            centered = ttnn.subtract(input_tensor, mean, memory_config=mc)
            variance = ttnn.multiply(centered, centered, memory_config=mc)
            variance = ttnn.sum(variance, dim=2, keepdim=True, scalar=1.0 / self.config.seq_len, memory_config=mc)
            stdev = ttnn.sqrt(ttnn.add(variance, self.config.revin_eps, memory_config=mc))
            normalized = ttnn.div(centered, stdev, memory_config=mc)
        return normalized, mean, stdev

    def _prediction_and_gating_weights(
        self,
        input_tensor: ttnn.Tensor,
        input_marks: ttnn.Tensor,
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
        *,
        return_channelwise_weights: bool = True,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        validate_timeseries_input(input_tensor, seq_len=self.config.seq_len, input_dim=self.config.input_dim)
<<<<<<< HEAD
        validate_time_marks(
            input_marks,
            seq_len=self.config.seq_len,
            expected_features=self._expected_time_features(),
        )
=======
        validate_time_marks(input_marks, seq_len=self.config.seq_len)
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
        self._ensure_router_parameters()

        batch_size = input_tensor.shape[1]
        gating_logits = apply_two_layer_mlp(
            extract_initial_marks(input_marks),
            self._tt_router_parameters,
            memory_config=self.activation_memory_config,
        )
        gating_flat, gating_weights = temporal_gating(
            gating_logits,
            batch_size=batch_size,
            channels=self.config.input_dim,
            num_predictions=self.config.num_experts,
            return_channelwise_weights=return_channelwise_weights,
        )
        if gating_weights is None and self.experts is not None:
            gating_weights = ttnn.reshape(gating_flat, (1, batch_size, self.config.input_dim, self.config.num_experts))
            gating_weights = ttnn.permute(gating_weights, (0, 1, 3, 2))

<<<<<<< HEAD
        c = self.config
        if c.base_model_type == "dlinear":
            prediction = self._predict_packed_dlinear(input_tensor, gating_flat, batch_size)
        elif c.base_model_type == "rlinear" and not c.individual:
            prediction = self._predict_packed_rlinear(input_tensor, gating_flat, batch_size)
        elif c.base_model_type == "rmlp" and not c.individual:
            prediction = self._predict_packed_rmlp(input_tensor, gating_flat, batch_size)
        else:
            if gating_weights is None:
                raise RuntimeError("individual expert path requires channel-wise gating weights")
            prediction = self._predict_individual_experts(
                input_tensor, input_marks, gating_weights, batch_size
            )
        return prediction, gating_weights

=======
        if self.config.base_model_type == "dlinear":
            projected = self._compute_dlinear_expert_projection(input_tensor)
            prediction = reduce_weighted_heads_batch_major(
                projected,
                gating_flat,
                batch_size=batch_size,
                channels=self.config.input_dim,
                pred_len=self.config.pred_len,
                num_predictions=self.config.num_experts,
                memory_config=self.activation_memory_config,
            )
        elif self.config.base_model_type == "rlinear" and not self.config.individual:
            projected, mean, stdev = self._compute_rlinear_expert_projection(input_tensor)
            prediction = reduce_weighted_heads_batch_major(
                projected,
                gating_flat,
                batch_size=batch_size,
                channels=self.config.input_dim,
                pred_len=self.config.pred_len,
                num_predictions=self.config.num_experts,
                memory_config=self.activation_memory_config,
            )
            if stdev is not None and mean is not None:
                prediction = ttnn.multiply(prediction, stdev, memory_config=self.activation_memory_config)
                prediction = ttnn.add(prediction, mean, memory_config=self.activation_memory_config)
        elif self.config.base_model_type == "rmlp" and not self.config.individual:
            projected, mean, stdev = self._compute_rmlp_expert_projection(input_tensor)
            prediction = reduce_weighted_heads_batch_major(
                projected,
                gating_flat,
                batch_size=batch_size,
                channels=self.config.input_dim,
                pred_len=self.config.pred_len,
                num_predictions=self.config.num_experts,
                memory_config=self.activation_memory_config,
            )
            if stdev is not None and mean is not None:
                prediction = ttnn.multiply(prediction, stdev, memory_config=self.activation_memory_config)
                prediction = ttnn.add(prediction, mean, memory_config=self.activation_memory_config)
        else:
            shared_normalized = None
            mean = None
            stdev = None
            if self.config.base_model_type == "rmlp":
                shared_normalized, mean, stdev = self._compute_shared_normalized_input(input_tensor)

            prediction = None
            for expert_index, expert in enumerate(self.experts):
                if shared_normalized is None:
                    expert_prediction = expert.forward(input_tensor, input_marks)
                else:
                    expert._ensure_parameters(input_tensor.device())
                    expert_prediction = expert._forward_normalized_prediction(shared_normalized)
                expert_weight = ttnn.slice(
                    gating_weights,
                    (0, 0, expert_index, 0),
                    (1, batch_size, expert_index + 1, self.config.input_dim),
                )
                weighted_prediction = ttnn.multiply(
                    expert_prediction,
                    expert_weight,
                    memory_config=self.activation_memory_config,
                )
                prediction = (
                    weighted_prediction
                    if prediction is None
                    else ttnn.add(prediction, weighted_prediction, memory_config=self.activation_memory_config)
                )
            if prediction is None:
                raise RuntimeError("expected at least one expert prediction")
            if shared_normalized is not None and mean is not None and stdev is not None:
                prediction = ttnn.multiply(prediction, stdev, memory_config=self.activation_memory_config)
                prediction = ttnn.add(prediction, mean, memory_config=self.activation_memory_config)
        return prediction, gating_weights

    def _compute_router_weights(self, input_marks: ttnn.Tensor) -> ttnn.Tensor:
        validate_time_marks(input_marks, seq_len=self.config.seq_len)
        self._ensure_router_parameters()
        batch_size = input_marks.shape[1]
        gating_logits = apply_two_layer_mlp(
            extract_initial_marks(input_marks),
            self._tt_router_parameters,
            memory_config=self.activation_memory_config,
        )
        _, gating_weights = temporal_gating(
            gating_logits,
            batch_size=batch_size,
            channels=self.config.input_dim,
            num_predictions=self.config.num_experts,
            return_channelwise_weights=True,
        )
        return gating_weights

>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
    def forward_prediction_no_trace(self, input_tensor: ttnn.Tensor, input_marks: ttnn.Tensor) -> ttnn.Tensor:
        prediction, _ = self._prediction_and_gating_weights(
            input_tensor,
            input_marks,
            return_channelwise_weights=False,
        )
        return prediction

    def forward_prediction(self, input_tensor: ttnn.Tensor, input_marks: ttnn.Tensor) -> ttnn.Tensor:
        if not self._trace_capture_enabled:
            return self.forward_prediction_no_trace(input_tensor, input_marks)

        state = self._prediction_trace_state
        current_ids = (id(input_tensor), id(input_marks))

        if state is None or state.input_ids != current_ids:
            self._release_prediction_trace()
            self._release_forward_trace()

            prediction = self.forward_prediction_no_trace(input_tensor, input_marks)
            ttnn.synchronize_device(self.device)

            trace_id = None
            try:
                trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
                prediction = self.forward_prediction_no_trace(input_tensor, input_marks)
                ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            except Exception:
                self._trace_capture_enabled = False
                if trace_id is not None:
<<<<<<< HEAD
                    with contextlib.suppress(Exception):
                        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
                        ttnn.release_trace(self.device, trace_id)
=======
                    try:
                        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
                        ttnn.release_trace(self.device, trace_id)
                    except Exception:
                        pass
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
                self._release_prediction_trace()
                return prediction
            self._prediction_trace_state = _MoLEPredictionTraceState(
                trace_id=trace_id,
                prediction=prediction,
                input_ids=current_ids,
            )

        pst = self._prediction_trace_state
        ttnn.execute_trace(self.device, pst.trace_id, cq_id=0, blocking=False)
        return pst.prediction

    def forward_no_trace(self, input_tensor: ttnn.Tensor, input_marks: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        prediction, gating_weights = self._prediction_and_gating_weights(
            input_tensor,
            input_marks,
            return_channelwise_weights=True,
        )
        if gating_weights is None:
            raise RuntimeError("full forward expects channel-wise router weights")
        return prediction, gating_weights

<<<<<<< HEAD
=======
    def forward_no_trace(self, input_tensor: ttnn.Tensor, input_marks: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        prediction, _ = self._prediction_and_gating_weights(
            input_tensor,
            input_marks,
            return_channelwise_weights=False,
        )
        return prediction, self._compute_router_weights(input_marks)

>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
    def forward(self, input_tensor: ttnn.Tensor, input_marks: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if not self._trace_capture_enabled:
            return self.forward_no_trace(input_tensor, input_marks)

        state = self._forward_trace_state
        current_ids = (id(input_tensor), id(input_marks))

<<<<<<< HEAD
        if state is None or state.input_ids != current_ids:
=======
        if state is None or state["input_ids"] != current_ids:
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f
            self._release_forward_trace()
            self._release_prediction_trace()

            output = self.forward_no_trace(input_tensor, input_marks)
            ttnn.synchronize_device(self.device)

            trace_id = None
            try:
                trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
                output = self.forward_no_trace(input_tensor, input_marks)
                ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            except Exception:
                self._trace_capture_enabled = False
                if trace_id is not None:
<<<<<<< HEAD
                    with contextlib.suppress(Exception):
                        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
                        ttnn.release_trace(self.device, trace_id)
                self._release_forward_trace()
                return output
            self._forward_trace_state = _MoLEForwardTraceState(
                trace_id=trace_id,
                output=output,
                input_ids=current_ids,
            )

        fst = self._forward_trace_state
        ttnn.execute_trace(self.device, fst.trace_id, cq_id=0, blocking=False)
        return fst.output
=======
                    try:
                        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
                        ttnn.release_trace(self.device, trace_id)
                    except Exception:
                        pass
                self._release_forward_trace()
                return output
            self._forward_trace_state = {
                "trace_id": trace_id,
                "output": output,
                "input_ids": current_ids,
            }

        ttnn.execute_trace(self.device, self._forward_trace_state["trace_id"], cq_id=0, blocking=False)
        return self._forward_trace_state["output"]
>>>>>>> 832f8d006a67a76ebe4bbdf3ffb366344dc9940f

    def forward_from_torch_input(
        self,
        torch_input: torch.Tensor,
        *,
        input_marks: torch.Tensor,
        device=None,
        return_router_output: bool = True,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
        if device is None:
            device = self.device
        elif device is not self.device:
            raise ValueError(
                "TtMoLE.forward_from_torch_input: 'device' must match the device this model was constructed with."
            )
        tt_input, tt_marks = upload_timeseries_and_marks_to_device(
            model=self,
            device=device,
            torch_input=torch_input,
            torch_input_mark=input_marks,
            memory_config=self.activation_memory_config,
        )
        if return_router_output:
            return self.forward(tt_input, tt_marks)
        return self.forward_prediction(tt_input, tt_marks)
