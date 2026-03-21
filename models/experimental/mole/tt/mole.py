# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.mole.reference.config import MoLEConfig
from models.experimental.mole.reference.mole import MixtureOfLinearExperts
from models.experimental.mole.tt.common import (
    TtRuntimeOptions,
    default_activation_memory_config,
    register_trace_release_hook,
    upload_timeseries_and_marks_to_device,
    validate_time_marks,
    validate_timeseries_input,
)
from models.experimental.mole.tt.dlinear import TtDLinearExpert
from models.experimental.mole.tt.rlinear import TtRLinearExpert
from models.experimental.mole.tt.rmlp import TtRMLPExpert


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
        # Kept for compatibility with existing demo callers.
        self.memory_config = self.activation_memory_config
        self.dtype = options.dtype
        expert_runtime_options = TtRuntimeOptions(
            memory_config=self.parameter_memory_config,
            activation_memory_config=self.activation_memory_config,
            dtype=self.dtype,
        )
        if config.base_model_type == "dlinear":
            self.model = TtDLinearExpert(
                config,
                reference_model=reference_model.model,
                runtime_options=expert_runtime_options,
            )
        elif config.base_model_type == "rlinear":
            self.model = TtRLinearExpert(
                config,
                reference_model=reference_model.model,
                runtime_options=expert_runtime_options,
            )
        elif config.base_model_type == "rmlp":
            self.model = TtRMLPExpert(
                config,
                reference_model=reference_model.model,
                runtime_options=expert_runtime_options,
            )
        else:
            raise ValueError(f"unsupported base_model_type: {config.base_model_type}")
        self._prediction_trace_state = None
        self._trace_capture_enabled = True
        register_trace_release_hook(device=self.device, hook=self._release_prediction_trace)

    def _release_prediction_trace(self) -> None:
        state = self._prediction_trace_state
        if state is None:
            return
        try:
            ttnn.release_trace(self.device, state["trace_id"])
        except Exception:
            pass
        self._prediction_trace_state = None

    def __del__(self):
        self._release_prediction_trace()

    def _prediction_and_gating_flat(
        self,
        input_tensor: ttnn.Tensor,
        input_marks: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        validate_timeseries_input(input_tensor, seq_len=self.config.seq_len, input_dim=self.config.input_dim)
        validate_time_marks(input_marks, seq_len=self.config.seq_len)
        prediction, _, gating_flat = self.model._forward_outputs(
            input_tensor,
            input_marks,
            return_channelwise_weights=False,
        )
        return prediction, gating_flat

    def forward_prediction(self, input_tensor: ttnn.Tensor, input_marks: ttnn.Tensor) -> ttnn.Tensor:
        if not self._trace_capture_enabled:
            prediction, _ = self._prediction_and_gating_flat(input_tensor, input_marks)
            return prediction

        state = self._prediction_trace_state
        current_ids = (id(input_tensor), id(input_marks))

        if state is None or state["input_ids"] != current_ids:
            self._release_prediction_trace()

            # Prime once before capture to avoid setup overhead inside the traced region.
            prediction, _ = self._prediction_and_gating_flat(input_tensor, input_marks)
            ttnn.synchronize_device(self.device)

            try:
                trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
                prediction, _ = self._prediction_and_gating_flat(input_tensor, input_marks)
                ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            except Exception:
                self._trace_capture_enabled = False
                try:
                    ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
                    ttnn.release_trace(self.device, trace_id)
                except Exception:
                    pass
                self._release_prediction_trace()
                return prediction
            self._prediction_trace_state = {
                "trace_id": trace_id,
                "prediction": prediction,
                "input_ids": current_ids,
            }

        ttnn.execute_trace(self.device, self._prediction_trace_state["trace_id"], cq_id=0, blocking=False)
        return self._prediction_trace_state["prediction"]

    def forward(self, input_tensor: ttnn.Tensor, input_marks: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        prediction, gating_flat = self._prediction_and_gating_flat(input_tensor, input_marks)

        # Fuse router-output path: average directly from [1, B*C, 1, T] flattened weights.
        batch_size = input_tensor.shape[1]
        averaged = ttnn.reshape(gating_flat, (1, batch_size, self.config.input_dim, self.config.t_dim))
        averaged = ttnn.sum(
            averaged,
            dim=2,
            keepdim=True,
            scalar=1.0 / self.config.input_dim,
            memory_config=self.activation_memory_config,
        )
        return prediction, ttnn.permute(averaged, (0, 2, 1, 3))

    def forward_from_torch_input(
        self,
        torch_input: torch.Tensor,
        *,
        input_marks: torch.Tensor,
        device,
        return_router_output: bool = True,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
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
