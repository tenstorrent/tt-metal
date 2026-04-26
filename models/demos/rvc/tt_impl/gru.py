# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn


@dataclass
class _LinearParams:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None


class GRU:
    """Inference-oriented GRU implemented from TT linear and elementwise ops."""

    def __init__(
        self,
        device: ttnn.MeshDevice,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        bidirectional: bool = False,
        dtype: ttnn.DataType | None = None,
    ) -> None:
        if not batch_first:
            raise ValueError("Only batch_first=True is supported")

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dtype = ttnn.bfloat16 if dtype is None else dtype
        self.num_directions = 2 if bidirectional else 1

        self._params: dict[tuple[int, int], dict[str, _LinearParams]] = {}
        self._one_cache: dict[int, ttnn.Tensor] = {}

    def _make_linear_params(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> _LinearParams:
        transposed_weight = weight.detach().to(torch.float32).transpose(0, 1).contiguous()
        weight_tensor = ttnn.from_torch(
            transposed_weight,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        bias_tensor = None
        if bias is not None:
            bias_tensor = ttnn.from_torch(
                bias.detach().to(torch.float32).reshape(1, 1, -1).contiguous(),
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return _LinearParams(weight=weight_tensor, bias=bias_tensor)

    def _apply_linear(self, x: ttnn.Tensor, params: _LinearParams) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        output_memory_config = (
            ttnn.DRAM_MEMORY_CONFIG if x.memory_config() == ttnn.DRAM_MEMORY_CONFIG else ttnn.L1_MEMORY_CONFIG
        )
        return ttnn.linear(
            x,
            params.weight,
            bias=params.bias,
            dtype=self.dtype,
            memory_config=output_memory_config,
        )

    def _get_one_tensor(self, hidden_size: int) -> ttnn.Tensor:
        if hidden_size not in self._one_cache:
            self._one_cache[hidden_size] = ttnn.from_torch(
                torch.ones((1, 1, hidden_size), dtype=torch.float32),
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._one_cache[hidden_size]

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], key: str, module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        base_key = f"{module_prefix}{key}" if module_prefix else key

        self._params.clear()
        layer_input_size = self.input_size
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                suffix = "_reverse" if direction == 1 else ""
                weight_ih = state_dict[f"{base_key}.weight_ih_l{layer}{suffix}"].detach().to(torch.float32).contiguous()
                weight_hh = state_dict[f"{base_key}.weight_hh_l{layer}{suffix}"].detach().to(torch.float32).contiguous()

                bias_ih = None
                bias_hh = None
                if self.bias:
                    bias_ih = state_dict[f"{base_key}.bias_ih_l{layer}{suffix}"].detach().to(torch.float32).contiguous()
                    bias_hh = state_dict[f"{base_key}.bias_hh_l{layer}{suffix}"].detach().to(torch.float32).contiguous()

                w_ir, w_iz, w_in = torch.chunk(weight_ih, 3, dim=0)
                w_hr, w_hz, w_hn = torch.chunk(weight_hh, 3, dim=0)

                if self.bias:
                    b_ir, b_iz, b_in = torch.chunk(bias_ih, 3, dim=0)
                    b_hr, b_hz, b_hn = torch.chunk(bias_hh, 3, dim=0)
                else:
                    b_ir = b_iz = b_in = None
                    b_hr = b_hz = b_hn = None

                self._params[(layer, direction)] = {
                    "x_r": self._make_linear_params(w_ir, b_ir),
                    "h_r": self._make_linear_params(w_hr, b_hr),
                    "x_z": self._make_linear_params(w_iz, b_iz),
                    "h_z": self._make_linear_params(w_hz, b_hz),
                    "x_n": self._make_linear_params(w_in, b_in),
                    "h_n": self._make_linear_params(w_hn, b_hn),
                }
            layer_input_size = self.hidden_size * self.num_directions

    def _gru_step(self, x_t: ttnn.Tensor, h_prev: ttnn.Tensor, params: dict[str, _LinearParams]) -> ttnn.Tensor:
        r = ttnn.sigmoid(ttnn.add(self._apply_linear(x_t, params["x_r"]), self._apply_linear(h_prev, params["h_r"])))
        z = ttnn.sigmoid(ttnn.add(self._apply_linear(x_t, params["x_z"]), self._apply_linear(h_prev, params["h_z"])))

        n_x = self._apply_linear(x_t, params["x_n"])
        n_h = self._apply_linear(h_prev, params["h_n"])
        n = ttnn.tanh(ttnn.add(n_x, ttnn.multiply(r, n_h)))

        one_minus_z = ttnn.subtract(self._get_one_tensor(self.hidden_size), z)
        return ttnn.add(ttnn.multiply(one_minus_z, n), ttnn.multiply(z, h_prev))

    def _run_direction(
        self,
        layer_input: ttnn.Tensor,
        h0: ttnn.Tensor,
        params: dict[str, _LinearParams],
        reverse: bool,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        batch_size, sequence_length, _ = layer_input.shape
        h_t = h0
        outputs = []
        time_indices = range(sequence_length - 1, -1, -1) if reverse else range(sequence_length)

        for t in time_indices:
            x_t = ttnn.slice(
                layer_input,
                (0, t, 0),
                (batch_size, t + 1, layer_input.shape[2]),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            h_t = self._gru_step(x_t, h_t, params)
            outputs.append(h_t)

        if reverse:
            outputs.reverse()

        direction_output = ttnn.concat(outputs, dim=1)
        return direction_output, h_t

    def __call__(
        self, input_tensor: ttnn.Tensor, hidden_state: ttnn.Tensor | None = None
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        batch_size, sequence_length, _ = input_tensor.shape

        if hidden_state is None:
            hidden_state = ttnn.from_torch(
                torch.zeros(
                    (self.num_layers * self.num_directions, batch_size, self.hidden_size),
                    dtype=torch.float32,
                ),
                dtype=self.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
        else:
            hidden_state = ttnn.to_layout(hidden_state, ttnn.ROW_MAJOR_LAYOUT)

        layer_input = input_tensor
        final_hidden_states = []

        for layer in range(self.num_layers):
            direction_outputs = []
            for direction in range(self.num_directions):
                hidden_index = layer * self.num_directions + direction
                h0 = ttnn.slice(
                    hidden_state,
                    (hidden_index, 0, 0),
                    (hidden_index + 1, batch_size, self.hidden_size),
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                h0 = ttnn.reshape(h0, (batch_size, 1, self.hidden_size))

                direction_output, h_n = self._run_direction(
                    layer_input=layer_input,
                    h0=h0,
                    params=self._params[(layer, direction)],
                    reverse=direction == 1,
                )
                direction_outputs.append(direction_output)
                final_hidden_states.append(ttnn.reshape(h_n, (1, batch_size, self.hidden_size)))

            layer_input = direction_outputs[0] if self.num_directions == 1 else ttnn.concat(direction_outputs, dim=2)

        final_hidden = ttnn.concat(final_hidden_states, dim=0)
        return layer_input, final_hidden


class TorchWrappedGRU:
    """Torch GRU backend with the same TT tensor interface as the TT implementation."""

    def __init__(
        self,
        device: ttnn.MeshDevice,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        bidirectional: bool = False,
        dtype: ttnn.DataType | None = None,
    ) -> None:
        if not batch_first:
            raise ValueError("Only batch_first=True is supported")

        self.device = device
        if self.device.get_num_devices() > 1:
            self.input_mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=0)
            self.output_mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
        else:
            self.input_mesh_mapper = None
            self.output_mesh_composer = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dtype = ttnn.bfloat16 if dtype is None else dtype

        self.torch_gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional,
        ).eval()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], key: str, module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        base_key = f"{module_prefix}{key}" if module_prefix else key

        gru_state_dict = {}
        num_directions = 2 if self.bidirectional else 1
        for layer in range(self.num_layers):
            for direction in range(num_directions):
                suffix = "_reverse" if direction == 1 else ""
                for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    full_key = f"{base_key}.{name}_l{layer}{suffix}"
                    if name.startswith("bias") and not self.bias:
                        continue
                    if full_key not in state_dict:
                        raise KeyError(f"Missing required parameter: {full_key}")
                    gru_state_dict[f"{name}_l{layer}{suffix}"] = (
                        state_dict[full_key].detach().to(torch.float32).contiguous()
                    )

        self.torch_gru.load_state_dict(gru_state_dict, strict=True)
        self.torch_gru.eval()

    def __call__(
        self, input_tensor: ttnn.Tensor, hidden_state: ttnn.Tensor | None = None
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        torch_input = ttnn.to_torch(input_tensor, mesh_composer=self.output_mesh_composer).to(torch.float32)
        torch_hidden = None
        if hidden_state is not None:
            torch_hidden = ttnn.to_torch(hidden_state).to(torch.float32)

        with torch.no_grad():
            output, hidden = self.torch_gru(torch_input, torch_hidden)

        tt_output = ttnn.from_torch(
            output,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.input_mesh_mapper,
        )
        tt_hidden = ttnn.from_torch(
            hidden,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.input_mesh_mapper,
        )
        return tt_output, tt_hidden
