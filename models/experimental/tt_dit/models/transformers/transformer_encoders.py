# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from ...layers.encoder import CLIPEncoderLayer
from ...utils.substate import indexed_substates

# encoderparallel config

# ccl manager


class CLIPTextEncoderTransformer:
    def __init__(self, config, mesh_device=None, parallel_manager=None, ccl_manager=None, parallel_config=None):
        self.config = config
        self.mesh_device = mesh_device
        self.parallel_manager = parallel_manager
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self._layers = []
        self._d_model = config.hidden_size

    def load_state_dict(self, state_dict, ccl_manager=None, parallel_config=None):
        layer_states = indexed_substates(state_dict, "layers")

        for layer_state in layer_states:
            # placeholder for parameters, CLIPEncoderLayer constructor expects this
            parameters = type(
                "obj",
                (object,),
                {
                    "layer_norm1_weight": None,
                    "layer_norm1_bias": None,
                    "layer_norm2_weight": None,
                    "layer_norm2_bias": None,
                    "mesh_device": self.mesh_device,
                },
            )
            layer = CLIPEncoderLayer(
                parameters,
                self.config,
                self.mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
            )
            layer.load_state_dict(layer_state)
            self._layers.append(layer)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        mesh_device: ttnn.Device,
        causal_attention_mask: ttnn.Tensor = None,
        parallel_manager=None,
        ccl_manager=None,
        parallel_config=None,
        output_hidden_states: bool = True,
    ):
        all_hidden_states = []

        # create causal attention mask if not provided (following SD 3.5 pattern)
        if causal_attention_mask is None:
            input_shape = (hidden_states.shape[0], hidden_states.shape[1])  # (batch_size, seq_len)
            causal_attention_mask = _create_4d_causal_attention_mask(
                input_shape, mesh_device, dtype=hidden_states.get_dtype()
            )

        if output_hidden_states:  # this is the transformer input
            all_hidden_states.append(hidden_states)

        for layer in self._layers:
            hidden_states = layer(
                hidden_states, causal_attention_mask, ccl_manager=ccl_manager, parallel_config=parallel_config
            )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        if output_hidden_states:
            return hidden_states, all_hidden_states
        return hidden_states
