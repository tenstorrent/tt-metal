# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Phi-1 Model Implementation for Tenstorrent Wormhole

Implements the full Phi-1 model using tt-metal primitives.
Leverages ttnn for tensor operations and device management.
"""

import torch
import ttnn
from typing import Optional, Tuple
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.demos.wormhole.phi_1.tt.phi_decoder import TtPhiDecoderLayer
from models.demos.wormhole.phi_1.tt.phi_embedding import TtPhiEmbeddings
from models.demos.wormhole.phi_1.config import PhiConfig
import tt_lib as ttl


class TtPhi:
    def __init__(
        self,
        device: ttnn.Device,
        state_dict: dict,
        base_url: str,
        max_position_embeddings: int,
        config: PhiConfig,
        rotary_dim: int,
        use_xformer_rotary: bool = False,
    ):
        self.device = device
        self.state_dict = state_dict
        self.base_url = base_url
        self.max_position_embeddings = max_position_embeddings
        self.config = config
        self.rotary_dim = rotary_dim
        self.use_xformer_rotary = use_xformer_rotary

        # Load embeddings
        self.embed_tokens = TtPhiEmbeddings(
            device=device,
            state_dict=state_dict,
            base_url=base_url,
            config=config,
        )

        # Load decoder layers
        self.layers = []
        for layer in range(config.num_hidden_layers):
            self.layers.append(
                TtPhiDecoderLayer(
                    device=device,
                    state_dict=state_dict,
                    base_url=f"{base_url}.h",
                    layer_num=layer,
                    config=config,
                    max_position_embeddings=max_position_embeddings,
                    rotary_dim=rotary_dim,
                    use_xformer_rotary=use_xformer_rotary,
                )
            )

        # Final layer norm
        self.final_layernorm_weight = torch2tt_tensor(
            state_dict[f"{base_url}.ln_f.weight"], device, tt_memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.final_layernorm_bias = torch2tt_tensor(
            state_dict[f"{base_url}.ln_f