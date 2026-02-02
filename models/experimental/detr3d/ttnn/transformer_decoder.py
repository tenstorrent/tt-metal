# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.detr3d.ttnn.multihead_attention import TtnnMultiheadAttention
from dataclasses import dataclass, asdict


@dataclass
class DecoderLayerArgs:
    d_model: int = None
    nhead: int = 4
    dim_feedforward: int = 256
    normalize_before: bool = True


class TtnnTransformerDecoderLayer(LightweightModule):
    def __init__(
        self,
        device,
        d_model,
        nhead=4,
        dim_feedforward=256,
        normalize_before=True,
        parameters=None,
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.normalize_before = normalize_before

        # Initialize attention layers with parameters
        self_attn_params = parameters.get("self_attn") if parameters is not None else None
        multihead_attn_params = parameters.get("multihead_attn") if parameters is not None else None
        self.self_attn = TtnnMultiheadAttention(d_model, nhead, device, parameters=self_attn_params)
        self.multihead_attn = TtnnMultiheadAttention(d_model, nhead, device, parameters=multihead_attn_params)

        # Load preprocessed parameters
        if parameters is not None:
            self.load_parameters(parameters)
        else:
            # Initialize weights as None (for backward compatibility)
            self.self_attn_weights = None
            self.cross_attn_weights = None
            self.ff_weights1 = None
            self.ff_weights2 = None
            self.ff1_bias = None
            self.ff2_bias = None
            self.norm1_weights = None
            self.norm2_weights = None
            self.norm3_weights = None
            self.norm1_bias = None
            self.norm2_bias = None
            self.norm3_bias = None

    def load_parameters(self, parameters):
        """Load preprocessed parameters from the preprocessor"""
        # Feedforward weights
        if "linear1" in parameters:
            self.ff_weights1 = parameters["linear1"]["weight"]
            self.ff1_bias = parameters["linear1"].get("bias", None)
        if "linear2" in parameters:
            self.ff_weights2 = parameters["linear2"]["weight"]
            self.ff2_bias = parameters["linear2"].get("bias", None)

        # Normalization weights
        if "norm1" in parameters:
            self.norm1_weights = parameters["norm1"]["weight"]
            self.norm1_bias = parameters["norm1"].get("bias", None)
        if "norm2" in parameters:
            self.norm2_weights = parameters["norm2"]["weight"]
            self.norm2_bias = parameters["norm2"].get("bias", None)
        if "norm3" in parameters:
            self.norm3_weights = parameters["norm3"]["weight"]
            self.norm3_bias = parameters["norm3"].get("bias", None)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else ttnn.add(tensor, pos)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                pos,
                query_pos,
            )
        else:
            return self.forward_post(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                pos,
                query_pos,
            )

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt = ttnn.to_memory_config(tgt, ttnn.L1_MEMORY_CONFIG)
        q = k = self.with_pos_embed(tgt, query_pos)

        # Self-attention using proper QKV projections
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        tgt = ttnn.add(tgt, tgt2, memory_config=ttnn.L1_MEMORY_CONFIG)
        tgt = ttnn.layer_norm(tgt, weight=self.norm1_weights, bias=self.norm1_bias, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Cross-attention using proper QKV projections
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
        )
        tgt = ttnn.add(tgt, tgt2, memory_config=ttnn.L1_MEMORY_CONFIG)
        tgt = ttnn.layer_norm(tgt, weight=self.norm2_weights, bias=self.norm2_bias, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Feedforward
        tgt2 = ttnn.linear(
            tgt, self.ff_weights1, bias=self.ff1_bias, activation="relu", memory_config=ttnn.L1_MEMORY_CONFIG
        )
        tgt2 = ttnn.linear(tgt2, self.ff_weights2, bias=self.ff2_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        tgt = ttnn.add(tgt, tgt2, memory_config=ttnn.L1_MEMORY_CONFIG)
        tgt = ttnn.layer_norm(tgt, weight=self.norm3_weights, bias=self.norm3_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(tgt2)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt = ttnn.to_memory_config(tgt, ttnn.L1_MEMORY_CONFIG)

        # Pre-norm self-attention
        tgt2 = ttnn.layer_norm(
            tgt, weight=self.norm1_weights, bias=self.norm1_bias, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask)
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        tgt = ttnn.add(tgt, tgt2, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Pre-norm cross-attention
        tgt2 = ttnn.layer_norm(
            tgt, weight=self.norm2_weights, bias=self.norm2_bias, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
        )
        tgt = ttnn.add(tgt, tgt2, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Pre-norm feedforward
        tgt2 = ttnn.layer_norm(
            tgt, weight=self.norm3_weights, bias=self.norm3_bias, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        tgt2 = ttnn.linear(
            tgt2, self.ff_weights1, bias=self.ff1_bias, activation="relu", memory_config=ttnn.L1_MEMORY_CONFIG
        )
        tgt2 = ttnn.linear(tgt2, self.ff_weights2, bias=self.ff2_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        tgt = ttnn.add(tgt, tgt2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(tgt2)

        return tgt


class TtnnTransformerDecoder(LightweightModule):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        device=None,
        use_norm=True,
        return_intermediate=False,
        decoder_args=DecoderLayerArgs(),
        parameters=None,
    ):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        # Create multiple decoder layers
        self.layers = []
        for i in range(num_layers):
            layer_params = parameters.layers[i] if parameters else None
            layer = decoder_layer(
                device,
                **asdict(decoder_args),
                parameters=layer_params,
            )
            self.layers.append(layer)

        # Final layer norm
        self.norm = None
        if use_norm:
            self.norm_weights = parameters.get("norm", {}).get("weight") if parameters else None
            self.norm_bias = parameters.get("norm", {}).get("bias") if parameters else None

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        pos=None,
        query_pos=None,
        transpose_swap=False,
    ):
        # Handle transpose_swap for memory tensor
        if transpose_swap:
            # memory: bs, c, h, w -> b, t, c
            memory = ttnn.reshape(memory, (memory.shape[0], memory.shape[1], -1))
            memory = ttnn.permute(memory, (0, 2, 1))
            if pos is not None:
                pos = ttnn.reshape(pos, (pos.shape[0], pos.shape[1], -1))
                pos = ttnn.permute(pos, (0, 2, 1))

        output = tgt
        intermediate = []

        # Pass through each decoder layer
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                pos=pos,
                query_pos=query_pos,
            )

            if self.return_intermediate and self.norm_weights is not None:
                intermediate.append(ttnn.layer_norm(output, weight=self.norm_weights, bias=self.norm_bias))

        # Apply final layer norm
        if self.norm_weights is not None:
            output = ttnn.layer_norm(output, weight=self.norm_weights, bias=self.norm_bias)
            if self.return_intermediate:
                # Replace last intermediate result with final normed output
                if intermediate:
                    intermediate[-1] = output

        if self.return_intermediate:
            # TTNN doesn't support torch.stack directly, using ttnn.concat for now
            intermediate = [ttnn.unsqueeze(t, 0) for t in intermediate]
            intermediate = ttnn.concat(intermediate, dim=0, memory_config=ttnn.L1_MEMORY_CONFIG)
            return intermediate

        return output
