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
            self.norm1_weights = None
            self.norm2_weights = None
            self.norm3_weights = None

    def load_parameters(self, parameters):
        """Load preprocessed parameters from the preprocessor"""
        # Feedforward weights
        if "linear1" in parameters:
            self.ff_weights1 = parameters["linear1"]["weight"]
            self.ff1_bias = parameters["linear1"].get("bias")
        if "linear2" in parameters:
            self.ff_weights2 = parameters["linear2"]["weight"]
            self.ff2_bias = parameters["linear2"].get("bias")

        # Normalization weights
        for i, norm_name in enumerate(["norm1", "norm2", "norm3"], 1):
            if norm_name in parameters:
                setattr(self, f"norm{i}_weights", parameters[norm_name]["weight"])
                setattr(self, f"norm{i}_bias", parameters[norm_name].get("bias"))

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
        return_attn_weights=False,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                pos,
                query_pos,
                return_attn_weights,
            )
        else:
            return self.forward_post(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                pos,
                query_pos,
                return_attn_weights,
            )

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        pos=None,
        query_pos=None,
        return_attn_weights=False,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)

        # Self-attention using proper QKV projections
        tgt2 = self.self_attn(q, k, tgt, tgt_mask)
        # tgt2 = self.self_attention(q, k, tgt, query_pos, tgt_mask)
        tgt = ttnn.add(tgt, tgt2)
        tgt = ttnn.layer_norm(tgt, weight=self.norm1_weights, bias=getattr(self, "norm1_bias", None))

        # Cross-attention using proper QKV projections
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
        )
        tgt = ttnn.add(tgt, tgt2)
        tgt = ttnn.layer_norm(tgt, weight=self.norm2_weights, bias=getattr(self, "norm2_bias", None))

        # Feedforward
        tgt2 = ttnn.linear(tgt, self.ff_weights1, bias=getattr(self, "ff1_bias", None))
        tgt2 = ttnn.relu(tgt2)
        tgt2 = ttnn.linear(tgt2, self.ff_weights2, bias=getattr(self, "ff2_bias", None))
        tgt = ttnn.add(tgt, tgt2)
        tgt = ttnn.layer_norm(tgt, weight=self.norm3_weights, bias=getattr(self, "norm3_bias", None))

        if return_attn_weights:
            return tgt, None  # TTNN doesn't return attention weights
        return tgt, None

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        pos=None,
        query_pos=None,
        return_attn_weights=False,
    ):
        # Pre-norm self-attention
        tgt2 = ttnn.layer_norm(tgt, weight=self.norm1_weights, bias=getattr(self, "norm1_bias", None))
        q = k = self.with_pos_embed(tgt2, query_pos)
        # q=k=v=tgt2=tgt
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask)
        # return tgt2, None
        tgt = ttnn.add(tgt, tgt2)

        # Pre-norm cross-attention
        tgt2 = ttnn.layer_norm(tgt, weight=self.norm2_weights, bias=getattr(self, "norm2_bias", None))
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
        )
        tgt = ttnn.add(tgt, tgt2)

        # Pre-norm feedforward
        tgt2 = ttnn.layer_norm(tgt, weight=self.norm3_weights, bias=getattr(self, "norm3_bias", None))
        tgt2 = ttnn.linear(tgt2, self.ff_weights1, bias=getattr(self, "ff1_bias", None))
        tgt2 = ttnn.relu(tgt2)
        tgt2 = ttnn.linear(tgt2, self.ff_weights2, bias=getattr(self, "ff2_bias", None))
        tgt = ttnn.add(tgt, tgt2)

        if return_attn_weights:
            return tgt, None
        return tgt, None


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
        return_attn_weights=False,
    ):
        # Handle transpose_swap for memory tensor
        if transpose_swap:
            # memory: bs, c, h, w -> t, b, c
            memory = ttnn.reshape(memory, (memory.shape[0], memory.shape[1], -1))
            memory = ttnn.permute(memory, (2, 0, 1))
            if pos is not None:
                pos = ttnn.reshape(pos, (pos.shape[0], pos.shape[1], -1))
                pos = ttnn.permute(pos, (2, 0, 1))

        output = tgt
        intermediate = []
        attns = []

        # Pass through each decoder layer
        for layer in self.layers:
            output, attn = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                pos=pos,
                query_pos=query_pos,
                return_attn_weights=return_attn_weights,
            )

            if self.return_intermediate and self.norm_weights is not None:
                intermediate.append(ttnn.layer_norm(output, weight=self.norm_weights, bias=self.norm_bias))

            if return_attn_weights:
                attns.append(attn)

        # Apply final layer norm
        if self.norm_weights is not None:
            output = ttnn.layer_norm(output, weight=self.norm_weights, bias=self.norm_bias)
            if self.return_intermediate:
                # Replace last intermediate result with final normed output
                if intermediate:
                    intermediate[-1] = output

        # Stack results if needed
        if return_attn_weights and attns:
            # TTNN doesn't support torch.stack directly, using ttnn.concat for now
            attns = [ttnn.reshape(t, (1, *t.shape)) for t in attns]
            attns = ttnn.concat(attns, dim=0, memory_config=ttnn.L1_MEMORY_CONFIG)

        if self.return_intermediate:
            # TTNN doesn't support torch.stack directly, using ttnn.concat for now
            intermediate = [ttnn.reshape(t, (1, *t.shape)) for t in intermediate]
            intermediate = ttnn.concat(intermediate, dim=0, memory_config=ttnn.L1_MEMORY_CONFIG)
            return intermediate, attns if return_attn_weights else None

        return output, attns if return_attn_weights else None
