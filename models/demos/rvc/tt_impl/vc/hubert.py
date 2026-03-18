# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch

import ttnn
from models.demos.rvc.tt_impl.conv1d import Conv1d
from models.demos.rvc.tt_impl.groupnorm import GroupNorm1D
from models.demos.rvc.tt_impl.layernorm import LayerNorm
from models.demos.rvc.tt_impl.linear import Linear


def pad_to_multiple(x: ttnn.Tensor, multiple: int, value: float = 0.0) -> tuple[ttnn.Tensor, int]:
    tsz = x.shape[1]
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if m.is_integer():
        return x, 0
    return ttnn.pad(x, [(0, 0), (0, remainder), (0, 0)], value=value), remainder


class MultiheadSelfAttention:
    """Optimized TT HuBERT attention using TTNN transformer attention operators."""

    def __init__(self, device: ttnn.MeshDevice, embed_dim: int, num_heads: int, self_attention: bool = False) -> None:
        self.device = device
        self.embed_dim = embed_dim
        self.k_dim = embed_dim
        self.v_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.self_attention = self_attention

        self.k_proj = Linear(device=device, in_features=self.k_dim, out_features=embed_dim)
        self.v_proj = Linear(device=device, in_features=self.v_dim, out_features=embed_dim)
        self.q_proj = Linear(device=device, in_features=embed_dim, out_features=embed_dim)
        self.out_proj = Linear(device=device, in_features=embed_dim, out_features=embed_dim)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        self.k_proj.load_state_dict(state_dict=state_dict, key="k_proj", module_prefix=module_prefix)
        self.v_proj.load_state_dict(state_dict=state_dict, key="v_proj", module_prefix=module_prefix)
        self.q_proj.load_state_dict(state_dict=state_dict, key="q_proj", module_prefix=module_prefix)
        self.out_proj.load_state_dict(state_dict=state_dict, key="out_proj", module_prefix=module_prefix)

    def __call__(self, query: ttnn.Tensor) -> ttnn.Tensor:
        src_len = query.shape[0]

        query_proj = self.q_proj(query)
        key_proj = self.k_proj(query)
        value_proj = self.v_proj(query)

        qkv_proj = ttnn.concat([query_proj, key_proj, value_proj], dim=-1)
        query_heads, key_heads, value_heads = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv_proj,
            num_heads=self.num_heads,
            transpose_key=False,
        )
        ttnn.deallocate(qkv_proj)

        ttnn.deallocate(query_proj)
        ttnn.deallocate(key_proj)
        ttnn.deallocate(value_proj)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            is_causal=False,
        )

        ttnn.deallocate(query_heads)
        ttnn.deallocate(key_heads)
        ttnn.deallocate(value_heads)

        attn_out = ttnn.transformer.concatenate_heads(attn_out)
        attn_out = self.out_proj(attn_out)

        return attn_out

    def deallocate(self) -> None:
        self.k_proj.deallocate()
        self.v_proj.deallocate()
        self.q_proj.deallocate()
        self.out_proj.deallocate()


class TransformerSentenceEncoderLayer:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:
        self.device = device
        self.embedding_dim = embed_dim
        self.activation_fn = activation_fn
        self.self_attn = MultiheadSelfAttention(
            device=device, embed_dim=embed_dim, num_heads=attention_heads, self_attention=True
        )
        self.layer_norm_first = layer_norm_first
        self.fc1 = Linear(
            device=device,
            in_features=embed_dim,
            out_features=ffn_embed_dim,
            activation=activation_fn,
        )
        self.fc2 = Linear(device=device, in_features=ffn_embed_dim, out_features=embed_dim)
        self.self_attn_layer_norm = LayerNorm(device=device, normalized_shape=embed_dim, eps=1e-5)
        self.final_layer_norm = LayerNorm(device=device, normalized_shape=embed_dim, eps=1e-5)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        self.self_attn.load_state_dict(state_dict, module_prefix=f"{module_prefix}self_attn.")
        self.fc1.load_state_dict(state_dict=state_dict, key="fc1", module_prefix=module_prefix)
        self.fc2.load_state_dict(state_dict=state_dict, key="fc2", module_prefix=module_prefix)
        self.self_attn_layer_norm.load_state_dict(
            state_dict=state_dict, key="self_attn_layer_norm", module_prefix=module_prefix
        )
        self.final_layer_norm.load_state_dict(
            state_dict=state_dict, key="final_layer_norm", module_prefix=module_prefix
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            attn_out = self.self_attn(query=x)
            x = ttnn.add(residual, attn_out, output_tensor=x)

            residual = x
            x = self.final_layer_norm(x)
            x = self.fc1(x)
            x = self.fc2(x)
            x = ttnn.add(residual, x, output_tensor=x)
        else:
            x = self.self_attn(query=x)

            x = ttnn.add(residual, x, output_tensor=x)

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.fc1(x)

            x = self.fc2(x)
            x = ttnn.add(residual, x, output_tensor=x)
            x = self.final_layer_norm(x)

        return x

    def deallocate(self) -> None:
        self.self_attn.deallocate()
        self.fc1.deallocate()
        self.fc2.deallocate()
        self.self_attn_layer_norm.deallocate()
        self.final_layer_norm.deallocate()


class ConvFeatureExtractionModel:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        conv_layers: list[tuple[int, int, int]],
        mode: str = "default",
        conv_bias: bool = False,
    ) -> None:
        if mode not in {"default", "layer_norm"}:
            raise ValueError("mode must be 'default' or 'layer_norm'")
        self.device = device
        self.mode = mode
        self.conv_layers_cfg = conv_layers
        self.conv_layers: list[Conv1d] = []

        in_d = 1
        for i, conv_config in enumerate(conv_layers):
            if len(conv_config) != 3:
                raise ValueError(f"invalid conv definition: {conv_config}")
            dim, k, stride = conv_config
            self.conv_layers.append(
                Conv1d(
                    device=device,
                    in_channels=in_d,
                    out_channels=dim,
                    kernel_size=k,
                    stride=stride,
                    padding=0,
                    dtype=ttnn.bfloat16,
                )
            )
            in_d = dim

        self.layer_norms: list[LayerNorm] = []
        self.group_norms: list[GroupNorm1D] = []
        if self.mode == "layer_norm":
            for dim, _, _ in conv_layers:
                self.layer_norms.append(LayerNorm(device=device, normalized_shape=dim, eps=1e-5))
        if self.mode == "default" and conv_layers:
            self.group_norms.append(
                GroupNorm1D(device=device, num_channels=conv_layers[0][0], num_groups=conv_layers[0][0])
            )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        for i, conv in enumerate(self.conv_layers):
            conv.load_state_dict(state_dict=state_dict, key=f"conv_layers.{i}.0", module_prefix=module_prefix)
            if self.mode == "layer_norm":
                self.layer_norms[i].load_state_dict(
                    state_dict=state_dict, key=f"conv_layers.{i}.1.1", module_prefix=module_prefix
                )
            elif self.mode == "default" and i == 0:
                group_norm = self.group_norms[i]
                if group_norm is None:
                    raise ValueError("GroupNorm module is not initialized.")
                group_norm.load_state_dict(state_dict=state_dict, key=f"conv_layers.{i}.1", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.mode == "layer_norm":
                x = self.layer_norms[i](x)

            elif self.mode == "default" and i == 0:
                x = self.group_norms[i].gp_slice(x)
            x = ttnn.gelu(x, output_tensor=x)

        return x

    def deallocate(self) -> None:
        for conv in self.conv_layers:
            conv.deallocate()
        for layer_norm in self.layer_norms:
            layer_norm.deallocate()
        for group_norm in self.group_norms:
            if group_norm is not None:
                group_norm.deallocate()


class PositionalConvEmbedding:
    def __init__(self, device: ttnn.MeshDevice, embed_dim: int, kernel_size: int, groups: int) -> None:
        self.device = device
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.conv = Conv1d(
            device=device,
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            groups=groups,
            dtype=ttnn.bfloat16,
            activation="gelu",
        )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        self.conv.load_state_dict(state_dict=state_dict, key="0", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        input_length = x.shape[1]
        output_length = input_length + 2 * (self.kernel_size // 2) - self.kernel_size + 1
        out = self.conv(x)
        return out

    def deallocate(self) -> None:
        self.conv.deallocate()


class TransformerEncoder:
    def __init__(self, device: ttnn.MeshDevice, args: dict) -> None:
        self.device = device
        self.embedding_dim = args["encoder_embed_dim"]
        self.required_seq_len_multiple = args.get("required_seq_len_multiple", 2)
        self.pos_conv = PositionalConvEmbedding(
            device=device,
            embed_dim=self.embedding_dim,
            kernel_size=args["conv_pos"],
            groups=args["conv_pos_groups"],
        )
        self.layers = [self.build_encoder_layer(args) for _ in range(args["encoder_layers"])]
        self.layer_norm_first = args["layer_norm_first"]
        self.layer_norm = LayerNorm(device=device, normalized_shape=self.embedding_dim, eps=1e-5)

    def build_encoder_layer(self, args: dict):
        return TransformerSentenceEncoderLayer(
            device=self.device,
            embed_dim=self.embedding_dim,
            ffn_embed_dim=args["encoder_ffn_embed_dim"],
            attention_heads=args["encoder_attention_heads"],
            activation_fn=args["activation_fn"],
            layer_norm_first=args["layer_norm_first"],
        )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        self.pos_conv.load_state_dict(state_dict=state_dict, module_prefix=f"{module_prefix}pos_conv.")
        self.layer_norm.load_state_dict(state_dict=state_dict, key="layer_norm", module_prefix=module_prefix)
        for i, layer in enumerate(self.layers):
            layer.load_state_dict(state_dict=state_dict, module_prefix=f"{module_prefix}layers.{i}.")

    def __call__(self, x: ttnn.Tensor, tgt_layer: int) -> ttnn.Tensor:
        x = ttnn.add(x, self.pos_conv(x))

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x, pad_length = pad_to_multiple(x, self.required_seq_len_multiple, value=0.0)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == tgt_layer:
                break

        if pad_length > 0:
            x = ttnn.slice(x, (0, 0, 0), (x.shape[0], x.shape[1] - pad_length, x.shape[2]))
        return x

    def deallocate(self) -> None:
        self.pos_conv.deallocate()
        self.layer_norm.deallocate()
        for layer in self.layers:
            layer.deallocate()


class HubertModel:
    def __init__(self, device: ttnn.MeshDevice, cfg: dict, task_cfg) -> None:
        self.device = device
        self.cfg = cfg
        self._is_generation_fast = False

        feature_enc_layers = eval(cfg["conv_feature_layers"])  # noqa: S307
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            device=device,
            conv_layers=feature_enc_layers,
            mode=cfg["extractor_mode"],
            conv_bias=cfg["conv_bias"],
        )
        feature_ds_rate = math.prod([stride for _, _, stride in feature_enc_layers])
        self.feat2tar_ratio = cfg["label_rate"] * feature_ds_rate / task_cfg.sample_rate

        self.post_extract_proj = None
        if self.embed != cfg["encoder_embed_dim"]:
            self.post_extract_proj = Linear(
                device=device,
                in_features=self.embed,
                out_features=cfg["encoder_embed_dim"],
            )

        self.feature_grad_mult = cfg["feature_grad_mult"]
        self.logit_temp = cfg["logit_temp"]

        final_dim = cfg["final_dim"] if cfg["final_dim"] > 0 else cfg["encoder_embed_dim"]
        self.encoder = TransformerEncoder(device=device, args=cfg)
        self.layer_norm = LayerNorm(device=device, normalized_shape=self.embed, eps=1e-5)
        self.untie_final_proj = cfg["untie_final_proj"]
        self.final_proj_linear = Linear(
            device=device,
            in_features=cfg["encoder_embed_dim"],
            out_features=final_dim,
        )

    @classmethod
    def build_model(cls, cfg, task, device: ttnn.MeshDevice):
        return cls(device=device, cfg=cfg, task_cfg=task.cfg)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.feature_extractor.load_state_dict(state_dict=state_dict, module_prefix="feature_extractor.")
        self.layer_norm.load_state_dict(state_dict=state_dict, key="layer_norm")
        if self.post_extract_proj is not None:
            self.post_extract_proj.load_state_dict(state_dict=state_dict, key="post_extract_proj")
        self.encoder.load_state_dict(state_dict=state_dict, module_prefix="encoder.")
        self.final_proj_linear.load_state_dict(state_dict=state_dict, key="final_proj")

    def __call__(self, source: ttnn.Tensor, output_layer: int) -> ttnn.Tensor:
        x = self.feature_extractor(source)

        x = self.layer_norm(x)

        if self.post_extract_proj is not None:
            x = self.post_extract_proj(x)
        out = self.encoder(x, tgt_layer=output_layer - 1)
        return out

    def final_proj(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.final_proj_linear(x)

    def eval(self):
        return self

    def float(self):
        return self

    def to(self, *_args, **_kwargs):
        return self

    def deallocate(self) -> None:
        self.feature_extractor.deallocate()
        self.layer_norm.deallocate()
        if self.post_extract_proj is not None:
            self.post_extract_proj.deallocate()
        self.encoder.deallocate()
        self.final_proj_linear.deallocate()
