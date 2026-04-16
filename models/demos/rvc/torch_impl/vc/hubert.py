# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.demos.rvc.utils.config import HubertPretrainingConfig, HubertPretrainingTask


def get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])


def relu_squared(x: torch.Tensor):
    return F.relu(x).pow(2)


def gelu_approximate(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x.float()).type_as(x)


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    if activation == "relu":
        return F.relu
    elif activation == "relu_squared":
        return relu_squared
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_approximate":
        return gelu_approximate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return nn.SiLU
    else:
        raise RuntimeError(f"--activation-fn {activation} not supported")


def pad_to_multiple(x, multiple, dim=-1, value=0):
    # Inspired from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py#L41
    tsz = x.size(dim)
    m = tsz / multiple
    remainder = math.ceil(m) * multiple - tsz
    if m.is_integer():
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2

    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder


class TransposeLast(nn.Module):
    def __init__(self):
        super().__init__()
        self.tranpose_dim = -2

    def forward(self, x):
        return x.transpose(-2, -1)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:
        super().__init__()
        # Initialize parameters
        self.embedding_dim = embed_dim

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadSelfAttention(
            self.embedding_dim,
            attention_heads,
            self_attention=True,
        )

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x = self.self_attn(
                query=x,
            )
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.fc2(x)

            x = residual + x
        else:
            x = self.self_attn(
                query=x,
            )

            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.fc2(x)

            x = residual + x
            x = self.final_layer_norm(x)

        return x


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: list[tuple[int, int, int]],
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert not (is_layer_norm and is_group_norm), "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Sequential(
                        TransposeLast(),
                        nn.LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x


class MultiheadSelfAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        self_attention=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=True)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        query: Tensor,
    ) -> Tensor:
        """Input shape: Time x Batch x Channel"""
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if (
            # The Multihead attention implemented in pytorch forces strong dimension check
            # for input embedding dimention and K,Q,V projection dimension.
            # Since pruning will break the dimension check and it is not easy to modify the pytorch API,
            # it is preferred to bypass the pytorch MHA when we need to skip embed_dim_check
            True
        ):
            return F.multi_head_attention_forward(
                query,
                query,
                query,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                None,
                None,
                False,
                0.0,
                self.out_proj.weight,
                self.out_proj.bias,
                False,
                None,
                False,
                None,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )[0]

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        kv_bsz = bsz  # need default value for scripting
        if k is not None:
            kv_bsz = k.size(1)
            k = k.contiguous().view(-1, kv_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, kv_bsz * self.num_heads, self.head_dim).transpose(0, 1)

        assert k is not None
        assert k.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = attn_weights

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)
        return attn


def make_conv_pos(e, k, g, is_batch_norm=False):
    pos_conv = nn.Conv1d(
        e,
        e,
        kernel_size=k,
        padding="same",
        groups=g,
    )
    std = math.sqrt(4 / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    if not is_batch_norm:
        pos_conv = nn.Sequential(pos_conv, nn.GELU())
    else:
        batch_norm = nn.BatchNorm1d(e)
        pos_conv = nn.Sequential(batch_norm, pos_conv, nn.GELU())

    return pos_conv


class TransformerEncoder(nn.Module):
    def build_encoder_layer(self, args: dict[str, Any]):
        return TransformerSentenceEncoderLayer(
            embed_dim=self.embedding_dim,
            ffn_embed_dim=args["encoder_ffn_embed_dim"],
            attention_heads=args["encoder_attention_heads"],
            activation_fn=args["activation_fn"],
            layer_norm_first=args["layer_norm_first"],
        )

    def __init__(
        self,
        args: dict[str, Any],
    ):
        super().__init__()

        self.embedding_dim = args["encoder_embed_dim"]
        self.required_seq_len_multiple = args.get("required_seq_len_multiple", 2)

        pos_conv_depth = getattr(args, "pos_conv_depth", 1)
        if pos_conv_depth > 1:
            num_layers = args["pos_conv_depth"]
            k = max(3, args["conv_pos"] // num_layers)

            def make_conv_block(e, k, g, num_layers):
                return nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv1d(
                                e,
                                e,
                                kernel_size=k,
                                padding="same",
                                groups=g,
                            ),
                            TransposeLast(),
                            nn.LayerNorm(e, elementwise_affine=False),
                            TransposeLast(),
                            nn.GELU(),
                        )
                        for _ in range(num_layers)
                    ]
                )

            self.pos_conv = make_conv_block(self.embedding_dim, k, args["conv_pos_groups"], num_layers)
        else:
            self.pos_conv = make_conv_pos(
                self.embedding_dim,
                args["conv_pos"],
                args["conv_pos_groups"],
                is_batch_norm=args.get("conv_pos_batch_norm", False),
            )

        encoder_layers = args["encoder_layers"]

        self.layers = nn.ModuleList([self.build_encoder_layer(args) for _ in range(encoder_layers)])
        self.layer_norm_first = args["layer_norm_first"]
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        x,
        tgt_layer,
    ):
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(x, self.required_seq_len_multiple, dim=-2, value=0)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == tgt_layer:
                break

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # undo paddding
        if pad_length > 0:
            x = x[:, :-pad_length]
        return x


class HubertModel(nn.Module):
    def __init__(
        self,
        cfg,
        task_cfg: HubertPretrainingConfig,
    ):
        super().__init__()
        self._is_generation_fast = False
        feature_enc_layers = eval(cfg["conv_feature_layers"])  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            mode=cfg["extractor_mode"],
            conv_bias=cfg["conv_bias"],
        )
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = cfg["label_rate"] * feature_ds_rate / task_cfg.sample_rate

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg["encoder_embed_dim"]) if self.embed != cfg["encoder_embed_dim"] else None
        )

        self.feature_grad_mult = cfg["feature_grad_mult"]
        self.logit_temp = cfg["logit_temp"]

        final_dim = cfg["final_dim"] if cfg["final_dim"] > 0 else cfg["encoder_embed_dim"]

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = nn.LayerNorm(self.embed)

        self.untie_final_proj = cfg["untie_final_proj"]
        self.final_proj = nn.Linear(cfg["encoder_embed_dim"], final_dim)

    @classmethod
    def build_model(cls, cfg, task: HubertPretrainingTask):
        """Build a new model instance."""
        return HubertModel(cfg, task.cfg)

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def forward(
        self,
        source: torch.Tensor,
        output_layer: int,
    ) -> dict[str, torch.Tensor]:
        """output layer is 1-based"""
        features = self.feature_extractor(source)

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x = features

        x = self.encoder(
            x,
            tgt_layer=output_layer - 1,
        )

        return x
