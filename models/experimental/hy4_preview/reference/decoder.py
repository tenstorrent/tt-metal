# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch ground-truth reference for one Hy4-preview decoder layer and a stack of them.

Pure torch, runs on CPU. Mirrors HF transformers ``models/hy_v4/modeling_hy_v4.py``
HYV4DecoderLayer.forward and the layer loop of HYV4Model.forward, composed from the sublayer
references next to this file (ihc.py, attention.py, moe.py):

    streams [S, n, hidden]                                      n = 4 residual streams (iHC)
      h, post = hc_attn.hc_pre(streams)                         collapse n -> 1
      a       = self_attn(RMSNorm(h; input_layernorm))          pre-norm, gated MLA with a sink
      streams = hc_attn.hc_post(a, streams, post)               expand 1 -> n, add each stream's residual
      h, post = hc_mlp.hc_pre(streams)
      m       = mlp(RMSNorm(h; post_attention_layernorm))       dense FFN on layer 0, MoE on the rest
      streams = hc_mlp.hc_post(m, streams, post)

Both pre-norms are HYV4RMSNorm with eps = ``rms_norm_eps`` (1e-5), unlike the two norms inside the
attention (1e-6).

Attention here is dense and causal: the DSA indexer is not modelled. That is exactly what HF
computes while every position a query can see is among its top-k picks. HYV4Indexer.forward takes
``topk = min(index_topk, T)`` over the T keys, and HYV4Attention.forward masks the keys that are
not picked; with T <= index_topk (2048) every key is picked and the mask is empty, for a "full"
layer and for the "shared" layers that reuse its picks alike. Past that length the picks matter and
this reference is no longer HF: Hy4DecoderStack refuses such a prompt.

Precision. The sublayers run in the dtype of the parameters (fp32 here). The streams keep the dtype
the caller holds them in: iHC computes both mixes in fp32 and casts the result back to the streams'
dtype, so fp32 streams stay fp32 end to end, and BF16 streams are rounded to BF16 after every
sublayer, which is what HF does with the BF16 checkpoint.

Checkpoint names under ``model.layers.{L}.`` (LAYER_PARAMETERS maps each to this module):
    hc_attn_layer.hc_pre.{hc_fn, hc_base, hc_scale}    hc_mlp_layer.hc_pre.{hc_fn, hc_base, hc_scale}
    input_layernorm.weight                             post_attention_layernorm.weight
    self_attn.*  (reference/attention.py)              mlp.*  (reference/moe.py)
The indexer's weights (``self_attn.indexer.*``) are not used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from models.experimental.hy4_preview.reference.attention import (
    AttentionConfig,
    Hy4Attention,
    causal_mask,
    rms_norm,
    rope_cos_sin,
)
from models.experimental.hy4_preview.reference.ihc import IHCConfig, IHCWrap, ihc_expand
from models.experimental.hy4_preview.reference.moe import Hy4FFN, Hy4MoE, MoEConfig

# What one layer records, in order. Shapes are for a prompt of S tokens; the HF harness
# (hf_harness.py) records the same keys from the HF modules.
STAGES = (
    "streams_in",  # [S, n, hidden]  the layer's input
    "attn_hc",  # [S, hidden]     hc_pre at the attention site: the collapsed stream
    "attn_post",  # [S, n]          its post weights, fp32
    "attn_in",  # [S, hidden]     after input_layernorm: the attention sublayer's input
    "attn_out",  # [S, hidden]     the attention sublayer's output
    "streams_mid",  # [S, n, hidden]  the streams after the attention sublayer
    "mlp_hc",  # [S, hidden]     hc_pre at the FFN site
    "mlp_post",  # [S, n]
    "mlp_in",  # [S, hidden]     after post_attention_layernorm: the FFN / MoE input
    "mlp_out",  # [S, hidden]     the FFN / MoE output
    "streams_out",  # [S, n, hidden]  the layer's output
)


@dataclass
class DecoderConfig:
    """Decoder hyperparameters. Defaults are the tencent/Hy4-preview config.json values."""

    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    ihc: IHCConfig = field(default_factory=IHCConfig)
    norm_eps: float = 1e-5  # input_layernorm / post_attention_layernorm ("rms_norm_eps")
    index_topk: int = 2048  # keys a DSA query keeps; dense attention is exact up to this length
    dense_layers: tuple = (0,)  # layers whose FFN is dense ("mlp_layer_types")

    def __post_init__(self):
        sizes = (self.attention.hidden_size, self.moe.hidden_size, self.ihc.dim)
        assert len(set(sizes)) == 1, f"attention / moe / ihc hidden sizes differ: {sizes}"

    @property
    def hidden_size(self) -> int:
        return self.ihc.dim

    def is_dense(self, layer: int) -> bool:
        return layer in self.dense_layers


def _layer_parameters(dense: bool) -> dict:
    names = {}
    for site, attr in (("hc_attn_layer", "hc_attn"), ("hc_mlp_layer", "hc_mlp")):
        for p in ("fn", "base", "scale"):
            names[f"{site}.hc_pre.hc_{p}"] = f"{attr}.{p}"
    names["input_layernorm.weight"] = "input_layernorm"
    names["post_attention_layernorm.weight"] = "post_attention_layernorm"
    for p in Hy4Attention.PARAMETER_NAMES:
        suffix = "" if p == "learnable_sink_param" else ".weight"
        names[f"self_attn.{p}{suffix}"] = f"self_attn.{p}"
    if dense:
        for p in Hy4FFN.PARAMETER_NAMES:
            names[f"mlp.{p}"] = "mlp." + p.removesuffix(".weight")
    else:
        for p, attr in Hy4MoE.PARAMETER_NAMES.items():
            names[f"mlp.{p}"] = f"mlp.{attr}"
    return names


# Checkpoint name (under ``model.layers.{L}.``) -> parameter path in Hy4DecoderLayer, for a dense
# layer (True) and a MoE layer (False).
LAYER_PARAMETERS = {True: _layer_parameters(True), False: _layer_parameters(False)}


def checkpoint_names(layer: int, dense: bool) -> list:
    """The full checkpoint names of one layer's parameters, in LAYER_PARAMETERS order."""
    return [f"model.layers.{layer}.{name}" for name in LAYER_PARAMETERS[dense]]


def dense_prefill_inputs(cfg: DecoderConfig, seq_len: int, positions=None):
    """(cos, sin, mask) for a prompt of ``seq_len`` tokens at ``positions`` (default 0 .. S-1), with the
    causal mask of dense attention. Refuses a prompt the DSA indexer would thin out."""
    assert seq_len <= cfg.index_topk, (
        f"a prompt of {seq_len} tokens is longer than index_topk = {cfg.index_topk}: "
        "the DSA indexer drops keys there, and dense attention is no longer what HF computes"
    )
    positions = torch.arange(seq_len) if positions is None else positions
    cos, sin = rope_cos_sin(positions, cfg.attention)
    return cos, sin, causal_mask(seq_len)


class Hy4DecoderLayer(nn.Module):
    """One decoder layer on a whole prompt. streams: [S, n, hidden] -> [S, n, hidden]."""

    def __init__(self, cfg: DecoderConfig, dense: bool, init: bool = True):
        """``dense`` picks the FFN (layer 0) over the MoE. ``init=False`` leaves the routed experts
        uninitialised, for a caller that loads them."""
        super().__init__()
        self.cfg = cfg
        self.dense = dense
        self.hc_attn = IHCWrap(cfg.ihc)
        self.hc_mlp = IHCWrap(cfg.ihc)
        self.input_layernorm = nn.Parameter(torch.ones(cfg.hidden_size))
        self.post_attention_layernorm = nn.Parameter(torch.ones(cfg.hidden_size))
        self.self_attn = Hy4Attention(cfg.attention)
        if dense:
            self.mlp = Hy4FFN(cfg.hidden_size, cfg.moe.dense_intermediate_size)
        else:
            self.mlp = Hy4MoE(cfg.moe, init=init)

    @property
    def parameter_names(self) -> dict:
        return LAYER_PARAMETERS[self.dense]

    def weights(self) -> dict:
        """Every parameter by checkpoint name (under ``model.layers.{L}.``), as the device layer takes them."""
        return {name: self.get_parameter(path).detach() for name, path in self.parameter_names.items()}

    def load_one(self, name: str, value: torch.Tensor) -> None:
        """Take one parameter by checkpoint name. A tensor that already has the parameter's dtype is
        adopted, not copied: one layer of routed experts is 39 GB in fp32."""
        p = self.get_parameter(self.parameter_names[name])
        assert value.shape == p.shape, f"{name}: {tuple(value.shape)} vs {tuple(p.shape)}"
        p.data = value.detach().to(p.dtype)

    def load(self, tensors: dict) -> "Hy4DecoderLayer":
        """Take every parameter from ``tensors`` (checkpoint name -> tensor); other entries are ignored."""
        for name in self.parameter_names:
            self.load_one(name, tensors[name])
        return self

    def attention_sublayer(self, h, cos, sin, mask, record=None):
        """h: the collapsed stream [S, hidden] -> the attention output [S, hidden]."""
        dtype = self.input_layernorm.dtype
        x = rms_norm(h.to(dtype), self.input_layernorm, self.cfg.norm_eps)
        out = self.self_attn(x, cos.to(dtype), sin.to(dtype), mask.to(dtype))
        if record is not None:
            record["attn_in"], record["attn_out"] = x, out
        return out

    def mlp_sublayer(self, h, record=None, expert_weights=None):
        """h: the collapsed stream [S, hidden] -> the FFN / MoE output [S, hidden].

        ``expert_weights`` [S, num_experts], MoE layers only, replaces the layer's own routing: a
        token's weight for every expert, zero where not picked. It lets a comparison follow the picks
        of another implementation, so that a near tie between the 8th and 9th score that went the
        other way there is not read as an arithmetic error.
        """
        x = rms_norm(h.to(self.post_attention_layernorm.dtype), self.post_attention_layernorm, self.cfg.norm_eps)
        if expert_weights is None:
            out = self.mlp(x)
        else:
            assert not self.dense, "a dense layer has no routing to replace"
            out = self.mlp.routed(x, expert_weights.to(x.dtype)) + self.mlp.shared(x)
        if record is not None:
            record["mlp_in"], record["mlp_out"] = x, out
        return out

    def forward(self, streams, cos, sin, mask, record=None, expert_weights=None):
        """streams: [S, n, hidden]; cos / sin: [S, rope]; mask: additive [S, S]. Returns [S, n, hidden]
        in the streams' dtype. ``record``, a dict, receives every STAGES tensor; ``expert_weights``
        is mlp_sublayer's."""
        rec = {} if record is None else record
        x = streams.unsqueeze(0)  # iHC keeps a batch dim
        rec["streams_in"] = streams

        h, post = self.hc_attn.hc_pre(x)
        rec["attn_hc"], rec["attn_post"] = h[0], post[0]
        out = self.attention_sublayer(h[0], cos, sin, mask, rec)
        x = self.hc_attn.hc_post(out.unsqueeze(0), x, post)
        rec["streams_mid"] = x[0]

        h, post = self.hc_mlp.hc_pre(x)
        rec["mlp_hc"], rec["mlp_post"] = h[0], post[0]
        out = self.mlp_sublayer(h[0], rec, expert_weights)
        x = self.hc_mlp.hc_post(out.unsqueeze(0), x, post)
        rec["streams_out"] = x[0]
        return x[0]


class Hy4DecoderStack(nn.Module):
    """Layers ``first`` .. ``first + num_layers - 1`` of the decoder, in prefill with dense attention.

    Starts from the embedding output and stops at the last layer's streams: the embedding, the
    stream-collapsing head, the final norm and the LM head are not here.
    """

    def __init__(self, cfg: DecoderConfig, num_layers: int, first: int = 0, init: bool = True):
        super().__init__()
        self.cfg = cfg
        self.first = first
        self.layers = nn.ModuleList(Hy4DecoderLayer(cfg, cfg.is_dense(first + i), init=init) for i in range(num_layers))

    def run(self, streams, positions=None, record=None):
        """streams: [S, n, hidden] entering layer ``first`` -> the streams leaving the last layer.
        ``record``, a list, receives one STAGES dict per layer."""
        cos, sin, mask = dense_prefill_inputs(self.cfg, streams.shape[0], positions)
        for layer in self.layers:
            rec = {}
            streams = layer(streams, cos, sin, mask, rec)
            if record is not None:
                record.append(rec)
        return streams

    def forward(self, embeds, positions=None, record=None):
        """embeds: the embedding output [S, hidden] -> the streams [S, n, hidden] after the last layer."""
        assert self.first == 0, "the embedding feeds layer 0; use run() to enter a later layer"
        streams = ihc_expand(embeds.unsqueeze(0), self.cfg.ihc.n)[0]
        return self.run(streams, positions, record)
