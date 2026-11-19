# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One Hy4-preview decoder layer on the SC1 layout, for a whole prompt (tenstorrent/tt-metal#65893).

Mirrors the pure-torch ground truth in ../reference/decoder.py, which mirrors HF's
HYV4DecoderLayer.forward, and composes the device sublayers that exist:

    streams                                              [1, 1, T, n * hidden/tp]  fp32
      h, post = hc_attn.hc_pre(streams)                  tt/ihc.py       1 all-reduce (hidden)
      a       = attention(input_layernorm(h))            tt/norm.py      1 all-reduce (hidden)
                                                         tt/attention.py 4 collectives + its core
      streams = hc_attn.hc_post(a, streams, post)        chip-local
      h, post = hc_mlp.hc_pre(streams)                   1 all-reduce (hidden)
      m       = mlp(post_attention_layernorm(h))         tt/norm.py      1 all-reduce (hidden)
                                                         tt/moe.py       2 (dense FFN) or 5 (MoE)
      streams = hc_mlp.hc_post(m, streams, post)         chip-local

Layout (tt/common.py MeshPlan; the 8x4 mesh: hidden axis = the 4-chip axis). The n = 4 residual streams are
packed on the last dim, each chip holding its own hidden slice of every stream: per chip
[1, 1, T, n * hidden/tp] with stream i at columns [i * hidden/tp, (i+1) * hidden/tp), the same on
every chip that shares a hidden coordinate. The single stream a sublayer sees is [1, 1, T, hidden/tp]. T tokens run down
the tile rows. streams_to_device / streams_to_torch are that contract in code.

Precision. The streams, both iHC mixes and both pre-norms are fp32, which is what HF computes
them in. A sublayer runs in BF16: the norm hands it a BF16 tensor and its output is cast back to
fp32 before hc_post. HF with the BF16 checkpoint also rounds the streams to BF16 after every
sublayer; here they stay fp32, so they carry less rounding than the HF golden, not more.

The attention seam. The layer does not know how attention is computed. It calls

    attention(x, **attention_inputs) -> [1, 1, T, hidden/tp]

with x the normed BF16 hidden state and ``attention_inputs`` whatever keyword arguments the caller
gave ``forward``, passed through untouched. The default is TtHy4Attention's dense core, whose
inputs are ``cos``, ``sin`` and an additive ``mask`` (dense_prefill_inputs builds them, causal).
Sparse prefill (#65887) and cached decode (#65888, #65889) plug in as another callable built from
the same checkpoint tensors (attention_weights) and take their own inputs the same way: positions,
the cache, the picks a "shared" indexer layer inherits from the "full" layer below it.

Dense attention is what HF computes only while the prompt is no longer than index_topk (2048)
tokens; see reference/decoder.py. dense_prefill_inputs refuses a longer prompt.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.hy4_preview.reference import decoder as reference
from models.experimental.hy4_preview.reference.attention import Hy4Attention
from models.experimental.hy4_preview.tt import frames
from models.experimental.hy4_preview.tt.attention import TtHy4Attention
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.ihc import TtIHCWrap
from models.experimental.hy4_preview.tt.moe import TtHy4FFN, TtHy4MoE
from models.experimental.hy4_preview.tt.norm import TtRMSNorm


# ---------------------------------------------------------------------------------------------
# Host <-> device layout of the streams and of a single hidden state
# ---------------------------------------------------------------------------------------------
def streams_to_device(plan: MeshPlan, streams: torch.Tensor, dtype=ttnn.float32):
    """Host streams [T, n, hidden] -> device, per chip [1, 1, T, n * hidden/tp].

    The hidden split cuts the last dim contiguously, so the columns are laid out chip-major, then
    stream: each chip receives its own hidden slice of every stream.
    """
    T, n, C = streams.shape
    tp = plan.hidden_tp
    packed = streams.float().reshape(T, n, tp, C // tp).permute(0, 2, 1, 3).reshape(1, 1, T, n * C)
    return ttnn.from_torch(
        packed.contiguous(),
        layout=ttnn.TILE_LAYOUT,
        device=plan.mesh_device,
        dtype=dtype,
        mesh_mapper=plan.shard_hidden(3),
    )


def streams_to_torch(plan: MeshPlan, x, n: int) -> torch.Tensor:
    """The inverse of streams_to_device: [T, n, hidden], read from the first chip of each hidden slice."""
    d = plan.to_torch_hidden_concat(x, dim=-1).float()
    T = d.shape[-2]
    return d.reshape(T, plan.hidden_tp, n, -1).permute(0, 2, 1, 3).reshape(T, n, -1)


def hidden_to_device(plan: MeshPlan, h: torch.Tensor, dtype=ttnn.float32):
    """Host [T, hidden] -> device, per chip [1, 1, T, hidden/tp]."""
    return ttnn.from_torch(
        h.float().reshape(1, 1, *h.shape).contiguous(),
        layout=ttnn.TILE_LAYOUT,
        device=plan.mesh_device,
        dtype=dtype,
        mesh_mapper=plan.shard_hidden(3),
    )


def hidden_to_torch(plan: MeshPlan, x) -> torch.Tensor:
    """The inverse of hidden_to_device: [T, hidden]."""
    return plan.to_torch_hidden_concat(x, dim=-1).float()[0, 0]


def dense_prefill_inputs(plan: MeshPlan, cfg: reference.DecoderConfig, seq_len: int, positions=None) -> dict:
    """The attention inputs of a prompt of ``seq_len`` tokens for the dense core, on every chip:
    cos / sin [1, 1, T, rope] as the interleaved tables the device attention takes (tt/frames.py) and the
    causal additive mask [1, 1, T, T], BF16."""
    cos, sin, mask = reference.dense_prefill_inputs(cfg, seq_len, positions)
    cos, sin = frames.rope_tables(cos, sin)

    def up(t):
        return ttnn.from_torch(
            t.float().reshape(1, 1, *t.shape).contiguous(),
            layout=ttnn.TILE_LAYOUT,
            device=plan.mesh_device,
            dtype=ttnn.bfloat16,
        )

    return {"cos": up(cos), "sin": up(sin), "mask": up(mask)}


def attention_weights(weights: dict) -> dict:
    """A layer's checkpoint-named tensors -> the nine tensors TtHy4Attention takes."""
    out = {}
    for name in Hy4Attention.PARAMETER_NAMES:
        suffix = "" if name == "learnable_sink_param" else ".weight"
        out[name] = weights[f"self_attn.{name}{suffix}"]
    return out


def _under(weights: dict, prefix: str) -> dict:
    return {name[len(prefix) :]: value for name, value in weights.items() if name.startswith(prefix)}


# ---------------------------------------------------------------------------------------------
class TtHy4DecoderLayer(LightweightModule):
    """One decoder layer. ``weights`` holds the layer's tensors by checkpoint name under
    ``model.layers.{L}.``, as reference.decoder.Hy4DecoderLayer.weights() returns them.

    ``dense`` picks the dense FFN (layer 0) over the MoE. ``weight_dtype`` is the dtype of the
    attention projections and of the FFN / shared-expert weights; ``expert_dtype`` is the routed
    experts' (None: TtHy4MoE's policy, BFP4 beside BFP8 weights); the iHC parameters and the norm
    weights are fp32. ``attention`` replaces the default dense TtHy4Attention (see the module docstring);
    ``self_attn.*`` is then not read from ``weights``.
    """

    def __init__(
        self,
        plan: MeshPlan,
        cfg: reference.DecoderConfig,
        weights: dict,
        dense: bool,
        weight_dtype=ttnn.bfloat8_b,
        attention=None,
        expert_dtype=None,
    ):
        self.plan = plan
        self.cfg = cfg
        self.dense = dense
        self.n = cfg.ihc.n
        self.stream_dtype = ttnn.float32
        self.sublayer_dtype = ttnn.bfloat16

        def ihc(site):
            p = _under(weights, f"{site}.hc_pre.hc_")
            return TtIHCWrap(plan, cfg.ihc, p["fn"], p["base"], p["scale"], dtype=self.stream_dtype)

        def norm(name):
            return TtRMSNorm(plan, weights[f"{name}.weight"], cfg.norm_eps, output_dtype=self.sublayer_dtype)

        self.hc_attn = ihc("hc_attn_layer")
        self.hc_mlp = ihc("hc_mlp_layer")
        self.input_layernorm = norm("input_layernorm")
        self.post_attention_layernorm = norm("post_attention_layernorm")
        if attention is None:
            attention = TtHy4Attention(
                plan, cfg.attention, attention_weights(weights), weight_dtype=weight_dtype, dtype=self.sublayer_dtype
            )
        self.attention = attention
        mlp = _under(weights, "mlp.")
        if dense:
            self.mlp = TtHy4FFN(plan, cfg.hidden_size, cfg.moe.dense_intermediate_size, mlp, weight_dtype=weight_dtype)
        else:
            self.mlp = TtHy4MoE(plan, cfg.moe, mlp, weight_dtype=weight_dtype, expert_dtype=expert_dtype)

    def attention_sublayer(self, h, record=None, **attention_inputs):
        """h: the collapsed stream, fp32 [1, 1, T, hidden/tp] -> the attention output, fp32, same layout."""
        x = self.input_layernorm(h)
        out = ttnn.typecast(self.attention(x, **attention_inputs), self.stream_dtype)
        if record is not None:
            record["attn_in"], record["attn_out"] = x, out
        return out

    def mlp_sublayer(self, h, record=None):
        """h: the collapsed stream, fp32 [1, 1, T, hidden/tp] -> the FFN / MoE output, fp32, same layout."""
        x = self.post_attention_layernorm(h)
        out = ttnn.typecast(self.mlp(x), self.stream_dtype)
        if record is not None:
            record["mlp_in"], record["mlp_out"] = x, out
        return out

    def forward(self, streams, record=None, **attention_inputs):
        """streams: fp32 [1, 1, T, n * hidden/tp] -> the same layout after the layer.

        ``attention_inputs`` go to the attention callable untouched. ``record``, a dict, receives the
        device tensor of every reference.decoder.STAGES entry; ``streams`` is not freed.
        """
        rec = {} if record is None else record
        rec["streams_in"] = streams

        h, post = self.hc_attn.hc_pre(streams)
        rec["attn_hc"], rec["attn_post"] = h, post
        out = self.attention_sublayer(h, rec, **attention_inputs)
        mid = self.hc_attn.hc_post(out, streams, post)
        rec["streams_mid"] = mid

        h, post = self.hc_mlp.hc_pre(mid)
        rec["mlp_hc"], rec["mlp_post"] = h, post
        out = self.mlp_sublayer(h, rec)
        streams = self.hc_mlp.hc_post(out, mid, post)
        rec["streams_out"] = streams
        return streams
