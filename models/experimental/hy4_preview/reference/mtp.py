# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch ground-truth reference for Hy4-preview's MTP layer (``model.mtp_layers.0``) and for drafting with it.

Pure torch, fp32, runs on CPU. The checkpoint holds one MTP layer (config.json ``num_nextn_predict_layers`` = 1):
a decoder block WITHOUT iHC (no ``hc_attn_layer`` / ``hc_mlp_layer`` tensors) with the backbone's gated MLA
attention (own sink, own gate, own DSA indexer) and a MoE (256 routed experts + 1 shared), plus four tensors of
its own: ``enorm``, ``hnorm``, ``eh_proj`` [hidden, 2 * hidden] and ``final_layernorm``. 27 tensors in all.

Source of the semantics. HF transformers' ``modeling_hy_v4.py`` does not implement the layer: HYV4PreTrainedModel
lists ``model.mtp_layers.*`` in ``_keys_to_ignore_on_load_unexpected``. The reference is vLLM 0.31.0, which the
model card serves with (``--speculative-config '{"num_speculative_tokens":3,"method":"mtp"}'``):

  vllm/models/hy_v4/nvidia/mtp.py
    HYV4MultiTokenPredictorLayer.forward (lines 372-392)
        e = enorm(inputs_embeds); h = hnorm(previous_hidden_states)
        x = eh_proj(cat([e, h], -1))                                 embedding first, hidden second
        y, residual = mtp_block(positions, x, residual=None)         HYV4DecoderLayer with enable_ihc = False (352-354)
        out = final_layernorm(y, residual)                           RMSNorm of y + residual
    HYV4MultiTokenPredictor.forward (494-511)
        inputs_embeds = where(positions == 0, 0, embed_tokens(input_ids))     the embedding is zeroed at position 0
    HYV4MultiTokenPredictor.compute_logits (513-521) and HYV4MTP.load_weights (741-744)
        logits = lm_head(out): the TARGET's lm_head and embed_tokens, no ``model.norm`` in between
    all three norms are RMSNorm(hidden, eps = rms_norm_eps) (342-343, 370)
  vllm/models/hy_v4/nvidia/model.py
    HYV4DecoderLayer._forward_normal (162-185): the plain pre-norm block
        a = x + self_attn(input_layernorm(x));  y = mlp(post_attention_layernorm(a));  residual = a
    HYV4Model.forward (388-393): what the target hands the drafter is ``norm(hc_head(streams))``, the final normed
        hidden the LM head reads. HYV4ForCausalLM defines no ``get_mtp_target_hidden_states`` override
        (vllm/v1/worker/gpu_model_runner.py 5145-5153; DeepSeek V4 uses that hook to pass its pre-head streams).
  vllm/models/hy_v4/nvidia/attention.py (313-334): the MTP layer is a DSA layer (its layer type is the backbone's
        last, "deepseek_sparse_attention") and always builds its own "full" indexer.
  vllm/v1/spec_decode/llm_base_proposer.py, SpecDecodeBaseProposer
    set_inputs_first_pass (834-854): the first pass runs the layer over every position the target just computed,
        with the tokens shifted by one: the row at position q takes the target's hidden h_q and the embedding of
        the token at q + 1 (the newly sampled token at the last row), RoPE and the layer's own KV entry at q.
    propose (615-774): draft 1 = argmax lm_head(row of the last position). For i >= 1 the SAME layer runs one row
        per request: its input ids are draft i, its position the previous one + 1, and its ``hidden_states`` the
        layer's OWN output of the step before (model_returns_tuple is False for this architecture, 1002-1016, so
        the hidden fed back is the final_layernorm output that produced draft i). Each step writes the layer's KV
        at its position and attends everything before it.
    1592-1603: ``index_share_for_mtp_iteration`` (steps 1+ reuse step 0's DSA picks) is read from the draft config;
        Hy4's config.json does not set it, so the indexer runs at every step.

So, with x the committed tokens (x_{p+1} the last one, sampled from the target's h_p) and M the layer:

    row q <= p (first pass)   o_q  = M(h_q, emb(x_{q+1})) at position q           its KV entry at q stays
    draft 1                   d_1  = argmax lm_head(o_p)                          predicts x_{p+2}
    draft j + 1               o'_j = M(o'_{j-1}, emb(d_j)) at position p + j      (o'_0 = o_p), attends the entries
                              d_{j+1} = argmax lm_head(o'_j)                      0 .. p and the draft rows p+1 .. p+j

The entries the draft rows write past p are overwritten by the next first pass, which re-runs the verified
positions with the target's hidden (that is what a device has to backfill).

Where the sources disagree or are silent:

* The two norms inside the attention (``q_a_layernorm``, ``kv_a_layernorm``): HF's HYV4RMSNorm default is 1e-6
  for the backbone, vLLM builds them with rms_norm_eps = 1e-5 (attention.py 387, 403) for the backbone and the MTP
  layer alike. This file follows HF (reference/attention.py AttentionConfig.norm_eps), the backbone's reference;
  the activations' mean square is of order one, so the choice moves values by about 1e-5 relative.
* SGLang (``--speculative-algorithm NEXTN --speculative-num-steps 3``) is not on this machine and was not read.
* Sampling: vLLM's proposer drafts greedily unless draft probabilities are requested; this file drafts greedily.

Attention here is dense and causal, as in reference/decoder.py: exact while a query sees at most ``index_topk``
(2048) positions, where the DSA top-k keeps every key. The layer's indexer weights are loaded (``indexer``) but
the selection is not run; ``draft`` refuses longer contexts.

Checkpoint names under ``model.mtp_layers.0.`` (MTP_PARAMETERS maps each to this module):
    enorm.weight  hnorm.weight  eh_proj.weight  final_layernorm.weight
    input_layernorm.weight  post_attention_layernorm.weight
    self_attn.*  (reference/attention.py, and self_attn.indexer.* of reference/indexer.py)   mlp.*  (reference/moe.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.hy4_preview.reference.attention import Hy4Attention, rms_norm, rope_cos_sin
from models.experimental.hy4_preview.reference.decoder import DecoderConfig
from models.experimental.hy4_preview.reference.indexer import Hy4Indexer, IndexerConfig
from models.experimental.hy4_preview.reference.moe import Hy4MoE


@dataclass
class MTPConfig:
    """MTP hyperparameters. Defaults are the tencent/Hy4-preview config.json values."""

    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    indexer: IndexerConfig = field(default_factory=IndexerConfig)
    norm_eps: float = 1e-5  # enorm / hnorm / final_layernorm ("rms_norm_eps")
    num_layers: int = 1  # "num_nextn_predict_layers"

    @property
    def hidden_size(self) -> int:
        return self.decoder.hidden_size


def _mtp_parameters() -> dict:
    names = {
        "enorm.weight": "enorm",
        "hnorm.weight": "hnorm",
        "eh_proj.weight": "eh_proj",
        "final_layernorm.weight": "final_layernorm",
        "input_layernorm.weight": "input_layernorm",
        "post_attention_layernorm.weight": "post_attention_layernorm",
    }
    for p in Hy4Attention.PARAMETER_NAMES:
        suffix = "" if p == "learnable_sink_param" else ".weight"
        names[f"self_attn.{p}{suffix}"] = f"self_attn.{p}"
    for attr, name in Hy4Indexer.CHECKPOINT_NAMES.items():
        names[f"self_attn.indexer.{name}"] = f"indexer.{attr}"
    for p, attr in Hy4MoE.PARAMETER_NAMES.items():
        names[f"mlp.{p}"] = f"mlp.{attr}"
    return names


# Checkpoint name (under ``model.mtp_layers.{i}.``) -> parameter path in Hy4MTP: all 27 tensors of the layer.
MTP_PARAMETERS = _mtp_parameters()


def checkpoint_names(layer: int = 0) -> list:
    """The full checkpoint names of MTP layer ``layer``, in MTP_PARAMETERS order."""
    return [f"model.mtp_layers.{layer}.{name}" for name in MTP_PARAMETERS]


@dataclass
class MTPCache:
    """The layer's own KV cache for one sequence, in position order: row q is position q, written by the row that
    took the target's hidden h_q. ``latent`` [P, kv_lora + rope] is (k_pass | k_rope after RoPE), the backbone's
    cache entry format (reference/attention_decode.py DecodeCache)."""

    latent: torch.Tensor

    def __len__(self) -> int:
        return self.latent.shape[0]


class Hy4MTP(nn.Module):
    """The MTP layer. A row is one position of one sequence: ``hidden`` [R, hidden] is what the layer is conditioned
    on there (the target's final normed hidden in the first pass, the layer's own output in a draft step) and
    ``embeds`` [R, hidden] the embedding of the token one position later. The output [R, hidden] goes straight to
    the target's LM head and predicts the token two positions after the row's."""

    def __init__(self, cfg: MTPConfig | None = None, init: bool = True):
        """``init=False`` leaves the routed experts uninitialised, for a caller that loads them."""
        super().__init__()
        self.cfg = cfg = cfg or MTPConfig()
        hidden = cfg.hidden_size
        self.enorm = nn.Parameter(torch.ones(hidden))
        self.hnorm = nn.Parameter(torch.ones(hidden))
        self.eh_proj = nn.Parameter(torch.empty(hidden, 2 * hidden))
        self.final_layernorm = nn.Parameter(torch.ones(hidden))
        self.input_layernorm = nn.Parameter(torch.ones(hidden))
        self.post_attention_layernorm = nn.Parameter(torch.ones(hidden))
        self.self_attn = Hy4Attention(cfg.decoder.attention)
        self.indexer = Hy4Indexer(cfg.indexer)
        self.mlp = Hy4MoE(cfg.decoder.moe, init=init)
        nn.init.normal_(self.eh_proj, std=0.006)

    # ------------------------------------------------------------------------------------------
    # weights
    # ------------------------------------------------------------------------------------------
    def load_one(self, name: str, value: torch.Tensor) -> None:
        """Take one parameter by its name under ``model.mtp_layers.{i}.``. A tensor that already has the
        parameter's dtype is adopted, not copied (the routed experts are 39 GB in fp32)."""
        p = self.get_parameter(MTP_PARAMETERS[name])
        assert value.numel() == p.numel(), f"{name}: {tuple(value.shape)} vs {tuple(p.shape)}"
        p.data = value.detach().to(p.dtype).reshape(p.shape)

    def load(self, tensors: dict) -> "Hy4MTP":
        """Take every parameter from ``tensors`` (name under ``model.mtp_layers.{i}.`` -> tensor)."""
        for name in MTP_PARAMETERS:
            self.load_one(name, tensors[name])
        return self

    @classmethod
    def from_checkpoint(cls, path=None, layer: int = 0, cfg: MTPConfig | None = None) -> "Hy4MTP":
        """The layer with the checkpoint's weights in fp32, read one tensor at a time."""
        from models.experimental.hy4_preview.reference import hf_harness

        mtp = cls(cfg, init=False).eval()
        for name, full in zip(MTP_PARAMETERS, checkpoint_names(layer)):
            mtp.load_one(name, hf_harness.checkpoint_tensors([full], path)[0])
        return mtp

    # ------------------------------------------------------------------------------------------
    # the pieces
    # ------------------------------------------------------------------------------------------
    def project(self, hidden: torch.Tensor, embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """eh_proj(cat(enorm(embeds), hnorm(hidden))), the block's input [R, hidden]. The embedding of a row at
        position 0 is zeroed first."""
        eps = self.cfg.norm_eps
        embeds = torch.where((positions == 0)[:, None], torch.zeros_like(embeds), embeds)
        e = rms_norm(embeds.float(), self.enorm, eps)
        h = rms_norm(hidden.float(), self.hnorm, eps)
        return F.linear(torch.cat([e, h], dim=-1), self.eh_proj)

    def latent(self, x: torch.Tensor, positions: torch.Tensor):
        """x: the block's input [R, hidden] at ``positions`` -> (attention input [R, hidden], q_resid [R, q_lora],
        cache entry [R, kv_lora + rope], cos, sin)."""
        a = rms_norm(x, self.input_layernorm, self.cfg.decoder.norm_eps)
        cos, sin = rope_cos_sin(positions, self.self_attn.cfg)
        q_resid, k_pass, k_rope = self.self_attn.latent(a, cos, sin)
        return a, q_resid, torch.cat([k_pass, k_rope], dim=-1), cos, sin

    def attend(self, a, q_resid, cos, sin, cache_latent, visible, extra=()):
        """Attention of R rows, each over its own keys: the first ``visible[r]`` rows of ``cache_latent`` [P, width]
        (the sequence's entries in position order) and its row of every tensor in ``extra`` ([R, width] each: a
        draft row's own entry and the entries of the draft rows before it). Returns the sublayer output [R, hidden].
        """
        attn, c = self.self_attn, self.self_attn.cfg
        R, P = a.shape[0], cache_latent.shape[0]
        q = attn.queries(q_resid, cos, sin)  # [H, R, qk]
        keys, values = attn.keys_values(*cache_latent.split([c.kv_lora_rank, c.qk_rope_head_dim], dim=-1))
        logits = torch.matmul(q, keys.transpose(-1, -2)) * c.scaling  # [H, R, P]
        hidden_keys = torch.arange(P)[None, :] >= visible[:, None]  # [R, P]
        parts = [logits.masked_fill(hidden_keys[None], torch.finfo(logits.dtype).min)]
        extra_values = []
        for latent in extra:
            k, v = attn.keys_values(*latent.split([c.kv_lora_rank, c.qk_rope_head_dim], dim=-1))  # [H, R, .]
            parts.append((q * k).sum(-1, keepdim=True) * c.scaling)
            extra_values.append(v)
        parts.append(attn.learnable_sink_param.to(logits.dtype).reshape(-1, 1, 1).expand(-1, R, -1))
        combined = torch.cat(parts, dim=-1)
        probs = F.softmax(combined - combined.max(dim=-1, keepdim=True).values, dim=-1)
        out = torch.matmul(probs[..., :P], values)  # [H, R, v]; the sink's share is dropped
        for i, v in enumerate(extra_values):
            out = out + probs[..., P + i : P + i + 1] * v
        out = out.transpose(0, 1).reshape(R, -1)
        return F.linear(out * attn.gate(a), attn.o_proj)

    def finish(self, x: torch.Tensor, attn_out: torch.Tensor, record: dict | None = None) -> torch.Tensor:
        """The residual, the MoE and the final norm: -> the layer's output [R, hidden]."""
        a = x + attn_out
        m_in = rms_norm(a, self.post_attention_layernorm, self.cfg.decoder.norm_eps)
        weights, indices = self.mlp.route(m_in)
        m = self.mlp.routed(m_in, self.mlp.dense_weights(weights, indices).to(m_in.dtype)) + self.mlp.shared(m_in)
        if record is not None:
            record.update(mlp_in=m_in, router_weights=weights, router_indices=indices)
        return rms_norm(a + m, self.final_layernorm, self.cfg.norm_eps)

    # ------------------------------------------------------------------------------------------
    # the two kinds of pass
    # ------------------------------------------------------------------------------------------
    def fill(self, hidden, embeds, positions=None, cache: MTPCache | None = None, record: dict | None = None):
        """The first pass: S consecutive positions of one sequence, the row at position q taking the target's final
        normed hidden h_q and the embedding of the token at q + 1. ``cache`` holds the entries of the positions
        before them (None: nothing is cached; ``positions`` may then start anywhere, for a window of a sequence).
        Returns (outputs [S, hidden], the cache with the rows' entries appended); the output of a row is its draft
        of the token at q + 2."""
        before = 0 if cache is None else len(cache)
        S = hidden.shape[0]
        positions = torch.arange(before, before + S) if positions is None else positions
        assert torch.equal(positions, positions[0] + torch.arange(S)), "a first pass runs consecutive positions"
        self._check_length(int(positions[-1]) + 1)
        x = self.project(hidden, embeds, positions)
        a, q_resid, latent, cos, sin = self.latent(x, positions)
        latent_all = latent if cache is None else torch.cat([cache.latent, latent], dim=0)
        visible = before + torch.arange(S) + 1  # the entries before the row, and its own
        out = self.finish(x, self.attend(a, q_resid, cos, sin, latent_all, visible), record)
        return out, MTPCache(latent_all)

    def step(self, hidden, embeds, positions, cache: MTPCache, visible, extra=(), record: dict | None = None):
        """One draft step of R rows (R anchors of the sequence ``cache`` belongs to, or one): row r at
        ``positions[r]`` takes the layer's previous output ``hidden[r]`` and the embedding of the previous draft, and
        attends the cache's first ``visible[r]`` entries, its row of every ``extra`` entry (the draft rows before
        it) and its own. Returns (outputs [R, hidden], the rows' own entries [R, width]); nothing is written to
        the cache."""
        self._check_length(int(positions.max()) + 1)
        x = self.project(hidden, embeds, positions)
        a, q_resid, latent, cos, sin = self.latent(x, positions)
        out = self.finish(x, self.attend(a, q_resid, cos, sin, cache.latent, visible, (*extra, latent)), record)
        return out, latent

    def _check_length(self, length: int) -> None:
        topk = self.cfg.decoder.index_topk
        assert length <= topk, (
            f"a context of {length} positions is longer than index_topk = {topk}: the layer's DSA indexer drops "
            "keys there, and dense attention is no longer what the layer computes"
        )

    def forward(self, hidden, embeds, positions=None):
        """``fill`` from position 0, outputs only."""
        return self.fill(hidden, embeds, positions)[0]


def greedy(outputs: torch.Tensor, lm_head: torch.Tensor) -> torch.Tensor:
    """The target's LM head on the layer's outputs [R, hidden] (fp32, no extra norm) -> greedy token ids [R]."""
    return F.linear(outputs.float(), lm_head.float()).argmax(dim=-1)


def draft(mtp: Hy4MTP, embedding, lm_head, hidden, next_tokens, depth: int = 3, anchors=None, record=None):
    """Draft ``depth`` tokens after every anchor of one sequence, as vLLM's MTP proposer would at that point.

    hidden [S, hidden]: the target's final normed hidden at positions 0 .. S - 1; next_tokens [S]: the token
    committed after each (``next_tokens[q]`` is at position q + 1, chosen from h_q); embedding / lm_head
    [vocab, hidden]: the target's. ``anchors`` (positions, default all): the positions p a proposal starts from,
    each seeing the layer's entries 0 .. p only.

    Returns (drafts int64 [A, depth], outputs [depth, A, hidden]): ``drafts[a, j]`` is the draft of the token at
    position p_a + 2 + j. ``record``, a list, receives one dict per draft depth with the MoE's ``router_indices``
    of the rows that ran (depth 0: all S rows of the first pass).
    """
    with torch.no_grad():
        S = hidden.shape[0]
        anchors = torch.arange(S) if anchors is None else torch.as_tensor(anchors)
        rec = {}
        out, cache = mtp.fill(hidden, embedding[next_tokens].float(), record=rec)
        if record is not None:
            record.append(rec)
        out = out[anchors]
        drafts, outputs, extra = [greedy(out, lm_head)], [out], []
        for j in range(1, depth):
            rec = {}
            out, latent = mtp.step(out, embedding[drafts[-1]].float(), anchors + j, cache, anchors + 1, extra, rec)
            extra.append(latent)
            drafts.append(greedy(out, lm_head))
            outputs.append(out)
            if record is not None:
                record.append(rec)
        return torch.stack(drafts, dim=1), torch.stack(outputs)


def accepted_lengths(drafts: torch.Tensor, tokens: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """How many of each anchor's drafts a greedy verify accepts: the leading drafts equal to the tokens the target
    committed. drafts [A, depth]; tokens [T]: the sequence by position; anchors [A]: draft j of anchor p stands for
    position p + 2 + j, which must exist in ``tokens``. Returns int64 [A], 0 .. depth."""
    depth = drafts.shape[1]
    target = torch.stack([tokens[anchors + 2 + j] for j in range(depth)], dim=1)
    return (drafts == target).long().cumprod(dim=1).sum(dim=1)
