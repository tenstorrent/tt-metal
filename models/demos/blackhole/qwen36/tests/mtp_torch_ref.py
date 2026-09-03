# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-torch reference for the Qwen3.5/3.6 MTP (multi-token prediction) drafter head.

Shared by the device PCC test (tests/test_mtp_tp.py) and the off-device acceptance oracle
(tests/mtp_cpu_check.py) so both score the drafter against the same math:

    h'  = fc( concat[ enorm(embed(token)), hnorm(hidden) ] )
    h'' = DecoderLayer(h')                  # one gated full-attention layer + SwiGLU MLP
    logits = LMHead( mtp.norm(h'') )
    chain  = mtp.norm(h'')  (default)  |  h''  (chain_postnorm=False)    # the next step's hidden

``chain_postnorm`` mirrors the device head's feed contract (Qwen36MTP.forward_decode under
QWEN36_SPEC_POSTNORM=1, "V3", the default): the next chained step is fused from mtp.norm's output
instead of the raw block output. Default True is the V3 contract; False restores V0.
``logits`` always consumes the raw block.

Two forms with identical math (pinned by tests/test_mtp_torch_ref.py, CPU-only):

* ``forward_sequence`` — all S slots in one causal pass; equals a drafter reading a fully
  warmed MTP KV cache over a prefix.
* ``forward_step``     — one slot against a supplied K/V cache; the autoregressive draft chain.

Slot semantics are the CALLER's choice: this class fuses whatever (hidden, token) pair it is
handed at whatever position it is handed. The two candidate conventions —- ``(h_i, t_i)`` and
``(h_i, t_{i+1})`` -— are what mtp_cpu_check.py measures, so nothing here assumes either.
"""
import json
import os

import torch
import torch.nn.functional as F

EPS = 1e-6
LAYER = "mtp.layers.0."


def load_head_sd(ckpt_dir):
    """Load only the tensors the MTP head needs: embedding, LM head, final norm, and mtp.* (15).

    Avoids the full-model load — remap gives tok_embeddings/output/norm, load_mtp_tensors the
    drafter weights.
    """
    from safetensors import safe_open

    from models.demos.blackhole.qwen36.tt.weight_mapping import load_mtp_tensors, remap_qwen36_state_dict

    head_keys = {"lm_head.weight", "model.language_model.embed_tokens.weight", "model.language_model.norm.weight"}
    wm = json.load(open(os.path.join(ckpt_dir, "model.safetensors.index.json")))["weight_map"]
    file_to_keys = {}
    for k, fn in wm.items():
        if k in head_keys:
            file_to_keys.setdefault(fn, []).append(k)
    raw = {}
    for fn, keys in file_to_keys.items():
        with safe_open(os.path.join(ckpt_dir, fn), framework="pt") as sf:
            for k in keys:
                raw[k] = sf.get_tensor(k)
    sd = remap_qwen36_state_dict(raw)  # -> tok_embeddings.weight, output.weight, norm.weight
    sd.update(load_mtp_tensors(ckpt_dir))  # -> mtp.* (15)
    return sd


def rms(x, raw_weight, eps=EPS):
    """Zero-centered RMSNorm reference: out = x * rsqrt(mean(x^2)+eps) * (1 + raw_weight)."""
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return xf * torch.rsqrt(var + eps) * (1.0 + raw_weight.float())


def apply_rope(x, rope_dim, theta, positions):
    """Partial HF split-halves RoPE on [S, H, HD] at the given absolute ``positions`` [S].

    Only the leading ``rope_dim`` of each head passes through the rotation; the tail is copied.
    For text (t==h==w) M-RoPE collapses to this 1D form, so it matches rot_mats_prefill's
    position_ids=None case.
    """
    inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    emb = torch.cat([torch.outer(positions.float(), inv)] * 2, dim=-1)  # [S, rope_dim]
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    xr, xp = x[..., :rope_dim], x[..., rope_dim:]
    r1, r2 = xr[..., : rope_dim // 2], xr[..., rope_dim // 2 :]
    xrot = torch.cat([-r2, r1], dim=-1)
    return torch.cat([xr * cos + xrot * sin, xp], dim=-1)


class MTPTorchHead:
    """Pure-torch MTP head in fp32. Dims are derived from the state dict; only the RoPE
    parameters (partial rope width + theta) have to be supplied, since weight shapes do not
    encode them."""

    def __init__(self, sd, rope_dim, rope_theta, eps=EPS, chain_postnorm=True):
        self.sd = sd
        self.eps = eps
        self.rope_dim = rope_dim
        self.rope_theta = rope_theta
        self.chain_postnorm = chain_postnorm
        fc = sd["mtp.fc.weight"]
        self.dim = fc.shape[0]
        assert fc.shape[1] == 2 * self.dim, f"unexpected mtp.fc shape {tuple(fc.shape)}"
        self.head_dim = sd[LAYER + "self_attn.q_norm.weight"].shape[0]
        # q_proj emits [query | gate] per head, hence the 2x.
        self.n_heads = sd[LAYER + "self_attn.q_proj.weight"].shape[0] // (2 * self.head_dim)
        self.n_kv_heads = sd[LAYER + "self_attn.k_proj.weight"].shape[0] // self.head_dim
        self.group = self.n_heads // self.n_kv_heads
        self.scale = self.head_dim**-0.5
        self.vocab_size = sd["output.weight"].shape[0]
        self._fp32 = {}

    def w(self, key):
        """fp32 view of a weight, converted once (the LM head alone is multiple GB)."""
        if key not in self._fp32:
            self._fp32[key] = self.sd[key].float()
        return self._fp32[key]

    def embed(self, tokens):
        """[S] token ids -> [S, dim] fp32. Looked up in the checkpoint dtype, then upcast."""
        return F.embedding(tokens, self.sd["tok_embeddings.weight"]).float()

    def fuse(self, hidden, tokens):
        """(hidden [S,dim], tokens [S]) -> fused [S,dim]. Concat order is [embedding, hidden]."""
        e = rms(self.embed(tokens), self.sd["mtp.pre_fc_norm_embedding.weight"], self.eps)
        h = rms(hidden, self.sd["mtp.pre_fc_norm_hidden.weight"], self.eps)
        return torch.cat([e, h], dim=-1) @ self.w("mtp.fc.weight").T

    def _qkv(self, fused, positions):
        """fused [S,dim] -> roped/normed (q, gate, k, v); q/gate [S,NH,HD], k/v [S,NKV,HD]."""
        xf = rms(fused, self.sd[LAYER + "input_layernorm.weight"], self.eps)
        S, HD = xf.shape[0], self.head_dim
        qg = (xf @ self.w(LAYER + "self_attn.q_proj.weight").T).reshape(S, self.n_heads, 2 * HD)
        q, gate = qg[..., :HD], qg[..., HD:]
        k = (xf @ self.w(LAYER + "self_attn.k_proj.weight").T).reshape(S, self.n_kv_heads, HD)
        v = (xf @ self.w(LAYER + "self_attn.v_proj.weight").T).reshape(S, self.n_kv_heads, HD)
        q = rms(q, self.sd[LAYER + "self_attn.q_norm.weight"], self.eps)
        k = rms(k, self.sd[LAYER + "self_attn.k_norm.weight"], self.eps)
        q = apply_rope(q, self.rope_dim, self.rope_theta, positions)
        k = apply_rope(k, self.rope_dim, self.rope_theta, positions)
        return q, gate, k, v

    def _attend(self, q, gate, k_all, v_all, q_offset):
        """Gated causal attention. q/gate [Sq,NH,HD]; k_all/v_all [Sk,NKV,HD]; query row i sits
        at absolute slot ``q_offset + i`` so it may attend to k rows 0..q_offset+i."""
        Sq, Sk = q.shape[0], k_all.shape[0]
        idx = torch.arange(self.n_heads) // self.group  # GQA expand
        kh = k_all[:, idx, :].permute(1, 0, 2)  # [NH,Sk,HD]
        vh = v_all[:, idx, :].permute(1, 0, 2)
        scores = torch.matmul(q.permute(1, 0, 2), kh.transpose(-1, -2)) * self.scale  # [NH,Sq,Sk]
        allowed = torch.arange(Sk)[None, :] <= (q_offset + torch.arange(Sq))[:, None]
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        ao = torch.matmul(torch.softmax(scores, dim=-1), vh).permute(1, 0, 2)  # [Sq,NH,HD]
        gated = ao * torch.sigmoid(gate)
        return gated.reshape(Sq, self.n_heads * self.head_dim) @ self.w(LAYER + "self_attn.o_proj.weight").T

    def _block_tail(self, fused, attn_out):
        """Attention residual + SwiGLU MLP residual -> the raw block output (what ``logits`` norms,
        and — through ``chain`` — the next chained hidden)."""
        h1 = fused + attn_out
        ff = rms(h1, self.sd[LAYER + "post_attention_layernorm.weight"], self.eps)
        mlp = (
            F.silu(ff @ self.w(LAYER + "mlp.gate_proj.weight").T) * (ff @ self.w(LAYER + "mlp.up_proj.weight").T)
        ) @ self.w(LAYER + "mlp.down_proj.weight").T
        return h1 + mlp

    def logits(self, block_out):
        """Raw block output -> [S, vocab] through mtp.norm + the shared LM head."""
        return rms(block_out, self.sd["mtp.norm.weight"], self.eps) @ self.w("output.weight").T

    def chain(self, block_out):
        """Raw block output -> the hidden the NEXT chained step is fused from.

        V3 (default, chain_postnorm=True): mtp.norm's output — the very tensor the LM head
        consumes — as Qwen36MTP.forward_decode chains under QWEN36_SPEC_POSTNORM=1 (the default).
        V0 (chain_postnorm=False): the block output itself, before mtp.norm.
        """
        if self.chain_postnorm:
            return rms(block_out, self.sd["mtp.norm.weight"], self.eps)
        return block_out

    def forward_sequence(self, hidden, tokens, positions=None, want_logits=True):
        """All S slots in one causal pass.

        Returns (logits [S,vocab] or None, chain [S,dim], k [S,NKV,HD], v [S,NKV,HD]). ``chain`` is
        the per-slot next-step hidden under the configured contract (see ``chain``). The k/v are the
        cache a later ``forward_step`` chain reads as its prefix.
        """
        S = hidden.shape[0]
        positions = torch.arange(S) if positions is None else positions
        fused = self.fuse(hidden, tokens)
        q, gate, k, v = self._qkv(fused, positions)
        block_out = self._block_tail(fused, self._attend(q, gate, k, v, 0))
        return (self.logits(block_out) if want_logits else None), self.chain(block_out), k, v

    def forward_step(self, hidden_row, token, position, past_k=None, past_v=None):
        """One slot against a K/V cache.

        Returns (logits [vocab], chain [dim], k_all, v_all) where ``chain`` is this slot's next-step
        hidden under the configured contract and k_all/v_all include this slot, ready to be passed
        straight back in as the next step's cache.
        """
        fused = self.fuse(hidden_row.reshape(1, -1), torch.as_tensor([int(token)]))
        q, gate, k, v = self._qkv(fused, torch.as_tensor([int(position)]))
        k_all = k if past_k is None else torch.cat([past_k, k], 0)
        v_all = v if past_v is None else torch.cat([past_v, v], 0)
        offset = 0 if past_k is None else past_k.shape[0]
        block_out = self._block_tail(fused, self._attend(q, gate, k_all, v_all, offset))
        return self.logits(block_out)[0], self.chain(block_out)[0], k_all, v_all


def mtp_reference(hidden, tokens, sd, args):
    """Sequence-form reference matching the device head's prefill path.

    hidden [1,S,dim], tokens [1,S] long -> logits [S, vocab]. Kept as a thin wrapper so
    test_mtp_tp.py's call signature is unchanged.
    """
    head = MTPTorchHead(sd, rope_dim=args.rope_head_dim, rope_theta=args.rope_theta)
    logits, _, _, _ = head.forward_sequence(hidden[0], tokens[0])
    return logits
