# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Torch reference for Kimi K3 attention residuals (AttnRes).

Softmax attention over residual-stream snapshots: one live stream `running_sum`
plus `S` write-once sealed snapshots `block_residual`, mixed by a softmax over
RMS-normalized scores. The invariants that bite:

  * Keys are RMS-normalized, values are **not** — the mixture is over raw `v`.
  * RMS is a per-(token, candidate) scalar, so the normalized tensor is never
    materialized: `score = rsqrt(mean(v²) + eps) · ⟨q, v⟩`.
  * `q` folds `res_norm.weight * res_proj.weight` at load time, which is why
    this file never applies a norm gain.

Two forms of the read live here. `attn_res` is the direct one. `inter_block` +
`merge` split it so the sealed half amortizes across a whole 12-layer block —
algebraically exact, and the form the device op is structured around. The direct
form is retained permanently as the only independent check on that split's
online-softmax algebra.
"""

import torch

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config

HIDDEN_SIZE = KimiK3Config.EMB_SIZE
NUM_LAYERS = KimiK3Config.NUM_LAYERS
BLOCK_SIZE = KimiK3Config.ATTN_RES_BLOCK_SIZE

# The read takes its epsilon off the `res_norm` module, which the model builds with
# `rms_norm_eps` — AttnRes has no epsilon of its own to drift from the layer norms'.
EPS = KimiK3Config.RMS_NORM_EPS


def fold_query(norm_weight, proj_weight):
    """Collapse `res_norm.weight` and `res_proj.weight` into one `[d]` query."""
    return norm_weight.reshape(-1).float() * proj_weight.reshape(-1).float()


def attn_res_scores(v, q, eps=EPS):
    """Scores for every candidate.

    Args:
        v: [N, C, d] candidates, fp32.
        q: [d] folded query.

    Returns:
        [N, C], `v.dtype`.
    """
    rms_inv = torch.rsqrt(v.pow(2).mean(-1) + eps)
    return (v * q).sum(-1) * rms_inv


def attn_res(running_sum, block_residual, q, eps=EPS):
    """The AttnRes read.

    Args:
        running_sum: [N, d] live residual stream.
        block_residual: [N, S, d] sealed snapshots. S == 0 is legal and is the
            identity, since a one-candidate softmax is 1.
        q: [d] folded query.

    Returns:
        [N, d] in `running_sum.dtype`.
    """
    v = torch.cat((block_residual, running_sum.unsqueeze(1)), dim=1)
    v_f = v.float()
    probs = attn_res_scores(v_f, q.float(), eps).softmax(-1)
    out = torch.matmul(probs.unsqueeze(1), v_f).squeeze(1)
    return out.to(v.dtype)


def attn_res_inter_block(block_residual, q_batch, eps=EPS):
    """Sealed-snapshot half of the mixture, shared by every read site in a block.

    Within one 12-layer block all read sites see the identical sealed set and
    the queries are static parameters, so this single pass over `block_residual`
    amortizes across the block's 24 reads.

    Returns the mixture in unnormalized online-softmax form — `e_i = exp(s_i - m)`
    — because `merge` still has to fold in a candidate whose score is unknown
    here.

    Args:
        block_residual: [N, S, d] sealed snapshots.
        q_batch: [R, d] one folded query per read site in the block.

    Returns:
        partials: [R, N, d] `Σ_i e_i v_i`.
        m: [R, N] shift. `-inf` when S == 0.
        z: [R, N] mass `Σ_i e_i`. Zero when S == 0.
    """
    v = block_residual.float()
    n, s, d = v.shape
    r = q_batch.shape[0]

    if s == 0:
        # An empty mixture. The -inf shift makes `merge`'s rescale factor exactly
        # zero, so it collapses to the live stream without a special case there.
        return (
            v.new_zeros(r, n, d),
            v.new_full((r, n), -float("inf")),
            v.new_zeros(r, n),
        )

    rms_inv = torch.rsqrt(v.pow(2).mean(-1) + eps)
    scores = torch.einsum("nsd,rd->rns", v, q_batch.to(v.dtype)) * rms_inv.unsqueeze(0)
    m = scores.amax(-1)
    e = torch.exp(scores - m.unsqueeze(-1))
    return torch.einsum("rns,nsd->rnd", e, v), m, e.sum(-1)


def attn_res_merge(partial, m, z, running_sum, q, eps=EPS):
    """Fold the live stream into a precomputed sealed-snapshot partial.

    Args:
        partial: [N, d] from `attn_res_inter_block`, for this read site.
        m: [N] shift from `attn_res_inter_block`.
        z: [N] mass from `attn_res_inter_block`.
        running_sum: [N, d] live residual stream.
        q: [d] folded query, the same one used to build `partial`.

    Returns:
        [N, d] in `running_sum.dtype`, equal to `attn_res` up to fp32 rounding.
    """
    v_live = running_sum.float()
    s_live = attn_res_scores(v_live.unsqueeze(1), q.to(v_live.dtype), eps).squeeze(1)

    m_new = torch.maximum(m, s_live)
    rescale = torch.exp(m - m_new)
    e_live = torch.exp(s_live - m_new)

    num = rescale.unsqueeze(-1) * partial + e_live.unsqueeze(-1) * v_live
    return (num / (rescale * z + e_live).unsqueeze(-1)).to(running_sum.dtype)


class AttnResStream(object):
    """The `block_residual` lifecycle, mirroring `_forward_attn_residual`.

    One live stream and write-once snapshots. Writes are plain `+=` with weight
    one; AttnRes rewrites only the read. `running_sum` is `None` between a seal
    and the next `accumulate` — the layer pipeline places no read site in that
    window, so `read` asserts rather than guessing.
    """

    def __init__(self, hidden_states, block_size=BLOCK_SIZE, eps=EPS):
        """Args: hidden_states: [N, d] token embeddings, the first live stream."""
        n, d = hidden_states.shape
        self.running_sum = hidden_states
        self.block_residual = hidden_states.new_zeros(n, 0, d)
        self.block_size = block_size
        self.eps = eps

    @property
    def num_sealed(self):
        return self.block_residual.shape[1]

    def read(self, q):
        assert self.running_sum is not None, "no live stream between seal and accumulate"
        return attn_res(self.running_sum, self.block_residual, q, self.eps)

    def seal(self):
        self.block_residual = torch.cat((self.block_residual, self.running_sum.unsqueeze(1)), dim=1)
        self.running_sum = None

    def accumulate(self, module_out):
        if self.running_sum is None:
            self.running_sum = module_out
        else:
            self.running_sum = self.running_sum + module_out


def attn_res_layer(stream, layer_idx, q_pre, q_post, attn_fn, mlp_fn):
    """One layer's residual bookkeeping, in reference order.

    `attn_fn` and `mlp_fn` stand in for everything between the reads; the two
    layernorms fold into them since each sits immediately before its module.

    The pre-attention read is skipped at `S == 0` (only layer 0), matching the
    reference. The pre-MLP read is unconditional — at layer 0 the seal has
    already run, so it mixes two candidates.
    """
    h = stream.running_sum
    if stream.num_sealed > 0:
        h = stream.read(q_pre)

    if layer_idx % stream.block_size == 0:
        stream.seal()

    stream.accumulate(attn_fn(h))
    h = stream.read(q_post)
    stream.accumulate(mlp_fn(h))


def attn_res_stack(hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns, block_size=BLOCK_SIZE, eps=EPS, hook=None):
    """Walk a whole stack's residual bookkeeping.

    Args:
        hidden_states: [N, d] token embeddings.
        q_pre, q_post: sequences of [d] folded queries, one per layer.
        q_out: [d] folded query for the single model-level read.
        attn_fns, mlp_fns: per-layer callables `[N, d] -> [N, d]`.
        hook: optional `(layer_idx, stream) -> None`, called after each layer.
            The depth harness uses it to record a per-layer PCC curve.

    Returns:
        [N, d] — what `model.norm` sees.
    """
    # Unequal sequences would walk their common prefix and still return a plausible [N, d].
    # This is the oracle every device gate is scored against, so a short walk has to raise
    # rather than quietly move the target.
    lengths = (len(attn_fns), len(mlp_fns), len(q_pre), len(q_post))
    assert len(set(lengths)) == 1, f"attn_fns/mlp_fns/q_pre/q_post have lengths {lengths}"

    stream = AttnResStream(hidden_states, block_size=block_size, eps=eps)
    for layer_idx, (attn_fn, mlp_fn) in enumerate(zip(attn_fns, mlp_fns, strict=True)):
        attn_res_layer(stream, layer_idx, q_pre[layer_idx], q_post[layer_idx], attn_fn, mlp_fn)
        if hook is not None:
            hook(layer_idx, stream)
    return stream.read(q_out)
