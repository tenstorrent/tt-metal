# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ground-truth CPU reference for Kimi K3 attention residuals (AttnRes).

Written from the published definition — a read mixes the live residual stream
with `S` write-once sealed snapshots by a softmax over RMS-normalized scores, and
mixes the **raw** candidates rather than the normalized ones.

Deliberately the naive form. It materializes the normalized keys, applies
`res_norm.weight` and `res_proj.weight` as two separate factors in the order the
modules do, and spells the softmax and the mixture out as arithmetic.
`torch_functional/` and `tt/` both fold the two weights into one query and pull
`rsqrt` out of the dot; a reference that shared that algebra could not detect an
error in it, so nothing here takes a shortcut those paths take.

Computes in fp64 by default and never narrows an input, so it can measure an
fp32 implementation. `eps` and `block_size` are required arguments rather than
module constants — the model config owns those values, and a second copy here
would be free to drift from it.

Intended import form, which keeps the short names unambiguous at the call site:

    from models.experimental.kimi_k3_attn_res.reference import attn_res_reference as ref
    out = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, eps)
"""

import torch

DTYPE = torch.float64


def _widen(t, dtype):
    """Promote to `dtype` without ever narrowing."""
    return t.to(torch.promote_types(t.dtype, dtype))


def candidates(prefix_sum, block_residual, dtype=DTYPE):
    """The mixture's inputs, live stream last.

    Args:
        prefix_sum: [N, d] live residual stream.
        block_residual: [N, S, d] sealed snapshots. S == 0 is legal.

    Returns:
        [N, S + 1, d] in `dtype`.
    """
    return _widen(torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1), dtype)


def scores(prefix_sum, block_residual, norm_weight, proj_weight, eps, dtype=DTYPE):
    """One score per candidate, as `res_proj(res_norm(v))`.

    Three separate steps because that is three separate module operations: the
    gainless RMS normalization, the `res_norm` gain, then the `[1, d]` `res_proj`
    linear. Pre-multiplying the two weight vectors is already half the fold.

    Returns:
        [N, S + 1] in `dtype`.
    """
    v = candidates(prefix_sum, block_residual, dtype)
    normalized = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    gained = normalized * _widen(norm_weight, dtype).reshape(-1)
    return (gained * _widen(proj_weight, dtype).reshape(-1)).sum(-1)


def softmax_rows(row_scores):
    """Row softmax, max-shifted.

    Written out rather than called because the shift convention is what the
    device's online form has to reproduce, and a reference should show it.
    """
    weights = (row_scores - row_scores.amax(-1, keepdim=True)).exp()
    return weights / weights.sum(-1, keepdim=True)


def read(prefix_sum, block_residual, norm_weight, proj_weight, eps, dtype=DTYPE):
    """The AttnRes read.

    Args:
        prefix_sum: [N, d] live residual stream.
        block_residual: [N, S, d] sealed snapshots. S == 0 is the identity, since
            a one-candidate softmax is 1.
        norm_weight: [d] `*_res_norm.weight`.
        proj_weight: [1, d] or [d] `*_res_proj.weight`.
        eps: `config.rms_norm_eps`.
        dtype: internal precision. Never narrows an input below its own dtype.

    Returns:
        [N, d] in `prefix_sum.dtype`.
    """
    v = candidates(prefix_sum, block_residual, dtype)
    probs = softmax_rows(scores(prefix_sum, block_residual, norm_weight, proj_weight, eps, dtype))
    # The mixture is over `v`, not over the normalized keys. An explicit weighted
    # sum keeps that visible where a matmul would hide which tensor is mixed.
    return (probs.unsqueeze(-1) * v).sum(1).to(prefix_sum.dtype)


class Stream(object):
    """The `block_residual` lifecycle: one live stream, write-once snapshots.

    Writes are plain `+=` with weight one — AttnRes rewrites only the read.
    `prefix_sum` is `None` between a seal and the next `accumulate`; the layer
    pipeline places no read site in that window, so `read` asserts rather than
    guessing what the live candidate would be.
    """

    def __init__(self, hidden_states, block_size, eps, dtype=DTYPE):
        """Args: hidden_states: [N, d] token embeddings, the first live stream."""
        num_tokens, hidden_size = hidden_states.shape
        self.prefix_sum = hidden_states
        self.block_residual = hidden_states.new_zeros(num_tokens, 0, hidden_size)
        self.block_size = block_size
        self.eps = eps
        self.dtype = dtype

    @property
    def num_sealed(self):
        return self.block_residual.shape[1]

    def read(self, query):
        """Args: query: `(norm_weight, proj_weight)` — unfolded, as the model stores them."""
        assert self.prefix_sum is not None, "no live stream between seal and accumulate"
        norm_weight, proj_weight = query
        return read(self.prefix_sum, self.block_residual, norm_weight, proj_weight, self.eps, self.dtype)

    def seal(self):
        self.block_residual = torch.cat((self.block_residual, self.prefix_sum.unsqueeze(1)), dim=1)
        self.prefix_sum = None

    def accumulate(self, module_out):
        if self.prefix_sum is None:
            self.prefix_sum = module_out
        else:
            self.prefix_sum = self.prefix_sum + module_out


def layer(stream, layer_idx, q_pre, q_post, attn_fn, mlp_fn):
    """One layer's residual bookkeeping.

    `attn_fn` and `mlp_fn` stand in for everything between the reads; the two
    layernorms fold into them since each sits immediately before its module.

    The pre-attention read is skipped at `S == 0`, which happens only at layer 0.
    The pre-MLP read is unconditional — layer 0's seal has already run by then,
    so it mixes two candidates.

    Args:
        q_pre, q_post: `(norm_weight, proj_weight)` pairs.
    """
    hidden = stream.prefix_sum
    if stream.num_sealed > 0:
        hidden = stream.read(q_pre)

    if layer_idx % stream.block_size == 0:
        stream.seal()

    stream.accumulate(attn_fn(hidden))
    hidden = stream.read(q_post)
    stream.accumulate(mlp_fn(hidden))


def stack(hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns, block_size, eps, dtype=DTYPE, hook=None):
    """Walk a whole stack's residual bookkeeping.

    Args:
        q_pre, q_post: sequences of `(norm_weight, proj_weight)` pairs, one per layer.
        q_out: the `(norm_weight, proj_weight)` pair for the single model-level read.
        attn_fns, mlp_fns: per-layer callables `[N, d] -> [N, d]`.
        hook: optional `(layer_idx, stream) -> None`, called after each layer.

    Returns:
        [N, d] — what `model.norm` sees.
    """
    stream = Stream(hidden_states, block_size=block_size, eps=eps, dtype=dtype)
    for layer_idx, (attn_fn, mlp_fn) in enumerate(zip(attn_fns, mlp_fns)):
        layer(stream, layer_idx, q_pre[layer_idx], q_post[layer_idx], attn_fn, mlp_fn)
        if hook is not None:
            hook(layer_idx, stream)
    return stream.read(q_out)
