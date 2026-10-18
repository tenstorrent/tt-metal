# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The AttnRes seal schedule, driven by the vendored HuggingFace read.

A read mixes the live residual stream with `S` write-once sealed snapshots by a
softmax over RMS-normalized scores, and mixes the **raw** candidates rather than
the normalized ones. That read is `_apply_attn_res`, the HuggingFace function
itself, so nothing here re-derives it — `hf_attn_res` below only restates its
argument list in plain tensors.

What this file holds is the part no HuggingFace function exposes: which layers
seal, which reads see how many candidates, and when the live stream is absent.
`attn_res.py` walks the same schedule against the folded query; everything here is
prefixed `hf_` because it walks it against the vendored read and the two weights
unfolded, which is the whole difference between the two.

    from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.hf_walk import hf_stack
    out = hf_stack(hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns, block_size, eps)
"""

import torch

from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.hf_attn_res import _apply_attn_res


# The underscore is upstream's own naming, kept because that function is vendored
# byte-identical. Its module holds no Tenstorrent code, which is what makes these two
# shims live here: they exist only so the vendored read can be called without importing
# `transformers` or building a `KimiRMSNorm`, since it touches nothing but `norm.weight`,
# `norm.variance_epsilon` and `proj.weight`.
class _NormShim:
    def __init__(self, weight, variance_epsilon):
        self.weight = weight
        self.variance_epsilon = variance_epsilon


class _ProjShim:
    def __init__(self, weight):
        self.weight = weight


def hf_attn_res(running_sum, block_residual, norm_weight, proj_weight, eps):
    """Call the vendored read with plain tensors.

    Args:
        running_sum: [N, d] live residual stream.
        block_residual: [N, S, d] sealed snapshots. S == 0 is legal.
        norm_weight: [d] `*_res_norm.weight`.
        proj_weight: [1, d] `*_res_proj.weight`.
        eps: `rms_norm_eps`.

    Returns:
        [N, d] in `running_sum.dtype`, computed in fp32 regardless of that dtype.
    """
    return _apply_attn_res(
        running_sum,
        block_residual,
        _ProjShim(proj_weight),
        _NormShim(norm_weight, eps),
    )


class HfStream(object):
    """The `block_residual` lifecycle: one live stream, write-once snapshots.

    Writes are plain `+=` with weight one — AttnRes rewrites only the read.
    `running_sum` is `None` between a seal and the next `accumulate`; the layer
    pipeline places no read site in that window, so `read` asserts rather than
    guessing what the live candidate would be.
    """

    def __init__(self, hidden_states, block_size, eps):
        """Args: hidden_states: [N, d] token embeddings, the first live stream."""
        num_tokens, hidden_size = hidden_states.shape
        self.running_sum = hidden_states
        self.block_residual = hidden_states.new_zeros(num_tokens, 0, hidden_size)
        self.block_size = block_size
        self.eps = eps

    @property
    def num_sealed(self):
        return self.block_residual.shape[1]

    def read(self, query):
        """Args: query: `(norm_weight, proj_weight)` — unfolded, as the model stores them."""
        assert self.running_sum is not None, "no live stream between seal and accumulate"
        norm_weight, proj_weight = query
        return hf_attn_res(self.running_sum, self.block_residual, norm_weight, proj_weight, self.eps)

    def seal(self):
        self.block_residual = torch.cat((self.block_residual, self.running_sum.unsqueeze(1)), dim=1)
        self.running_sum = None

    def accumulate(self, module_out):
        if self.running_sum is None:
            self.running_sum = module_out
        else:
            self.running_sum = self.running_sum + module_out


def hf_layer(stream, layer_idx, q_pre, q_post, attn_fn, mlp_fn):
    """One layer's residual bookkeeping.

    `attn_fn` and `mlp_fn` stand in for everything between the reads; the two
    layernorms fold into them since each sits immediately before its module.

    The pre-attention read is skipped at `S == 0`, which happens only at layer 0.
    The pre-MLP read is unconditional — layer 0's seal has already run by then,
    so it mixes two candidates.

    Args:
        q_pre, q_post: `(norm_weight, proj_weight)` pairs.
    """
    hidden = stream.running_sum
    if stream.num_sealed > 0:
        hidden = stream.read(q_pre)

    if layer_idx % stream.block_size == 0:
        stream.seal()

    stream.accumulate(attn_fn(hidden))
    hidden = stream.read(q_post)
    stream.accumulate(mlp_fn(hidden))


def hf_stack(hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns, block_size, eps, hook=None):
    """Walk a whole stack's residual bookkeeping.

    Args:
        q_pre, q_post: sequences of `(norm_weight, proj_weight)` pairs, one per layer.
        q_out: the `(norm_weight, proj_weight)` pair for the single model-level read.
        attn_fns, mlp_fns: per-layer callables `[N, d] -> [N, d]`.
        hook: optional `(layer_idx, stream) -> None`, called after each layer.

    Returns:
        [N, d] — what `model.norm` sees.
    """
    # Truncating here would be worse than in `attn_res_stack`: the gate that compares the two
    # hands both walks the same sequences, so a short walk agrees with itself and passes.
    lengths = (len(attn_fns), len(mlp_fns), len(q_pre), len(q_post))
    assert len(set(lengths)) == 1, f"attn_fns/mlp_fns/q_pre/q_post have lengths {lengths}"

    stream = HfStream(hidden_states, block_size=block_size, eps=eps)
    for layer_idx, (attn_fn, mlp_fn) in enumerate(zip(attn_fns, mlp_fns, strict=True)):
        hf_layer(stream, layer_idx, q_pre[layer_idx], q_post[layer_idx], attn_fn, mlp_fn)
        if hook is not None:
            hook(layer_idx, stream)
    return stream.read(q_out)
