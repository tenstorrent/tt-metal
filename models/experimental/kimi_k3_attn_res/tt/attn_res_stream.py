# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-side `block_residual` lifecycle for Kimi K3 attention residuals, plus
the layer and stack drivers that walk it.

Interface-compatible with `torch_functional.attn_res` — same `prefix_sum` /
`num_sealed` / `read` / `seal` / `accumulate` / `block_size` on the stream, same
signature on `attn_res_layer` — so the depth harness drives both backends through
one loop and the seal schedule and read count cannot diverge between them. The
in-layer order is per-backend, because only this one owns tensors; the harness's
per-layer PCC curve is what gates that half.
"""

import ttnn

BLOCK_SIZE = 12


class TtAttnResStream(object):
    """One live stream and write-once snapshots, on device.

    Writes are plain `+=` with weight one; AttnRes rewrites only the read.
    `prefix_sum` is None between a seal and the next `accumulate` — the layer
    pipeline places no read site in that window, so `read` asserts rather than
    guessing.

    **The stream owns its tensors.** Construction transfers ownership of
    `hidden_states`, and `accumulate` takes ownership of `module_out`. That makes
    the first `seal` a zero-copy ownership move into `block_residual` instead of a
    clone, and lets every later `seal` free what it superseded.

    Args:
        op: a `TtAttnRes`.
        hidden_states: `[1, 1, N, d]` token embeddings, the first live stream.
        block_size: layers per block; seals fire at `layer_idx % block_size == 0`.
    """

    def __init__(self, op, hidden_states, block_size=BLOCK_SIZE):
        self.op = op
        self.prefix_sum = hidden_states
        self.block_residual = None
        self.block_size = block_size

    @property
    def num_sealed(self):
        return 0 if self.block_residual is None else self.block_residual.shape[1]

    def read(self, q):
        assert self.prefix_sum is not None, "no live stream between seal and accumulate"
        return self.op.forward(self.prefix_sum, self.block_residual, q)

    def seal(self):
        assert self.prefix_sum is not None, "nothing to seal"
        if self.block_residual is None:
            self.block_residual = self.prefix_sum
        else:
            grown = ttnn.concat([self.block_residual, self.prefix_sum], dim=1)
            ttnn.deallocate(self.block_residual)
            ttnn.deallocate(self.prefix_sum)
            self.block_residual = grown
        self.prefix_sum = None

    def accumulate(self, module_out):
        if self.prefix_sum is None:
            self.prefix_sum = module_out
        else:
            total = ttnn.add(self.prefix_sum, module_out)
            ttnn.deallocate(self.prefix_sum)
            ttnn.deallocate(module_out)
            self.prefix_sum = total

    def deallocate(self):
        for tensor in (self.prefix_sum, self.block_residual):
            if tensor is not None:
                ttnn.deallocate(tensor)
        self.prefix_sum = None
        self.block_residual = None


def attn_res_layer(stream, layer_idx, q_pre, q_post, attn_fn, mlp_fn):
    """One layer's residual bookkeeping, in reference order.

    `attn_fn` and `mlp_fn` stand in for everything between the two reads. Each
    module's own input layernorm folds into its callable — that norm is distinct
    from the `*_res_norm` already folded into `q_pre` / `q_post`, and both exist
    in the checkpoint. A callable borrows the tensor it is handed and must not
    free it; ownership of what it returns passes to the stream.

    The pre-attention read is skipped at `S == 0`, which is only layer 0. `h`
    aliases the stream's own `prefix_sum` there and the seal below takes
    ownership of it, so freeing it would free `block_residual`.
    """
    h, borrowed = stream.prefix_sum, True
    if stream.num_sealed > 0:
        h, borrowed = stream.read(q_pre), False

    if layer_idx % stream.block_size == 0:
        stream.seal()

    stream.accumulate(attn_fn(h))
    if not borrowed:
        ttnn.deallocate(h)

    h = stream.read(q_post)
    stream.accumulate(mlp_fn(h))
    ttnn.deallocate(h)


def _block_sites(layers, q_pre, q_post, q_out, block_size):
    """Queries grouped by the sealed set they read, in the order the walk consumes them.

    A `block_residual` value outlives the block that installed it by one read. The seal
    fires in the middle of layer `L`, between its two reads, so the sealed set spans layer
    `L`'s post-attention read through layer `L + block_size`'s pre-attention read. Groups
    are cut on the read sequence every `2 * block_size` sites for that reason, not on layer
    boundaries — and the trailing group is short whenever the stack does not divide evenly.
    """
    order = []
    for layer_idx in range(layers):
        if layer_idx > 0:
            order.append(q_pre[layer_idx])
        order.append(q_post[layer_idx])
    order.append(q_out)

    sites = 2 * block_size
    return [order[start : start + sites] for start in range(0, len(order), sites)]


def attn_res_stack_split(op, hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns, block_size=BLOCK_SIZE, hook=None):
    """Walk a whole stack's residual bookkeeping through the split read form.

    Same schedule and same reads as `attn_res_stack`, equal to it up to rounding, with the
    sealed half of every read hoisted into one `inter_block` per block. A sealed snapshot is
    write-once, so that half is loop-invariant across a block's read sites; only `merge`,
    which folds in the live stream, has to stay per site.

    Layer 0's pre-attention read is skipped for want of a snapshot, so every executed read
    runs at `S >= 1` and `inter_block`'s non-None precondition holds for all of them. The
    direct form is never needed as a fallback.

    Args: as `attn_res_stack`.
    """
    stream = TtAttnResStream(op, hidden_states, block_size=block_size)
    blocks = iter(_block_sites(len(attn_fns), q_pre, q_post, q_out, block_size))
    pending = iter(())
    partials = shifts = masses = None

    def read():
        site, q = next(pending)
        return op.merge(partials, shifts, masses, stream.prefix_sum, q, site)

    for layer_idx, (attn_fn, mlp_fn) in enumerate(zip(attn_fns, mlp_fns)):
        h, borrowed = stream.prefix_sum, True
        if stream.num_sealed > 0:
            h, borrowed = read(), False

        if layer_idx % block_size == 0:
            stream.seal()
            # The read just above is the outgoing block's last site, so replacing `pending`
            # and freeing its batches here strands nothing. `inter_block` has to follow the
            # seal, not precede it — the snapshot it batches over is the one just installed.
            queries = next(blocks)
            if shifts is not None:
                ttnn.deallocate(partials)
                ttnn.deallocate(shifts)
                ttnn.deallocate(masses)
            partials, shifts, masses = op.inter_block(stream.block_residual, queries)
            pending = enumerate(queries)

        stream.accumulate(attn_fn(h))
        if not borrowed:
            ttnn.deallocate(h)

        h = read()
        stream.accumulate(mlp_fn(h))
        ttnn.deallocate(h)

        if hook is not None:
            hook(layer_idx, stream)

    out = read()
    ttnn.deallocate(partials)
    ttnn.deallocate(shifts)
    ttnn.deallocate(masses)
    stream.deallocate()
    return out


def attn_res_stack(op, hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns, block_size=BLOCK_SIZE, hook=None):
    """Walk a whole stack's residual bookkeeping.

    Args:
        op: a `TtAttnRes`.
        hidden_states: `[1, 1, N, d/tp_factor]` token embeddings, placed with
            `op.stream_mapper`. Ownership passes to the stream.
        q_pre, q_post: folded queries from `op.to_query`, one per layer.
            `q_pre[0]` is never read — layer 0 has no sealed snapshot — so a
            93-layer stack holds 187 queries and takes 186 reads.
        q_out: folded query for the single model-level read.
        attn_fns, mlp_fns: per-layer callables, see `attn_res_layer`.
        block_size: layers per block; seals fire at `layer_idx % block_size == 0`.
        hook: optional `(layer_idx, stream) -> None`, called after each layer.

    Returns:
        `[1, 1, N, d/tp_factor]` — what `model.norm` sees. The stream is freed
        here; the returned tensor is the caller's.
    """
    stream = TtAttnResStream(op, hidden_states, block_size=block_size)
    for layer_idx, (attn_fn, mlp_fn) in enumerate(zip(attn_fns, mlp_fns)):
        attn_res_layer(stream, layer_idx, q_pre[layer_idx], q_post[layer_idx], attn_fn, mlp_fn)
        if hook is not None:
            hook(layer_idx, stream)

    out = stream.read(q_out)
    stream.deallocate()
    return out
