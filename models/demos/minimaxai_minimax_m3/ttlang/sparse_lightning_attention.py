# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TT-Lang kernel(s) for the MiniMax-M3 sparse "lightning attention" indexer.

The genuinely ttnn-missing piece of ``sparse_lightning_attention`` is the
*lightning indexer*: the dynamic, per-query, block-granularity key selection
that no ttnn SDPA op accepts. This module authors the expressible core of that
indexer as a TT-Lang tile-math kernel that runs in the functional simulator
(device-free):

    block_scores[B, S_q, n_blocks]
        = max over (index heads, keys-in-block) of  idx_q @ idx_k^T
          with strict token-causal masking, plus the local-block +inf boost.

``block_scores`` is *exactly* the tensor that feeds the top-k block selection in
``functional._lightning_indexer_block_indices``. The top-k selection itself, the
expansion to a dense additive attention mask, and the masked GQA SDPA are NOT
expressible with TT-Lang tile primitives (no top-k / argsort / scatter exists in
``ttl.math`` or ``ttl.block`` in tt-lang-sim 1.1.3 — see NOTES.md) and are done
host-side in the harness. That split is honest: the tt-lang kernel produces the
block-selection *scores*; the discrete selection + standard additive-mask SDPA
slot in around it (and map directly onto ttnn ops in the device phase).

Run the harness (``test_sparse_lightning_attention_sim.py``) with the tt-lang-sim
interpreter::

    /data/ttlang-venv/bin/python .../test_sparse_lightning_attention_sim.py

TT-Lang API note: the installed wheel is the *sim-only* build, so the DSL
entry points live under ``ttl.sim`` (``from ttl.sim import ttl, ttnn``) rather
than at the top level. The functional sim executes the kernel as pure Python.
"""

from ttl.sim import ttl

TILE = 32


def make_block_score_op(
    seq_len: int,
    index_head_dim: int,
    index_n_heads: int,
    block_size: int,
    n_blocks: int,
):
    """Build the TT-Lang operation that computes boosted indexer block-scores.

    Tile / block layout (single Tensix node, grid=(1, 1)):
      * ``idxq`` is the post-norm / post-rope index queries reshaped to a 2D
        DRAM tensor ``[index_n_heads * seq_len, index_head_dim]`` (head ``h``,
        query ``s`` lives on row ``h * seq_len + s``). Each DFB entry is a
        ``(1, index_head_dim/32)`` patch — one 32-query tile-row across all
        index-head channels.
      * ``idxk`` is the post-norm / post-rope index keys ``[seq_len,
        index_head_dim]``. Each DFB entry is a ``(block_size/32,
        index_head_dim/32)`` patch — one whole 128-key block.
      * ``cmask`` is a precomputed additive *token-causal* mask
        ``[seq_len, seq_len]`` (0 where key<=query, -inf where key is in the
        future). Each DFB entry is the ``(1, block_size/32)`` slice covering one
        query tile-row vs. one key block. Token-level causality is applied here,
        BEFORE the block max-pool, matching the reference exactly.
      * ``out`` is ``[seq_len, n_blocks * 32]``; tile ``(qr, b)`` holds the
        (broadcast) block-max scalar for query-tile-row ``qr`` and key-block
        ``b``. The harness reads column ``b * 32`` to recover ``block_scores``.

    Per output tile the kernel:
      1. for each index head ``h``: ``sc = idxq[h, qr] @ idxk[b]^T`` (a
         ``(1, block_size/32)`` row of key scores), add the causal mask slice,
         then ``reduce_max`` over the key (column) dim -> a ``(1, 1)`` per-block
         maximum (this is the within-block ``amax`` over the 128 keys);
      2. ``ttl.math.max`` accumulates across the ``index_n_heads`` heads (the
         second ``amax`` in the reference);
      3. the local-block boost: the 32 queries of tile-row ``qr`` all share the
         block ``q_block = (qr*32)//block_size`` (since 32 < block_size and
         blocks are 128-aligned), so when ``b == q_block`` the per-query own
         block is forced to ``+inf`` (``local_blocks == 1``); broadcast to the
         output tile and store.
    """
    St = seq_len // TILE  # query tile-rows
    Dt = index_head_dim // TILE  # index-head-dim tiles
    bst = block_size // TILE  # key-tiles per block (128/32 = 4)
    H = index_n_heads
    POS_INF = float("inf")

    @ttl.operation(grid=(1, 1))
    def block_score_op(idxq, idxk, cmask, out):
        qdfb = ttl.make_dataflow_buffer_like(idxq, shape=(1, Dt), block_count=2)
        kdfb = ttl.make_dataflow_buffer_like(idxk, shape=(bst, Dt), block_count=2)
        mdfb = ttl.make_dataflow_buffer_like(cmask, shape=(1, bst), block_count=2)
        odfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            for qr in range(St):
                q_block = (qr * TILE) // block_size
                for b in range(n_blocks):
                    with odfb.reserve() as ob:
                        acc = None
                        for _ in range(H):
                            with (
                                qdfb.wait() as qb,
                                kdfb.wait() as kb,
                                mdfb.wait() as mb,
                            ):
                                # scores for this (query tile-row, key block, head)
                                kt = ttl.block.transpose(kb)  # idx_k^T
                                sc = qb @ kt  # (1, bst) row of key scores
                                sc = sc + mb  # token-causal additive mask
                                # within-block amax over the block_size keys
                                bm = ttl.math.reduce_max(sc, dims=[-1], shape=(1, 1))
                                acc = bm if acc is None else ttl.math.max(acc, bm)

                        if b == q_block:
                            # local-block (+ sliding) boost -> always selected.
                            # Keep ``acc`` (the consumed scores) in the dataflow by
                            # max-ing it with +inf rather than discarding it; the
                            # strict sim requires every waited block to reach a store.
                            acc = ttl.math.max(acc, ttl.block.fill(POS_INF, shape=acc.shape))

                        ob.store(ttl.block.broadcast(acc, dims=[-1], shape=ob.shape))

        @ttl.datamovement()
        def read():
            for qr in range(St):
                for b in range(n_blocks):
                    for h in range(H):
                        with (
                            qdfb.reserve() as qb,
                            kdfb.reserve() as kb,
                            mdfb.reserve() as mb,
                        ):
                            ttl.copy(idxq[h * St + qr, 0:Dt], qb).wait()
                            ttl.copy(idxk[b * bst : (b + 1) * bst, 0:Dt], kb).wait()
                            ttl.copy(cmask[qr, b * bst : (b + 1) * bst], mb).wait()

        @ttl.datamovement()
        def write():
            for qr in range(St):
                for b in range(n_blocks):
                    with odfb.wait() as ob:
                        ttl.copy(ob, out[qr, b]).wait()

    return block_score_op
