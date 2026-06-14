# MiniMax-M3 `sparse_lightning_attention` — TT-Lang authoring notes

## What this delivers

The only ttnn-missing piece of the MiniMax-M3 sparse layer is the **lightning
indexer**: dynamic, per-query, *block-granularity* key selection that no ttnn
SDPA op accepts. This directory authors the expressible core of that indexer as
a TT-Lang tile-math kernel and validates it END-TO-END against the bit-exact
reference golden, in the **functional simulator only** (device-free — no chips
touched).

- `sparse_lightning_attention.py` — the TT-Lang kernel
  (`make_block_score_op`): computes the indexer **block-scores**
  `[S_q, n_blocks]` =
  `amax_heads( amax_keys-in-block( idx_q @ idx_k^T + token_causal_mask ) )`
  with the `local_blocks=1` `+inf` boost applied.
- `test_sparse_lightning_attention_sim.py` — sim harness: runs the kernel in the
  functional sim, composes the full sparse attention host-side, asserts
  block-score bit-exactness, block-index set-equality, and end-to-end PCC.

## Validation result (functional sim)

```
[block_scores]  finite max_abs_diff = 0.0000   pcc = 1.000000   +inf_match=True  -inf_match=True
[block_indices] set-mismatch queries: 0/2560   match = True
[end-to-end]    output PCC vs golden = 1.000000
```

`status = ok`, `sim_pcc = 1.0`, `block_index_match = True`.

The block-scores match the reference to `max_abs = 0.0` because the sim runs with
its default **float32 promotion** (see "sim quirks"), so the kernel's tile matmul
+ reduce executes the identical fp32 math as the reference indexer.

## Tile / block layout chosen (grid = (1,1), single Tensix node)

`seq_len 2560 = 80 query-tile-rows`, `index_head_dim 128 = 4 tiles`,
`block_size 128 = 4 key-tiles`, `n_blocks 20`, `index_n_heads 4`.

| tensor | DRAM shape | DFB entry | meaning |
|--------|-----------|-----------|---------|
| `idxq` | `[H*S, D]` (row `h*S+s`) | `(1, D/32)` | one 32-query tile-row, all idx channels, one head |
| `idxk` | `[S, D]` | `(block_size/32, D/32)` = `(4,4)` | one whole 128-key block |
| `cmask`| `[S, S]` additive | `(1, block_size/32)` = `(1,4)` | token-causal slice (q-row × key-block) |
| `out`  | `[S, n_blocks*32]` | `(1,1)` | tile `(qr,b)` = broadcast block-max scalar; harness reads col `b*32` |

Per output tile `(qr, b)`: for each index head `h`, `sc = idxq[h,qr] @ idxk[b]^T`
(via `ttl.block.transpose` + `@`), add the causal mask slice, `reduce_max` over
the column (key) dim → `(1,1)` within-block max; accumulate across heads with
`ttl.math.max`. Token-level causality is applied *before* the block max-pool
(exactly matching the reference), via a standard additive mask. The local-block
boost forces `b == (qr*32)//block_size` to `+inf` (all 32 queries of a tile-row
share one 128-block, so the boost is per-tile-row, not per-query).

## What compiles in the SIM vs. known gaps

**Runs in the functional sim (this kernel):** tile `@` matmul,
`ttl.block.transpose`, additive masking (`+`), `ttl.math.reduce_max(dims=[-1])`
(within-tile amax + across-tile amax), `ttl.math.max` (head reduction),
`ttl.block.fill`, `ttl.block.broadcast`. All validated bit-exact.

**Genuine SIM gap — top-k block selection is NOT expressible in TT-Lang.**
tt-lang-sim 1.1.3 exposes no `topk` / `argsort` / `sort` / `argmax` / `scatter`
primitive in `ttl.math` or `ttl.block` (greppable: the only `sort` in the sim
package is the scheduler's profiling sort). The discrete top-16-block selection,
the `scatter`-based expansion of selected blocks into the dense additive
`[B,1,S_q,S_k]` mask, and the masked GQA SDPA are therefore done **host-side** in
the harness. This is the honest decomposition the task anticipated: the kernel
emits the *selection scores*; the argmax-like selection and the standard
additive-mask attention wrap around it.

**Sim quirks worth knowing:**
- *Sim-only wheel*: the DSL is under `ttl.sim` (`from ttl.sim import ttl, ttnn`),
  NOT at the top level (`ttl.operation` is absent there). `HAS_TT_DEVICE=False`.
- *float32 promotion*: importing `ttl.sim` rebinds `torch.bfloat16 → torch.float32`
  (toggle via `ttnn.set_disable_float32_promotion`). Two consequences:
  1. `ttnn.from_torch(dtype=ttnn.bfloat16)` actually stores fp32, so the kernel
     computes in fp32 — hence the perfect block-score match. On real hardware the
     bf16 matmul would introduce the usual bf16 rounding (still well within the
     PCC>0.99 bar; the reference golden itself was generated in bf16 with the
     indexer scores cast to fp32 before the amax, matching this kernel's contract).
  2. It **corrupts `torch.load` of bf16 checkpoints** (record-size mismatch). The
     harness works around this by loading the golden with native torch *before*
     importing `ttl.sim`.
- *Strict dataflow checker*: every block from `wait()` whose value is consumed
  must reach a `store`. The local-boost branch therefore does
  `acc = max(acc, fill(+inf))` instead of discarding `acc` and substituting a
  fresh `+inf` block.

## How it slots into the ttnn (device) phase — drop-in

The kernel is a **drop-in producer of the indexer block-scores**, from which the
ttnn phase builds the attention mask consumed by ttnn SDPA:

1. ttnn produces `idx_q`/`idx_k` (index q/k proj → Gemma RMS qk-norm → partial
   rope) — all standard ttnn matmul/rmsnorm/rope ops.
2. **This TT-Lang kernel** → boosted `block_scores [S_q, n_blocks]`.
3. ttnn / host top-k over `n_blocks` (small: 20) → block indices → expand to the
   additive `[B,1,S_q,S_k]` mask (`functional._build_block_mask`).
4. **Standard ttnn GQA SDPA** with that additive `attn_mask` (scale
   `head_dim**-0.5`) — the masked SDPA is an ordinary additive-mask attention; no
   custom op needed.

## Dense-GQA fallback (exact for seq ≤ 2048)

When `n_blocks ≤ topk_blocks` (i.e. `seq_len ≤ block_size * topk_blocks =
128*16 = 2048`), every key block is always selected, the block-sparse mask
reduces **exactly** to the dense causal mask, and `sparse_lightning_attention`
collapses to plain causal GQA (`gqa_attention_forward`). So for any prompt
≤ 2048 tokens the indexer/top-k machinery can be skipped entirely and a standard
causal ttnn SDPA is bit-exact — the indexer path (and this kernel) is only needed
beyond 2048 tokens. (The golden here is `seq_len=2560` → 20 blocks > 16, so the
sparsity path is genuinely exercised: e.g. query 2559 keeps blocks
`{0,1,2,3,6,7,9,10,11,12,14,15,16,17,18,19}`, dropping 4 of 20.)
