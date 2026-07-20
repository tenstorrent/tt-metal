# 06 · Fusion, Residuals, and Op-Count Reduction

Every captured-trace op costs ~0.6 µs of host dispatch turnaround. A 24-layer transformer
has 300+ ops; removing 50 saves ~30 µs of pure turnaround plus the device cost of each op.
This is a real lever in the **host-bound** regime (small batch, decode) and free correctness
everywhere. This file collects the fusion patterns that held across BGE-M3, ViT, and Swin-L.

---

## 1. Fuse `add + Norm` into the norm's residual input {#fuse-residual-norm}
<!-- route
op_class: eltwise,reduction
rank: count
lever_type: single-shot
-->

The single most reusable fusion. Any norm that accepts `residual_input_tensor`:

```python
# 2 ops → 1 op, mathematically identical:
y = ttnn.layer_norm(x, residual_input_tensor=r, weight=γ, bias=β, ...)
```

Apply it at every `Norm(x + r)` in the graph — embedding position fold, both block
residuals. No PCC change. (See 02 §6.) If your layout keeps the residual block-sharded
(ViT), an explicit sharded `add` that avoids a reshard is equally fine — the goal is "no
DRAM round-trip," achieved either by fusion or by layout matching.

---

## 2. Fuse activation into the producing matmul {#fuse-activation-matmul}
<!-- route
op_class: eltwise
rank: count
lever_type: single-shot
-->

GELU/SiLU into FF1, ReLU into a projection — always fuse via `fused_activation`. The SFPU
LUT hides in the packer schedule (+~0.8 µs vs bare matmul, vs +6–40 µs as a separate op).
See 05 §2. Same idea for any unary that immediately follows a matmul.

---

## 3. Fuse dtype casts into reshards — always {#fuse-cast-reshard}
<!-- route
op_class: datamove
rank: count,time
lever_type: single-shot
-->

`interleaved_to_sharded`, `sharded_to_interleaved`, and `to_memory_config`-style ops accept
`output_dtype=`. Never run a standalone `typecast` next to a reshard:

```python
# Bad:  y = ttnn.typecast(x, bf16); z = ttnn.interleaved_to_sharded(y, cfg)
# Good: z = ttnn.interleaved_to_sharded(x, cfg, output_dtype=ttnn.bfloat16)
```

This is also the mechanism for the bf8b-norm precision fix (02 §5): the bf16 cast that
protects the reduction is *free* because it rides the reshard you were doing anyway.
BGE-M3: −82 µs on the S→I output path alone.

---

## 4. Fuse adjacent unary ops via `unary_chain` {#fuse-unary-chain}
<!-- route
op_class: eltwise
rank: count
lever_type: single-shot
-->

When two scalar/unary ops are adjacent (e.g. `hardsigmoid(x)` then `x − 0.5`, or
`2·x + 1`), `ttnn.unary_chain` runs them in one kernel:

```python
y = ttnn.unary_chain(x, ops_chain=[
    ttnn.UnaryWithParam(ttnn.UnaryOpType.HARDSIGMOID),
    ttnn.UnaryWithParam(ttnn.UnaryOpType.SUB_UNARY_SFPU, 0.5),
])
```

Swin-L DyHead used this in the scale-attention and task-attention heads (`hardsigmoid+sub`,
`mul+add`), each saving one dispatch per call across 6 blocks. Small but strictly positive,
and exactly the kind of op-count win that matters in a host-bound trace.

---

## 5. Pre-fuse anything that depends only on weights — at load time {#fuse-weight-folds}
<!-- route
op_class: eltwise,embedding
rank: count
lever_type: single-shot
-->

Any per-forward op whose output depends only on weights (not input data) can be done once
at weight-load:

| Pre-fusion | Saved per forward |
|---|---|
| Constant token-type embedding into word-embedding weight (when `type_vocab_size==1`) | 1 lookup + 1 add (~25 µs) |
| Bias / scale folds into a linear's bias | the fold op |
| `1/sqrt(d_k)` attention scale folded into a weight | small |
| bf8b weight conversion (done once at init) | per-call cast |

BGE-M3's token-type fold removed a lookup + a `BinaryNg` add per forward (−111 µs wall at
batch 1, because two dispatch ops also leave the trace pipeline).

---

## 6. Skip whole op-chains with config flags {#fuse-skip-chains}
<!-- route
rank: count
lever_type: single-shot
-->

If the caller can guarantee a precondition, short-circuit the op chain:
- **`attn_mask=None`** when inputs are unpadded → SDPA's compile-time constexpr strips the
  mask read + add from the kernel (−237 µs at batch 1). See 04 §8.
- Skip token-type / position prep when folded or deterministic.

These are opt-in (they change the contract), so gate them behind a flag and document the
precondition.

---

## 7. Cross-block sharded handoff {#fuse-cross-block-handoff}
<!-- route
op_class: datamove
rank: time
memory: sharded
lever_type: single-shot
-->

Thread the last norm's sharded output of block N into block N+1's first norm as its
residual, so the block boundary has no I→S reshard. BGE-M3: −53 µs. Only works when both
sides share the shard config — the ViT "everything block-sharded" pipeline is the extreme
form (no DRAM round-trips anywhere in the encoder).

---

## 7b. Eliminate redundant layout conversions — make producers emit the consumer's layout {#layout-coherence}
<!-- route
op_class: datamove
rank: time,count
lever_type: structural
-->

A `Tilize`/`Untilize` (or any op whose `INPUT_0_LAYOUT != OUTPUT_0_LAYOUT`) does **no math** — it
exists only because a producer emitted a layout the consumer can't accept. The profile surfaces
this as `layout_churn` (count + ms + % of device time) and per-bucket `[layout-churn N× = X ms]`.
When that share is large, it is the single biggest **reducible** device cost — unlike matmul/attention
it can go to ~zero, because the conversion is pure plumbing, not compute.

**Fires when:** the datamove bucket carries a high `layout_churn` share (many `Tilize`/`Untilize`
ops, often ≈ one per matmul — a tell-tale that each op round-trips ROW_MAJOR↔TILE).

Recipe — drive each hot conversion to its source and remove it:
1. For each hot `Tilize`/`Untilize` in `top_ops`, find the **producer** op feeding it.
2. Make the producer emit the layout (and `memory_config`/`dtype`) the consumer needs directly:
   - pass `output_layout=ttnn.TILE_LAYOUT` / the consumer's `memory_config` on the producing op
     (matmul, reshard, `to_memory_config`, eltwise) instead of a standalone convert after it;
   - keep tensors in `TILE_LAYOUT` end-to-end; only un-tilize at a genuine ROW_MAJOR boundary
     (host readback, an op that truly requires row-major), once, not repeatedly.
3. If a convert is unavoidable, ride it on a reshard/cast you already do (see §3) rather than as
   its own op.

Each removed conversion saves its full device time **plus** ~0.6 µs dispatch. The win shows up as
the datamove bucket shrinking while matmul/attention counts stay fixed — exactly the
fusable-only reduction the comparability guard now accepts as a real win (not a partial capture).

---

## 8. When fusion does NOT help

| Situation | Why |
|---|---|
| Device-bound model (large batch) | removing dispatch ops doesn't move wall; the device is the bottleneck |
| Sharding a one-shot / small tensor | per-shard descriptor dispatch overhead > device saving (BGE-M3 embed shard: device −20 µs, wall **+46 µs**) |
| Handoff into an op that reshards internally | the "saved" reshard happens anyway (+106 µs) |
| `addcmul`-style ternary fusion with `(B,1,1,1)` broadcast | Swin-L: standalone PCC 1.0 but trace+2CQ chain PCC collapsed to 0.42 — broadcast precision bug |

**Rule:** op-count reduction is a host-bound lever. In the device-bound regime, spend your
time on fidelity, subblock unlock, and removing typecasts instead (see 05, 04 §5).

---

## 9. The dispatch-cost arithmetic

On Blackhole each captured-trace op ≈ 0.6 µs turnaround. BGE-M3 batch 1 went 368 → 318 ops;
50 × 0.6 = 30 µs of turnaround removed, plus each op's device time, totaling ~130 µs wall.
At batch 32 the same op removals gave ~0 ms (device-bound). **Measure the regime before
investing in op-count work.**

---

## 10. Quick reference

| Pattern | When | Win type |
|---|---|---|
| `Norm(x, residual=r)` | always (or sharded add, no reshard) | strict, correctness-neutral |
| Fused activation in matmul | always | strict |
| Fused dtype cast in reshard | always | strict |
| `unary_chain` for adjacent unaries | always (small) | strict, host-bound |
| Load-time weight folds | when output depends only on weights | strict |
| `attn_mask=None` fast-path | unpadded inputs (opt-in) | host-bound |
| Cross-block sharded handoff | shared shard config | host-bound |
| Don't shard one-shot tensors | — | avoids regression |
| Don't fuse into reshard-internally ops | — | avoids regression |
