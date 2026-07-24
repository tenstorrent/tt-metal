# 06 · Fusion, Residuals, and Op-Count Reduction

Every captured-trace op costs ~0.6 µs of host dispatch turnaround. A 24-layer transformer
has 300+ ops; removing 50 saves ~30 µs of pure turnaround plus the device cost of each op.
This is a real lever in the **host-bound** regime (small batch, decode) and free correctness
everywhere. This file collects the fusion patterns that held across BGE-M3, ViT, and Swin-L.

---

## 1. Fuse `add + Norm` into the norm's residual input

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

## 2. Fuse activation into the producing matmul

GELU/SiLU into FF1, ReLU into a projection — always fuse via `fused_activation`. The SFPU
LUT hides in the packer schedule (+~0.8 µs vs bare matmul, vs +6–40 µs as a separate op).
See 05 §2. Same idea for any unary that immediately follows a matmul.

---

## 3. Fuse dtype casts into reshards — always

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

## 4. Fuse adjacent unary ops via `unary_chain`

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

## 5. Pre-fuse anything that depends only on weights — at load time

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

## 6. Skip whole op-chains with config flags

If the caller can guarantee a precondition, short-circuit the op chain:
- **`attn_mask=None`** when inputs are unpadded → SDPA's compile-time constexpr strips the
  mask read + add from the kernel (−237 µs at batch 1). See 04 §8.
- Skip token-type / position prep when folded or deterministic.

These are opt-in (they change the contract), so gate them behind a flag and document the
precondition.

---

## 7. Cross-block sharded handoff

Thread the last norm's sharded output of block N into block N+1's first norm as its
residual, so the block boundary has no I→S reshard. BGE-M3: −53 µs. Only works when both
sides share the shard config — the ViT "everything block-sharded" pipeline is the extreme
form (no DRAM round-trips anywhere in the encoder).

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
