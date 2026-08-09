# AttnRes — API specification

**Contract, fixed at Phase 3 (2026-07-30). Do not rewrite this file.** Deviations
discovered later are recorded as Learnings and Decisions in `bringup_log.md`; the spec
stays as-written so the log shows where reality diverged from the contract.

---

## 1. Tensor contract

`d = 7168`, `Bk = 12`, `eps = 1e-5`, `N = B*T`, `S ∈ [0, 8]`.

### torch (`torch_functional/`)

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| `prefix_sum` | `[N, d]` | bf16 or fp32 | the single live residual stream |
| `block_residual` | `[N, S, d]` | matches `prefix_sum` | sealed snapshots; `S=0` is legal and is a real `[N, 0, d]` tensor |
| `q` | `[d]` | fp32 | folded pseudo-query (§3) |
| return | `[N, d]` | `prefix_sum.dtype` | |

Internal arithmetic is fp32 regardless of input dtype; the cast back to input dtype
happens only on return. This matches the reference.

### ttnn (`tt/`)

| Tensor | Shape | layout | memory |
|---|---|---|---|
| `prefix_sum` | `[1, 1, N, d]` | TILE | DRAM interleaved (Phase 5); sharded is a Phase-8/9 knob |
| `block_residual` | `[1, S, N, d]` | TILE | DRAM interleaved |
| `q` | `[1, 1, 1, d]` | TILE | DRAM |
| return | `[1, 1, N, d]` | TILE | matches `prefix_sum` |

**Candidates live on dim 1** (D10). `S+1 ≤ 9`; a last-dim candidate axis tile-pads
9 → 32 and admits `exp(0) = 1` from padding zeros into a last-dim softmax. Reductions
over `d` stay last-dim; the softmax over candidates is hand-rolled on dim 1.

Anything that crosses a collective stays TILE:
`composite_common::use_composite_all_gather` returns true *unconditionally* for
ROW_MAJOR, which routes to `all_broadcast` and a documented unrecoverable erisc stall
on a partial cluster-axis line.

---

## 2. Invariants

1. **Keys are normalized; values are not.** `rsqrt` scales the score only. The mixture
   is over raw `v`. This is the single most likely porting bug.
2. **RMS is a per-`(token, candidate)` scalar**, so the normalized tensor `k` never
   needs to exist: `score = rsqrt(mean(v²) + eps) · ⟨q, v⟩`.
3. **`block_residual` is write-once.** A sealed snapshot is never mutated, so its
   `rms_inv` is loop-invariant and may be folded at seal time.
4. **Writes into `prefix_sum` are plain `+=` with weight 1.** AttnRes rewrites the read
   only; it does not touch the write path.
5. **`α` is row-stochastic** over `S+1` candidates. `Σ_i α_i = 1` per token, per read.
6. **`S=0` is identity.** One candidate ⇒ `α = [1]` ⇒ `out = prefix_sum`. The reference
   short-circuits it; so do we, and the numeric ladder covers `S=0` anyway.
7. **`prefix_sum` is `None` between the seal and the attention output.** Steps 2–4 of
   the layer pipeline have no live stream. No read site falls in that window.
8. **`block_residual` does not persist across prefill chunks.** It is per-forward-pass
   state, unlike a KV cache.

---

## 3. Weight folding

```
q_l = res_norm.weight * res_proj.weight.squeeze(0)      # [d] * [d] -> [d], fp32
```

Load-time only. Two `[d]` bf16 tensors collapse to one fp32 `[d]`; the RMSNorm gain
multiply disappears from the runtime path.

Folding reassociates the score: the reference computes `Σ_j (v_j · rms_inv · w_j)`,
the folded form computes `rms_inv · Σ_j (v_j · q_j)`. Algebraically identical,
**not bit-identical in fp32**. Hence the three-rung gate (§6, D9).

### HF weight-name map

| HF key | folded into |
|---|---|
| `language_model.model.layers.{l}.self_attention_res_norm.weight` `[d]` | `q_pre[l]` `[d]` |
| `language_model.model.layers.{l}.self_attention_res_proj.weight` `[1,d]` | ″ |
| `language_model.model.layers.{l}.mlp_res_norm.weight` `[d]` | `q_post[l]` `[d]` |
| `language_model.model.layers.{l}.mlp_res_proj.weight` `[1,d]` | ″ |
| `language_model.model.output_attn_res_norm.weight` `[d]` | `q_out` `[d]` |
| `language_model.model.output_attn_res_proj.weight` `[1,d]` | ″ |

374 source tensors, all bf16 → 187 folded queries. `q_pre[0]` is loaded but never used
(the `l=0` pre-attention read is skipped at `S=0`).

---

## 4. torch API

```python
# reference/attn_res_reference.py — unfolded fp64 ground truth. Gates only, never shipped.
# Imported as `attn_res_reference as ref`; takes the two weights separately.
ref.read(prefix_sum, block_residual, norm_weight, proj_weight, eps, dtype=float64) -> Tensor
ref.scores(prefix_sum, block_residual, norm_weight, proj_weight, eps, dtype=float64) -> Tensor
ref.Stream(hidden_states, block_size, eps, dtype=float64)          # read() takes a weight pair
ref.layer(stream, layer_idx, q_pre, q_post, attn_fn, mlp_fn)
ref.stack(hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns, block_size, eps)

# reference/hf_attn_res.py — upstream's `_apply_attn_res` verbatim, plus tensor shims.
# The external anchor. fp32-locked (upstream spells `.float()`), so it gates the
# equation and never the precision. Licensed under reference/LICENSE-Kimi-K3.
hf_attn_res(prefix_sum, block_residual, norm_weight, proj_weight, eps) -> Tensor

# torch_functional/attn_res.py
fold_query(norm_weight, proj_weight) -> Tensor                    # [d], fp32 or wider

attn_res_scores(v, q, eps) -> Tensor                              # [N, C]

attn_res(prefix_sum, block_residual, q, eps) -> Tensor
    # folded, single-pass; the production reference

attn_res_inter_block(block_residual, q_batch, eps) -> (partials, m, Z)
    # once per 12-layer block, for all read sites in it
    # block_residual [N,S,d], q_batch [R,d] -> partials [R,N,d], m [R,N], Z [R,N]

attn_res_merge(partial, m, Z, prefix_sum, q, eps) -> Tensor
    # per read site; online-softmax rescale of a 2-candidate mixture

class AttnResStream:
    # the block_residual lifecycle, mirroring _forward_attn_residual exactly
    __init__(hidden_states, block_size=12, eps=1e-5)
    read(q) -> Tensor            # AttnRes over (prefix_sum, block_residual)
    seal()                       # append prefix_sum, set it to None
    accumulate(module_out)       # prefix_sum += out, or = out after a seal
    S -> int                     # sealed snapshot count
```

Both forms of the op are permanent. The naive form is the **only** independent check on
the online-softmax merge algebra, so unlike KDA (which deleted its composed op at
`4384da3d4db`) it is not disposable.

## 5. ttnn API

Mirrors the torch API parameter-for-parameter, with `mesh_device` and the standard
distribution arguments prepended in the module constructor rather than the call:

```python
class TtAttnRes(LightweightModule):
    __init__(mesh_device, hidden_size=7168, eps=1e-5, torch_queries=None,
             cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear,
             input_memcfg=None, stats_memcfg=None,
             weight_cache_path=None, cache_name_prefix=None)

    forward(prefix_sum, block_residual, q) -> ttnn.Tensor          # naive form
    inter_block(block_residual, q_batch) -> (partials, m, Z)
    merge(partial, m, Z, prefix_sum, q) -> ttnn.Tensor
```

The ctor signature deliberately tracks
`deepseek_v3_d_p/tt/tt_distributed_rms_norm.py:134-148` — same distribution shape
(reduce over `d`, communicate statistics only), so the same knobs must be reachable.

`topology` is accepted as a scalar for single-axis use, but Phase 8 replaces it with a
**per-axis tuple**: Galaxy production prefill is `dims:[8,4] dim_types:[LINE, RING]`,
and a scalar `Ring` deadlocks a TP all-gather on a column wrap link with no physical
fabric edge.

---

## 6. Numeric validation plan

Each rung is measured against the rung below (D9, amended by D11):

| Rung | Compares | Gate |
|---|---|---|
| 0 | `ref` vs closed forms and structural properties | see `tests/test_attn_res_reference.py` |
| 0b | `ref.read` vs upstream `hf_attn_res`, fp32 | rel err ≤ 1e-5 and PCC ≥ 1 − 1e-9 |
| 1 | `attn_res` (folded) vs `ref.read`, both fp64 | rel err ≤ 1e-13 |
| 1a | `attn_res` (folded) vs `ref.read`, fp32 | rel err ≤ 1e-5 and PCC ≥ 1 − 1e-9 |
| 1b | folded and unfolded fp32 vs `ref.read` fp64 | `err(folded) ≤ max(4·err(unfolded), 1e-5)` |
| 2 | `inter_block` + `merge` vs rung 1 | rel err ≤ 1e-5 |
| 3 | `AttnResStream` lifecycle | seal schedule, snapshot growth, read count |
| 4 | `tt/` composite vs rung 1 | PCC ≥ 0.9999 |
| 5 | 93-layer depth harness | `PCC(TT, fp32) ≥ PCC(bf16, fp32) − ε` |

Rung 0 is the exception to "measured against the rung below" — it is the root, so
nothing above it can gate it without making the ladder circular. What pins it are
three closed forms where the answer is known outright (a zero query gives the plain
mean of the candidates, a saturated query selects exactly one, and constant-along-`d`
candidates give `(a/√(a²+eps))·Σⱼ gainⱼ·projⱼ`) plus scale invariance of the scores,
output sensitivity to a candidate's scale, and a convex-hull bracket. The third
closed form is load-bearing: a `sum`-for-`mean` slip in the RMS is a pure softmax
temperature error, and dropping the `res_norm` gain is invisible to the rest — both
would otherwise be caught only by agreeing with the implementation under test.

Rung 0b is the only rung that reaches outside the module. Rungs 0 and 1–5 all compare
things we wrote, so every one of them is consistent with the whole ladder solving the
wrong equation; 0b is not. It runs at fp32 because the vendored function widens with
`.float()` and so computes in fp32 whatever it is handed — which is also why it anchors
the equation in one direction only and `ref` stays the root for precision.

Rung 1 runs at fp64 as well as fp32. At fp32 alone an algebra error near the
rounding floor is indistinguishable from rounding; at fp64 the two forms must agree
to ~1e-14, which is what turns rung 1 from a smoke test into a proof of the
reassociation.

No rung above 1 is bit-exact. Folding reassociates the score and the online-softmax
split reassociates the mixture; both change fp32 rounding. `1e-5` is the fp32
dot-product noise floor `√d · ε_fp32` at `d = 7168`, and a real algebra error clears
it by orders of magnitude — measured rung-1a error is 1.5e-7 … 4.0e-7.

Both the error metric and the ground truth must be computed outside fp32:
`torch.corrcoef` in fp32 caps near 0.99999988 on 458 k elements, and a reference that
widened with `.float()` would compute in fp32 whatever dtype it was handed and could
not serve as its own high-precision reference — which is why `ref` promotes rather
than casts.

Sweep `S ∈ {0, 1, 4, 8}` and `d ∈ {256, 7168}` on every rung. `S=0` and `S=8` are the
boundary cases (identity, and the widest mixture); `d=256` is the toy dim for fast
iteration, `d=7168` is production. Correctness at toy dims, **perf at production dims**
— a KDA hang existed only because its layer test used toy `K=64` while its op test used
`K=128`.
