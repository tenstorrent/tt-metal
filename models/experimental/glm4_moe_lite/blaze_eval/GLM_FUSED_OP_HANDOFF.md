# Handoff: `GLMQANormQBProjection` — deferred q_a RMSNorm fused into the q_b projection

**Status:** source-only, isolated, CPU-verified. Never run on hardware. No shared file touched, no
commit made. The silicon test is **opt-in behind an env var** and skipped by default.

**Why this cluster.** `GLMQKVAProjection` proved a GLM-shaped FusedOp is assemblable, but its own
measurement is the argument for building this one: the two `_a` projections *share* an input rather
than chaining, so there was no DRAM round-trip between them, and fusing them contributed **~0.4 µs
of the 9.43x — about 4%**. The rest was `DRAMStreamingMatmul` beating ttnn's matmul. As
`BLAZE_EVALUATION.md` puts it, the mechanism-2 case "still rests on a *chained* fusion (norm ->
proj …) where an intermediate genuinely round-trips today. That is the next op to build." This is
that op: `q_a → RMSNorm → q_b` is the only place in GLM's decode attention where a norm feeds
straight into a matmul, and the normalized q_a exists solely to be consumed by the next op.

No performance is claimed here. Only the mechanism is stated; you have the only measurement.

---

## 1. Op and source paths

| | |
|---|---|
| op | `GLMQANormQBProjection`, `name = "glm_qa_norm_qb_projection"` |
| implementation | `blaze/ops/glm_qa_norm_qb_projection/op.py` (new) |
| exports | `blaze/ops/glm_qa_norm_qb_projection/__init__.py` (new) |
| CPU tests | `tests/blaze/fused_ops/glm_qa_norm_qb_projection/test_glm_qa_norm_qb_derivation.py` (new) |
| silicon test | `tests/blaze/fused_ops/glm_qa_norm_qb_projection/test_glm_qa_norm_qb_silicon.py` (new, opt-in) |

Self-registering by direct import, exactly like `glm_qkv_a_projection` — **`blaze/ops/__init__.py`
is not modified**. Nothing outside these two new directories was created or edited.

Public surface, all device-free except `emit`:

```python
from blaze.ops.glm_qa_norm_qb_projection import (
    GLMQANormQBProjection,  # .emit(...) / .golden(...) / .golden_from_unfolded(...)
    derive_layout,          # hparams -> every shape/arg, no device
    fold_gamma_into_qb,     # offline W' = gamma . W_q_b, zero-padded in K
)
```

## 2. Exact model calls it replaces

`/home/ttuser/sdawle/tt-metal/models/experimental/glm4_moe_lite/tt/attention_decode.py`

```
327:    q_a = w.q_a_layernorm(q_a, mode="decode")
328:    q = attn_linear(q_a, w.w_q_b, device=device, cfg=cfg, force_no_tp=cfg.attn_dp)
```

Both inside `q_projection(...)`. Supporting definitions:

- `tt/layer_weights.py:641-651` — `q_a_layernorm = RMSNorm(dim=hparams.q_lora_rank, eps=hparams.rms_norm_eps, weight_key="q_a_layernorm")`. A plain RMSNorm with a learned gamma over 768, which is what makes the fold valid.
- `tt/layer_weights.py:714-720` — `w_q_b` from `q_b_proj.weight` (HF stores `[N, K]`; `fold_gamma_into_qb` wants `[K, N]`, so it must already be transposed).

Line 329's `ttnn.deallocate(q_a, ...)` becomes unnecessary. Lines 331+ (`reshape` → `permute` →
nope/rope slice) are **out of scope and unchanged** — this op stops at `q`, the same way
`GLMQKVAProjection` deliberately stops before the Gather.

## 3. The transform

FlashNorm deferred normalization, the same identity `DeferredRMSNormMatmul` and
`qa_projection(defer_norm=True)` already use:

```
(1/RMS(q_a)) * (q_a @ (gamma . W_q_b))  ==  RMSNorm(q_a, gamma) @ W_q_b
```

gamma folds into the weight offline, so the matmul runs on the **un-normalized** q_a and the
`1/RMS` scalar is applied inside the matmul's DST epilogue:

```
q_a ─┬─► SumOfSquares(scalar=1/sqrt(768), epsilon)  ──► 1/RMS  [1 tile, same core, no NOC]
     │                                                    │
     └─► DRAMStreamingMatmul(W', scalar=1/RMS) ◄──────────┘ ──► q [1, 5120]
```

`DRAMStreamingMatmul` already carries a `scalar` CB input for the MoE routing weight
(`out = scalar * (act @ W)` — `common.py:390`, `kernels/op.hpp:403`), and a per-row `1/RMS` is the
same shape of value read from the same lane. **No new kernel, no kernel edit.**

**No cross-core traffic at all.** `DRAMStreamingMatmul` wants its activation *replicated* across
the 8 DRAM-bank workers, so every core already holds the full K and computes the identical
`1/RMS` from its own replica. There is no Gather and no Mcast — deliberately, since the Gather is
what deadlocks `glm_routed_expert` at GLM's dims (F11) and `DeferredRMSNormMatmul` reaches for
both. The price is a redundant 1024-element reduce per core against the 640×1024 matmul it then
runs.

### What disappears

| | today | with this op |
|---|---:|---:|
| dispatches (norm + q_b) | 2 | **1** |
| q_a activation passes | 2 (norm reads, matmul reads the copy) | **1** |
| normalized-q_a materialization | written then read back | **never exists** |
| elementwise normalize pass | 1 full pass over q_a | **0** — folded into the matmul epilogue |

The intermediate that stays in L1 is `q_a` itself: it is read once by the reduce and once by the
matmul out of the *same* L1 shard, and the normalized copy is never formed in L1 *or* DRAM. The
boundary that disappears is the write+read of normalized q_a between `attention_decode.py:327`
and `:328`.

Honest scoping: getting un-normalized q_a into the replicated bank-worker layout is still a
reshard, and that cost is real and pre-existing for any `DRAMStreamingMatmul` adoption
(`BLAZE_EVALUATION.md` §"No blaze op has been substituted"). This op does not remove it — it
amortizes one reshard over norm *and* matmul instead of paying separately for a norm.

## 4. Shapes, layouts, dtypes, grids

`derive_layout(q_lora_rank=768, num_attention_heads=20, qk_head_dim=256, num_banks=8)` returns,
verified on CPU:

```
logical_k=768   padded_k=1024   norm_num_tiles=1   norm_tile_shape=(32,32)
rms_scalar=0.036084391824351615  (== 1/sqrt(768))
n_total=5120    per_core_n=640   per_core_n_tiles=20
act_num_pages=32  subblock_k=8   num_subblocks_k=4  subblock_w=4
```

| tensor | shape | dtype | layout | grid |
|---|---|---|---|---|
| `q_a` (in) | `[1,1,1,1024]`, replicated to `[1,1,8,1024]` | `bfloat16` | `HEIGHT_SHARDED` L1, shard `(1,1024)`, tile `1x32` | 8 DRAM-bank workers |
| `q_b_weights` (in) | `[1,1,1024,5120]` | `bfloat8_b` | `WIDTH_SHARDED` DRAM, shard `(1024,640)`, column-major tile-shuffled | 8 DRAM banks |
| `q_out` (out) | `[1,1,1,5120]` | `bfloat16` | `WIDTH_SHARDED` L1, shard `(1,640)`, tile `1x32` | same 8 workers |
| `1/RMS` (internal) | 1 tile/core | `bfloat16` | scratch CB, 2048 B/core | same 8 workers |

Build the three tensors with the already-validated helpers
`_make_act_tensor` / `_make_weights_tensor` / `_make_output_tensor` from
`tests/blaze/micro_ops/common/test_dram_streaming_matmul.py` — the silicon test does exactly this.

### Why K is padded 768 → 1024

`SumOfSquares` reinterprets the `1x32`-tile activation row into standard compute tiles, and that
needs a width that is a multiple of 512 (HALF) or 1024 (FULL). **768 is a multiple of neither**:
`interpret_tile(768)` returns *one HALF tile*, silently covering 512 of the 768 elements and
producing a wrong RMS. This is the trap in this cluster.

Padding to 1024 lands on exactly one FULL 32×32 tile — the geometry
`test_deferred_rmsnorm_matmul` validates on silicon at its `K=1024` case. The op deliberately
rounds to FULL rather than using `interpret_tile_padded`, which minimizes padding and would answer
HALF×2 (a correct but differently-validated path used by `TpRmsNorm`).

Zeros are inert in both consumers: they add nothing to the sum of squares, and they multiply the
zero-padded rows `768:1024` of `W'`. **The mean must still divide by the logical 768**, so the
reduce scalar is `1/sqrt(768)`. Dividing by 1024 instead scales every q by
`sqrt(1024/768) = 1.155` — a 15.5% error; `test_the_wrong_reduce_width_would_be_caught` pins
exactly that magnitude so the tolerance can never be relaxed past it.

## 5. Validation

### 5a. CPU — safe, no device, run this first

```bash
cd /home/ttuser/sdawle/tt-blaze && source env.sh
python -m pytest tests/blaze/fused_ops/glm_qa_norm_qb_projection/test_glm_qa_norm_qb_derivation.py -q
```

Expect **23 passed in ~0.2 s**. Opens no device (no `device` fixture; marked
`device("cpu")`, `level("infra")`).

### 5b. Silicon — bounded, opt-in

Skipped unless `GLM_QA_NORM_QB_HW=1`. Do this only when the Galaxy is idle and healthy.

```bash
cd /home/ttuser/sdawle/tt-blaze && source env.sh && unset TT_MESH_GRAPH_DESC_PATH
GLM_QA_NORM_QB_HW=1 timeout 300 python -m pytest \
  tests/blaze/fused_ops/glm_qa_norm_qb_projection/test_glm_qa_norm_qb_silicon.py -q -x -s 2>&1 | tail -40
```

`timeout 300` is the whole point of the bound: `o_proj` completes in ~23 s and this is a smaller
matmul, so anything past ~60 s is a hang, not slowness. **Kill it, do not wait.**

### 5c. Post-run control (mandatory if it hung) — F12

A hang degrades the device, and open/close still succeeds on a degraded one. Re-run a known-good
case before believing any result:

```bash
timeout 300 python -m pytest glm47_all_shapes_check.py -q     # expect 6/6, ~12 s
```

If that now hangs, the device is degraded, not the op: `tt-metal/python_env/bin/tt-smi -r`, then
re-run the control, then retry.

### Required PCC checks

The silicon test reports **two** PCCs deliberately, so a failure is diagnosable without a second
run:

| check | against | gate | meaning if it fails |
|---|---|---:|---|
| `pcc_vs_device` | `(1/RMS) * (q_a @ W'_bf8)` — same quantized weight | ≥ 0.99 | the **fusion** is wrong (CB handover, scalar lane, reduce width). Not precision. |
| `pcc_vs_model` | `RMSNorm(q_a, gamma) @ W_q_b` — the model call | ≥ 0.99 | if `pcc_vs_device` passed, this is the folded-weight bf8 precision shift, not the op. |

Both must pass to adopt. Treat `pcc_vs_model ≥ 0.99` as the adoption bar (the same bar
`test_deferred_rmsnorm_matmul` uses, and for the same folded-weight reason); the 0.9999 the pure
matmuls hit is not reachable once gamma folds into a bf8 weight.

## 6. Known risks

1. **`fp32_dest_acc_en=True` must be set on the `FusedProgram`, not the emit — highest-impact
   risk.** The reduce sums 1024 squares; RMSNorm at GLM's dims measured **PCC 0.9865** accumulating
   in bf16 against **0.9999** in fp32. `emit` takes an `fp32_dest_acc_en` too, but that one only
   reaches the matmul — the norm's `DST_ACCUM_MODE` comes from the program's
   `ComputeConfigDescriptor`, which is authoritative (`blaze/ops/rmsnorm/op.py:143`). Set it in
   **both** places. This is the documented silent-precision-loss trap in this exact area.

2. **Two CB views over one activation tensor — the main un-run interaction.** `SumOfSquares` sees
   q_a as one FULL `32x32` tile; the matmul sees it as 32 `1x32` pages. Both come from
   `f.cb_from_tensor` on the same tensor. Statically verified safe: `_try_reuse_tensor_cb` reuses a
   cb_id only when `(data_format, tile)` **match** *and* the grids are **disjoint** — here the tiles
   differ *and* the grids are identical, so each view gets its own cb_id over the same L1 address
   (read-only aliasing). This mirrors `DeferredRMSNormMatmul`, which hands the same tensor to
   `SumOfSquares` and `Mcast` and passes on silicon. **If it does misbehave**, the fallback is to
   emit the matmul's default-geometry CB first, or pass `SumOfSquares` an explicit
   `tile=`/`page_size=` (it accepts both) instead of letting it re-derive.

3. **bf16 scalar quantization — expected, not a defect.** The epilogue reads the scalar as bf16
   (`kernels/op.hpp:322` keeps 16 of 32 bits), so `1/RMS` carries ~8 mantissa bits. It multiplies
   the whole output row *uniformly*, and PCC is invariant under uniform scaling, so it should cost
   the gate almost nothing — but the op is not bit-exact against an fp32 norm. Do not chase this if
   PCC passes.

4. **Untested at m=32 (F1).** Everything here is `m=1` decode. `DRAMStreamingMatmul` is
   numerically broken at m=32 (PCC 0.0074), so **keep m=1**; this op inherits that limit and does
   nothing to fix it.

5. **`subblock_k` is left to the default** (8, giving 4 K-subblocks). `ab_glm_qkv_a_bench.py` uses
   `K/32/2` instead (16 here, 2 subblocks). Both divide `Kt=32`; if perf disappoints, this is the
   first knob, and it is a plain `emit` kwarg.

6. **Weight prep is a load-time transform, and the fold must precede quantization.** Fold gamma in
   float, zero-pad K, *then* cast to bf8 and apply the DRAM width-shard + column-major tile
   shuffle. Folding after quantization loses the point of doing it in float.
   `fold_gamma_into_qb` returns float and leaves the cast to you, on purpose.

7. **20 heads / F3 is not touched.** This op stops at `q [1,5120]` and never reasons about heads,
   so `layout_plan`'s `n_heads_per_device % 8 == 0` does not apply. Splitting `q` into
   NoPE/RoPE for `scattered_q_heads` remains yours and is unchanged.

## 7. What you need to integrate

Nothing of mine is on your path — these are additive. In dependency order:

1. **Weight prep** (yours to place): for each layer, `W'_q_b = fold_gamma_into_qb(w_q_b_KxN,
   q_a_layernorm_gamma, padded_k=1024)`, then bf8 + DRAM width-shard + column-major tile shuffle
   (`_make_weights_tensor`'s `_shuffle_tensor_tiles`). Adds 256 zero rows/layer — 256×5120 bf8 ≈
   **1.3 MB/layer** of DRAM, the cost of the padding.
2. **`q_a` must be produced zero-padded to 1024 and replicated on the 8 bank workers.** Cleanest
   route, if you want it: zero-pad `W_q_a`'s N from 768 → 1024 inside `GLMQKVAProjection`'s weight
   prep, and the pad appears in `q_a` for free with no runtime op. That file is yours; I did not
   touch it. Otherwise pad at the reshard.
3. **Call site** — replace `attention_decode.py:327-328` with one `GLMQANormQBProjection.emit`,
   passing `logical_k=hparams.q_lora_rank` (**768, not the padded 1024** — the signature is shaped
   to make that the obvious choice) and `epsilon=hparams.rms_norm_eps`.
4. **Program config** — `fp32_dest_acc_en=True` on the `FusedProgram` (risk 1).
5. **Consume `q_out`** `[1,1,1,5120]` width-sharded 640/core, into your existing nope/rope split.

Not required, and I did not change them: `blaze/ops/__init__.py`, any registry, any op you own.

## 8. If you want the A/B

`ab_glm_qkv_a_bench.py` is the closest template. The ttnn side must be **`ttnn` RMSNorm followed by
`dram_sharded_linear`** — two dispatches — not a bare matmul, or the comparison silently drops the
norm and flatters this op. Mirror `w.q_a_layernorm` + `attn_linear(q_a, w.w_q_b)` exactly, and
report the norm and the matmul separately alongside the fused number, the way the `~0.4 µs`
decomposition was reported for `GLMQKVAProjection` — that decomposition is what makes the result
interpretable, and here it is the actual question: **whether removing a real round-trip beats the
~12% ceiling.**
