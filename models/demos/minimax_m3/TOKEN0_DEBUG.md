# MiniMax-M3 prefill "token-0" bug — debug log

**Status:** root not yet fixed. Localized to the MoE `combine`/`dispatch` padding contract, but
reframed (2026-06-30) as likely a **systematic, residual-propagated** error, not an isolated L35 event.

**Symptom:** full-60-layer bf8 prefill (SP=8 × TP=4 × EP=32, MSA off / `FORCE_DENSE=1`) emits token id 0
(`'\x00'`) for every step. Mechanism of the *visible* failure: the residual stream is healthy through
L34, **explodes at L35** (`max|x| ≈ 1.27e38`), tips to `inf → nan` a few layers later, the final RMSNorm
returns zeros → all-zero logits → `argmax = 0`.

---

## The MoE pipeline (per layer, layers 3–59)

`tt_minimax_moe.py:246-293`:

```
1. gate              -> scores, indices (top-4 of 128 experts per token)
2. routing_setup     -> per-expert token counts / region offsets
3. dispatch_module   -> dispatched_buffer  [1,1,4096,6144]   capacity buffer (4096 = padded slots)
4. to_layout/tile    -> dispatched_buffer_tiled [4096,6144] bf8
5. routed_expert     -> expert_outputs    [4096,6144]        FFN per expert (composite OR fused)
6. combine_module    -> combined_output   [1,1,256,4,6144]   gather each token's 4 expert results
7. reduce_module     -> routed_output     [1,1,256,1536]     weighted-sum over top-4 + reduce-scatter (TP)
8. (mlp.py)          -> all-gather across TP -> add to residual
```

---

## What the data definitively shows

### Per-layer residual (whole-tensor `max|x|`), full-60 run (`/tmp/sanitize_full60.log`)
```
L0=147  L1=228  ... steady growth ...  L29=11712  L30=11840  L31=11968
L32=12160  L33=12224  L34=12288   <- still healthy, growing ~128/layer
L35=1.27e38   <- EXPLOSION (whole tensor); LAST-token row still 800 here
L36+ = 1.27e38 (whole), LAST-row now 7.4e30 -> propagates to inf/nan -> token 0
```

### MoE stages at L35 vs healthy L34 (`MOE_STAGE_DBG`, all-32-device scan)
```
            dispatch            expert (FFN out)        combine            -> residual
L34   finite_all=FALSE      max=3.376e38 (finite)    max=35.0   OK        ->  764    OK
L35   finite_all=FALSE      max=inf                  max=1.22e38 BAD      ->  1.27e38 BOOM
```

### Per-MoE-layer expert vs combine (the key proof)
```
EVERY MoE layer:   expert (FFN out) = 3.376e38 / inf   (uninitialized padding, ALWAYS present)
combine L3..L34:   O(10-100)   OK   (combine does NOT read the padding)
combine L35:       1.223e38    BAD  (combine reads the padding -- ONLY here, in this run)
combine L36..:     O(10-100)   OK
```

**Conclusions locked down by this data:**
1. The **FFN/expert is NOT the source.** Its output is huge at *every* layer; if it were the bug, the
   combine would track it. The combine is fine everywhere except L35. (This invalidated the entire
   "composite transient / bf8 re-pack" theory and the fused-kernel rewrite — see below.)
2. The garbage originates as **uninitialized dispatch capacity padding** (step 3), present at all layers,
   harmless until something **reads** it.
3. `combine_module` (step 6) is the op that turns good data into wrong output — by reading padding.
4. In *this run* it only misbehaves at L35 → triggered by **L35's specific routing distribution**
   (the `metadata` / `token_counts` / `region_offsets` it produces), not a precision issue.

---

## Hypotheses: assumed → checked → result

| # | Assumed | How checked | Result |
|---|---------|-------------|--------|
| 1 | MoE/bf4/bf8/dispatch precision | full runs at various dtypes | ❌ ruled out (earlier session) |
| 2 | Layer-35 weights corrupt/stale-cache | DEBUG_LAYERS attn vs mlp probe | ❌ not weights — it's the MLP/MoE path |
| 3 | Composite expert: `ttnn.linear(down)` transient + bf16→bf8 in-place insert reads stale DRAM | single-chip poison test (`test_poison_composite_vs_fused.py`); built + ran **fused** swiglu_oai kernel replacement, full-60 run | ❌ **WRONG.** Fused kernel (no transient/re-pack) explodes *identically* (`expert=inf → combine=1.3e38`). The FFN was never the cause. (Cost: a full night + the whole fused-kernel branch merge.) |
| 4 | Combine **output** buffer uninitialized | `init_zeros=True` on TtCombineModule | ❌ no effect (zeros the result, not the padding it reads) |
| 5 | All-gather (async) corrupts | swapped raw AG → all_gather_async; swap-correctness test | ❌ AG is just the messenger; garbage exists pre-AG (PCC 1.0 vs raw) |
| 6 | Sanitize the dispatch **input** padding `where(isfinite,x,0)` | applied to `dispatched_buffer_tiled` (bf8), full-60 run | ❌ no effect: `[MOE expert]` still inf. Wrong buffer AND broken op (see below) |
| 7 | `ttnn.where(cond, x, 0)` works on bf8 | tiny op test | ❌ **bf8 `where` is BROKEN** — zeros the *true* branch too. Returns all-0. (bf16 `where` is correct.) |
| 8 | Padding garbage is always inf/nan | single-layer pre-flight | ❌ can be **finite-huge** (`3.376e38`) → `isfinite` misses it → must clamp by **magnitude** (`\|x\|≥1e30`) |
| 9 | Single layer + real L35 weights + real dumped input reproduces it | `test_moe_real_l35_repro.py` (+ DBG_POISON) | ❌ stays finite (21.6). **Footprint-dependent**: only the full-60 DRAM footprint provides the resident-garbage condition |
| 10 | NLAYERS=37 reproduces it | full run, 37 layers | ❌ clean. Confirms footprint dependence |

### Things confirmed TRUE
- Bug is **footprint-dependent** (only full 60-layer load; single-layer & 37-layer are clean).
- Garbage = **uninitialized dispatch capacity padding** = resident DRAM (other layers' weight bytes).
- The expert FFN output is garbage-padded at **every** MoE layer; only the **combine read** matters.
- A coherent token *was* produced once (fused run, contained finite explosion) — but it's a **coin flip**
  (finite garbage → RMSNorm contains it → coherent; inf garbage → nan → token 0). **Not reliable.**

### Verified op facts (reusable)
- `ttnn.where(cond, x_bf8, scalar)` → returns 0 for the true branch. **Do not use `where` on bf8.**
- `ttnn.where`, `ttnn.abs`, `ttnn.lt`, `ttnn.isfinite` all correct on **bf16**.
- Working sanitize pattern: `typecast bf8→bf16 → where(lt(abs(x),1e30), x, 0) → typecast bf16→bf8`
  (preserves valid O(1e2) values, zeros inf/nan/finite-huge).

---

## Reframe (2026-06-30) — DON'T band-aid; it's systematic

`max|x|` only catches the **overflow**. It is blind to a small per-layer error that degrades PCC. Known
separately: **KV-cache PCC degrades layer by layer** — i.e. there is a systematic error, and the
**residual is the propagation channel** (K/V are projected from the residual, so a corrupted residual
compounds across layers and shows up as KV-cache PCC decay).

Working theory: possibly **two faces of one disease** —
1. a **systematic** small error injected each layer (→ the layer-by-layer KV-cache PCC decay), and
2. the **L35 overflow** (combine reads inf padding), the one layer where the garbage is large enough to
   blow up rather than merely degrade.

The sanitize (zero expert-output padding) fixes the *overflow* but masks whether the combine/dispatch
contract is leaking *small* errors at every layer. **Open question:** is the combine reading padding only
at L35, or a little at every layer (which `max|x|` can't see, but PCC would)?

---

## Methodological correction & next steps

**Measure PCC per layer, not `max|x|`.** Track, against the torch reference, per layer:
- `attn_out` PCC, `mlp_out` (MoE) PCC, and `residual` PCC,

to find (a) *where* the error enters (attention vs MoE), (b) whether it's **every layer** or only L35,
(c) how it **compounds** through the residual.

**Need to anchor first (open):**
1. Which run/test showed the **KV-cache PCC layer-by-layer degradation**? MSA on or dense? Start from
   those numbers.
2. Do we have a **per-layer torch reference** to diff the residual against now, or stand it up first?

**Fix levels (decide after the systematic measurement):**
- *Root, dispatch:* zero the capacity buffer's padding in the dispatch op → combine reads 0 everywhere.
- *Root, combine:* understand why L35's routing makes the combine index unwritten slots (metadata/offset).
- *Band-aid (rejected as the answer):* magnitude-clamp the expert output before combine.

---

## Key files / artifacts
- Fix-in-progress wiring: `models/demos/minimax_m3/tt/experts_throughput/tt_minimax_moe.py`
  (`_moe_stage_dbg` gated by `MOE_STAGE_DBG=1`; sanitize block ~L262-281)
- Probes: `model.py` (`_dbg_layer_stats` whole+lastrow, `_dbg_logits`), `layer.py` (`_dbg_sub`,
  `_dbg_dump` → `/tmp/m3_dump`), `mlp.py` (post-AG all-device probe)
- Single-layer repro: `models/demos/minimax_m3/tests/unit/test_moe_real_l35_repro.py`
- Op tests: `tests/.../deepseek_prefill/test_poison_composite_vs_fused.py`, `test_swigluoai_m3dims.py`
- Logs: `/tmp/sanitize_full60.log`, `/tmp/fused_fix_full60.log` (both have full per-layer + MoE-stage probes)
- Env to reproduce probes: `DEBUG_LAYERS=1` (per-layer residual), `MOE_STAGE_DBG=1` (32-device MoE-stage
  scan — 2× the run time), `DBG_SUB_RANGE=33-37 DBG_SUB_POS=62,4129` (sub-layer at the L35 hotspot)
