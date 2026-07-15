# T2U Optimization Log (`perf_t2u.md`)

Living record of Text-to-Unit (`tt_text_to_unit.py`) device-perf optimizations for **P150 (tp=1)** and
**BH-QB (tp=4)**. Metric: total device kernel time from the Tracy ops CSV (`DEVICE KERNEL DURATION`),
summed across rows. BH-QB is a 4-device *merged* report (per-device ≈ ¼); ratios are apples-to-apples.
Validation: `test_text_to_unit.py::...max_seq_pcc` (encoder seq 4096, decoder unit seq ≈16640), logits
PCC gate 0.99.

Run: `MESH_DEVICE={P150|BH-QB} python -m tracy -p --op-support-count 100000 -r -v -m pytest <test>`

## Current baseline (all landed wins in place)

| target | total device time | logits PCC |
|---|---:|---:|
| P150 (1-dev) | **377.2 ms** (was 524.2) | 0.9972 |
| BH-QB (4-dev sum) | **856.4 ms** | 0.9972 |

_Updated after #11 (P150 tp=1 linear reroute off the m≤32 DRAM-sharded matmul onto the 512-row mcast/tuned
path): P150 524.2→377.2 ms (−28.0%), op count ~47250→1830. BH-QB unchanged (tp>1 already uses the mcast
path; the global 256→512 chunk bump was tried but reverted — it clashes with the co-resident decode trace,
S2ST "static CB clash" in demo_perf_sweep)._

_Updated after the attention-mask trio (#8 mask build, #9 no-op encoder mask, #10 fold scale into Q)._
_Cumulative from the pre-#8 baseline: P150 −9.9%, BH-QB −20.1%; PCC unchanged (both are bit-exact / no-op)._

### Where the time goes (current, post-#10)

| op | P150 ms (%) | BH-QB ms (%) | note |
|---|---:|---:|---|
| SDPA | 207.5 (40%) | 241.3 (28%) | attention (fused); decoder seq≈16640 dominates |
| Matmul | 150.0 (29%) | 67.3 (8%) | FFN/linear — **not TP-split on P150**; 9377 tiny 256-row chunks |
| Conv2d | 45.8 (9%) | 183.1 (21%) | decoder conv, DRAM width-sliced (#7) |
| BinaryNg | 19.8 (4%) | 75.7 (9%) | residual/mask adds @ seq≈16640 (mask×8 gone via #10) |
| Concat | 19.6 (4%) | 34.5 (4%) | chunk stitching (dense-mask concats gone via #8) |
| AllGather | 0 | 88.0 (10%) | TP comm — **BH-QB only** |
| ReshapeView | 6.8 | 27.1 | incl. a `[1,S]→[S,1]` RM transpose in the unit-upsample gather |
| Slice/PaddedSlice/S2I/Copy | ~40 | ~63 | chunk + conv glue |

Key structural facts:
- **P150 is matmul- & SDPA-heavy** (Matmul 150 ms, no TP split; SDPA 207 ms); **BH-QB is conv/SDPA/comm-heavy**
  (TP splits the matmuls 4×, so Conv2d + SDPA + AllGather dominate).
- **Attention (SDPA) is now the single biggest op on both** and is decoder-dominated (seq≈16640): ~232 ms of
  the BH-QB SDPA is the 6 decoder layers, ~24 ms the 6 encoder layers.
- The 9377 P150 matmuls / 4916 BH-QB matmuls + their Slice/S2I/Concat are all driven by the **256-row linear
  chunking** over the 16640 unit seq (see backlog #1).

## Optimizations tried

| # | change | target | P150 | BH-QB | PCC | status |
|---|---|---|---|---|---|---|
| 1 | `_hard_upsample_gather` (one-hot matmul → gather) | expand glue | −12 ms, −195 ops | (same, small) | 0.9728→0.9749 | ✅ landed |
| 2 | SDPA flash-chunk cap 32→256 | SDPA | 562→215 ms SDPA (2.6×), total 975→640 | applies via fused | →0.995 (fixed pre-existing fail) | ✅ landed |
| 3 | Fused SDPA for tp>1 (was DRAM matmul+softmax) | attention | n/a (already fused) | **2796→1259 ms (2.22×)** | 0.9972 | ✅ landed |
| 4 | SDPA output → L1 | activation placement | 0.0% | 0.0% | 0.9972 | ❌ no-op, reverted |
| 5 | Encoder residual+LN+attn → L1-resident | activation placement | — | −0.2% | 0.9972 | ❌ no-op, reverted |
| 6 | FFN/linear fidelity HiFi4→HiFi2 | matmul math | +0.1% | — | 0.9964→0.9962 | ❌ no-op, reverted |
| 7 | conv1d **DRAM width-slice** (`Conv2dDRAMSliceWidth`) vs manual chunk loop | decoder conv glue | 620.6→582.1 (**−6.2%**) | 1259.3→1072.5 (**−14.8%**) | 0.9964 / 0.9972 | ✅ **landed** |
| 8 | decoder additive-mask build: `repeat_interleave`+`where` → **broadcast outer-product** | Concat / Ternary | 582.1→541.1 (with #9) | 1072.5→944.7 (**−11.9%**) | 0.9964 / 0.9972 (bit-exact) | ✅ **landed** |
| 9 | drop **all-zero (no-op) encoder mask** → `attn_mask=None` | encoder SDPA mask stream | (with #8) | 944.7→926.2 (−2.0%) | unchanged (no-op) | ✅ **landed** |
| 10 | **fold SDPA scale into Q**, pass `scale=1.0` (skip per-layer mask×1/scale) | decoder SDPA mask rescale | 541.1→524.2 | 926.2→857.2 (−7.4%) | unchanged (bit-exact, pow-2 scale) | ✅ **landed** |
| 11 | **P150 tp=1 linear reroute**: DRAM-sharded (m≤32) → 512-row mcast/tuned chunked (`SEAMLESS_T2U_TP1_MCAST_LINEAR`, default on) | Matmul + S2I/Copy/I2S glue | 524.2→**377.2 (−28.0%)** | n/a (tp>1 unchanged) | 0.9972 (unchanged) | ✅ **landed** |

_(P150 rows: #8 and #9 were measured together — 582.1→541.1 — not split; BH-QB rows are per-change.)_

**#8 detail** (`_expand_4d_padding_additive_b1`): HF `_expand_mask` built the dense `[1,1,S,S]` decoder
additive mask with `ttnn.repeat_interleave(row, S, dim=2)` — implemented internally as *hundreds* of
concats (S=16640 → a 32→1504→16640 concat hierarchy, ~72 ms merged) — plus a full-size `ttnn.where`
(20 ms Ternary) and a `multiply(-1)`. But every query row of a padding mask is identical, so build one
`[1,width]` additive row (`where(m>0.5, 0, FLOOR)`) and broadcast it across `S` with a single
outer-product `ttnn.multiply(ones_col[1,1,S,1], row[1,1,1,width])` — one `[S,S]` write, no concat, no
full-size where. **Concat 2388→256 ops, −73 ms; Ternary 20 ms → 0.** Validated `max_abs_err=0` in
isolation vs. the old build at S=256 and S=16640 (bit-exact); isolated wall 531 ms → 3.7 ms.

**#9 detail** (`_additive_mask_is_noop`): a batch-1 full-length sequence has an all-ones padding mask →
all-zero additive mask that contributes nothing. Detect it once (a `max(abs(mask))` reduce + tiny host
readback) and pass `attn_mask=None`, so the 6 encoder SDPA calls skip streaming a dense `[4096,4096]`
zero mask (and its per-call rescale, see #10). **Eager path only** — the readback is illegal during
`begin_trace_capture`, so it is gated on `not trace_no_profiler` (trace/demo keeps the mask as passed).

**#10 detail** (`_t2u_scaled_dot_product_attention`): ttnn's `scaled_dot_product_attention`
(`ttnn/cpp/.../sdpa/sdpa.cpp`) pre-multiplies **any provided mask by `1/scale` on every call** (the
flash kernel folds `scale` into the softmax exponent as `exp((QK+mask−max)*scale)`, so the mask must be
pre-divided to stay unscaled). With the shared, cached decoder mask this re-materialized the dense
`[16640,16640]` mask once per layer — 6× `[16640,16640]×8` = ~71 ms merged, invisible to a Python
`ttnn.multiply` patch (it is a C++ `ttnn::multiply` inside the op). The wrapper skips it when
`effective_scale == 1.0`, so we fold the scale into Q instead: `q *= scale` then call SDPA with
`scale=1.0`. For head_dim=64, `scale = 1/8 = 0.125` is a **power of two**, so `q*scale` only decrements
the bf16 exponent — bit-exact, no mantissa rounding. **BinaryNg 145.8→75.7 ms (BH-QB), 36.8→19.8 (P150).**

**#7 detail** (ported from HiFi-GAN commit `3cea14f4904`): `_conv1d_same` used a manual Python chunk
loop (256-row windows → per-chunk slice-in/conv/slice-out/concat) for wide decoder convs, generating
3024 convs of `[1,1,262,1024]` + surrounding glue. Replaced with one `ttnn.conv1d` call using
`slice_config=Conv2dSliceConfig(Conv2dDRAMSliceWidth, num_slices=0)` — device slices the timeline in
DRAM, manages halo internally. Op count collapsed (BH-QB 48480→20856), `UntilizeWithUnpadding` 75→2 ms,
`Tilize` 49→15 ms, `Slice`/`I2S` roughly halved. Gated by `SEAMLESS_T2U_CONV1D_DRAM_SLICE` (default on).
Confirms the through-line: **op-count reduction is the lever.**

**Decisive lesson (from #4/#5/#6 + HiFi-GAN):** this workload is **op-overhead / data-movement bound at
the op level, NOT math- or placement-bound.**
- Fidelity is a dud everywhere tried (SDPA, HiFi-GAN convs, T2U linear): individual ops are tiny
  (P150 matmuls: median 15 µs, 83% ≤ 20 µs) so they're dispatch-bound, not FLOP-bound → HiFi2 changes 0%.
- L1/sharding activations is a dud (#4/#5): kernel runtime doesn't depend on where I/O lives.
- **The only things that worked were OP-COUNT reductions** (#1 gather, #2/#3 fused SDPA = fewer/larger
  ops). So the strategy is: **reduce the number of ops** (fewer, larger matmuls/convs/chunks), not tune
  fidelity or memory config.

## Strategy / backlog (prioritized)

Through-line: **reduce op count** (fewer, larger ops). Ranked by impact × breadth. Fidelity/placement
levers are struck through — measured duds, do not revisit.

1. **Reduce FFN/linear matmul count on P150** (150 ms / 29%, **9377 tiny matmuls**, median 15 µs). The
   `_linear_*_chunked` paths split each linear into many small 256-row chunk-matmuls; each pays fixed
   dispatch overhead + a per-chunk Slice/S2I/Concat. Raise `_T2U_LINEAR_CHUNK_ROWS` 256→512 (halves the
   chunk count; the tuned `_T2U_TUNED_MATMUL` configs were actually swept at M=512) so the same FLOPs
   run in far fewer ops. Biggest single P150 lever. **Caveat / why not yet done:** the constant comment
   warns *"512-row chunks clash with decode-trace L1"* — 512 was rejected because, co-resident with the
   full-model text-decoder KV cache + Metal trace, the larger per-core CB tipped L1 over. It passes in
   this **isolated** T2U test but must be validated against the **full seamless model** (with the decode
   trace live), not this PCC test, before landing. Consider making it adaptive (512 eager / 256 under trace).
2. ~~**Reduce decoder Conv2d count**~~ ✅ **DONE (#7)** — conv1d DRAM width-slice. Conv 3024→96 ops.
3. **SDPA dense-mask streaming (decoder).** Partially addressed: **#9** dropped the all-zero *encoder*
   mask, and **#10** removed the per-layer mask *rescale*. What **remains**: the decoder still streams a
   dense `[1,1,16640,16640]` (~554 MB) bf16 mask into all 6 SDPA calls (~232 ms of the BH-QB SDPA).
   **Caveat:** ttnn SDPA's `attn_mask` is `[b,nqh,s,s]` with **only b/nqh broadcastable — the two `s`
   dims are not** (confirmed from the op docstring), so a `[1,1,1,S]` broadcast mask is rejected; a dense
   `[S,S]` is mandatory. Real reductions need an SDPA-kernel change (a native padding-mask / `valid_len`
   path, or `bfp8`/`bfp4` mask dtype to cut streaming bytes). The tail is only ≤255 padded keys of 16640,
   so dropping the mask entirely and instead zeroing padded-key contributions post-hoc is *not* correct
   (zero keys still get `exp(0)` softmax weight) — do not try it.
4. **AllGather (88 ms, BH-QB only)** — TP-comm; hard, low priority (needs TP-strategy change; see the
   text-encoder skill notes on `num_links`/`Ring`, capped at 2 on BH-QB).
5. **`ReshapeView` in the unit-upsample gather (27 ms BH-QB / 6.8 ms P150).** `_hard_upsample_gather`
   reshapes `frame_idx` `[1,S]→[1,S,1]`, which in ROW_MAJOR is a physical `[1,S]→[S,1]` transpose
   (~21 ms merged at S=16640) feeding the searchsorted `le`. Low priority, fiddly (layout-sensitive).

Struck (measured no-ops — see Optimizations tried #4/#5/#6):
- ~~matmul/SDPA/conv math fidelity~~ (ops are dispatch-bound, not FLOP-bound)
- ~~L1-resident / sharded activations~~ (kernel runtime independent of I/O placement)

### What cannot be further optimized cheaply (caveats)
- **SDPA compute itself** (the flash matmuls at seq≈16640) is real FLOPs, not glue — only the *mask*
  streaming around it is addressable, and that needs a kernel change (backlog #3).
- **Encoder-mask drop (#9) is eager-path only.** It needs a host readback to detect the all-zero mask,
  which is illegal during `begin_trace_capture`; the trace/demo path therefore keeps the mask. To get
  the win under trace, thread a host-known `encoder_mask_is_noop` flag through `capture_forward_trace`.
- **Fold-scale-into-Q (#10) is bit-exact only because `scale = 1/√64 = 0.125` is a power of two.** If a
  future config changes `head_dim` off a perfect square whose reciprocal-sqrt isn't a power of two,
  `q*scale` would round in bf16 — re-gate on PCC (the code already only applies it when a mask is present).
- **Fidelity and activation placement are settled duds** (#4/#5/#6) — do not revisit.

### Env flags for A/B (already wired)
- `SEAMLESS_T2U_SDPA_CHUNK` (default 256) — flash chunk cap.
- `SEAMLESS_T2U_TP_FUSED_SDPA` (default 1) — fused SDPA vs DRAM matmul for tp>1.
- `SEAMLESS_T2U_CONV1D_DRAM_SLICE` (default 1) — conv1d DRAM width-slice vs manual chunk loop (#7).
