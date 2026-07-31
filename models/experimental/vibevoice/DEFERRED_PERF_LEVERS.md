# Deferred VibeVoice perf levers — measured, rejected, and how to bring them back

Two non-bit-exact optimizations were built, measured, and **removed from the code** on 2026-07-31.
Both are real speedups. Both degrade a 100-minute render in the same way. This file is the record so
they can be revisited without redoing the measurement work.

Baseline for everything below: commit `072b6a91ab4`, `demo.py --demo 4p_climate_100min`, no flags
(`VV_CONV_HS=1`, `VV_POST_L2_PROGCFG=1`, `VV_TRACE_SEGMENT=1` are the shipping defaults).
Baseline audio is preserved at `ops_list/warnings.wav`, sha256 `8462bd1bf65c646e…f056d65`.

---

## Summary — why both were removed

| render | ms/tok | tok/s | gen wall | AR tokens | audio | voiced p5 | p90 | frac<½·median |
|---|---|---|---|---|---|---|---|---|
| **baseline (shipping)** | **47.81** | **20.92** | 2503 s | 42498 | **93.17 min** | 0.0363 | 0.1084 | 0.052 |
| depthwise mul+reduce only | 45.05 | 22.20 | 2147 s | 40596 | 89.40 min | 0.0320 | 0.1175 | 0.064 |
| both (dw + sharded norm) | — | — | — | 42498 | 84.40 min | 0.0294 | 0.1231 | 0.085 |

**The failure signature, identical for both and additive:** low voiced-RMS percentiles fall while
**p90 rises** — the dynamic range widens, so quiet speakers get quieter (this is the "Maya's voice is
low" report). More sub-half-median voiced seconds, fewer voiced seconds overall, shorter audio.

For the depthwise change, AR tokens fell 42498 → 40596 (−4.5%) alongside the −4.0% duration, so
**turns end early — content is lost, not merely spoken faster.** That makes it a correctness
regression, not a cosmetic one.

**The counter-intuitive part worth remembering:** the depthwise mul+reduce is 11.6× faster on the op
*and* lands closer to the fp32 reference in isolation (rel_rms 3.6e-3 vs the conv2d path's 6.0e-3).
Per-op accuracy did not predict long-form behaviour. The only gates that worked here were
byte-identity (cheap) or a full 100-min render with the percentile analysis below (43 min).

---

## Lever 1 — out_w==1 streaming depthwise conv → multiply + reduce

**Worth: −2.76 ms/tok (+6.1% tok/s), ~2.7 ms/frame.** Was `VV_DW1_MULREDUCE=1` in
`tt/ttnn_semantic_tokenizer.py`.

### Why it is fast

A streaming depthwise conv whose cache-padded input is exactly `K` wide produces ONE output column:
`out[c] = Σ_k x[k,c] · w[c,k] + b[c]` — a broadcast multiply and a K-row reduce. `ttnn.conv2d` cannot
express that cheaply: with `groups > 1` **and** a bias it misses the compact 1d-depthwise path
(`is_1d_depthwise_conv()` in `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp` requires
`!has_bias`), so `convert_conv_weight_tensor_to_grouped_layout` expands the weight to a dense
block-diagonal `[K·C, C]`. For C=2048 that is **58 MB of DRAM read per call to do 14336 MACs —
99.95% of it zeros**, measured at 187.8 µs. Multiply+reduce is 16.2 µs.

In the deployed-frame profile this shows up as `Conv2d 32 x 14336 x 2048`, **3024 µs / 16 ops =
7.8% of the frame** — the single largest non-matmul bucket.

### The code that was removed

Module constant, and in `TTConv1d.__init__` (weight pre-transposed to `[1,1,K,C]`):

```python
self._dw1: Optional[tuple] = None
if cw.groups == self.in_ch == self.out_ch and self.stride == 1 and _DW1_MULREDUCE:
    w_kc = cw.weight.reshape(self.out_ch, K).t().contiguous().reshape(1, 1, K, self.out_ch)
    self._dw1 = (
        ttnn.as_tensor(w_kc.to(tdtype), device=device, dtype=compute_dtype,
                       layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        None if cw.bias is None else ttnn.as_tensor(
            cw.bias.to(tdtype).view(1, 1, 1, -1).contiguous(), device=device,
            dtype=compute_dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
    )
```

In `TTConv1d.__call__`, immediately after `conv_padding = (0, 0, cp, extra_pad)` and before the
`VV_CONV_SINGLE_BLOCK` block. The tilize zero-fills rows `K..31` and the pre-transposed weight is
zero there too, so tile-pad rows cannot contribute to the reduce:

```python
if self._dw1 is not None and use_cache and T_padded == self.K:
    w_kc, b_c = self._dw1
    xt = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    out = ttnn.sum(ttnn.multiply(xt, w_kc, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                   dim=2, keepdim=True, compute_kernel_config=_HIFI4)
    if b_c is not None:
        out = ttnn.add(out, b_c, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return out
```

### Bit-exact alternatives worth trying instead

The prize is removing the 2048× redundant weight expansion, not this particular formulation:

1. **Unfuse the bias** so `is_1d_depthwise_conv()` accepts the conv, then add the bias as a separate
   `ttnn.add`. Keeps `ttnn.conv2d`'s own accumulation order, so it has a real chance of being
   bit-exact — the reduce order is what the mul+reduce version changes. **Untested; highest-value
   next experiment.**
2. Fix `is_1d_depthwise_conv` / the grouped-layout expansion in ttnn so a biased depthwise conv takes
   the compact path. Correct fix, wider blast radius, needs a rebuild.
3. A decomposition into 7 shift-multiply-accumulates was already tried on a *different* shape family
   and is device-slower (see the perf skill's data-movement dead-ends) — do not repeat it.

---

## Lever 2 — width-sharded RMSNorm for the 1-tile-tall decode norms

**Worth: ~1.5 ms/frame** (LayerNorm bucket 4.30 → 1.44 ms). Was `tt/ttnn_norm.py` +
`VV_SHARDED_NORM=1`, wired at 7 call sites (3 in `ttnn_vibevoice_lm.py`, 2 in
`ttnn_diffusion_head.py`, 2 in `ttnn_semantic_tokenizer.py`).

### Why it is fast

ttnn's *interleaved* layernorm family parallelises over ROWS. A decode-step norm is 1 tile tall
(single token) or 2 (CFG batch-2) and 48–64 tiles wide, so it lands on **one core**, which reads and
writes the whole row serially. Measured on Blackhole (fp32-dest HiFi4, DRAM in/out, reshard round
trip **included** in every number):

| shape | 1 core | sharded | speedup |
|---|---|---|---|
| `[1,1,32,1536]` | 25.4 µs | 6.99 µs (6 cores) | 3.6× |
| `[2,1,32,1536]` | 25.4 µs | 10.08 µs (6 cores) | 2.5× |
| `[1,1,32,2048]` | 33.4 µs | 7.45 µs (8 cores) | 4.5× |

**8 width-tiles per core was optimum in all three cases** — more cores lose to the cross-core
reduction, fewer leave the single core bandwidth-bound. Derive the grid as `width_tiles // 8`, never
`width_tiles` (sharding 48 tiles onto 48 cores *regresses*, which is the failure mode an older note
recorded as "sharded norm can't help decode").

### Accuracy

Not bit-exact on the 1536-wide shapes: `maxabsdiff` 1.6e-2 vs the 1-core kernel, because the
cross-core reduction combines partials in a different order. The 2048-wide post-diffusion norm *was*
bit-exact. PCC against an fp32 reference was unchanged to 6+ decimals (the sharded variant was
marginally *closer* on two of three shapes) — which, again, did not predict the long-form outcome.

### Reconstruction sketch

`ttnn.create_sharded_memory_config(shape=(row_tiles*32, width//ncores),
core_grid=ttnn.CoreGrid(y=1, x=ncores), strategy=ttnn.ShardStrategy.WIDTH,
orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)` plus
`ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(ncores,1),
subblock_w=1, block_h=row_tiles, block_w=width_tiles//ncores, inplace=False)`, converting back to the
caller's memory config afterwards, and falling back to plain `ttnn.rms_norm` for any shape that
doesn't match.

Two shape gotchas that cost time: `row_tiles` must be counted off the **tile-padded** height
(`shape[0] * shape[1] * ((shape[2] + 31) // 32)`, not `shape[2] // 32`, or real `[2,1,1,1536]`
tensors are rejected), and `block_h` must equal that row-tile count (`block_h=1` `TT_FATAL`s for
B=2).

### Not separately bisected

The 84.40-min render had both levers on. The depthwise lever alone gives 89.40 min, so the sharded
norm plausibly accounts for the remaining ~5 min, but **a sharded-norm-only 100-min render was never
run.** That is the one missing datapoint if this lever is revisited.

---

## How to gate a revisit

1. **Prefer byte-identity.** `sha256sum` against `ops_list/warnings.wav` costs seconds; a render costs
   43 minutes. Any change that survives byte-identity needs no audio QA at all.
2. If a change cannot be bit-exact, run the full render and compare **distributions**, not waveforms —
   the AR loop is chaotic, so frames 0–3 match at PCC 0.9999 and then legitimately fork. Waveform PCC
   against a prior build is not a valid gate.
3. Check, in order: **AR token count** and **audio duration** (content loss), then voiced-RMS
   percentiles p5/p10/p25/p50/p75/p90 over seconds with RMS > 0.02 (per-speaker level), then clipping
   / anomalous minutes. The p5-down-with-p90-up pattern is the fingerprint of this regression.
4. Per-op PCC, isolated fp32 comparison, and energy/clipping QA bands **all passed** for both rejected
   levers. Do not rely on them.

## Reference artifacts

| file | what |
|---|---|
| `ops_list/warnings.wav` | 93.17-min baseline, sha256 `8462bd1b…` |
| `output/4p_climate_100min/4p_climate_100min_tt_DW1.wav` | depthwise-only render, 89.40 min |
| `output/4p_climate_100min/4p_climate_100min_tt_ALLCHANGES.wav` | both levers, 84.40 min |
| `ops_list/render100_gamma_tile.log` | baseline render log |
| `ops_list/render100_dw1_mulreduce.log` | depthwise-only render log |
| `ops_list/deployed_frame_05_gamma_tile_tt_perf_report.txt` | deployed-frame op profile, shipping config (4235 ops / 38.9 ms) |

Deployed-frame profiles are captured by running the frame eagerly during trace warmup and exiting
before `begin_trace_capture` — trace replay emits no per-op profiler data, and `--no-trace` profiles a
different graph. See `tests/perf/capture_deployed_frame.sh` and its `.hook.patch`.
