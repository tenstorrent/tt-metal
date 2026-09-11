# DFlash drafter — device operator mapping (T3K)

Per-operator device profile of the **ttnn DFlash drafter alone** ([tt/dflash/drafter.py](tt/dflash/drafter.py)),
with every op traced back to the ttnn call that emitted it. The 27B target is **not** in this
capture — it is 94% of a real speculative step, so an end-to-end report says nothing about the
drafter.

## How this was produced

```bash
ls generated/profiler/reports/ | sort > /tmp/reports_before.txt
MESH_DEVICE=T3K DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \
  python -m tracy -p --op-support-count 100000 -r -v -m \
    pytest "models/demos/blackhole/qwen36/tests/perf/test_profile_dflash_drafter.py::test_profile_dflash_drafter[wormhole_b0-device_params0-1x8-ctx64]"

D=2026_09_11_05_29_09
tt-perf-report generated/profiler/reports/$D/ops_perf_results_$D.csv \
  --start-signpost start --end-signpost stop --no-color
```

Report dir `2026_09_11_05_29_09`, **174** ops in the signposted window (IDs 6391–7773).

> **This capture excludes the harness's own uploads, and earlier ones did not.** Until now the
> measured step ran `ttnn.from_torch` of 5 taps + the noise block *inside* the signposts — six
> `TilizeWithValPadding`, **180 μs (1.7 %)** — which production never pays: `TtDrafter.propose`
> takes the taps on-device from the target's `_record_tap` and the noise from
> `target.embed_device`. They are hoisted out of the window in
> [test_profile_dflash_drafter.py](tests/perf/test_profile_dflash_drafter.py), so **10.34 ms is the
> drafter and 10.54 ms was the drafter plus the test.** (An earlier revision of this file put those
> six ops at 198 μs; the correct figure is 180 — five taps at 33 μs plus a 15 μs block.)
(`profile_log_device.csv` deleted afterwards — it is 10–20 GB on larger captures.)

**Capture verified as the drafter** before analysis, by op-count signature against the source
structure — all five counts are exact, which no other module in this repo would match:

| expected from source | count |
| --- | --- |
| `fc` + 6 projections × 5 layers (k and v are **one fused** `kv_proj`) | **31** Matmul |
| `hidden_norm` + (input_ln, q_norm, k_norm, post_attn_ln) × 5 + final `norm` | **22** LayerNorm |
| the single tap gather (drafter is replicated) | **1** AllGatherAsync |
| one per layer | **5** SDPA |
| q and k per layer | **10** RotaryEmbeddingHf |

## Run configuration

| | | evidence |
| --- | --- | --- |
| Mesh | T3K, `(1, 8)` | `parametrize_mesh_tp()` |
| Parallelism | **none — replicated on all 8 devices** | 1 collective in the window; [weights.py](tt/dflash/weights.py) |
| Block | 16 slots (1 anchor + 15 drafted) | `cfg.block_size` |
| New context rows/step | 16 | test `n_new` |
| Committed KV history | 64 rows → SDPA k/v length 96 | `ctx64` param; `96 = 64 + 16 + 16` |
| Hidden / heads | 5120, 32 q heads / 8 kv heads × 128 | GQA 4:1, all local |
| MLP intermediate | 17408 | SwiGLU |
| Layers | 5 — 4 causal sliding-window (2048) + 1 bidirectional | `cfg.layer_types` |
| Weight dtype | `BFLOAT8_B`, except **`BFLOAT4_B` for MLP gate/up**; DRAM-interleaved | CSV `INPUT_1`; [weights.py](tt/dflash/weights.py) `MLP_DTYPE` |
| Activation dtype | `BFLOAT16`, DRAM-interleaved | CSV `INPUT_0` |
| Math fidelity | matmul/SDPA **HiFi2**, norms/RoPE **HiFi4** | CSV `MATH FIDELITY` |

## Where the time goes

**Total device time: 10.34 ms per drafter step.**

| Total % | Op code | Time | Ops | Cores |
| --- | --- | --- | --- | --- |
| 26.5 % | `Matmul 32 x 5120 x 17408` (gate 284 μs + up 265, **BFP4**) | 2,745 μs | 10 | 61 |
| 22.2 % | `Matmul 32 x 17408 x 5120` (down, BFP8) | 2,292 μs | 5 | 54 |
| 15.3 % | `LayerNorm` | 1,580 μs | 22 | **1** |
| 6.5 % | `Matmul 32 x 25600 x 5120` (`fc`) | 670 μs | 1 | 54 |
| 5.5 % | `Matmul 32 x 4096 x 5120` (o, **auto** progcfg) | 570 μs | 5 | 54 |
| 5.4 % | `Matmul 32 x 5120 x 4096` (q, 1D 8x8) | 560 μs | 5 | 64 |
| 3.2 % | `ReshapeView` | 335 μs | 10 | 64 |
| 2.7 % | `Matmul 32 x 5120 x 2048` (**fused** k\|v, 1D 8x8) | 275 μs | 5 | 64 |
| 2.4 % | `NlpCreateHeads` (tied k/v split) | 251 μs | 5 | **1** |
| 2.3 % | `BinaryNg` (SwiGLU mul, residual adds) | 242 μs | 19 | 64 |
| 1.6 % | `AllGatherAsync` (tap gather) | 161 μs | 1 | 6 |
| 1.5 % | `RotaryEmbeddingHf` | 154 μs | 10 | 32 |
| 1.1 % | `SDPA` | 116 μs | 5 | 64 |
| 0.9 % | `Tilize` (kv_src retilize ×5, RoPE ×2) | 98 μs | 7 | — |
| 0.6 % | `UntilizeWithUnpadding` (kv_src concat operands) | 63 μs | 6 | 54 |
| 0.2 % | `TilizeWithValPadding` (q cos/sin, mask) | 24 μs | 3 | 1 |
| ~2.0 % | Concat / Slice / Transpose / Copy | 204 μs | 55 | — |

Rolled up: **MLP 48.7 %**, attention projections 13.6 %, norms **15.3 %**, `fc` 6.5 %, layout churn
~9.4 %, SDPA 1.1 %, collective 1.6 %.

**Matmul FLOPs efficiency: 12.21 % mean** (9.00–17.25 %).

Note the gate/up split: `gate` is 284 μs and `up` 265 μs for the same shape and dtype, because
`gate` carries the explicit progcfg that packer-fuses its SiLU (see `_layer_mlp`); the 19 μs
difference is that config, and it buys the removal of a separate 18 μs `Unary`.

> **All 185 μs of remaining tilize/untilize is load-bearing** (16 ops): the `kv_src` row-axis
> concat (5 retilize at 18 μs + 5 operand untilize at 10.5 + 1 hoisted `ctx_rm` untilize), the RoPE
> slices (2 at 4 μs for the K span, 2 at 8 μs for Q's — `TilizeWithValPadding` because q_len 16 pads
> to a tile), and the 7 μs `_sliding_mask` upload. At session start the same capture carried
> **511 μs across 44 ops**.

> **Cumulative against the first capture** (`2026_09_09_12_35_01`, 13.52 ms / 230 ops):
> **−2,845 μs, −21.0 %.** Three changes, in order:
>
> 1. `k_proj`/`v_proj` fused into one `kv_proj` matmul + a tied head-split (36 → 31 matmuls):
>    **−293 μs** (`2026_09_10_06_33_06`, 13.23 ms).
> 2. Explicit 1D program configs on `q_proj` and `kv_proj` with L1 outputs (`_proj_pc`, swept by
>    [tests/perf/test_dflash_attn_matmul_sweep.py](tests/perf/test_dflash_attn_matmul_sweep.py)):
>    **−373 μs**, of which −360 μs is matmul (`kv_proj` 116 → 55 μs, `q_proj` 121 → 112 μs) and the
>    rest is q's reshape/transpose getting cheaper from an L1 input
>    (`2026_09_10_06_58_22`, 12.86 ms).
> 3. **BFP4 weights for MLP gate/up** — the only lever the MLP sweep found, and NOT a program config
>    ([weights.py](tt/dflash/weights.py) `MLP_DTYPE`): **−2,180 μs**, gate+up 4,840 → 2,640 μs.
>    `down_proj` stays BFP8; at BFP4 the full drafter drops to 0.9891 PCC and fails the suite's 0.99
>    gate. Cost of the change that shipped: full-drafter PCC **0.9978 → 0.9942**.
>
> 4. **Removed the redundant layout work** (`2026_09_10_09_01_42`, 10.52 ms): **−165 μs and −30
>    dispatches**, 215 → 185 ops, at bit-identical PCC. Q's `cos`/`sin` were being sliced off the
>    *tilized* tables once per layer — untilize + slice + retilize, five times for an identical
>    result — and are now sliced once per step off the resident ROW_MAJOR tables (130 μs / 30 ops →
>    20 μs / 4 ops). The `kv_src` concat's ROW_MAJOR conversion of `ctx_kv` is likewise hoisted out
>    of the layer loop. `UntilizeCodegen` is gone from the capture entirely.
>
> The shape of the problem has changed as a result. The MLP is no longer the whole story at 47 %,
> **`LayerNorm` is now 15.0 % on ONE core** — the largest single unexploited item in the capture.

### Four levers, in order

1. **Every matmul runs M = 32 for 16 real rows.** The CSV shows `32[16]` — padded 32, logical 16.
   A 16-slot block wastes half of every matmul on tile padding, so **`block_size = 32` would cost
   the same device time as 16** and draft twice as many tokens. This is the cheapest lever in the
   whole DFlash stack and it is a config change, not a kernel change. (It interacts with acceptance:
   more drafted slots is only useful if the drafter can predict that far, which is unmeasured.)
2. **LayerNorm runs on 1 core** — 22 norms, 1,580 μs, 11.7 % of the step. A sharded norm (the
   target's own `get_norm_config` path does this) should take this to ~2 %.
3. ~~**MLP is 55 %, so give it explicit / DRAM-sharded program configs.**~~ **SWEPT, AND FALSE —
   the MLP's win came from the weight dtype instead.**
   [test_dflash_mlp_matmul_sweep.py](tests/perf/test_dflash_mlp_matmul_sweep.py) found **auto
   optimal at all three MLP shapes**, the best of ~14 explicit candidates losing by 1.7–5.5 %. The
   `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` this entry used to call "the standard
   fix" is **+207 % on `down`** (1,371 μs vs 447) and **illegal on `gate`** — it rejects a fused
   activation. Fusing `gate|up` into one matmul also loses (975 μs vs 959 μs split, before the
   slice + silu it would need), so the `kv_proj` trick does not generalize. What DID pay was **BFP4
   on gate/up (−2,180 μs, now shipped)**; `down` stays BFP8 because BFP4 there fails the PCC gate.
   The MLP is now 46 % and, on the auto config, has no known lever left short of a smaller weight.
4. **174 eager dispatches per step, and this is the top lever — device time is only ~9 % of wall.**
   MEASURED at **120 ms/step** un-profiled against 10.3 ms of device time
   ([test_dflash_drafter_wall_time.py](tests/perf/test_dflash_drafter_wall_time.py), T3K, 30 steps),
   i.e. **~0.6 ms of wall per dispatch**. That is the exchange rate to optimize against: removing
   one op is worth roughly 30x more than saving one microsecond of device time. Today's
   230 → 174 dispatches is therefore the largest win in this document, worth ~30 ms of wall against
   the 3 ms of device time the same changes saved. It also means a **trace** is the single biggest
   remaining lever — but **the drafter as written cannot be traced**, and an earlier revision of
   this entry was wrong to say "the block is a fixed shape and should trace well". The block is
   fixed; the KV history is not. MEASURED
   ([test_dflash_drafter_trace_probe.py](tests/perf/test_dflash_drafter_trace_probe.py)): the
   per-layer history grows `(1,8,48,128)` → `(1,8,64,128)` every step, so `shape_stable=False` and
   capture dies on *"Cannot load new binaries during trace capture"* — every step compiles programs
   it has never seen, which no amount of warm-up fixes. Tracing needs the fixed-capacity KV buffer
   written in place that the layout notes below have wanted anyway, plus a resident mask (the
   `_sliding_mask` host upload is illegal in a capture) and offset-free RoPE.

   *Old text, kept because the framing was wrong:* Device time is 10.52 ms but an un-profiled step measures
   ~100 ms wall (`test_dflash_throughput`: `propose` 0.6 s / 6 steps, which also includes the LM
   head and a host argmax) — so roughly **13 % device utilization**. The block is a fixed shape, so
   it should trace well.

> ⚠️ **Do not read op-to-op gaps from this capture.** Tracy inflates host-side gaps enormously: the
> window's gap total is 3.66 s for a step that really takes ~100 ms, and the report's "device busy
> 0.4 %" is a profiling artifact. Device *times* are accurate; for the busy fraction use the
> un-profiled wall clock above.

## Prologue — `project_taps`

`ttnn` source: [drafter.py:217-253](tt/dflash/drafter.py#L217) (`project_taps`).

| Perf ID | Perf OP Code | ttnn op — block / why / config | In shape | Out shape | Weight dt / fid |
| --- | --- | --- | --- | --- | --- |
| 6502–6542 | `TilizeWithValPadding` ×6 | **test harness only** — `ttnn.from_torch` of 5 taps + the noise block. In production the taps arrive already on device from the target's `_record_tap`; only the noise embedding is uploaded. 33 μs each | — | — | — |
| 6549 | `Concat` | `ttnn.concat(taps, dim=-1)` — joins this device's 5 tap slices, [drafter.py:230](tt/dflash/drafter.py#L230) | 5 × `16x640` | `16x3200` | — |
| 6560 | `AllGatherAsync` | `tpc.tuned_vocab_all_gather(dim=3)` — **the drafter's only collective**, [drafter.py:237](tt/dflash/drafter.py#L237). Gathers to device-major column order, which is why `fc`'s rows were permuted at load time | `16x3200` DRAM-IL | `16x25600` DRAM-IL | — |
| 6569 | `Matmul 32 x 25600 x 5120` | `ttnn.linear(gathered, weights.fc)` — the tap projection, [drafter.py:248](tt/dflash/drafter.py#L248). 669 μs, the largest single matmul in the step, at 72 % of its DRAM roofline. **SWEPT ([test_dflash_fc_matmul_sweep.py](tests/perf/test_dflash_fc_matmul_sweep.py)) and left on auto**: best explicit config +1.6 %, DRAM-sharded +22.6 %, `prefill_mlp` +39.8 %. bf4 would be −19.9 % but takes the full drafter to 0.9796 PCC (4 of 5 suite tests fail) — `fc` feeds every layer, so unlike the MLP its error propagates five times | `16x25600` BF16 | `16x5120` BF16 | BFP8 / HiFi2 |
| 6576 | `LayerNorm` | `hidden_norm` — standard RMSNorm, gain used as-is (**not** the target's zero-centred fold), [drafter.py:251](tt/dflash/drafter.py#L251). 125 μs on 1 core | `16x5120` | `16x5120` | BF16 / HiFi4 |

## Per-step setup — RoPE slice + mask

| Perf ID | Perf OP Code | ttnn op — block / why / config |
| --- | --- | --- |
| 6581, 6601 | `Slice` ×2 | `_rope_slice` — cos/sin for `[start-n_new, start+q_len)` sliced out of the **resident** ROW_MAJOR tables, [drafter.py:167](tt/dflash/drafter.py#L167). Host trig happens once at construction, not per step |
| 6589, 6606 | `Tilize` ×2 | `ttnn.to_layout(TILE)` on the sliced cos/sin — the slice must be ROW_MAJOR (non-tile-aligned rows), so it is tilized after |
| 6654 | `TilizeWithValPadding` | `_sliding_mask` upload, [drafter.py:289](tt/dflash/drafter.py#L289). Built **once per step**, not per layer: all four sliding layers see the same `(q_len, kv_len)` because their histories commit in lockstep |

## Repeating layer (×5)

`ttnn` source: `_layer_attention` [drafter.py:309](tt/dflash/drafter.py#L309), `_layer_mlp` [drafter.py:417](tt/dflash/drafter.py#L417).
IDs below are layer 0; each subsequent layer repeats the pattern.

### Attention

| Perf ID | Perf OP Code | ttnn op — block / why / config | In shape | Out shape | Weight dt / fid |
| --- | --- | --- | --- | --- | --- |
| 6659 | `LayerNorm` | `input_layernorm` on the **noise branch only** — `kv_source` goes to `kv_proj` raw, which is the drafter's own asymmetry, [drafter.py:498](tt/dflash/drafter.py#L498). 125 μs, 1 core | `16x5120` | `16x5120` | BF16 / HiFi4 |
| 6668–6686 | `Untilize`, `Concat`, `Tilize` | `ttnn.concat([ctx_rm, hidden_rm], dim=-2)` — the K/V source spans context then block, [drafter.py:334](tt/dflash/drafter.py#L334). 16 + 16 rows are both part-tile, so the concat cannot happen in TILE and the round trip is real: untilize 10 μs + concat 7 + retilize 18. Only **one** untilize now — `ctx_kv`'s ROW_MAJOR copy is made once per step in `forward`, not per layer. **Keep the concat**: the alternative (two matmuls, one per row group) costs 2 × 55 μs because M = 16 pads to 32 either way, against 55 + 35 here, and it would double the 50 μs head-split too |
| 6694 | `Matmul 32 x 5120 x 4096` | `q_proj` — Q from the block only. **Explicit 1D (mcast_in0) progcfg on the full 8x8 grid** + L1 output, [drafter.py:182](tt/dflash/drafter.py#L182). 112 μs against 121 μs on auto | `16x5120` | `16x4096` L1 | BFP8 / HiFi2 |
| 6702 | `Matmul 32 x 5120 x 2048` | **fused `kv_proj`** — k and v in one matmul; they read the same rows, so the weights are concatenated on the output dim at load time ([weights.py](tt/dflash/weights.py) `_kv_proj`). Note **M = 32 real here** (ctx 16 + noise 16), the only matmul not padding-dominated. Same explicit 1D progcfg + L1 output. **55 μs** — against 2×88 μs for the split pair on auto, and against 116 μs for this fused shape on auto: N = 2048 is 64 tiles and auto lands at 33 % of the DRAM roofline, which one program config fixes. The two changes together are **−188 μs per step** on this matmul alone | `32x5120` | `32x2048` L1 | BFP8 / HiFi2 |
| 6707–6730 | `ReshapeView`, `Transpose`, `Copy` | `_heads` for **Q only** — `[1,1,16,4096] → [1,32,16,128]`, [drafter.py:255](tt/dflash/drafter.py#L255). 25 + 5 μs from an L1 input (29 + 6 from DRAM), then a 3 μs `Copy` back to DRAM that did **not** exist before the L1 output — see the optimization note |
| 6736 | `NlpCreateHeads` | `_kv_heads` — `nlp_create_qkv_heads(kv_tied=True)` splits the fused `[k\|v]` block into head-major K and V in **one** op, [drafter.py:261](tt/dflash/drafter.py#L261). Verified bit-exact against the reshape+transpose form. 50 μs on **1 core** (its work split is `B * S/TILE_HEIGHT` = 1 block) against 57 μs for the four ops it replaces — a wash on device time; the reason to keep it is 4 fewer dispatches per layer in an untraced module |
| 6742 | `LayerNorm` | `q_norm` over head_dim, fused gain. 9 μs | `32x16x128` | same | BF16 / HiFi4 |
| 6748 | `LayerNorm` | `k_norm` over head_dim. 7 μs | `8x32x128` | same | BF16 / HiFi4 |
| 6761–6766 | `Copy`, `RotaryEmbeddingHf` | `apply_partial_rope_prefill(k, …)` — full 128-dim rotation at θ=1e7, HF half-split. `rope_dim == head_dim`, so this is the fused `rotary_embedding_hf` with no pass-through concat |
| 6772–6780 | `Copy`, `RotaryEmbeddingHf` | Q's rotation. Q takes only the trailing `q_len` positions, which is what pins the block to its absolute positions — but its cos/sin are now sliced **once per step** in [`forward`](tt/dflash/drafter.py#L460), off the ROW_MAJOR tables. This row used to also carry `UntilizeCodegen` + `Slice` + `TilizeWithValPadding` ×2 per layer, 26 μs × 5, for a slice whose result never varied by layer |
| 6790–6813 | `Concat` ×2, `Slice` ×2 | KV-history prepend then commit: `concat([hist, k])` for SDPA, then `slice` to `hist_len + n_new` to persist only accepted rows — the block's own K/V is scratch, [drafter.py:378](tt/dflash/drafter.py#L378) |
| 6828 | `SDPA` | `scaled_dot_product_attention(attn_mask=mask if sliding else None, is_causal=False)`, [drafter.py:390](tt/dflash/drafter.py#L390). `is_causal=False` is required: Q is the block only, so SDPA's own causal alignment would be wrong. **24 μs — 0.9 % of the step** | q `32x16x128` L1-IL, k/v `8x96x128` | `32x16x128` | — / HiFi2 |
| 6837, 6845 | `Transpose`, `ReshapeView` | head concat back to `[1,1,16,4096]`, [drafter.py:408](tt/dflash/drafter.py#L408). 7 + 41 μs on 64 cores. **Deliberately not `nlp_concat_heads`**, which is bit-identical here and one op instead of two but runs on **1 core** at this shape (63 μs) — measured +69 μs over the step |
| 6857 | `Matmul 32 x 4096 x 5120` | `o_proj` — a **plain local matmul**; replicated, so no all-reduce. 114 μs, and **deliberately still on the auto progcfg**: the sweep found nothing that beats it at this shape | `16x4096` | `16x5120` | BFP8 / HiFi2 |
| (per layer) | `BinaryNg` | `ttnn.add(residual, attn)` |

### MLP — 46 % of the step

| Perf ID | Perf OP Code | ttnn op — block / why / config | In shape | Out shape | Weight dt / fid |
| --- | --- | --- | --- | --- | --- |
| (per layer) | `LayerNorm` | `post_attention_layernorm`. 125 μs, 1 core | `16x5120` | `16x5120` | BF16 / HiFi4 |
| 6880 | `Matmul 32 x 5120 x 17408` | `gate_proj` with `activation="silu"` — which does **not** actually fuse into the packer on the auto config; it emits the separate `Unary` below. [drafter.py:447](tt/dflash/drafter.py#L447). **264 μs, 61 cores** at BFP4, against 490 μs at BFP8 | `16x5120` | `16x17408` | **BFP4** / HiFi2 |
| 6897 | `Matmul 32 x 5120 x 17408` | `up_proj` — same shape, no activation. 266 μs at BFP4 against 469 | `16x5120` | `16x17408` | **BFP4** / HiFi2 |
| (per layer) | `UnaryDeviceOperation` | the fused SiLU's unary stage (5 ops, 95 μs total) |
| (per layer) | `BinaryNg` | `ttnn.mul(gate, up)` — SwiGLU gate | 2 × `16x17408` | `16x17408` | — |
| 6908 | `Matmul 32 x 17408 x 5120` | `down_proj` — plain local matmul, no all-reduce. **457 μs, 54 cores**, and now the **hottest single op in the step**. Stays BFP8: at BFP4 it is 358 μs but the full drafter falls to 0.9891 PCC, under the 0.99 gate ([weights.py](tt/dflash/weights.py) `MLP_DOWN_DTYPE`) | `16x17408` | `16x5120` | BFP8 / HiFi2 |
| 6917 | `BinaryNg` | `ttnn.add(residual, mlp)` |

## Epilogue

| Perf ID | Perf OP Code | ttnn op — block / why / config |
| --- | --- | --- |
| 7971 | `LayerNorm` | final `norm`, [drafter.py:519](tt/dflash/drafter.py#L519). 125 μs. Its output feeds the **target's** LM head (`TtTarget.lm_head_device`), which is deliberately outside this window — that projection belongs to a target report |

## Optimization notes

- **Padding, not compute, is the top waste.** `32[16]` on every activation means a 16-slot block
  pays a 32-row tile everywhere. Doubling `block_size` to 32 is free in device time.
- **LoFi buys nothing on any of the seven matmuls, because none of them is compute-bound.** Swept
  crossed with in0 placement and weight sharding
  ([test_dflash_fidelity_sharding_sweep.py](tests/perf/test_dflash_fidelity_sharding_sweep.py)):
  every LoFi-vs-HiFi2 delta on the shipped path is <= 0.9 %, inside the +/-0.3 % run-to-run floor,
  while per-matmul PCC drops 0.999967 -> 0.999880. The tell was in the report all along — **60-69 %
  of DRAM bandwidth against 9-17 % of FLOPs peak.** Fidelity only reaches the critical path on the
  *sharded* path, where LoFi is worth -18 % on `up` — and that path is 64 % slower overall, so the
  win is unreachable. Keep HiFi2.
- **Precision has been spent where it was cheapest, and that budget is now used up.** BFP4 for MLP
  gate/up, BFP8 everywhere else, BF16 activations, HiFi2 matmuls. Full-drafter PCC against the fp32
  host reference: 0.9978 all-BFP8 → **0.9942 as shipped** → 0.9891 if `down_proj` also went BFP4,
  which fails the 0.99 gate. Norms and RoPE stay at HiFi4 — leave them; they are 16 % of the step
  now and a drafter's per-channel gains and position phases are the last place to trade accuracy.
  The BFP4 trade was checked on the metric that actually matters: **acceptance is unchanged at
  5.089 tok/step** (bf8 5.089, 56 steps each, full 27B on T3K, greedy) and greedy output ids are
  bit-identical — [tests/reference/test_dflash_acceptance.py](tests/reference/test_dflash_acceptance.py),
  whose docstring carries the resolution and greedy-only caveats.
- **The step is not drafter-bound end to end: 50 of 56 speculative steps take the ROLLBACK path.**
  A partial acceptance costs `generate.py` a second target forward, and the target is ~94 % of a
  step, so ~89 % of steps pay roughly double. That dwarfs everything left in this capture — the
  whole drafter is 10.68 ms against a target forward measured in hundreds of ms — and the fix is in
  the speculative loop (`generate.py`'s restore-and-replay, or the per-candidate-state select its
  docstring already sketches), not in `tt/dflash/`.
- **The collective is not a problem** — 168 μs, 1.2 %. The earlier tensor-parallel drafter was
  replaced partly on a mistaken "CCL-bound" reading (see README-DFLASH.md); this profile is what
  that claim should have been checked against. Replication costs one gather and nothing else.
- **SDPA is free** (0.9 %) even at 96 k/v positions, so the sliding-window mask and the
  bidirectional layer cost essentially nothing. Growing the drafter's context is cheap.
- **Layout churn is ~10.9 %** (Tilize 301, Reshape 336, NlpCreateHeads 250, Untilize 63, Concat 71,
  Transpose 56, Slice 37, Copy 39 μs), and 198 μs of that Tilize is the harness, not the drafter.
  What remains is now genuinely load-bearing, and the audit that got it there is worth repeating
  elsewhere: **two of the three layout costs were per-layer recomputation of a per-step value**, not
  op-selection mistakes. Q's cos/sin slice (130 μs, 30 ops) and `ctx_kv`'s ROW_MAJOR conversion
  (4 of 5 untilizes) both produced the same bytes in every one of the five layers. Nothing about the
  op names said so — only the fact that the loop body did not depend on `layer_idx` for those lines.
- **The fused-op head shuffles are single-core at this shape — do not re-propose
  `nlp_concat_heads`.** Both `nlp_create_qkv_heads` and `nlp_concat_heads` split work as
  `num_blocks = B * S/TILE_HEIGHT`, so a 16-slot block (one tile row) gets **one core**, while
  `reshape`+`transpose` spread over 64. Measured over 5 layers: `nlp_concat_heads` 314 μs vs 245 μs
  for the `transpose`+`reshape` pair it would replace (**+69 μs**, reverted), and
  the tied k/v split 251 μs vs 287 μs (kept for the dispatch count, not the device time). These ops
  are the right answer at prefill sequence lengths, not at a drafted block.
- **At M = one tile row, the auto program config is not a baseline — it is a bug surface.** All
  three `_layer_attention` matmuls were on auto; an explicit 1D `mcast_in0` config on the full 8x8
  grid took `kv_proj` from 116 to 55 μs (33 % → 72 % of the DRAM roofline) and `q_proj` from 121 to
  112 μs at identical PCC, while `o_proj` was already optimal on auto. Nothing distinguishes the
  three without measuring, so the sweep
  ([test_dflash_attn_matmul_sweep.py](tests/perf/test_dflash_attn_matmul_sweep.py)) is the artifact,
  not the numbers. Losers recorded there so they are not retried: DRAM-sharded weights (+11 % to
  +24 %, plus an in0 reshard op), the 2D prefill factories (+23 % to +37 %), and
  `COMPUTE_HIFI2_NO_FP32_ACC` (slower *and* costs PCC).
- **The SiLU is packer-fused, and that is a dispatch trade, not a speed one.**
  `activation="silu"` on the auto config does NOT fuse — it emits a separate `Unary` (18 μs x 5).
  An explicit progcfg with `fused_activation=SILU` does, at +23 μs on the gate matmul: **+21 μs
  device, −5 dispatches**, which at ~0.6 ms of wall per dispatch is a clear net win (see lever 4).
  Also measured and rejected: `ttnn.swiglu` is **not** a fused kernel — it decomposes into
  Slice + Unary + BinaryNg (5 ops, +62 μs) and returns the tile-PADDED height, which a 16-row block
  cannot use ([test_dflash_swiglu_fusion_sweep.py](tests/perf/test_dflash_swiglu_fusion_sweep.py)).
- **Open, self-inflicted: 17 μs of `Copy`.** `q_proj`'s L1 output pays for itself (q's reshape
  29 → 25 μs, transpose 6 → 5), but `_heads` ends with `to_memory_config(_DRAM)`, which was free
  when its input was already in DRAM and is now a real 3 μs L1→DRAM copy per layer. Q's next three
  consumers (`q_norm`, RoPE, SDPA) all read L1 — RoPE currently pays its own DRAM→L1 `Copy` to get
  there — so keeping Q in L1 through SDPA should remove both copies rather than move one. Not yet
  measured.
- **Every hint in tt-perf-report's own "Matmul Optimization" section is neutral or harmful here.**
  All three are now measured across all seven matmul shapes, and none of them is applied:
  - *"Try a DRAM-sharded program config"* (printed on 22 rows): **+11 % to +207 %**, plus an in0
    reshard op, and **illegal on `gate`** — the DRAM-sharded matmul rejects a fused activation.
  - *"If possible place input 0 in L1"* (printed on all 8 `SLOW` rows): noise at five shapes and
    **+415 % on `gate`/`up`** — a 5x regression on the two rows it is printed on most insistently
    ([test_dflash_in0_l1_sweep.py](tests/perf/test_dflash_in0_l1_sweep.py)). PCC identical, so it is
    the auto config resolving to a far worse blocking once in0 is in L1.
  - *"use HiFi4 with BF16 activations for full accuracy"*: an accuracy suggestion, not a perf one;
    the fidelity/no-fp32-acc variants lost on time **and** on PCC.

  The hints are shape-family heuristics and this model is off their beaten path in one specific way:
  **M is one tile row.** Treat that section as a list of hypotheses to sweep, never as advice.
- **The auto config is not uniformly bad — it is uniformly unknown.** Three call sites are now
  swept with identical candidate families, and the answer flips by call site: in
  `_layer_attention` explicit 1D configs took `kv_proj` −54 % and `q_proj` −9 %, while in
  `_layer_mlp` (all three shapes) and in `fc` every explicit candidate LOST, by 1.6–5.5 % for the
  best and by +207 % for the DRAM-sharded family tt-perf-report itself suggests. **7 of the 8
  matmuls in this model are best left on auto**; the exception was not predictable from shape,
  dtype or roofline efficiency. Neither "auto is fine" nor "always hand-write a progcfg" is a rule
  here — the sweeps run in ~90 s and are the only reliable answer.
- **BFP4 is affordable in exactly one place in this model, and it is not the biggest matmul.**
  Every projection has now been measured at bf4 against the full-drafter 0.99 PCC gate:

  | bf4 on | full-drafter PCC | time it would buy | verdict |
  | --- | --- | --- | --- |
  | MLP `gate`+`up` | **0.9942** | −2,180 μs | **shipped** |
  | `o_proj` | 0.9899 | −105 μs | fails by 0.0001 |
  | `down_proj` | 0.9891 | −443 μs | fails |
  | `fc` | 0.9796 | −130 μs | fails |
  | `kv_proj` | 0.9788 | −29 μs | fails |
  | `q_proj` | 0.9718 | −235 μs | fails |
  | `q`+`kv`+`o` together | 0.9545 | −370 μs | fails badly |

  The ordering tracks **position in the graph**, not matmul size: a SwiGLU input tolerates it, the
  two residual writers (`o_proj`, `down_proj`) both land at ~0.989 just under the gate, the
  attention-logit projections are worst because their error moves the SDPA logits where softmax
  amplifies rather than averages it, and `fc` is bad because every layer reads it. Do not try bf4
  anywhere else here without re-running the ladder.
- **bf4's verdict depends on position in the graph, not on the matmul.** Same dtype, same kind of
  weight-streaming op, −20 % either way, opposite conclusions: on MLP gate/up it cost 0.9978 →
  0.9942 PCC and shipped; on `fc` it cost 0.9942 → **0.9796** and failed 4 of 5 tests, because `fc`
  is the drafter's entrance and every layer reads its output. Do not generalise a dtype win from
  one call site to another.
- **Fusing weights that share an activation is the lever that paid — in exactly one place.** `k_proj`/`v_proj` read the
  same `kv_src` rows, so one `[k|v]` weight is one matmul: −273 μs, and its `[k|v]` column layout is
  exactly what the tied head-split consumes. `q_proj` cannot join it — Q comes from the *normed*
  block while K/V come from the raw `[ctx, block]` rows, a 16-vs-32-row shape difference, and
  slicing Q's 16 rows back out of a 32-row tile is not tile-aligned. It does **not** generalize:
  `gate`/`up` share an activation too, and fusing those measured slower (lever 3).
