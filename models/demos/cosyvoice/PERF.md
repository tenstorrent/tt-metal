# Performance and accuracy

Every measured performance and accuracy figure for this port. `README.md` and
`docs/VALIDATION.md` link here rather than quoting. Unmet requirements, open defects and
workarounds are in `docs/VALIDATION.md`.

Two parts, with different provenance:

* Part I — the certification run. One commit, one day, three boards, five
  configurations each. Every figure is reproducible by re-running the suite, and every
  numeric threshold is asserted by a test.
* Part II — the engineering record: the A/B measurements behind the design. An A/B
  compares against something no longer in the tree, so each row carries the board and
  the date it was measured on.

---

# Part I — the certification run

## 1. What was run, and where

| | |
|---|---|
| Commit | ``c84b4f151b6`` |
| Date | 2026-08-30 |
| tt-metal | `571b1e0395` (the C++/ttnn build the model was overlaid onto) |
| Boards | Blackhole `p150a`, Blackhole `p150b`, Wormhole n300 |
| Configurations | host; `pcc`+`e2e`; `perf` × {default, `COSYVOICE_FF2_GRID=8x2`, `COSYVOICE_KV_INPLACE=1`} |

```bash
pytest models/demos/cosyvoice/tests/ -k "not device"                    # host tier
pytest models/demos/cosyvoice/tests/pcc/ models/demos/cosyvoice/tests/e2e/ -v
pytest models/demos/cosyvoice/tests/perf/ -v -s                         # and again under each flag
```

### The boards

| property | Blackhole `p150a` | Blackhole `p150b` | Wormhole n300 |
|---|---|---|---|
| Form factor | single card | Quietbox, 4 cards, one used | T3000, 4 cards, one **chip** used |
| Cooling | active | passive | — |
| Compute grid | **13 × 10 = 130** | **13 × 10 = 130** | **8 × 8 = 64** |
| Device memory | 32 GB | 32 GB | 12 GB |
| Selection | — | `TT_VISIBLE_DEVICES=0` | `TT_VISIBLE_DEVICES=0` |

`p150a` and `p150b` are the same silicon and differ only in cooling; the passive board
runs a few per cent slower on identical work. That is the same order as several of the
optimisations in Part II, so the two stay separate columns and neither fills the
other's empty cells. PCC matches to ten digits across both.

The n300 result is one Wormhole B0 chip; nothing in this port is multi-chip
(`docs/VALIDATION.md` *The device matrix*).

### Test counts

| tier | count | needs | result |
|---|---:|---|---|
| host | 113 | nothing | 113 passed on all three boards |
| `pcc` + `e2e` | 150 | `/dev/tenstorrent` | 148 passed 2 skipped / 149 passed 1 skipped |
| `perf` | 14 | `/dev/tenstorrent` | 14 passed, on each of the three configurations, on all three boards |

The device tier re-runs the host tier (it lives in `tests/pcc/`), which is why 150 is
not 113 + 37. This run skipped end-to-end batched synthesis, which now runs.

## 2. The requirements, and the verdict on each

Thresholds are quoted from the bring-up scope and asserted through
`tests/perf/gates.py`: a met threshold against the requirement, so a regression fails;
an unmet one against a recorded band in both directions, so neither a regression nor a
stale published figure passes. Nothing is `xfail`-ed.

The band is a reference, not a copy of the figures below: the same board measures a
few per cent apart from day to day. The published figures must sit inside it; when one
does not, the run fails and both are updated together.

| requirement | target | `p150a` | `p150b` | n300 |
|---|---:|---:|---:|---:|
| Semantic token generation | `>= 30 tok/s` | `201.3 tok/s` ✅ | `192.1 tok/s` ✅ | `130.6 tok/s` ✅ |
| Semantic token generation (stretch) | `>= 60 tok/s` | `201.3 tok/s` ✅ | `192.1 tok/s` ✅ | `130.6 tok/s` ✅ |
| Real-time factor | `< 0.5` | `0.342` ✅ | `0.362` ✅ | `0.552` ❌ |
| Real-time factor (stretch) | `< 0.2` | `0.342` ❌ | `0.362` ❌ | `0.552` ❌ |

Best of each configuration; §3 breaks them out. `RTF < 0.5` is the only verdict that
differs by architecture, and the gap is the compute grid: 64 cores against 130.
`RTF < 0.2` has a floor rather than being simply unmet (§3.4).

## 3. End-to-end real-time factor

RTF is compute seconds per second of audio produced, measured on the captured
utterance: 164 generated tokens, 3.27 s of audio at 22 050 Hz.

The LLM runs once per token, and a second of speech is 50 tokens, so its contribution
is `50 / tok_s` whatever the utterance length. The flow decoder runs ten Euler steps
over the whole utterance at once and the vocoder runs once, so both get cheaper per
second of audio as utterances lengthen.

### 3.1 Per stage, at each board's default

| stage | `p150a` | `p150b` | n300 |
|---|---:|---:|---:|
| LLM (14-block AR decoder, traced, fused attention) | `5.72 ms/token × 164 = 0.938 s` · RTF `0.287` | `5.93 ms/token × 164 = 0.973 s` · RTF `0.297` | `7.85 ms/token × 164 = 1.288 s` · RTF `0.393` |
| Flow decoder (10 Euler steps, traced, fused SDPA) | `0.256 s` · RTF `0.078` | `0.285 s` · RTF `0.087` | `0.451 s` · RTF `0.138` |
| HiFT vocoder | `0.047 s` · RTF `0.014` | `0.059 s` · RTF `0.018` | `0.072 s` · RTF `0.022` |
| **Total** | **`1.241 s` · RTF `0.379`** | **`1.317 s` · RTF `0.402`** | **`1.811 s` · RTF `0.553`** |

The LLM is 76 % of an utterance at `p150a`'s default settings, the flow decoder 21 % and
the vocoder 4 %. That split is why Part II's optimisations target the decode step, why
batching (§4) targets only the LLM, and why `RTF < 0.2` is out of reach (§3.4).

### 3.2 Across the three configurations

| configuration | `p150a` | `p150b` | n300 |
|---|---:|---:|---:|
| default | `0.379` | `0.402` | `0.553` |
| `COSYVOICE_FF2_GRID=8x2` | `0.354` | `0.378` | `0.552` |
| `COSYVOICE_KV_INPLACE=1` | `0.342` | `0.362` | `0.564` |

The defaults differ by architecture: `kv_inplace_default` reads `device.arch()` and
turns the in-place KV cache on for Wormhole and off for Blackhole, because the trade
differs by part (Part II §1.4). The n300 default row is therefore the in-place cache,
and its explicit row measures the same configuration; the two would drift apart only if
the default stopped being read.

### 3.3 Semantic-token throughput

| configuration | `p150a` | `p150b` | n300 |
|---|---:|---:|---:|
| default | `174.8 tok/s` | `168.5 tok/s` | `127.3 tok/s` |
| `COSYVOICE_FF2_GRID=8x2` | `190.7 tok/s` | `184.0 tok/s` | `130.6 tok/s` |
| `COSYVOICE_KV_INPLACE=1` | `201.3 tok/s` | `192.1 tok/s` | `128.0 tok/s` |

Against a `>= 30 tok/s` requirement and a `>= 60 tok/s` stretch target. The untraced
control, measured in the same process, is in §6.

### 3.4 Why `RTF < 0.2` is not reachable here

`0.2` on this utterance is a budget of `0.654 s`. The flow decoder alone spends about
`0.256 s` of it with fused SDPA and the cross-utterance trace cache; its cost is 64
transformer blocks × 10 Euler steps, and halving the step count buys `1.43×` at a PCC
below every threshold here (Part II §2.2). The LLM's share would need the decode step
under `1.5 ms`, against a best measured `4.97 ms`; Part II §1.3 has what limits it.

The threshold is asserted against a recorded band, so an improvement fails the test
until this table moves with it.

## 4. Batched decode

At one row, every decode matmul multiplies a weight matrix by a single row, so each
weight read and each op serves one utterance. Batching makes one step serve `B` rows.
The step time grows; the figure that matters is the per-utterance cost.

Per-utterance decode cost, `max_len = 384`, mean of 32 steps, at each board's default:

| batch | `p150a` | `p150b` | n300 |
|---:|---:|---:|---:|
| **1** | `5.70 ms` (1.00×) | `6.02 ms` (1.00×) | `11.47 ms` (1.00×) |
| **2** | `4.58 ms` (1.24×) | `4.82 ms` (1.25×) | `8.64 ms` (1.33×) |
| **4** | `4.11 ms` (1.39×) | `4.32 ms` (1.39×) | `7.47 ms` (1.53×) |
| **8** | `3.77 ms` (1.51×) | `3.97 ms` (1.52×) | `6.96 ms` (1.65×) |

At `B = 8` the per-utterance cost is `1.51×`–`1.65×` lower on the same kernels, with
nearly the same ratio on both architectures.

It compounds with `COSYVOICE_FF2_GRID`, though both act on the same matmul: batching
amortises the weight read, and the grid flag changes how the reduction is split. With
the flag on, `p150a` reaches `3.17 ms` per utterance at `B = 8` (`1.66×` against that
configuration's own `B = 1`) and `p150b` reaches `3.39 ms` (`1.63×`).

`test_device_batched_decode_matches_single` checks correctness at ragged prefixes: four
sequences with prompt lengths `209, 177, 241, 193`, stepped together and alone, since an
equal-length batch cannot tell a correct per-row mask from a lucky one. Worst
hidden-state PCC: `0.9999998808` (p150a) / `0.9999998808` (p150b) / `0.9985343218`
(n300). It also asserts the deviation does not compound across steps, which separates
per-step rounding from a wrong mask or a mis-strided cache.

Only the LLM is batched. It runs once per token and is most of an utterance; the flow
decoder and the vocoder run once per utterance, and batching them would pad every
utterance to the longest mel, a cost the flow decoder pays in full because it is linear
in mel length.

## 5. Streaming

Two claims, checked separately.

Content: `test_device_streamed_matches_non_streamed` compares concatenated streamed
audio with non-streamed audio for the same tokens, in mel space and in the energy
envelope, plus continuity at chunk seams. Mel-space PCC: `0.901830` (p150a) /
`0.901541` (p150b) / `0.902374` (n300).

Schedule: `test_device_streaming_first_audio_latency` checks that audio starts before
generation finishes. Both schedules run in one process on one device over the same
tokens, with all three stages real: the AR decoder is prefilled from the captured
prefix and stepped for every token.

| | `p150a` | `p150b` | n300 |
|---|---:|---:|---:|
| LLM alone, all 164 steps | `0.956 s` | `1.025 s` | — |
| batch schedule, first audio = total | `1.569 s` | `1.744 s` | — |
| streaming, first audio | `1.313 s` | `1.493 s` | — |
| streaming, total | `2.139 s` | `2.464 s` | — |
| first-audio gain | **`1.19×`** | **`1.17×`** | — |
| cost of interleaving, on the total | `1.36×` | `1.41×` | — |

The n300 column is empty because this test hangs Wormhole (`docs/VALIDATION.md`). The
shipped path, `CosyVoiceTTNN.synthesize_streaming`, runs on n300, where
`test_device_streaming_generates_the_same_tokens_as_batch` passes, as does the content
check above.

Interleaving makes the total worse: one device, one command queue, no overlap of
compute, and a chunk's flow and vocoder work pauses token generation while it runs. What
it improves is when the first sample reaches the caller.

The gain is modest here because first audio is floored by one chunk —
`token_hop_len + token_overlap_len` = 120 tokens, 2.40 s of speech — and the utterance
is only 164 tokens. That floor is constant while the batch path's first audio is the
whole utterance, so the gap widens with length. That scaling is not measured
(`docs/VALIDATION.md` has why).

## 6. Trace capture, the KV cache, and weight dtype

All three are A/B'd inside the certification run, in one process.

| | `p150a` | `p150b` | n300 |
|---|---:|---:|---:|
| AR decode step, untraced (fixed-width cache) | `19.29 ms` | `23.60 ms` | `19.97 ms` |
| AR decode step, traced | `5.72 ms` | `6.03 ms` | `11.70 ms` |
| trace speedup | **`3.37×`** | **`3.92×`** | **`1.71×`** |
| moving KV cache, traced | `5.71 ms` | `6.20 ms` | `10.80 ms` |
| in-place KV cache, traced | `4.96 ms` | `5.39 ms` | `8.31 ms` |
| `bfloat16` weights, traced step | `5.48 ms` | `5.83 ms` | `8.35 ms` |
| `bfloat8_b` weights, traced step | `5.48 ms` | `5.62 ms` | `7.75 ms` |

Trace capture is the largest lever in the port and is bit-exact:
`test_device_traced_matches_untraced` checks PCC `1.0` before any of these timings
count. The in-place KV cache is second and is not bit-exact (worst PCC `0.9987` over 72
steps, non-accumulating), which is why it ships as an architecture-dependent default
rather than as the only mechanism.

`bfloat8_b` weights are a memory option, not a speed one: halving the weight width
moves the traced step by about a per cent either way across the three boards, inside
run-to-run noise. `COSYVOICE_WEIGHT_BF8` stays available for the 352 → 176 MB it saves.
Part II §1.3 has what limits the step instead.

### The growing KV cache, and why the fixed-width one exists

A cache that grows by one slot per token gives every decode step a new attention shape,
and TTNN's program cache is keyed on shape, so every token pays a JIT compile. On the
growing cache, in this run:

| | `p150a` | `p150b` | n300 |
|---|---:|---:|---:|
| cold pass, mean of 32 — what a real utterance gets | `41.51 ms` | `65.68 ms` | `61.46 ms` |
| warm pass, second time over the same sizes | `19.47 ms` | `23.20 ms` | `20.55 ms` |
| compile share of the cold pass | `53.1 %` | `64.7 %` | `66.6 %` |

The cold figure depends on what the machine's JIT cache already holds, so it is not
portable; the ratio is the point. `forward_chunk_fixed` holds the key width at
`max_len`, leaving two shapes for a whole utterance.

## 7. Accuracy

| module | PCC |
|---|---:|
| tokens → waveform (reference excitation) | `0.9951367159` |
| flow: tokens → mel | `0.9993962895` |
| whole HiFT vocoder | `0.9996373743` |
| LLM AR prefill, 209 tokens | `0.9997530373` |
| LLM AR decode step | `0.9989617190` |
| traced vs untraced decode | `1.0000000000` (bit-exact) |
| iSTFT vs captured golden | `0.9999298811` |
| batched vs single-row decode | `0.9999998808` (p150a) / `0.9999998808` (p150b) / `0.9985343218` (n300) |

| check | value | target |
|---|---:|---:|
| Token agreement, teacher-forced | `99.04 %` | `> 95 %` ✅ |
| Token agreement, through the KV cache | `100.00 %` | `> 95 %` ✅ |
| Streaming vs batch generation, greedy | `100.00 %` | — |
| WER (English) | `0.00 %` | `< 3.0` ✅ |
| Speaker similarity (mean, 10 utterances) | `83–96` | `> 60` ✅ |

WER and speaker similarity come from `scripts/eval_wer_sim.py` in the reference venv —
whisper `large-v3` and `WavLMForXVector`, neither of which tt-metal's `python_env`
carries — so they are the two figures here that no test in this tree asserts.

The waveform-PCC check injects the reference excitation. f0 error integrates into phase
drift, and holding drift under a tenth of a cycle across 72 192 samples needs mean f0
error below `0.03 Hz`, tighter than Tensix HiFi4 delivers. That is a property of the
model; with a self-computed excitation the energy envelope (`0.9975`) and RMS (within
6 %) still hold.

## 8. Generation modes and speech quality

All four modes run on device across five languages: 20 cases, all synthesising.

| mode | prompt | on device |
|---|---|---|
| zero-shot | reference audio | ✅ 5/5 |
| cross-lingual | reference audio, different language | ✅ 5/5 |
| SFT | speaker id, no prompt audio | ✅ 5/5 |
| instruct | speaker id + description, no prompt audio | ✅ 5/5 |

Scored with whisper `large-v3`; CER for CJK, WER for English.

| mode | zh | en | ja | ko | yue |
|---|---:|---:|---:|---:|---:|
| zero-shot | `3.03` | `0.00` | `5.56` | `3.12` | `64.52` |
| cross-lingual | `6.06` | `0.00` | `2.78` | `0.00` | `100.00` |

Cantonese is a model limitation, not a port defect: the PyTorch reference scores worse
on the same text through the same ASR, `83.87 %` against this port's `64.52 %`
zero-shot. Excluding Cantonese, the CJK mean is `3.90 %` zero-shot and `2.95 %`
cross-lingual. `docs/VALIDATION.md` records the open `cross_lingual yue` early stop.

## 9. Tuning flags

Every default is measured. Defaults are read from the code, not from this document;
each row names where its figure lives.

| flag | default | what it does | what it is worth |
|---|---|---|---|
| `COSYVOICE_KV_INPLACE` | follows `device.arch()` — on for Wormhole, off for Blackhole | writes the KV cache with `ttnn.update_cache` instead of rebuilding it | §3.2, §6 |
| `COSYVOICE_FF2_GRID` | unset | explicit core grid for the FFN's second linear at decode (`T == 1` only) | §3.2; Part II §4.2 |
| `COSYVOICE_SDPA_DECODE` | `1` | fused `sdpa_decode` for the AR decoder's relative-position attention | Part II §1.1 |
| `COSYVOICE_SDPA` | `1` | fused SDPA in the flow estimator | Part II §2.1 |
| `COSYVOICE_CFM_TRACE_CACHE` | `1` | keeps the CFM estimator trace across utterances of the same mel length | Part II §2.2 |
| `COSYVOICE_GN_PERMUTE` | unset (matmul form) | restores the permute-based GroupNorm | Part II §2.3 |
| `COSYVOICE_FLOW_STEPS` | `10` | Euler solver depth | Part II §2.2 |
| `COSYVOICE_FIDELITY` | `HiFi4` | math fidelity for the matmuls | §7 |
| `COSYVOICE_HIFT_TRACE` | unset (off unless a caller opts in with `enable_trace()`) | forces vocoder trace capture on or off | `TtHiFTGenerator.enable_trace` |
| `COSYVOICE_WEIGHT_BF8` | `0` | `bfloat8_b` decoder linear weights | §6 — a memory option, not a speed one |
| `COSYVOICE_FLOW_BF8` | `0` | `bfloat8_b` flow-estimator weights | carries its own measurement |
| `COSYVOICE_FP32_ACC` | `1` | fp32 accumulation in the vocoder convolutions | Part II §3.2 |
| `COSYVOICE_CONV_PREPARE` | unset (per-geometry verification) | overrides the prepared-weight verdict | Part II §3.2 |
| `COSYVOICE_INPUTS` | unset | where the prompt `.npz` files live, for the two API tests | `docs/VALIDATION.md` |

Two flags are opt-in because the best setting is not portable: `COSYVOICE_FF2_GRID`, and
`COSYVOICE_KV_INPLACE` on Blackhole.

## 10. Known limitations

In `docs/VALIDATION.md`: the unmet requirements, the open defects, and the workarounds
the tree carries for TTNN behaviour.

---

# Part II — the engineering record

A/B measurements behind the design. Each compares against something no longer in the
tree, so each carries the board and the date it was measured on. A/Bs the suite
re-measures on every run — traced against untraced, the two KV-cache mechanisms,
`bfloat8_b` against `bfloat16` — are in Part I §6 instead.

## What made the difference, ranked

| change | effect |
|---|---|
| fixed-width KV cache | **73×** on the first pass — a growing cache gives every token a new shape and so a fresh JIT compile; 98.9 % of the cold cost was compilation. §1.5. |
| trace capture, both compute stages | `3.37×` (`p150a`) / `3.92×` (`p150b`) / `1.71×` (n300) on the decode step. Part I §6. |
| fused decode attention (`sdpa_decode` with the rel-pos term as its mask) | `−17.1 %` on the step; `3.3×`–`3.4×` on the attention block alone. §1.1. |
| hoisting `linear_pos` out of the decode step | `15.71 → 8.25 ms`. §1.2. |
| in-place KV via `ttnn.update_cache` | `1.30×` on n300, `1.15×` on Blackhole — the one change that pays more on the slower part. Part I §6. |
| cached CFM trace | `2.37×` on the solver on `p150b`, `1.67×` on n300. §2.2. |
| fused SDPA in the flow estimator | faster and more accurate on every check — `0.707 → 0.600 s`. §2.1. |
| `rel_shift` → one slice at `T = 1` | seven ops become one. §1.2. |
| QKV fusion | flow decoder `1.075 → 0.719 s`; the same change on the AR decoder was a wash, since the split op's data movement costs what it saves at `T = 1`. §1.2. |

## 1. The AR decode step

The largest stage, and the one every remaining lever sits in.

### 1.1 The relative-position attention is a fused kernel

ESPnet relative-position attention decomposes into `(q + u)K^T + (q + v)P^T`. At
`T = 1` the second term is a `[B, h, 1, W]` vector over the key axis — an additive
bias, which is what `ttnn.transformer.scaled_dot_product_attention_decode` accepts as
`attn_mask` with `is_causal=False`. The score matmul, the bias add, the masked softmax
and the context matmul collapse into one kernel that never materialises the score
matrix.

The bias goes in unscaled: `sdpa_decode` adds the mask to the raw `q·kᵀ` and applies
`1/sqrt(d_k)` afterwards (`sdpa_flash_decode.cpp`), which matches the reference's
`(ac + bd) / sqrt(d_k)`. Under torch's SDPA order — scale, then add — the bias would
need pre-dividing, and the raw term would give attention wrong by a factor of 8 on this
model. `tests/pcc/test_rel_pos_attention.py` checks that identity both ways, and
`test_device_fused_attention_matches_explicit` checks the device path.

| measured | `p150b`, 2026-08-06 |
|---|---|
| attention block, key width 384 | `1.563 ms` explicit → `0.460 ms` fused (**3.4×**) |
| attention block, key width 448 | `1.817 ms` explicit → `0.557 ms` fused (**3.3×**) |
| whole decode step | `6.73 → 5.58 ms` (`148.5 → 179.2 tok/s`) |
| end-to-end RTF | `0.533 → 0.477` |

The fused path matches the explicit chain at PCC `0.9988`–`0.9999`, and exact-token
agreement through the KV cache rises from `95.83 %` to `100.00 %`.
`COSYVOICE_SDPA_DECODE=0` restores the explicit chain, and
`test_device_fused_attention_matches_explicit` keeps the two comparable.

### 1.2 Token-independent work, hoisted out of the traced step

`linear_pos(pos_emb)` projects `2·max_len − 1` rows through `[d_model, d_model]` —
about 536 MFLOP at `max_len = 256`, against roughly 1 MFLOP each for q, k and v — and
depends only on `max_len`, so it runs outside the traced step rather than on every
token. With the head-split transpose also hoisted out (most of the gain), `rel_shift`
reduced to a single slice at `T = 1`, and `transpose_b` and `scale_mask_softmax` folded
into their matmuls:

| measured | `p150b`, 2026-08-06 |
|---|---|
| decode step | `15.71 → 8.25 ms` |
| throughput | `63.6 → 121.3 tok/s` |

QKV fusion is stage-dependent. It helps the flow decoder (`1.075 → 0.719 s` at
`T ≈ 600`, batch 2) and is a wash on the AR decode step (`8.29 → 8.31 ms` at `T = 1`,
where splitting back into heads costs about what the fused matmul saves).

### 1.3 What limits the step

* Not weight bandwidth. `bfloat8_b` weights measure `1.00×` at two different effective
  bandwidths, so `COSYVOICE_WEIGHT_BF8` ships as a memory option (352 → 176 MB).
* Not the four linears. They are 34 % of the step (`2.82 ms` across 14 layers) and near
  TTNN's default grid optimum.
* A per-op dispatch floor of ~6.3 µs, flat in tensor size, across the ~280 non-linear
  ops that make up the rest: about `2.1 ms` of an `8.25 ms` step. (`p150b`, 2026-08-06.)

That floor is why batching (Part I §4) still pays: it does not make the step cheaper,
it makes one step serve more utterances.

### 1.4 KV-cache layout: tile alignment, not bandwidth

`slice` + `concat` on a `[1, 16, 256, 64]` cache cost ~`228 µs` against `19–64 µs` for
every other non-linear op: 0.5 MB moved in 134 µs, about 3.7 GB/s, two orders below
what the byte count implies. Two measurements locate it:

* slicing at a tile-aligned row is `11–16×` cheaper than at row 1
  (`78.3 → 7.0 µs`, `207.4 → 13.1 µs`);
* `bfloat8_b`, half the bytes, is identical to the last decimal.

A layout cost, not a bandwidth one. `TILE_LAYOUT` tiles the last two dimensions, so
`[1, h, T, d_k]` puts time on a tiled axis and appending one row re-tiles the buffer.
Time-major `[1, T, h, d_k]` puts it on a free one:

| measured, `p150b`, 2026-08-06 | |
|---|---|
| slice + concat | `207.2 → 19.7 µs` |
| permute back for the matmuls | `+13.9 µs` |
| traced decode step | `8.26 → 6.73 ms` (`121.4 → 148.5 tok/s`) |
| trace speedup | `2.54× → 3.10×` |
| end-to-end RTF | `0.610 → 0.533` |

Bit-exact against untraced. The in-place `ttnn.update_cache` write is faster still
(`3.7 µs`, 56×) and is the second mechanism, measured in Part I, but it needs 65
captured traces where this needs one; both ship, and `kv_inplace_default` picks by
architecture.

### 1.5 The fixed-width cache — 73× on the first real pass

A growing cache gives every decode step a new attention shape, and TTNN's program cache
is keyed on shape, so every token pays a JIT compile:

| `p150b`, 2026-08-06 | mean/step | tok/s |
|---|---:|---:|
| growing cache, cold — what a real utterance gets | `2595.34 ms` | `0.4` |
| growing cache, warm — second pass over the same sizes | `28.32 ms` | `35.3` |
| **fixed-width cache, first pass** | **`34.10 ms`** | **`29.3`** |

98.9 % of the growing-cache cold cost was compilation. `forward_chunk_fixed` holds the
key width at `max_len` (rounded to a multiple of 128), leaving two shapes for a whole
utterance, with the live tokens at the end of the buffer because ESPnet's `rel_shift`
assumes the queries are the last of the key positions.
`test_device_fixed_shape_cache_matches_the_growing_one` guards the equivalence, and the
right-alignment lets a ragged batch be one `valid` per mask row (Part I §4).

### 1.6 The per-token tail outside the trace

`0.352 ms`, 2.7 % of a token (`p150b`, 2026-08-06): output-head matmul `0.043`, logits
device→host `0.142`, RAS sampling on host `0.075`, embedding row→device `0.092`.
`ttnn.sampling` could remove at most `0.217 ms` of that, 1.7 % of a token, and would
give up exact agreement with the reference's sampler, so sampling stays on the host.
`nucleus_filter` is optimised instead: `0.245 → 0.075 ms`, bit-identical.

## 2. The flow decoder

### 2.1 Flash attention

The estimator's self-attention has no mask and no relative-position term, so
`ttnn.transformer.scaled_dot_product_attention` is a drop-in.

| `p150b`, 2026-08-06 | explicit chain | fused SDPA |
|---|---:|---:|
| flow decoder | `0.707 s` | **`0.600 s`** |
| end-to-end RTF | `0.647` | **`0.611`** |
| `solve_euler` PCC | `0.9992047752` | **`0.9993701398`** |
| flow tokens → mel PCC | `0.9992029011` | **`0.9993962895`** |
| CFM estimator, first / last step | `0.9998326979` / `0.9991904460` | **`0.9998480374` / `0.9994887951`** |

Faster and more accurate on every check. `scale=1.0` because `1/sqrt(d_head)` is folded
into the fused QKV weight's q half. `COSYVOICE_SDPA=0` restores the explicit chain.

### 2.2 Trace-cache reuse across utterances

The stage is not linear in solver depth: `T(n) ≈ 0.350 s + 35.8 ms/step`. Halving the
10-step solver buys `1.43×`, not `2×`, at PCC `0.9825`, below every threshold here, so
`COSYVOICE_FLOW_STEPS` exists and is unused by default.

The fixed `0.350 s` is trace capture: 46.6 % of the solve, against 52.9 % for the
replay. Keeping the trace across utterances of the same mel length is worth `1.67×` on
the solver (`0.601 → 0.359 s` steady state) and takes n300 end-to-end RTF from `0.736`
to `0.628` (n300, 2026-08-06). It is correct across utterances with different
conditioning, because the trace bakes a buffer address that is refilled in place: PCC
`1.0000000000` over three consecutive solves. `COSYVOICE_CFM_TRACE_CACHE=0` turns it
off, which `synthesize_batch` requires (`docs/VALIDATION.md`).

### 2.3 GroupNorm as a matmul

In a traced, per-block-class profile — untraced timings are dominated by host dispatch
and can invert the ranking — the permute-based GroupNorm costs about 7× the convolution
beside it (`0.2197` / `0.3809 ms` against conv1d's `0.0320` / `0.0556 ms` on `p150b` /
n300, one resnet block at `T = 141`). 33 GroupNorms run per Euler step, roughly 36 % of
the estimator.

The permute re-tiles under `TILE_LAYOUT`. Computing the channel sum as a matmul against
a `[C, G]` indicator avoids that:

| 2026-08-18 | `p150b` | n300 | PCC vs torch |
|---|---:|---:|---:|
| `[2, 141, 256]`, permute → matmul | `0.2202 → 0.1012` (**2.18×**) | `0.3820 → 0.1874` (**2.04×**) | `0.999988854` |
| `[2, 282, 256]`, permute → matmul | `0.3993 → 0.1056` (**3.78×**) | `0.6691 → 0.2045` (**3.27×**) | `0.999992251` |

`1.41×` on the whole stage on `p150b`, `1.34×` on n300. `COSYVOICE_GN_PERMUTE=1`
restores the permute form. The matmul form needs its variance clamped
(`docs/VALIDATION.md`); the clamp costs 2–8 % and no PCC.

## 3. The vocoder

The cheapest stage.

| op | shape | latency (`p150b`, 2026-08-06) |
|---|---|---:|
| iSTFT | 18 049 frames → 72 192 samples (3.27 s of audio) | **`1.115 ms`** |
| iSTFT | 1 024 frames → 4 092 samples | `0.853 ms` |
| `ConvTranspose1d` | 512→256, k=16, s=8, L=282 (`ups[0]`) | **`3.886 ms`** |

The inverse transform is a matmul — fixed window and hop make it an exchange matrix
plus overlap-add — and contributes RTF `0.00034`. The `conv2d`-at-`H=1` op standing in
for an absent `ttnn.conv_transpose1d` dominates the stage instead, at 3.5× the whole
iSTFT per upsample layer, of which there are two.

### 3.1 `phase_mod1` against `ttnn.cumsum`

On this branch's tt-metal, `ttnn.cumsum` is 2000× less accurate than torch's over the
real 72 192-sample f0 scan (`max|d| 5.62e-01` against `2.44e-04`, both against an fp64
reference). Phase is `2π · (cumsum mod 1)`, so `0.56` absolute is over half a cycle.
`phase_mod1()` reduces each block mod 1 before accumulating: PCC `0.843 → 0.99999745`,
and faster, because the plain scan runs serially on one core.

| `cumsum` + `mod 1`, 2026-08-13 | `p150b` | n300 |
|---|---:|---:|
| plain, one core | `40.4 ms` | `73.3 ms` |
| `phase_mod1` | `5.9 ms` | `12.5 ms` |

`docs/VALIDATION.md` has the upstream status.

### 3.2 Per-geometry verification of prepared conv weights on Wormhole

Disabling `ttnn.conv1d` weight preparation to avoid the Wormhole defect
(`docs/VALIDATION.md`) costs the flow stage `0.683 → 1.723 s` on n300, since the same
`TtConv1d` backs the estimator's traced convolutions. Verifying each `(length, batch)`
geometry once instead is free at the utterance level. With it, the vocoder runs in
`0.077 s` against `0.084 s` and the streamed-vs-non-streamed mel PCC on n300 is `0.9024`
against `0.218`, matching Blackhole's `0.9019`. Reproducer:
`scripts/repro_conv1d_wormhole.py`, no model involved.

## 4. Measured and not shipped

### 4.1 Multi-chip tensor parallelism

A two-chip Megatron-sharded decoder on an n300 pair: `1.18×` on the decode step alone,
PCC `0.99994` (2026-08-14). It does not compound with `COSYVOICE_FF2_GRID`: tensor
parallelism halves the FFN's second linear to `K = 2048`, and the core-grid win that is
`1.50×`–`2.11×` at `K = 4096` falls to about `1.03×` there.

### 4.2 Explicit core grids, almost everywhere

Explicit `core_grid` lost to TTNN's default in 10 of the 12 combinations swept. The
exception is the FFN's second linear at decode, and it wins by being smaller: at
`M = 1` a 4096-deep reduction spread over the whole grid leaves each core a sliver and
the gather dominates. `8x2`, sixteen cores, measures `1.98×` the default on `p150b` and
`1.50×` on n300 (2026-08-17).

It ships as `COSYVOICE_FF2_GRID` rather than as a default because the best shape is not
portable: `4x8`, the same core count transposed, manages only `1.15×` on n300.
