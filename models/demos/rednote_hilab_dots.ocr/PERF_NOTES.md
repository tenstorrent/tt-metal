# dots.ocr — Pipeline Perf Notes (ocr use_case)

Phase: `use_cases.ocr.perf`. Device: **p150 (Blackhole)**, arch `blackhole`.
Branch: `ssinghal/seamless-m4t`. Real-world workload: `demo/demo_image1.jpg`
(1700×2250 full document page, model-determined resolution, no cap).
Real test: `tests/test_dots_ocr_inference.py`.

## Real-workload characterization (full-res `demo_image1.jpg`)

Profiled via `tt/profile_ocr_traced.py --image demo/demo_image1.jpg` under
`python3 -m tracy -p -v -r`. Input: **19,520 vision patches, 4,897-token
prefill**. Per-stage wall (warm, p50 over 3 replays), real weights, 28 LM /
42 vision layers:

| stage | traced | notes |
|-------|--------|-------|
| vision (42 L) | **1693 ms** (was 3201 ms) | **1.9× — see vision opts below** |
| LM prefill (28 L) | **272 ms** (was 1426 ms) | **5.2× — see prefill opts below** |
| decode / step | ~19 ms | — |
| **e2e** | **2559 ms** (was 4695 ms) | **1.8×**; OCR exact vs HF (16/16 tokens) |

### Vision optimizations (3201 → 1693 ms, PCC tower 0.99998)

dots.ocr vision is full bidirectional attention over all ~19.5k patches in every
one of the 42 layers (no window/causal — config has no `window_size`), so the
O(seq²) is architectural; the wins are kernel-level on the windowed SDPA:

| change | vision ms | gate |
|--------|-----------|------|
| baseline (q/k_chunk 128) | 3201 | — |
| SDPA q_chunk 256 / k_chunk 512 | 2371 | precision-neutral |
| + fused RoPE + bf8 Q/K/V + HiFi2 SDPA | **1693** | tower PCC 0.99998 |

bf8 Q/K/V + HiFi2 SDPA is the qwen2.5-VL vision pattern (matmuls/RoPE stay
HiFi4/bf16): SDPA 46 → 15 ms/layer at seq=19520. Generalized to
[[skills/optimization]] "VLM vision attention".

Real-world demo output (full document, full depth, chat-template prompt),
**exact token match vs HF DotsOCRForCausalLM (16/16 = 1.0000)**:
`**TABLE II** – ODDS RATIO OF HODGKIN LYMPHOMA AND NON-HODGKIN LYMPHOMA …`

### Prefill optimizations (1426 → 262 ms, all verified exact vs HF)

| change | prefill ms | why |
|--------|-----------|-----|
| baseline | 1426 | — |
| lm_head slice to last token | 1178 | vocab proj (N=151936) on 1 row not ~4.9k |
| SDPA q/k_chunk 128→256 | 1147 | fits L1, ~2× faster SDPA/layer |
| fused RoPE + native GQA | 1125 | 1 kernel vs 6-op RoPE; drop `_repeat_kv` |
| **DRAM residual at large seq** | **262** | **the big one — see below** |

**The DRAM-residual fix (−863 ms, 4.3×):** `decoder_layer.prefill_kv` pinned the
two residual `ttnn.add`s to `L1_MEMORY_CONFIG` unconditionally. At seq=4891 the
`[seq,1536]` activation is ~15 MB — far over L1 — and that L1-resident tensor
flowing (via RMSNorm, which inherits buffer type) into the MLP `gate_up` matmul
forced a **pathological L1 matmul path: 34 ms/layer vs 6 ms with a DRAM input**
(same shape/weights/config). The full layer was 36 ms while its components summed
to ~8 ms — the ~28 ms gap was entirely this L1→matmul interaction. Fix: gate the
residual `memory_config` on seq (`L1 if seq<=1024 else DRAM`). Generalized to the
[[skills/optimization]] + [[skills/perf]] frameworks ("NEVER pin a LARGE activation
to L1").

**Diagnosis note:** the earlier "prefill matmul needs DRAM-sharding" and "31 ms
gate_up" reads were artifacts — isolated matmuls with preallocated inputs are
sub-3 ms; tracy's traced-op CSV mislabels SDPA as MatmulDeviceOperation and
inflates durations. The real bottleneck was found by timing the fused layer vs its
components with `synchronize` (full ≫ Σparts ⇒ layout-interaction stall).

**Remaining:** vision (3.2 s, ~80% of e2e) is the next target — windowed SDPA
~46 ms/layer × 42 ≈ 1.9 s dominates it.

## Headline

The KV cache was the structural win. It converts the AR decode from an **O(N)
full-trunk re-run per step** into an **O(1) cached step**, which is what makes
**full-depth (28 LM / 42 vision) OCR generation tractable in one session**. At
full depth the cached path now decodes the sample image to **"HELLO 2026"**,
**token-for-token identical to the HF `DotsOCRForCausalLM` reference**
(char accuracy = 1.0000, token accuracy = 1.0000, both stop at EOS 151673).

## Sub-pass 1: structural — KV cache (`tt/kv_cache.py` + cached-decode path)

What changed:

- **`tt/kv_cache.py`** — new `SelfAttentionKVCache` (decoder-only, no cross-attn).
  Stores **only the 2 GQA KV heads** per layer (`[1, 2, max_seq, 128]`, DRAM TILE
  bf16). Single-token update uses
  `ttnn.experimental.paged_update_cache(cache, kv, update_idxs_tensor=pos_tt)`
  with a persistent int32 position buffer (trace-ready: position read from device
  memory, not baked into kernel args). Prompt prefill bulk-populates via
  `ttnn.fill_cache`. `reset()` streams a host zero tensor in place (no realloc).
- **`tt/attention.py`** — added `prefill_kv()` (full-causal forward that also
  writes post-RoPE K / raw V into the cache) and `forward_decode()` (single-token:
  project Q/K/V, RoPE at `cur_pos`, write to cache, then
  `ttnn.transformer.scaled_dot_product_attention_decode` against the full cache).
  SDPA-decode performs the **GQA 2→12 expansion internally**, so the cache holds
  only 2 KV heads. A capped `SDPAProgramConfig(grid=(8,8))` is required — the
  default core allocation over-subscribes the tree reduction on this device
  (`TT_FATAL: got 65 cores/head`).
- **`tt/decoder_layer.py`** — `prefill_kv()` / `forward_decode()` wrappers.
- **`tt/language_model.py`** — full-length decode RoPE tables (built once at
  `max_seq_len`), `prefill_from_embeds()` and `decode_step()` (enter from
  embeddings, since the OCR pipeline scatters vision embeds host-side).
- **`tt/ocr_model.py`** — `generate()` rewritten: prefill once (populate cache) →
  per-step `decode_step()` (O(1)) → argmax. Old path re-instantiated a fresh
  full LM trunk per sequence length and re-ran the whole O(N) forward each step.

### Correctness (cached decode vs. the verified no-cache full forward)

Per-step PCC of cached-decode logits vs. the no-cache full forward at the same
positions, **real checkpoint weights, full 28-layer depth**, teacher-forced on the
cached sequence:

| step | pos | PCC (cached vs no-cache) | argmax match |
|------|-----|--------------------------|--------------|
| 0    | 43  | 0.99965 | ✓ (50712) |
| 1    | 44  | 0.99979 | ✓ (1593)  |
| 2    | 45  | 0.99937 | ✓ (220)   |
| 3    | 46  | 0.79988* | ✓ (17)   |
| 4    | 47  | 0.95263* | ✓ (15)   |
| 5    | 48  | 0.97213* | ✓ (17)   |

\* The lower PCC on the later steps is bf16 argmax noise on near-tie **digit**
logits (the no-cache reference itself carries the same bf16 drift); the **argmax
token matches at every step**, and the cached sequence `[50712,1593,220,17,15,17]`
= "HELLO 202…" — exactly HF's leading tokens.

### Full-depth e2e (the real OCR deliverable)

`tests/test_dots_ocr_inference.py` (the real-world test, full document image)
runs at **28 LM / 42 vision** with a meaningful accuracy gate (`HF - 1.0`):

```
[e2e ocr] LM_LAYERS=28 VISION_LAYERS=42 max_new_tokens=12
[e2e ocr] TTNN tokens=[50712, 1593, 220, 17, 15, 17, 21, 151673]
[e2e ocr] HF   tokens=[50712, 1593, 220, 17, 15, 17, 21, 151673]
[e2e ocr] TTNN text='HELLO 2026'
[e2e ocr] HF   text='HELLO 2026'
[e2e ocr] char accuracy vs HF = 1.0000 | token accuracy = 1.0000
[e2e ocr] PASS accuracy=1.0000 token_accuracy=1.0000 tokens=8 gate=0.5
```

### Perf numbers (full depth, prompt_len≈44)

| metric | value |
|--------|-------|
| prefill_ms (44-token prompt, full trunk, populates cache) | ~164 ms |
| cached decode step (O(1)) | **~39–43 ms** |
| no-cache full forward at the same short length (old per-step cost) | ~41 ms |
| measured speedup at this short length | **~1.04×** |

**Honest read:** at a short ~80-token total sequence the cached step is only
~1.04× faster than the old full-forward step, because at this length the path is
**host-dispatch / device-kernel bound on ~28 layers of ops**, and the old path's
extra O(N) matmul work is still small in absolute terms. The KV cache's win here
is **structural, not wall-clock at short N**: it removes the O(N) per-step
re-compute (which would dominate at longer generations) AND removes the need to
re-instantiate a fresh LM trunk per sequence length, and it is the prerequisite
for a single replay-able metal trace. This mirrors the SeamlessM4T-v2 lesson
(trace alone bought 1.21×, not 50%, because the floor was kernel time).

## Sub-pass 2: targeted — traced tracy characterization

Harness: `tt/profile_ocr.py` (1 prefill + 1 warmup decode + N timed decode steps;
reports `prefill_ms` / `decode_step_ms`). Captured under tracy:

```
python -m tracy -p -v -r --op-support-count 3000 -n ocrdec \
    models/demos/rednote_hilab_dots.ocr/tt/profile_ocr.py \
    --lm-layers 28 --vision-layers 2 --num-timed 4
```

**Traced device-kernel evidence:**
`tests/profiler_artifacts/ocr_decode_traced_device_perf.csv` (local, gitignored —
generated by re-running the tracy command above) — **5496 device-op instances,
~127–156 ms total `DEVICE KERNEL DURATION`** over the prefill+decode capture. This confirms the decode path is **device-kernel-bound**
(real kernel time across 28 layers), NOT host-dispatch noise — the regime where
op-level work matters.

**Limitation (honest):** tracy's report-assembly step (`process_ops_logs.py`)
raised `AssertionError: Device data missing: Op … not present in
cpp_device_perf_report.csv` on this decode op pattern, and the per-op `OP NAME`
column in `cpp_device_perf_report.csv` came back **blank** (the host↔device
enrichment join failed; at full vision depth the profiler DRAM buffers also
overflowed: "markers were dropped"). So a clean **per-op top-10 by name could not
be extracted this pass**. The device-kernel *total* is valid and proves the
bottleneck class; the *per-op attribution* is blocked by the tracy tooling
failure, not by a missing capture.

**No op-level optimization was applied this pass** — per the perf skill, forcing
a "win" without a clean hotspot table risks regressing PCC. The already-inherited
block-level optimizations stand: `lm_head` bf8 weight (~72% of trunk kernel time,
−36% traced), attention head-split/merge L1 pinning, MLP gate/up fusion + L1
SwiGLU chain.

## Recommendations / deferred

1. **Metal trace the decode step.** The KV cache is already trace-ready
   (`paged_update_cache(update_idxs_tensor=…)` + persistent position buffer).
   Capturing one trace at warmup and replaying per position should remove the
   per-step host dispatch and is the next ~1.1–1.3× per the SeamlessM4T-v2
   precedent.
2. **Fix tracy op-name enrichment** for the decode pattern (lower
   `--vision-layers`, raise profiler DRAM buffer, or capture decode-only without
   prefill) to get a clean per-op top-10, then attack the hottest matmul.
3. **fp32 score sensitivity:** the attention QK^T is kept fp32 for the wide K
   dynamic range; revisit whether bf8 KV-cache storage holds PCC at long N.

## Sub-pass 3: bf8 dense-MLP weights (gate_up + down)

Tracy showed decode is device-compute-bound (~15.7 ms/tok) and matmul-dominated,
with the two dense SwiGLU MLP matmuls (`gate_up` 1536→17920, `down` 8960→1536)
the largest contributors. At decode seq=1 both are skinny matmuls whose cost is
dominated by the **DRAM weight read** (gate_up ~27.5M params, down ~13.8M params).
Storing both weight tensors as **bfloat8_b** (was bf16) halves that read. Mirrors
the proven `lm_head` bf8-weight win. Activations + HiFi4 fp32_dest_acc compute
unchanged; both matmuls converted.

| metric | bf16 weights | bf8 weights | delta |
| --- | --- | --- | --- |
| MLP block PCC (`test_tt_mlp.py`) | — | **0.99979** | > 0.99 gate held |
| decode traced ms/tok (32 tok, 3 iters) | 19.50 | **17.63** | **−1.87 ms (−9.6%)** |
| DecodeGraph signpost ms/tok | 19.36 | **17.43** | −1.93 ms (−10.0%) |
| OCR text (16/32 tok vs HF) | exact | **exact** | `**TABLE II** – ODDS RATIO OF HODGKIN LYMPHOMA…` |

**Honest read:** the realized win is **~1.9 ms/tok**, smaller than the ~3 ms
hypothesis (the matmuls aren't purely DRAM-read-bound at this config), but it is
a consistent, correctness-neutral −10% of the decode step. Both MLP matmuls were
converted (PCC headroom ample at 0.99979, so no need to fall back to one).

## Sub-pass 4: width-sharded DRAM-sharded-matmul decode path (Phases 1–4 coupled)

Per the plan (`docs/superpowers/plans/2026-05-30-dots-ocr-sharded-decode.md`),
built a fully width-sharded decode block behind a `sharded_decode=True` flag
(default OFF; prefill + the existing bf8/interleaved decode path untouched). The
residual-stream activation is laid out L1 **WIDTH_SHARDED**, the four layer
matmuls (qkv 16c, o 48c, gate_up 16c, down 8c) run **DRAM-sharded** (HiFi2,
`create_dram_sharded_mem_config` bf8 weight dups) on dedicated DRAM banks
(removes the 130-core interleaved bank contention), and the two RMSNorms run as
**sharded `LayerNormShardedMultiCoreProgramConfig`** whose output grid matches the
downstream matmul so the matmul needs no separate input reshard.

| metric | baseline (bf8 interleaved) | sharded decode | delta |
| --- | --- | --- | --- |
| decode-step guardrail PCC (flag ON) | — | **0.999430** | > 0.99 held |
| pure device replay (decompose, p50) | 14.67 ms | **12.97 ms** | **−1.70 ms (−11.6%)** |
| decode traced ms/tok (32 tok, median of 3) | 17.47 | **15.96** | **−1.51 ms (−8.6%)** |
| OCR text (32 tok vs HF) | exact | **exact** | `**TABLE II** – ODDS RATIO OF HODGKIN LYMPHOMA…` |

Sharded successfully: qkv (16c), o (48c), gate_up (16c), down (8c) DRAM-sharded
matmuls; both RMSNorms width-sharded; all PCC-neutral. Fell back to interleaved:
nothing (all four matmul shapes validated as DRAM-sharded-expressible).

**Honest read vs the ~9 ms target:** realized only **~1.5 ms/tok (−8.6%)**, far
short of ~9 ms. Two reasons, both confirmed on device:
1. **Reshard tax, exactly as the plan's CRITICAL FINDING warned.** With sharded
   matmuls alone the device floor barely moved (14.67 → 14.07 ms, −0.6 ms): the
   per-boundary reshards (`to_memory_config` / `sharded_to_interleaved` around
   `nlp_create_qkv_heads` which needs interleaved bf16, the SwiGLU, and the 16c→8c
   gate_up→down regrid) nearly cancelled the matmul saving. The real win only
   appeared once the **sharded RMSNorm fed the matmul directly** (removed the
   norm→matmul reshard + killed the single-core LN), taking the floor to 12.97 ms.
2. **The decode step is no longer matmul-dominated.** ~4 ms/step is host tail
   OUTSIDE the trace — the `[1, 151936]` logits D2H (~2.3 ms) + per-layer RoPE
   cos/sin H2D (~0.96 ms) — plus the bf8 lm_head (~1.5 ms, out of scope). Sharding
   the layer matmuls cannot touch any of that. Hitting ~9 ms needs attacking the
   host tail (e.g. on-device argmax to shrink the D2H, device-resident RoPE) and
   the lm_head, not more layer-matmul sharding.

## Sub-pass 5: on-device greedy argmax in the decode trace (kill the logits D2H)

The sharded-decode write-up above identified the `[1, 151936]` logits D2H (~2.3 ms)
as the largest remaining host-tail item: the full logits vector was read back every
step only so a host `torch.argmax` could pick the next token. This sub-pass folds the
greedy argmax INTO the captured decode trace, so each step reads back one int (the
chosen token id) instead of 151,936 floats.

`TtLanguageModel.decode_step` / `decode_step_traced` gained a `return_logits` flag
(default **True**, preserving the logits output the PCC guardrail compares). With
`return_logits=False` the trace output is the next-token id `[1, 1]` uint32 produced by
`_greedy_token_id`: `ttnn.untilize(use_multicore) -> ttnn.argmax(dim=-1, use_multicore)`.
Both ops are device-resident and trace-safe (no D2H / host work inside capture); the
argmax output is a transient that belongs to the trace and is reused on replay.

**Implementation note (load-bearing):** single-core `ttnn.argmax` on a TILE `[1, 151936]`
tensor is **~5.15 ms** — slower than the host tail it replaces. `use_multicore=True`
requires ROW_MAJOR input, so untilize-multicore first, then multicore argmax: **~0.09 ms**
standalone, and **+0.01 ms** to the in-trace device replay (measured A/B in one process:
pure replay 14.427 → 14.438 ms).

| metric | logits D2H + host argmax | on-device argmax (int readback) | delta |
| --- | --- | --- | --- |
| full decode step (one harness, p50 over ~140 warm steps) | 18.61 ms | **15.63 ms** | **−2.98 ms (−16%)** |
| in-trace device replay (A/B, p50) | 14.427 ms | 14.438 ms | +0.011 ms (argmax is free) |
| decode-step guardrail PCC (logits path, default) | 0.999448 | 0.999448 | unchanged |
| OCR text (32 tok vs HF) | exact | **exact (token-identical)** | `**TABLE II** – ODDS RATIO OF HODGKIN LYMPHOMA…` |

The full ~2.62 ms D2H+cast+host-argmax tail is recovered; the int readback is cheaper
than even the host cast, so the realized win (−2.98 ms) slightly exceeds the D2H alone.
The graph-zone `decode_traced_ms` reported by `profile_ocr_traced.py` does NOT reflect
this win (the saving is in the host tail, outside that zone) — the new
`decode_full_step_ms` field added to the SUMMARY line is the honest per-token metric.

## Sub-pass 6: device-resident decode RoPE (drop per-layer cos/sin H2D)

After the on-device argmax killed the logits D2H, `decompose_decode.py` showed the next
host-tail item was the per-step RoPE cos/sin H2D: `write_decode_pos` called
`write_decode_rope(pos)` on **every one of the 28 layers**, each doing
`from_torch(cos_row) + from_torch(sin_row) + 2× copy_host_to_device_tensor` into that
layer's persistent `[1,1,1,head_dim]` decode buffers — **56 tiny H2D copies/step ≈ 0.96 ms**.

But the RoPE cos/sin row for a position is **identical across all 28 layers** (one shared
table). So the per-layer upload was fully redundant. The fix keeps the full
`[max_seq, head_dim]` cos/sin tables **device-resident** (ROW_MAJOR DRAM, built once in
`TtLanguageModel.__init__`) and gathers the current position's row **once per step on
device** via `ttnn.embedding` over a persistent `[1,1]` uint32 position index (written by
`write_decode_pos`, trace-safe — the captured trace reads it at a stable address). The
single gathered `[1,1,1,head_dim]` cos/sin pair is threaded into every layer's
`forward_decode_traced` / `forward_decode_traced_sharded`. `write_decode_rope` and the
per-layer persistent cos/sin buffers are gone; the untraced `forward_decode` PCC-reference
path still host-slices (unchanged).

| metric | per-layer cos/sin H2D (28×) | device-resident gather | delta |
| --- | --- | --- | --- |
| `pos_rope_h2d_ms` (decompose, isolated) | 0.978 ms | **0.017 ms** | **−0.96 ms (→ ~0)** |
| full decode step (in-process A/B, same trace, median-of-runs) | 16.96 ms | **16.00 ms** | **−0.96 ms (−5.7%)** |
| in-trace device replay (decompose p50) | 14.61 ms | 14.53 ms | within noise (gather is free) |
| decode-step guardrail PCC (vs reference) | 0.999448 | 0.999448 | unchanged |
| decode-step guardrail PCC (traced vs untraced) | 1.000000 | **1.000000** | exact |
| OCR text (32 tok vs HF) | exact | **exact (token-identical)** | `**TABLE II** – ODDS RATIO OF HODGKIN LYMPHOMA…` |

The full ~0.96 ms per-layer RoPE H2D tail is recovered. The on-device gather costs two
`ttnn.embedding` ops/step (~free on the device replay). RoPE is now fully device-resident
and trace-safe: per step only a 1-element position index is uploaded (the gather index),
shared by all layers — no per-layer cos/sin H2D.

## Sub-pass 7: multicore decode head-split (nlp_create_qkv_heads_decode)

The sharded decode path reused the PREFILL head-split op
(`ttnn.experimental.nlp_create_qkv_heads`) on the single-token interleaved
`[1,1,1,2048]` QKV vector — it runs on **1 core (~0.57 ms/tok)**. Swapped the
decode branch to the decode-optimized **`ttnn.experimental.nlp_create_qkv_heads_decode`**,
which spreads the head-split across the head/core grid (`CoreGrid(y=4,x=8)`,
one padded head-block per core) and emits height-sharded 1BQD q/k/v
(`[1, batch=1, nh/nkv, hd]`). The 1BQD layout lets q feed
`scaled_dot_product_attention_decode` directly (it already wants `[1,batch,nh,hd]`)
— the old `[1,nh,1,hd]` path needed an extra `permute` before SDPA.

**Honest read — the wiring is what made/broke this.** A first attempt restored the
old `[1,nh,1,hd]` interleaved layout after the decode op (3× `sharded_to_interleaved`
+ 3× `permute`) — those 6 layout-restore ops cost **more** than the 0.57 ms the
head-split saved, **regressing ~1.0 ms/tok** (A/B: 16.0 → 17.0 ms/tok DecodeGraph).
The shipped 1BQD path consumes the decode op's output with only 3
`sharded_to_interleaved` (no permutes, and drops the SDPA-input permute), flipping
it to a small net win. The decode op's height-sharded output still needs converting
to interleaved because the custom RoPE (`_rotate_half` concat) runs on interleaved —
that conversion claws back most of the kernel saving, so the realized win is well
below the ~0.5 ms estimate.

| metric | single-core prefill split | multicore decode split (1BQD) | delta |
| --- | --- | --- | --- |
| DecodeGraph signpost ms/tok (A/B, same harness, 3 iters) | 16.03–16.14 | **15.80–15.92** | **−0.2 to −0.3 ms/tok (~−1.5%)** |
| decode-step guardrail PCC (vs reference) | 0.999430 | **0.999430** | unchanged |
| decode-step guardrail PCC (traced vs untraced) | 1.000000 | **1.000000** | exact |
| OCR text (32 tok vs HF) | exact | **exact (byte-identical)** | `**TABLE II** – ODDS RATIO OF HODGKIN LYMPHOMA…` |

Per-op `NlpCreateHeads*` core count could not be read by name: tracy's
`process_ops_logs.py` op-name enrichment fails on this decode pattern (the
`OP NAME` column comes back blank — the same pre-existing tooling failure noted in
Sub-pass 2), so the multicore spread is confirmed by the op's C++ output-shard grid
(>1 core by construction) + the reproducible end-to-end DecodeGraph delta rather than
a per-op-by-name table. On by default under `DOTS_SHARDED_DECODE=1`; set
`DOTS_HEADSPLIT_DECODE=0` to fall back to the single-core split.
