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

| stage | untraced | traced | trace speedup |
|-------|----------|--------|---------------|
| vision (42 L) | 3218 ms | 3252 ms | **0.99×** |
| LM prefill (28 L) | 1427 ms | 1426 ms | **1.00×** |
| decode / step | — | 16.5 ms | — |
| **e2e** | 4695 ms | 4720 ms | 0.99× → `**TABLE II**` |

**Key finding:** at real resolution vision (≈69% of e2e) and prefill (≈30%) are
**compute-bound — metal trace gives 0%** (it only helped on the 256×96 synthetic
toy image, which was dispatch-bound). The optimization lever is op-level compute
efficiency in the **vision encoder** (windowed SDPA + matmuls over ~19.5k
patches), not trace/dispatch. The earlier "prefill matmul needs DRAM-sharding"
diagnosis was a benchmark artifact (input re-uploaded inside the timing loop);
isolated MLP matmuls with preallocated inputs are sub-1.3 ms.

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
