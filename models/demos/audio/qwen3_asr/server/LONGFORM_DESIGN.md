# Long-form support — design (not yet deployed)

Two failures block long audio today. Root causes (from static analysis of tt_transformers;
no device experiments — chip 3 is busy serving):

## Failure 1 — prefill seq length (`a_shape[-1]==b_shape[-2]` in MLP)
`models/tt_transformers/tt/mlp.py` prefill path:
```python
if mode == PREFILL and seq_len >= prefill_len_cutoff:   # 512 on Blackhole
    x = ttnn.reshape(x, [1, seq_len // prefill_len_cutoff, prefill_len_cutoff, -1])
```
So for `seq_len >= 512` the MLP requires **seq_len % 512 == 0** (else the reshape changes the
last dim — e.g. 768 → inner dim 3072 ≠ 2048 → matmul contraction mismatch). Attention separately
requires **seq_len % 256 == 0** (8 cores × tile 32; the earlier 384 "48-row shard" failure).
**Valid prefill pads: 256, or any multiple of 512** (512, 1024, 1536, 2048).

**Fix (Tier 1, STAGED in `tt/qwen3_asr_decoder.py`):** pad to `256 if S<=256 else ceil(S/512)*512`.
This lifts the single-shot cap from ~512 tokens (~20 s audio) to `max_seq_len` (2048 → **~150 s**),
memory permitting. One-line change; needs a server restart to take effect.

## Failure 2 — conv2d `bank_manager` OOM over a long run
ttnn.conv2d allocates per call; over many requests DRAM fragments → allocation fails (also worse
with bigger conv batches = longer segments). A fresh server doesn't hit it; a long-lived one does.

## Tiered strategy

**Tier 1 — prefill pad fix (staged).** Segments up to ~150 s work single-shot. Biggest win, smallest change.

**Tier 2 — server-side chunking (arbitrary length).** Accept any-length upload; the server splits at
silence into windows sized so prefill stays ≤ a chosen cap (e.g. ≤1024 → ~80 s, or ~30 s for lower
latency/memory), transcribes each, and stitches. Fold in the eval's robustness (move into the server
so every client benefits):
  - `detect_and_fix_repetitions` on each window;
  - drop non-speech windows (low RMS) and outlier-language windows (single-speaker hallucinations);
  - majority-language vote for the response `language`.
Cut at silence (no overlap) → plain concatenation; if forced cuts are needed, add small overlap + dedup.

**Tier 3 — memory hygiene.** `ttnn.deallocate` conv/encoder intermediates per request; cap conv batch
via the window size; optionally clear the program cache / light device reset every N requests. Goal:
no fragmentation OOM across an all-day server.

**Tier 4 — proper long single-shot / streaming (deferred, higher effort).**
  - raise `max_seq_len` (more KV/DRAM) for longer single prefills; or
  - paged-attention chunked prefill (tt_transformers supports it) for unbounded prefill; or
  - the AuT windowed-encoder (`n_window_infer`) + chunked decode for true streaming.

## Status: Tier 1-3 DEPLOYED & VERIFIED (2026-06-11)
- **Tier 1** ✅ 40 s single-shot (prefill ~1024) transcribes; the 768-pad MLP failure is gone.
  Pad rule `256 if S<=256 else ceil(S/512)*512` (`tt/qwen3_asr_decoder.py`).
- **Tier 2** ✅ `qwen3_asr_server.py`: SINGLE_CAP=45 s; longer uploads → silence windows (~38-45 s) →
  per-window infer, non-speech (RMS<0.005) skipped, repetition hallucinations dropped, majority lang.
  Verified: 137 s Korean → 4 chunks (CER 0.056 vs YT); 180 s Chinese (Tsai) → 5 chunks, coherent.
- **Tier 3** ✅ real leak was L1_SMALL (conv config tensors). Fix: `Conv2dConfig(config_tensors_in_dram
  =True, deallocate_activation=True)` + `open_device(l1_small_size=65536)`. 10-req soak → 0 OOM.

## Remaining (Tier 4, deferred)
Very-long single-shot (>~150 s) needs a higher `max_seq_len` (more KV/DRAM) or paged chunked-prefill;
true streaming via the AuT windowed encoder. Also: reuse one decode trace in-server (→ rtf ~0.05).
