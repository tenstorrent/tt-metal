# AutoFix: long prefill and bounded cache tail

## Starting evidence

- Source: `AUTODEBUG.md` sections 1–2.
- The autoport sent every prefill length through non-chunked SDPA even though
  canonical Gemma4 caps that path at 32,768.
- The autoport sent tile-padded K/V to bounded `paged_fill_cache`, allowing
  padding rows to wrap over live modulo-cache history.

## Hypothesis experiments

### Long prefill dispatch

- Prediction: 32,768 may use non-chunked SDPA; 32,800 must use full or sliding
  chunked attention.
- Host experiment:
  `pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py -k prefill_attention_dispatch_host`
- Result: passed.
- Verdict: source hypothesis verified; runtime fix implemented.
- Fix: full attention chunks Q against the populated paged cache; sliding
  attention uses overlapping square causal+sliding slices and retains only the
  new rows.
- Device verification:
  `GEMMA4_LONG_ATTN_TEST=1 pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py -k long_prefill_attention_correctness`
  passed for both sliding and full attention.

### Exact bounded-cache tail

- Prediction: logical lengths produce `(tile-aligned prefix, exact tail)` and no
  physical padding row is written.
- Host experiment:
  `pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py -k bounded_cache_fill_plan_host`
- Result: all cases `1,31,32,33,1023,1024,1025,1055,1056` passed.
- Verdict: source hypothesis verified; runtime fix implemented.
- Fix: fill the aligned prefix, then serially write each real tail token with a
  one-user height-sharded `paged_update_cache` and a device-resident int32
  one-user position tensor. Serial writes avoid concurrent read-modify-write
  races within one cache tile.
- Device verification:
  `GEMMA4_RANGE_DOWNLOAD=1 pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py -k bounded_modulo_prefill_tail_cache_integrity`
  passed.

## Current status

Source, host tests, and both serialized device regressions pass. TT hardware
commands were serialized and executed by the parent stage, not this subtask.
The long full-attention regression uses the natural
`[blocks,2,128,512]` cache. Long-prefill chunked SDPA with the optional shared
HMA physical cache view remains unverified.
