# h44p increment 3: MOE_G=8 + COLSPLIT auto-enabled. Diff: changes_auto.diff (on main 5af47395bb9, clean). Files: tt/prefill_layer.py, tt/prefill_model.py, tt/prefill_handoff.py

## What changes
* DSV41_MOE_G default 8 (was 1; =1 forces the old path). DSV41_COLSPLIT default "auto" (=0 forces off, =1 forces on and asserts if impossible).
* `colsplit_active(U, C)` (tt/prefill_layer.py): colsplit iff MOE_G == 8 and U*C % 256 == 0 (a multiple of 8 chunks of 32 tokens per mesh row). The model decides per chunk size in `alloc_inputs(C)` (sets `model.cs`
  and `layer.colsplit`), the paged sink / ragged head use the same function, so prompts of different lengths in one process switch automatically (re-capture happens anyway when C changes).
* Fallback when the shape does not allow colsplit: replicated path; with G=8 front-end when chunks%8==0, else a T=32 front-end `pmoe.g1` (own shared buffers for all layers, expert weights shared): no error, e.g. U=1 with chunk 128.
  The DEMO therefore needs no env.
## Verified
* 4 layers, U=1, sparse on, shapes alternating in one process (128 -> fallback g1, 512 / 2048 -> colsplit): logits vs forced old path (DSV41_MOE_G=1): PCC 1.00001 (128), 0.99764 (512, 3/4 argmax), 0.99914 (2048).
* e2e test, 40 layers, ISL 4096, B=16 (U=4, chunk 1024 -> colsplit), NO env, dump s4096b1f (prefill2_e2e_auto_4k.log): first-token PCC 0.99180, argmax 16/16; teacher-forced decode PCC 0.99447 / 0.99212 / 0.99098 (16/16 each);
  state ring 0.980 / latents 0.9954; warm prefill 25.05 s (2616 tok/s, 65.5k tokens); closed loop 45.4 ms/token.
* demo/text_demo.py session (no env), 40 layers (prefill2_demo_auto_b16.log, _b64.log): gsm8k_b16: TTFT 757 ms (ISL<=111), decode 57.6 ms/token, 14/16 correct vs gold (scored by /mnt/tt-data/ssinghal/wt/h44p/score_gsm.py: first 16 test questions);
  isl4k_b16 (ISL 3720): TTFT 24.9 s (2391 tok/s), decode 45.0 ms/token; isl8k_b16 (ISL 7443): TTFT 44.3 s (2688 tok/s), decode 45.5 ms/token (closed loop ran; no dump at 8k so no PCC).
  gsm8k_b64 (U=16, chunk 256): TTFT 5.74 s, decode 68.4 ms/token = 935 tok/s, 64/64 correct vs gold.
## DRAM note (for the 32k/64k B=16 OOM work)
Colsplit shrinks the transient stream working set: allocated MiB/bank growth over the build, 40 layers U=1: chunk 1024 +137 (G=1) -> +46 (colsplit); chunk 2048 +237 (G=8) -> +80. T=256 moe buffers +3 MiB/bank. Keep U*C a multiple of 256 so it stays on.
## (3) Avoiding the 4->32 row stream padding: analysis, not implemented
The padded fp32 stream chunk [32,1,4,D] is 21 MB/chunk (8x waste: compact [1,1,128,D] tile layout = 2.6 MB, exactly 4 tiles). After the column split each chip holds 1/8 of the chunks, so the padding costs ~46 MiB/bank at 1024 tokens/row (was 137) and the mHC phases are
~1/8 of their former share of the layer. A compact layout needs new mHC kernels (readers/writers/TensorAccessor tile mapping of mhc_proj/collapse/expand assume [T,1,4,D] pages) or a per-layer padded<->compact conversion (to_layout RM + reshape + to_layout, 3 ops each way per 32-token chunk, ~20 MB of traffic):
that would eat most of the saved time and only pays for memory at >= 8k tokens/row. Recommendation: not worth it now; the memory lever with the best return is a smaller DSV41_PREFILL_ROW_TOKENS together with colsplit.
## Not verified
ISL 8k PCC vs dump (dump unfinished), B=64 / 128 at ISL >= 1k, closed-loop GSM8K beyond 64 questions.
