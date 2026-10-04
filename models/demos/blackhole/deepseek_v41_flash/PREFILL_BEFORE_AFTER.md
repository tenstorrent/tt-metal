# h44p: before/after of the default prefill path (MOE_G=8 + COLSPLIT auto, main c37eb3b994e), notes only (no new package diff)
Demo = demo/text_demo.py session scenarios, 40 layers, same host per column pair, "before" = DSV41_MOE_G=1 DSV41_COLSPLIT=0, "after" = defaults. Logs /mnt/tt-data/ssinghal/dsv4-logs/prefill2_{base_b16,base_b64,demo_auto_b16,demo_auto_b64,def_b64_4k}.log. GSM8K scored by wt/h44p/score_gsm.py (extract answer vs gold of the first N test questions).
| scenario | TTFT before | TTFT after | prefill tok/s before -> after | decode ms/token before -> after | accuracy |
| gsm8k_b16 (ISL<=111) | 1357 ms | 757 ms | 770 -> 1380 | 59.4 -> 57.6 | 14/16 both |
| gsm8k_b64 (ISL<=130, U=16 chunk 256) | 9518 ms | 5740 ms | 413 -> 686 | 76.8 -> 68.4 | 64/64 both |
| isl4k_b16 (ISL 3720) | 42.4 s | 24.9 s | 1404 -> 2391 | n/a -> 45.0 | outputs identical for 16/16 users |
| isl8k_b16 (ISL 7443) | 81.0 s | 44.3 s | 1470 -> 2688 | n/a -> 45.5 | outputs identical for 16/16 users |
| isl4k_b64 (ISL 3720, U=16 chunk 256) | 144.0 s | 79.7 s | 1653 -> 2988 | 67.5 -> 61.9 | same first token; text same, only line-break placement differs in the 64 outputs (0/64 byte-identical) |
GSM8K b16: 6/16 outputs byte-identical, mean common prefix 292 chars (late divergence, normal bf16 noise). Gains: 1.8x (short), 1.7-1.8x (4k-8k), 1.8x (U=16).
## Gaps closed
* Column split at U=16 (B=64): active (users*chunk = 4096 % 256 == 0) in gsm8k_b64 and isl4k_b64; correct (64/64) and 1.8x faster.
* ISL 8192 through paged hand-off + closed loop vs a NEW CPU dump (dsv4-prefill-s8192b1r; the earlier s8192b1 dump had died at layer 24; 86 min on .46 CPU): e2e test, B=16, default env (prefill2_e2e_auto_8k.log):
  first-token PCC 0.98306 argmax 16/16; teacher-forced decode PCC 0.99155 (argmax 0/16: near-tie flip, 16 identical users) / 0.98405 (16/16) / 0.97635 (16/16); state ring 0.983 latents 0.9954; warm prefill 49.0 s (2675 tok/s); closed loop 44.5 ms/token.
  ISL 16k: no dump (a CPU dump of 16k B=1 would take ~4-6 h; not made); 16k timing only in earlier traced ladders.
## Profile of one layer at 4096 tokens/row (U=4, 16 x 1024, eager, device profiler, layers 2-3; tests/prefill_prof_summary.py; dirs prefill2_prof_{def,base})
| phase ms/layer (mean over 32 devices) | before (G=1, no colsplit) | after (default) |
| MoE (router per slice + dispatch + moe_compute + combine + CCL) | 104.7 (52%) | 63.2 (71.4%) |
| attention (S-wide projections, SDPA, state writes, rs/ar) | 16.9 | 15.1 (17.0%) |
| mHC expand+mixes+collapse (ffn) | 35.1 | 4.0 |
| mHC mixes+collapse (attn) | 17.9 | 3.2 |
| expand out | 12.9 | 1.7 |
| shared expert | 14.1 | 1.4 |
| total kernel time | 201.6 | 88.5 (2.28x) |
Inside the after-MoE phase (device 0, grouped by core count): the 48-core `moe_compute` op is ~2.1-2.9 ms per call (16 calls/layer) = ~65% of the phase (~46% of the layer); 80-core dispatch/combine ops ~0.3 ms x 16 = 7.5% of the phase; per-slice router/typecast tiny ops (6 / 32 cores) ~4-6%. The CSV has no op names in this build, so groups are by core count.
Per token: 88.5 ms / 16384 tokens = 5.4 us/token/layer = 216 us/token for 40 layers = 4.6k tok/s kernel-only; observed traced end to end 2.6-2.8k tok/s (idle/launch gaps, CCL waits, indexer layers, head/host). Roofline (tests/prefill_roofline_model.py, 34 GFLOP/token at 4k, HiFi2 peak 350 TFLOP/s/chip x 50%): 165k tok/s all-mesh, i.e. ~35x above the kernel time and ~60x above the end-to-end rate.
## Next biggest item
1. moe_compute itself (T=256 tokens/device/call, 2.6 ms per 1024 tokens = 2.5 us/token/layer): at ~16 tokens per expert the 32-row tile padding wastes the matmul; T=512 fails in moe_core_placement (needs a kernel/placement change by the moe_compute owner; estimated 2.27 -> ~1.5 us/token by h45p). It is ~46% of the layer now.
2. Attention (17%): the S-wide projections/o-proj run on all 8 columns for 1/8 of the heads each; the compressed-layer SDPA is dense-masked (with the indexer only >512 entries); a flash-style fused path or bfp8/HiFi2 fidelity would be the lever. Indexer layers add ~+25 ms/chunk in eager (sparse test: layer 2 33 ms vs 6 ms per 512-token chunk).
3. Gaps between ops inside the trace (kernel 4.6k vs observed 2.7k tok/s): ~40% of wall time is not device kernel time at U=4 chunk 1024: candidates are CCL waits (reduce-scatter/all-gather per 256-token group), host syncs per chunk, the per-chunk uploads (host/chunk 0.1-0.4 s at U=1, 1 s at U=16).
