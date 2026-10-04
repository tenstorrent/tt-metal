# h44p prefill performance increment: grouped MoE (G=8) + column-split. Diff: changes_perf.diff (git apply -p1 on main 5a15bdd2ccd: clean)

Env gates (all default OFF = unchanged behaviour): DSV41_MOE_G=8 (h45p's grouped moe_compute + big shared expert, merged from /mnt/tt-data/ssinghal/wt/h45p/changes.diff),
DSV41_COLSPLIT=1 (column split of the token-wise work; needs DSV41_MOE_G=8 and chunk*users_per_row/32 a multiple of 8, e.g. U=1 chunk >= 256, U=4 chunk >= 64).
Files: tt/prefill_layer.py (grouped MoE, forward_cols), tt/prefill_model.py (per-column token layout, Engram per group, colsplit-aware traced head: one head trace per owner column),
tt/prefill_attention.py (rs_tokens: reduce-scatter over tokens instead of all-reduce), tests (scen test: SAVE_LOGITS, h45p's profiling/scaling tests).

## Verified at 40 layers, traced default run(), sparse path on, U=1 (4 users, same prompt), replay-only, run 2 of 2 (logs prefill2_g8_u1.log, prefill2_cs_u1_40.log)
| ISL / chunk | baseline G=1 TTFT, PCC | G=8 TTFT, PCC | G=8 + colsplit TTFT, PCC, tok/s |
| 2048 / 512 | 4.92 s, 0.99601 | 4.35 s, 0.99601 (identical) | 3.18 s, 0.99415, 2576 |
| 4096 / 1024 | 9.61 s, 0.99417 | 8.44 s, 0.99417 (identical) | 5.86 s, 0.99175, 2794 |
| 4096 / 2048 | - | 8.51 s, 0.99375 | 5.93 s, 0.98644 (lowest), 2761 |
| 8192 / 1024 | 18.84 s (no dump) | 16.68 s | 12.06 s, 2717 |
| 16384 / 1024 | 37.6 s (dense, older) | - | 23.99 s, 2731 |
| 65536 / 1024 | 196 s (dense, older) | - | 102 s, 2569 (timing only, synthetic Engram rows) |
Replay per 1024-token chunk: 2.09 s -> 1.82 s (G=8) -> 1.15-1.44 s (colsplit). Compile+capture 27-140 s (not in TTFT). Free DRAM after 64k: 765 MiB/bank.
U=4 (16 users): ISL 128: replay 1.12 s -> 0.52 s, PCC vs dump 0.98096 (14/16) vs 0.97198 (14/16) baseline. ISL 1024 chunk 256: TTFT 10.69 s -> 6.10 s (2686 tok/s), chunk 512 11.03 -> 6.51 s;
logits vs G=1: PCC 0.989 (12/16 and 16/16 argmax), the same size as the chunk-size / sparse-on-off noise (0.987-0.990) measured earlier on those random prompts.
G=8 alone is bit-identical in the 2 layer h45p tests and gave identical first-token PCC at 40 layers. Colsplit changes rounding (reduce-scatter instead of all-reduce): 4 layers incl. Engram layer 1, eager: PCC 0.999995, 16/16 vs colsplit off.
## Not verified / caveats
Colsplit with the paged hand-off sink (state_sink gets replicated kv/latents so it should work; not run), with a ragged head subclass (it overrides last_logits: its input xs are per-column chunks now), U=8+, G=8 with chunk counts not a multiple of 8 (raises).
mHC for T > 32 was NOT written: with the column split each column runs mHC on R/8 tokens with the existing T=32 kernels, so the kernel rewrite is no longer the first lever. Head + readback is now 1.1-1.2 s of a 6 s U=4 prefill (16 users x 129k-vocab gather): next cheap target.
