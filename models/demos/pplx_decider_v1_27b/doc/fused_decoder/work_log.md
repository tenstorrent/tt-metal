# Fused decoder work log (stage 2, prefill only)

Model `perplexity-ai/pplx-decider-v1-27b`, one Blackhole p150a (13x10 grid, 8 DRAM banks).
Branch `gtobarTT/pplx-decider-bringup`. Baseline: stage 1 commit `3d944442cb4`.
Precision policy unchanged: `act_bf16__w_bfp8_all__hifi2`.
Hashes are post-rewrite: the bring-up commits were re-authored to gtobarTT after stage 2 (tree
unchanged; pre-rewrite stage 1 was `248d1a8202c`, stage 2 code `6def56eea90`).

Labels: **measured** means a command and its output recorded here; **inferred** means it follows
from code or arithmetic and was not run.

## Timeline and host conditions

- Start 18:03 UTC, timebox 2 h.
- 18:16 UTC: another agent started a CPU-only HF golden job (`tt-golden_pplx_bf16`, about 10 cores busy).
  From then on, every device run here was pinned with `taskset -c 12-23` or `taskset -c 14-23`. Host
  dispatch noise at small S grew. Decisions on small changes therefore use Tracy device time, not
  wall time. The final before/after table interleaves baseline and fused runs under the same load.

## Method

- One change per measurement. A change is kept only if it is faster and PCC holds.
- PCC: the real-weight tests in `tests/pcc` on the affected modules (S=128/2048/8192, and other
  layers where relevant), logged to `artifacts/pplx_decider/stage2/pcc_<tag>.jsonl`.
- Wall time: `tests/perf/test_prefill_perf.py::test_layer_and_module_perf` (2 warm-ups, median of 7),
  logged to `artifacts/pplx_decider/stage2/perf_<tag>.jsonl`.
- Device time: Tracy capture of `test_profile_layer` (one warmed layer between signposts) and
  `tt-perf-report`, under `artifacts/pplx_decider/stage2/tracy/<tag>_<layer>_S<len>/`.
- Iteration helpers were scratch scripts outside the repo. Their commands are the pytest and Tracy
  commands in the README.

## Op-sequence analysis (from the stage 1 S=2048 profiles)

`linear_attention` (30 device ops, 10.95 ms): 5 minimal_matmuls (56 %), GDN prep and scan (17 %),
6 BinaryNg (8.6 %). Data movement worth removing: the in-proj output slices (qkv 211 µs, z 127 µs),
the untilize before the conv (211 µs), and the SiLU*up mul (472 µs).

`full_attention` (33 device ops, 9.95 ms): 5 minimal_matmuls (61 %), SDPA (13 %). Data movement:
the packed q|k|v|gate slices (161 + 121 µs), the partial-RoPE slice/rotate/slice/concat on q
(71 + 103 + 97 + 125 µs) and k (88 µs), and the SiLU*up mul (469 µs).

## Fusions tried (in order)

Device times are Tracy device µs per 2048-token chunk. Wall times are harness medians in ms.

| # | change | result | decision |
|---|---|---|---|
| 1 | MLP gate+up as one tile-pair-interleaved weight [5120, 34816] and `minimal_matmul(fuse_swiglu=True)`: SiLU*up runs in the matmul epilogue. | Device: 2 x 1327 + 472 (mul) = 3127 -> 2800 µs. Wall MLP: S2048 4.83 -> 4.52, S8192 20.81 -> 19.96. **S128 1.09 -> 1.31.** At 128 rows the fused matmul takes 815 µs (232 GB/s weight read); ttnn.linear did gate + up + mul in about 650 µs. PCC unchanged (mlp L0 S2048 0.999987). | **kept** |
| 1a | Short-row config for the fused matmul (K_block 16). | Micro wall 0.909 -> 0.830 ms, but Tracy device time 815.2 -> 817.8 µs. | rejected (no device gain; config tuning is stage 3) |
| 1b | Short rows via one `ttnn.linear` on the interleaved weight, then reshape/slice de-interleave and SiLU*mul. | Micro: 1.90 ms at M=128 against 0.64 ms for the stage 1 path. | rejected |
| 2 | Attention: split the packed q\|k\|v\|gate projection into a q\|k\|v matmul and a gate matmul. Removes 2 full-width slices. | Device: 1235 + 161 + 121 = 1517 -> 767 + 507 = 1274 µs. Wall attention: S2048 4.70 -> 4.54, S8192 35.59 -> 35.03; S128 0.75 -> 0.81. PCC unchanged. | **kept** |
| 3 | GDN: split the packed in-proj into 3 matmuls: qkv, z, b\|a. | Wall GDN: S2048 5.65 -> 5.56; S128 0.90 -> 1.04 (the 128-column b\|a minimal_matmul is inefficient). | rejected for 3b |
| 3b | GDN: 2 matmuls, qkv and packed z\|b\|a. Removes the 211 µs qkv slice; the z slice (126 µs) stays. | Device: 1279 + 211 -> 859 + 509 µs; layer device time 10.95 -> 10.49 ms (with fusion 1). Wall GDN S2048 5.48, S128 0.93. PCC unchanged. | **kept** |
| 3c | GDN: 3b, plus z alone and b\|a via ttnn.linear, to drop the z slice. | b\|a linear 136 µs (64 cores) against the 126 µs z slice; layer device time 10485 vs 10487 µs. | rejected (no gain, one more op) |
| 4 | Fused partial RoPE. The weight adapter permutes the q/k head dims (q, k rows and the q/k norm weights): neox pair (j, j+32) becomes (2j, 2j+1). One full-width `rotary_embedding_llama` then runs per tensor, with cos=1 and sin=0 on the 192 pass-through dims. q.k is invariant under the shared permutation; v is untouched. | Device: q 393 -> 234 µs, k 88 -> 32 µs; 6 ops fewer per chunk. Wall attention S2048 4.54 -> 4.32, S8192 35.03 -> 33.48. Needs `fp32_dest_acc_en=False` (op limit for head_dim > 128). PCC unchanged to 1e-6 (attention L3/L63, all tested S). Micro: the full-width rotate-half `rotary_embedding` is 347 µs on q against 272 µs for the llama op. | **kept** |
| 5 | Sigmoid of the attention gate as a minimal_matmul epilogue, plain mul afterwards. | Gate matmul 507 -> 652 µs, mul 174 -> 169 µs. | rejected |
| 6 | RoPE cos/sin tables stored TILE [1, 1, 8192, 256], so a chunk is one tile-row slice (no per-call tilize). | 2 tilize ops fewer per chunk (about 14 µs). PCC unchanged. | **kept** |
| - | Non-chunked causal SDPA instead of the chunked paged SDPA (micro, same q/k/v). | 2048: 1.377 vs 1.345 ms; whole 8192 in one call: 16.39 vs 16.32 ms. | no gain, not adopted |

The non-bucket lengths S=100 and S=2100 (one tile-padded chunk, and 2048 + 52 tail) were checked
ad hoc after fusions 1-6: L0 0.999987 / 0.999993, L3 0.999991 / 0.999994 (**measured**,
`pcc_oddlen.jsonl`).

## Short-row investigation (S=128)

- **Micro, wall time, M=128, gate+up shapes** (`taskset -c 14-23`):
  - `ttnn.linear` 17408-wide: 0.326 ms. Two of these plus the mul (the stage 1 path): 0.638 ms.
  - `ttnn.linear` 34816-wide: 0.604 ms.
  - Fused `minimal_matmul`: 0.83-0.91 ms across block configs. Several configs fail validation.
- **Tracy device time:** the fused matmul takes 815 µs at S=128.
- **Cause.** `minimal_matmul_program_factory.cpp:251-259`: for M <= N, the 13 cores on grid x read
  the weight and multicast it down the columns. Weight-read parallelism is capped at 13 cores, about
  232 GB/s.
- **Not fixable in this model's code without a second weight copy** (inferred: about 12 GB over 64
  layers). This is a stage 3 candidate.

## Final verification (on the kept set)

| check | command | result |
|---|---|---|
| PCC suite | `taskset -c 12-23 pytest models/demos/pplx_decider_v1_27b/tests/pcc -q -p no:cacheprovider` | 124 passed, 949 s, exit 0; min 0.999897; no drop > 1e-4 vs stage 1 (deltas -1.4e-6..+3.8e-6) |
| fallback audit | `tests/pcc/test_runtime_audit.py`, part of the suite | passed; 0 host calls |
| determinism | `tests/pcc/test_prefill_contract.py`, part of the suite | passed; bit-identical repeat |
| before/after perf | interleaved A1 B1 A2 B2 (A = `3d944442cb4` tt/ and tests, B = HEAD) | README table; logs `stage2/perf_ab_*.jsonl`, `stage2/logs/ab_perf.log` |
| device profiles | Tracy `test_profile_layer` L0/L3 x S128/2048/8192 | `doc/fused_decoder/perf/`; raw `stage2/tracy/final_*` |
| watcher 10 s | `TT_METAL_WATCHER=10 ... test_decoder_layer.py -k "S2048 and (L0 or L3)"` | 2 passed, exit 0, 0 error lines |
| watcher 1 s | `TT_METAL_WATCHER=1 ... -k "(S2048 or S8192) and (L0 or L3)"` | 4 passed, exit 0, 22 dumps, 0 error lines |

## Not tried, or blocked at model level (stage 3 candidates)

See the README "Stage 3 candidates" table.
