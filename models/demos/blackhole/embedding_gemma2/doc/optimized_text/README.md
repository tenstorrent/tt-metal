# E5a: optimized text tower (stage report)

Scope: the EmbeddingGemma 2 text tower (`tt/attention.py`, `masks.py`, `mlp.py`, `linear.py`,
`per_layer_input.py`, `encoder_layer.py`, `rope.py`, `text_model.py`, `config.py`) on one Blackhole
P150a (13x10 grid), sjc-snva-tp100, 2026-10-27. The delivered path is the optimized one; the
functional baseline is the E4 commit `9e1cb0dd79`. Vision and audio towers are unchanged (the
shared `MLP` keeps its E2 path unless built with `optimized=True`; `RMSNorm` is unchanged).

Labels: **[M]** measured (command and log named), **[I]** inferred from code (file:line), **[G]** guess.

## Result

Traced device time per forward (`tests/perf/bench_sustained.py`, inputs resident,
`execute_trace` back to back, one sync per window). Burst = median of 5 windows of 20 replays,
each after 3 s idle. Sustained = median of 5 windows after a 10 s soak (100 replays per window
for the baseline and the per-change rows; 30 for the final row). Window spread is under 0.1 %
in every row, so any change above 0.05 ms is outside noise. This card did not throttle on this
workload: sustained equals burst within 0.03 ms everywhere (the bge_m3 throttle was at about
200 A; this model draws less) **[M]** `e5a/bench.jsonl`.

| batch x bucket | E4 burst ms | E4 sustained ms | final burst ms | final sustained ms | speedup | final tokens/s |
|---|---|---|---|---|---|---|
| 1 x 128 | 8.752 | 8.749 | 4.923 | 4.920 | 1.78x | 26.0k |
| 1 x 512 | 18.314 | 18.308 | 7.315 | 7.311 | 2.50x | 70.0k |
| 1 x 2048 | 58.984 | 58.973 | 22.053 | 22.058 | 2.67x | 92.9k |
| 8 x 512 | 74.498 | 74.483 | 29.417 | 29.485 | 2.53x | 139.2k |
| 1 x 8192 | 411.701 | 411.850 | 109.810 | 109.956 | 3.75x | 74.6k |
| 1 x 65536 (max bucket, new) | — (bucket did not exist) | — | 2230 (eager, warm) | — | — | 29.4k |

Final rows: HEAD `93f4074a48`, label `final`; E4 rows: label `baseline_E4` (1x8192 measured with
the E4 `tt/` files checked out over HEAD, `e5a/scripts/final.sh`). The 65536 row is one warm eager
forward, not a trace.

## Op-topology audit (one encoder layer, 1x512)

| step | E4 ops (device us, Tracy) | candidate | action | evidence |
|---|---|---|---|---|
| input norm | rms_norm (10.1, 16 cores) | sharded norm | not done: interleaved norm is row-parallel (16 tile rows = 16 cores); L1 input does not help at 1x512 (10.0 us) | **[M]** tracy c4 |
| QKV | already one fused linear (25 us, HiFi4 bf16, default config) | bfp8 + HiFi2 + 2D mcast in0_block_w 8 | kept | 9.9 us **[M]** tracy c4 |
| head split | nlp_create_qkv_heads (11.3) | — | kept, output in L1 (5.0) | **[M]** |
| q/k/v norm | 3 x rms_norm (9.1/6.6/6.0) | fuse with RoPE | no ttnn prefill op fuses head norm + RoPE (`rotary_embedding_llama_fused_qk` is decode only) **[I]**; L1 outputs (6.5/5.8/5.0) | **[M]** |
| RoPE | rotary_embedding x2 (28.6/19.9) | rotary_embedding_hf, rotary_embedding_llama (interleaved, weights permuted) | rejected: same device time (12.2 / 12.0 / 12.2 us at q [1,4,512,256] in a trace); llama has head_dim <= 256. Kept rotary_embedding with L1 tables and L1 in/out | **[M]** `logs/rope_micro.log` |
| SDPA | 1 op, dense [B,1,S,S] bf16 mask, q256/k128 on 8x8, HiFi4 (309 sliding / 252 full) | chunk/grid sweep, HiFi2, mask dtype, block map, native window, windowed mode | kept chunk table + HiFi2 (68 / 123); bfp4 masks; block map for buckets >= 4096 | see "SDPA" |
| concat heads | nlp_concat_heads (6.9) | SDPA `output_concat_heads=True` | kept: op removed | **[M]** |
| o_proj | linear (28.9) | bfp8/HiFi2/2D config | kept (9.0) | **[M]** |
| post-attn norm + residual add | rms_norm + add | rms_norm `residual_input_tensor` computes norm(x + r), not x + norm(a) | not applicable **[I]**; L1 placement kept | |
| MLP gate/up | 2 linear (26/25) + gelu (20.7) + mul (27) | packed gate|up (N=4096), gelu in matmul epilogue, gelu as multiply pre-activation | kept separate gate/up with gelu fused (packer) above 1024 tokens, gelu as multiply input activation at <= 1024 tokens | see "MLP" |
| MLP down | linear (53.4) | 2D config / minimal_matmul | kept (15.1 at 1x512; minimal_matmul at >= 2048 tokens) | **[M]** |
| PLE input | slice + rms_norm per layer (3.3 + 10) | batched [24] projection + 1 norm | rejected: broadcast bmm 604 us vs 115 us for the one wide linear | **[M]** `logs/small_ops.log` |
| PLE block | gate linear + gelu + mul + proj + norm + add | gelu fusion as MLP | kept | |
| layer_scalar | multiply (4.7) | SFPU post-activation of the PLE residual add | kept: op removed | **[M]** |
| per forward | mask build: subtract, multiply, repeat, add (bf16 S x S) | bfp4 broadcast adds | kept (no repeat) | |

Ops per forward at 1x512: 743 (E4) -> 647 (final) **[M]** tracy. Decode, KV cache, paging,
sampling, MoE and CCL items of the optimize skill do not apply to this one-pass encoder on one chip.

## SDPA

Sweep: `e5a/scripts/sdpa_sweep.py` (q/k chunk 32-512, grids 8x8 / 12x10 / 13x10, HiFi4 vs HiFi2)
and `sdpa_variants.py` (mask handling). PCC vs an fp32 torch golden of the same inputs, valid rows.

| shape | E4 config (us) | best (us) | config kept |
|---|---|---|---|
| sliding 1x128 | 48.2 | 18.7 | q32 k128 8x8 |
| full 1x128 | 65.1 | 32.8 | q64 k128 8x8 |
| sliding 1x512 | 316.0 | 72.6 | q32 k64 8x8 |
| full 1x512 | 252.6 | 123.1 | q32 k32 8x8 |
| sliding 1x2048 | 1232.2 | 356.7 | q64 k128 13x10 |
| full 1x2048 | 1760.1 | 549.7 | q64 k64 13x10 |
| sliding 8x512 | 358.1 | 200.9 | q128 k128 13x10 |
| full 8x512 | 895.5 | 287.5 | q128 k128 13x10 |

Rule for other shapes (`attention._sdpa_heuristic`): the smallest power-of-two q chunk that keeps
batch x heads x S / q_chunk at or below the core count (cap 128 for head_dim 512, 256 otherwise),
k chunk 128, full grid above 64 work units. HiFi2 + fp32 accumulation: PCC 0.99990+ per op.

Mask handling (sliding layer, PCC identical across variants unless noted) **[M]** `logs/sdpa_variants.log`:

| variant | 1x512 | 1x2048 | 1x4096 | 1x8192 | 8x512 |
|---|---|---|---|---|---|
| dense bf16 mask | 72.8 | 616.4 | 1101.3 | 4041.6 | 364.7 |
| dense bfp4 mask | 72.5 | 612.4 | 1096.1 | 4032.1 | 362.4 |
| bf16 mask + block map | 75.5 | 834.8 | 502.2 | 806.2 | 368.9 |
| bfp4 mask + block map | 74.9 | 802.3 | 448.1 | 736.1 | 367.2 |
| native `sliding_window_size` 1024, no padding | 110.9 | 1565.2 | — | 4019.5 | 804.9 |
| `cu_window_seqlens` (padding only, no band) | 168.9 | 1560.6 | — | 4044.8 | n/a (1-D) |

Native window semantics vs HF `|q - k| <= 512`: window 1024 and 1025 both match (PCC 0.999928 at
2048, 1.000000 at 8192 vs the exact band golden); 1026 does not (0.999311 / 0.999090) **[M]**.
The native window was rejected: 2.2-2.5x slower than the dense mask at the measured shapes, no
faster at 8192, and it cannot express key padding (`sdpa_device_operation.cpp:142`). Windowed mode
handles padding without a mask but is 2-3x slower and takes one window list for the whole batch.
Kept: bfp4 dense masks everywhere and the band block map for sliding layers at buckets >= 4096,
batch 1 (the map's batch dim must equal the mask's) **[I]** `masks.py`.

## Matmul search

`e5a/scripts/mm_sweep.py` (M = 128 / 512 / 2048 / 4096; every projection role; bf16 HiFi4,
bfp8 HiFi2 with and without fp32 accumulation, bfp8 LoFi, bfp4 LoFi; default, minimal_matmul and
2D multicast configs on 13x10 / 13x8 / 8x8 / 8x10 with in0_block_w 1-8) and `mm_grid.py` (fp32
accumulation, subblock <= 4, large M). Selected examples (us, eager back-to-back; small ops are
host bound in this harness, so the Tracy rows below are the reference):

| role (M=512) | E4 default bf16 HiFi4 | kept | PCC (per op, vs fp32) |
|---|---|---|---|
| QKV sliding 512->2048 | 26.5 | 17.7 (2D 13x10 bw8, bfp8 HiFi2) | 0.99990 |
| o_proj full 2048->512 | 56.0 | 20.6 (2D 8x8 bw8) | 0.99997 |
| MLP down 2048->512 | 55.7 | 20.4 | 0.99990 |
| PLE all 512->12288 | 118.2 | 76.0 | 0.99985 |

| role (M=4096) | default | kept |
|---|---|---|
| QKV sliding | 141.1 | 100.1 (2D 13x10 bw8, fp32 acc) |
| o_proj / down | 100.8 / 178.9 | 57.5 / 84.4 (minimal_matmul) |
| PLE gate (512->512) + gelu | 62.2 (no gelu) | minimal_matmul with fused gelu 218 us -> rejected; 2D config with gelu, or minimal + gelu in the multiply |

Precision policy (kept): bfp8 weights, HiFi2, fp32 accumulation in the 2D configs and
minimal_matmul. Rejected: bfp4 weights (per-op PCC 0.9931-0.9943 on every role, below the 0.995
per-layer bar); LoFi (0-5 % faster, 5e-5 lower PCC per op; left to the E6 sweep). bf16 weights with
the same configs were measured end to end: 2 % slower at 1x512, 7.5 % at 8x512, with better
accuracy (min embedding cosine 0.99938 vs 0.99751, min last-hidden PCC 0.99859 vs 0.99663)
**[M]** `results_acc_*.jsonl`, `bench.jsonl` labels `c2_matmul_bf16` / `c2_matmul_bfp8`. E6 should
weigh this trade.

## L1 placement

`config.act_memory_config`: an intermediate goes to L1 interleaved when it needs <= 64 KB per L1
bank and the forward has <= 4096 tokens; the residual stream follows the same rule. Above 4096
tokens some ops fill L1 with circular buffers (at 8192 SDPA's CB region ends at 1484160 B of
1572864; an L1 activation clashed) **[M]**, so everything stays in DRAM there. RoPE cos/sin for
buckets <= 512 also live in L1 (3 MB in total).

## Change ledger

Traced device ms, burst median (sustained median in brackets where measured); each row is
cumulative over the rows above. Source: `e5a/bench.jsonl` (label in the last column).

| # | change | 1x128 | 1x512 | 1x2048 | 8x512 | decision | commit / label |
|---|---|---|---|---|---|---|---|
| 0 | E4 baseline | 8.752 (8.749) | 18.314 (18.308) | 58.984 (58.973) | 74.498 (74.483) | — | `9e1cb0dd79` / `baseline_E4` |
| 1 | SDPA chunk/grid table, HiFi2, `output_concat_heads` | 7.868 (7.865) | 12.713 (12.711) | 35.983 (35.990) | 67.759 (67.747) | kept | `ebe6ede2c1` / `c1_sdpa` |
| 2 | projections: bfp8, HiFi2, 2D configs / minimal_matmul, gelu fused in the gate matmul | 5.643 (5.640) | 9.984 (9.980) | 30.426 (30.427) | 40.916 (40.912) | kept | `93c41d6bce` / `c2_matmul_bfp8` |
| 2b | same with bf16 weights | 5.949 | 10.190 | 30.535 | 43.989 | rejected for speed (accuracy better; E6) | `c2_matmul_bf16` |
| 3 | layer_scalar into the add + minimal_matmul for narrow roles (incl. activation) | 5.572 | 10.123 | 32.275 | 45.219 | regression at >= 2048 tokens: minimal_matmul + fused gelu 218 us (Tracy) -> fixed in 5a | `c3_lscalar_mmrule` |
| 4 | L1 intermediates (<= 64 KB/bank, <= 4096 tokens) | 5.170 (5.166) | 7.918 (7.912) | 25.248 (25.248) | 36.493 (36.489) | kept | `c4_l1` |
| 5a | activation matmuls back to the 2D config | 5.170 | 7.926 | 23.639 | 33.484 | kept | `c5_gelumul0_resl10` |
| 5b | + residual stream in L1 | 5.131 | 7.739 | 22.189 | 30.902 | kept | `c5_gelumul0_resl11` |
| 5c | + gelu as multiply pre-activation everywhere | 5.026 | 7.671 | 22.628 | 31.054 | partly: kept <= 1024 tokens only | `c5_gelumul1_resl11` |
| 5 | final c5 (hybrid gelu) | 5.026 (5.022) | 7.670 (7.666) | 22.192 (22.193) | 30.900 (30.925) | kept | `93c41d6bce` / `c5` |
| 6 | bfp4 masks + block map (>= 4096) | 5.026 | 7.650 | — | 30.652 | kept; 1x8192 175.536 -> 109.732 | `29efe95c78` / `c6_only_noropeL1`, `c6_bfp4mask_noblockmap` |
| 7 | L1 RoPE tables (buckets <= 512) | 4.924 (4.921) | 7.316 (7.312) | 22.059 (22.061) | 29.418 (29.439) | kept | `29efe95c78` / `c6_bfp4mask_blockmap` |
| 8 | 65536 bucket (row-blocked band builder) | — | — | — | — | kept (capability) | `93f4074a48` |

Rejected or not adopted (evidence in the sections above):

- TT native `sliding_window_size`: 1.5-2.5x slower than the dense mask, cannot mask padding.
- `cu_window_seqlens` windowed mode for padding: 2-3x slower; one window list per batch.
- Block map below 4096: slower at 2048 (834.8 vs 616.4 us).
- SDPA `fp32_dest_acc_en=False` (-13 to -18 %) / `exp_approx_mode` (-10 %): per-op PCC 0.9988-0.9997; E6.
- bfp4 weights (per-op PCC 0.993); LoFi (0-5 %); bf16 weights (row 2b).
- minimal_matmul with a fused activation (218 us vs 40 us at M = 4096).
- PLE batched [24]-head projection + one norm (bmm 604 us vs 115 us).
- `rotary_embedding_hf` / `rotary_embedding_llama` (same device time; llama limited to head_dim 256).
- L1 activations above 4096 tokens (CB clash at S = 8192).
- Packed gate|up: priced only (see "Not done").

## Gates

All on HEAD `93f4074a48` with `source e1/env.sh` plus `HF_MODEL=<pinned 914f7f89 snapshot>
HF_HUB_OFFLINE=1` (see `work_log.md`, "Environment incident"). Logs in `e5a/logs/gate_*.log`,
metrics in `e5a/results_gates.jsonl`.

| gate | command | result |
|---|---|---|
| G1 PCC, trace, audit (text, vision, audio, full model) | `pytest models/demos/blackhole/embedding_gemma2/tests/pcc -q -k "not s32768"` | 342 passed, 1 deselected (494.7 s) |
| G2 full model | `pytest models/demos/blackhole/embedding_gemma2/tests/pcc/test_full_model.py` | 30 passed (49.3 s) |
| G3 demo | `pytest models/demos/blackhole/embedding_gemma2/demo/demo.py -s` (env.sh `HF_MODEL`, `HF_HUB_OFFLINE=1`, HF `refs/main` restored to `914f7f89`) | 2 passed; retrieval argmax TT == HF on all 4 tasks |
| G4 watcher | `TT_METAL_WATCHER=10 pytest tests/pcc/test_{attention,encoder_layer,mlp,embedding,text_model,trace}.py -k "not s32768"` | 98 passed; `watcher.log` has 0 lines matching error/assert/sanitiz/overflow/hang |

Accuracy on G1 (min over cases, text tower; bars unchanged): per-layer PCC 0.999902 (layer 0,
padded b3; E1 0.999978), attention 0.999877, MLP 0.999929, PLE inputs 0.999960, embedding
cosine 0.996575 (8-row batch, the 7-token row; E1 0.999607), last-hidden PCC 0.996917 (E1 0.998819
at 32768, 0.99930 at 512), trace replay == eager bit-identical (16 cases), full-model cosine min
0.999264. Fallback audit (`test_fallback_audit.py`, in G1): passed (0 torch calls, 0 host ops).
The lower end-to-end cosine is the bfp8 weights (row 2 vs 2b of the ledger), not the SDPA/L1
changes (c1: 0.999705).

## tt-perf-report (before / after)

One warmed eager forward between the `PERF_TEXT` / `PERF_TEXT_END` signposts
(`tests/perf/perf.py::test_tracy_forward[b1_s512|b8_s512]`, run by `e5a/scripts/tracy_case.sh`):

```
python -m tracy -r -p -v -m pytest models/demos/blackhole/embedding_gemma2/tests/perf/perf.py::test_tracy_forward[b1_s512] -q -s
tt-perf-report <ops_perf_results.csv> --start-signpost PERF_TEXT --end-signpost PERF_TEXT_END --csv forward_perf_report.csv
tt-perf-report <ops_perf_results.csv> --start-signpost PERF_TEXT --end-signpost PERF_TEXT_END --no-color > forward_perf_report.txt
```

"before" = E4 `tt/` files (`git checkout 9e1cb0dd79 -- .../tt`, restored afterwards), "after" =
HEAD. Tables with advice, CSVs and stacked CSVs: `tracy/{before,after}/{b1_s512,b8_s512}/`; raw
ops CSVs stay in `tt-ops/runs/embeddinggemma-2/e5a/tracy/`. Device time in us (sum of kernel time).

1 x 512: 743 -> 645 ops, 19,836 -> 7,013 us

| op class | before us (count) | after us (count) |
|---|---|---|
| SDPA | 7174 (24) | 1745 (24) |
| matmul | 6605 (171) | 1715 (171) |
| RMSNorm | 2095 (217) | 1858 (217) |
| eltwise binary | 1374 (149) | 835 (126) |
| RoPE | 1324 (48) | 630 (48) |
| eltwise unary (gelu, scalar) | 638 (51) | 5 (3) |
| heads, concat, slice, other | 625 (83) | 225 (56) |

8 x 512: 743 -> 645 ops, 86,366 -> 29,045 us

| op class | before us (count) | after us (count) |
|---|---|---|
| SDPA | 10488 (24) | 4326 (24) |
| matmul | 49977 (171) | 13523 (171) |
| RMSNorm | 6303 (217) | 3331 (217) |
| eltwise binary | 7473 (149) | 3905 (126) |
| RoPE | 5088 (48) | 2236 (48) |
| eltwise unary | 3073 (51) | 5 (3) |
| heads, concat, slice, other | 3965 (83) | 1719 (56) |

Every op in both windows is a device op (no host op rows). Remaining advice in the "after" tables:
"High Op-to-Op Gap" (eager dispatch; the traced forward is 7.31 ms against 7.01 ms of kernel time)
and "SLOW" on the M = 512 matmuls (10-15 us, 38-40 % of FLOP peak at HiFi2; the M x N tiles per
core are small: 104 cores for 16 x 64 output tiles). Normalization is now the largest class at
1 x 512 (26 %): see "Not done".

## Context contract

`doc/context_contract.json`: supported max bucket 32768 -> 65536. bfp4 masks are 0.5625 B per
element: three live masks (band, sliding, full) at 65536 take 7.2 GB of 32 GB DRAM; at 131072 they
would take 29 GB, plus a 34 GB bf16 band on the host, so 131072 is infeasible. 65536 has no HF
reference (the fp32 CPU reference needs about 4 x the 24 GB measured at 32768); evidence there is
padding equivalence (a 32768-token prompt in bucket 32768 vs padded into 65536: cosine 1.000000,
max diff 3.0e-8) and a finite, repeatable 65536-token forward (2.2 s eager) **[M]**
`logs/long_ctx.log`. HF-validated maximum stays 32768.

## Not done / remaining

- Sharded (block / width) RMSNorm with a sharded residual stream: norms are 25 % of 1x512 device
  time (217 ops at 9-10 us, row-parallel on 16 cores) and the largest remaining item.
- SDPA `fp32_dest_acc_en=False` (-13 to -18 %) and `exp_approx_mode` (-10 %) trade 3e-4 to 5e-4
  per-op PCC; LoFi matmuls; bf16 vs bfp8 weights: E6 precision sweep.
- Packed gate|up (N=4096) was priced, not built: 29.6 us vs 2 x 17.7 us at M=512, but it needs two
  slices and loses the fused gelu epilogue.
- Block map for batch > 1 (needs a per-batch map), pooling before the 512->768 projection
  (0.1 % of 1x512; `last_hidden` is an API output).

## Artifacts

- `tt-ops/runs/embeddinggemma-2/e5a/`: `bench.jsonl` (every burst / sustained row), `results_*.jsonl`
  (PCC per run), `logs/`, `scripts/` (sweeps and harnesses), `tracy/<stage>/<case>/` (raw ops CSV).
- `doc/optimized_text/tracy/{before,after}/{b1_s512,b8_s512}/`: tt-perf-report table with advice,
  CSV, stacked CSV.
