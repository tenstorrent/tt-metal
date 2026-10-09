# Stage 2: fused decoder layers (prefill only)

Model `perplexity-ai/pplx-decider-v1-27b`, one Blackhole p150a. Branch `gtobarTT/pplx-decider-bringup`.
The baseline is the stage 1 functional commit `248d1a8202c`. The precision policy is unchanged
(`act_bf16__w_bfp8_all__hifi2`: BF16 activations, BFP8 weights, HiFi2 matmuls with fp32
accumulation, HiFi4 norms), and the prefill chunk is still 2048. Labels: **measured** means a
command and its output are recorded here or in [`work_log.md`](work_log.md); **inferred** means it
follows from code or arithmetic.

## Approach

- **Fusions are made in the existing modules.** They live in `tt/` (TTv2 style, one
  implementation, as in bge_m3). There is no parallel `fused_decoder.py`; stage 1 is commit
  `248d1a8202c`, and every before/after number compares against that commit.
- **Prefill only.** Decode, paged-decode trace and KV-cache items of the graph-fusing goal are N/A.
  The length contract is the buckets 128 / 1024 / 2048 / 4096 / 8192 at batch 1.
- **One change per measurement.** A change is kept only if it is faster and PCC holds; see the
  work log for each measurement.

## What changed (kept)

| module | change | why it is faster (Tracy device time, one 2048-token chunk) |
|---|---|---|
| `tt/mlp.py`, `tt/weight_adapter.py`, `tt/common.py` | gate and up are one tile-pair-interleaved weight [5120, 34816]; `minimal_matmul(fuse_swiglu=True)` computes `silu(gate) * up` in the epilogue | 2 x 1327 µs matmul + 472 µs mul -> one 2800 µs matmul |
| `tt/attention.py`, `tt/weight_adapter.py` | q\|k\|v is one matmul and the output gate a second one, instead of one packed q\|k\|v\|gate output sliced twice | 1235 + 161 + 121 µs -> 767 + 507 µs |
| `tt/attention.py`, `tt/rope.py`, `tt/weight_adapter.py`, `tt/optimizations.py` | partial RoPE is one full-width `rotary_embedding_llama` per tensor. The adapter permutes the q/k head dims so each neox pair (j, j+32) becomes the adjacent pair (2j, 2j+1). The cos/sin tables are 1 and 0 on the 192 pass-through dims and are stored TILE (no per-chunk tilize) | q: slice + rotary + slice + concat 393 µs -> 234 µs; k: 88 -> 32 µs; 8 fewer ops per chunk |
| `tt/gated_deltanet.py`, `tt/weight_adapter.py` | the in-projection is a qkv matmul plus a packed z\|b\|a matmul, instead of one packed output with a 211 µs qkv slice | 1279 + 211 µs -> 859 + 509 µs |

The q/k permutation is exact: q and k get the same permutation, so q.k does not change, and the
RMSNorm over head_dim is permutation invariant once its weight is permuted too. v and the gate are
not permuted. The K cache stores permuted keys, which only this module reads. `rotary_embedding_llama`
rejects fp32 dest accumulation for head_dim > 128, so RoPE uses its own HiFi4 config without fp32
accumulation (`AttentionOptimizations.rope_compute_kernel_cfg`).

Rejected after measurement: a 3-way GDN in-projection split, `b|a` through `ttnn.linear`, the
attention sigmoid as a matmul epilogue, a short-row config for the fused SwiGLU matmul, and a
linear-plus-de-interleave SwiGLU for short rows. Numbers are in the work log.

Stage 1 review fixes:

- `make_actual_start` is copied into `tt/gated_deltanet.py`. The import from
  `models.demos.qwen38_27b_qb2` is gone. The bucket contract does not make it dead: the conv op
  needs the scalar.
- `PPLX_DECIDER_PCC_LOG` now also sets the `tests/pcc_report.py` default, and the new
  `PPLX_DECIDER_PROBE_LOG` sets the context-probe log. The defaults are the old paths. The golden
  directory was already `PPLX_DECIDER_GOLDEN_DIR` in `reference/hf_reference.py`.

## PCC (measured)

`pytest models/demos/pplx_decider_v1_27b/tests/pcc -q -p no:cacheprovider` on the fused working
tree. That tree was then committed as `6def56eea90`; the pre-commit `black` hook re-wrapped one
call in `tt/attention.py` and changed no code:
**124 passed in 949 s, exit 0**, with 123 PCC records and 0 failures. Log:
`artifacts/pplx_decider/stage2/logs/pcc_suite_final.log`; records:
`artifacts/pplx_decider/stage2/pcc_suite_final.jsonl`.

- The lowest PCC is 0.999897 (`gated_deltanet` L61 S8192). It is the same case as in stage 1, and
  the value moved by +5e-8.
- Per-module minimums:
  - decoder layer: 0.999988 (`linear_attention`) and 0.999967 (`full_attention`);
  - gated attention 0.999927, MLP 0.999933, GDN 0.999897, readout 0.999961.
- Compared with stage 1 (the stage 1 log, records before 18:00 UTC on 2026-10-09; 108 matching
  keys), the per-case PCC delta ranges from -1.4e-6 to +3.8e-6. **No case dropped by more than 1e-4.**
- Non-bucket lengths checked ad hoc: S=100 and S=2100 give L0 0.999987 / 0.999993 and L3 0.999991 /
  0.999994.

## Warmed prefill latency, before -> after (measured)

Harness: `tests/perf/test_prefill_perf.py::test_layer_and_module_perf`, the same as stage 1: batch 1,
eager, 2 warm-ups, then the median of 7 passes, with `synchronize_device` around each pass.

Baseline and fused code ran interleaved, A1 B1 A2 B2, under the same host load. A is the stage 1
`tt/` and `tests/` (checked out from `248d1a8202c` for the run, then restored). B is HEAD. Each cell
is the mean of the two run medians, in ms. Logs: `artifacts/pplx_decider/stage2/perf_ab_{A1,B1,A2,B2}.jsonl`.

| target | kind | S=128 | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---|---|---|---|---|---|
| decoder layer | linear_attention | 2.11 -> 2.37 (+12.3 %) | 6.18 -> 6.03 (-2.4 %) | 11.02 -> 10.64 (-3.4 %) | 22.44 -> 21.67 (-3.5 %) | 46.71 -> 45.23 (-3.2 %) |
| decoder layer | full_attention | 1.91 -> 2.18 (+14.2 %) | 5.39 -> 5.03 (-6.6 %) | 10.07 -> 9.26 (-8.0 %) | 22.70 -> 21.09 (-7.1 %) | 57.96 -> 55.20 (-4.8 %) |
| GDN mixer | linear_attention | 0.90 -> 0.93 (+4.0 %) | 3.10 -> 3.09 (-0.5 %) | 5.62 -> 5.53 (-1.7 %) | 11.64 -> 11.46 (-1.6 %) | 24.11 -> 23.81 (-1.2 %) |
| attention mixer | full_attention | 0.75 -> 0.81 (+7.7 %) | 2.36 -> 2.14 (-9.0 %) | 4.80 -> 4.28 (-10.8 %) | 12.00 -> 11.05 (-7.9 %) | 35.67 -> 34.41 (-3.5 %) |
| MLP | linear_attention | 1.08 -> 1.30 (+20.6 %) | 2.81 -> 2.65 (-6.0 %) | 4.83 -> 4.52 (-6.4 %) | 9.83 -> 9.18 (-6.6 %) | 21.48 -> 20.26 (-5.7 %) |
| MLP | full_attention | 1.08 -> 1.31 (+20.6 %) | 2.79 -> 2.62 (-5.9 %) | 4.84 -> 4.50 (-7.1 %) | 9.87 -> 9.20 (-6.8 %) | 22.27 -> 21.09 (-5.3 %) |

The two runs of each side agree within 0.04 ms at S <= 4096. Run to run, S=8192 moves up to 0.7 ms.

### Why S=128 got slower

**Both layer kinds got slower at S=128, and the MLP accounts for most of it.**

- The fused SwiGLU exists only in `minimal_matmul`, so the MLP uses it at every length.
- At 128 rows the projection is bound by weight reads. Tracy: 815 µs for 189 MB, about 232 GB/s.
  `ttnn.linear` did gate + up + mul in about 650 µs.
- Cause: `minimal_matmul` puts N on the 13-wide core-grid axis when M <= N, so 13 cores read and
  multicast all of the weight (`minimal_matmul_program_factory.cpp:251-259`). `ttnn.linear`
  spreads the read over 80-128 cores.
- Block-size changes do not move the device time: 815.2 vs 817.8 µs.
- Keeping a second, non-interleaved gate/up copy for short rows would add about 12 GB over 64
  layers. That does not fit next to the 27 GB model.
- The two-matmul attention and GDN in-projections add one dispatch each, which costs 0.03-0.06 ms
  at S=128.

**Net effect over 64 layers (48 linear + 16 full, inferred from the medians):**

| S | stage 1 (ms) | stage 2 (ms) | change |
|---:|---:|---:|---:|
| 128 | 132 | 149 | +17 ms (+12.7 %) |
| 1024 | 383 | 370 | -13 ms (-3.4 %) |
| 2048 | 690 | 659 | -31 ms (-4.5 %) |
| 4096 | 1440 | 1378 | -63 ms (-4.4 %) |
| 8192 | 3169 | 3054 | -115 ms (-3.6 %) |

The fix belongs to stage 3, on the op or config side (see the candidates below). Until then, the
done criterion "beat stage 1 at every bucket for the dominant layer kind" is **not met at S=128**.
It is met at 1024, 2048, 4096 and 8192 for both kinds.

## Device profiles and op counts (measured)

Captures: Tracy plus `tt-perf-report` of one warmed layer between `PREFILL_START` and `PREFILL_END`.
The fused reports are in [`perf/`](perf/) (`*_perf_report.txt` and `.csv`, at S = 128, 2048 and 8192).
The baseline numbers come from `../functional_decoder/perf/` (S=2048 and S=8192) and from new S=128
captures of the stage 1 code (`artifacts/pplx_decider/stage2/tracy/base_*`). Every capture shows
0 host ops.

| layer kind | S | device ops before -> after | device time before -> after |
|---|---:|---:|---:|
| linear_attention | 128 | 30 -> 28 | 2.010 -> 2.269 ms |
| linear_attention | 2048 | 30 -> 28 | 10.954 -> 10.487 ms (-4.3 %) |
| linear_attention | 8192 | 125 -> 117 | 44.511 -> 42.734 ms (-4.0 %) |
| full_attention | 128 | 33 -> 22 | 1.809 -> 2.130 ms |
| full_attention | 2048 | 33 -> 22 | 9.952 -> 9.172 ms (-7.8 %) |
| full_attention | 8192 | 137 -> 93 | 53.500 -> 50.370 ms (-5.9 %) |

Per chunk, the `linear_attention` layer loses the qkv slice and the SiLU*up mul and gains one
matmul. The `full_attention` layer loses 4 slices, 2 concats, 2 tilizes and the SiLU*up mul, gains
one matmul, and replaces 2 rotate-half ops with 2 llama rotary ops.

## Runtime audit, determinism, watcher (measured)

- **Fallback audit.** `tests/pcc/test_runtime_audit.py` passes in the suite: 0 host conversions
  and 0 torch ops in a measured pass, for L0 and L3 at S=128 and S=4096. Tracy shows 0 host ops in
  all 6 fused captures.
- **Determinism.** `tests/pcc/test_prefill_contract.py` (8192 -> 128 -> 2048 -> 4096 on one
  instance, then a repeat) passes in the suite. The repeat is bit-identical (`torch.equal`). The
  contract PCCs are >= 0.999990.
- **Watcher.** `TT_METAL_WATCHER=10` on `test_decoder_layer.py -k "S2048 and (L0 or L3)"` gives
  2 passed, exit 0, and no error, assert, sanitize, overflow or hang entry in the log
  (`artifacts/pplx_decider/stage2/watcher/generated/watcher/watcher.log`). A 1 s interval run at
  S2048 and S8192 gives 4 passed, exit 0, 22 dumps and no errors (`.../stage2/watcher_1s/`).

## Stage 3 candidates (not done here)

| candidate | measured cost now | why not in stage 2 |
|---|---|---|
| Short-row fused SwiGLU (S=128 regression) | 815 µs vs about 650 µs unfused | Needs a `minimal_matmul` that reads the weight on more than 13 cores for M <= N, or DRAM-sharded weights. This is op or sharding work. |
| Input/post RMSNorm core count | 2 x 192 µs on 64 cores at S=2048; 2 x 82 µs on 4 cores at S=128 | The interleaved norm parallelises over tile rows only, so it needs a sharded program config. |
| Matmul block configs | down 63 %, qkv 62 %, o_proj/out_proj 60 % of peak FLOPs | Grid tuning is deferred to stage 3 by the brief. |
| SDPA | 1.27 ms per 2048 chunk; 4 chunks at S=8192 | A plain causal SDPA is not faster (1.38 vs 1.35 ms; whole 8192: 16.39 vs 16.32 ms). Needs SDPA program config work for head_dim 256. |
| Larger prefill chunk (one chunk per bucket) | input chunk slices plus the final concat, about 0.8 ms at S=8192 | `test_prefill_contract.py` pins the chunk to 2048 (acceptance check). This is a design decision. |
| GDN conv input untilize | 211 µs | `qkv_causal_conv1d_silu` takes ROW_MAJOR input only. Needs an op change. |
| GDN g/beta relayout inside `chunk_gated_delta_rule` | 2 x 111 µs `ReshapeView` | Op internal ([BH, T] -> [BH, NC, C, 1]). |
| GDN `sigmoid_gated_rms_norm` + `* z` | 330 + 168 µs | Needs a SiLU-gated variant of the op. |
| GDN z slice from the z\|b\|a output | 126 µs | `minimal_matmul_split` needs equal chunk widths; `ttnn.linear` for b\|a costs the same (136 µs). |
| Residual adds | 2 x 139 µs | No residual epilogue in `minimal_matmul`, and `rms_norm(residual_input_tensor=)` does not return the sum. |
| Head split/concat | 160 + 130 µs on 64 cores | The prefill `nlp_create_qkv_heads` / `concatenate_heads` have no fused alternative. Needs a sharded config. |
| Single-chunk requests without the paged cache | 2 x 22 µs `paged_fill_cache` | Too small to matter. The module does not know whether more chunks follow. |
| Trace capture per bucket | host dispatch at S=128 | Not a graph fusion. |

## Limitations

- **S=128 is slower than stage 1** for both layer kinds, by about 0.26 ms per layer. See "Why S=128
  got slower".
- **Batch 1, eager, prefill only, 8192 context**, as in stage 1.
- **Host load.** From 18:16 UTC a CPU golden job shared the host. Device runs were pinned with
  `taskset -c 12-23` / `14-23`, and before/after was measured interleaved. Wall times at S=128
  include host dispatch.
- **Layer scope.** PCC covers layers 0, 3, 61 and 63. The full 64-layer check is stage 6.

## Commands

The environment is the same as stage 1 (`source python_env/bin/activate; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD HF_HOME=/local/ttuser/gtobar/hf`).

| purpose | command |
|---|---|
| PCC suite | `pytest models/demos/pplx_decider_v1_27b/tests/pcc -q` |
| PCC vs stage 1 | compare `pcc_suite_final.jsonl` with the stage 1 `logs/pcc_results.jsonl` by (module, layer, seq_len) |
| warmed perf | `PPLX_DECIDER_PERF_LOG=<log> pytest models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py -q -s -k test_layer_and_module_perf` |
| baseline perf | `git checkout 248d1a8202c -- models/demos/pplx_decider_v1_27b/tt models/demos/pplx_decider_v1_27b/tests/{test_utils.py,pcc,probe,pcc_report.py}`, run the line above, then `git checkout HEAD -- models/demos/pplx_decider_v1_27b/tt models/demos/pplx_decider_v1_27b/tests` |
| device profile | `python -m tracy -r -p -v --no-web-server -o <dir> -m pytest "models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py::test_profile_layer[L0_linear-S2048-device_params0]"` (S128, S2048 and S8192; L0_linear or L3_full) |
| perf report | `tt-perf-report <ops_perf_results.csv> --start-signpost PREFILL_START --end-signpost PREFILL_END --no-advice [--csv out.csv]` |
| watcher | `TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=<dir> pytest models/demos/pplx_decider_v1_27b/tests/pcc/test_decoder_layer.py -q -k "S2048 and (L0 or L3)"` |

Artifacts are under `/local/ttuser/gtobar/artifacts/pplx_decider/stage2/`: `logs/`, `pcc_*.jsonl`,
`perf_*.jsonl`, `tracy/` and `watcher*/`.
