# GLM-4.7-Flash fused decoder — work log

Stage: fused-decoder (graph fusing; single Blackhole chip, 1x1 mesh, device 0).
Starts from the completed functional decoder (commit 11d5578c175, work log in
`../functional_decoder/`). Skill: `.agents/skills/graph-fusing/SKILL.md`.
Date: 2026-08-31.

## Method

Followed the graph-fusing loop: (1) swept the tt-metal op library and the
reference models (gpt_oss experts, deepseek_v3 mla1d + moe, common/modules/moe,
tt_transformers attention) for dedicated ops; (2) wrote out the functional op
sequence from the committed tt-perf-report CSVs (66.4 ops/step moe decode,
1167 us device + 419 us gaps; prefill 82% all-expert sparse matmul); (3)
classified candidates; (4) verified every rewrite on device (PCC + traced
latency); (5) iterated until no graph-level candidate remained.

## Session log

- Device health: `tt-smi -ls` 4x p300c Blackhole visible; single-chip stage
  uses device 0. Contract probes + all runs on device 0 only.
- Wrote `probe/fusing_contract_probe.py` — 10 on-device probes; all passed:
  matmul `activation="silu"` (PCC 0.999937); binary lhs-SILU via
  `[ttnn.UnaryOpType.SILU]` (the string form is rejected by the binding);
  topk idx = uint16 TILE, converts to single-stick RM; indexed sparse_matmul
  gate-like (compact 6D out) and down-like (compact-A) with topk-derived ids;
  `ttnn.gather` fp32-by-uint16 exact; broadcast-batched wq_b matmul;
  `concatenate_heads`; `nlp_concat_heads_decode`; `slice_write`; mixed-dtype
  (bf16 x fp32) router matmul; block-union mask via view-reshape + max.
- Op-library findings recorded:
  - sparse_matmul hardcodes `FUSE_ACTIVATION=0` (factory line 444) — no fused
    activation; fold SiLU into the eltwise multiply instead.
  - sparse_matmul indexed/gather mode (docs in matmul_nanobind.cpp + op tests
    test_sparse_matmul_indexed.py): compact output, ids read on device,
    program-cache-safe -> trace-safe; requires is_input_b_sparse, uint16
    single-stick ids, num_active <= E.
  - `ttnn.experimental.moe_compute` single-card path exists but its matmul
    output fails PCC on Blackhole (skipped in CI, tt-metal#50038) — blocked.
  - deepseek mla1d fuses wq_a+wkv_a into one matmul (adopted) and runs the
    b={20} absorbed matmuls DRAM-sharded-batched (deferred: program config).
- Built `tt/fused_decoder.py` as a FunctionalDecoder subclass; first on-device
  smoke passed for both kinds and both fold_uk variants at functional PCC.
- A/B round 1 (traced, synthetic): fold_uk=True slower everywhere (moe decode
  1.652 vs 1.335) -> REJECTED. Dense decode regressed 1.008 -> 1.232 with the
  batched wq_b -> isolated with traced micro-benches:
  qpath flat+I/R/T 123 us vs batched 372 us (M=1 tile); outpath
  nlp_concat_heads 238 vs 247 us; wqkv_a fused 44 vs 65 us. Decode q path
  reverted to flat wq_b; prefill keeps the batched layout (its win is the
  deleted reshape+permute at M=2048). Explicit-config rescue of the batched
  decode matmul is blocked: broadcast batch (in0 batch 1) TT_FATALs with any
  explicit program config and with the batched DRAM-sharded config.
- A/B round 2: packed gate_up (one sparse matmul, one in0 mcast pass):
  decode indexed 180 -> 144 us, prefill grouped stage 84.9 -> 68.1 ms/chunk
  (probe scripts inline; results also visible in the final perf reports) ->
  ADOPTED everywhere (weights [1,E,2048,3072] replace gate+up; a second
  unpacked copy would cost 226 MB/layer at bf4 -> not duplicated).
- A/B round 3: routed_scaling 1.8 folded into down weights (removes a scalar
  multiply per router invocation); B=1 identity-transpose elision (3 fewer
  dispatches/step; cos/sin/kv-half (1,2) transposes of [1,1,1,d] are
  identities).
- Bug found during bring-up: `nlp_concat_heads_decode` with batch 32 threw
  `bad optional access` — the 32-core row-wise corerangeset splits into 3
  ranges on the 13-wide grid, flipping the op into subcoregrid mode which
  requires an explicit sub_core_grids. Fix: build the input shard grid as a
  single rectangle (8x4 for batch 32). Also: the op pads the user dim to a
  full tile, so the output is sliced back to B before wo (kept: the pair
  still beats the untilize/reshape/tilize it replaced).
- Prefill accumulator: rolling concat -> preallocated tensor + slice_write
  (deepseek mla1d idiom); same in the MoE down-split loop.
- Prefill MoE sparsity: real per-block union mask (gate/up) + per-chunk expert
  union (down), nnz=None. Correctness is exact because non-selected experts
  have exactly 0 routing weight in the same bf16 tensor used for the combine
  (bf16 flush-to-zero consistency argument recorded in the module docstring).
  Perf-report follow-up: the big pre-sparse-matmul UnaryDeviceOperation rows
  (~2.0 + 1.4 ms/chunk) are the op's own output zero-fill and exist
  identically in the functional baseline (checked both CSVs) — not a
  regression; op-internal, out of stage scope.
- Verified the lhs-SILU eltwise fold is genuinely in-kernel: mul 1.37 ms,
  mul+lhs-silu 1.57 ms, separate silu+mul 2.80 ms at [1,64,1024,1536].
- Test suite `tests/test_fused_decoder.py`: the functional suite re-pointed at
  FusedDecoder (Harness gained a decoder_cls kwarg) + fused-vs-functional
  dense equivalence (PCC 1.000000 prefill and decode) + 96-replay traced
  stress with bitwise repeatability. 20 synth + 3 real passed.
- Long-context ladder rerun with GLM47_DECODER=fused (test_long_context.py
  now selects the decoder via env var): 5 passed — 8k anchor vs full HF
  (0.999459 prefill), 202751 moe + dense control + aligned 202752; dense
  32/32 rows at bar everywhere; every below-bar moe row proven an exact
  alternate-top-4 routing flip; 0 unexplained. moe 202751 prefill 90.3 s
  (2246 t/s) vs functional 95.7 s.
- Perf evidence (final code): standalone wall-clock runs (8 windows,
  `perf_wallclock_*.json`) + a tracy session -> per-window tt-perf-report
  tables via `--start/end-signpost PERF_{mode}_{kind}_{impl}`; raw 15 MB ops
  CSV disk-only. Final: moe decode 1.532 -> 1.035 ms/tok wall (1166.9 ->
  787.3 us device), dense 1.008 -> 0.969 (923.5 -> 890.4); moe prefill
  268 -> 210 ms, dense 19.5 -> 15.3 ms.
- Watcher (separate from profiling, final code): TT_METAL_WATCHER=2 over
  decode PCC (both kinds), moe-512 prefill, cache content, traced decode +
  stress — 6 passed, 26 dumps, 0 exception/assert/sanitize/fault lines.
  Raw watcher.log (1 MB) disk-only; bit-exact .gz committed.
- context_contract.json: added the `fused_decoder` section —
  capability_reduction none, identical cache contract, +0.36 GiB weight
  footprint (prefill wq_b copy) accounted against the 32 GiB budget.

- Post-evidence probe: `nlp_create_qkv_heads_decode` with num_kv_heads=0 (the
  q-only head-split candidate) segfaults host-side (core dump). Device health
  re-verified per tt-device-usage: no stale pytest procs (only the tracy web
  UI server), `tt-smi -ls` shows all 8 board entries, open/compute/close
  smoke OK (`DEVICE_SMOKE_OK`). Recorded as an exact op-contract blocker.

## Commands (exact)

```bash
# contract probes
python models/autoports/zai_org_glm_4_7_flash/probe/fusing_contract_probe.py
# correctness (final code)
pytest models/autoports/zai_org_glm_4_7_flash/tests/test_fused_decoder.py -q -s -m "not real_weights"
pytest models/autoports/zai_org_glm_4_7_flash/tests/test_fused_decoder.py -q -s -m "real_weights"
GLM47_DECODER=fused pytest models/autoports/zai_org_glm_4_7_flash/tests/test_long_context.py -q -s -m long
# perf (wall clock, then tracy for the ops CSV)
pytest models/autoports/zai_org_glm_4_7_flash/tests/test_fused_perf.py -q -s
python -m tracy -r -p -v -m pytest models/autoports/zai_org_glm_4_7_flash/tests/test_fused_perf.py -q -s
tt-perf-report --arch p150 --start-signpost PERF_DECODE_MOE_FUSED --end-signpost PERF_DECODE_MOE_FUSED_END \
  --csv .../decode_fused_perf_report.csv <ops_csv>   # x8 windows
# watcher (separate run)
TT_METAL_WATCHER=2 TT_METAL_LOGS_PATH=.../logs/watcher pytest .../test_fused_decoder.py -q -s \
  -m "not real_weights" -k "decode_pcc or (prefill_pcc and moe-512) or cache_content or traced"
```

## Stage review

- Independent $stage-review (fresh subagent, reviewer mode): **clean-pass**,
  no required work. The reviewer re-derived all headline numbers from the
  committed CSVs/JSON/logs, verified the rejected-candidate evidence against
  the op sources (FUSE_ACTIVATION=0, indexed-mode num_active <= E validation,
  batch-mismatch TT_FATAL, moe_compute #50038 CI skip), and controlled every
  anomaly (the post-evidence fused_decoder.py edit was proven docstring-only
  by bytecode diff of the cached pyc; the 202k moe end window matches the
  functional baseline with 0 unexplained rows; the pos-514 tie and the 3
  batch-32 ties are individually proven).
- Post-review fixes (concerns, not required work):
  1. README runtime-fallback audit corrected: prefill's slice_write
     accumulator lowers to an untilize + slice_write + tilize composite at
     chunk boundaries (~330 us of the 15.2 ms dense window) and decode keeps
     a small pair inside the post-concat-heads batch slice (~25 us/step);
     both measured, both net wins over what they replaced.
  2. README prefill PCC table re-transcribed from the final
     logs/pytest_fused_synth.log (four 6th-decimal transcription slips).
  3. concat_heads_mem now rejects batch != max_batch_size instead of
     silently reusing the cached shard config.
  4. Stacked-report double extensions fixed (*_stacked.csv/.png).
  5. The nlp_create_qkv_heads_decode num_kv_heads=0 blocker repro added to
     probe/fusing_contract_probe.py as a subprocess-guarded case; rerun of
     the full probe suite: 10 contracts OK + blocker reproduced (this time
     as a hang, killed at 120 s; first observation was a fast core dump —
     both recorded). Device smoke re-verified OK after each event.
- Post-fix sanity: fused decode/traced tests re-passed after the
  concat_heads_mem guard (see final commit state).

## Checkpoint commit

- LOCAL CHECKPOINT COMMIT (never pushed), repo /home/stisi/tt-metal, branch
  ttmodelmanager/glm47-flash-probe, stage-owned paths only
  (models/autoports/zai_org_glm_4_7_flash), all pre-commit hooks passed:
  1efaf46d8a4eb69ee161d9c6775199f10593fd68 — fused decoder stage (91 files;
  also untracks the __pycache__ .pyc files the functional-stage commit had
  swept in; the 14 MB raw tracy ops CSV and the 1 MB raw watcher.log remain
  disk-only per the repo 500 KB commit limit). Pre-commit reformats were
  behavior-neutral: 5-test device smoke re-passed on the committed code
  (decode PCC both kinds, moe-512 prefill, cache content, traced decode).
  This work-log commit record is folded in by a single amend (same pattern
  as the functional stage); the amended SHA is the one recorded by
  `git log` on the branch.
