# Optimized decoder work log

## Scope and baseline

- Model: `microsoft/Phi-3.5-mini-instruct`
- Stage: optimized decoder only.
- Functional checkpoint: `8adaf288329`.
- Stage-owned paths: `tt/optimized_decoder.py`,
  `tests/test_optimized_decoder.py`, this directory, and the optimized section
  of `doc/context_contract.json`.
- Device health: `timeout 60 tt-smi -ls --local` showed four Blackhole p300c
  devices. Hardware commands were serialized.

Current functional baseline command:

```bash
pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/functional_decoder_perf.py
```

Result: prefill-128 1.917198 ms; traced decode context-128 batch 1
1.050973 ms and batch 32 1.216832 ms.

## Candidate ledger

0. Mandatory shard-advisor seed (this pass): activated the pinned
   `mvasiljevic/shard-advisor-dram-sharding` tt-mlir checkout at
   `618cd4e75d`, then captured the rewritten batch-32 dense attention+MLP
   block. The report contains 38 ops, 35 final choices, and one output spill.
   DRAM sharding was considered and advised for all five linears. The
   authoritative IR recommends 8-bank DRAM-sharded weights/inputs,
   `in0_block_w=12` for hidden-input projections, and
   `in0_block_w=32` for down. That exact program seed passed real-weight PCC
   at batch 1 and 32, but measured 555.650 us and 756.075 us versus the final
   16-core/block-16 default's 521.679 us and 722.930 us in the same
   environment. The DRAM-sharded family and L1 chaining recommendation are
   applied; the advisor's 8-core/block-32 geometry and 96/86-core output
   layouts are rejected by whole-layer traced latency. Artifacts:
   `shard_advise/report.json`, `shard_advise/final_ir.mlir`, and
   `shard_advise/report.txt`; A/B logs:
   `logs/shard_advise_{seed,default}_b{1,32}.log`.
1. Existing draft cumulative candidate: packed QKV/gate-up, width-sharded L1
   residual/norm, BFP4/LoFi DRAM-sharded decode matmuls, BFP8 cache.
   Correctness passed; current-run timings were 0.5545-0.5549 ms batch 1 and
   0.746720 ms batch 32 after the selected 16-core geometry.
2. DRAM-bank-derived geometry: changed program geometry from the activation
   grid to 12 DRAM banks. The first run failed because input shard width 3 was
   not divisible by `in0_block_w=6`. The adapted run used blocks 3 for
   hidden-input roles and 8 for down, passed PCC 0.9999983, but measured
   0.593255 ms batch 1, slower than 0.556875 ms. Reverted.
3. Large-M prefill static program at non-aligned 32769 failed with
   13,041,408 B circular buffers against 1,572,864 B L1. Reusing TTNN's
   bounded default program above 4096 rows passed 32769 and 131071; retained.
4. Real-weight precision/fidelity: BFP4/LoFi 0.556810 ms and PCC 0.9998000;
   BFP8/LoFi 0.582679 ms and PCC 0.9999436; BFP8/HiFi2 0.735706 ms and
   PCC 0.9999891. BFP4/LoFi retained as fastest passing policy.
5. Precision-locked geometry: 8, 16, and 32 working cores were compared.
   With blocks capped at 8 they measured 0.578351, 0.573131, and 0.556750 ms.
   Enabling larger legal blocks gave block 16 for the down path: 8 cores hit
   an L1 CB allocation failure; 16 cores passed at 0.555009 ms. Three
   200-replay confirmations were 0.554667/0.554502/0.554893 ms versus
   0.556522/0.557047/0.556610 ms for 32 cores. The 16-core candidate is final.
6. Fidelity closure: real-weight BFP4/HiFi2 passed PCC 0.9998080 at
   0.729356 ms versus BFP4/LoFi PCC 0.9998000 at 0.556810 ms; LoFi retained.
7. Projection topology: split QKV initially failed concat grid validation, was
   adapted through explicit per-projection interleaving, and passed at
   0.569967 ms (0.564816 ms when combined with split gate/up), slower than the
   packed-QKV final path. Separate gate/up won: three 200-replay runs were
   0.550049/0.550409/0.550487 ms versus packed
   0.555681/0.555158/0.554898 ms. Separate gate/up is the final default.

The final runtime rows prove BFP4/LoFi reached packed QKV, output, separate
gate and up, and down. All deciding precision/geometry PCC runs use official real layer-0
weights from snapshot `2fe192450127e6a83f7441aef6e3ca586c338b77`.

## Commands and results

Mandatory shard-advisor capture:

```bash
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
export TT_METAL_ROOT=/home/mvasiljevic/tt-metal
export PYTHONPATH=/home/mvasiljevic/tt-metal:$PYTHONPATH
cd "$TTMLIR_ADVISOR_HOME"
source tools/ttnn-jit/integrations/agentic-research/shard-advise/scripts/bootstrap.sh
SHARD_ADVISE_BATCH=32 ttnn-advise capture \
  /home/mvasiljevic/tt-metal/models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/shard_advise/advise_phi35.py:decode \
  --out /tmp/phi35-shard-advice-b32-third
```

The first helper attempt failed before capture because the advisor venv lacks
the test-only `safetensors` package; the helper was made self-contained. The
second reached the decoder but exposed a dynamic `gate.memory_config()` query
that the compiler cannot resolve before layout assignment. Replacing that
query with the already-declared phase-specific output memory config made the
runtime compiler-traceable without changing topology. The third capture
succeeded. These setup/tracer issues are not candidate rejections.

Optimized correctness:

```bash
pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py
PHI35_RUN_LONG_PREFILL=1 PHI35_LONG_PREFILL_LEN=32769 pytest -q -s ... -k long_prefill_page_table
PHI35_RUN_LONG_PREFILL=1 PHI35_LONG_PREFILL_LEN=131071 pytest -q -s ... -k long_prefill_page_table
PHI35_RUN_LONG_CONTEXT=1 pytest -q -s ... -k full_context_decode_current_position_and_page_table
```

Long runs at 32769, 131071, exact-limit 131072, and full-context decode pass.
Same-shape batch-32 prefill is 20.277 ms optimized versus 26.165 ms functional.

Watcher:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_LOGS_PATH=$PWD/models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/watcher_shard_advise \
PHI35_RUN_LONG_CONTEXT=1 pytest -q -s \
models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py \
-k 'synthetic_prefill_decode_pcc_and_traced_decode or non_aligned_prefill_and_decode_pcc or decode_batch32_traced_pcc or repeated_input_determinism or full_context_decode_current_position_and_page_table'
```

Post-advisor result: 5 passed in 29.52 s. The generated watcher log contains
no kernel error/assert/hang. Nanobind reference-leak diagnostics appear only
during Python teardown after the pass and successful device close; they are
classified as binding teardown noise, not a model or watcher failure.

Profiler:

```bash
PHI35_HOST_TIMING_ITERS=20 python -m tracy -r -p -v -m pytest \
models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py \
-k synthetic_prefill_decode_pcc_and_traced_decode
tt-perf-report tracy/final/ops_b1.csv --start-signpost PERF_DECODE \
--end-signpost PERF_DECODE_END --no-summary
```

The same command was run with `-k decode_batch32_traced_pcc`. Reports and
CSV-console provenance are under `tracy/final/`.

Post-advisor final-default validation:

- Full default suite: 7 passed, 3 opt-in tests skipped; see
  `logs/final_tests_shard_advise_pass.log`.
- Watcher: 5 passed with `TT_METAL_WATCHER=10`, including full-context
  decode; see `watcher_shard_advise_console.log` and
  `watcher_shard_advise/generated/watcher/watcher.log`.
- Fresh Tracy: `tracy/final/ops_b1_shard_advise.csv` and
  `ops_b32_shard_advise.csv`; advice-enabled reports use the matching
  `_shard_advise.txt` suffix. Batch-1 decode contains 62 device ops, zero host
  ops, and totals 499 us device time. All five dense matmul rows show
  `LoFi BF16 x BFP4 => BF16`; profiled host timing is 566.429 us.

## Optimize checklist

- [x] Optimized decode trace replay, no functional or host fallback.
- [x] Width-sharded L1 residual/norm/attention/MLP chain.
- [x] DRAM-interleaved prefill and explicit 2D configs for bounded large-M.
- [x] Topology audit and repeated-projection decisions recorded.
- [x] `$shard-advise` run this pass on the rewritten dense block; its five
  DRAM-sharded matmul picks and L1 chain were seeded, measured at batch 1/32,
  and applied or rejected with evidence.
- [x] Packed QKV retained and separate gate/up selected by whole-layer A/B.
- [x] Explicit memory, program, SDPA, and compute-kernel configs.
- [x] BFP4/LoFi DRAM-sharded dominant decode matmuls verified in runtime rows.
- [x] Batch-1 and batch-32 traced decode measured; batch 1 wins and 32 improves.
- [x] BFP8 paged cache and cache-consuming traced correctness.
- [x] Non-aligned and advertised context capability preserved.
- [x] Stress/repeat and watcher-clean runs.
- [x] Advice-enabled `tt-perf-report` reviewed; the material MLP projection
  group was attacked with geometry and packed/split topology trials.
- [x] Device/e2e/roofline accounting recorded in README.
- [x] Real-weight BFP4/BFP8 crossed with LoFi/HiFi2 precision/fidelity sweep.
- [x] Precision-locked 8/16/32-core and block-through-16 geometry sweep.
- [x] Packed/split QKV and gate/up whole-layer A/B; measured winners selected.
- [ ] Independent `$stage-review` clean pass and local final checkpoint commit
  (first post-advisor review requested provenance/commit remediation).

## Commits

- Repo: `tt-metal`
- Branch: `skillexp-nofuse-advise`
- Restored optimized baseline commits: `41544a0e7b1`, `0fa46fae7f7`
- Post-advisor first review:
  `doc/optimized_decoder/stage_review_shard_advise_first.md`, verdict
  `more-work-needed` (provenance/commit remediation).
- Final post-advisor checkpoint and clean rereview: pending below.
- Push: not performed
