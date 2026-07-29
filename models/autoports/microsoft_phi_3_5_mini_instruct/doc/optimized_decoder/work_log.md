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
TT_METAL_LOGS_PATH=$PWD/models/autoports/microsoft_phi_3_5_mini_instruct/doc/optimized_decoder/watcher \
PHI35_RUN_LONG_CONTEXT=1 pytest -q -s \
models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_optimized_decoder.py \
-k 'synthetic_prefill_decode_pcc_and_traced_decode or non_aligned_prefill_and_decode_pcc or decode_batch32_traced_pcc or repeated_input_determinism or full_context_decode_current_position_and_page_table'
```

Result: 5 passed in 27.52 s, watcher clean.

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

## Optimize checklist

- [x] Optimized decode trace replay, no functional or host fallback.
- [x] Width-sharded L1 residual/norm/attention/MLP chain.
- [x] DRAM-interleaved prefill and explicit 2D configs for bounded large-M.
- [x] Topology audit and repeated-projection decisions recorded.
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
- [x] Independent `$stage-review` clean pass and local checkpoint commit.

## Commits

- Repo: `tt-metal`
- Branch: `skillexp-probe`
- Optimized-decoder checkpoint: `44de174c510`
- Review: `doc/optimized_decoder/stage_review.md`, verdict `clean-pass`
- Push: not performed
