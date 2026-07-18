# Optimized multichip decoder work log

## Scope and provenance

- Model: `tiiuae/Falcon3-7B-Base`
- Stage: optimized-multichip-decoder only
- Starting branch: `mvasiljevic/model/tiiuae-falcon3-7b-base`
- Starting repository HEAD: `74b73791498c5e838ebcfb6266c71324424a7bbf`
- Completed multichip implementation commit: `b2666fe1505`
- Completed multichip audit commit: `cdbf998b3d6`
- Target: four Blackhole p300c devices, mesh `1x4`, TP axis 1,
  `FABRIC_1D_RING`, two links
- Final implementation SHA256:
  `1bb774f48c3dd19e9c4ba0550e6eb279809973ff0fc856552cd06f16d2b1a199`
- Final performance-test SHA256:
  `c7701566daa7895b2cb1f4940dd8cca4a4243ade11c0e16c2b97900ea1b98b38`

No full-model, generator, LM-head, or vLLM work was started. The unrelated
pre-existing edit to `.agents/skills/forge-functional-decoder-from-ir/SKILL.md`
was preserved and excluded from this stage.

## Device discipline

The `$tt-device-usage` workflow was used for every hardware phase. Mesh runs
were serialized. Health checks preceded baseline work and followed all final
runs. Every test closed the mesh and disabled fabric. Watcher and Tracy were
never enabled together. A stalled historical AGMM links=2 experiment was not
repeated; the prior stage had already triaged and recovered it.

Representative health command:

```bash
tt-smi -s
```

Final health at `2026-07-18T19:43:05Z`: four p300c devices, DRAM healthy,
GDDR corrected/uncorrected error counters zero. Firmware bundle 19.8 is newer
than the latest fully tested 19.5 bundle. TT-Metal also warns that the host has
about 17.5 MB free in `/dev/shm` when requesting a 16-MB MPI segment.

## Fresh baseline

The baseline was captured before modifying the completed multichip decoder:

```bash
for batch in 1 32; do
  FALCON3_RUN_MULTICHIP_PERF=1 \
  FALCON3_MULTICHIP_PERF_BATCH=$batch \
  FALCON3_MULTICHIP_RESULTS_DIR=models/autoports/tiiuae_falcon3_7b_base/doc/optimized_multichip_decoder/results/baseline \
  pytest -q models/autoports/tiiuae_falcon3_7b_base/tests/test_multichip_decoder.py::test_warmed_multichip_trace_performance
done
```

| Batch | Warmed prefill | Traced warmed decode |
| ---: | ---: | ---: |
| 1 | 0.822899863 ms | 0.356703736 ms |
| 32 | 3.125764895 ms | 0.578790270 ms |

The baseline implementation/test hashes are respectively
`b249847705594bbf49e795eecdc1669a0016929929952d0767c9939cef1dd573`
and `b6b10e32580ff5e4f9fdb465099272727a147049d5f191e91bd0dcae91ded557`.

## Operation-topology audit and coherent families

The starting graph had packed rank-local QKV, separate same-input gate/up
matmuls, row-parallel O/down, two standard BF16 Ring all-reduces, and an
inter-layer public-output conversion from a replicated-mesh L1 width-sharded
residual through interleaved L1 to DRAM.

The audit proceeded by coherent families:

1. Collective placement and persistence: replace both decode reductions with
   persistent asynchronous all-reduce and preallocate role-specific buffers.
2. CCL payload and links: compare BF16/BFP8 and one/two links on that same
   persistent path.
3. Residual contract: retain the reduced residual's replicated mesh placement
   and width-sharded L1 device layout between decoder layers instead of
   restoring the old public DRAM boundary.
4. Lower movement: carry row-parallel output through reduce-scatter, local
   residual add, distributed RMSNorm, statistics all-gather, full activation
   all-gather, and the next real QKV. The full downstream family, not an
   immediate restore microbenchmark, is the rejection gate.
5. Fused CCL/matmul: measure real next-QKV fused AGMM provenance and exact O
   and padded-down fused MMRS shapes. Adapt the MMRS probe to the current
   `persistent_output_buffers=[intermediate, output]` API after its first stale
   API error, then rerun eager, trace, PCC, and profiles.
   Separately build real local-output O/down weights and cross both selected
   BFP8 row-parallel boundaries with explicit async all-gather+matmul and
   fused AGMM while keeping the resulting residual hidden-fractured.
6. Matmul topology: compare the exact fresh shard-advisor report against the
   selected DRAM-sharded family, and compare rank-correct packed versus
   separate gate/up projections.
7. Precision and advice: execute BF16/HiFi4 after adapting the initial L1
   overflow; independently cross attention/MLP weight dtype, LoFi/HiFi2 at
   the same dtype, attention/MLP activation dtype, grid-x 10/subblock, and L1
   prefill inputs.
8. Per-role DRAM-sharded geometry: sweep divisor-valid QKV/O/gate-up/down
   core counts, retain physical grid, shard, block, PCC, and wall-time fields,
   then repeat the narrow winner before changing defaults.

The full decision matrix is in `README.md` and
`results/candidate_summary.csv`. Key sequential outcomes were:

| Candidate | Batch-32 decode | Outcome |
| --- | ---: | --- |
| Starting standard BF16 CCL/public output | 0.578790 ms | baseline |
| Persistent async BF16 CCL/public output | 0.533160 ms | keep family |
| Persistent async BFP8 CCL/public output | 0.529814 ms | keep dtype |
| Same persistent BF16 family, one link | 0.551102 ms | reject; use two links |
| Direct stackable L1 residual | 0.504766 ms | keep contract |
| Selected 4/4/24/8 geometry, before activation-dtype selection | 0.503439 ms | keep geometry |
| BFP8 attention+MLP activations | 0.491488 ms | selected |

The exact role sweep tried QKV 4/8/16, O 4/8/12/16/24/48, gate/up
8/12/24/48, and down 8/12/24/48. O16 maps to O12's physical 6x2/12-core
program and O48 maps to O24's 8x3/24-core program. QKV4 and O4 together
measure 0.302523 ms at batch 1
and 0.503421 ms at batch 32 with HF decode PCC 0.999998255/0.999998283.
Paired default-grid repeats are 0.303151/0.303066 ms; paired O4 repeats are
0.302576/0.302774 ms. O12 narrowly led O4 before the activation sweep, so it
was retried under the winning BFP8 activation family: O4 wins at batch 1
(0.286602 vs 0.288198 ms) and batch 32 (0.491443 vs 0.495796 ms). The final
default is QKV/O/gate-up/down 4/4/24/8.
Every artifact records the DRAM-sharded matmul program class, exact shape,
grid, input shard tiles, `in0_block_w`, per-core M/N, input/output layout,
and eight-bank DRAM weight layout. Output-subblock fields are not exposed by
this TTNN program class, so no synthetic values are reported.

The lower-movement boundary was finally repeated under the selected
persistent-BFP8 policy, not just the earlier BF16 family. Persistent-buffer
reduce-scatter, local residual add, distributed RMSNorm, BFP8 statistics
all-gather, BFP8 hidden all-gather, and selected real QKV measure 0.108096 ms
versus 0.043906 ms for the selected replicated all-reduce boundary at PCC
0.999851. Physical BFP8 Ring traffic sent per rank is 159,936 versus 156,672
bytes because next-QKV compatibility still requires the hidden gather. The
fractured family is 2.462x slower. Two clean non-Watcher BF16 repeats also
measure 0.109990/0.109965 ms fractured versus 0.097506/0.097512 ms
replicated. No lower-movement family was rejected by an immediate restore to
the old public boundary.

Persistent resources were also tested with a real owner and borrower. Both
buffers and semaphores have identical object identity, the borrower owns none,
sequential owner/borrower outputs have PCC 1.0, three borrower trace replays
pass, and the owner runs again at PCC 1.0 after borrower release. This proves
the 835,584-byte per-device pair is safely shared across layers.

The exact fused-MMRS probe command family was:

```bash
FALCON3_RUN_FUSED_MMRS=1 FALCON3_FUSED_MMRS_TRACE=1 \
pytest -q models/autoports/tiiuae_falcon3_7b_base/tests/test_optimized_multichip_decoder_mmrs_probe.py

FALCON3_RUN_FUSED_MMRS=1 FALCON3_FUSED_MMRS_TRACE=1 \
FALCON3_FUSED_MMRS_NON_FUSED=1 \
pytest -q models/autoports/tiiuae_falcon3_7b_base/tests/test_optimized_multichip_decoder_mmrs_probe.py
```

Eager and trace passed for O `[3072,3072]` and padded global down
`[24576,3072]`, M=32, BFP4 weights, TP4, Ring, two links. Minimum trace PCC is
0.9927695/0.9932234 for O/down. The fused primitive saves 2.532 us for O and
1.271 us for down, smaller than the complete fractured boundary loss.

The distinct gathered-input/local-output O/down family was then run with real
layer-14 weights and selected BFP8 activation/CCL semantics:

```bash
FALCON3_RUN_MULTICHIP_O_DOWN_AGMM=1 \
FALCON3_MULTICHIP_RESULTS_DIR=models/autoports/tiiuae_falcon3_7b_base/doc/optimized_multichip_decoder/results/candidates/o_down_agmm \
pytest -q models/autoports/tiiuae_falcon3_7b_base/tests/test_optimized_multichip_decoder_agmm_decomposition.py
```

The first current-API call rejected rank-2 local-output weights. They were
adapted to rank-4 weights and the input sharding was matched to the proven TP4
AGMM layout; the retry passes eager correctness and traced replay. Both O and
down alternatives do local BF16 residual addition and leave a hidden-sharded
mesh residual, with no replicated restore. Explicit BFP8 async AG+local-output
matmul totals 0.173024 ms; fused BFP8 AGMM totals 0.149221 ms; selected
row-parallel matmul+persistent BFP8 AR totals 0.089642 ms. Fused O is nearly
tied at 0.033311 vs 0.032602 ms, but fused padded down is 0.115910 vs 0.057040
ms. Fused PCC versus selected is 0.999879/0.999154 and all traces are bitwise
deterministic. This closes the alternative output-projection decomposition
without measuring an immediate restore.

The final selected-topology precision matrix independently varied weight
dtype/fidelity by attention and MLP. Batch-1 traced results were BFP8
HiFi2/LoFi 0.401050/0.320416 ms, attention-BFP8/LoFi 0.395538 ms,
MLP-BFP8/LoFi 0.325894 ms, BFP4 attention HiFi2 0.308197 ms, and BFP4 MLP
HiFi2 0.389421 ms. All-BFP4/LoFi remains fastest. The activation matrix then
held those weights/fidelity and topology fixed: attention-only BFP8 is
0.295602 ms, MLP-only BFP8 is 0.293329 ms, and combined BFP8 is 0.286602 ms
with HF decode PCC 0.999998147. Combined BFP8 became the default and was
reproduced at 0.286597 ms in the final exact run.

## Shard advisor

The mandatory advisor capture uses exact local-rank decode dimensions at
batch 32. The bootstrap required the embedded tt-mlir `tt-metal` library on
`LD_LIBRARY_PATH`; this is recorded in the compressed
`shard_advise/pipeline.log.gz`.

```bash
python models/autoports/tiiuae_falcon3_7b_base/doc/optimized_multichip_decoder/shard_advise/advise_falcon3_tp4_local.py
```

The report contains 23 ops, 20 choices, and one spill. Its proposed family
includes QKV 11x4, O 11x9, gate/up 11x9, down 11x9, and a 96-core residual.
The exact report family measures 0.644761 ms decode; keeping the old residual
inside the advisor matmul family measures 0.641023 ms. Both lose to the
starting and final DRAM-sharded paths. The advisor remains executable through
`decode_matmul_mode="shard_advisor"` and
`advisor_residual_mode={"report","legacy_32core"}` for provenance, but is not
the default.

## Correctness and stress commands

Final correctness artifacts were generated with the default source and real
layer-14 weights. The final non-Watcher suite command was:

```bash
FALCON3_RUN_MULTICHIP_TOPOLOGY=1 \
FALCON3_RUN_MULTICHIP_FRACTURED_BOUNDARY=1 \
FALCON3_RUN_MULTICHIP_FRACTURED_SELECTED_POLICY=1 \
FALCON3_RUN_MULTICHIP_HETEROGENEOUS_POSITIONS=1 \
FALCON3_RUN_MULTICHIP_LONG_PREFILL=1 \
FALCON3_RUN_MULTICHIP_MAX_CONTEXT=1 \
FALCON3_RUN_MULTICHIP_BASELINE=1 \
FALCON3_MULTICHIP_RESULTS_DIR=models/autoports/tiiuae_falcon3_7b_base/doc/optimized_multichip_decoder/results/final_correctness \
pytest -q -rA models/autoports/tiiuae_falcon3_7b_base/tests/test_multichip_decoder.py
```

Result: 10 passed and the two separately executed perf/profile gates skipped
in 72.63 seconds. Every generated JSON has the final implementation and test
hashes above.

Final selected metrics are listed in `README.md`. Stress matches the changes:
unaligned paged cache crosses a page boundary; heterogeneous positions test
per-user RoPE/cache behavior; trace replays are deterministic; the 1,025-row
test crosses the internal MLP chunk; the full 32,768 prefill and final decode
prove context capacity; stacked decode proves the inter-layer layout.

## Final performance and profiler commands

```bash
for batch in 1 32; do
  FALCON3_RUN_MULTICHIP_PERF=1 \
  FALCON3_MULTICHIP_PERF_BATCH=$batch \
  FALCON3_MULTICHIP_RESULTS_DIR=models/autoports/tiiuae_falcon3_7b_base/doc/optimized_multichip_decoder/results/final \
  pytest -q models/autoports/tiiuae_falcon3_7b_base/tests/test_multichip_decoder.py::test_warmed_multichip_trace_performance
done

FALCON3_RUN_MULTICHIP_PROFILE=1 python -m tracy -r -p -v \
  -o models/autoports/tiiuae_falcon3_7b_base/doc/optimized_multichip_decoder/tracy/final_default \
  -m pytest models/autoports/tiiuae_falcon3_7b_base/tests/test_multichip_decoder.py::test_multichip_profile_signposts

tt-perf-report models/autoports/tiiuae_falcon3_7b_base/doc/optimized_multichip_decoder/tracy/final_default/ops.csv \
  --start-signpost MULTICHIP_PREFILL --end-signpost MULTICHIP_PREFILL_END --no-color

tt-perf-report models/autoports/tiiuae_falcon3_7b_base/doc/optimized_multichip_decoder/tracy/final_default/ops.csv \
  --start-signpost MULTICHIP_DECODE --end-signpost MULTICHIP_DECODE_END --no-color
```

Final exact-default results:

| Batch | Warmed prefill | Traced warmed decode | vs single chip | TP efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.902849250 ms | 0.286596501 ms | 2.2472x | 56.18% |
| 32 | 3.047774080 ms | 0.491488329 ms | 1.5636x | 39.09% |

Batch-1 single-call prefill is noisy and the selected changes are decode-only.
The controlled source-identical prefill A/B used 50 synchronized executions
per sample: persistent resources measured 0.906946/0.857275 ms, while
standard/no-buffer measured 0.861957/0.887350 ms. The distributions overlap;
persistent allocation does not explain the historical 0.822900-ms sample.
The official table still reports the final default's own five-sample run.

The final profiler capture was regenerated after the code settled. Stable
artifacts are `tracy/final_default/ops.csv`, `prefill_perf_report.{txt,csv}`,
and `decode_perf_report.{txt,csv}`. Advice was acted on as described above;
HiFi2/HiFi4 accuracy advice was also tested as complete precision policies.
The merged prefill table reports 441 us device work, 1,005 us gaps, and
14.1%/72 GB/s modeled DRAM roofline. Three decode trace replays report 825 us
device work, 144 us gaps, and 22.3%/114 GB/s. The decode table shows BFP8
projection outputs, BFP8 gated down input, and BFP8 async all-reduce rows.

## Watcher and fallback audit

The source-level fallback test inspects all hot methods for `torch`,
`from_torch`, `to_torch`, and `OptimizedDecoder.`. It passes. The final
serialized Watcher suite ran the default correctness test plus topology,
fractured boundary, heterogeneous positions, long prefill, max context, and
direct optimized-baseline checks. Performance and Tracy tests remained manual
and were run separately.

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
FALCON3_RUN_MULTICHIP_TOPOLOGY=1 \
FALCON3_RUN_MULTICHIP_FRACTURED_BOUNDARY=1 \
FALCON3_RUN_MULTICHIP_FRACTURED_SELECTED_POLICY=1 \
FALCON3_RUN_MULTICHIP_HETEROGENEOUS_POSITIONS=1 \
FALCON3_RUN_MULTICHIP_LONG_PREFILL=1 \
FALCON3_RUN_MULTICHIP_MAX_CONTEXT=1 \
FALCON3_RUN_MULTICHIP_BASELINE=1 \
FALCON3_MULTICHIP_RESULTS_DIR=models/autoports/tiiuae_falcon3_7b_base/doc/optimized_multichip_decoder/results/watcher \
pytest -q -rA models/autoports/tiiuae_falcon3_7b_base/tests/test_multichip_decoder.py
```

Result: 10 passed, 2 skipped in 80.84 seconds. `watcher_raw.log` has no match
for error/fatal/exception/hang/timeout/assert/NoC-failure/stuck-waypoint.
The final `tt-smi -s` at 19:43:05Z reports four healthy p300c DRAM devices,
zero corrected/uncorrected GDDR errors, and no reset requirement.

## Completion checklist

- [x] Measured final path is TP4 on the target `1x4` mesh.
- [x] Prefill, decode, K, and V PCC preserve the accepted dense-layer baseline.
- [x] Warmed prefill and traced warmed decode reported before and after.
- [x] Operation-topology audit completed before local tuning.
- [x] Human and CSV `tt-perf-report` outputs preserved.
- [x] Actionable profiler advice implemented and measured.
- [x] Async CCL, fused CCL/matmul, buffers, placement, link, residual,
  activation, DRAM-sharded matmul, precision, and fidelity families closed.
- [x] Lower-movement residual family measured through distributed norm and
  the next real projection.
- [x] No gather, reshard, layout conversion, or all-reduce between selected
  decoder layers; residual contract documented.
- [x] Non-aligned public sequence and paged-cache behavior preserved.
- [x] 32,768-token context preserved with physical execution evidence.
- [x] Runtime fallback audit and risk-matched stress pass.
- [x] Watcher-clean evidence and post-run device health preserved.
- [x] MoE gate marked not applicable because the model is dense.
- [x] Independent `$stage-review` returned `clean-pass`; see `stage_review.md`.
- [ ] Local stage commit: recorded after creating the commit.

No applicable optimization is deferred. Any independent review finding is
work until a clean re-review.
