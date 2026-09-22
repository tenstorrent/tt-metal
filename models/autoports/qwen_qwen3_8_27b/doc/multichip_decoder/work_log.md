# Multichip decoder work log

Stage 4, Qwen/Qwen3.8-27B, revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
Starting clean checkout: 1477a0eecf4f6654cebb4010aa89b852d2ca75fb.
Scope: multichip_decoder.py, stage tests and docs. No full-model/vLLM work.

- Read multichip, device usage, tracing and optimization skills, baseline code,
  baseline README/context contract, LLM report section 3.3 and common modules.
- Enabled plugin inventory: codex-home/config.toml explicitly enables both
  tt-autodebug and tt-model-bringup. Experiment wrapper runs packaged environment.py.
- `timeout 60 /home/mvasiljevic/tt-metal/python_env/bin/tt-smi -ls --local`
  lists four Blackhole p300c devices; repo python_env has no tt-smi entrypoint.
- `topology_initial.log`: 1x4 FABRIC_1D mesh opens/closes successfully; UMD reports
  P300_X2 and four degree-two vertices. Unknown B850M-C motherboard warning
  falls back to PCI bus IDs; successful topology mapping/open controls it.
- Recorded `mesh_plan.md` before implementation. TP4 ownership divides all
  head/intermediate dimensions. Compare replicated and hidden-sharded residuals.
- New runner initially used unsupported `device_ids`; corrected to
  `physical_device_ids` per current distributed.py. No device opened in that failure.
- Exact subsequent commands and raw output are in commands.log and named logs.

Status: implementation and validation in progress; no performance/PCC pass claimed.

## Initial runtime investigations

- `replicated_l0`: prefill reached CCL, then Python Shape slice failed. Converted
  shape to list; reset_1 returned 0 and list_after_reset_1 saw all four chips.
- `replicated_fixed_l0`: prefill completed. Decode failed in native
  get_worker_noc_hop_distance: multi-reader bank assignment requires a unit mesh.
  AutoFix invoked; fresh xhigh source-only agent `dram_mesh_diagnosis` investigates
  model-local alternatives. reset_2 launched after process/device teardown.

- `reader1_l0` passes prefill/decode and local-state parity (min0.999957),
  repeated and changed-input trace parity. Decode0.7023ms vs baseline0.8268ms.
- `reader1_l3` passes (KV PCC1.0); decode0.5833ms vs baseline0.6602ms.
- `sharded_l0` carries width1280 residual through add/distributed norm and
  gathers only normalized projection inputs. Passes but decode0.8118ms,
  slower than replicated control0.7023ms; prefill1.7515ms vs1.9776ms.
- Initial changed-input runner allocated after capture, producing allocator
  warning. Moved all refresh buffers and eager reference work before capture;
  `reader1_l3` and `sharded_l0` have no such warning. No precision change.
- Fresh diagnosis written to AUTODEBUG_dram_mesh.md. AutoFix hypothesis agent
  verifies raw both-kind results and sets one-reader default; exact multi-reader
  factory requires native changes outside the authorized stage scope.

## Profiler findings and topology tuning

- AutoFix's one-reader defaults are integrated. No native source changed.
- Ring_l0 completes both PCC and trace gates, decode0.6828ms. A wrapper edited
  while its child ran produced a post-child shell parse error; the raw child
  JSON and normal mesh teardown prove runtime success. `bash -n` passes the
  finalized wrapper. Subsequent wrappers are left immutable while running.
- Initial profiler launch lost JSON quoting inside Tracy's child shell. Added
  --policy-file, then profile_ring_l0 completed with operation CSV and tables.
- Default merged tt-perf-report selected rows across devices and introduced a
  spurious670us cross-device gap. `tests/multichip_profile_tables.py` preserves
  per-device rows and signposts. Device0 shows620us kernels+60us gaps versus
  same-run0.713ms host trace latency. This resolves the merged-gap anomaly.
- Real profile rows confirm BFP4/LoFi: attention88us, gate/up89us each,
  output29us, down43us. Each RS is16us and AG19us. Projection geometry is
  therefore the first tuning target; CCL is material but not dominant.
- geometry16_l0 combines larger blocks with carried residual, passes but
  loses0.8606ms. The serialized sweep separates geometry, carry, minimal
  matmul, CCL precision, and stack-compatible sharded residual families.

- AGMM adapted sender grouping succeeds:0.8428ms, minimumPCC0.999942.
  MMRS succeeds0.7531ms with the same sharded residual/consumer contract.
- Persistent CCL's first call used the wrong kwarg for the cluster-axis overload:
  persistent_output_tensor is required when mesh_device is supplied. Corrected
  that exact overload mismatch and reset_4/list_after_reset_4 completed.
- Updated harness allocates all changed-input buffers before trace, changes
  page tables and batch positions, and adds a two-layer linear/full fixture.
  New fixture names distinguish continuation and stack; baseline_contract_l0
  regenerated the baseline for this expanded input contract.
- tune_output4/6/12 pass, with decode0.5292/0.5234/0.5272ms vs control0.5540ms.
- tune_down34 is an L1-rejected candidate: native circular buffers end at
  1548800 while live allocation begins1308416 on[0,0]-[7,9]. Input storage is
  four cores with[32,1088] BF16 shards. Down block17/8-core is retained pending
  further tuning. reset_5 follows clean process teardown; no hang observed.

## Combined topology and geometry closure

- Sweep precision stays BF16 activations/BFP4 projections/LoFi unless the
  candidate explicitly tests CCL BFP8 or HiFi2. Real checkpoint weights and
  recorded HF inputs are used throughout. `candidate_summary.json` indexes
  actual per-run JSON and policy, including unsuccessful speed candidates.
- Packed gate/up lowers launch count: tune_packed_l0 0.5137ms. Combining
  output block6, persistent CCL, and two links reaches 0.4855ms in
  combined_packed_links2_l0. Final default reproduction is still required.
- Output block4/6/12 and projection block4/8/16/32 were tested; down34 hits
  an exact L1 capacity assertion. HiFi2 passes but loses to LoFi (0.6800ms).
- The BFP8 fused-AGMM family initially inherited carry_residual=True, whose
  optimized implementation assumes full hidden5120. Hidden1280 cannot reshape
  to5120. This is a policy incompatibility, not an AGMM op failure. Adapt the
  candidate to carry_residual=False, keeping the residual hidden-sharded through
  the normal add/distributed-norm consumer. reset_6 and device list succeed.
- Expanded runner compares both stacked layers' states and changed-input cache
  or recurrence state, with bitwise eager-versus-trace state equality. At large
  context it samples first/end physical pages plus every unowned page and
  compares complete output tensors. Page allocation caps at the advertised
  context, rather than adding two virtual tokens beyond capacity.

## Default and strengthened cache validation

- selected_candidate_l0/l3 reproduce0.4857/0.3706ms versus baseline_state_l0/l3
  0.8211/0.6610ms. Final default adopts that policy and avoids allocating unused
  gate/up copies. Final runtime SHA234077ed508b95ef959ee68bead94e6ab19e22fb226b389091413f932e5ac5c2.
- AGMM BFP8 failed due to a native raw-byte dtype mismatch: requested BF16
  output also determines its gather buffer, but input was BFP8. AutoFix source
  diagnosis plus dtype=xx.dtype and explicit BF16 result cast fixes it.
  agmm_bfp8_dtype_fixed_l0 passes0.8469ms. MMRS with genuine BFP8 output/RS
  payload passes0.7572ms (mmrs_bfp8_payload_l0). Both retain hidden-sharded
  residuals through distributed norm and include packed MLP. They remain slower.
- Initial regression failed full-attention S1 whole-cache PCC while complete
  output PCC exceeded0.999999. `cache_padding_failure.log` and
  `regressions_padding_failure.log` preserve the failure. The native paged fill
  writes a physical32-row tile; future rows beyond logical prefix are not
  semantic state. Baseline future cache rows also contained nonzero padding
  (max6.5 at row1); AutoFix verified the exact source contract.
- Corrected cross-baseline cache comparison to valid prefix rows, retaining
  bitwise raw-cache replay and exact unchanged-row checks. Changed-position
  tests now overwrite initialized prefix rows (n-1 minus per-user offset),
  and move physical cache contents with changed page tables. This removes the
  former synthetic hole at n and preserves logical history. reset_8/list pass;
  the complete regression matrix reruns with refreshed baseline fixtures.

- Refreshed regression matrix passes28/28 in571.29s (`regressions.log/xml`),
  covering both layer kinds, all planned tails/continuations, B3/B8/B32, and
  linear->full stacked execution for B1 and B3. No runtime implementation
  change was needed for the cache-padding investigation; fixture semantics
  were corrected and documented in AUTOFIX_cache_padding.md.
- Capacity planning raises trace/activation reserve from2GiB to9GiB: a full
  prompt hidden tensor is2.5GiB, and input/output/concatenation may retain three
  streams. New planned total20,158,234,624 bytes/device leaves14,020,496,384
  bytes headroom. The hardware probe holds the full plan additionally to its
  real layer/inputs/cache, in <=1GiB slabs to avoid giant tensor shape arithmetic.
- The capacity runner separately snapshots the normal final-position decode
  state, so last-row KV writes are compared along with complete outputs.

- The first max-context multi-chip prefill completed, but a warmed rerun OOMed
  at concatenation with the prior2.5GiB output still live plus a full9GiB
  activation budget duplicated in the empty reservation. Preserved in
  capacity_overreservation_failure.log and capacity_overreservation_runs.log.
  Reset9/list complete. The runner releases previous device outputs before
  repeat and reserves only full-model persistent memory plus2GiB trace/CCL;
  real full-length activation buffers are already present in the experiment.
- Final planned trace/activation headroom is12GiB; tiled norms and conv taps
  receive a corrected128MiB reserve. Total23,496,900,608B/device; extra probe
  reservation12,759,482,368B/device. No sequence/context reduction. Source-only
  AutoFix reviews tensor lifetime and concat working memory independently.

- All eight single-layer capacity runs pass; both kinds retain262144-token
  prefill and final valid decode position262143. Linear prefillPCC0.999996465,
  full-attention prefillPCC0.999999183; final-position output/state comparisons
  and changed-input replay pass. `capacity_runs.log` indexes these controls.
- Native unaligned concat retains TILE chunks, row-major copies, row-major
  output, and retiled output simultaneously. The conservative final budget
  includes six hidden streams for retained original stack input,2GiB trace/CCL,
  and1GiB scratch:18GiB activation/trace total. Full plan29,939,351,552B leaves
  4,199,336,960B versus actual post-trace DRAM allocator34,138,688,512B.
- The two-layer stack at262143 also passes, with the original full prompt held
  while full attention consumes linear output and all planned persistent
  model memory is reserved. Artifacts: capacity_stack_s262143{,_baseline}.json.
  This directly exercises the retained-input lifetime in the revised plan.
- Watcher initially fails during mesh-open fabric initialization, before model
  code: ACTIVE_ETH program29072B exceeds config buffer26624B. Preserved in
  watcher_fabric_size_failure.log and matching directory. Per optimize skill,
  retry with TT_METAL_WATCHER_DISABLE_ETH=1, retaining worker/NoC instrumentation.
  Reset10 follows normal driver teardown; this is a scoped instrumentation
  limit, not a model runtime corruption or a context-capability reduction.

- Four watcher cases pass with source-built worker/NoC instrumentation and
  ETH-only exclusion: linear2049 tail, B32 linear, B32 full, and B3 stack.
  `validation_summary.json` indexes28 regressions, four watcher runs, and five
  mesh capacity artifacts; minimum PCC0.9998691678. Watcher logs explicitly
  show all four devices checked and disabled features ETH.
- Final profiler's broad validation capture overflowed per-core DRAM marker
  buffers; runtime completed, but enrichment failed for missing markers.
  `profile_marker_overflow.log` and `tracy/profile_marker_overflow` preserve it.
  Reset11/list succeed. Profile-only mode now uses one warmed repeat and
  ReadDeviceProfiler flushes outside signposted windows. Non-profiler benchmark
  repetition remains unchanged; no invalid-duration data is used for claims.

## Final combined collective experiment

- Packed MLP + direct all-reduce measures0.47299ms linear decode versus the
  previous RS/AG default0.48735ms (`direct_ar_packed.json`). The selected
  hidden-sharded packed family remains slower: BF16 CCL0.59902ms and BFP8
  CCL0.60921ms (`sharded_packed{,_ccl8}.json`).
- Selected direct all-reduce uses a model-scoped shared TT_CCL workspace,
  avoiding64 duplicated16KiB/core allocations. Mesh identity is checked.
  The contract permits a single ordered CQ0 model stream; no concurrent
  independent traces may share the workspace. Source audit is recorded in
  AUTOFIX_shared_ccl.md, including the existing Llama shared-buffer precedent.
- B32 full-attention changed-input/page-table/position parity and trace pass
  (`ar_batch32_full.json`). Final source adds the mesh ownership guard and
  refreshes all hardware gates. The new optional queued-stress harness compares
  100 evolving-state eager calls against100 queued trace calls through B32
  linear/full layers with the exact same workspace object.

- Final direct-AR regressions28/28 pass in490.709s, sourcee36087219b20d68d515d15e9cadc82fc3b38413d1b53a96dcc8520024f0af503.
- New B32 stack stress first fails in single-chip eager oracle after capture: retained trace L1 allocations clash with ChunkGdnPrep CB (1103872 vs1115136). Preserved stress_baseline_trace_lifetime_failure.log. Moved eager evolving-state oracle before capture and explicitly released each prior output; TP4 kernel had not run. Reset12/list then rerun the same100-iteration test.

- Coherent packed direct-AR BFP8 candidate adapts input, persistent workspace,
  and native output dtype together, then casts the residual back to BF16.
  `direct_ar_packed_bfp8.json` passes PCC and initially ties0.4728ms. Paired
  50-replay controls resolve the variance: BF16 medians0.46848/0.46639ms, BFP8
  0.47894/0.47468ms. Both BFP8 runs pass. BF16 remains selected; experimental
  sourcebac850be386a1540383a605d406180278cec4d9f39a117fc80e0e398ec08ebad
  is saved in sources/, while final sourcee360872 is restored.
  Exact samples/IQR are in direct_ar_dtype_ab_*.json and
  direct_ar_dtype_comparison.json. The final direct-AR branch uses BF16;
  `ccl_dtype` applies to RS/AG and fused experimental branches in that source.

- B32 TP4 stack then exposed a real L1 layout limit before stress/capture:
  second-layer packed MLP static CB ends1338368B, dynamic allocation starts
  921344B. See stack_batch32_l1_failure.log and AUTOFIX_stacked_batch_layout.md.
  Native public [B,1,H] TILE conversion expands packed rows to B*32, retaining
  multiple10MiB L1 tensors through the next layer at B32.
- Reset13/list and mesh smoke succeed. Policy public_dram_batch=2 uses DRAM
  for the existing batched public conversions, preserving compact internal L1
  operations and unchanged B1 layout. stress_stack_batch32_dram passes100
  evolving eager and100 queued trace calls, bitwise output/state, final trajectory
  output PCC0.99998122 and recurrence0.99982654. No batch reduction.
- Final default sourcee8898b5a80cd8f182c82a9fd94d194d6636522378b003cca37552c66770f02ce.31-case regression refresh reuses existing baseline
  fixtures only when their recorded optimized source hash matches the unchanged
  baseline; new B2/B8/B16 stack cases generate fresh baseline fixtures. Command:
  QWEN_REUSE_OPTIMIZED_BASELINE=1 python_env/bin/pytest -q -x --confcutdir=models/autoports/qwen_qwen3_8_27b/tests models/autoports/qwen_qwen3_8_27b/tests/test_multichip_decoder.py --junitxml=models/autoports/qwen_qwen3_8_27b/doc/multichip_decoder/regressions.xml

## Final acceptance refresh

-31/31 regressions pass326.88s. Final B32 stack stress passes100 evolving
  eager and100 queued trace calls, including final-state/output PCC against
  the frozen optimized reference.
- All five final TP4 capacity cases pass with12,759,483,008B/device actually
  reserved plus real layer weights/activations/state. Both kinds retain262144
  prefill, final valid decode position262143, and the full retained-input stack.
  context_contract.json and memory_capacity_plan.json are validated at final
  runtime sourcee8898b5a80cd8f182c82a9fd94d194d6636522378b003cca37552c66770f02ce.
- Four final source-built worker/NoC watcher cases pass; only ETH is excluded
  for the previously proven fabric configuration-size blocker.
- validation_summary.json passes with minimum PCC0.9998265382 including
  stress (31regressions +4watcher +5capacity +1queuedstress).
- Final warmed medians (20prefill/50trace): linear single2.04161/0.82293ms,
  TP4 1.29714/0.46539ms; full single1.71578/0.66192ms, TP4 1.22896/0.35063ms.
  Decode speedups1.7682x/1.8878x, efficiencies44.21%/47.19%.

- All four final profiler runs pass with no dropped-marker/enrichment failure.
  Device0 decode: linear425.601us kernels +49.185us gaps; full315.055 +40.875us.
  Direct AR totals31.583/31.641us; matmul188.322/180.946us; movement82.841/39.348us.
  All dominant rows confirm BF16×BFP4/LoFi. Final utilization, SLOW output-row
  investigation, native subblocks, and non-failing metadata/UI-copy diagnostics
  are recorded in profiler_interpretation.md. Per-device tables avoid merged
  clock artifacts; performance_summary.json uses uninstrumented warmed medians.
- Final tt-smi list sees all four p300c devices (final_device_health.log).
  Pre-commit passes on stage Python/shell files; host AST/bash syntax checks
  pass. No C++/CMake changes, so no build is required.

## Independent review and checkpoint

Fresh xhigh stage-review returned **clean-pass**, with no required work
remaining. Report: stage_review.md. It independently checked final source and
runner hashes,31 regressions, queued B32 stress, five capacity probes, four
watcher runs, final profiler rows/benchmarks, and archive provenance. Review
findings were repaired and rereviewed before this verdict.

Only stage-owned multichip runtime/tests/docs/context-contract changes are
included in the local checkpoint. The optimized single-chip baseline remains
unchanged. No full-model or vLLM implementation was begun; no push is authorized
or performed.

- The commit hook normalized whitespace/newlines in76 generated text artifacts. Original byte-exact copies are preserved under SHA256-addressed archive paths in raw_archive_manifest.json; normalized repository text has identical whitespace-token content. Runtime/tests were unchanged. The first commit attempt stopped on these automatic formatting edits; files are restaged for the same checkpoint.

### Local checkpoint

| Repository | Branch | Stage checkpoint SHA |
| --- | --- | --- |
| tt-metal | mvasiljevic/qwen38-full-bringup | `226db8ef4347b4485d131cc113aa5199d2dc995e` |

Command: `git commit -m 'Add Qwen3.8-27B TP4 multichip decoder and validation'`.
All commit hooks pass; the hook transcript is checkpoint_checks.log.
This checkpoint contains the independently reviewed implementation and evidence.
A following docs-only commit records this SHA and completion status. No push.

Stage state: **multichip-decoder complete**. Full-model and vLLM work remain
separate future stages.
