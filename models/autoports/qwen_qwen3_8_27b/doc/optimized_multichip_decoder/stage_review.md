# Stage Review

Verdict: clean-pass

Independent final review of stage 05, `optimized-multichip-decoder`, Qwen/Qwen3.8-27B, completed 2026-10-24. This verdict covers the frozen v5 TP4 decoder on four Blackhole p300c ASICs, mesh 1×4. The reviewer inspected the live worktree on `exp26-qwen38-bringup`, based on `ad9ed5d3c46f649ae4a0fa0804e899e7999d8180`. No device experiments, implementation edits, or child reviews were performed by this reviewer. This report supersedes preliminary review findings; it does not approve full-model or vLLM behavior.

## Required Work

None. The earlier material gaps in prefill CCL families, precision-matched projection geometry, native RS grids and role-specific K blocks, repeated RoPE operations, and affected branch coverage have been closed with source inspection, adapted candidates, measurements, and fresh final-source acceptance.

Local checkpoint commits and recording their SHAs follow this verdict under the stage-review workflow. They are not claimed to exist at review time.

## Other Concerns

- Final acceptance contains 67 distinct cases. I independently read every case JSON, checked its recorded source hashes against the frozen files, and recomputed the minimum over output and available state PCC values. All values are finite and exceed the inherited 0.995 gate. The minimum is **0.9993398185689135**, the value-cache PCC in `packed_prefill_multichunk_b8_l3.json`. Every case reports a passing dynamic fallback audit and three exact restored replays.
- Coverage includes both meaningful layer kinds; B1/4/8/16/32 at 4096+128; odd batches and ragged/remapped pages; logical lengths 1/2/3/31/32/33/129/2049/4095/4096/4097; continuation; both mixed-layer stack orders at B1/8/32; native 262144-token capacity; and native-context runs with another 2 MiB of L1 allocation. The final B8/S4097 case specifically covers packed prefill followed by a one-token second chunk and eight cache-consuming decode steps.
- The production-helper probes pass 18 exact convolution/history checks, 54 exact decode-RoPE Q/K checks, and 14 exact prefill-RoPE Q/K checks. Checks include all four ASICs, odd batches, distinct/permuted positions, and untouched rotary tails. The prefill packing identity proof deliberately uses uniform FP16 rotary accumulation; original production precision is independently covered by complete-layer PCC, as explained in the anomaly ledger.
- Dedicated lifetime probes total **2524 nonblocking replays**: workspace 1514, mixed decode 324 per layer kind, and mixed prefill/decode 362. Allocation tracking includes program-cache allocations. These probes use no corruptible-buffer acknowledgments; survivor traces remain exact after other traces retire. The mixed-prefill probe restores empty state for prefill and separately captured filled state for decode, so it exercises meaningful state transitions. Six additional mixed-stack cases use a narrowly identified `traced2` output acknowledgment with an explicit consume-before-next-trace invariant.
- All four watcher files contain completed polls and detach records for devices 0–3, without assertion or corruption findings. Watcher and profiling runs are separate. No unclassified warning or error family was found in the 67 accepted logs.

The final device timings below were independently recomputed as the maximum of the four complete per-device firmware envelopes, using each profile's reported clock. They include operations and gaps; they do not subtract clocks across ASICs or sum four devices' latency. Final operation CSV hashes and source provenance match their receipts.

| Workload | Linear prefill µs | Linear decode µs | Full-attention prefill µs | Full-attention decode µs |
|---|---:|---:|---:|---:|
| Before, B1, 4096+128 | 8237.612593 | 509.787407 | 10626.325185 | 421.190370 |
| Final v5, B1, 4096+128 | 5667.222222 | 460.834815 | 5919.607407 | 368.265185 |
| Final v5, B32, 4096+128 | 211522.113333 | 2035.431852 | 187301.459259 | 1010.193333 |

B1 prefill improves 1.454×/1.795× and decode 1.106×/1.144× for linear/full attention. Final ordinary host prefill/decode medians are separately reported as 5870.45/445.78 µs and 6127.69/359.46 µs. Final defaults reproduce the paired candidate prefill selection. B32 linear's last grid change is effectively flat in device profiles. B32 full prefill is also effectively flat versus archived v4 (+0.19%); the small paired host improvement does not establish a device-prefill improvement. Full B32 decode improves from archived v4's 1686.13 µs to 1010.19 µs.

Actual final rows confirm BFP4 weights and LoFi matmul descriptors, the selected ordinary decode readers, separate prefill gate/up with fused gate SiLU and one shared gather, and selected role configurations. B1 native prefill RS has 108 participating cores (88 MM + 20 CCL), 11×8/N15, H4/sub2×3, outK48/downK17. B32 native decode RS has 22 participating cores (10 MM + 12 CCL), 10×1/N16/sub1×4, outK6 linear/outK24 full and downK17. Every B32 ASIC has one linear decode convolution or two full-attention RoPE operations; full-attention prefill also has two RoPE operations. Forty compiled-kernel descriptor references in the runtime audits were hash-checked.

The applicable optimization families have evidence, rather than rejection at a first API error:

| Family | Independently inspected closure |
|---|---|
| Residual layout and collective placement | Replicated DRAM/L1, hidden-sharded residuals consumed by distributed norm, direct async AR, RS, gathered-input/local-output projections, activation/CCL precision, and persistent/transient storage are compared as coherent families. Sharded candidates are not rejected by immediately restoring the old residual contract. See `topology_audit.md`, `residual_contract.md`, and candidate records. |
| Async CCL and owner lifetime | Two-slot AR ownership and preallocated decode/prefill RS pools are shared across layers; complete out/down invocation ordering, failure invalidation, preparation before capture, and retention until trace retirement are implemented and stress-tested. |
| Fused matmul/CCL | Native reader defects were investigated; full-width output blocks and interleaved/padded weight adaptations produce valid comparisons. Native RS is selected. Adapted fused AG and coherent L1 alternatives pass but lose complete-layer latency. See `AUTODEBUG_fused.md`, `AUTOFIX_fused.md`, and prefill triage/fix records. |
| Packed/separate projections and graph work | Real split-mix, packed versus separate gate/up, fused gate SiLU, minimal SwiGLU, shared-AG consumers, and fused AG+gate sharing its gathered output were tested. The selected separate prefill pair retains packed decode weights. |
| DRAM-sharded decode geometry | Precision-matched storage-core counts, wider legal K divisors, readers 1/2/3, owned padding, and whole-layer confirmations support the selected ordinary matmuls. `matmul_search.csv` has 188 rows. |
| Prefill projection geometry | Selected BF8 inputs/CCL and BFP4/LoFi weights are retained across larger K, output-block, subblock, grid, and padded/interleaved adaptations. Larger legal endpoints were not dismissed using incompatible BF16-input capacity evidence. See `prefill_projection_geometry.md` and selection records. |
| Native RS geometry | 8/10/11-wide grids and role-isolated K/H/subblocks were measured. Larger downK68/K136 decode endpoints have source-derived native-CB capacity bounds; even N15 requires 1,484,288 bytes before persistent allocations, above the reported 1,461,120 allocatable bytes/bank. K34 is legal and measured. |
| Repeated convolution and RoPE | Packed convolution and decode/prefill RoPE include packing, extraction, state, and layout costs in complete-layer comparisons. Threshold controls preserve the original loop where it wins. Exact intermediate proofs and final-source acceptance cover the selected paths. |
| Precision and attention | BFP4/LoFi weights, BF8 communication/KV, FP32 recurrence, and LoFi/FP16 SDPA have real-weight output/state evidence. SDPA chunk/core/fidelity candidates and explicitly mapped long-context pages were tested. |
| Context and serving-oriented memory | Native context is preserved. Extra separate weights, shared pools, L1 use, and transient RoPE allowance are accounted for. The B32×32768 full-stack planning estimate is 30,879,801,344 bytes/device against 34,178,731,008 physical bytes, including 12 GiB activation/CCL and 1 GiB trace reserves. This is correctly labeled a plan, not a full-model allocation measurement. |

All evidence links in 961 general candidate rows, 500 prefill rows, and 188 geometry rows resolve to preserved files. The local manifest has 1766 existing artifacts with matching recorded sizes, totaling 12,257,144,986 bytes. The telemetry packet's final device, roofline, and PCC values agree with the measured receipts; its review-pending completion note is appropriate until this verdict is recorded.

## Hard-Check Gaps

- The exact odd-batch full-attention mix placement transition B3/S1344→1345 (4032→4128 padded working rows) is not a paired boundary test. This is nonblocking: final B1/S4096 reaches the largest selected L1 working allocation, B3/S4095 exercises the DRAM arm, and S2049 exercises the nontrivial output-block divisor rule. Source inspection found no distinct untested transformation at that placement comparison. The new packed-prefill cross-chunk risk has separate direct coverage in case 67.
- The candidate search is not an exhaustive Cartesian product of every layer, batch, grid, and K value. The shared projection contracts, legal adaptations, isolated controls, paired repeats, and final-default measurements cover the material families. The report does not claim globally optimal geometry for every batch.
- Layer PCC against the accepted single-chip TTNN reference is not a full-model language-quality evaluation. Generation, vLLM trace reuse, serving throughput, and full-stack allocation remain outside this stage's explicit goal.

## Anomaly Ledger

- **Observed anomaly:** Initial native fused projections hang or return invalid values under some weight/output layouts.
  **Evidence:** `AUTODEBUG_fused.md`, `AUTOFIX_fused.md`, `AUTOTRIAGE_prefill.md`, `AUTOFIX_prefill.md`, `AUTODEBUG_column.md`, preserved failed logs and adapted candidate JSONs.
  **Affected path:** Native AG/MM, MM/RS, and larger-K prefill weight readers.
  **Control or comparison:** Valid interleaved/padded copies preserve quantized values; full-width output blocks produce passing whole-layer controls.
  **Likely subsystem:** Native descriptor/K ordering, output-width addressing, and prefill AG receiver reuse/row-base contracts.
  **Investigation performed:** Source and triage inspection followed by shape, layout, padding, output-block, chunk, and weight-packing adaptations.
  **Resolution:** Fixed for selected native RS; controlled for rejected AG alternatives by valid slower measurements. An initial API failure is not the rejection basis.

- **Observed anomaly:** Early L1 failures appeared to rule out coherent prefill placement; some larger-K endpoints also failed allocation.
  **Evidence:** Prefill source/harness receipts, `prefill_ccl_findings.md`, geometry logs, and `decode_rs_geometry.md`.
  **Affected path:** Prefill live activations and fused/ordinary matmul CB capacity.
  **Control or comparison:** Release unused eager outputs after their host read and retain only required warm/captured outputs; rerun coherent placement. Adapt legal H/K/grid values.
  **Likely subsystem:** Harness tensor lifetime for the former failures; actual per-bank capacity for the documented endpoints.
  **Investigation performed:** Alias/consumer inspection and fresh adapted measurements; native decode CB bounds were derived separately from prefill bounds.
  **Resolution:** Fixed lifetime artifact; legal families measured. Remaining endpoint rejections have measured or source-derived physical bounds.

- **Observed anomaly:** SDPA512 produced NaNs with an inadequately mapped padded page table.
  **Evidence:** `AUTODEBUG_sdpa.md`, `sdpa_pad_aligned512_l3.json`, effective-chunk fields, native-context runs.
  **Affected path:** Full-attention decode tails.
  **Control or comparison:** Explicitly mapped 144-page/4608-token capacity makes the 512 candidate pass, but it is slower than 256.
  **Likely subsystem:** Rounded kernel reads into undefined page-table padding.
  **Investigation performed:** Reader bounds and actual mapping capacity inspected; selected chunk guard respects explicit mappings.
  **Resolution:** Fixed. No public context reduction or borrowed pages.

- **Observed anomaly:** Mixed-stack second capture encountered a program-cache/lifetime problem; scratch reuse also required cross-trace ownership guarantees.
  **Evidence:** `AUTODEBUG_trace.md`, six `stack_b*_l*.json` cases, `final_workspace_lifetime.json`, mixed-trace probes, workspace source.
  **Affected path:** Retained traces and shared communication allocations.
  **Control or comparison:** Warmup with retained output before the second capture; exact output-only acknowledgment in the stack stress; dedicated no-acknowledgment lifetime probes.
  **Likely subsystem:** Program preparation/live allocation interaction. The exact missing C++ program hash was not captured, so the detailed source diagnosis remains a hypothesis.
  **Investigation performed:** Source lifetime/alias review, preparation before capture, alternating replay and survivor-retirement checks with program-cache tracking enabled.
  **Resolution:** Fixed operationally and controlled by final stress evidence; no whole-forward/capture exemption.

- **Observed anomaly:** Initial packed decode RoPE returned interleaved L1 Q rejected by SDPA.
  **Evidence:** Preserved original adapter, repaired candidate logs, `packed_rope_selection.json`, `final_packed_rope_exact.json`.
  **Affected path:** Packed full-attention decode.
  **Control or comparison:** Restore the original DRAM final-concatenation boundary; compare shared and separate tables against the per-user loop.
  **Likely subsystem:** SDPA input memory contract.
  **Investigation performed:** Kernel/API source inspection, repaired exact Q/K/tail probes, B2/B4 threshold controls, real-layer comparisons and final odd-batch tests.
  **Resolution:** Fixed. The optimization is selected for B≥4; positions remain distinct.

- **Observed anomaly:** Packed prefill RoPE slightly changes original-policy key/output PCC despite rowwise semantics.
  **Evidence:** `rope_optimization.md`, rotary factory source around lines 811–835, uniform-FP16 candidate proof, `final_prefill_rope_exact.json`, final full-attention acceptance.
  **Affected path:** Multi-tile prefill rotary embedding.
  **Control or comparison:** Uniform FP16 rotary accumulation gives 14/14 exact all-device Q/K/tail comparisons; original production policy passes complete-layer output/cache PCC. Decode packing remains exact.
  **Likely subsystem:** Upstream core_group_1 uses default compute settings while group 2 receives requested settings; folding sequence rows changes the groups' assignment.
  **Investigation performed:** Re-derived row/core assignment and kernel configuration; checked unchanged value-cache PCC and explicit intermediate identity under uniform settings.
  **Resolution:** Controlled numerical difference, not user-row mixing. Production precision remains unchanged; this is not a claim that the upstream discrepancy is repaired.

- **Observed anomaly:** Higher-precision attention weights can lower PCC against the existing BFP4 reference; minimal SwiGLU also fails the inherited gate.
  **Evidence:** Precision/combined candidate records and preserved numerical fields in `candidate_summary.csv`.
  **Affected path:** Rejected precision and fusion candidates.
  **Control or comparison:** Real target weights and recorded activations; complete output/state gates and whole-layer latency.
  **Likely subsystem:** Different quantization/accumulation relative to the accepted reference.
  **Investigation performed:** BFP8 and higher-fidelity combinations measured; the BFP8 value-cache PCC near 0.9941 is below 0.995 and the candidate is slower.
  **Resolution:** Controlled rejection. No faster passing low-precision candidate is rejected solely by random synthetic semantics.

- **Observed anomaly:** Fused profiler rows show scalar HiFi4, utilization above 100%, and generic advice to increase 108 of 110 cores.
  **Evidence:** Final CSVs/tables, `kernel_fidelity_evidence/`, both runtime audits, `perf_advice.md`, native program factories.
  **Affected path:** Fused-op report attribution/advice.
  **Control or comparison:** Exact compiled bmm descriptors encode LoFi; the reduction descriptor encodes HiFi4. Selected prefill uses 88 MM plus 20 CCL cores.
  **Likely subsystem:** Scalar report fields represent the last compute kernel and aggregate worker count.
  **Investigation performed:** Matmul attributes and descriptor hashes checked; MM/CCL placement re-derived. An 11×9 origin-anchored MM grid overlaps CCL at row 8; 12×8 exceeds the worker-grid width.
  **Resolution:** Controlled. Whole-layer rooflines use all four ASICs and appropriate useful-work/traffic estimates. The remaining two cores do not invalidate the measured 8/10/11-wide search; this is not proof that arbitrary kernel redesign is impossible.

- **Observed anomaly:** A final-profile attempt hit disk-full during CPU postprocessing, and external cleanup removed historical uncompressed operation/report CSVs and host Tracy data.
  **Evidence:** `profile_disk_failure_v4/`, retained before-profile timing JSONs, printed advice tables, logs, compressed device/timing CSVs, final four v5 captures, local manifest.
  **Affected path:** Evidence retention, not accepted device correctness.
  **Control or comparison:** All four final profiles were captured afresh; operation hashes, source hashes, and per-device envelopes were checked. Historical baseline numbers remain unchanged.
  **Likely subsystem:** Host storage/postprocessing.
  **Investigation performed:** Surviving artifacts audited; original failed attempt retained; final raw capture and compact report existence verified.
  **Resolution:** Controlled with an explicit historical-data limitation. Missing historical files are not fabricated or presented as present.

- **Observed anomaly:** One candidate's source hash was recorded after an adapter edit; moved archives and failed runs also left broken table links, and selected-overrides documentation became stale.
  **Evidence:** `prefill_source_provenance/`, archived receipts/source, candidate table resolver code, `selected_default_receipt.json`, final policy files.
  **Affected path:** Candidate provenance and summary presentation.
  **Control or comparison:** Imported source reconstructed and hash-verified; affected candidate rerun. Later candidates hash at import. Final defaults have independent frozen-source measurements.
  **Likely subsystem:** Evidence collection/bookkeeping.
  **Investigation performed:** Source archives and resolver targets inspected; all 1649 evidence-table rows now resolve; final acceptance and profile hashes independently compared.
  **Resolution:** Fixed. Selection does not depend on the ambiguous historical row.

- **Observed anomaly:** Profile-enabled and ordinary host timings differ; an archived B32 host run has an isolated tail spike.
  **Evidence:** Archived v4 and final v5 run JSONs, profile timings, README/work log.
  **Affected path:** Performance interpretation.
  **Control or comparison:** Same-regime paired ordinary timings and separately measured complete device envelopes.
  **Likely subsystem:** Measurement/runtime variability; the host spike's cause was not measured.
  **Investigation performed:** Regimes and individual samples compared; final reported numbers use current defaults. Flat B32 prefill/grid results are explicitly reported as flat.
  **Resolution:** Controlled. No unsupported cause or cross-regime speedup is claimed.

- **Observed anomaly:** Accepted logs include AICLK settling, rank-normalizing ND-view, motherboard discovery, and subset-MMIO warnings.
  **Evidence:** Accepted logs, UMD firmware acceptance near lines 235–254, `tensor_ops.cpp`, README/runtime audit, four watcher dumps.
  **Affected path:** Setup metadata and existing tensor-view representation.
  **Control or comparison:** Firmware accepts the observed 1331/1337/1343 MHz settling within 5%; views retain equivalent 2D sharding; TP4 profiles and PCC remain valid.
  **Likely subsystem:** Firmware polling, metadata normalization, and discovery/reference setup.
  **Investigation performed:** Source and all final accepted warning families inspected; watcher completion and all-device runtime rows checked.
  **Resolution:** Controlled. Nominal peak and profile-header clock conversion remain clearly labeled; these warnings are not treated as host/model fallback.

## Scope Inspected

- **Goal/skill paths:** Original `/home/mvasiljev/agentic-research-data/experiment-26/bringup-logs/qwen38-27b/20261023-astra-medium-exp26-clean/05-05-optimized-multichip-decoder.prompt.txt`; installed `stage-review`, `optimize`, `agentic-serving`, and `tt-device-usage` skills under `.codex/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.21/skills`, with startup and relevant tracing contracts.
- **Artifact paths:** This directory's README/work log, topology/residual/geometry contracts, candidate and prefill tables, selection receipts, before/final correctness JSONs, all 67 final case JSONs and logs, four watcher dumps, exact/lifetime probes, four final profile operation CSVs/tables/timing receipts, kernel descriptors, context check, memory ledger, local manifest, and preserved failure/source archives. Also `../context_contract.json` and telemetry packet `a07f750e-34af-4431-bb22-90aaafd34e60.json` in the experiment's telemetry directory.
- **Code paths:** `tt/multichip_decoder.py`, `tt/optimized_decoder.py`, `tt/decode_workspace.py`, `tt/fused_decode.py`, `tt/fused_prefill.py`; runner, validation, candidate adapters, exact/lifetime probes, profile extraction/audits, evidence summarizers, and performance reporting. Relevant native CCL/matmul, rotary, SDPA, tensor-view, allocator, and UMD sources were inspected for the reported contracts and anomalies.
- **Commands run:** Read-only `rg`, file reads, source/diff inspection, local Python artifact analysis, SHA256 comparisons, streaming zstd reads, CSV envelope/count calculations, manifest existence/size checks, and `git diff --check`. No reviewer hardware use or model execution. Only this report was written by the reviewer.

Frozen reviewed source identities:

| File | SHA256 |
|---|---|
| `tt/multichip_decoder.py` | `d999166db24800aec9c16001201f26baad683c76150f5401db4834e4c937e82e` |
| `tt/optimized_decoder.py` | `d355691a69232c06f4c072bdd417352b081f5bea3f64d603646c74071fb9f2ac` |
| `tt/decode_workspace.py` | `bdf291c2e95e6f31a0842076a5a3176989c675035aa460a8bfb572490c4f3d80` |
| `tt/fused_decode.py` | `aaabf3a36c6029bdfbc97ce1342ed3931cd4da729f6a4939dc789ce5db193038` |
| `tt/fused_prefill.py` | `80de6c24646acd943025cadab48cc70b3fc93e3c83c4d2716c1c2c81df1a18bd` |
| `tests/run_multichip_decoder.py` | `b0dae32ef8060644693ab295312956df84fe76fa1f2e280ae1b019c62f49c4ae` |

## Residual Risk

- The shared owner requires the documented serialized CQ and complete-invocation protocol, preparation before capture, and retention through all live trace retirements. Later full-model code must preserve these constraints and the residual layout rather than adding boundary conversions.
- Native context is demonstrated at B1 and larger batches separately at the stated workloads. The complete batch/context Cartesian product and full-stack memory occupancy are not measured here.
- The upstream rotary core-group precision discrepancy and native fused-AG reader limitations remain source-level limitations outside the selected working contracts. Later kernel changes require renewed numerical/geometry validation.
- Historical uncompressed profiling files lost during external cleanup cannot be claimed as preserved. Current final captures are complete; compact baseline reports and compressed raw device/timing evidence survive.
- Roofline percentages are modeled useful-work/compulsory-traffic ratios to nominal physical peak, not hardware utilization counters. Small timing differences and B16 prefill ties should not be presented as substantial gains.
