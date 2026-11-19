# AutoFix: resident PLE, recurrent precision and traced decoder

Status: repair in progress. Source diagnosis: `AUTODEBUG.md`.

The failing contract is a complete single-chip decoder with all PLE weights
resident, exact n-gram IDs, no runtime host fallback, and PCC >= 0.995. Native
BF16 PLE exceeds device capacity; native BFP embedding/gather does not supply
a usable compact lookup path. Context reduction cannot repair weight capacity.

## Discriminating experiments

1. **Compact integer payload with device gather.** Stripe packed UINT32 words
   over 32 tile rows and gather columns before selecting a stripe. The small
   `probe_packed_lookup.py` exact-word/nibble probe passes. Full-table-shaped
   allocation plus other-layer-weight dummy storage leaves 2626587648 bytes.
   This disproves a blanket physical-impossibility conclusion. Complete NF4
   lookup is still being exercised; the first run found use of padded volume
   in a logical reshape, corrected to the product of logical dimensions.
2. **NF4 quantization accuracy.** A CPU diagnostic over real rows selected
   across all 128 shards gives PLE PCC 0.995866–0.995867 for prefill and
   0.996223 for decode using absmax scales. LSQ scales improve these to
   0.996119–0.996120 and 0.996368. These are component diagnostics; complete
   TTNN decoder acceptance is not inferred from them.
3. **Exact hash without 64-bit device arithmetic.** Two UINT32 limbs plus
   integer remainder reproduce all valid token products and HF n-gram IDs.
   `ngram_hash_device.json` passes changed-input replay and independent history
   reset/continuation. The custom TTNN generic op keeps all runtime data on
   device and preallocates its buffers.
4. **Linear decoder trace correctness.** Real layer 0 passes 33-token prefill
   at PCC 0.9999147654 and three changed-input decode replays at PCC
   0.9999481440, 0.9999409914, and 0.9999201894. Allocation tracking is enabled.
   Only the exact sole-live-trace output is acknowledged; it is consumed
   before the next replay. This is an initial trace check, not all-path proof.

AutoFix subagents own separate hash/QSA and PLE arithmetic experiments;
hardware runs are serialized by the parent. No library edits, resets, or
remote changes are part of this repair. Final validation commands, remaining
risks and resolution will be recorded here after integration.

## Current discriminating results

- Full-table native gather completed normally after roughly 1791 seconds;
  live triage had shown continued progress. Direct striped NoC reads replace
  the full-source scan. Exact complete-table reconstruction, changed IDs,
  traced replay and watcher checks pass (`packed_embedding_full_direct.json`).
- Complete PLE decoding with absmax NF4 fails the first decode PCC gate
  (`ple_decoder_probe.json`). CPU controls reproduce the failure. LSQ row
  scales, smaller absmax groups and several affine/mixed-precision formats
  fail independent holdouts. Learned group-20 LSQ scales pass two CPU seeds
  and continuation holdouts (`ple_decoder_smaller_group_continuation.json`,
  `ple_decoder_group20_original.json`); exact device reconstruction also
  passes (`grouped_embedding_device.json`). Complete table preparation and
  full decoder validation with BFP8 expert storage remain required.
- Long GDN accuracy fails with the native fused recurrence, including when
  exact HF q/k/v/decay/beta operands are injected (`gdn_core_device_1025.json`).
  HF chunk-vs-recurrent and convolution-rounding controls do not explain the
  error. The lowered core truncates carried state/decay through FPU source
  formats. A model-local candidate preserves state in FP32 with SFPU decay
  and subtraction. Its first two versions failed and remain separate
  artifacts (`gdn_core_model_1025.json`, `gdn_core_model_reconfig_1025.json`).
- Independent source review identified an illegal mixed use of precise
  unpack CBs as FPU operands. The repair uses shared-allocation aliases:
  original state CBs retain Default mode for matrix multiplication, and
  precise-copy aliases load FP32 state into DST. This follows the existing
  matmul kernel's alias/read-pointer pattern. First-token and 32-token tests
  now pass (`gdn_core_model_alias_1.json`, `gdn_core_model_alias_32.json`).
  The integrated 1025-token GDN attention test now passes every output row
  (minimum PCC 0.99992985), final traced decode (0.99997807), exact repeat and
  runtime audit (`gdn_model_device_1025.json`). Advertised-context and full
  decoder regressions remain work.
- QSA batch 32 passes all whole-output and contribution gates, independent
  positions and deterministic trace replay (`qsa_batch32.json`). Profiling
  first exposed a whole-trace-vs-per-op mode mismatch, then insufficient
  default profiler record capacity. The next measurement uses standard
  per-op profiling with an explicitly increased program-support count.

No numerical threshold has been lowered. No failing result is accepted as a
known limitation. AutoFix remains active.

## Router-sensitive prefill investigation

After the recurrence repair, `linear_contract_fp32.json` still fails an extra
initial-prefill per-row contribution check (row 1: 0.99178004). The matched
HF full/chunked/token-at-a-time control `hf_chunk_router.json` preserves expert
sets and scores above 0.99998 contribution PCC, refuting chunking as an excuse.

`decoder_intermediates_norm.json` and `decoder_intermediates_conv.json` isolate
the upstream boundaries with exact HF substitutions. Source inspection finds
two GDN rounding differences: gated normalization must round normalization
and learned-weight multiplication separately to BF16, then apply FP32 sigmoid;
depthwise convolution must accumulate products before BF16 rounding. These
semantic corrections are now under full-decoder regression.

An additional exact-input counterfactual proves a residual-write discrepancy.
With exact HF block output and injection, native BF16 multiply/add produces
only 0.89373 / 0.89159 BF16 element equality across the two requests. Explicit
FP32 multiply followed by BF16 rounding, then FP32 addition followed by BF16
rounding gives exact equality and zero maximum error for both. This test-local
candidate is recorded in `decoder_intermediates_residual.json`; it also removes
the primary failing row's expert mismatch. The verified arithmetic correction
is integrated into `GatedResidual.write`. Shared QSA correctness and performance
must be refreshed after this change; no prior measurement is relabeled final.

`linear_contract_rounding.json` closes the complete layer-0 gate with the
combined semantic corrections: prefill minimum contribution PCC 0.99958813,
continuation windows above 0.99995 with failing dropped-history controls,
three changed-input traced decode contributions at least 0.99988490, exact
repeats/reset/isolation, live interleaved prefill/decode, and clean watcher.
No router policy or threshold changed. Full GDN context must be rerun after
the norm/convolution changes; its earlier successful recurrence result remains
explicitly distinguished. PLE precision and remaining stage gates are ongoing.

### Long PLE precision control (2026-11-12, active)

Independent exact-checkpoint PLE/GDN reference replay exposed a group20 precision floor beyond the original short holdouts. At49152 continuous rows, minimum PLE output PCC remained0.99689066, but GDN output reached0.99497098 and attention residual contribution0.99472475. Exact first1024-row fixture replay was bit-identical. The full continuous CPU sweep and a finer grouping with compressed scales within the same30.72GB lookup budget are being evaluated. These findings block acceptance; no threshold reduction or context cap is substituted for numerical repair.

### Current precision repairs and open regressions

Group20 failed the independent long context and remains rejected. Fixed group10 plus8-bit LUT scales fits the same30.72GB table budget, passes full262144 CPU state-path holdout and the complete resident-table short PLE decoder (`ple_group10_contract.json`); device long context and largest resident prefill remain work.

The final GDN recurrence/norm/convolution state path passes262144 and262143+traced final decode (`gdn_final_device_262144.json`). Complete layer0 batch32 later exposes discrete routing sensitivity. HF BF16 router rounding and a device implementation of the installed ATen tie policy fix independently proved semantic differences. Exact-input GR nonlinear/mean controls justify FP32 intermediates at HF rounding boundaries. The remaining three batch failures are not accepted: `router_projection_intermediates.json` proves an additional native projection error, and `router_upstream_intermediates.json` narrows upstream drift to GR projections even on exact HF post-attention input. Test-local precise GR controls hold precise router arithmetic fixed before any further runtime integration.

The original QSA262144 run fails row222275 at PCC0.1833. Exact-prefix substitutions and CPU selected-key counterfactuals isolate compressed indexer projection drift. SFPU projection and precise128-wideRoPE have separate exact-input support. The reconstructed-prefix32-row window passes with precise projection; native score arithmetic on exact HF operands retains the critical self key and every selected-key output passes. Native scoring remains unchanged. Final precise-projection/precise-RoPE continuation cuts3/4/5/31/32/33 all pass (`qsa_continuation_boundaries_precise.json`, min resumed-rowPCC0.9999787211), including negative controls, traced changed inputs and clean watcher. Full262144/262143 tests remain open.

All new final commands retain source/argv/environment hashes in `provenance/`. Numerical gates are unchanged. AutoFix remains active; no failed stage gate is carried as accepted.

### Follow-up causal isolation (2026-11-12 19:41 UTC)

The precise projection/RoPE full262144 QSA run still fails78 rows. Four exact-input windows now reproduce the three worst failures using native scores and literal HF attention. SFPU score products/reduction restore all128 selected sets and >0.9999999 at the three failed rows (`qsa_score_windows_causal_summary.json`). Integrated `QSA.indexer_scores` into both prefill/decode with bounded8192-key scratch. The measured65536-key alternative gave no meaningful speed benefit for8x scratch. Runtime trace and full-context checks remain required.

Accurate GR down/up and router projections plus HF-exact SiLU/sigmoid boundaries fix batch requests16/23. Request29 still fails; `gdn_recurrence_isolation_v3.json` shows replacing recurrence alone on actual operands fixes it while preserving the passing16/23 cases. Literal HF normalization on identical operands does not help. The recurrence precision repair is active; no CPU diagnostic injection is in runtime. Existing GDN long-context evidence predates the nonlinear change and must be refreshed.

### Recurrence repair and wider decode regression

Exact FP32 SFPU recurrence now passes first-token state bit-for-bit and3-token/chunked state within4.657e-10, and fixes focused full-decoder requests16/23/29. It is the runtime default; the old FPU-dot backend is only an explicit diagnostic selection. Full32-request validation (`linear_sfpu_batch32.json`) then passes every prefill row but exposes request3 decode contribution0.9835732579. Repeat outputs remain exact and runtime audit is clean. Exact-prefix-state and activation-boundary controls are required; this failed decode gate is not carried or waived.

### Precise GDN projection trace workaround

linear_gdn_hybrid_batch32.json completes capture and two repeated complete batch32 decode replays with runtime audit and watcher enabled. Every decode row passes and outputs are bit-exact between restored-state repeats. Prefill request16 retains exactly the earlier failing PCC; other rows pass. Thus the hybrid block policy fixes capture packaging without concealing the numerical gate. Integrated the tested256-output BF16 parent/256 single-token/32 multi-token policy in tt/precise_gdn_projection.py. Same weight payload and K reduction tree; no shared trace-code edit. This supports a graph-size-sensitive workaround, not a measured proof of the old4GiB byte count. request16_precise_gdn_bisection probes the remaining accuracy failure.

### PLE physical-capacity lifetime repair

Fresh probes establish5664pass/5665fail, but complete public5664fails the83,886,080B MoE workspace: per-bank free15,499,328B, largest10,477,824B versus10,485,760B required. This proxy boundary is not accepted. Source inspection identifies gate/up (20MiB each) dead afteractivation and activated (20MiB) dead afterdown projection, all retained through the later80MiB weighted-output allocation. Added only Python reference releases at exact last use in SparseMoe.__call__, with no forced deallocation/math/layout/precision changes. The same5664 complete public test is rerunning; before/after evidence is required before accepting repair. Capacity lifecycle reproduction is also being extended to include prepared decode resources and a repeated prefill allocation.

### Chunk-I/O preparation repair (2026-11-12 22:32 UTC, active)

Fresh isolated diagnosis: `chunk_input_debug/AUTODEBUG.md`, launched with the installed AutoDebug script using a complete Codex0.163 installation. Root verified native slice hash inputs, descriptor-only generic hashing, interleaved accessor metadata, and lazy per-workload kernel-binary DRAM allocation against source. The native-only from_torch diagnostic independently measured338 programs ×16,384B =5,537,792B. Exact retained-byte slopes in the original capacity sweep follow the count of new extraction/writer programs; source and allocation ledgers refute per-N ttnn.full growth as the primary explanation.

The violated contract is bounded preparation for arbitrary public logical lengths while decode traces remain live. The prioritized minimal repair replaces only long-input hidden/token extraction with model-local fixed-descriptor copies, moves writer count to runtime, and executes all three I/O descriptors at setup before large I/O/capture. Arithmetic/dtypes remain unchanged. Tests are being strengthened to first capture an unseen plan after only fixed setup, interleave changed eager plans with the live trace, verify all valid counts/offsets/padding and stable retained memory, and exercise the complete public path plus real-HF checks. No maximum context or stage acceptance is inferred until those gates pass.

## Whole-trace profiler follow-up, 2026-11-12

Fresh isolated AutoDebug completed successfully (582.74s): `trace_profiler_debug/AUTODEBUG.md`, recorded by `provenance/trace_profiler_autodebug.json`. Numerical prefill replay is exact, but v1 timing is rejected: empty device CSV plus host Tracy32K dynamic-source overflow. Source inspection confirms trace-only builds use PROFILE_KERNEL=5, replay GO arms the producer, the host reads DRAM_AND_L1, and CPP_POST_PROCESS/DISABLE_PUSH_TO_TRACY do not suppress CSV serialization. Parent verified the cited producer, reset and consumer code. No causal environment repair is yet proved. Next discriminating experiment is a separate one-core device-only trace canary with read-only pre-drain control/marker snapshots, matching loaded-library/HAL provenance and two replay ordinals. This avoids full-layer host-Tracy source overflow while locating the missing producer/consumer boundary. No firmware changes, device resets, core source edits or acceptance waiver; AutoFix remains active.

### Device-only full-layer contrast succeeded (2026-11-12)

The isolated canary and then `layer1_prefill_device_trace_v2` establish that native trace producer/drain/export works in this environment. Full real layer1 prefill has exact eager/replay output and four-state equality, with native endpoint coverage for all110 participating cores and two replay ordinals. Its closed report measures one warmed whole32-row trace at459,938 microseconds, including internal gaps; no per-op timings are claimed. Capture/report logs, native CSV and exact provenance are losslessly archived.

The failed host-Tracy capture remains separately rejected: dynamic-source32K overflow is observed, while the specific cause of its empty device CSV remains unresolved. This contrast does not prove a firmware/environment fix or repair that earlier path. Device-only profiling is a demonstrated alternative; remaining decoder modes, capacity/HF acceptance and final review are still work. No runtime arithmetic, firmware, core source or acceptance threshold changed in this archival step.

### All six device-only profile slots have measured evidence (2026-11-12)

The full-layer native path now succeeds for prefill32/decode1 of kinds0,1,3, with all110-core/two-ordinal endpoint checks, exact eager/replay output/state and runtime audit. Whole-trace milliseconds are0:424.197505/47.196239,1:459.937690/48.554689,3:919.896021/194.279724 (prefill/decode); experts are BF16 for0/3 and BFP8 for1. Human-readable tt-perf-report tables, native CSV and exact capture/report provenance exist under the six `tracy/layer{0,1,3}_{prefill,decode}_device_trace_v2` directories. Internal gaps are included and per-op timings are unavailable.

This completes measured evidence for those profile slots on their recorded runtime, not final stage acceptance. A pending capacity repair may require source-closure review and refreshed measurements. The alternative device-only path does not establish the cause of the rejected host-Tracy empty device CSV; observed32K source overflow remains separate. No blanket firmware diagnosis, environment repair or threshold waiver is inferred.

## Sparse down workspace contrast (2026-11-12, active)

Fresh complete public3707 fails at its final27-token chunk: an80MiB down-output requires10,485,760 contiguous bytes per bank, with20,606,080 total free but only8,779,392 contiguous. This proves fragmentation, not total-DRAM exhaustion; no capacity reduction is accepted. Test-only early reservation passed96 ordinary-versus-reserved exact eager cases and15 changed-input trace cases across all32 logical counts with856 clean watcher lines (`sparse_down_workspace_control_v1.json`). Full-model reservation v6 then correctly rejected a guard mismatch before testing capacity. Metadata-only v7 confirms actual activation shape[1,1,1,512,32,640], versus the isolated rank4 control; a rank-preserving exact-alias control is required. Runtime remains unchanged, and the failed experiments are preserved. Separately, complete real-weight3712 prefill and three changed-input traced decodes pass strict whole/contribution PCC, exact repeats and audit (`ple_public_hf_3712_v5.json`). This does not eliminate the nonaligned capacity failure.

### Workspace experiment shape correction remains test-only (2026-11-12)

The isolated rank4 workspace v1 control passed96 eager comparisons/15 replays, but integrated preparation exposed rank6 activations. Reserved-public v6 and metadata v7 stopped at guards before testing capacity. The v2 control incorrectly inferred a rank6 output from that input: the existing both-inputs-sparse native contract yields rank4 down output. Its failure after96 rank4 comparisons is preserved, not called an allocator limit. The next test-only control must retain native output shape and zeroing semantics; no runtime integration or capacity gain is yet proved.

### Persistent pool experiment rejected (2026-11-13)

V3 isolated control now proves192 eager and30 trace exact comparisons across both input ranks/native4D output. However full-public reserved3707v9 fails in its first32-row chunk: the persistent80MiB pool leaves only16,598,016 total free bytes for a20,971,520-byte precise-residual reduction slice. This is an earlier total-free-memory failure, unlike the ordinary tail's fragmentation failure. The reservation hypothesis is therefore refuted as a capacity repair for this case; no runtime integration or capacity reduction is accepted. Preserve both the passing isolated control and failed integrated experiment while checking ordinary per-residue boundaries.

### Ordinary2555 tail failure confirms range remains open (2026-11-13)

The terminal2560 and per-residue searches identify a count27 hole at2555, and fresh complete-public2555 reproduces it at actualtail79. Total per-bank free18,518,144 exceeds the10,485,760-byte request, but largest7,864,320 does not: ordinary fragmentation remains, distinct from the rejected reserved-pool total-memory failure. Candidate fixed-residue boundaries are not a monotonicity proof or accepted maximum. Aligned3712 real-HF success is retained separately. A query-row scheduling experiment is active; no runtime change or memory repair is yet claimed.

### Query-row scheduling exactness proved in isolation (2026-11-13)

The closed real-GR control passes all 384 eager comparisons and 120 traced executions over 20 plans, including live eager interleaving. Observed down-product padded geometry halves from 40MiB at Q32 to 20MiB at Q16; the corresponding first-half slice reduction from 20MiB to 10MiB remains source-derived. Watcher scans are clean over 5,093 lines. This supports testing row scheduling while retaining the unchanged per-row K reduction. No runtime integration, whole-decoder capacity gain or accepted bound is claimed; the fresh split-only full-public3707 run is still separate work.

### Split-only schedule rescues the reproduced3707 case (2026-11-13)

After isolated exactness, the test-only split schedule passes complete public3707 twice (116 chunks per pass), including tail27 and the decode trace lifecycle, in126.81s. No persistent down-output pool is used. This supplies a direct counterexample to treating the ordinary3707 fragmentation as an unavoidable capacity limit. The minimal split-only direction is supported; HF parity, full residue/range validation and any runtime integration remain separate gates. No accepted context reduction follows from this diagnostic.

### Split-only range still has a residue26 fragmentation hole (2026-11-13)

The closed320-probe/32-residue test-only search improves the previously failing27/29/30 residues, but residue26 fails3482 after candidate3450. Its largest per-bank block10,482,176 is3,584 bytes below the10,485,760-byte request despite22,173,568 free. Identical post-probe baselines distinguish this from retained growth. Thus split-only is not yet a continuous-range proof. The active combined experiment is separate; no runtime integration, final bound or capability reduction is accepted.

### Combined split plus persistent pool also fails (2026-11-13)

The combined3707 experiment fails before MLP on the first Q32 chunk: GDN beta reshape needs786,432 contiguous bytes per bank, with1,315,968 free but only565,248 contiguous. Unlike the earlier reservation-only residual-slice failure, this is fragmentation with sufficient total free memory. Persistent reservation overlaps another upstream peak and is rejected. The next test-only contrast places reservation at SparseMoe entry after upstream consumers; it remains unproved and runtime is unchanged.


### Test-only per-call reservation avoids the demonstrated persistent overlap

`ple_capacity_split_percall_public_3707_v1` completed both full public constant-input 3,707-row passes (116 chunks each) and decode lifecycle in125.83s. Split16 plus a SparseMoe-entry per-call down pool passes this candidate, unlike the rejected persistent pool that overlapped GDN. This proves only the tested capacity/lifecycle schedule; no runtime integration, HF accuracy or continuous maximum is accepted. Result/log/provenance are archived. Per-call-alone isolation is pending separately.


### Per-call-only isolation does not rescue tail27

`ple_capacity_percall_public_3707_v1` completed its wrapper in79.68s (exit0) but failed the actual3707 tail27 activation allocation. Required2,621,440 bytes per bank exceed the largest block2,119,168 despite4,877,440 free. This refutes per-call reservation alone as a sufficient repair for this candidate; it does not establish a physical total-memory bound. The passing3707 combined split16/per-call schedule remains test-only, with its all-residue search pending separately. Closed failure evidence is archived; runtime is unchanged.


### Combined reservation search refutes a larger uniform range

The completed split16-plus-per-call search (`ple_capacity_residues_split_percall_v1`,683.51s,320 probes/32 residues) reproduces split-only candidate boundaries, including residue26's3450pass/3482fail. Reservation changes the failing allocation to activation multiply20MiB, with the same3,584-byte contiguous shortfall per bank; total free6,444,928 exceeds required2,621,440. No uniform3712 claim or accepted physical bound follows. Closed evidence is archived. A separate in-place private-gate activation control is pending; no runtime edit or numerical equivalence is asserted for it yet.


### Private expert activation reuse passes exact isolated control

`expert_activation_control_v1` passes96 real SparseMoe eager comparisons at all Q1..32 and36 trace executions across six changed-input/interleaved plans. All240 observed in-place SiLU/multiply operations alias the private gate buffer exactly with zero allocation delta; watcher5,209 lines is clean under both recorded scans. This establishes the tested scheduling equivalence only. No runtime repair or capacity bound is accepted; activation-only full-public3707 is being tested separately. Closed evidence is archived.


### Activation-only policy shifts the tail failure to reduction cloning

`ple_capacity_activation_public_3707_v1` fails actual tail27 after successful activation and sparse-down allocation. Native `sum(down)` requests an80MiB clone: free12,610,688/largest6,157,952/request10,485,760 bytes per bank. The failure is contiguous-allocation fragmentation, not total-free exhaustion. Closed evidence is archived; no runtime integration or accepted bound follows. A proposed private-padding/aligned-view reduction requires its own numerical control before any claim.


### Per-call reservation does not remedy the reduction clone

Adding per-call reservation to the exact activation-reuse policy (`ple_capacity_activation_percall_public_3707_v1`) still fails actual public3707 tail27 on native `sum(down)`80MiB cloning: free12,610,688/largest8,779,392/request10,485,760 bytes per bank. The closed failure rejects this combination as sufficient for the clone allocation; runtime remains unchanged. Evidence is archived. Private-padding/aligned-view reduction equivalence remains to be tested.


### Private padding removes the clone in an exact isolated control

`private_reduction_control_v1` passes96 eager checks at all Q1..32 and36 trace executions on real expert-down shapes, including poisoned NaN/Inf/finite padding. Valid inputs and native reduction outputs remain bit-exact. For every nonaligned count, NORMAL allocator graphs measure additional peak84,213,760→163,840 bytes; all64 raw graphs are archived. The candidate retains the native sum call/configuration and uses alias-only physical-height/logical-height views around a private in-place padding clear. Audit/allocation tracking pass and watcher1,137 lines is clean under both scans. This is test-only proof; complete decoder/capacity/HF gates remain separate, runtime unchanged.


### Minimal private-buffer policy integrated for PLE, final validation pending

Test-only activation plus private reduction passes both complete public3707 constant-input prefills and decode lifecycle in124.57s, with no GR split or reservation. The corresponding PLE-only runtime policy is integrated after exact isolated controls. Full integrated HF/capacity/tracing/batch/performance gates remain required. Earlier six-mode source closure and the old batch5 clone-based bound are historical, not evidence for the repaired allocation path. No supported context reduction is accepted.


### Integrated real-HF lifecycle completes with one literal contribution failure

`ple_public_hf_3707_v6` exits1: whole-output PCC passes globally and at every row (minimum0.9950039280), but contribution row55 is0.9947104757, the sole failing row. All3 decodes, one capture/six replays, exact repeats/state and runtime audits pass. This is not accepted accuracy or capacity. Failure evidence is archived; independent actual-packed CPU diagnosis remains pending without a claimed cause or threshold change.


### Repaired default runtime yields consistent proxy candidates, not an accepted bound

The320-probe all32-residue search completes in580.67s: every r reports4736+r pass/4768+r fail, with identical restored allocator baselines. At4769 the first MLP GR20MiB slice fails contiguous allocation despite sufficient aggregate free bytes. Evidence is archived. Fresh public/HF boundary and terminal checks remain required; neither search consistency nor the planned CPU closure helper proves continuous support or a hard physical limit.


### Fresh public boundary evidence, with accuracy still separate

The repaired default runtime completes both complete4768-row constant-input prefills plus traced decode lifecycle; fresh4769 fails its first public chunk in the same MLP GR contiguous allocation. Both closed records are archived. The CPU3707 packed diagnostic reproduces a contribution anomaly but also introduces a separate whole-row failure, so no matched-device classification is inferred. CPU4768/4767 controls pass but cannot substitute for device HF evidence. Accepted capacity and hard-physical-limit claims remain unset.


### Integrated4768 literal accuracy passes; neighboring/terminal proof pending

`ple_public_hf_4768_private_v1` passes every literal whole/contribution prefill row (minima0.9954640871/0.9951774922), three traced decodes through4771, audits and exact repeats. The closed evidence and external-output identity are archived. This does not retroactively clear the3707 row55 failure or complete the prepared-range/capability contract; neighboring HF4767, terminal32 and final review remain separate.


### Nonaligned neighbor4767 closes strict HF correctness

`ple_public_hf_4767_private_v1` passes every literal whole/contribution prefill row (minima0.9958494904/0.9956877869), three traced decodes through4770, audits and exact repeats. Evidence/external-output identity are archived. Together with4768 this closes the two requested HF boundary lengths; the current terminal32 sweep and final review remain pending, with no accepted range inferred here.


### Terminal32 passes with precise proxy scope

All4737..4768 terminal public-chunk lifecycles pass in `ple_capacity_private_terminal_4768_v1` (130.56s), including full resident I/O and decode/repeat. These are chunk proxies, not32 complete public-prefill runs. Combined with complete public4768, adjacent failure and strict HF4767/4768, the fixed-resource source argument supports the recorded prepared-profile range. Formal dependency/provenance closure remains pending; no universal maximum, accepted capability reduction or stage verdict is inferred. Closed evidence is archived.


### Current batch and evidence closure remain distinct from acceptance

Normal batch5 passes explicit controls while preserving its literal contribution finding; Watcher batch5 fails only end-trace overlap. Normal batch6 fails larger trace overlap, with aggregate-resource analysis pending. All closed records are archived. CPU capacity closure validates source/log identities of the six boundary records but explicitly declines continuous-range/hard-limit/reduction acceptance. Matched3707 TT/packed rows pass, providing attribution while preserving both the original literal row55 failure and the control's separate literal row33 failure. No final stage verdict follows.


### Final QSA all-output pass remains open after one failing row

The full262144 QSA prefill completes but fails row105584 PCC0.9944236875, despite global0.9999747948 and finite outputs. The fail-closed all-row assertion prevents boundary/tail/decode execution and prefix export. Evidence/output identity are archived; this cannot be called a successful full-context gate. Fresh CPU and AutoDebug diagnosis is in progress, without a proven cause or relaxed acceptance.

### QSA cutoff diagnosis and launcher recovery (2026-11-13)

The fresh source-only AutoDebug launcher first failed its sandbox preflight because this host disallows unprivileged namespaces (`qsa_full_context_autodebug_v1`). The explicitly authorized retry (`v2`) selected the old CLI, whose code-mode helper was absent. After verifying that PID 2603810 was this run's CPU-only Codex child, it was terminated with SIGTERM; no device process was terminated or reset. Retry `v3` prepends the installed matching CLI/helper directory `/home/ttuser/qwen38-flash/codex-venv/lib/python3.12/site-packages/codex_cli_bin/bin` and runs with `AUTODEBUG_SKIP_CHILD_SANDBOX=1`. All three logs and provenance records are retained. The fresh report remains pending.

CPU-only controls (`qsa_failed_row_selection_cpu.json`, `qsa_two_row_precision_cpu.json`, `qsa_two_row_midpoint_cpu.json`) localize a plausible cause without changing runtime or acceptance. On row 105584, a single near-cutoff block replacement raises saved-TT agreement from 0.994424 to 0.999973. Uniform FP64-rounded and balanced-FP32 indexer projections both predict that replacement using otherwise unchanged HF query normalization, RoPE and keys. The critical exact projection is -0.1333008091605734, just below the BF16 midpoint -0.13330078125; literal CPU BF16 GEMM rounds to -0.1328125, while the more exact controls round to -0.1337890625. Row 58531 provides an independent analogous control. Score precision and query-normalization reduction controls on identical operands retain the original selected sets. These are source-bound CPU observations, not proof of actual device-selected IDs or a passing full-context gate.

A uniform all-row FP64 indexer reference control is running separately. It must verify the original full selection hash before reusing any unchanged attention output. A test-only exact-runtime prefix-state recovery harness is being prepared to exercise the boundary checks omitted by the failed run. It must preserve the failed all-output result and prove state/output equivalence; it cannot relabel that run as passed. Chip 0 remains exclusively owned by the GDN full-context test during this CPU work.


### QSA actual-device cause and boundary recovery (2026-11-13)

The fresh AutoDebug v3 report completed successfully. Direct captures now close the causal gap: `qsa_failed_projection_device_v1` shows the production balanced-FP32 projection agrees with its CPU arithmetic model at all four named coordinates, while literal oneDNN BF16 projection differs at sensitive midpoints. `qsa_window_105568_current_v1` confirms the device selects block25012 instead of23259 at row105584; top-k agrees with CPU on identical scores, all2049 IDs are valid/unique/causal, and cache gathering is exact. Independent HF attention using those actual IDs matches saved full-run TT output at PCC0.9999731261 (`qsa_actual_capture_diagnosis_v1`). The original all-row PCC failure remains open.

Uniform FP64 projection is rejected as a repair: `qsa_fp64_indexer_control_full_v2` fixes105584 but creates row16748 PCC0.993546327 against TT. Literal HF itself differs from that uniformly higher-precision reference below0.995 on those rows. Projection order therefore matters; higher precision alone is insufficient. Source-derived fixed K1024 blocks with high-then-low pair accumulation reproduce literal BF16 on307200 tested projections (`qsa_projection_orders_cpu_v3`); a full-history CPU check and standalone device candidate are pending. No per-row correction, reference rewrite or threshold relaxation has been applied.

`qsa_boundary_recovery_v1` passes an explicitly reduced state-reconstruction path: short controls prove all five mutable states bit-exact to ordinary forward, final32 outputs are bit-exact to the saved full TT run, the31-row tail has minimum HF PCC0.9999137521, and one traced attention capture/four A/B/A/B replays pass with exact repeated states. The exported recovery state has its own version2 provenance and retains `strict_full_prefill_passed:false`. `qsa_full_decoder_recovery_v1` then passes full public layer3 decode at position262143 on two changed inputs, one capture/four replays; minimum whole/contribution PCC0.9999422431/0.9999366403, clean audit and allocation tracking. No HF state is injected. These boundary results do not clear the failed full-prefill accuracy gate.
