# Optimized multichip decoder work log

Stage 5, started 2026-11-13. Baseline checkout: 8502f42b, clean worktree.
Target Qwen/Qwen3.6-27B revision 8f33b98937c6031bb49e713dee3e8e185aafd26f.
Four idle Blackhole P300c chips, 1x4 FABRIC_1D, Linear axis 1, two links.
Enabled plugin inventory: CODEX_HOME/config.toml enables tt-model-bringup
0.1.23 and tt-autodebug 0.1.8. Shell environment uses lane-env.sh and the
installed environment.py. Packages remain read-only. No full model or vLLM.

Accepted inherited decoder hash: 082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44.
PCC floor remains 0.995 for output, contribution, per-user and cache/state.
Context 262144 and arbitrary logical sequence lengths remain supported.
Dense model: all MoE conditions are inapplicable.

The original multichip stage used checklist stop rule 1. This stage explicitly
uses the requested >=70% roofline plus last two kept <3%, or eight hours of
device work, starting with the first candidate measurement. Inherited evidence
is useful prior knowledge, not a substitute for final-default measurements.

## Startup

Read model-bringup startup, optimize method/perf-report, multichip, device usage
and repository LLM multi-device/optimization guidance. fuser found no chip
owners; bounded `timeout 60 tt-smi -ls --local` returned all four P300c chips.
No reset performed. Device jobs will remain serialized and bounded.

## Operation topology audit before tuning

See topology_audit.md. Initial priority is physical flatten/unflatten movement
in batched prefill, then compact decode norm/collective and projection paths.
Historical same-source profiler tables are in ../multichip_decoder/perf/
delivery_profile_retry_*. New baseline timing will use the existing numerical
runner and matching accepted reference tensors; only TP4 is measured here.

## Initial measurements and investigation

`bash doc/optimized_multichip_decoder/run_baseline.sh` (full path in scripts)
completed six TP4 before rows at B1/S512, B32/S33 and B32/S513 for both kinds.
All pass unchanged PCC. Before decoded ms: GDN .574264/.732729/.733056;
full attention .489789/.701178/.748529. Warmed prefill ms: GDN
2.138881/11.155843/59.001102; full 1.618333/10.998635/44.630081.
See before_*.json and .command.json for source hashes and exact commands.

AutoDebug child sandbox preflight failed because this container disallows
unprivileged namespaces. No model started. Retried with explicit
AUTODEBUG_SKIP_CHILD_SANDBOX=1 under the existing unrestricted execution and
skill-requested subagent authorization. Child remains inspection-only.
Reports: autodebug_launch.log, autodebug_retry.log, debug_flat_prefill/.

First candidate measurement began 2026-11-13T14:52:41.525539Z. Serialized
candidate durations are recorded in candidate_commands.jsonl. No stop rule
has fired. Family1 measures existing policy combinations before source edits.

## First family results

Family1 existing policies: L1 MLP intermediates increase GDN decode from
.5740 to .5855ms (B1), .7327 to .7443ms (B32). BFP8 CCL increases B1
decode to .5852ms and fails synthetic B32 per-user decode PCC .991332;
aggregate/state PCC passes. This is not a real-activation precision rejection.
precision1.json schedules matched recorded-activation controls, including
separate attention/MLP activation precision and topology combinations.
The failed run completed all device work and closed cleanly; failure was the
host numerical assertion, with no transport/runtime error. No reset was used.
Remaining family1 entries are unmeasured, not rejected.

Family2 full attention: remove the [1,B,H]->[B,1,H]->[1,B,H] round trip,
pack gate with QKV at setup, and compare gather-first local width-sharded norm
against distributed norm. B32/S33 control .701476ms; direct .638161ms;
packed .626116ms; gather-first without sharded norm .653066ms; coherent
packed/direct/gather plus sharded norm .523585ms. All unchanged PCC gates pass.
B1/S512 selected family .473398ms versus .489818ms; B32/S513 .572283ms.
Weights remain BFP8. These are policy candidates, not final-default numbers.
Full QKV/gate packed width3584 replaces widths2048+1536 with unchanged head
ownership and setup-only weight packing; old projection copies are removed.

AutoDebug completed its prefill report; findings were checked against current
source. AutoFix begins with flat_finish and flat_prefill_correction as separate
increments. Initial B32/S33 GDN: control11.3034ms, flat finish9.8088ms,
flat finish plus corrected projection9.3817ms. Minimum PCC .996754 in each;
decode unchanged (~.733ms). Pending broader shape/profile/watch validation.

Persistent CCL first attempt failed before launch: TTNN Shape does not accept Python slices. Converted Shape to tuple for output sizing and retried; original shape_error log retained. Device closure completed normally. This is not a rejection.

Persistent CCL adapted shape API runs eager correctly but first trace replay fails output PCC .63743 and state .79933/.96664. This is a trace correctness defect, not a precision rejection. Snapshot and AutoDebug investigation are in debug_persistent/. No default selects this policy. Other topology candidates continue without persistent buffers while diagnosis runs.

### Continuation: flat-prefill integration
Prior candidate suite finished successfully; old process handles are gone and no device owners remain. Flat-prefill CPU candidate integrated behind an opt-in policy. Persistent AutoDebug ended without a report; diagnosis remains open. Installed skills now version 0.1.24; explicit user stop rule remains authoritative over changed skill defaults.

### Flat prefill and recurrence smoke
`flat2.json` six-case TP4 suite passed at source c170388df1cdc4b03330a7f85a8502c8cdb3dafcf04bfc8f89679836b1955408. GDN B32/S33 prefill 9.4169 ms (family3 legacy candidate) → 8.3872 ms; minimum PCC unchanged .9967542. Full attention B32/S33 8.3645 ms, essentially unchanged from flat_finish. Full results in flat2_summary.csv. B1 full attention policy still regresses prefill relative to original single-batch norm route and requires phase policy adjustment. These are opt-in candidates, not final defaults.

Recurrence smoke on TP4 B32 BF16 passed exact output/state checks for all six independent source changes, eager and changed-input traces. Matched traced medians (us): skip_dead_padding 141.89→140.52; skip_constants 141.94→137.03; grouped_reads 141.98→141.77; gate_cache 141.81→141.77; exp_row 142.10→141.69; exp_scalar 141.97→138.85. Cumulative variants now undergoing diverse-batch BF16/FP32 checks. No numerical-policy changes selected. Persistent CCL AutoDebug relaunched because prior child terminated without producing its report.

Recurrence cumulative probe: all 48 cases passed bitwise eager and changed-input replay parity (B1/2/3/10/17/31/32, BF16/FP32, extra B3 one-core user-wrap). Exact commands in recurrent_cumulative.command.json. Source transforms remain opt-in/probe-only; full decoder policy integration and zero/poison/watcher checks pending. Prepared readers1.json with per-role reader counts 1/2/3 and activation cores 4/8/16 (down uses17 instead of16 to divide K=136 tiles), not yet run.

### Persistent CCL repair accepted for further evaluation
All eight persistent_fix1.json cases pass (includes four newly executed family3_mmrs_persistent tags). Corrected native Linear staging allocation follows padded input shape with dimension0 doubled. This is a caller repair; native validation/docs gap is recorded in AutoDebug for follow-up. No native kernel changes were needed. Latency and PCC summarized in AUTOFIX.md. Broader ownership, stress and watcher validation remains required. Precision1 resumed after pausing cleanly between jobs for this repair.

### Remaining work (stage still active; stop rule has not fired)
Integrate and measure cumulative_exp_scalar recurrence; zero/poison controls; fix B1 prefill norm policy; compare coherent fused AG+MM family; finish real precision and per-role K/readers/activation-shard sweeps with retries for material API failures; combine kept policies; reprofile final default with tt-perf-report/advice/roofline; evaluate explicit user stop rule; context smoke and memory contract; awkward public lengths/list/resume; inter-layer stack and mutable-resource ownership; trace allocation positive control, stress and watcher; runtime fallback audit; independent stage review/fixes; local stage commit. README numbers are provisional candidates. No full-model/vLLM work authorized or begun.

Precision driver interruption: parent expanded wrapper provenance hashes while an invocation was waiting. The shell resumed at a shifted file offset and reported unexpected EOF after the device job closed successfully and wrote passing precision1_attn_hifi2_l3_b32_s33.json. This is a host wrapper mutation error, not a device/API failure. bash -n passes on the complete current wrapper. Do not edit any live wrapper again; resume skips completed passing JSON and runs remaining candidates. AutoDebug retry exited0 with report; its sizing diagnosis was verified and repaired above.

Precision1 completed all20 configurations: last full-attention BFP4/LoFi candidate fails a state PCC gate despite strong output PCC; preserved failure JSON/log, no default change. Higher fidelity/geometry retry remains to distinguish quantization from compute error. Integrated1 opt-in hooks now combine corrected persistent buffers, flat prefill, cumulative_exp_scalar raw-packed GDN recurrence, and attention decode_or_single norm policy to restore B1 prefill route. Exact combined checks pending.

### Continued integrated search
Integrated1 all six cases pass; latency table in integrated1_summary.csv. Geometry2 runs each dominant projection K-block divisor with this same BF8/BF16 candidate, preserving native compact residual and persistent buffers. Precision1 real activations: BFP8 CCL valid but slower; BFP4 MLP gives ~7–8% decode gain but is stage8 evidence only per optimize instruction; attention BF4 full-attention fails KV PCC .99378/.99465. Precision2 prepared with current integrated topology and higher fidelity adaptation, not run yet. Stack/resource tests now prepare native compact decode and isolate persistent scratch per shallow-copied resource owner; device validation pending final selection.

### Collective placement and geometry adaptation
Geometry2 GDN input K2/5/10/16/20 all pass real-activation PCC; K2 .685115 ms, K5 .685250, K10 .690693, K16 .691822, K20 .694290. K32 hits L1 collision (static end1521600 vs live buffer1457216); activation cores16 moves live start1514560 but still collides; readers3 and cores32 adaptations require1718208 bytes >1572864 physical L1. Placement family changed compact collective input/output together to DRAM (not only an immediate layout restore). Matched B32S33 DRAM vs L1: GDN .673525 vs .685905 ms; full .481988 vs .488953, unchanged minimum PCC. K32/c32/DRAM still exceeds static physical L1; c16/DRAM remains to try. New interim profile captures this coherent candidate. These allocation validation failures closed devices normally; no in-flight hang/reset occurred.

### Native MMRS NoC ownership repair
Kernel-only NoC0 experiment passes watcher for direct native M1056 attention and MLP, eager plus50 queued replays. MLP PCC .9999927946; native plan recorded in mmrs_noc0_mlp_m1056.json. Timings are watcher-contaminated and not performance evidence. Canonical host factory now routes both operands through reader NoC0; reader consumes both compile-time arguments and asserts ownership. Host build and fresh-cache revalidation follow. Existing failing watcher control remains in prior-stage debug_narrow_watcher.

### Coherent fused all-gather families
New fused_ag_candidate.py measures sharded residual/add, distributed norm, AG+MM, SwiGLU, down+RS through the retained sharded residual contract. B32 GDN separate sharded family .305686 ms vs best-current finish .264523; minimal fused .533441 vs .264881; native fused SwiGLU .536692 vs .264407. All pass eager contribution PCC and exact first/last replay. Conventional mcast1d attempt rejects Linear fusion in all_gather_async_default_program_factory.cpp:220; inspection confirms both conventional factories share this constraint. Adapted minimal and native-SwiGLU measurements are legal Linear evidence, so no blanket first-API rejection. Actual physical endpoint Ring connectivity is not established; no blind ring experiment. B1/full-kind/config adaptations remain. Details in fused_ag_candidate.md.

### Interim profile and accounting
interim_profile2 bothkind B32S33 TP4 profiles pass; CSV and advice tables in tracy/ and perf/. GDN decode matmuls274.583us, recurrence~125us, othergeneric~38us, binary64.139us, RS35.134us, AG31.684us. Prefill is now dominated by native MMRS18calls1379us, remaining reshape743us, matmul676us, recurrencechunk627us, conv604us, slice569us, nativeSwiGLU549us, AG443us. Full attention prefill MMRS1375us, reshape535us, slice499us, AG441us, nativeSwiGLU383us, RoPE274us64calls. TT-perf-report advice inputL1 and K/subblock tuning are scheduled. Profile helper corrected to --policy-file after first Tracy command quoting failure; the earlier artifact remains a host CLI failure, not a model/runtime result.

Accounting helper includes read-once matrices plus mandatory GDN state RW and visible attention KV traffic, validates matrix allocations, and excludes setup-only redundant prefill copies from decode bandwidth. Current B32 lower bounds .285792ms GDN/.195296ms full; instrumented profile roofline fractions .40044/.37874 are provisional, not final unprofiled claims. Explicit original stop rule remains unmet. Capacity audit preserves262144, ~20.57GB/device includes4GiB reserve for integrated1; final policy and32K–64K smoke pending.

MMRS2 all16 integrated full-layer cases pass unchanged output/contribution/per-user/state gates and changed-input traced replay. Both kinds B32/S33/S513; rows128/512/2048/16384 and max32768. Complete before/after table mmrs2_summary.csv. Increasing max without increasing128chunk regresses long prefill, confirming family interaction. Best whole/chunk2048 materially reduces prefill; no decode change. Current larger-row selections remain opt-in pending resource/watch and final default validation.

Fresh native MMRS probe now passes with explicit command env afterlane setup; mmrs_fresh_mlp_m1056.log reports0/198 cachehits, proving fullcold rebuild. RoPE distinct-user TP4 watcher probe passes ten B/S/head cases including2x3,3x33,17x31,32x33,3x513, NaNdeadpadding and changedinput traces. First full-layer RoPEB32S33 5.358574ms vs7.964114ms baselinepolicy; decode.482481 unchanged. Mainclass nowhasopt-in batched_prefill_rope hook; integrated3 pending. PackedGDNprimary+correction firstfull-layer pass.665329ms vs~.673ms, setupquantizedvaluesexact; candidateonly, extraresident38,993,920bytes/layer and furthergeometry/B1 checks pending.

Geometry3 gateupK32 exceedsphysicalL1:1,728,448B>1,572,864B; c16/r2same, c8/r3adapt1,718,208Bstilltoo large. Nonpipelineadaptprepared; smallerK2/5/10/16/20passed withK2fastestsofar. Readers2 outputc16/r1 initiallyviolates shardKtiles3 vsblock2 divisibility; adaptedalloutc16entries toblock3 andretainfailurelog ratherthanrejectgeometry. Initialinputr1 ~.948ms vsr2 .672ms/r3 .668ms, samePCC. Integrated3 allsixpass withnewRoPEhook+nativewholechunks; B1prefillregressionstillunderinvestigation beforedefaultselection.

Joint GDN prefill initial failure was TTNN Shape slicing in hostvalidation, corrected by tuple(shape). RetrypassesallunchangedB32S33 gates: prefill8.919ms/decode.664908ms, old128-rowMMRSpolicy. Botholdweightsremoved; candidatejointweightonly38993920bytes/rank. Largerrowcaps/cleaner2Dgrids withintegrated3 pending. Reader3 rowoutput exposed setup bug: paddeddecodeN wascopiedintoMMRS andscatteredatwrongboundaries; setupnowcropslogicalN beforeMMRS. Retainedreaderfailure andresumed with adaptation.

### Convolution and reader-family continuation
All remaining readers2 cases completed on TP4, including shape-adapted output and down projections. Per-role choices differ: GDN input c8/r3 0.667746 ms versus c8/r2 0.672230; GDN MLP gate/up and down retain c8/r2 (0.672779/0.672445). Full attention input c4/r2 0.481341, gate/up c4/r2 0.480875, down c8/r2 0.482130. Sub-percent differences need integrated repeats before selection. One-reader controls and extra reader padding are retained in the search table, not conflated with physical logical widths.

Local fused convolution passed all 21 watcher/trace-allocation controlled cases: batches1/2/3/17/31/32, zero QKV, zero history, poisoned dead tail padding, and B3 one-core wrapping. Exact output/history parity and changed-input traces passed. Full decoder B32/S33 allocation-tracking run passed all unchanged gates at 0.625784 ms decode /7.529235 ms prefill. Uninstrumented B1/S512 passed at 0.523660 ms decode /2.125275 ms prefill. Production class-method integration remains opt-in until cumulative tests.

Streamed recurrence state writes passed exact eager/state/changed-trace B32 checks. Matched seven-sample median 132.719 us baseline versus132.594 us candidate is only0.094%; no selected-path change justified. It adds four state-write barriers per head, so source simplification alone does not establish a speedup. No device hang/reset occurred.

Wrapper provenance now includes every model/test Python source plus both modified native MMRS source files. The optional MMRS L1 advice candidate now moves each chunk into L1 rather than staging the entire flat input, avoiding unnecessary whole-prefill capacity pressure. Neither change selects a new default.

### Owner-safe integrated4
Joint projection now uses class dispatch and explicit executing-owner arguments; removed prototype shallow decode view and bound methods. Full ordinary decoder runner integrated4 six-case suite passes unchanged gates. GDN B1/S512:2.046851ms prefill/0.512771ms decode; B32/S33:6.951146/0.614172; B32/S513:46.442908/0.614326. Full attention B1/S512:1.565881/0.454069; B32/S33:4.232219/0.482535; B32/S513:35.684964/0.532796. All are candidates under explicit policy, not final defaults. Fused-MMRS threshold now256 (strictly greater), recovering B1 prefill versus the stage-entry baseline. Candidate L1 joint-prefill input7.161284ms did not improve matched DRAM7.147625ms; retain DRAM.

Capacity helper now models actual joint active matrices, excludes retained inactive primary/correction from decode traffic, and crops MMRS-copy accounting to logical columns. Actual run allocation shapes agree. Joint active matrix bytes118,374,400/device; resident GDN matrices including MMRS/SwiGLU copies197,754,880/device. Context remains262144. Full independent resource/allocator validation remains pending.

Geometry3 down K34 initially collides with live L1 at1494080 versus static end1521344; larger-core, three-reader, and nonpipeline adaptations are prepared. This is not a first-error rejection. Earlier K2/8/17 cases passed and closed normally. No reset.

GDN down K34 adaptations: c17/r2 static1757632B >1572864B physical; c8/r3 static1959104B >1572864B physical; c8/r2 nonpipeline lowers static to1504960B but still overlaps live L1 allocation1494080B. The extra core/reader geometries also pad N, so their capacity cost is part of the family evidence. Smaller K2/8/17 remain valid measured alternatives. Validation errors closed normally; these were not hangs. Per-role larger-block trials continue for the full-attention kind.

Recurrence pair-column ownership smoke passes exact output/state/changed-input trace at B32 BF16. Seven-sample isolated medians132.7977us whole-head control→130.1166us pair candidate (~2.02%). It is not integrated or selected yet; diverse-batch/zero/poison/watcher and full-layer comparison must precede that decision.

### Continued stage: epilogue and precision
Packed SwiGLU corrected trace-lifetime probe passes eight exact eager/changed-trace cases with allocation tracking1 and skip-program-cache0. Uninstrumented matched epilogue medians (B32): GDN28.1705us→12.3186us; full24.2871us→12.3244us. Register variant beats local BF16 roundtrip (~12.69us). Integration is opt-in fused_packed_swiglu; ordinary six-case cumulative suite integrated5 underway. No default change.
Full-attention BF4 projection HiFi2 remains below unchanged KV-state PCC: prefill0.99378755/0.99466155, decode0.99377869/0.99465379, while output passes. The failure is preserved in precision3_attn4_hifi2_l3_b32_s33.json; higher-fidelity adaptation and remaining CCL crosses use precision3_unrun.json. Geometry3 full-attention K32 gate/up exceeded L1; four down-projection rows remain in geometry3_full_tail_unrun.json. Earlier geometry adaptations and physical limits remain recorded.

### Integrated epilogue and prefill-convolution evidence
Integrated5 ordinary six-case suite passes; same-path before/after table is integrated5_summary.csv. Decode saves roughly5–7us versus integrated4. Packed SwiGLU B1..32 both source graphs passes all64 watcher10/trace-allocation1 cases (skip program cache0) in packed_swiglu_watcher_all; exact eager and changed-input replay.
Native BF16 recurrence pack fails strict persisted-state bit equality; unchanged FP32 control passes. Separate unchanged full-layer PCC test native_pack_precision_l0_b32 passes (decode0.606886ms vs integrated4 0.614172ms; recurrent-state PCC~0.99943). This is precision evidence only, not an exact optimization or selected default. Longer stress remains necessary before selection.
Batched prefill convolution first probe failed in reference configuration: native operation rejects packer_l1_acc=True. Corrected reference config toFalse; fresh retry passes exact explicit arithmetic/rawhistory/splitresume/changedtraces at3x2,3x33,32x33 with tracking1. Uninstrumented native→candidate: B1S51260.287→302.312us; B3S3399.604→139.623us; B32S33880.540→709.351us; B32S5122526.479→7863.622us. A universal selection would regress long inputs. Complete regular and coherent flat-packed paths are being compared, and long-schedule adaptation is open.
Advice5 rebases actionable L1/minimal matmul trials onto integrated5; all10 complete gates pass. Results are kept in original JSON/commands and the search table. Decode minimal fused SwiGLU materially loses to the packed epilogue; no default selection from that family.

### Affine/gate and coherent prefill continuation
Affine6 allsixordinaryTP4case gates pass, with B32 GDN decode0.593706ms vs integrated5 0.607480ms and B1 0.494141 vs0.505730ms. Fullattentioncontrols unchanged~0.477ms. Focusedaffine watcher10/tracking1 suite passes28cases including6batches×4modes andB3onecore; initial CLI wrongmode extreme was corrected to extreme_gate in freshartifact.
Regular batched prefill convolution fullcases pass: B1S5122.413214ms, B32S336.523885ms, B32S51351.586926ms. Coherent flatpacked path B32S33 passes5.846927ms (integrated5control7.011243ms), with packedandZ remainingcompact; no fullpacked immediate restore. Long/native-tile adaptation remains open. Tiled-batch preservingfamily B1passes butB32S33 first2D launch requires1852352B L1>1572864B; geometryadaptation underway.
FullattentionBF4HiFi4 stillfailsKVstate0.993788/0.994668 despitehighoutputPCC, retainedprecision3_attn4_hifi4_l3_b32_s33.json. Thusfidelityalone doesnotrepairquantizationloss; remainingCCLprecisioncrosses andnewMLP/rowchunkfamiliesarequeued underfreshnames.

### Batched native-math and L1 adaptations
Tiled native-math convolution probe passes allfourmeasuredshapes; native→candidate ms: B1S5120.059440→0.057147;B3S330.099817→0.039452;B32S330.881152→0.127967;B32S5122.523896→0.790432. Thisadapts the earlierper-tokenlongregression with32-time-rowtilereuse, and keepsfull4160packedinputcompact. Watcher10/tracking1passes sixcases1x1/3x2/3x33/17x31/32x33/3x513, includingpoisonunusedpackedcolumns, exactrawhistory, splitresume andchangedtraces.
Tiled_conv7 fullsixcasegatespass; GDN B32prefill5.351771ms(S33)/44.019973ms(S513),decode~0.5937ms. B1prefill2.607164ms regressesversusaffine6 2.028475 despitefocusedkerneltie; retainlegacyB1 bydefaultunless explicitexperimentflag. Puregeneratedsourcecached once andbufferaddresseshoisted outsideper-coredescriptorloop; newtimingstillpending. Noarithmeticalchange.
Nativebatchedcachefillpasses focusedtracking1 anduninstrumentedB1/B32S33/S513, withcompletecacheoracle andchangedtables/inputs. B32S33 eager1052.871→56.438us,trace287.159→16.822us;B32S513 eager1394.363→100.141us. B1slightlylosesandretainsoldloop. Owner-safeintegration/ownershipproofpending.
MLP L1 firstfailure washelperDRAM-onlyvalidation. AdaptedexistingTensorAccessorreader/writer tointerleavedorwidth-shardedL1, preservingnativearithmetic; width-shardedoutput[32,544] directlymatchesdownprojectioninput. BothkindL1fullcasespass: GDN0.606970ms,full0.472276ms versusmatchingpackedfull0.477147ms. Oldflat_prefill guarddisabledcompactionwheneverl1_mlp; nowpermitsrows>32 whereinheritedL1decodeconditioncannotapply. Freshcumulativecomparisonrequired.
Tiled-batchpreservingfamily adaptedlegalgateupblocks passesbothkindsB32S33butloses: GDN9.642227ms/full6.631245ms vsintegrated5 7.011243/4.046740ms;GDNlong65.750552ms vs46.311885ms. ThismeasuresretainedpaddedB/Snorm/dense/residuallayout, notanimmediaterestore. Full six-caseevidencein tiled_batch2 manifest/results.

### Resume: verified terminal sweep and Q/K normalization probe
Previous prefill5 driver is terminal: cap4096/chunk128/full-attention failed physical L1 validation (1608640 B >1572864 B), devices closed normally; later jobs did not launch. Successful prior rows remain preserved. CPU investigation is adapting both the SDPA alignment-induced single-token schedule and the capacity failure, rather than rejecting this family on its first error. Compact Q/K norm first TP4 batches1/3/32 passes native/torch PCC and exact changed-input replay with allocation tracking1. Instrumented timings are not final performance evidence; uninstrumented comparison/adaptation remains. Projection chunk queue prefill_chunks7 started from its source-verified manifest, serialized.

Roofline wording corrected in report generators: source audit supports Blackhole1.35GHz and512GB/s per chip for this P300c configuration; the earlier blanket P150-mismatch caveat was unsupported. Historical artifacts remain unchanged. Peaks remain models, not sustained measurements, and mandatory traffic accounting does not count redundant movement toward the stop threshold.

Projection queue first nine jobs pass (all eight2D role jobs and joint1D); source-verified receipts retain row/family/PCC/replay evidence. Joint1D is slower at128/1024/4096 rows (325/2519/9968us versus220/616/2392us in the same run). The inventory helper initially failed on a new dictionary-valued source map; canonicalized its deduplication key without discarding provenance. Retry candidate_search_resume1725.csv inventories719 declared/executed variants with zero parsing warnings; pending includes superseded manifests and is not an assertion that327 independent families remain. Hardware ledger snapshot device_time_resume1724.json records1.1467h conservative occupancy, below original8h device cap.

Conventional fused AGMM adaptation: independent CPU audit identified receiver coverage/direction assumptions beyond the Linear guard; conventional_agmm_linear_audit.md gives the concrete schedule proposal. Invoked installed AutoDebug launcher before AutoFix. First preflight failed because this host disallows unprivileged namespaces; no model started. Retried with explicit AUTODEBUG_SKIP_CHILD_SANDBOX=1 under this session's existing unrestricted execution authorization, keeping the child inspection-only. Original failure and retry logs retained.

Prefill_chunks7 completed all28 jobs and376 measured configurations on TP4; every recorded role/row case passes unchanged PCC and exact changed-input/first-last trace gates. CSV: prefill_chunks7.csv; full command/source receipts perjob. This is a kernel-family table, not final whole-layer speedup evidence. Follow-up serialized queue resume_probe_queue8.sh begins only after validating all28 passing receipts, then runs watcher probes and the aligned temporal adaptation.

Reviewed bulk-cache integration applied opt-in: CertifiedPagedState shares revocation across shallow/deep copies, exposes public page tables conservatively, uses ordered capture fallback, and permits fresh private setup tables. B2/S33 watcher10/tracking1 full-layer cache_owner_b2_v1 passes oracle and exact complete-cache/alias/remap/resume/mutable-trace checks. Parallel Q/K norm passes four watcher batches; L1-width packed SwiGLU passes64 watcher cases. Onecore tiled-convolution first command had an unsupported poison flag and failed before opening devices; corrected retry passes3x33/3x257 with watcher and allocation tracking. All four watcher sidecars exist. Suspected runner table lifetime issue was refuted by source ordering: replacement occurs before first trace capture, so no speculative workaround retained. integrated8 six-case candidate and three L1 comparisons are prepared, not measured or defaults.

Integrated8 allsixordinaryTP4cases pass, with actual full-attention bulk dispatch1initial/3timed and exact timed output/cache checks. B32fullattention prefill3.590ms(S33)/35.471ms(S513),decode0.47782/0.52700ms; GDN prefill5.583/43.224ms,decode0.59384/0.59402ms. GDN B1prefill1.958/decode0.49399ms. FullB1prefill1.876ms is slower than prior1.58–1.62ms despite bulk path being B>1-only; controlled repeat required before selection. Before/aftercandidate table integrated8_summary.csv; not finaldefaults. Reconnected after runnerrestart to verified live PIDs, didnotduplicatejobs.

Prefill alignment8 all3adaptedcap512/2048/4096 casespass. K128short-capacityschedule and boundedheight solve both measuredpathologies: cap512pre-fill38.671ms vsold691.903ms; cap2048=37.887ms;4096nowpassesphysicalallocation. They remain slower than current512temporalschedule. Decoder K512andlongcapacity remaininherited. Classadapter updated for instance-method table helper afterownershipintegration, avoiding publictableexposure; nodefaultchange. RuntimeownershipfocusedCPUaudit reports111reachablefunctions/0violations and6leasechecks; extended negativecontrol coverage targetsnewhelpers.

Integrated8 L1 fullattention all3casespass: B1prefill1.687/decode0.447630ms; B32S33 3.505/0.472588;S51335.070/0.526652. FourB1ABBAcacheflag repeatsallpass: disabled1.713/1.622ms,enabled1.629/1.666ms,decodeall~0.449ms. No systematic B1cacheflagpenalty identified; finaldefaulttimingsstillrequired. ParallelQKnormuninstrumented totalgroupnative→candidateus: B1 17.640→16.006,B3 18.018→16.555,B17 23.258→21.372,B32 26.061→25.674. Main adds opt-in fused_compact_qk_norm after watcher/allocation tests; qknorm9 full3casescurrentlyrunning. No defaultselection.

### Resume: completed Q/K integration and native fused-CCL repair
qknorm9 allthree full-attention rows pass unchanged PCC/state/replay gates: B1S512 prefill1.699154ms/decode0.442682ms; B32S33 3.511754/0.471305ms; B32S513 35.339211/0.522083ms. These remain opt-in candidates. Native AutoDebug completed its inspection-only report in debug_conventional_agmm/AUTODEBUG.md: the observed Linear guard masks eight-slice Ring receiver assumptions and missing persistent scratch admission. It refutes alleged cross-launch accumulation of program-local semaphores. AutoFix is implementing an explicit bounded Linear schedule/ABI and admission protocol; no hardware claim yet. Three isolated CPU tasks own schedule repair, admission repair and a focused probe, with no overlapping native edits or device jobs.
Prepared precision_geometry5 is running serialized (30 real-activation geometry/precision rows, BF4 stage8-only). Prepared mlp9 rebases packed/split, reader/fidelity and activation precision comparisons onto current cumulative policies. ag_adapted9 uses recorded text inputs and actual decoder attention partials, preserves the compact residual through the complete finish, includes admission cost, and shares the selected packed SwiGLU epilogue across applicable families. Old synthetic-family artifacts remain intact.

Precision_geometry5 completes30 real-activation comparisons:28pass, two full-attention BF4 rows fail KV-state PCC at0.993788/0.994662 (prefill), as the prior fidelity-only trials did. A fresh missing-row manifest preserves the first failure and executes the remaining reader geometry. SummaryCSV precision_geometry5_summary.csv. BF8 compact workers/fidelity choices show only small gains or losses; BFP4MLP gains are stage8 evidence, not defaults. The final expected-failure row ran during a CPU native build; it supplies numerical failure evidence only, with no timing ranking.
Native LinearAGMM buildv1 fails a C++ span equality expression in new host placement validation; changed to std::ranges::equal and buildv2 is running. No runtime rejection is inferred. quality_candidates10 begins B32 full cache ownership and all-batch QK watcher checks on the existing installed runtime, followed by the separate512-step recurrence precision oracle/control/candidate. Native build is CPU-only and will not be installed while this device queue is active.

B32cacheownership exposed an L1/public-layout resume collision, not an aliasing correctness failure. Same22-case watcher/tracking rerun cache_owner_b32_l1resume_v3 passes after moving only non-decode compact attention output to DRAM before `[B,1,H]` expansion. Native/compact decode keeps its prior L1 contract. Runtimeaudit now explicitly requires parallel QK helper and passes113reachablefunctions/0violations plus negativecontrols; runtime sentinels still require finaldefault attachment. NativeAGMM buildv2/install completes. Relative freshcache path caused firmware build failure before operation execution; absolutecache retryagmm_linear_2d_small_smoke_v2 underway. No resets.


Native Linear AGMM focused followup10 passes all eight jobs: 1D/2D B1/B32 K4/K8 projection schedule patterns at N8704, plus 1D persistent-buffer rebinding, 50 queued changed calls and 50 three-call trace replays with deliberate rank skew. This complements 2D ownership/skew v6 and two fresh-cache smoke tests (including nine host rejection checks). All watcher10/allocation-tracking1/skip-program-cache0; devices closed normally. Summary agmm_linear_followup10_summary.json. Coherent ag_adapted9 real-text whole-finish comparison is running; native correctness does not establish speedup.


Coherent `ag_adapted9` finishes all32 real-text TP4 B1/B32 comparisons, both layer kinds: separate distributed norm, minimal fused AGMM K4/K8, fused SwiGLU, and repaired conventional 1D/2D multicast K4/K8. Every output/contribution/per-user/replay gate passes. All candidates are slower than the matched current finish (about0.256–0.261ms): distributed separate about0.298–0.303ms; conventional 1D about0.33ms, 2D about0.35–0.41ms; minimal about0.59–0.68ms. Exact table `ag_adapted9_summary.csv`. The measured boundary retains hidden-sharded residuals and includes admission, norm, projection, activation and both reduce-scatter/adds. This is a performance rejection after legal shape/layout/weight/grid/block adaptations and native AutoFix, not a first-API-error rejection. Independent review found scratch logical/coordinate admission gaps; guards added after this queue closed, canonical buildv3 underway.


Current-policy `mlp9` completes all10 rows with unchanged gates. Packed GDN decode0.59350ms versus split0.61789ms, one-reader split0.71137ms and split LoFi0.61696ms; full attention0.47137ms versus0.50978/0.57931/0.51010ms. BF8 activation variants0.59402/0.47127ms give no material gain. Packed remains preferred for decode. `mlp9_summary.csv` retains prefill too; GDN split prefill is slightly faster but does not offset decode cost or justify retaining duplicate projection families yet.

Q/K norm all B1..32 watcher10 passes. Native recurrence packing passes the separately labeled512-step real-text whole-layer precision test: final recurrent-state PCC0.99715560 versus explicit-RNE control0.99717005; output0.99953617 versus0.99951686. Both retain exact repeated-run/changed-input trace gates. The earlier strict bit-equality failure is unchanged; native packing is not called an exact transformation. Current-policy timing and broader ownership comparisons are prepared in `recurrence12.json`.

Independent `recurrent_epilogue_audit.md` identifies feasible cast/norm/affine/gate launch fusion, with native reduction and four BF16 boundaries requiring proof. An isolated two-file CPU implementation is assigned; production remains unchanged. Prepared `sdpa14.json` compares decode grids and K chunks at B1S512/B32S513; unlike the optimized single-chip defaults, the TP4 policy does not set short_capacity_decode_k_chunk and currently resolves it to512. The runner now records the resolved plan and actual cache dtypes; profiler evidence remains separate. Four remaining down-block choices are rebased in `geometry13_full_down.json`.

### Resume: recurrence12 and remaining scalar epilogues

Resumed the same active goal and serialized queue after the previous context
boundary. The explicit original >=70% roofline AND two kept gains <3%, OR8h
occupied-device rule still controls; neither condition has been met. No default
selection or stage acceptance is implied by the candidate measurements.

`recurrence12` compares original head ownership, exact column/pair ownership,
and native state packing on three regression fixtures and two real-text
fixtures. Early completed rows show column ownership reducing B1 traced decode
from0.4942 to0.4762ms, with B32 gains below1%. Native packing is about1% faster
at B32, but is classified as a precision change because persisted state bits
differ;512-step real-text quality evidence is retained in the prior receipts.
No PCC threshold has changed. Final decisions await the complete table.

CPU-only isolated recurrent epilogue work produced `local_norm_gate_candidate.py`
and `probe_local_norm_gate.py`. `norm_gate_cpu_v1.json` records384 geometry and192
actual host-descriptor checks; compilation/numerical equivalence/performance
remain unverified until the parent device probe. The native reference and both
head/tile candidates warm programs before capturing matched traces. Transient
RMS intermediates are allocated inside capture; no allocation acknowledgment
or skip-program-cache exemption is used.

Added a between-job STOP check to the final validation driver so a parent may
safely interrupt a future queue without terminating outstanding device work.

### Recurrence, SDPA and fused epilogue results (resume14)

`recurrence12`:20/20 complete whole-layer cases passed, including real-text
fixtures. `sdpa14`:24/24 passed. At B32/S513, grid8x4,K128 measured0.497197ms
versus grid8x8,K5120.522277ms in the same sweep; B1/S512K128/256 rows measured
about0.437ms versus0.4426ms K512. `sdpa15_short.json` checks the B32/S33 grid
choice before integration. Full-attention down projection K2/8/17 measured
0.471430/0.473212/0.474591ms at B32/S33. K34 failed allocation: static CB end
1521344 overlapped a live L1 allocation starting1322240. This is not numerical
evidence; `geometry15_adapt.json` prepares larger-core/reader/no-pipeline/DRAM
adaptations of that material block family. Small-block results and failure
console/provenance are preserved.

`norm_gate_smoke_v1` and `beta_decay_smoke_v1` both passed TP4 watcher10 with
allocation tracking1/skip0, bit-exact against native paths at B1/B3, address
rebind and changed-input trace. Norm/gate also passed one-core B3. Wider and
extreme-value coverage is queued; no production hook/default changed yet.

`integrated14_profile_l{0,3}_b32_s33` now profiles the actual current candidate
policies with advice tables/CSV/Tracy/provenance. GDN:40ops,574us kernels+24us
gaps, recurrent125us, norm epilogue18us, beta/decay preparation8ops. Full:
49ops,443us kernels+34us gaps; native partial-RoPE staging survives the selected
Q/K norm fusion. Dominant matmul rows achieve~74–86% modeled DRAM bandwidth,
but whole-layer compulsory-traffic fractions remain~.45/.39, below the explicit
70% stop target. Do not confuse per-op utilization with stage completion.

### Instrumentation harness repair, not a kernel rejection

`norm_gate_wide15`97 cases and `beta_decay_wide15`192 cases passed exact
comparisons, watcher10/tracking1/skip0. `norm_gate_perf15` mistakenly supplied
TT_METAL_WATCHER=0. rtoptions.cpp1467–1474 enables watcher whenever present and
sets interval0, causing~94s startup contention and invalidating timing. The
job then completed both numerical cases and closed all4devices normally; no
timeout, reset, killedprocess, or candidatekernel hang occurred. STOP prevented
launch of subsequent jobs. Initial tt-triage attempted default/tmp inspector
metadata; actualmetadata was generated/inspector. This failed diagnostic attempt
is retained and is not a device-failure diagnosis. Bounded watcher head/tail
and source-bytecount preserve the relevant evidence.

`zero_watcher_command_audit.json` found exactly one affected command. Its
performance validity is explicitly revoked in norm_gate_perf15_invalidation.json.
Launcher now rejects watcher=0; three new probes detect watcher by presence.
`norm_gate_perf16` retries with watcher/profiler unset. Existing passing broad
watcher runs are unaffected; newprobe source changes only instrumentation
classification. The pending qkrope smoke and short-SDPA rows use fresh outputs.

### Resume17: epilogue and new kernel admission

`epilogue16_summary.csv` records12/12 whole-layer passes. Beta/decay fusion:
B1/S5120.493938→0.484212ms; B32/S33 regression0.593818→0.579474ms;
real-text0.593679→0.578291ms. Head/tile norm-gate fusion is slower at the
existing compact recurrence boundary (real-text0.611587/0.627068ms).
`norm_gate_perf16` confirms standalone native/head/tile B1 medians14.055/
34.052/32.796us and B32 24.959/41.179/58.051us with watcher/profiler unset.
A coherent head-major recurrence writer→norm-reader family is being adapted
before rejecting the material movement opportunity.

`hidden_norm_fused_smoke_v1` passed exact B1/B3 watcher10/tracking1/skip0,
including50 queued generations and changed traces. Wider coverage and timing
remain pending. `native_recurrent_public_b1_smoke_v1` compiled but its probe
allocated a live buffer after capture; tracker rejected replay at line269.
All four devices closed normally. This is an admission/lifetime repair, not a
performance or arithmetic rejection. `qkrope_smoke_v4` further localized the
poison-fixture failure to full physical upload, before sharding or candidate.
TTNN from_torch defaults preserve_nan_values=False and uses device tilization;
explicit preserved-NaN upload is being added. All failed runs are retained.

### Resume18: admitted fusions and legal geometry boundaries

`qkrope_wide17` passes all32batches with exactselected-chain/tail comparisons,
physical NaN padding, rebound addresses, changed traces and allocation positive
control. `qkrope_perf17` selectedparallel-norm+native-RoPE→fused medians:
B1 42.307→18.617us; B32 76.449→25.844us. Whole-layer `fusions18` is pending.
`hidden_norm_wide17` passes96batch/value cases; standaloneperf native→fused
B1 23.475→33.330us and B32 22.585→32.026us. Do not extrapolate those deltas:
first matched whole-layer rows show B1~0.8% gain and B32~0.1% regression.
Paired whole-layer profiles will resolve the discrepancy before decisions.

`native_recurrent_public_b1_smoke_v2` passes tracker-enabled replay including
the unsafe-allocation rejection positivecontrol. Nativeoutput and reset sources
are allocated beforecapture; no corruptible-allocation exemption. Wider batch,
queued and whole-boundary timing gates remain pending.
`head_major_chain_smoke_v1` passes bit-exact B1/B3 and onecoreB3 recurrence,
state/readout/norm-gate outputs, queuedwork and changed traces. Command source
SHA1632adca8b0faa477eaa5bccc3718b4a0c8d5ec02ce30078c9404b488c259643
matches current helper despite a late-delivered agent completion message.

All four `geometry15` K34 adaptations fail with physical evidence and close
normally. c17/r2needs1757632B>1572864; c8/r3needs1959104B>1572864;
nopipelineends1504960 vs liveL1start1322240; DRAMoutputends1521344 vs1494272.
The driver now recognizes both max-L1 and live-allocation collision signatures,
retains exactcommands and records physical_rejection sidecars. Smaller K2/8/17
actual-text whole-layer comparisons pass; K2 remains faster. These are repeated
legalization attempts, not a first-API-error rejection.

Source now has opt-in fused_hidden_norm/fused_compact_qk_rope hooks and an actual
kv_dtype allocator. Batchedfill admits matchingBFP4/BFP8 payload/cache dtypes,
keeping BF16 decode updates and ownership rules. Defaults remain BFP8; candidate
KVprecision must pass unchanged output/peruser/state gates before any selection.

### Resume21 cumulative status

`fusions18` completed18/18PCC passes. Full-attention QK/RoPE fusion changes
B1/S5120.437504→0.411757ms, B32/S330.470333→0.419523ms,
B32/S5130.497227→0.446285ms, real-textB32/S330.469897→0.419008ms.
GDN hidden-norm fusion has~0.75–0.8%B1 gain and0.06–0.23%B32 regression;
no B32 selection is justified yet. Paired profiler files are prepared.

`native_recurrent_wide18` passed18batch/value cases and50queued updates;
minimum all recorded PCC0.9998360387. Watcher retainedlogs for this run,
qkrope_wide17,hidden_norm_wide17 andhead_major_chain_smoke_v1 contain no
Watcher-detected/assert/NoC-error lines. Complete-source/runtime gate remains
separate. `runtime_audit_resume21` passes129reachable functions,0violations
andnegativecontrols; finalruntime sentinel attachment remains pending.

`kv18_l3_regression_bf4_b1_s512` fails unchangedPCC gates: prefill0.983986,
decode0.996179; prefillK/V0.985474/0.985336 anddecodeK/V0.985478/0.985334.
Resolved allocations explicitly reportBFP4 forbothK/V. Devicesclosednormally;
no invalid performance result is ranked. Otheractualtext/regressionrows and
a directTP4 pack/unpack floor diagnostic are queued. Publiccontext/defaultKV
remainunchanged; no maxcontextclaim followsfromthis rejectedfirstrow.

Originalstoprule remainsunmet: `device_time_resume1908` conservativeoccupancy
2.09776h (TP4countedonce), below8h; wholelayerroofline below70%. No completion,
finaldefault,stage-reviewverdictorcommit is claimed.

### Resume 19:20 UTC — coherent epilogue decision and probe repair

Head-major recurrence→norm/gate passed `head_major_wide19.json` (258 exact cases, all B1–32, BF16/FP32 persistence, normal/zero/scaled inputs, queued updates and tracked replay). Complete chain `head_major_perf19.json` native/head/tile median µs: B1 63.262/72.178/70.417; B3 67.807/78.011/76.159; B17 103.662/113.335/111.756; B32 149.641/154.903/153.024. Reject this lower-movement family on coherent-chain evidence; no restore to the old readout layout was inserted into candidate timing. This is a measured regression, not an API rejection.

`native_recurrent_perf19` closed all devices after a harness NameError (`tracking` missing); added explicit tracking flag and watcher-presence check. Retained failure, restarted as `native_recurrent_perf19_v2`. Its custom controls now explicitly use the best measured cumulative-exp variant. No kernel arithmetic changed. Added opt-in static column runtime and dispatch-admission policy hooks for forthcoming whole-layer comparisons; production defaults remain unselected.

### KV precision evidence and reference provenance

All four `kv18` BF4-cache rows completed and closed normally; unchanged quality gates rejected each. Real-text B32/S33 whole output passes (.999797 prefill/.999848 decode), but K/V state fails. `kv20_text_quantization` isolates pack loss on the accepted single-chip TTNN BFP8 cache from recorded real-text activations: BF8 round-trip is exact; BF4 minimum rank PCC is .976607–.977965. These are **not HF cache tensors**: initial probe field `pcc_hf` was mislabeled. `kv20_reference_provenance_correction.json` records exact provenance without overwriting original evidence, and future probe output uses `pcc_reference`. No runtime fallback or numerical gate is relaxed. Keep BFP8 cache; the actual BF4 dtype executed successfully, so rejection is measured representation loss rather than an API/layout dismissal.

### Paired operation profiles (`fusions20`)

Four actual TP4 real-text B32/S33 profiles completed with profiler enabled and watcher absent. Every profile passed correctness. `perf/fusions20_{control,fused}_l{0,3}_b32_s33/` has advice-enabled prefill/decode tables, CSV/provenance and accounting; compact comparison is `fusions20_profile_comparison.csv`. GDN hidden-norm fusion removes two launches (33→31 ops) but total kernel+gap changes only576.491→574.833µs, consistent with negligible uninstrumented B32 whole-layer gain. Full-attention QK/RoPE fusion reduces material operation/movement cost (see comparison CSV), consistent with the uninstrumented fusions18 improvement. Matrix advice marks all four dominant decode matmuls optimized; those reports do not prove whole-layer70% roofline. Runtime/profiler numerical checks are kept separate from finaldefault acceptance.

### Resume 19:42 UTC — continue existing stage, precision diagnosis and wide checks

Resumed existing queue without restarting prior experiments. `packed_qkrope_wide22` passes all32 batches (changed/rebound inputs, poisoned padding, tracked traces); `hidden_norm_dispatch_wide22` passes96 cases. Uninstrumented `packed_qkrope_perf22` split-fused/direct-packed complete-boundary medians in µs are B1 33.407/28.024, B3 33.613/28.383, B17 38.766/35.277, B32 42.402/36.425. `hidden_norm_dispatch_perf22` native/dispatch medians are B1 23.495/33.310, B3 23.431/33.397, B17 23.283/32.957, B32 22.505/32.051. The dispatch protocol adaptation does not independently improve hidden norm. Whole-layer gdn22/attention22 comparisons remain queued; none of these results selects final defaults.

`column_runtime_wide22` is running four-path exact comparisons under watcher10 and allocation tracking1/skip0. STOP is an explicit between-job queue checkpoint; no device job is interrupted. The initial1200s bound may be too short for132 cases because allocation tracking collects garbage on each replay. Preserve its exact partial results and outcome, then split remaining batches with bounds based on observed duration if needed. Do not treat a timeout as a numerical rejection.

Independent AutoDebug completed `debug_native_precision/AUTODEBUG.md` after a sandbox preflight failure and authorized unsandboxed source-only retry. Its leading diagnosis is FP32-state FPU-readout operand narrowing amplified by cancellation; prepared BF16 also has a CB8 direct-unpack/FPU-consumer conflict. No hardware claim follows from that source diagnosis. AutoFix now tests a readout-only SFPU adaptation, keeping native update and BF16 persistence unchanged and preserving all .995 per-head/per-user gates. The first queued zero-state diagnostic will save all22 original-path comparisons before failing, so missing prepared metrics can be resolved. Separately, CPU persistence reproduction shows that long BF16/FP32 trajectories can diverge even for exact dtype-matched math; the existing failed comparison remains failed.

Using tt-enable-tracing for uncovered prefill dispatch advice: a separate actual-TP4 prefill probe is being prepared, with all buffers/programs warmed before capture, fresh state per measured replay, changed-input and post-prefill decode continuity checks, allocation positive control, and separate instrumented correctness/performance runs. Existing eager-prefill before/after semantics remain unchanged. No full-model/vLLM work is started.

### 19:52 UTC — bounded wide run and clean health recovery

`column_runtime_wide22` reached its1200s timeout after55 completed exact cases, with continued progress and no numerical failure. Original report/log/watcher remain unchanged; `column_runtime_wide22.timeout.json` records incomplete status. No manual signal or reset was used. `fuser` confirmed no device owners; bounded `health_after_column23` completed exact arithmetic on allfour ranks and closed normally. Prepared `column_remaining23.sh` covers unfinished B14–32 pluslow-core cases in three3600s bounded groups; already completed B1–13 cases are retained. The three completed B14 cases repeat only because batch is the probe selection unit.

Releasing the between-job STOP checkpoint after health success; starting `prediction23.sh`: full original zero-state diagnostics, readout-only AutoFix smoke, recurrence-prediction smoke, then25GDN and12attention whole-layer comparisons. Independent source review found no arithmetic/ownership blocker in precise readout; it caught watcher=0 in the new probe/example. Corrected detector to environment presence and performance command to unset watcher before any device execution. No quality gate changed.

Checkpoint reason: Parent queue checkpoint after column_runtime_wide22: run native precision diagnostics and prediction smoke before pending gdn22/attention22. Do not interrupt current device job. Remove only after driver closes.

`prepared_native_zero23` completed all22 comparisons before the unchanged numerical failure: public native and both prepared state dtypes have worst per-head outputPCC0.9887767045; statePCC remains above0.999997 forBF16 and0.99999986 forFP32. Bothcustom controls pass. This confirms the compact preparation alone does not repair the first-step readout. `native_readout_zero_smoke_v1` then hit a compile-only missing SFPU declaration; all drivers closed normally. Added the defining `compute_kernel_api.h` header and retried fresh `native_readout_zero_smoke_v2`; neither failure is an optimization rejection or changed gate.

`native_readout_zero_smoke_v2` passes after the include repair. First-step per-headPCC0.9887767→0.9999997 forbothpersistentdtypes with original/corrected statebitidentity; secondstep also passes. This discriminates readout narrowing from native update changes. Broaderquality/perf pending. Started `whole23.sh` on the releasedmesh: prediction smoke followedbygdn22/attention22.

### Resume24 — complete whole-layer comparisons

`whole23.sh` completed normally: prediction smoke6 exact cases, all25GDN and12full-attention whole-layer PCC gates pass. Compact timings: `gdn22_timing24.csv`, `attention22_timing24.csv`; individual JSON/commands/logs remain authoritative. GDN column1 improves B1 decode0.484161→0.466628ms andB32S330.578870→0.575519ms; real-text agrees. Dispatch admission adds a smallB1 gain but no consistentB32 benefit. Full-attention direct-packed QK/RoPE has smallB32 improvement0.419471→0.417983ms, butB1 decode0.411243→0.412949ms; isolated fusion win does not justify claiming a whole-layerB1 win. Defaults remain unselected.

Independent CPU composition audit `prediction_column_composition_audit23` proves source transforms commute and768 ownership partitions are complete/disjoint; combined hardware/perf remains pending. Independent final-validation readiness audit repaired actualHF continuation checks and mixed-row contributions, preserving prior self-reference/drop-history gates and documenting reference provenance. Remaining native/build provenance, coverage receipt, boundary instrumentation and B1 prefill statistical checks are being repaired before final gates. No acceptance requirement is relaxed.

### Resume24 — first candidate gates and measurement provenance

`attention_gate_smoke_v1` passesB1/B3 exact native output, poison, changed/rebound inputs,50queued traces and allocation positive control underwatcher10/tracking1/skip0. `focused24_remaining.sh` serializes broaderreadout, predictionperf, attentionwide/perf, jointprecision and tracedprefill; rootownsallfourchips.

`native_readout_stress24` passes eightB1/3/10/32 zero-state/normal cases, four changedsteps plus50queuedreplays. Corrected/original native state remains bitidentical. Original-control failures are retained separately; candidate quality .995 unchanged. `prediction_perf24` passes8exactB1/3/17/32×BF16/FP32 cases; BF16 mediansbaseline/candidateµs52.717/51.401,56.514/55.311,89.413/86.891,131.811/126.912. Added optional static-column composition to the same independent probe after this jobclosed, backedby `prediction_column_composition_audit23`; CPUsource checks pass. `combined24.sh` is prepared, not run.

Immutable originalTP4 source exported fromHEAD8502f42bca20 into `stage4_control_multichip_decoder.py`, verifiedSHA082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44. It is a test-only original-control implementation for same-current-harness paired B1 prefill measurements, never a runtime fallback or single-chip replacement. Finaldefault jobs will have no control-source or policyoverride. Newrun_job provenance inventories staged/unstaged native sources and buildbinaries; loadedbinary receipts are recorded after timings.

### Resume25 — completed evidence and next coherent families

`adapt24.sh` closed normally through all six traced-prefill workloads before its between-job STOP. The full-attention B1 quality probe initially failed a harness counter assertion: native single-user paged_fill_cache does not increment the batched ordered counter. The repaired probe instruments actual native fill calls during capture, still requires zero bulk calls, and requires the ordered counter for B>1. Fresh `traced_prefill23_l3_b1_s512_quality_v2` passes; no timed-forward algorithm or quality threshold changed. Eager/traced median prefill µs for GDN B1S512, B32S33, B32S513: 2305.475/1615.286, 5803.239/4526.670, 44179.652/43057.979. Full attention: 2014.700/1417.028, 3914.772/3303.584, 35419.489/35716.356. Thus traced prefill improves the shorter workloads, but full-attention B32S513 is slightly slower. These are separate traced-prefill experiments with reset/copy boundary, not replacements for final same-harness eager-prefill numbers.

`attention_gate_wide24` passes64 exact cases (all B1–32 × logical heads6/32, poisoned padding, changed/rebound inputs, queued replay). Native/direct canonical L1 gate medians µs B1/B3/B17/B32: 26.655/17.753, 26.781/18.135, 28.456/21.887, 20.147/26.759. A real row-band adaptation distributes each output tile across two disjoint half-tile writers. CPU ownership224 cases and watcher/tracker smokeB1/17/32 pass. `attention_gate_bands_perf24` gives native / one-band48 / two-band96 µs B1 26.652/17.759/24.202, B3 26.791/18.132/24.726, B17 28.459/21.889/28.712, B32 20.141/26.711/31.505. Reject the row-band adaptation on measured latency; do not claim B32 improvement. New off-default decoder hook preserves actual o_proj canonical input and excludes fused-MMRS restore for the direct-L1 family. Whole-layer comparison is queued in `attention_gate25.json`, both QK families and all four relevant regression/text workloads.

Joint output BF16 adaptation passes24 exact combine cases,14 actual-weight projection cases and2 actual B32S33 projection cases (`joint_output_*_v1`). The phase-specific full-layer matrix `joint_output23.json` remains queued. Prediction register retention passes isolated timing and changed-state tests; a new off-default hook composes its exact reader/compute transform with full column reader/compute/writer ownership. Combined hardware is queued, not claimed passed.

New coherent fused distributed norm→activation gather→actual projection-input/projection family passes first B1 TP4 watcher/tracker smoke `fused_dit_norm_b1_smoke25`. Broader B1/3/17/32, both kinds and both actual norm weights, separate quality/perf are queued in `norm_whole25.sh`. Inputs in these probes are labeled synthetic residuals with real model weights; no actual post-attention-residual claim. The separate native minimal fused RMSNorm family has a source-proven invalid stats iterator, mapping and phase-init bugs; installed AutoDebug report is `debug_fused_rms_minimal/AUTODEBUG.md`. AutoFix repair preparation is in progress without changing native source/binaries under active probes. Neither family is rejected on its first API failure.

`joint_conv_smoke25` failed the finite PCC input assertion after normal device closure. Investigation preserves poison/finite/exact gates and will record per-path diagnostics. No performance result or rejection follows from this first failure. Root owns the entire four-chip mesh and all jobs remain bounded and serial. Final defaults/gates/review/commit and the strict original stop condition are still outstanding.

### Resume25 — native AutoFix verified, convolution dtype repair

All mesh jobs closed before applying `debug_fused_rms_minimal/repair.patch`. Canonical build `cmake --build build_Release --target ttnn -j 12` and both runtime/pybind installation components completed; `fused_rms_minimal_build25.json` records all18 prior/new native sources and installed binaries, with complete build log. Old build3 binary hashes are now historical, not current final-build proof. The native repair replaces invalid iterator/core partitioning with bounded gathered-page mapping, admits one sender-local stats shard containing all rank tiles and one link, and replaces the invalid second kernel hardware startup with sender-local phase data-format reconfiguration. Independent AutoDebug/AutoFix and CPU mapping/admission evidence are under `debug_fused_rms_minimal/`.

`fused_minimal_norm_b1_smoke25` passes96 strict norm/projection comparisons, two changed banks, actual FP32 gamma rebinding, six trace allocation checks and effective positive control; mesh closed normally. First coherent norm/projection traced µs B1 input control50.828/146.846 versus minimal41.920/141.321; post control50.983/155.833 versus minimal42.166/150.733. These are within-layer boundary candidates, not whole-decoder or final-default results. Broader minimal and DiT cases continue serially with separate watcher quality and uninstrumented timing. No isolated native API error is used as a performance rejection.

Joint-convolution diagnostic v2 is explicitly INVALID as a quality gate: a diagnostic indentation defect omitted some per-user output comparisons, and an unequal-constant PCC diagnostic raised before complete capture. Preserve its failed artifacts; no pass claim. Corrected v3 records all12*B comparisons before failure and saves `joint_conv_smoke25_v3.first_failure.pt`. Native output and candidate corrected rest/history are exact, but candidate convolution output has1920/2560 mismatches and240 nonfinite values per rank. Source establishes that copy_init does not change unpacker dtype: FP32 joint-combine startup persisted while reading BF16 history/taps/token. Added explicit source-A format transitions before joint combine and before convolution. No arithmetic, rounding, state layout or quality gate changed. Fresh `joint_conv_smoke25_v4` passes bothB1/B3, changed/rebound inputs,50queued traces, watcher and allocation positive control. Wider timing/quality remains to run.

Independent hook audit finds selected attention gate admission/ownership and prediction-column composition consistent. Prediction transforms apply to every S1 call, including prefill/resume tails. Added constructor admission requiring packed batched convolution including B1 and cumulative-exp baseline for prediction, so an invalid policy fails before any convolution state mutation. Defaults still off. `remaining_bottleneck_audit25.md` identifies new traversal candidates and explicitly warns that existing S33 profiles and decode roofline bounds cannot explain S513 prefill latency. Fresh S513 profiles are queued before attributing any long-prefill saving.

### Resume26 — continuation, extended quality failure retained

Continuing the same active stage and artifacts. Original prompt remains stricter than the installed skill/current goal summary: stop requires ≥70% whole-layer roofline plus last two kept gains<3%, or8h occupied device time. Neither is established. No final-default or completion claim.

All32 minimal norm quality/perf cases pass. Native admission probe passes12 expected guards plus valid rebinding; storage rows1/9 at B1/B32 pass watcher/tracker/50queue. DiT full-attention B1 post-norm original setup failed a256-byte L1 overlap caused by retained verification outputs. Only snapshot copies moved to DRAM; actual canonical projection input stays L1 in both arms. Fresh quality/perf v2 pass; remaining full-attention B3/17/32 jobs are serialized in resume26.sh. Cross-regime absolute timings are not merged; each candidate has its own matched control.

native_readout_perf25 fails the unchanged per-head output gate after702 fixed-input recurrent updates. Both custom BF16/FP32 paths pass; original and precise-readout native updates remain bitidentical but their long output trajectories fail. Original/corrected native FP32 worst-head output PCC0.193892/0.177455; BF16 0.992189/0.992535. Complete-boundary B1 custom≈52.5µs, original native≈88–89µs, precise readout≈97–98µs are failed-candidate diagnostics, not accepted optimized numbers. All devices closed normally. Independent source diagnosis and discriminating AutoFix preparation requested; original failed report preserved. combined26.sh omits the duplicate unrun native_readout_perf24 and retains independent prediction/column work.

Unapplied bounded runtime norm integration reviewed and guarded against minimal_swiglu=True; prefill-only minimal SwiGLU remains supported. Whole-decoder off/minimal/DiT27-job comparison is prepared in norm_whole26.json; not run yet. New live-row and compact prefill-QKV smoke probes follow remaining DiT jobs.

Applied bounded norm integration patch331a86e80493dc49c6779d846c3230e74b0c6905d44d328d6ea8fc114a102f4e after all prior jobs closed; all model Python AST checks pass. First minimal whole B1 smoke rejected mismatched reference metadata because requested stress50 was absent from existing reference. This was a fixture mismatch after successful first prefill/decode, not a numerical/native failure. Fresh frozen-TTNN reference jobs explicitly include stress50; fresh minimal retry uses _v2 output. No metadata bypass or gate change.

All remaining DiT focused quality/perf cases pass, giving32 cases per norm family. fused_norm_timing26.csv contains64 traced matched boundary/projection rows and explicit snapshot-memory regime. Original L1-overlap failure remains retained.

recurrent_live_row_b1_smoke25 failed an oracle check while baseline/candidate prepared tensors were exact on all4 ranks. Independent source diagnosis shows the probe mapped SFPU scalar slots to contiguous eight columns, but native reduction expansion selects even/odd columns. Corrected oracle maps c to2*(c//16)+c%2, with native source evidence and CPU inverse mapping in recurrent_live_row25_v2.cpu.json. Helper/kernel and all exact/.995 gates unchanged; v2 hardware remains pending.

Device ledger device_time_resume26.json records2.877566h conservative occupied time, excluding uncertain endpoints and host postprocessing. Stop condition remains unmet.

### Resume26 — complete norm comparisons, remaining integration gates

All27 whole-decoder off/minimal/DiT comparisons pass unchanged output/contribution/per-user/state and changed-input traced replay gates (`norm_whole26_timing.csv`). GDN B1 regression traced decode is0.467117/0.450127/0.453307ms; actual-text is0.466958/0.450065/0.453140ms. GDN B32S33 is0.575952/0.565775/0.568183ms; S513 is0.575230/0.565905/0.568252ms. Full attention B1 is0.411646/0.403780/0.408923ms, B32S33 is0.419200/0.411387/0.412794ms, S513 is0.446352/0.439386/0.443497ms. Minimal is the leading norm candidate, not the selected final default. Its64-owner L1/resource feasibility, all-batch mixed-trace ownership, final context and final stress remain mandatory. Prefill sample variation is retained verbatim; this decode-only hook does not establish a prefill speedup.

Allfour integrated B1 norm stress50 watcher/tracker runs pass after generating explicitly matching frozen single-chip TTNN reference trajectories. These reference jobs are only oracles; each candidate executes the actualTP4 path. `prefill_qkv_smoke25` passes four exact B1/B3 × S33/S512 movement cases with changed inputs, queued replay and tracker positive control. Broad nonaligned geometry/perf and whole-prefill integration remain pending. The new root hook proposal is unapplied and separately CPU-reviewed; no default changed.

Long-prefill B32S513 profiles run serially with watcher unset. Native seed-only precision AutoFix is prepared in `native_seed_autofix25.md`: it mirrors the native decayed-state DST result into a direct-FP32 seed CB while retaining original FPU prediction/update math, and records all nine raw trajectory checkpoints before evaluating any failed gate. Hardware is pending.

### Resume27 — native recurrence decision and bounded ownership

`fused_norm_owner26_l0.json` and `_l3.json` both pass the actual-owner watcher/tracker probe: minimal and DiT, two owners each, both real affine phases/projections, B1/3/17/32 changed inputs and queued replay, all32 host views sharing bounded physical storage, and a tracker positive control. The 64-owner simultaneous allocation/execution check remains pending and is not inferred from these two-owner runs.

`native_seed_smoke26` passes, while fixed702 and perf runs retain overall failures. Independent analysis in `native_seed_autofix26_result.md/.json` distinguishes their outcomes: direct-FP32-seed BF16 passes all sampled fixed checkpoints, including min-head output PCC0.997482984 at702, but costs98.74123us versus52.70483us custom (1.87348x). Seed FP32 fails accumulated-update quality and costs99.30520us versus52.53952us (1.89010x). Actual-Q contraction of saved seed-FP32 state agrees with readout to0.9999999999743; FP64 updates of actual prepared operands with FP32 state persistence agree with the original oracle to0.9999999999016. Thus this FP32 failure is in the remaining native update arithmetic, not a reason to weaken the oracle. Reject these measured variants on cost (FP32 also quality); this does not reject an unimplemented full native arithmetic rewrite. All device jobs closed normally; no reset.

The live-row v2 smoke passes exact preparation/output/state and independent unchanged oracle after correcting only the erroneous host scalar-slot mapping. `joint_conv_perf25` passes its complete-boundary quality/timing collection at B1/3/17/32, FP32 L1 joint input and48/80/110 workers; whole integration awaits matched cost analysis.

Applied opt-in GDN combine-output windows from `gdn_combine_window26/candidate.patch`, default false. Actual B32S513 executes temporal512 (16384 flattened rows, four4096 pieces) plus a separate S1 tail. The profile's706us concat copies260MiB/rank; the proposed writer owns disjoint windows of one invocation-local output. No BF16 projection policy is admitted by this first variant. `combine_window27.json` prepares three matched complete-layer shapes, including two nonaligned boundary cases, plus separate watcher runs. Hardware pending.

The stack-resource harness now accepts explicit per-kind policy dictionaries in addition to its existing flat-policy format. `norm_resources26.sh` prepares four independent64-owner watcher/tracker experiments for minimal/DiT at B1/B32. These remain resource/isolation probes sharing representative weights, not a full-resident-weight capacity claim.

`joint_conv_result26.md/.json/.csv` rejects the measured FP32-L1 fused combine/conv family: best80-core rest-only B1/3/17/32 costs120.9165/129.0143/188.1749/249.8193us versus28.5856/30.0226/39.3864/48.6973us existing complete boundary. All41,220 recorded rank/user/tensor checks across smoke/perf are exact, finite andPCC1. Both full-packed and lower-movement rest-only families were measured without a restore in the latter, at48/80/110 workers. BF16/DRAM input is source-supported but not measured and is not claimed rejected. No whole-layer integration is justified for this4.23–5.13x slower measured implementation.

Native output-admission build27 installs successfully. First `minimal_norm_native_validation27` stops before any native negative because public TensorSpec rejects a logicalB1/paddedheight64 width-sharded view at construction (`Shard height64 must match physicalheight32`); devices close normally. The v2 fixture records this exact construction rejection separately with native_guard_executed=false, retaining22 constructible native negatives and the defensive native padded-shape check/source proof. No generic API failure is counted as a native guard success.

The admission v2 fixture also exposed row-major width-shard height needing logical1 rather than tiled32. v3 adapts only that malformed-output fixture to a constructible RM shard(1,160), so the native TILE-layout guard can actually execute. All guards/thresholds retained. v2 devices closed normally; no negative was falsely counted.

Admission v3 proves first20 native cases, then the host output is rejected earlier by `device_operation.hpp:564` with exact `Device Operations expect device tensors as inputs`. v4 classifies this framework guard explicitly (native_guard_executed=false), preserving exact expected-message and unchanged cache-count checks. It does not claim the defensive native allocated-device check executed. Cross-mesh case and healthy B1/B32 remain required. No device/kernel error or reset occurred.


### Resume27 — applied prefill candidates, native admission and audit

At21:24:26UTC `prefill_qkv26_quality_g3` closes normally; allfour groups pass64 exact TP4 cases, B1..32 × S33/S513, watcher10, allocation tracking1/skip-cache0, changed inputs, rebinding, queued50. `candidates26.sh` then exits at the requested between-job STOP; its later manifests remain unrun. No process was killed.

Applied native output-validation patch, `direct_prefill_qkv27.patch`, `prefill_swiglu27.patch` (both main and flat dispatch), and `flat_gather_prefill27.patch`. All candidates remain opt-in; selected defaults are unchanged. Main SHA55986ca3acf2b97969c42422f1a0e3ad5e24932bcec8fc311535530126279f50. The native build/install receipt is `fused_rms_minimal_build27.json`; it supersedes build25 binary pins. Only admission changed in this rebuild; kernel math is unchanged. Updated the norm-owner docstring and failure-latch comment to describe normalization phases precisely.

`minimal_norm_native_validation27_v4` passes22 exact-message/no-program-cache-growth rejection cases (21 native,1 earlier device-operation framework host-tensor check), plus the separately classified impossible padded-height TensorSpec construction, and8 positive B1/B32 input/gamma/stats/output/semaphore rebindings. Prior v1/v2/v3 fixture errors remain on disk, with explicit classification above; none counts as a native guard success.

`runtime_audit_resume27.json` exposed three static proof gaps rather than an observed host fallback: unpropagated packed `_project_compact` owner and imported source-string count/replace. Audit repair additionally follows the previously unbound FusedDecodeNormOwner.normalize path, verifies actual constructor/install/owner identity and model argument, distinguishes setup from runtime, and requires AST-proven literal string provenance. `runtime_audit_resume27_v2.json` passes154 reachable functions, zero violations and48/48 mutation controls, including new prefill QKV/SwiGLU helpers and misleading literal/owner/binding substitutions. Runtime sentinel is still not_provided and must attach final-default runs; this is not final runtime acceptance.

Active serialized continuation is `resume27_v4.sh`: QKV boundary timing,64-owner norm resources, same-harness B1 diagnostics, combine windows+watcher, long geometry, flat-preserving normalization/movement, direct QKV whole-layer, SwiGLU bulk/tail, pending attention gate/joint precision/recurrence comparisons and S4097. Every job remains bounded and checks STOP between jobs. All numerical and quality gates remain unchanged.


### User-directed wrap-up28

The user explicitly ended optimization search at the stage time/token cap; exact instruction is preserved in `user_wrapup28.json`. This supersedes the earlier optimization stop threshold, without changing PCC, trace, watcher, context, final-default performance or review gates. Last recorded occupied-device time was3.2575525h (`device_time_resume27.json`); no8h or70% claim is made. Pending search manifests will not run. Required final gates remain work.

`norm_resources26_minimal_b1` failed during64-owner construction with L1 allocation exhaustion (10240B required per bank;5440B free), before any prefill. All devices closed normally. Minimal fused normalization is therefore not eligible for the selected default on present evidence. AutoDebug is finishing its source-only investigation; no optimization repair sweep will be started after the wrap-up instruction. Its first launcher failed namespace sandbox preflight before starting a model; explicit sandbox-skip retry is inspection-only under existing authorization. Earlier two-owner and whole-layer passes do not establish64-owner feasibility.

Selected final28 defaults are the measured `traced23` policies plus GDN column-grain1, with `distributed_decode_norm=off` for both kinds. The exact selected overlays are in `final28_selected_overrides.json`; no runtime policy/file fallback is introduced. Final-source measurements remain required. Black formatting and compileall completed for stage-owned Python; stage-owned Pyflakes undefined-name checks pass102 files. A standalone RoPE probe closure now receives its input banks explicitly, removing a lint lifetime ambiguity; runtime RoPE is unchanged. The all-repository scan contains historical prior-stage probe warnings and is not used to claim a clean stage-owned scan. CPU runtime audit passes154 reachable functions and zero violations; final runtime sentinel attachment remains pending.

### Required HF-contract repair after wrap-up

Final28 passed20 correctness jobs, then independent HF-contract GDN B1S31 failed contribution row12 at0.9860137708810336 (whole output0.9993891826645709). Immutable stage4 reproduces exactly the same S31 scores (`final28_contract_stage4_l0.json`); this is inherited, but remains a failed gate. The complete unchanged HF contract passes with only GDN attention_fidelity=HiFi4 (`final28_contract_hifi4_l0.json`), improving the failing row to0.9997051475414599. No input, reference or threshold changed. Source diagnosis is `final28_contract_diagnosis.md`; fresh bundled AutoDebug is verifying the cause in `debug_contract28/`. Final29 selects that conservative precision repair; broader phase-specific fidelity changes remain untried. Final28 successes are preserved but are not relabeled final29. Required affected gates and final-default measurements must use the new source.

### Fixed paired B1 timing uncertainty

All five planned GDN pairs completed on final29: mean control2.395564ms versus final2.194060ms (8.41% lower), but the exact one-sided median bound is+0.037668ms, so the predeclared95% zero-margin nonregression inference is inconclusive. `final29_b1_measurement_review.md` preserves all pairs and distinguishes point improvement from confidence. No best-sample selection, estimator change or additional optional timing series is authorized by the wrap-up. This statistical uncertainty is explicitly submitted to the independent stage review for classification under the user-authorized non-blocking carry rule; it is not claimed passed. Numerical/trace/context gates are unaffected and remain mandatory.

Full attention likewise has lower paired mean1.761020→1.701235ms but positive median confidence upper bound+0.057086ms. Both fixed series are preserved in `final29_b1_uncertainty.json`; neither confidence gate is falsely called passed.

Fresh bundled AutoDebug completed: `debug_contract28/AUTODEBUG.md` verifies the fidelity-only contrast and recommends the already-selected broad GDN HiFi4 default as the smallest demonstrated repair. `AUTOFIX_contract29.md` records the change and required final closure. The earlier norm report completed separately: ordinary lockstep L1 allocations share one address pool; row placement does not reduce reservation. Its exact allocation ledger reproduces the OOM, and ephemeral stats/output scratch is only an untried proposal. Norm remains off; no context capacity reduction is claimed.

### Final29 required gates

Command `python -m models.autoports.qwen_qwen3_6_27b.tests.run_optimized_multichip_validation --prefix final_validation29 --run` completed all68 jobs with `pass:true`. Manifest and summary are `final_validation29_manifest.json` / `final_validation29_summary.json`; no selected-source/quality changes occurred during the run. Both real-HF contracts, ten correctness workloads, B2/B32 five-layer stacks, B1/B32 independent64-owner resource tests, both32K context smokes, eight repeated stress processes (two trajectories each), and twelve nonaligned watcher workloads passed. References are separate baseline jobs. Strict tracking remains1/skip-program-cache0, and watcher runs are separate from timing/profiling. `final29_capacity.py --write` with all six final performance artifacts succeeded; context stays262144, with maximum-context evidence inherited from stage4 and full-model capacity still a source estimate. `final29_capacity_update.log` records the updater result.

### Carried to next stage

DRAM-sharded profiler core/FLOP annotations assume8 workers where the selected two-reader path uses16. Do not use affected Cores, FLOPs[%], or compute-bound labels for utilization or rejection decisions. Correct the tool accounting in a later stage; retain raw output and measured latency. See `final29_profiler_limitations.md`.

#### Not tried (stop rule)

The user's cap stops optional search. Expected integrated gains are unmeasured; historical component gains do not establish whole-layer improvement. The following are untried ideas, not measured rejections: ephemeral/DiT fused-norm scratch for64owners; phase-specific GDN HiFi4; lower-movement prefill norm; prediction/column/live-row recurrent combinations; BF16-DRAM joint combine/convolution; long-prefill input-L1, row-cap, concat/window and geometry crosses under the actual selected fidelity; QKV batched/fragment movement adaptations; SwiGLU bulk/tail configurations; attention-gate whole-layer combinations; broader persistent-buffer signature capacity; and datatype-stage BFP4 MLP selection. Detailed prepared manifests, historical measured alternatives and exact next experiments are in `final29_handoff.md`, `final28_handoff.md`, `final29_candidate_decisions.md` and the final advice inventory. None is claimed to have run after the cap.

B1 prefill95% confidence remains inconclusive for both kinds despite improved means. Retain all pairs and use a prespecified independent confirmation series in a later authorized stage if required; no optional repeats occurred here. Rounds1–2 independently classify this as nonblocking uncertainty; round3 accepted it as nonblocking carry. All numerical/trace/watcher/context gates remain passed and unchanged.

Read-only final lint (`final29_readonly_lint.json`) passes Black across102 stage-owned model Python files. Isort/autoflake report import-order/unused-import cleanup; these nonfunctional items are carried under the cap, with logs retained. Earlier undefined-name/compile checks are `final28_stage_lint.json`. Final evidence hashes intentionally freeze source and raw artifacts. The local archival checkpoint will skip mutating formatting hooks and the500KB artifact-size policy so raw logs/CSV and the immutable stage4 control remain byte-exact; applicable nonmutating hooks will still run. This is an explicit artifact-preservation exception, not a numerical gate waiver.

All six `final29_profile_*` jobs completed with actual1x4 default execution, raw ops hashes, tt-perf-report tables/CSV and accounting. `final29_remaining.sh` exited0. Decode intra-layer CCL categories total66–69us/step; device kernels span0.393–0.601ms across workloads, and the limited traffic lower-bound/host ratio is43–45%. These instrumented figures are not final uninstrumented latency or proof of70% roofline. `final29_profile_summary.json` and the six per-profile accounting files retain scope/limitations.

### Final artifact consistency check and package-loader repair

The direct checker hit an ImportError in its runpy-loaded planner's relative resource-validator import. `check_final29.py` supplies only that exact planner's qualified run_name and executes the unchanged checker. `final29_checker_packaging_fix.md` explains the mechanical entrypoint repair; all frozen source/receipts stay unchanged. Repaired CPU check completed25 groups, with only B1 confidence inconclusive and pending review/commit errors. Actual output remains `pass:false` in `final_gate.json`; no failure is relabeled. All device gates/profiles are complete before independent review starts.

### Independent review round1 and dispositions

Fresh xhigh `final29_review_round1.md` returns more-work-needed solely for carry/review sequencing, with no correctness/accuracy or missing required device evidence and no additional device run requested. It rechecked8,935 source/native pins. Both B1 confidence bounds remain inconclusive, now independently classified as nonblocking performance uncertainty; the original checker result remains false. The report also traces the profiler's175.65% annotation to its fixed8-worker model versus actual16-worker/two-reader DRAM matmul. `final29_profiler_limitations.md` documents exactly which annotations cannot be used; latencies, dtype/fidelity and byte-based traffic accounting are unchanged. Raw tool output is preserved. Tool correction and optional prespecified B1 confirmation are carried to the next authorized stage, not claimed fixed or passed. Preparation-tense docs now reflect completed gates. Fresh diff-only follow-up review will check these dispositions. Shared native fixes will receive a separate local `Shared-code fix:` commit with their regression probes, followed by the model/evidence checkpoint; neither will be pushed.

### Independent review round2

`final29_review_round2.md` closes the documentation remedies and identifies no new correctness, accuracy, or missing-device-evidence work. Its more-work-needed verdict is procedural: obtain the fresh round3 carry sign-off and local checkpoints. The original false checker output is preserved. The owner explicitly resumed only for review sign-off and commits; no new tuning or device runs are authorized. Raw final report CSVs/tables will be included in the archival checkpoint.

### Independent review round3 acceptance

Fresh xhigh `final29_review_round3.md` returns **pass-with-carried**, Required Work empty. Its accepted carry comprises B1 confidence uncertainty, DRAM-sharded profiler annotation correction, and the untried optimization/import-cleanup inventory above. No correctness or accuracy gate is waived, no failed check is relabelled, and no additional hardware run or tuning occurs. The original `final_gate.json` is preserved byte-exact. Local checkpoint/provenance recording follows this acceptance.

Local shared-code checkpoint: repo `/home/ttuser/qwen36-27b/tt-metal`, branch `main`, SHA `3c8d8198482286af9c23688a1132d5caf4efa8f3`. Contains16 native CCL files and3 focused regression probes. Applicable nonmutating commit hooks passed; frozen formatting/artifact-size exception above applied. No push. Commit output: `final29_shared_commit.log`.

Local model/evidence checkpoint: repo `/home/ttuser/qwen36-27b/tt-metal`, branch `main`, SHA `421b68be97e410ac9d84619f2dae081d11447193`. Includes final default, stage-owned tests, all review reports, raw tables/CSV and provenance logs; no weight binaries. Shared-code ancestor is `3c8d8198482286af9c23688a1132d5caf4efa8f3`. Applicable nonmutating commit hooks passed with the documented frozen-artifact exception. No push. `final29_model_commit.log` records hooks and commit output. Final acceptance metadata will be committed separately; the manifest byte inventory deliberately covers frozen sources/context, not self-referential closure documents.

### Final postcommit acceptance

Command `python models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/check_final29.py --manifest models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/final_manifest.json` produced `final29_postcommit_gate.json`:26 groups pass, including source/native, required validation, review and committed-source provenance; exit1/`pass:false` is retained for the sole B1 confidence inference. Both layer kinds remain statistically inconclusive. Original `final_gate.json` SHA0ddbb0607936b7033cb1e7e8c8b7d57fe5348d2c8b9dbc41a60e8f8503105ea4 is unchanged. `final29_acceptance.json` separately records round3 **pass-with-carried**, no Required Work and no quality waiver. This final CPU provenance check introduced no new device work or tuning. Final closure metadata is a separate local documentation commit atop the logged shared/model checkpoints; all commits stay local.
