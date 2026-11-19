# Qwen3.8-Flash-Next functional decoder

**ACCEPTED — clean-pass, independent review round 2.** This report covers only the
functional decoder. [Review verdict](stage_review_round2.md) closes the round-1 timing
finding; decoder, context, performance and component evidence is complete. Local
checkpoint SHAs are recorded in the work log. PCC acceptance remains **>= 0.995**. Controlled
fresh-prefill contribution discrepancies below that threshold are disclosed separately;
they are not strict literal-HF passes.

Reference: `Qwen/Qwen3.8-Flash-Next`, revision
`812d96e10a0dd82b69d749e68693698cf6c36daa`, Transformers 5.22.2, `qwen4_exp_text`. The
actual configuration has hidden width 2560, four residual streams (external width
10240), 512 experts with top-10 routing and advertised context 262144. Representative
real layers are 0 (GDN), 1 (GDN plus PLE) and 3 (indexed QSA). See [checkpoint
inventory](checkpoint_inventory.json), [context contract](../context_contract.json) and
[work log](work_log.md).

## Public API and state contract

Implementation: [functional_decoder.py](../../tt/functional_decoder.py).

```python
FunctionalDecoder.from_state_dict(
    state_dict, *, hf_config, layer_idx, mesh_device,
    packed_embedding=None, expert_dtype=ttnn.bfloat16,
)
model.create_state(*, hash_lengths=tuple(range(1, 33)), capacity=None, page_table=None)
model.prefill_forward(hidden_states, *, state, token_ids=None)
model.decode_forward(hidden_states, *, state, token_ids=None)
```

Weight loading, table conversion/upload, state allocation, hash-program preparation and
trace preparation are setup operations. Runtime consumes device inputs and weights; no
Torch, `from_torch`, `to_torch` or host fallback is allowed inside an audited
prefill/decode pass. Reference computation and result readback are outside that
boundary.

- Hidden input/output is BF16 TILE `[1,1,S,10240]`. Prefill returns every logical
  row and processes chunks of at most 32; decode consumes `[1,1,1,10240]`. Logical
  lengths need not align to tiles/pages/chunks. Padding must not be supplied as
  valid rows. Length and accumulated position must stay within the advertised
  context and the physically supported allocation profile.
- Tuple inputs with matching independent states support 1..32 requests at the API
  level. PLE also takes a matching tuple of token tensors. Requests may have
  different lengths/history; outputs are tuples. Physical limits are measured
  separately and do not follow from the API maximum.
- PLE token words are device UINT32 row-major `[1,S,16]`: word 0 is the raw token
  ID and word 1 is 1 for each valid logical token. A complete resident 320001536-row
  packed table is required; selected-row CPU lookup is only an HF test facility.
- Fresh requests need fresh/reset state. Continuation preserves all GDN
  convolution/recurrent, PLE convolution/token-history, or QSA KV/indexer state.
  Restoring a position alone does not restore a prefix. State and input buffer
  addresses remain stable during replay; changed inputs are copied into warmed
  buffers outside the captured pass.
- QSA uses page size 32, a next-absolute-position device tensor, and paged K/V.
  Capacity defaults to 262144 and must be a multiple of 128 within 128..262144.
  A setup page permutation covers all allocated pages. In-place page-table
  refresh requires correspondingly remapped K/V contents. The caller maintains
  valid positions and cache ownership; arbitrary mask holes and multimodal
  position schemes are outside this causal-text contract.
- Canonical PLE continuous-range capacity prepares **all 32** hash signatures
  before the measured lifecycle. A custom `hash_lengths` subset restricts the
  usable chunk signatures; every used tail must have been prepared. Decode,
  changed-input copy and relevant prefill programs must be warm before capture.

## Precision and resident storage

External activations, ordinary weights and QSA K/V are BF16. GDN recurrent state is
FP32. Model-local FP32 SFPU arithmetic and explicit BF16 rounding boundaries implement
sensitive GDN recurrence/projections, gated residuals, router, QSA indexer
projection/RoPE and indexer scores. These are correctness repairs; no optimized-decoder
stage is claimed.

Evidence for layers 0/3 uses BF16 experts. PLE resident tests use BFP8 expert storage
and a learned four-bit embedding codebook, groups of 10 values and an eight-bit scale
LUT. The complete packed table occupies 30,720,147,456 bytes; exact BF16 storage alone
would require 102,400,491,520 bytes. The measured device DRAM is 34,178,731,008 bytes.
Cache construction, exact packing/lookup verification and real-HF numerical checks are
distinct gates; compression does not waive PCC. See
`packed_embedding_group10_scale8_cache_verification.json` and the work log.

## Capability and evidence status

The precision runtime reached short-gate validation with
`precise_residual_projection.py` SHA256
`8af1023e7dcbfae6523769fbfa7245348f4bf022d4b303309417649a2e47046a`. A subsequent
native-slice preparation defect required bounded device readers: new logical input
shapes and chunk offsets retained device programs. The native-only `from_torch`
diagnostic recorded 338 new programs and 5,537,792 bytes after 324 exact-copy checks.
See `chunk_input_native_growth.json`; no capacity limit from the earlier proxy is
accepted. The repaired reader passes 644 exhaustive cases, including physical padding,
changed/repeated traced inputs and 38 rejected invalid contracts
(`chunk_input_final_v6.json`). Its three I/O programs retain 49,152 bytes with zero
growth across those cases. The PLE capacity and boundary evidence below includes this
reader repair; the six closed GDN/PLE short records below have explicit source closure, while
PLE advertised state-context validation now closes at262144 tokens. This single hash is not complete runtime provenance. Earlier passing records
remain supporting evidence until their full source identities are refreshed or
explicitly dependency-reviewed.

| Layer kind | Complete public decoder prefill | Reduced long-prefix/state path | Complete public traced final decode |
|---|---|---|---|
|0: GDN|Current-source strict short/public continuation gates pass; batch 32 retains the explicitly controlled fresh-contribution finding.|All 262144 attention rows pass on current sources; prefix minimum PCC 0.99997920. Full decoder weights remain resident; prefix omits token-local full decoder/MoE computation.|At position 262143: complete public traced decode passes, minimum whole/contribution PCC 0.999996996/0.999996960; `gdn_final_full_context.json`.|
|1: GDN + PLE|Full public literal HF at lengths 4768/4767 and terminal 32-length lifecycle pass; prepared-profile range 1..4768 source argument and CPU evidence closure exist; final review pending.|All262143 reduced PLE/hash/GDN prefix rows pass; minima PLE0.99769008, GDN0.99578416 and attention contribution0.99577355. Prefix omits token-local MoE.|At position262143: full public traced decode passes whole/contribution PCC0.997859125/0.997826090; `ple_final_full_context.json`.|
|3: indexed QSA|Current literal short contract, public continuation and batch 32 pass.|All 262144 attention rows from empty TT state pass, minimum PCC 0.99820900. The restored device checkpoint also passes a 31-row tail to prefix 262143 and final attention decode.|Normal complete public traced decode at position 262143 passes with minimum whole/contribution PCC 0.999942243/0.999936640; `qsa_ordered_full_decoder_v1.json`.|

The skill permits a reduced layer-level long harness when complete execution would be
too slow. Such evidence is explicitly paired with complete short public decoder,
nonzero-history continuation and complete final public decode. It does **not** establish
full decoder/MoE output accuracy on every long-prefix row. Final bridges must build TT
state through actual device prefix execution; any test-boundary export/reuse requires
matching source, state, page and position provenance. No HF state is injected into TT
acceptance runs.

The current [GDN long-context bridge](gdn_final_full_context.json) checks all 262143
prefix attention outputs plus the original final attention row. It then restores
the actual TT prefix state for two changed full-decoder inputs, repeated A/B/A/B
through one capture and four replays. Outputs and both recurrent state buffers
repeat exactly; all 264 audited passes complete with allocation tracking enabled.
This normal-mode run has no Watcher evidence and is not a full-decoder long-prefill
claim. Its source and log checks pass; independent stage review remains pending.

The [PLE long-context bridge](ple_final_full_context.json) also closes normally.
Its262143 prefix rows start from empty TT state with the complete table and decoder
weights resident. Every hash is exact, and every PLE/GDN/attention-contribution row
passes0.995. The complete public final decode restores that actual TT prefix for two
identical-input replays of one trace; whole/contribution PCC is0.997859125/0.997826090,
output and all four state buffers repeat exactly, and final hash/history matches HF.
All1028 audited passes complete. This is262143 reduced prefix rows plus a full final
public decode, not262144 separate reduced-component readbacks or full MoE on every
prefix row. Changed-input tests and Watcher coverage are separate short runs.

The PLE memory repair is integrated. `SparseMoe(reuse_private_buffers=False)` preserves
the ordinary path by default; the public decoder enables reuse only for PLE. It applies
SiLU and multiplication in place on the private expert gate buffer, then clears padding
on the private weighted-down buffer and uses alias views around the unchanged native
expert sum. No GR query split or output reservation is required. Isolated controls
preserve exact outputs at every Q from 1 to 32, including changed traced inputs and
interleaved eager calls. NORMAL allocation graphs measure the nonaligned reduction's
additional peak falling from 84,213,760 to 163,840 bytes.

The current literal HF checks pass every prefill row at lengths 4768 and 4767
(`ple_public_hf_4768_private_v1.json` and `ple_public_hf_4767_private_v1.json`). Minimum
whole/contribution PCC is 0.9954640871/0.9951774922 at 4768 and
0.9958494904/0.9956877869 at 4767. Both runs pass three traced decodes, reaching
contexts 4771 and 4770 respectively, with exact repeated outputs/states and clean
audits. The separate 3707 run retains a literal contribution failure at row 55 (PCC
0.9947104757), while every whole-output row passes. Its completed independent matched
comparison passes all TT/packed rows (minimum contribution PCC 0.997952722; row 55 PCC
0.9999113071). The literal failure and the CPU control's separate literal row 33 failure
remain explicit; this is attribution evidence subject to review, not a strict pass.
Exact records and earlier experiments are in the work log.

Capacity evidence includes two complete constant-input 4768-row public prefills and
traced decode, a fresh 4769 allocation failure, strict HF checks at both boundary
lengths, and all 32 terminal lengths from 4737 to 4768. The terminal test exercises
public chunks, decode and repetition under full resident I/O; it is a chunk proxy with
offset 0 chunks and the actual terminal tail, not 32 complete public prefills. The
default-runtime residue search observes 4736+r passing and 4768+r failing for each
residue r from 1 to 32, with stable post-probe allocator baselines. The 4769 failure is
MLP GR temporary fragmentation, not aggregate exhaustion.

These records and the [capacity source
review](../../tests/capacity_contract_source_review.md) support an engineering range of
1..4768 for the specified prepared profile, rather than a universal maximum across
implementations or allocator histories. [`final_capacity_qsa_closure_v4.json.xz`](final_capacity_qsa_closure_v4.json.xz) closes the
source/dependency checks, including the unreachable QSA constructor change; final range and capability-reduction review remain pending. A
separate physical obstruction prevents advertised resident prefill: hidden input alone
needs 5,368,709,120 bytes versus 514,902,528 free after the recorded model/state
allocation. Historical allocation policies and maxima remain in
[work_log.md](work_log.md); they do not bound the repaired implementation.

Batch limits are a **separate allocation profile**. Normal PLE batch 5 is now the
largest tested passing tuple batch under the recorded resident/single-trace policy. It
passes explicit packed-backend controls while retaining
`literal_hf_strict_passed=false`; final review must assess that finding. Normal batch 6
fails, and the [current batch-capacity analysis](ple_private_batch_capacity_bound.md)
independently gives a conservative total-memory obstruction: resident model, actual
six-request trace, six mandatory states and 100 MiB of simultaneously live expert
input/output require 34,190,803,328 bytes versus 34,178,731,008 available, an excess of
12,072,320. This calculation excludes the removed reduction clone and omits other
allocations conservatively. It applies to the recorded policy, not hypothetical
alternative encodings or capture schedules.

Watcher batch 5's smaller trace-overlap failure is an instrumentation limit, not a
four-request device bound. The old [batch 5
diagnosis](batch5_trace_capacity_diagnosis.md) is superseded. Batch 5 uses its recorded
short prompts and per-request preparation; it does not imply five maximum-length prompts
or five states with the batch 1 resource profile with all 32 signatures prepared. No
Cartesian product of separately tested capacities is claimed. GDN batch 32 is closed through the narrow source proof below; QSA batch 32 now passes literal gates on the ordered runtime. [`final_batch_capacity_qsa_closure_v4.json.xz`](final_batch_capacity_qsa_closure_v4.json.xz)
revalidates the normal PLE 5/6 records and lower bound without accepting a numerical
exception or promoting Watcher overhead to a device limit.

## Correctness evidence and disclosed contribution anomalies

The ordered QSA projection changes two lines inside the QSA constructor only.
For recorded linear layers 0/1, the [live source-closure inventory](linear_qsa_gate_evaluations_v3.json.xz)
retains the original QSA file hash, proves that exact constructor change is
unreachable, and checks every shared runtime file against its frozen hash.
Twelve numerical/source gates now satisfy their declared checks with this
explicit proof; controlled findings and physical-capacity review remain separate.
The checker rejects additional edits, changed shared code, unknown closure kinds,
stale bindings and non-linear layer records. This is not stage acceptance.

Whole-output and residual-subtracted contribution PCC are reported separately. Every
literal whole-output gate remains >= 0.995. Resumed-prefill and decode contribution
gates also remain literal >= 0.995, including every logical row.

Current-source [layer 0 public continuation](public_continuation_layer0_v7.json)
passes strict literal HF for independent prefixes 31/33, resumed windows 8/7,
and changed traced inputs. Minimum resumed-row contribution PCC is 0.9999600053;
minimum decode contribution PCC is 0.9999452233. Dropping history fails both
negative controls. Independent lengths 65/97/129 execute for the first time while
the tuple trace remains live, with minimum whole/contribution PCC
0.9985389113/0.9980036616. Repeated outputs and states are exact, with state
isolation across requests. One capture and 12 replays acknowledge exactly two
outputs and perform zero capture-output readbacks before the first replay.
Runtime/allocation audits pass; the 3700-line Watcher log has zero scan hits.
This short allocation profile does not establish maximum-length live-trace
prefill or maximum tuple capacity.

The separate [layer 0 short contract](linear_final_contract_v4.json) passes fresh
lengths 1/31/32/33/65, resumed windows after positions 1/31/32/33 and changed-input
traced decode. Minimum fresh whole/contribution PCC is 0.9996449947/0.9996082187;
resumed/decode contribution minima are 0.9999601245/0.9999952912. All history-negative,
repeat and isolation checks pass, with 39 audited calls, one capture and 12 replays.
Its 3249-line Watcher log has zero scan hits.

The [PLE short contract](ple_final_contract_v4.json) passes strict literal
whole/contribution gates for fresh lengths 1/31/32/33/65, resumed windows and
changed-input traced decode. Fresh, resumed and decode contribution minima are
0.9962700009, 0.9965371490 and 0.9977497458. Six prefills interleaved with a live
trace preserve both requests' state; repeated output and all four state buffers
are exact. One capture/7 replays, 39 audited calls and allocation tracking pass;
the 4415-line Watcher log has zero scan hits. These inputs are distinct from
the controlled fresh findings in the independent public/batch probes.

Layer 1's current-source [v7 continuation run](public_continuation_layer1_v7.json)
passes strict literal resumed/decode gates (minimum contribution PCC
0.9967340231/0.9969496727), with both history-negative controls and exact state
isolation/repeats. Its independent fresh-prefill results require the explicit
matched control described below; `literal_hf_strict_passed=false`. One capture
and 12 replays acknowledge two outputs with zero pre-replay capture readbacks.
The 4380-line Watcher log has zero scan hits; runtime/allocation audits pass.
The additional [batch 4 Watcher run](ple_final_batch4_controlled_v4.json), with
prompt lengths 1/2/3/1, passes literal whole-output and decode-contribution gates.
Fresh request 2 row 0 contribution remains 0.9949769974; all actual-packed-control
comparisons pass (fresh minimum contribution 0.9992804527). The report retains
`literal_hf_strict_passed=false`, exact repeats and a clean runtime audit. Its
1400-line Watcher log has zero scan hits. This is separate from normal-mode batch 5
capacity evidence and does not establish a maximum-length batch profile.

Layer 3 pre-ordered page/cache-remapping evidence remains historical support only;
its ordered-runtime v7 continuation and v4 short contract now pass literal gates; v4 batch 32 also passes literal gates.

The current-source [GDN batch 32 run](linear_final_batch32_v4.json) covers 63 fresh
rows and 32 traced decode requests. Every literal whole-output row passes; minimum
decode contribution PCC is 0.9999741912. Request 16 retains the matched-backend
fresh-contribution finding below, so `literal_hf_strict_passed=false`. Independent
literal reruns are bit-exact, repeated outputs are exact and runtime audit passes.
This Watcher run has 4002 log lines and zero scan hits.

**Fresh-prefill contribution** discrepancies below are carried for independent review,
not accepted as strict passes:

| Supporting artifact | Literal HF whole-output minimum | Literal fresh contribution failure | Independent matched control | Literal decode contribution minimum |
|---|---:|---:|---|---:|
|`linear_final_batch32_v4.json`|0.9951083660|Request 16 aggregate/row 0: 0.9906083941|Only HF attention residual-down accumulation changed to FP64; reproduces 0.99061060 versus literal HF. TT/control row 0: 0.99998516.|0.9999741912|
|`ple_final_batch4_controlled_v4.json`|0.9971301556|Request 2 row 0: 0.9949769974|Only HF lookup values changed to actual packed-cache values; reproduces 0.9949601293. TT/control row 0: 0.9999611974.|0.9975344539|

The controlled batch reports retain `literal_hf_strict_passed=false`. The GDN control
isolates a BF16 midpoint/backend rounding sensitivity and resulting expert-set change;
see `gr_same_operand_rounding_cpu.json` and `gr_backend_sensitivity_cpu.json`. The PLE
control isolates a **storage approximation**, with actual cache words/scales checked
against the quantizer; see `ple_batch4_quantization_cpu.json`. It keeps checkpoint BF16
experts, whereas TT uses BFP8 experts, so these are not identically quantized models.

Expanded live-trace fresh prefills also expose two PLE contribution failures: 65-token
row 2 gives 0.9949381948 and 129-token row 1 gives 0.9947543740, identically on
repetition (`public_continuation_layer1_v5_diagnostic.json`, strict exit 1). Whole
outputs, literal continuation/decode and state/isolation checks pass. An independent
fixed control uses actual packed lookup plus FP64 accumulation at all six
runtime-precise GR/GDN projection boundaries for every 65/97/129-token case. It
reproduces those failures at the same rows: 0.99484086 and 0.99472564 versus literal HF.
TT/control contribution at the two rows is 0.99995285 and 0.99995047; every TT/control
row across all six readbacks is >= 0.9978749156. See `ple_public_backend_matched.json`
and its archive/provenance. Literal reruns and device repetitions are bit-exact; router
selection is never forced. FP64 is a mathematical backend control, not bitwise emulation
of the runtime FP32 tree. Current-source `public_continuation_layer1_v7.json`
repeats both literal failures and passes the explicit control gates, with
`literal_hf_strict_passed=false` and `fresh_backend_all_matched_passed=true`.
Every literal fresh whole-output row remains >= 0.9960471988. These findings
remain subject to independent stage review and are not strict literal passes.

Controls use independent full HF execution with the same inputs/weights and fresh
caches; the independent literal rerun must be bit-exact. They neither force router
choices nor inject HF intermediates into TT. A fresh failure can be classified only when
reproduced at the same aggregate/row granularity, literal whole outputs pass, and
TT/control gates pass. Explicit PLE control mode additionally requires every TT/control
phase/row to pass. Decode/resumed contribution thresholds are never relaxed. Final
review must assess these classifications; a diagnostic `passed` is not a blanket literal
pass or a lower model-specific PCC threshold.

The **pre-ordered** full QSA run (`qsa_final_device_full_boundary.json`) remains a
failure: all 262144 rows were compared, global PCC was 0.9999747948, but row 105584
was 0.9944236875. Its assertion prevented boundary/tail/decode and prefix export.
Actual device captures localize the failure to indexer projection rounding at a tiny
top-k cutoff: the selected block set changes, while gathered K/V and same-operand
selection agree. A uniform FP64 reference control fixes that row but introduces a
failure at 16748, so it was rejected as a replacement oracle.

The integrated repair instead uses one fixed arithmetic order derived from the pinned
literal CPU BF16 backend: adjacent products update FP32 high member then low member;
K-block partials are added in order, followed by explicit BF16 round-to-nearest-even.
The shape policy is 512-wide K blocks for Q1 and 1024-wide blocks for Q2..32, without
row/column exceptions or forced selections. The independent full CPU projection proof
is bit-exact across 262144 x 640 outputs (`qsa_blocked_projection_full_cpu_v1.json`).
[Device operator v3](ordered_projection_device_v3.json.xz) is exact for all Q1..32 at ten
real starts and 352 traced replays, including changed/repeated inputs and live eager
interleaving; audits pass and prepared program count stays 33. The
[documentation-only closure](ordered_projection_v3_documentation_closure.json.xz)
binds the validated operator to its current documentation revision. These are
component proofs, not a full decoder pass; see [backend source evidence](onednn_projection_order.md)
and [Q1 order evidence](onednn_q1_projection_order.md).

The [repaired 105568 window](qsa_window_105568_ordered_v1.json) raises row 105584 PCC
to 0.9999760135. It deliberately injects an exact HF prefix and is **diagnostic only**.
The repaired [from-empty full run](qsa_ordered_device_full_boundary_v1.json) now passes
all 262144 literal-reference attention rows: minimum PCC 0.9982089996, zero rows below
0.995. The actual saved row 105584 reaches PCC 0.9999759609 in an independent
[CPU readback check](qsa_ordered_repaired_row_cpu_v1.json), which binds the output,
HF fixture and TT prefix export hashes. No reference or threshold was substituted.
The device checkpoint at position 262112 supports both the complete 32-row ending
and a restored 31-row ending at prefix 262143; tail minimum PCC is 0.9999139905 and
traced final attention decode is 0.9999723434. Runtime fallback audits are clean.

The [normal full public bridge](qsa_ordered_full_decoder_v1.json) reuses that actual
TT version 1 prefix, with current runtime/fixture hashes verified. Complete layer 3
traced final decode passes two changed inputs in A/B/A/B order: minimum whole PCC
0.9999422431 and contribution PCC 0.9999366403, one capture/four replays, exact repeated
outputs and all five state buffers, clean runtime audit and allocation tracking.
The first CPU preflight failed on a metadata key before opening hardware; repaired
[preflight v2](qsa_ordered_full_decoder_preflight_v2.json) passes and preserves the
failed v1 record. Historical recovery-origin results remain diagnostic history;
they are not the source of this normal-prefix pass. These long-prefix checks omit
full decoder/MoE computation on prefix rows and do not imply whole-stage acceptance.

## Final warmed performance

These measurements use the final formatted runtime on device 0, with hardware jobs
serialized. These fresh clean-timing processes explicitly disable trace allocation
tracking, traceback diagnostics and runtime-audit wrappers; Watcher is absent. Separate
tracked correctness runs retain allocation and runtime-fallback audits. Eager/replay
output and state equality checks run outside the measured interval.
All six captures and report conversions close successfully. The offline validator
[`performance_evidence_offline_v4.json`](performance_evidence_offline_v4.json)
reparses native endpoints, regenerates the CSV/table/latency values, and verifies the
capture/conversion source and log identities.

| Complete layer | Warmed prefill, 32 tokens | Warmed traced decode, one token after a 32-token prefix | Final report root |
|---|---:|---:|---|
|0: GDN|424.137609 ms|47.199169 ms|`tracy/layer0_{prefill,decode}_device_trace_v4/`|
|1: GDN + PLE|455.555837 ms|48.118907 ms|`tracy/layer1_{prefill,decode}_device_trace_v4/`|
|3: indexed QSA|969.702974 ms|249.930357 ms|`tracy/layer3_{prefill,decode}_device_trace_v4/`|

Each `report/` directory contains a human-readable `{mode}_perf_report.md` view, the
exact original `{mode}_perf_report.txt.xz`, and filtered
`{mode}_perf_report.csv.xz`, `whole_trace_ops.csv.xz`, `raw_device.csv.xz`,
`latency_summary.json` and conversion `provenance.json`. Readable Markdown views strip
trailing display whitespace only; exact official table bytes remain hash-bound in the
archive and are restored for byte-level validators. Parent `profile.json` records
shapes, precision, trace identity, actual protocol events, worker inventory and exact
warm/measured replay equality against the eager output and state. Executed capture/report
commands and process/log/source identities are in
`provenance/layer{0,1,3}_{prefill,decode}_device_trace_v4{,_report}.json`;
`plans/qwen_perf_v4_commands.json` is the command plan, not execution evidence.

The metric is **one warmed whole-trace device interval**, including internal gaps,
from native TRACE-KERNEL timestamps. The official tt-perf-report `Device Time [μs]`
column is divided by 1000. It is not a per-operation kernel sum, throughput benchmark,
or latency distribution. Actual pre-drain markers prove both replay ordinals on all
110 independently inventoried worker cores. The native parser's header lists a
120-core architectural maximum; the measured row records 110 participating cores.
No roofline or utilization claim uses the architectural maximum.

The ordinary host Tracy path overflowed its source-location capacity; the failed
capture is retained. Firmware19.8.1 does not support the attempted streaming profile,
and that capture is rejected. The native whole-trace fallback records actual harness
markers and device timestamps, without inventing host Tracy events or per-op rows.
Batch is1; input/output is BF16. Layers0/3 use BF16 experts, PLE uses BFP8 experts and
the full resident packed table. QSA has the full262144 capacity allocated. Watcher runs
are separate. Historical v2/v3 timings remain preserved and are not the final clean-regime results.
[Round1 review](stage_review_round1.md) identified the former tracking-enabled
measurement regime; [round2](stage_review_round2.md) accepts the v4 correction.
Native profiler trace-ID bookkeeping remains enabled to identify measured replays;
it is distinct from the disabled allocation safety tracker.

## Source, audit and CI closure

The exact four-file clang-format19.1.4 transition is recorded in
`plans/format19_transition_v1.json`, with before/after snapshots, formatter/config
identities and explicit `format19_binding_v1.json`. Whitespace changes and braces
around existing single-statement bodies preserve predicates and operation order.
This is a narrow source-semantic proof, not identical compiled-binary or timing proof.
The validator rejects any unlisted edit. Fresh `chunk_input_final_v6` and
`chunk_output_final_v5` reruns close exact current-source copy/canary evidence without
a source exception. Original run hashes are never rewritten.
Final performance uses newly compiled formatted source.

Post-format CPU summaries are `final_capacity_qsa_closure_v4.json`,
`final_batch_capacity_qsa_closure_v4.json`, `final_watcher_summary_format_v2.json`,
`synthetic_final_decoders_v4_format_summary_v2.json`, and
`linear_qsa_gate_evaluations_v3.json`. The last retains all12 declared linear gates and
composes the exact formatting proof with the unreachable QSA-constructor proof.
These mechanical results require the independent semantic review described below.

The Watcher summary verifies successful separate `TT_METAL_WATCHER=10` runs for all
three kinds, complete worker dumps and detach, no profiler overlap, and zero failure
scan hits. Synthetic CI has six cases with zero failures/errors/skips: four complete
GDN/QSA decoder cases at lengths1/33 plus traced decode, and two PLE arithmetic cases.
Minimum synthetic PCC is0.999617517. This is not a full synthetic resident PLE decoder.
The post-format summary explicitly records raw source identity as false and exact
format-equivalence closure as true; it does not relabel historical source hashes.

Pinned formatting checks cover149 Python and8 native files; full scoped precommit
passes without modifications. `final_cpu_helpers_postformat_v2` records105 passing CPU
tests and84 subtests. The clean-timing adapter adds10 passing CPU tests, including
negative diagnostic-setting checks (`clean_timing_cpu_v1`). Correctness probes forbid Torch operations, `from_torch`, `to_torch`
and TTNN fallback inside each measured forward/capture/replay; setup/input refresh and
reference/readback remain explicit external boundaries. Repeated outputs/state, changed
inputs, independent requests, dropped-history negatives, and QSA coherent page remap
are tested by the public continuation runs.

## Evidence access and acceptance

[Final gate bindings](final_gate_bindings.json) specify artifacts, source dependencies,
numerical predicates and separate semantic review obligations. The artifact record and
its recorded command/log are evidence; a command plan or an agent-written claim is not.
[Context contract](../context_contract.json) distinguishes the262144-token autoregressive
state context from PLE's owner-declared prepared resident-call range1..4768, supported by
physical input bytes, boundary failures and finite source/probe evidence. It does not
claim universal allocator optimality, maximum prefill while a decode trace is resident,
or a Cartesian product of maximum batch and context.

Independent [round1](stage_review_round1.md) and [round2](stage_review_round2.md)
reviews close with **clean-pass** after correcting the timing regime. Local checkpoint
SHAs, exact commands, failures, numerical controls and chronology are retained in
[work_log.md](work_log.md). No push is authorized. Controlled fresh-contribution
findings retain the reviewers' scoped numerical assessment; physical capacity and
source-format arguments do not waive numerical requirements.

Archived evidence can be restored after checkout with the model-local helper:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.restore_evidence_archives \
  --root models/autoports/qwen_qwen3_8_flash_next --verify-only
python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.restore_evidence_archives \
  --root models/autoports/qwen_qwen3_8_flash_next \
  --source doc/functional_decoder/ordered_projection_device_v3.json
```

Omit `--source` to restore every archived original; this can require substantial disk
space. The helper verifies multipart manifest identities, compressed parts, and restored
SHA256/byte counts. It preserves identical existing files and refuses conflicting files,
unsafe paths, and symlink destinations. Originals are restored byte-for-byte; recorded
commands, timestamps, and source provenance are never rewritten. Restore is per file,
not a transaction across all selected entries: a later error can leave earlier verified
files restored. Temporary spools keep memory bounded but require additional disk space.

Existing evidence evaluators use the repository location recorded in provenance
(`/home/ttuser/qwen38-flash/tt-metal`). Restoring into a different checkout does not
relocate those recorded absolute paths or guarantee every evaluator can run there.
The portable claim is exact artifact-byte verification, not an automatically relocated
re-execution environment. Restored files receive new filesystem timestamps; checks
requiring original creation/modification times cannot be rerun from restored bytes
alone. Preserve the already recorded summary/provenance rather than manufacturing
historical timestamps. Large external checkpoint, packed-table and CPU/device fixture
paths remain external dependencies with recorded identities.
