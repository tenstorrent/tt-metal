# Fused decoder work log

Stage 2, Qwen/Qwen3.8-Flash-Next revision812d96e10a0dd82b69d749e68693698cf6c36daa.
Status: in progress; no fused acceptance claimed.

## Startup

Functional stage accepted with independent round2 clean-pass; runtime checkpoint
747f83791c95c757bf4f3d439dfc3e768d4f5c9b. Initial working tree clean.
Enabled plugin inventory: CODEX_HOME/config.toml enables tt-model-bringup and
tt-autodebug. Installed roots0.1.23 and0.1.8; environment.py exports validated.
Hardware: four p300c Blackhole devices visible; fuser finds no device holders.
This stage uses only chip0, serial, with the existing p150 single-chip descriptor.
No reset authorized or performed. No push. Scope: fused_decoder.py, tests/docs only.

Functional bars and context scopes remain unchanged, including documented numerical
controls and resident PLE physical limits. Preserve all precise projection, recurrence,
router tie-breaking, state/cache and bounded I/O implementations initially.
Baseline clean trace intervals (ms) from accepted v4: layer0 prefill32=424.137609,
decode=47.199169; layer1=455.555837/48.118907; layer3=969.702974/249.930357.
Fresh same-harness baseline/candidate runs remain required.

## Initial topology audit

|Subgraph|Inputs/output and movement|Candidate|
|---|---|---|
|Gated residual read, twice|BF16 streams reshape/permute; FP32 square/mean/add/rsqrt/mul/weight; precise down/up; BF16 nonlinear boundaries|Dedicated RMSNorm, preserve per-stream weights and routing|
|GDN|Precise packed QKV; four-tap FP32 convolution; q/k norm; FP32 recurrent custom op; norm/gate; precise out|Dedicated norm, binary activation, shared-LHS z/a/b packing|
|PLE|Resident packed embedding, hash, convolution; GDN follows|Preserve resident storage and lifetimes; inspect arithmetic fusion|
|QSA|Q/gate,K,V projections; indexed selection, paged updates and SDPA; sigmoid gate; permute/reshape|Head creation/concat, packed projections, fused gate multiply|
|MoE router|Precise router, ATen-compatible top10, stable softmax, scatter; tile→row-major active union|TopK already dedicated custom; sparse mask RM conversion required by sparse_matmul|
|MoE experts|Two same-LHS sparse matmuls, SiLU then multiply, sparse down, weighted sum|Packed gate/up; input activation into multiply; preserve sparse active-expert dispatch|
|Shared expert|Gate/up projections, SiLU multiply, down, sigmoid route multiply|Packed projections; binary activation merging|
|Residual write|FP32 multiply with BF16 boundary, permute/reshape, FP32 add with BF16 output|Preserve intermediate BF16 rounding; inspect structural views|

Candidate1 implements only MoE activation merging, separately selectable expert/shared.
No candidate has been measured or accepted yet.

## Candidate measurements (initial, not final sign-off)

Exact bounded argv/environment/source hashes live in each `*_provenance.json`;
stdout/stderr and exit markers are under `logs/`. Tracking runs include program-cache
allocations. Clean timing runs disable allocation tracking/diagnostics and runtime
audit; these are host-observed synchronized trace intervals, not per-op device times.
The benchmark resets request state outside timing and alternates two decode inputs.
No trace is live across candidate changes. The only acknowledged allocation is the
capture output, overwritten in full by its sole trace and read before subsequent work.

- `norm_v1.json`: native FP32 RMSNorm followed by per-stream FP32 weight passes
  component PCC at logical1/31/32, minima0.99999839, with lower component trace time.
- `candidates_layer0_v1.json`: the native norm fails integrated prefill33: whole
  minimum0.99401647, contribution0.99162400 (row8). Activation merging alone is exact;
  every candidate has exact eager/replay output and state. Native norm is not accepted.
- `activation_layer0_clean_v1.json`:20 alternating replays/candidate, length32.
  Baseline46.793518ms; expert activation46.649960; shared activation46.792484;
  both46.646244. All functional equivalence checks pass exactly. Gain is small.
- `candidates_layer3_v1.json`: tracked prefill33 and8 changed-input replays.
  Baseline250.117279ms; activations249.956833; plus dedicated concat249.939529;
  plus sigmoid gate249.932648; both QSA rewrites249.928131. All outputs/state checks
  pass; clean timing and HF checks are still required before acceptance.
- `packing_layer0_clean_v1.json`: same real weights, length32,20 replays. Baseline
  46.780205ms; activation46.641506; packed shared gate/up/route46.595108; packed
  routed experts46.649494; both packing46.591492. Routed packing has no clear decode
  win and worsens measured eager prefill; do not infer benefit from fewer matmuls.
- `reduce_layer0_v1.json`: fused score-weighted DeepSeek reduction adapted to the
  sparse_matmul512-expert DRAM output, identity expert mapping, per-token dense
  scores. Baseline46.893963ms; packed shared46.719418; fused reduction45.504047.
  Prefill33 contribution PCC0.99999839; changed/repeated traced outputs and states
  match eager exactly. This is a promising candidate, not final acceptance.

## Norm failure investigation

AutoDebug uses the installed launcher, in this stage document directory. The first
launch failed before a model because unprivileged namespaces are unavailable. The
explicit authorized sandbox skip was then used. The home-directory CLI could not
spawn its missing codex-code-mode-host; no investigation or hardware action occurred.
Retry uses `/usr/local/bin` first in PATH, whose installed CLI includes that binary.
Logs: `autodebug_norm_launcher.log`, `autodebug_norm_launcher_retry.log`, and
`autodebug_norm_launcher_systemcli.log`. No host configuration was changed.
A lower-risk candidate preserves the original reduction and merges only epsilon-add
and rsqrt; its integrated test is in progress. Native normalization is never silently
accepted on its component PCC alone.

## Additional topology candidates and cumulative checks

`gdn_packing_v1`: z/a/b packing has exact full-decoder equivalence at33; tracked
baseline46.968445ms, packed46.839821ms. `qsa_packing_v1`: packed Q/gate/K/V and
adapted dedicated head splitting both pass exact full-decoder equivalence at33;
tracked250.068634/250.025513/250.061432ms for baseline/packed/split. The dedicated
split interprets48 query-sized heads then restores the actual24 query+gate pairs;
no gate rows become attention queries.

`qsa_rope_v1` fails the partial-tail concatenation because the native op exposes
padded32 rows. `qsa_rope_v2` slices the native output back to logical count and
supplies the original compute config: ordinary rotary passes0.99882472 contribution
PCC, but the decode-layout variant still exposes padded heads. Both original errors
are retained; no first API failure is used as a rejection. The decode variant now
slices both logical head and token extents; rerun pending.

`norm_diagnosis_v1` reproduces native norm's failure on current source and observes
actual changed expert membership at the failing row. The exact smaller partial
fusion has no changed IDs. `norm_rhs_matmul_v1` tests the second, RHS-activation
partial norm: exact output/state checks pass; tracked baseline46.965122ms versus
46.942287ms. Simultaneous sparse/dense matmul activation folding fails contribution
PCC0.95091987; the sparse factory does not forward its program's activation into
its kernel. A target-shape raw-output diagnostic is prepared. The independent dense
shared-expert-only trial passes exact equivalence (`gdn_conv_shared_activation_v2`).

Dedicated `qkv_causal_conv1d_silu` is adapted to physical32 rows internally, then
sliced to the original logical count. It consumes caller history with a fixed local
start0; original state-copy uses only actual logical rows. Initial setup used the
wrong config namespace; v2 uses `ttnn.QkvCausalConv1dSiluProgramConfig`. The complete
layer fails contribution PCC0.99454182 under FP32 accumulation. Its fused SiLU omits
the original BF16 convolution-output boundary. A separate supported non-FP32-dest
accumulation control is pending; native precise projection policy stays unchanged.

`ple_candidates_v1` uses the full resident table and real BFP8 experts, loading only
non-table weights on the CPU. PLE sigmoid/multiply and SiLU/add fusions, with or
without the exact partial norm, pass exact decoder equivalence at33 but have no
tracked speed win (baseline47.837120ms, activation47.849503, norm47.896095). Fused
weighted reduction passes0.99999940 contribution PCC and46.842220ms. Clean timing
will determine whether any PLE-only partial fusion is retained.

The cumulative `best_v1` policy is now explicitly constructed and passes all three
real-HF short harnesses. Layer0 uses the original linear-contract sweep1/31/32/33/65,
continuation negative controls, independent requests, repeated changed-input traces
and interleaved prepared prefill. Its minimum whole/contribution PCC is
0.999645174/0.999608517. Layer1 passes the full resident PLE33 +3decode fixture.
Layer3 passes1/31/32/33/65 with full262144 KV capacity. Exact selection sidecars prove
the factories returned FusedDecoder and FusedSparseMoe; they do not silently select
the functional class. Final policy/performance/context/Watcher/review gates remain.

Beginning with `gdn_conv_shared_activation_v1`, relevant candidate runtime and test
sources are copied before launch under `sources/<run>/`. Earlier candidate hashes
and source history are retained in provenance but lacked full source bodies; final
acceptance will use source-archived current-path measurements, not those historical
numbers alone.

## Accurate reduction discovery and cumulative v2

The current checkout's generic reduction has an accurate FP32 SFPU path:
`reduction/generic/device/reduce_op.cpp` selects it when
`fast_and_approximate_mode=False`, FP32 input, FP32 destination accumulation,
and a supported architecture. The original precise projections explicitly expanded
balanced additions into thousands of slices/adds. `FusedPreciseProjection` retains
weight storage, FP32 products and final BF16 conversion, replacing only the tree
with dedicated accurate `ttnn.sum`. This changes summation order slightly; it does
not route the products through the TF32 FPU reduction.

`sfpu_reduction_layer0_v1` exposed a Python integration error: the GDN `linear`
helper recognizes `PreciseGDNProjection` by class. The fused adapter now derives
from that interface and overrides the complete runtime call; no functional tree
fallback executes. `sfpu_reduction_layer0_v2` passes all independent and combined
rewrites at33, including repeated changed-input traced output/state equality:

|Projection reduction|Decode ms (tracked)|Captured ops|Min contribution PCC vs functional|
|---|---:|---:|---:|
|Functional|46.930805|8514|1|
|Residual down/up|30.936201|4214|0.999997497|
|Router|44.375374|7730|1|
|GDN QKV/output|36.660282|6004|0.999996722|
|All above|18.170811|920|0.999996722|

These PCC values are not claims of bitwise equality with functional; exact replay
flags compare each candidate's trace to its own eager computation. `best_v2` adds
the combined reductions to `best_v1`, and all three real-HF short harnesses pass
(`best_v2_layer{0,1,3}_hf.json` plus selection/source/provenance/log artifacts).
Long-context, larger-batch, final clean timing and Watcher remain pending.

`sparse_activation_contract_v1` tests actual target gate weights at1/32 rows:
setting sparse matmul's SiLU field returns exactly the unactivated output, with
max error0.0256348/0.0329590 against standalone SiLU. This isolates the reason
for the combined matmul-activation failure. Binary-input SiLU remains valid.

`gdn_layout_conv_bf16_v1`: metadata-only recurrence reshapes pass with no measured
speed gain (46.893951ms versus packed46.875088ms); the BF16-destination native
conv+SiLU candidate passes the short functional comparison at0.999080181
contribution PCC and46.810081ms. Its full-HF and cumulative checks are pending.

## Remaining QSA pattern adaptations

`qsa_remaining_patterns_v1` completes the shape/layout adaptations rather than
rejecting on the initial API errors. All candidates pass repeated eager/replay
output and state equality at33, including the one-row final chunk.

|Candidate|Tracked decode ms|Min prefill contribution PCC vs functional|
|---|---:|---:|
|Functional|250.020432|1|
|Four-window concat + accurate mean|250.052811|1|
|Decode QKV heads, setup query/gate reorder + L1/DRAM adaptation|250.076691|1|
|Decode concat heads, height-shard32x256 + return to DRAM|250.066071|1|
|Native main Q/K rotary + logical extent slicing|249.932938|0.998824716|
|Native rotary token_index0 + decode layout adaptation|250.022614|0.998824716|

The dedicated decode head operations are correct, but their required layout
adaptations lose the speed comparison. Ordinary native rotary remains under
cumulative/HF evaluation; the exact indexer rotary path was never replaced.

`combined_layer{0,1,3}_clean_v2` uses explicit constructor policies on shallow
copies of the real baseline setup objects. Components are separately copied before
policy application; immutable weights are shared. No trace survives a variant
switch. This avoids doubling resident checkpoint storage while exercising the
actual FusedDecoder factory. Three warmed eager prefill samples and20 changed-input
trace replays are recorded; a final functional baseline checks drift. Allocation
tracking, traceback diagnostics and runtime monkeypatch auditing are disabled for
these clean timing runs; tracked correctness remains separate.

## Projection peer merging and native convolution controls

`projection_groups_layer0_v1` combines adjacent immutable output-weight blocks at
setup, retaining the original bounded prefill product blocks. Only one-token decode
uses the combined parent. Actual FP32 reduction and BF16 output boundaries remain.
Groups1/2/4/8 pass short full-path equivalence and traced determinism, with tracked
16.6071/14.5106/13.4651/12.9589ms. `projection_groups_layer0_clean_v2` measures
8/16/32/64 in clean timing:12.8615/12.4636/12.4125/12.3151ms; functional controls
46.7948/46.7852ms. All pass. Group64 combines each real matrix completely;
prefill still slices32-output GDN/router/down blocks or256-output residual-up blocks.
No duplicate matrix is retained by the constructed final component. The diagnostic
same-process harness deliberately keeps baseline objects for comparison; standalone
factory acceptance must establish actual resident capacity.

`best_group8_layer0_hf` and `best_group8_conv_layer0_hf` pass the original HF linear
contract. `best_rope_v2_layer3_hf` passes the original QSA HF short sweep; native
rotary's numerical delta is real, not bitwise equivalence. `best_group64_conv_layer1_hf`
passes with the full resident PLE table. Both `best_group64_conv_continuation0/1`
pass live request interleaving, nonaligned continuation, changed/repeated trace
execution, runtime fallback audit and allocation tracking. Layer1 keeps the
previous stage's explicit matched fresh-prefill backend controls and strict
whole/resume/decode bars; those controls are not relabeled literal HF equality.

The dedicated convolution's source builds diagonal tap matrices, accumulates four
matmuls in destination registers, then applies SiLU on the pack thread before
packing. Unlike the original graph, there is no explicit intervening BF16
convolution-output tensor. The supported `fp32_dest_acc_en=False` control passes
where `True` failed. Earlier shorthand “BF16 destination” names that configuration;
it is not a claim that every internal hardware accumulator is16-bit. All other
projection/recurrent compute precision is unchanged.

`group64_variants_layer0_clean_v1` passes all candidates: functional46.7698ms,
group64 12.2997ms, plus native convolution12.1554ms, metadata-only recurrence
reshape12.2756ms, both12.1164ms; final functional control46.7776ms. The combined
path also improves warmed prefill32 to152.98ms from422.43ms. Wider checks of the
combined path and final-source measurements remain required.

## Final-v1 gate failure; final-v2 convolution repair

The selected native-convolution default passed all real-HF short tests, but the
new default-only synthetic matrix found GDN length1 decode step1 below threshold.
The17-job long gate plan `final_v1_gate_plan.json` was prepared but never launched;
it is stale after the repair. Exact owner controls and a fresh installed AutoDebug
session isolate native convolution, not projection sums/grouping or recurrence
layout. See `autofix_ci/AUTODEBUG.md` and the updated `AUTOFIX.md` for the numerical
contract and source-supported repair. Native SiLU cannot silently stand in for the
original BF16 boundary followed by FP32 SiLU.

An accurate tap-axis reduction alternative is correct but slower. The model-local
SFPU specialization preserves both BF16 boundaries and the original pairwise order,
uses the existing compute primitives in one generic program, and is bitwise equal
and faster at component scale. The global native op is unchanged. Final-v2 gates
must use the actual new default and frozen source; old final-v1 results are not
relabelled acceptance of the repair.

## Final-v2 batch gate: QKV reduction order

The frozen final-v2 queue passed reduced context65 and public continuation for
layers0/1/3, then stopped on `final_v2_batch0_32`: request3 decode whole PCC
0.99280417 and contribution0.98352855. Determinism remains true. The ordinary
functional baseline passes the identical32-request fixture, request3 contribution
0.99997801. No accuracy gate or backend-control rule is changed.

`probe_fused_batch_control.py` substitutes original projection objects only at
setup; every other fused component and the unchanged batch probe remain active.
Restoring both GDN projection trees passes, as does restoring QKV alone
(request3 contribution0.99997956). Restoring only the GDN output tree still fails.
Exact command/source/log provenance is in `batch0_control_*_provenance.json`;
compact comparisons are in `autofix_batch/controls.json`. The installed AutoDebug
launcher is independently inspecting this failure, without running the device.
The failed final-v2 candidate is not eligible for final performance selection.

## Exact indexer: remove padded-row broadcast and SFPU work

Source inspection of `ordered_projection_reader.cpp` found that each of2560 K
steps expanded both operands across all32 physical output rows, even for a one-token
decode. A model-local reader initializes both ping-pong slots once, then materializes
only the valid logical rows. Weight order, high/low BF16 pair order, K512/K1024
partial boundaries and FP32 arithmetic are unchanged. The original compute/writer
are reused. This is a structural removal of padded broadcast work, not a precision
change or a host projection fallback.

`ordered_valid_rows_component_v1` passes strict bit-exact HF/ordered references at
counts1/2/31/32, with changed/repeated trace inputs. Complete QSA clean comparison
`ordered_valid_rows_layer3_clean_v1` passes: functional249.5997ms,
previous fused226.9833ms, valid-row reader169.5910ms. All20 replays match their eager
output/state exactly. An additional candidate limits SFPU tile math to the top
faces for logical counts<=16 using the established row-vector idiom in
`models/demos/gpt_oss/tt/fused_decode/kernels/stream_qkv_rope_compute.cpp`.
Counts>16 retain full-tile math; every logical count1..32 is being checked without
changing addition order. The candidate remains opt-in until correctness and
complete-decoder timing gates pass.

## QSA score and selection fusions; final-v4 freeze

The exact indexer valid-row reader is retained. Limiting SFPU face work and moving
copy initialization out of the K loop both pass exact component tests but do not
produce a material integrated win (`ordered_valid_compute_layer3_clean_v1`,
`ordered_copy_once_layer3_clean_v1`); neither is enabled by default.

The score graph contains four peer head graphs, each doing eight key blocks and
an explicit128-wide FP32 product/addition tree. Independent accurate-SFPU-sum,
merged-head/tree, and combined candidates all pass the original full-capacity
score gate with changed/repeated traced inputs. Combined scores preserve all
selected512 sets across counts1/3/7/31/32 in all four sensitive captured windows
205504,165856,215424,222272; maximum observed error is of order2e-6, below the
unchanged1e-4 component limit. Source and command provenance accompany
`indexer_score_{sum,merge,both}_component_v1` and the three extra window records.
`indexer_score_layer3_clean_v1` shows functional249.6898ms, valid-row reader169.6874,
accurate sum152.1868, merged exact tree165.5508, both150.4782. These are complete
public decoder traced medians, not only score timings.

The existing large-row dedicated TopK requires BF16 inputs. FP32 QSA scores
cannot use it directly. `topk_bf16_score_adapter_cpu.json` records that rounding
real scores to BF16 changes124/128 top512 sets across the same four windows;
that is a CPU information-loss control, not a device-op timing. The native FP32
K512 path is single-core (multi-core eligibility K<=64). The model-local fused
selector preserves exact nonnegative FP32 ordering using integer bit comparisons,
selects512 with a bounded device heap, and folds causal masking, four-token index
expansion, sentinels and the incomplete-group tail into the same device program.
Device-valued positions bound reads; there is no host position/score readback.
It defines ascending key-index ties within the original non-stable TopK contract.
The first shape-binding smoke failed before kernel launch (`shape[-2:]` is not
supported); scalar indexing repaired it. No device reset occurred.

`qsa_selector_component_v2` passes all28 exact cases at seven logical counts,
including dynamic traced bounds, invalid NaN tails, FP32-ULP differences,
deterministic ties and repeated inputs. Complete decoder
`qsa_selector_layer3_clean_v1`: functional249.5891ms, previous correct combined
score candidate150.4485ms, fused selector15.2347ms. All replay outputs/states match
eager exactly. `selector_acceptance_plan_v1` then passes real-HF lengths1/31/32/33/65,
normal batch32, public continuation, reduced context65, and Watcher component checks.
Its Watcher log has a completed dump and a clean detach; full-decoder Watcher remains
in the final queue. No full-context acceptance is inferred from these short checks.

The shared `FusedChunkedQSA.from_component` setup now serves both the full decoder
and the reduced-attention harness, preventing a prefix harness from omitting a
selected score/projection/selector fusion. Delivered tests assert actual fused
QSA, indexer, selector and MoE classes/flags. Old experiment policies explicitly
freeze all newly changed defaults so historical variants do not silently adopt
later fusions.

Final-v4 default runtime is frozen at SHA256
`42dfe986a782ad369fc11515aaff115068345a6e6f768925a0012a97106293cb`.
`final_v4_gate_plan.json` queues46 serial, bounded jobs: default-only synthetic and
real-HF gates; nonaligned/context/continuation/batch/PLE-capacity checks; separate
Watcher runs; twelve clean native trace captures plus twelve official reports;
and fresh advertised-context GDN/PLE/QSA bridges. Source snapshots are in
`sources/final_v4_frozen/` and per-job archives. Completion is still pending.
The optimization stop criterion and independent stage review are not yet satisfied.

### Final-v4 capacity gate failure — 2026-11-13 10:38 UTC

Frozen42dfe986 runtime passed synthetic8/8, short real-HF0/1/3, reduced context65, continuation0/1/3, batch0/3=32 and PLEbatch5. Prepared public PLE4768 failed at GR precise_down product allocation:41943040 bytes total,5242880 required perbank,5178752 largest free block. The serial gate queue stopped before4767, Watcher, native profiles and fullcontext. No reset occurred; process closed device. Advertised capacity is unchanged and this is an unresolved regression, not an accepted physical-limit reduction. Isolated AutoDebug source investigation launched in autofix_capacity; accurate-reduction product lifetime and grouped-weight fragmentation are hypotheses. Exact logs/provenance and frozen sources remain preserved.

### Capacity controls after AutoDebug

Matched functional4768 current-environment control passed all gates. Redundant same-FP32-cast removal plus release of consumed products did not suffice: capacity_repair_v1_ple_4768 still fails40MiB allocation, largest5191040 versus5242880 bytes/bank. This corroborates fragmentation as a remaining constraint; no capability reduction is accepted. Next control retains original GR weight blocks via residual_decode_group=1, while keeping accurate SFPU reductions and other final choices. AutoDebug completed successfully and reviewed its own claims; report autofix_capacity/AUTODEBUG.md.

### Compact expert graph prototype

Source inspection found indexed/gather sparse_matmul already supports compact active group output, including compact activation input for the down projection. Prototype sorts the existing selected10 IDs ascending on device (preserving original expert accumulation order), packs their already-BF16 scores, uses indexed gate/up/down, and reduces10 groups. V1 failed host validation because its required-but-unused sparsity operand had10 entries rather than512; no sparse kernel was launched. The corrected v2 uses a resident512-entry dummy mask while indices select10 actual groups. Runtime source remains unchanged by this prototype; tests/probe_fused_sparse_compact.py owns the experiment, and its v1 source is preserved in sources/probe_fused_sparse_compact_v1.py.txt.

Compact v2 component passed8/8 exact output cases, four each real layer0 BF16 and layer1 BFP8 weights; repeated trace output exact. Expanded512 MoE~7.62ms, compact10~5.85ms. This is component evidence, not full-path acceptance or clean final ranking. Revisit gate/up shared-LHS packing in the new compact graph before adoption.

### Capacity and next graph controls

All original precise projection blocks (projection_decode_group=1) restore prepared public4768 with all gates unchanged;4767 pending in the same plan. Shared-LHS gate/up packing with compact selected experts is exact but slower (~6.02ms vs5.85ms compact-unpacked); reject. New recurrence prototype folds head replication, padded per-token layout, scalar expansion and output packing into its reader/writer. Its SFPU recurrence compute is unchanged. It reads compact q/k/v and scalars directly, broadcasts each needed row into existing L1 circular buffers, and writes compact token-major output. This targets the current prefill32 recurrence output25MiB and q/k/v25MiB each, avoiding their DRAM expansion; it may restore grouped-weight capacity but needs exact component/integrated/Watcher/perf evidence. Test source probe_fused_gdn_compact.py; no production adoption yet.

### Integrated compact candidate freeze, 2026-11-13 11:07 UTC

Runtime SHA25658f560c36147210f3e42483a1c9d198a744dedfcf1db021bc1c3b2cb0e6fd970 adds explicit compact_experts and fuse_gdn_io options, both still defaultFalse pending full ranking. best_compact_both selects both. Production code now owns compact routes and recurrence I/O; component tests call those actual integrated helpers/paths. gdn_compact_component_v1 prototype was bitwise exact for output/final-state at1/2/7/16/17/31/32; traced1 .143406->.126092ms,32 2.446399->1.793432ms. Frozen integrated component reruns and synthetic8/8 passed. Serial16-job compact_integrated_plan_v1 continues realHF0/1/3, PLE4768/4767 with grouped weights, clean comparison0/3, separate L1 comparisons to avoid simultaneous extra grouped matrices, then separate Watcher0/1/3. Exact selected variants, source snapshots and environments are recorded by run_fused_plan/run_recorded.

Integrated compact source58f560c passed components, synthetic8/8 and short real-HF0/1/3. Group64 PLE4768 advanced through GR and recurrence but failed expanded-prefill MoE down80MiB allocation:10,485,760 required perbank vs9,297,920 largest. Thus compact recurrence removes the former obstruction but group64 capacity remains unaccepted. Next PLE control combines both new fusions with residual_decode_group=1. No advertised capacity reduction. Deferred ranking/Watcher move to compact_capacity_ranking_v2.json, preserving the failed queue.

### Direct packed setup and projection widening candidates

Residual-group1 with compact recurrence/MoE still fails4768 at the same80MiB prefill down allocation (largest9,297,920 vs10,485,760 bytes/bank). Retaining all original blocks remains proven capacity-preserving, but grouped allocations are still fragmented. Projection mixed-unpack component is exact for QKV/output at1/2/7/31/32; traceddecode QKV3.91283->3.36663ms and output.894592->.558832ms, prefill32 QKV165.376->159.799ms, output24.5014->21.3003ms.

New direct_packed_weights setup option takes ownership of each replaced projector, releases its original blocks, and uploads grouped weights directly from the same checkpoint tensor at the same dtype. This avoids concatenation relocating135MiB of live precise weights. No setup host conversions occur in runtime. The comparison harness copies projector objects so this ownership transfer cannot mutate its retained functional reference. A new integrated component check compares direct-upload outputs as well as unpack widening. Runtime 3dc6c4e64005ca22a489691e9b5b8a31948dc6a842aa3b4b801fa7d00805d25c frozen for direct_unpack_plan_v1 (15 jobs): components, synthetic/HF, setup-only capacity control, combined capacity, clean ranking, Watcher. No new default adopted yet.

Direct setup-only4768 and direct+unpack4768/4767 all passed unchanged gates. Clean ranking stopped before timing with a benchmark harness UnboundLocalError: checkpoint weights had been deleted before the new direct-setup call. Fix retains only the small projection checkpoint subset as setup_weights; the functional reference objects remain copied/preserved. No failed timing rows are ranked. Source frozen_v1 retains the failing harness. Corrected ranking gets v2 tags; Watcher had not launched and retains its unused tags.

Memory snapshots qualify the setup finding: direct_setup before preparation has514,214,400 free bytes and44,140,928 largest/bank, versus final-v4 concat514,140,672 and44,131,712. The largest initial hole is nearly unchanged, so direct loading does not fully defragment DRAM. After preparation direct setup has374,681,088 free and30,146,944 largest/bank (v4:362,415,616 /29,590,912). Its changed allocation/program-cache history and removed device concats restore actual capacity; no exact address-specific causal attribution is claimed.

## Packed expert width adaptation and direct unpack ranking
The prior queue `packed_width_direct_ranking_v2.json` completed all 13 jobs, including Watcher on all three layer kinds. Direct/unpack traced decode: layer0 11.708 ms, layer1 13.078 ms, layer3 13.174 ms; bracketed functional controls 46.74/47.67/249.61 ms. All acceptance checks passed.
Packed compact expert gate/up on the original 5-core grid was slower than separate projections. Matching the doubled output width with 10/20/40 column cores preserved bitwise outputs; 40 cores reached 2.296 ms BF16 and 2.292 ms BFP8 (component). This changes only the grid for the fused width; K blocking, fidelity, and down projection remain unchanged. Runtime direct setup now releases original expert matrices before uploading the same tile-aligned packed checkpoint tensor.
`ttnn.swiglu` is implemented in unary_composite_op.cpp as split, swish(second), multiply(first): it is not a dedicated fused kernel, has reversed gate/up ordering, and adds an activation launch compared with the retained binary activation. Source assessment rejects this API replacement.
Single-policy clean timing explicitly records same-candidate eager/state reference; it does not claim a functional/HF comparison. Separate real HF and capacity probes retain the original acceptance gates.

Packed40 short synthetic8/8 and HF0/1/3 pass; public4768 fails MoE down80MiB contiguous allocation (10,485,760B/bank required,10,031,488B largest,19,385,216B aggregate free). It is not accepted. Packed-activation reader prototypev1/v2 found a test-only tile-count bug: Tensor.volume includes padded dimensions, so dividing by logical height overcounted tail tiles and corrupted buffers. Both runs closed normally; no reset. Corrected to explicit logical prefix dimensions times ceil(height/32) times width/32; rerun under Watcher before integration. No runtime uses this prototype.

## Packed activation reader integration
`packed_silu_component_v6_watcher.json` proves exact native SFPU SiLU/BF16/multiply behavior for1/10/512groups and1/7/31/32rows. The initial FPU multiply candidate differed by up to0.5 on broad random3-sigma input; using the same native accurate SFPU multiply and BF16 unpack-to-FP32 mode repairs that delta exactly. Descriptor setup is64CB slots on this checkout. Integrated rank6 component `packed_silu_integrated_component_v2.json` passes.
The packed activation reader improves but does not fix PLE4768 allocation history:10,160,512B/bank largest versus10,485,760 needed,19,514,240 free. New independent inspection is in `autofix_packed_capacity/`; no reset.
`packed_silu_v3_layer{0,1,3}_hf.json` all pass. Same-process clean layer0 comparison in `packed_silu_layer0_clean_v3.json`: functional prefill421.718/decode46.7673ms, direct/unpack288.950/11.6824ms, packed+activation248.473/8.12075ms, functional tail424.786/46.7698ms. Packed and direct/unpack have the same measured prefill contribution PCC minimum0.9999974966 versus functional, and repeated trace outputs/states pass. Capacity acceptance remains open for PLE.

## Packed-capacity diagnosis and placement control
Fresh independent `autofix_packed_capacity/AUTODEBUG.md` exactly reconciles the218,567,168-byte live ledger at down allocation. No packed or slice buffer remains there. Leading hypothesis: packed path allocates surviving activation while40MiB temporary is live, unlike direct path's retained first20MiB gate. It also identifies changed old-MLP shared-buffer ownership at setup; current-source `current_direct_control_v1_ple_4768.json` passes, so that revision does not itself break the direct policy.20core and residualgroup32 controls still fail with nearly identical holes; no capacity reduction is accepted.
`packed_late_addresses_plan_v1.json` and `packed_early_plan_v1.json` isolate activation placement and record first-public-chunk addresses through `probe_fused_allocation.py`. This diagnostic does not retain tensor references or rank its timing.
The alternative compact-prefill union prototype (`prefill_compact_union_layer0/1_v1.json`) is bitwise exact for1/2/7/16/17/31/32 and repeated traces under Watcher. It uses10*T fixed slots, ascending unique selected experts, and zero-weight padding. Its larger-row timings under Watcher are worse due to padding work; clean timing and routing-packet CPU comparisons remain untried. It is not integrated. Prefer the minimal placement repair if it passes.

`packed_early_v2_ple_4768.json` passes the full unchanged gate. The isolated address evidence (`packed_late_addresses_v1_addresses.json`, `packed_early_v2_addresses.json`, derived `packed_activation_placement_proof_v1.json`) confirms the surviving activation was the separator: late activation address4,251,381,376 leaves10,160,512B/bank; early activation address4,246,138,496 leaves15,403,392B/bank before down. Both have exactly156,113,920 total free bytes. Down requires10,485,760B/bank and now allocates at4,248,759,936. This fixes capacity through allocation order without extra live storage, copying, reduced context, altered arithmetic, or slower core geometry. Initial early-v1 exited before execution because TTNN Shape does not support Python slice syntax; converting shape metadata to lists fixes it. Neighbor4767 and remaining gates are still required.

### 2026-11-13 12:33 UTC — allocation repair ranking and remaining fusion
The early activation destination preserves 4768/4767 PLE capacity; `packed_early_wide_rank_profile_v1_progress.json` records all correctness/ranking jobs passing. Same-session traced medians: layer0 packed40 8.142249 / packedSiLU 8.134844 / early 8.132350 ms; layer3 9.623250 / 9.614032 / 9.616728 ms. PLE early 9.463904/9.464396 ms versus late 9.464536/9.461990 ms; this sub-0.04% noise is not a speedup claim. Late placement is not capacity-correct, while early placement is.
The compact prefill union prototype is exact (all seven logical lengths, both expert dtypes, including exact CPU routing-packet checks), but the fixed10*T indexed work loses badly on long chunks: layer0 T32 39.8324→68.6024 ms, layer1 46.9169→68.5929 ms. T2 improves, but the general replacement is rejected; a selective small-tail specialization is a hand-off candidate. No capability reduction.
The full per-op Tracy retry `packed_early_tracy_layer0_decode_v1` failed collection: warm/setup DRAM marker overflow and malformed host CSV. Its timings are not accepted. Whole-trace timing remains the end-to-end authority; a stage-owned per-op reporter now enables official advice.
The exact FP32 half-tree primitive `balanced_reduce_v1` passed8 random/cancellation cases and4 replays each under Watcher. A separate `probe_fused_qkv_tree.py` prototype combines exact products and the same2560-wide tree, with a setup-only permutation of BF16 weights. It is not yet part of the runtime and has no accepted performance claim. Full262144 context bridge plan `packed_early_context_plan_v1.json` is running serially with the proven repaired policy.

### 2026-11-13 13:02 UTC — full GDN context passed; exact QKV prototype
`packed_early_context_v1_gdn_context` passes262143 actual TT prefix rows and complete public final decode at262143. Prefix minimum0.99997919798; fourA/B/A/B traced decodes have minimum whole/contribution0.99999700524/0.99999696949, with exact reset output/state. This matches the inherited reduced-prefix/full-final-token scope. The plan stopped normally between jobs via its stopfile so the next prototype could usechip0; PLE/QSA fullcontext remains pending.
QKV tree CPU factorization passes8 random/cancellation cases (`qkv_tree_order_cpu_v1.json`). First kernel attempt `qkv_tree_v1` failed compilation because typecast init needs explicit numeric datatype template parameters; no kernel executed and the device closed normally. The correctedv2 uses the exact same supported init/cast pair as the already-validated convolution kernel. It is currently running the seven logical lengths underWatcher. Watcher latency is diagnostic only; clean timing and integrated HF/capacity gates are required before adoption.

### 2026-11-13 13:18 UTC — QKV clean component win and trace repair
`qkv_tree_v2.json` passes seven logical counts under Watcher. Clean `qkv_tree_clean_v3.json` preserves bitwise output for two independent inputs, repeated and alternating-input traces at every count. Traced T1 old/new3.369502/0.350303ms; T32 159.891809/8.750679ms. No full-decoder claim yet.
`packed_early_tracy_layer0_decode_v2` successfully captures the current packed policy with unchanged inline kernels materialized into content-addressed files, avoiding Tracy's multiline-source CSV defect. `report_v3` enables official advice and supplies the exact10 active experts;279ops sum8.075147ms kernels+0.173068ms gaps. Sparse matmul reported active10/512, bandwidth18.9%/5.1%; no70% roofline claim.
The MoE cleanup prototype's dtype-override zeros_like took the small-tensor host-upload branch inside trace capture. Live triage and independent diagnosis are preserved under `autofix_moe_cleanup/`. The600-second timeout ended the job; no early kill/reset. Post-timeout trivial device smoke passed. Repair uses device empty+fill and closes capture in finally; fresh Watcher/clean controls are running before any integration.

### 2026-11-13 13:24 UTC — exact QKV and cleanup integrated for gates
The repaired prototype plan completed all5jobs. BF16/BFP8 cleanup each passes35 exact rows under Watcher and clean replay checks. Combined cleanup traced T1 saves~2microseconds; T32 saves~17/26microseconds. This is a small graph cleanup, not a claimed material speedup.
`FusedExactQKV` now owns a single setup-permuted BF16 matrix and fuses the products, original half-tree and final BF16 boundary. The existing exact tree remains selectable. `fuse_moe_cleanup` removes a redundant zero-input cast, folds score permutation into row-major reshape and shared gate sigmoid into multiply. Both new options remain disabled by default pending full ranking. `exact_qkv_integrated_plan_v1.json` selects both for realHF0/1/3, batch32/PLE5, prepared4768/4767 and Watcher, and compares oldbest, QKV-only and combined policies with clean timing.

### Final-v5 default freeze
`exact_qkv_integrated_plan_v1_progress.json` completed all14jobs. RealHF0/1/3, batch32/PLE5, prepared4768/4767 and Watcher0/1 pass. Layer0 baseline/oldbest/QKV/combined traced medians46.774662/8.136646/5.117432/5.117803ms; the0.000371ms difference between QKV and cleanup is noise, so no decode gain is claimed for cleanup. Combined prefill78.710ms versus78.751ms QKV-only and248.685ms oldbest. Layer3 oldbest/cleanup9.647720/9.646938ms is similarly a tiny change. Final repeated192-sample ranking will report reproduced numbers without claiming a material cleanup win.
Default adoption pins every historical policy to the previous constructor defaults before applying its overrides. `final` alone follows the delivered defaults. The target-shape test asserts packed40, compact routes/recurrence, exact packed activation, early placement, direct unpack, exact QKV and MoE cleanup. Component probes now call the integrated implementations. Historical source bytes are retained in `sources/pre_final_format_v1/`.
Autoflake, isort, repository Black and all applicable pre-commit hooks pass before the final measurement. Runtime SHA256 `06df08406a50a4608210cb00b99af745f29281905a9144feda928e287a09f1bd` is frozen by `final_v5_source.json`. The48-job final gate plan includes integrated component Watcher, synthetic, HF, context smokes, continuations, batches, capacity, Watcher, fresh functional HF controls and12 whole-trace capture/report pairs. Separate frozen plans cover clean ranking, six per-op captures/reports, and full-context bridges.

### Whole-trace protocol adaptation and short-tail follow-up
The final-v5 component/synthetic/HF/continuation/batch/capacity/Watcher gates all pass. Fresh functional HF controls pass too; `final_v5_pcc_comparison.json` records the matched minima. QSA prefill contribution baseline/fused0.997434378/0.997524381; decode0.999578536/0.999532342. PLE prefill0.996177051/0.996170249 and decode0.997693265/0.997695353; GDN changes are below3e-7. No threshold or control changed.
The first fused whole-trace capture fails before measurement because the functional profiler worker-coverage check reads `.block`, absent from the grouped FusedPreciseProjection. Device closes normally; no reset. AutoDebug is verifying the actual grouped-product work split before a stage-test-only protocol adaptation. The failed capture is excluded; its baseline companion succeeded.
`probe_fused_tail2_full.py` isolates the previously exact compact-union prototype only for two-row MoE calls; all other counts dispatch the production FusedSparseMoe. Four full-decoder Watcher cases (layers0/3, public lengths2/34) pass functional-output/state/replay checks. The first plan rejected a stale runtime hash before starting hardware; corrected `tail2_full_watcher_plan_v2.json` completed all4jobs. Clean full-path timing at2/34/66 and component counts3..6 will decide whether any short-tail specialization is worthwhile. Production runtime remains frozen; no unmeasured candidate is default.

The indexed sparse-matmul mask cannot remove compact-union padding work in this checkout: `reader_bmm_tile_layout_in1_sender_writer_padding.cpp:94-105,268-272,311-329` uses the indices list in the mask plumbing and unconditionally computes every index slot; the mask itself is never read in indexed mode. The matching in0 reader has the same mode switch. Non-indexed compact output requires an exact host-known nnz; the per-token union count is device data, so supplying10*T as nnz is not valid. A device mask adapter cannot express dynamic compact slot skipping with the current API. No shared TTNN kernel is changed in this fused-decoder-only scope. Fixed-slot compaction remains data-dependent and is evaluated only at short tails where full-path evidence supports it; longer unions remain rejected.


## 2026-11-13 14:16 UTC — short-tail candidates and profiler repair

`tail_union_rank_plan_v1` completed10/10 jobs, including full public lengths2/34/66 on GDN/QSA and component3/4/5/6 for both expert dtypes under Watcher and clean traces. At public2, GDN prefill19.37885->17.75228ms and QSA22.82017->21.69957ms. At66, GDN169.10748->169.21591ms is noise/slightly slower; QSA393.16245->391.20331ms improves. Decode kernels are unchanged by the prototype; measured shifts are noise. Component3/4 improves both dtypes (~1.5ms), while5/6 is marginal/distribution-dependent. Fixed-capacity padding therefore remains rejected for general prefill.

The production helper is now integrated behind `compact_prefill_limit` (default1, no selected-policy change). The production early activation allocation and shared cleanup are retained. Frozen runtime22945d216f66b4e16239094fbc6965fe5179d7cdf249c11092df1a88d4f2d09a is being tested by `integrated_tail4_rank_plan_v1.json`: actual runtime routing/kernel, both dtypes/Watcher, then whole public2/3/4/34/35/36 versus v5 and functional controls. This does not introduce public alignment constraints.

Profiler repair control completed4/4 fresh baseline/fused prefill captures and official strict reports with unchanged v5 runtime. The fused projection geometry is proved from actual320-row FP32 weights rather than a fabricated `.block` attribute. See `autofix_trace_protocol/{AUTODEBUG,AUTOFIX}.md`, `trace_protocol_repair_control_plan_progress.json` and `tracy/{baseline,final}_layer0_prefill_final_v5b/`. No reset or relaxed endpoint inventory was needed.

The repaired native whole-trace L0 prefill report gives functional424.122685ms versus selected v5 fused78.502730ms for32logical rows (`tracy/{baseline,final}_layer0_prefill_final_v5b/report/latency_summary.json`). These native intervals include internal gaps and correspond to27893 versus1002 captured operations. They corroborate the clean host timings but do not replace final-source layer/mode coverage.


## 2026-11-13 14:24 UTC — residual write-back fusion candidate

Source audit of `GatedResidual.write` identifies adjacent FP32 casts/product, BF16 boundary, stream permutation/reshape, FP32 residual add and BF16 boundary. A model-local prototype `tests/probe_fused_residual_write.py` combines these into one kernel, preserving both rounding boundaries. The reader maps block/injection/residual tiles, SFPU column broadcast multiplies in FP32, explicit typecast implements the first RNE boundary, FP32 addition plus explicit RNE implements the second, and the writer emits finalstream order. Norm/read semantics are unchanged. `residual_write_component_plan_v1.json` prepares Watcher and clean tests at1/2/3/4/7/16/17/31/32/33/65, two input families (random and BF16 midpoint/cancellation), with12 alternating-input trace replays. This is a test-only prototype; no selected runtime change yet.

The short-tail whole-path sweep already shows that a4-row compact limit is too broad: GDN public4 prefill22.235->23.985ms regresses despite random-component gains. Candidate limit3 is the provisional selection; actual public2/3 improves on GDN andQSA. The fixedslot sparse API explains distribution dependence. Remaining chunk-boundary comparisons continue and the4-row regression is retained as rejected evidence.

Public35 GDN also rejects the3-row union threshold as a general default: prefill97.721146->98.295776ms (see exact samples in integrated_tail4_layer0_n35_clean_v1.json). The provisional choice is now2, whose public2/34 wins and66 difference is within~0.1ms on169ms. This is distribution-dependent sparse work, not a correctness issue. Fullpath and component artifacts retain the rejected3/4 thresholds; no host branch on route contents is introduced.


## 2026-11-13 14:38 UTC — integrated exact residual writer

`residual_write_component_plan_v1` passes4/4: prototype Watcher and clean11-length checks plus repeated-identical-route Watcher checks for both expert dtypes. Residual write is bitwise exact for random and BF16 midpoint/cancellation inputs, including12 alternating-input replays. T1 original0.0762716->fused0.0227690ms, T32 original0.1250301->fused0.0233646ms. This is component latency, not a claimed full-decoder gain.

Existing-op exploration: `ttnn/cpp/ttnn/operations/eltwise/ternary/ternary.cpp:331` dispatches native addcmul; `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h` forms product-plus-residual without an intermediate BF16 rounding boundary. Exact counterexample: BF16 block1.0078125, injection1.5, residual-1.5. Required product1.51171875 rounds to1.515625 (tie-to-even), then adds to0.015625. A multiply-add without the intermediate boundary gives0.01171875. Layout adaptation cannot restore the omitted boundary without separating the multiply. The dedicated writer preserves it and directly emits the stream layout. No shared TTNN/LLK source was edited.

The prototype is integrated behind `fuse_residual_write=False` by default; policies `best_writer_v5` and `best_writer_tail2` enable it, with the latter also selecting2-row compact MoE tails. Factory ownership transfer drops the original component reference before later packed-weight replacement, avoiding unnecessary duplicate owners on PLE. RuntimeSHA8f8a4e2adccf66cb43135a6d1f11d8ed65d472ce1c3ce46f8e815ffb20f145ea is frozen for `integrated_writer_plan_v1.json` (17jobs). It covers actual integrated writer Watcher/clean, expanded HF lengths on GDN/QSA, PLE34 and2, batch32/PLE5, prepared4768/4767 plus4738, Watcher0/1, and clean comparisons against the best correct v5 candidate. Pre-commit passed (`logs/precommit_writer_candidate_v1.log`). No default adoption or final acceptance claim yet.


## 2026-11-13 14:46 UTC — expanded PLE34 failure isolated

`integrated_writer_plan_v1` stopped at job4: literal HF prefill row5 below0.995 on newly added length34. The gate remains failed. AutoFix collected five matched device controls and the existing CPU packed-backend counterfactual. Functional baseline fails the same row; v5/writer/tail/combined fused full outputs are bit-identical. Actual packed lookup values in otherwise unchanged HF reproduce the failure, and a six-projection FP64 control does not repair it. Fused versus packed-HF all-row minimumPCC is0.9994620whole/0.9994261contribution. Exact evidence and unresolved acceptance classification are in `autofix_ple_tail/AUTOFIX.md`, `ple34_control_comparison_v1.json`, and `ple34_packed_backend_v1.json`. Independent AutoDebug is running. No fixture or threshold was changed.

`integrated_writer_remaining_plan_v1` separately restores the original33-token HF/Watcher workload and continues unrun candidate/capacity gates; the original33HF gate passes. The34 failure is not erased or converted to a pass. The prepared length2 capacity clone also had its incompatible `--prepare-all-tails` option removed before first execution (that option requires at least32rows); no delivered public input restriction is added.


## 2026-11-13 14:58 UTC — writer gain and QSA expanded-fixture controls

`integrated_writer_rank_layer0_v1.json` passes all functional/state/replay checks. Functional prefill422.691224ms/decode46.779762ms; previous best78.828728/5.118903ms; writer78.667969/5.016447ms; writer+tail2policy78.598816/5.017188ms. This32-row workload has the same graph for the two writer policies; their difference is noise. Batch0_32, PLEbatch5, prepared4768/4767, WatcherGDN2/34/66 and originalPLE33 pass in `integrated_writer_remaining_plan_v2`.

Expanded QSA34 and66 HF probes fail only prefill row30 and44 respectively, while all trace/state/decode checks pass. Five matched device controls preserve those failures. Functional baseline fails the same rows; every prior/new fused full output is byte-identical. `qsa_tail_control_comparison_v1.json` verifies exact source/oracle/input identity and byte equality. `qsa_tail_backend_cpu_v1.json` independently reproduces the literal oracle, then changes only attentionGRdown accumulation toFP64 and rounds to the sameBF16 boundary. This reproduces the literal whole/contribution outliers; fused/control all-row minima are0.99860567/0.99856806 at34 and0.99706912/0.99696749 at66. MLPdown-only does not reproduce them; all-GRFP64 changes the34 result, so the conclusion is narrowly scoped backend sensitivity, not a universal FP64 oracle. Literal gates remain failed. See `autofix_qsa_tail/AUTOFIX.md`; independent AutoDebug is pending. No threshold or fixture is weakened.


## Owner wrap-up — 2026-11-13 15:12 UTC

Owner explicitly ended tuning at the stage wall cap and requested closure within about one hour. This supersedes the earlier 8-hour-device/70%-roofline stopping condition. No further performance candidates are pursued. Required remaining gates run once, with 32K context smokes and no repeat maximum-context/capacity sweeps. Existing completed maximum-context and capacity evidence is retained. No device reset has been used.

Owner acceptance clarification: PCC>=0.995 applies to whole-output/per-layer-kind aggregate; per-row/user/step is a diagnostic floor>=0.99 with top-1 agreement. Matched original-HF reproduced per-row misses are recorded limitations. Raw historical stricter per-row failed runs remain failed. `wrap_accuracy.json` independently recomputes aggregate metrics from saved tensors and explicitly distinguishes decoder hidden-coordinate argmax from unavailable token logits.

Selected default: existing `best_writer_tail2` candidate, exact residual writer enabled and compact union threshold2. Threshold3/4 and general union were slower on full-decoder inputs. Added physical-padding/standard-tile validation to the residual writer; arithmetic unchanged. `wrap_final_plan.json` freezes source and runs default-path synthetic/Watcher/batch checks, clean same-harness candidate/default ranking, six official per-op profiles/reports, and32K smokes. Earlier candidate checks retain same arithmetic and explicit flags; no need to repeat every unchanged gate.

Evidence policy now uses logs plus one closure manifest. Raw profiling files remain locally available; no additional archive pass or per-artifact compression will run. Untried ideas are hand-off notes, not blockers under the owner wall cap.

# Owner wrap-up hand-off

The owner wall-cap instruction stops candidate tuning on 2026-11-13. The selected
configuration enables the exact residual writer and a compact-expert threshold
of two physical rows. This note makes no 70% roofline claim. Required closure
uses `wrap_final_plan.json`; pending jobs are not described here as passes.

## Not tried (stop rule)

- Further matmul geometry, DRAM reader/shard and fidelity combinations: leave to
  the separate optimized-decoder stage. No measured gain is claimed here.
- General dynamic-size compact expert unions beyond two physical rows: fixed
  padded unions are already measured slower at full tiles, and thresholds 3/4
  regress some complete public lengths. A new indexed-kernel contract that skips
  padded slots could help; the current indexed sparse API computes each slot.
- Further residual-read fusion and direct output geometry for precise projection
  reductions: potential dispatch/movement savings are unmeasured. Read-side
  stream grouping changes order and cannot simply be deleted.
- Additional QSA GR accumulation-order controls and higher-precision PLE table
  formats: the existing HF controls and literal failures remain on disk. Further
  precision experiments are not needed solely to force every row above 0.995
  under the owner's whole-output/aggregate gate. Do not erase the original
  diagnostic results or present changed-arithmetic controls as literal HF.
- Additional max-context and capacity sweeps: not run under owner direction.
  KV dtype/layout and context262144 are unchanged; existing maximum-context
  evidence is retained and final-source closure uses 32K smokes.
- Additional trace/per-op profiling variants: final selected-path reports are
  required once; new candidate-specific profiling is deferred.

## Acceptance and evidence policy

Whole-output (or per-layer-kind aggregate) PCC must be at least0.995. Per-row,
per-user and per-step PCC is diagnostic, with floor0.99 and top-1 agreement.
Matched HF-reproduced row misses are recorded as known limitations. This policy
is an explicit owner update; historical strict-row probe outcomes remain intact.
This decoder stage has no LM head or generated-token contract, so any top-1
metric must identify its output domain rather than call hidden-state argmax a
predicted token.

Keep logs and one final manifest. Do not create new per-artifact archives or
repeat historical archival work. Final review, local stage-owned commit and
commit-SHA logging remain required. No push is authorized.



## Final timing and report closure — 2026-11-13 15:25 UTC

`wrap_final_rank_layer0.json`, `wrap_final_rank_layer3.json` and `wrap_final_rank_layer1_final.json` pass repeated state/output checks. Default traced decode is5.018603/6.341421/9.548145ms for GDN/PLE/QSA; warmed prefill78.399826/95.834846/184.512177ms. Baselines46.771099/47.679710/249.688477ms decode and423.757202/463.053467/970.489502ms prefill. PLE baseline is the earlier serialized same-harness BFP8 resident-table control, not an interleaved two-model run. See final_performance.json for exact source paths and gains. Final candidate/default agree within timing noise; all beat previous correct v5 decode.

Six final-source tt-perf-report jobs pass under tracy/wrap_final_ops_layer{0,1,3}_{prefill,decode}/report/. Kernel sums for prefill are78.003737/92.397247/183.022748ms; decode4.966203/6.290691/9.404568ms. Decode profiler kernel+gap5.100040/6.538532/9.657134ms is instrumented and not substituted for clean replay. Prefill uses segmented drains and has no aggregate-gap claim. All official tables/CSV have advice enabled. Exact commands are wrap_final_plan.json; each run has its log and existing run_recorded provenance.

Source pre-commit passed (`logs/wrap_all_source_precommit.log`). Staged artifact checks passed with only mutating trailing-whitespace/end-of-file hooks skipped (`logs/wrap_staged_precommit.log`) to preserve exact raw logs and profiler CSV; those formatting checks were run on production/test source. Repository500KiB cap is respected: large raw results/launcher captures remain locally available and are named in manifest.json, with concise metrics and command logs committed. No new compression/archive pass is performed.


## Final gates completed — 2026-11-13 15:35 UTC

The frozen final queue completed21/21 jobs without a failed gate. Synthetic14/14, final QSA Watcher/batch32, same-harness ranking, all six per-op capture/report pairs, and all three32768 context smokes pass. GDN32767-prefix minimum-row PCC0.99998319 and complete final decode minimum0.99999718; QSA32768-row whole PCC0.99997601, minimum-row0.99986607, three decode minimum0.99998142. PLE reduced-prefix plus complete public final decode, replay and exact state/output reset pass. All context runs use trace allocation tracking with program-cache allocation inclusion; report/clean timing runs remain separate. Exact scopes and PLE metrics are in final_gate_summary.json. Context262144 and preparedPLE4768/4767 are preserved. No reset performed. No more hardware work is planned before review unless a correctness finding requires it.


## Independent review closure — round1

Fresh xhigh stage-review returned **clean-pass**, no Required Work: `stage_review_round1.md`. It independently verified final source/plan/log provenance, all21 completed jobs, saved-output aggregate PCC/equality, allocation-tracked replay, capacity/tails and all six CSV totals. Controlled row sensitivity, decoder-only token-top1 scope, 32K/reduced-prefix context scope and deferred optimization advice remain explicitly documented concerns; none requires additional stage work. No re-review or new hardware run is needed. Source checks, human-doc checks and staged checks passed. Only raw-artifact-mutating whitespace/newline hooks are skipped for the checkpoint to preserve the measured logs/CSV; source/human docs received those checks separately. No push is performed.

## Local checkpoint

Repo `/home/ttuser/qwen38-flash/tt-metal`, branch `main`: `17553c7c59bacbd53375579e1d49467ac29df8a7` — validated fused runtime, tests, docs, logs, official filtered reports and clean-pass review. No serving/plugin/vLLM repo was changed. No push. A following documentation-only commit records this SHA; it does not alter reviewed source or measured behavior. Large raw profiler captures and oversized historical JSON remain local as declared in the manifest.
