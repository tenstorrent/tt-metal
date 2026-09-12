# Optimized multichip decoder work log

Stage 5, Qwen/Qwen3.8-27B, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Starting checkpoint:
`1623f9cb595` (completed multichip decoder). Working tree was clean.

## Initial operation-topology audit

Source: `tt/multichip_decoder.py`, inherited `tt/optimized_decoder.py`.
Measured reference: `../multichip_decoder/profiler_interpretation.md` and
the four-device tables under `../multichip_decoder/tracy/final_profile_l{0,3}`.
Times below are prior device0 decode microseconds, not new stage measurements.

| Boundary | Current topology / cost | Candidate and constraints | Action / evidence |
| --- | --- | --- | --- |
| Input norm → attention | Width-sharded BF16 residual; packed QKVG or QKVZBA; BFP4/LoFi DRAM matmul, 36–44 us | Preserve packed head grouping; compare wider K blocks and multiple bank readers | Read prior geometry controls; investigate unit-mesh reader assertion with AutoFix |
| Local attention → output | Local heads, native recurrence or paged BFP8 SDPA; output matmul 19 us; direct async AR | AGMM with output-column weights; MMRS carrying hidden1280 through distributed norm; BF16/BFP8 payloads | Prior adapted family measurements exist; recheck against current best contract |
| Residual add → post norm | Replicated hidden5120, on-device L1 sharding; no inter-layer CCL | Hidden1280 residual and distributed norm, followed by gathered normalized input | Prior sharded carry-forward path measured; inspect complete next-consumer contract |
| MLP norm → gate/up | One packed BFP4/LoFi projection, 82 us, split + SiLU/multiply | Separate gate/up with fused elementwise; precision-locked geometry and reader sweep | Prior packed/separate evidence exists; current packed projection dominates math |
| MLP down → residual | BFP4/LoFi DRAM matmul, 43 us; async AR | RS carry-forward / fused MMRS / AGMM; persistent buffers | Two AR rows total 31.6 us; shared 1.25 MiB L1 workspace; audit dtype propagation |
| Helper layout transitions | 82.8 us linear / 39.3 us full total movement | Carry compact residual and avoid conversion at helper boundaries | Audit inherited packing/unpacking and final default traces |
| Prefill | 2048-token internal chunks; real logical tails; DRAM projections with minimal/2D selection; RS+AG | Larger coherent matmul blocks, placement, chunk and collective families | Preserve public non-aligned input and context262144; compare short/long/tail workloads |

All families must retain real model shapes, TP4 ownership, accepted PCC and
trace replay. Single-chip timing is never the before result for this stage.
No full-model or vLLM implementation is in scope.

## Environment

- Enabled plugin inventory: `codex-home/config.toml` enables tt-autodebug and
  tt-model-bringup. Package setup uses their installed environment.py.
- `timeout 60 /home/mvasiljevic/tt-metal/python_env/bin/tt-smi -ls --local`
  succeeded and listed all four Blackhole p300c chips. The checkout's
  Python environment has no tt-smi executable, so the existing health utility
  was used. No reset was needed. No active device test was found.
- Hardware commands are serialized; watcher and profiler use separate runs.

## Execution record

The entries below preserve the experiment sequence, including temporary failures
and decisions superseded by later controls. The final review/checkpoint entry
and README identify the accepted source and measurements.

## Baseline and early findings

- Fresh TP4 S128/B1: linear prefill1.41584ms, traced decode0.47484ms;
  full prefill1.23377ms, traced decode0.35129ms. Both pass real-weight
  parity, changed-input bitwise trace replay, and state/cache ownership.
  Commands/source and samples are in before_l{0,3} artifacts.
- The initial stack baseline completed device work but the host reference
  fixture for S128 was missing. Generated stack_reference (single-chip
  correctness oracle only) and reran before_stack_retry on TP4. No model
  or hardware failure occurred; the fixture is outside runtime timing.
- CCL dtype audit: the earlier BFP8 direct-AR experiments used explicit
  casts (source bac850be...), but the current default source had removed
  those optional casts. Restored policy propagation so new CCL candidates
  really exercise their stated payload. Default BF16 math is unchanged.
- Required build-wrapper preflight: `.github/scripts/copilot-build.sh`
  fails because Docker is unavailable. Contrary to the generic environment
  note, an existing native clang20/Ninja build and Python TTNN library are
  installed. No compiler/dependency installation is needed; any native
  repair will be compiled with the existing narrow build target.

## Native multi-reader repair

AutoFix diagnosis `AUTODEBUG_dram_mesh.md` verified the coordinate-free hop
query was applied to a non-unit mesh. Applied `dram_mesh_native.patch.gz`:
the DRAM factory opts into per-coordinate descriptor construction and uses
the corresponding local physical chip for multi-reader placement/NoC arguments.
The Python descriptor binding keeps its old optional core-range argument.
Remote multi-reader descriptors fail explicitly; single-reader physical-device
selection remains unchanged. No tensor copies or unit-mesh fallback are added.

`/usr/local/lib/python3.12/dist-packages/cmake/data/bin/cmake --build build_Release --target ttnn --parallel 4`
compiled and linked both Python extensions successfully (`native_build.log`).
Loaded build-library hashes are in `native_library.sha256`.
Hardware validation and reader selection are in progress.

### Installation and padding controls

- The first native_reader{2,3} runs still loaded installed old libraries:
  traceback retained optional<CoreRangeSet>. The build target only linked
  build_Release/ttnn; Python loads source-tree _ttnn.so + build/lib/_ttnncpp.so.
  Installed components ttnn-runtime and tt_pybinds with CMake, prefix
  `$PWD/build_Release`. `native_install.log` and `native_installed_library.sha256`
  record the correction. Those early native_reader1 results also used old
  libraries and are not post-fix evidence.
- Installed_reader2_l0 reached a new host assertion: Worker7-3 has no storage
  area assigned. N130 tiles with16 readers×9 tiles produces144 tiles, but
 9 active storage cores×15 tiles provide135. The last reader owns padding only.
  AutoFix verified this arithmetic and is repairing zero-write worker handling.
- Adapted padded_reader2_l0 adds zero columns to each local weight and slices
  before semantic consumers. Passes exact prior PCC and trace checks; decode
 0.439073ms vs one-reader control0.470098ms. This is a candidate, not final.
- Split attention controls passed: linear0.507169ms and full0.395991ms,
  compared with packed controls0.470098/0.351291ms. Complete local head groups
  and B/A padding are owned by setup. Real-weight outputs and states pass.

### Installed native tail fix and precision-locked reader matrix

`native_tail_build.log`, `native_tail_install.log`, and
`native_tail_installed_library.sha256` identify the compiled/loaded tail-worker
fix. Real TP4 decoder matrix uses no logical output padding. Alternating
readers1/2/3/3/2/1 gives linear decode469.27/438.06/479.42/478.92/438.58/469.08us
and full352.97/320.92/360.95/359.82/325.24/353.69us. Output/state PCC is
unchanged for every reader count. Real-activation isolated role traces include
input/output layout costs: all four roles improve with two readers.
`readers_matrix.json` records policies and `readers_matrix.log` the serial run.
The native focused test initially had8 Python namespace errors in the new
descriptor-inspection helper and4 passing one-reader tests; repaired the helper
to use the actual nested binding. This was not a device failure.

A stack-compatible sharded residual candidate keeps hidden1280 through residual
adds and distributed norms, with L1 collective/residual storage for B1 decode.
`sharded_l1_retry` passes PCC>=.99990189 but costs1.026962ms for the two-kind
stack versus replicated .7913ms. Host comparison gather stays outside timing;
this is a genuine carry-forward candidate, not an immediate full-residual restore.
Further comparisons will combine this family with selected reader geometry.

Native focused regression retry: **12 passed**,84 deselected in11.94s. Includes readers1/2/3, TP4 and offsetTP2, alignedN1024 and padding-workerN4160; all-rank output PCC, allocation rebinding/cache hits, and coordinate-specific physical placement. See native_mesh_tests_retry.log. Exit0 with common nanobind shutdown leak diagnostics; no device corruption, cache, or execution assertion. Geometry down34 now passes with2readers; previous one-reader L1 failure is not used to reject it.

### Precision-locked geometry and coherent residual layouts

`geometry_matrix.json` and `cumulative_matrix.json` carry BFP4/LoFi/FP32acc
through every trial. Measured Q/gate/up core counts5/10/20/40/80, corresponding
K blocks32/16/8/4/2; output4/8/12/24/48 with blocks12/6/4/2/1; down4/8/17/34/68
with blocks34/17/8/4/2, for both two and three readers. Full-attention Q repeated
separately because N3584 differs from linearN4160. All pass real-weight parity.
The cumulative mixed-reader candidate is linear0.430170/full0.312511ms.
Matched residual/norm/projection grids were measured as two-layer stacks:
cores10 AR80/10:0.769822/0.750592ms; cores20 AR80/20:0.751809/0.744525ms;
cores40 AR80/40:0.711807/0.711733ms. The latter removes additional local
reshards while preserving replicated mesh residual ownership and no layer-boundary
collective. Final comparisons remain pending.
Full Q5/readers2 had noisy host samples (0.324–2.84ms) and median0.666ms;
its isolated Q trace is51.47us vs39.6us for Q10. That row is not a reliable
whole-layer magnitude estimate; the independently stable Q microtrace rejects
its geometry. No performance headline uses that outlier.

### Precision diagnostics and shared-memory cleanup

Per-role BF4/HiFi2, BF8/LoFi, BF8/HiFi2, and reduced-accumulation trials
are in precision_matrix and precision_remaining_matrix. Full attention BF8
projection weights legitimately differ from the frozen BF4 cache reference:
output PCC>.9999, changed-input trace/state equality and cache ownership pass,
but raw key/value PCC~.993/.994. The initial strict check failed after successful
device execution. Retried with explicit --cache-pcc-diagnostic: only cache
PCC is diagnostic; output/per-user gates and every exact replay/ownership check
remain mandatory. Final default runs will not use this diagnostic option.
Rejection is based on higher latency, not cache-PCC differences.

MPI warned about /dev/shm capacity during some trials. Inspected every owned
live process's mappings/fds; all inaccessible entries were verified zombies.
669 unreferenced, owned sm_segment files consumed49,324,032 bytes of64MiB.
Removed only those segments, preserving TT locks and unrelated objects;
usage is now168KiB. Metadata/guards: shm_inspection.json and shm_cleanup.json.
No device failure/reset occurred. Matmul and CCL work runs on the local mesh;
clean-shm timing controls will verify comparison stability. Some single-replay
host samples have millisecond scheduling outliers. Added optional queued timing
(5 samples of20 replays) alongside the unchanged primary metric; final queued
stress checks evolving-state equality separately. No outlier-based speedup is
used to select a candidate.

Existing unit-mesh native controls:11 passed in6.09s (native_unit_mesh_control.log).
Their nanobind shutdown leak diagnostics match the new mesh tests, controlling
that warning as existing module/binding teardown behavior.

### Collective families after geometry selection

Clean-shm old-policy controls reproduce linear0.466275/full0.351031ms, consistent
with the earlier one-reader controls. Current replicated-residual stack control
is0.705983ms. All families use real TP4 head/channel ownership and two-layer
carry-forward evaluation, including exact trace replay and changed cache/state.

BF16 RS+AG replicated costs0.747247ms in DRAM and0.743841ms in L1. RS with
hidden1280 carried through the next residual/norm costs0.996953ms in DRAM,
0.954393ms in L1 without persistent buffers, and the matching persistent-L1
variant is recorded in topology_matrix.log. BF8 payload variants are also
measured, including normalized-activation gather payloads (stats remain BF16).

The first AGMM trial with2links hit a sender-axis layout assertion: a10-core
axis with10senders/link forms1group, not2. Adapted to5senders/link (or4 for
the8-core non-transposed axis); retry passes. K-block adaptation uses16 for
output and17 for down so each divides its local K48/136 tiled width.
Fused AGMM and MMRS use correctly partitioned row/column weights, persistent
activation/output buffers, transpose alternatives, BF16/BF8 communication,
and combined attention/MLP activation8 variants. MMRS additionally exercises
the rolling L1 output window with two shared caller-owned progress/credit
counter arrays, removing its DRAM handoff. These adapted paths pass but lose.

Fused distributed RMSNorm (local stats+fabric gather+normalization in one op)
reduces the BF16 sharded-RS stack to0.774564ms. Combined fused norm+AGMM is
1.085864ms BF16 /1.035203ms BF8; fused norm+L1-window MMRS is1.008271ms
BF16 /1.053027ms BF8. This comparison preserves sharded residual ownership
through the next layer; comparison-only host gather is outside timing.

Packed MLP wins against separate gate/up with fused binary SiLU0.727490ms
and matmul SiLU epilogue0.736567ms; native fused SwiGLU minimal matmul costs
1.002424/1.002775ms for the two tested K-block policies. Additional tuned
separate-projection controls are being completed. Ring2links beats ring1link
0.758840ms and linear2links0.715221ms. Actual BF8 directAR reaches0.705608ms
but lowers PCC with no clear latency separation fromBF16 control.

### Long-prefill 2D adaptation and device ownership audit

The unblocked 8x8 K4 M=2048 packed MLP configuration requested 2,070,528 B circular buffers versus 1,572,864 B L1. Retrying the 2D family with `out_block_h=1` bounds accumulation storage while preserving real shapes and weights; see `prefill_remaining_matrix.json`. The source-only AutoFix agent accidentally initialized UMD through a linked TensorSpec capacity probe at 02:19:46.648–02:19:47.811 UTC. The preceding model process had closed by 02:18:18.703, so no model timing overlapped; future linked probes remain prohibited. No mesh, kernels, reset, or model operations were issued by that probe.

Minimal prefill K16/N16/M8 exceeded L1 (1,979,392 B requested versus 1,572,864 B). K16/N16/M4 and K32/N8/M4 had already passed. Retrying M8 with N8 reduces its output and weight tile buffers (`prefill_remaining2_matrix.json`).

### Closing topology and precision comparisons

Column AGMM was tested after fused distributed norm, retaining local hidden1280
through the layer boundary and gathering only at packed attention/MLP matmuls.
Both BF16 and BF8 payloads, K8/20/40 and N8/16 pass (1.248–1.351ms stack),
without an immediate replicated-residual restore. Tuned separate MLP gate/up
(core10 and40, two readers, fused binary/epilogue alternatives) reaches0.719038ms
at best versus the packed0.705983ms control. Split attention's narrow N32 output
conversion falsely rejected a valid full-shard allocation; zero-padding the
split weights and slicing output adapts that contract. Both kinds pass strict
PCC at0.468754/0.348073ms, slower than packed0.423/0.306ms.
`AUTODEBUG_narrow_output.md` records the allocation proof. Its broader native
validator patch is excluded because the measured model adaptation closes this
optimization and the selected packed outputs do not use that conversion shape.

FP16 accumulation was compared FP32/FP16/FP16/FP32 in the same cumulative stack.
Primary medians:0.710863/0.704580/0.705147/0.705453ms. Queued medians:
0.691322/0.690548/0.690735/0.691576ms, with overlapping sample ranges.
The sub-0.2% queued difference is not a material, reproducible improvement;
FP32 retains the stronger state PCC (.999928 versus .999892).

BFP4 KV trials use real K/V, correct fill casts, BF16 decode updates, and both
original/changed-input cache-consuming trace replays. Strict raw cache PCC
against BFP8 fails at~.9815 while output PCC remains above.9998. Diagnostic
reruns preserve output/per-user, cache ownership and exact replay gates.
At logical4097 decode is0.344728ms versus BFP8 control0.347499ms; short-context
BFP4 is0.307207ms, similar to prior BFP8 controls. Retaining BFP8 preserves
the accepted raw-state parity; no material short-context speedup was observed.
The diagnostic reports are explicitly excluded from final acceptance evidence.

### Full Ethernet watcher adaptation and recovery

AutoFix found supported `TT_METAL_FABRIC_OPT_LEVEL=Os` and
`TT_METAL_WATCHER_NOINLINE=1` size controls that retain all watcher checks.
Each fabric optimization experiment uses a fresh `TT_METAL_CACHE`: the current
JIT key omits per-kernel optimization level, so reusing it could silently measure
the old binary. Compile commands are retained to verify actual flags.

`watcher_full_eth_os` fits and passes real TP4 linear prefill/decode PCC plus
changed-input/cache/state replay under watcher10, with no disabled features.
However process teardown aborts134: device0 ETH29-25 never returns to base
firmware; heartbeat remains fabric marker0xdcba3800. This is NOT watcher-clean
signoff. Watcher had detached before RISC teardown. Saved inspector YAML,
watcher logs, native stack and compile commands remain under that run directory.
The process aborted itself before triage; current tt-triage requires live RPC
or serialized capnp and cannot consume these YAML logs (capture.log).

Recovery in `triage_watcher_os/`: bounded local list succeeded (four chips),
`timeout -k 10 180 .../tt-smi -r` succeeded, post-reset local list again shows
four chips, and a bounded normal Ring MeshShape(1,4) open/close passed exit0.
No stale process needed killing. AutoFix is investigating the shutdown path;
next isolated control is O3 plus watcher noinline, with every check enabled.

`watcher_full_eth_noinline` (fresh cache, O3 plus watcher noinline) also passes decoder gates then aborts134 at the same ETH29-25 shutdown wait. This refutes an Os-only explanation. Both reports have explicit exit_status134 markers; saved model JSON is not a clean process result. Hardware is being preserved for direct post-abort mailbox/register capture before another reset.

Post-abort ExaLens capture succeeded using NOC1: all16 active ETH cores have
128KiB L1 snapshots plus four debug-bus PC samples per ERISC (no halt/reset).
SYSTEM_NOC selection was unsupported in this ExaLens build; the initial
attempt stopped with an index error before memory reads. Offline mailbox/ELF
analysis identifies `DebugAssertNCriscNOCPacketTagClearedTripped`, ERISC1 at
active_erisck.cc51 on all16 cores. ERISC0 waits for its subordinate to finish.
The assertion concerns sticky packet-tag configuration, not outstanding write
completion counters. Write/atomic barriers do not clear those configuration
registers. The fabric router lacks the cleanup already used by fabric mux.

The second bounded list/reset/list and normal Ring mesh smoke passed. An empty
watcher10 O3+noinline Ring mesh also closes cleanly (`watcher_empty_mesh.log`),
so the assertion needs workload traffic that sets transaction tags. AutoFix
is preparing a minimal router teardown fix that retains all assertions and
synchronization. Normal SDPA tuning resumes while that source work continues.

### Teardown repair and final candidate confirmation

Applied `watcher_packet_tags.patch.gz`: after existing router write/atomic barriers,
clear only this ERISC's owned NoC packet tags before peer synchronization.
`AUTOTRIAGE_watcher_os_teardown.md` records all16 terminated-router mailboxes,
ERISC1 assertions, and all32 generated ownership maps. The host command
`cmake --build build_Release --target ttnn --parallel 4` exits0 (no host rebuild
needed for the JIT kernel edit); fresh-cache watcher compilation and a traffic
run are the executable kernel check. `watcher_full_eth_tags_fix` uses O3 and
watcher noinline with every watcher feature enabled.

The 13-case SDPA sweep passes. At logical4094, K128 reduces full-attention
decode to0.319400ms (8x8) /0.319655ms (11x10), versus K64 0.325401/0.325887ms.
K32 is slower. Short grids4x1 through11x10 differ by only1–3us. A paired
queued-timing confirmation is recorded separately before default selection.
Short prefill1D, L1 input placement, and chunk4096 are also combined before
selection; previous long-prefill L1 overflows were adapted with smaller output
blocks and then measured, as recorded in the prefill matrices.

`watcher_full_eth_tags_fix` model JSON passes and UMD cluster destruction
completes normally at02:56:58.768. However the shell wrapper itself exits2
with an unexpected-EOF error because its provenance loop was edited while
Bash was still reading it after the long child command. `bash -n` now passes;
a repeat with the wrapper frozen is required for an unambiguous exit0 artifact.
This wrapper-lifecycle error is separate from the repaired native teardown.

Paired SDPA confirmation passes both ordinary and queued timings. Logical4094
K64/K128/K64/K128 primary medians are0.326724/0.319596/0.325416/0.321099ms;
queued medians0.312850/0.306771/0.313005/0.307394ms. Select K128, retaining
the existing internal mapped-page divisibility adaptation for logical tails.
Short8x2 versus11x10 queued medians overlap at0.2934–0.2938ms; keep8x2.
All SDPA outputs/cache comparisons remain above.9999991.

Combined prefill1D tests (60 warmed samples) give linear K8 L1/DRAM
1.320051/1.384769ms. Full K20 L1/DRAM1.358665/1.239446ms and K40
1.260553/1.282419ms. Select short linear K8 L1 and full K20 DRAM.
The L1 option is scoped to the bounded short1D branch; it must not move
unbounded prefill inputs into L1. The long prefill path keeps its previous
minimal/2D program policy, with chunk4096 selected from measured tail results.

The frozen default runs with empty policy overrides reproduce the selected
decode candidate: linear0.421545ms, full0.307652ms, stacked0.705277ms.
Warmed prefill128 is1.304411/1.268657/2.704205ms. Full-attention short-prefill
host timing is slightly above the original1.233765ms measurement; final
reporting retains the measured default value rather than an earlier faster
sample. PCC minima are.999957/.9999994/.999928. The default source now
scopes L1 input placement to its64–256-token1D branch. Shorter lengths retain
the prior DRAM-sharded prefill path, and larger chunks retain the long-prefill
programs. The full31-case regression begins with the final source.

### Final correctness and native regressions

The final default31-case suite passes in403.10s: both kinds atS1/31/32/33,
2047/2048/2049/4097, continuation33/129, B3 continuation, B8 and B32, plus
stacks atB1/2/3/8/16. Each run includes strict PCC, per-user PCC and changed
input/cache/state trace checks; no policy override or cache diagnostic is used.
`final_correctness_pytest.log` and all `final_*.json`/exit markers are retained.

The expanded native mesh suite passes18 cases in19.30s, including readers1/2/3,
full TP4 and offset TP2 views, regular/tail/narrow N shapes and allocation-cache
generations. `native_final_mesh_tests.log` closes the source review's12-versus18
coverage gap. Existing11 unit-mesh controls remain valid: no native matmul edit
has occurred since their passing run. The only subsequent kernel change was
fabric teardown, exercised by full traffic watcher and TP4 model regressions.

`run_final_validation.py` serializes final stress, full watcher, capacity,
profiler and longer timing matrices. Any failed child stops subsequent device
work. Watcher and profiler environments remain separate.

Final stress passes100 eager versus100 queued trace iterations through both
layer kinds atB1/S128 andB32/S257. Evolving outputs and state are bitwise equal;
minimum PCC is.99975026/.99982654 and minimum per-user output PCC
.99999273/.99998909. The B1 reference was regenerated from the unchanged
single-chip oracle only to add stress-state keys; it is not TP4 timing evidence.

All four final watcher cases pass: linear non-aligned2049, linearB32/S257,
fullB32/S257 and stackedB3/S33. Each uses watcher10, O3+noinline and every
watcher feature enabled; each wrapper exits0 after native driver close.
`final_watcher.log`, per-case logs/JSON/environment/source/exit markers and
generated watcher logs retain evidence. No profiler was enabled in these runs.

The capacity plan includes7,015,956,480B projection copies per rank,
2,281,701,376B full-model KV, recurrence/constants/terminal reserves and18GiB
activation/trace allowance. Total30,069,112,832B <34,138,688,512B physical DRAM,
leaving4,069,575,680B planned headroom. Capacity probes reserve12,889,243,648B
plus their real full-length inputs, outputs and state. The five final-default
maximum-context probes are now running; validation status requires all five
strict reports and successful process exit markers.

All five final capacity probes pass and `memory_plan_validated.log` reports
`validated`. Linear exact-max prefill PCC.999996465; full exact-max prefill
PCC.999999183. Near-max prefill+last-position decode minima are.999970317
linear and.999999166 full; two-kind stack minimum.999929965. Exact state and
changed-input trace gates pass at262143, followed by decode at262144.
`doc/context_contract.json` preserves262144 with no public alignment restriction.
The resource plan uses current multi-reader bank padding and the shared40-core
collective workspace. No full-model construction was needed for the reserved
capacity probe.

Prefill profiler advice about dispatch gaps will also be exercised on this
final TP4 path with an optional caller-owned prefill trace. `$tt-enable-tracing`
was read; stable tensors, exact-signature warmup, state restoration, changed
inputs and repeated replay are required. The ordinary warmed eager-prefill
metric remains the comparable headline.

### Final profiles and boundary/accounting audit

Six final default profiles and three coherent-family profiles pass. Each keeps
separate warmed prefill/decode tables and CSVs for every device, with advice
enabled. `inter_layer_profile_audit.json` verifies four internal AR operations
for the two-kind stack and direct final-residual-add → next-input-norm adjacency
on every rank. There is no layer-boundary gather, reshard or AR.

`performance_accounting.json` reconciles stored-weight+KV lower bounds, each
device's kernel+gap span, and the same signposted trace's synchronized host
latency. Instrumented profiling times are separate from unprofiled headline
medians. `final_matmul_rows.csv` proves BF16/BFP4/LoFi from actual rows.
The installed profiler assumes8 compute workers for all DRAM matmuls, causing
~101% MLP utilization; actual attributes show2/3 readers per bank (16/24
workers). The supplemental CSV corrects this denominator without rewriting
the original tables. This is a concrete tt-perf-report improvement candidate.

Final clang-format changes only native whitespace. Rebuilt `ttnn` and installed
both runtime components again (`final_formatted_native_build/install.log`,
`final_installed_library.sha256`); the expanded18-case native suite passes
again (`native_formatted_mesh_tests.log`). The model source and its context
validation hash are unchanged. The optional prefill-trace runner path preserves
normal eager-prefill/decode behavior and adds input/state bitwise checks.

Current profiles also motivate a bounded GDN comparison at the actual TP4
local12-value-head shape. The monolithic op rejects the default flat rank3
contract by design; `gdn_multichip_probe.py` adapts it to rank4 heads with
explicit L2 normalization before calling the native fused op. These layout
and normalization costs remain inside the measured forward. This test-only
shim does not alter the normal model's TTNN dispatch.

### Supplemental GDN and prefill tracing advice

The adapted monolithic GDN probe passes real-weight PCC (minimum 0.999955773)
but costs 0.551348 ms decode versus approximately 0.422 ms for the phased
implementation. Its rank-4 adaptation and explicit Q/K normalization are inside
the measured forward. `gdn_monolithic_adapted` is rejected on measured latency,
not its initial flat-input API rejection. `gdn_scan_serial` gives 0.424225 ms
decode and 1.332651 ms prefill, with no repeatable decode advantage.
`gdn_prep_serial` at logical 2049 increases prefill to 7.432725 ms versus
approximately 5.205 ms; decode is 0.423534 ms. The final model retains native
phased GDN and the parallel preparation/scan defaults. Exact commands, fresh
kernel-cache environments, source snapshots and exit markers accompany all
three controls.

The optional caller-owned prefill trace passes stable-input, changed-input and
state bitwise checks for both kinds at 128 and 2049, plus B3/S33 and B32/S257
mixed stacks. Eager prefill remains the headline comparison. The first full
short trace run has large host outliers (2.009047 ms median versus a separate
instrumented 0.772044 ms replay); ordinary decode in that process also has
outliers. A fresh AutoDebug source investigation and controlled reruns are
recorded below before this anomaly can close.

### Independent review repair

The first independent xhigh review found four redundant BF16-to-BF16 casts per
short-prefill layer. The selected 1D projection branch now checks the input
dtype before invoking `typecast`, matching the inherited projection path.
This removes actual device operations without changing the configured dtype.
Previous final reports are preserved in `before_review_cast_fix/`.
All 31 model checks, stack stress, full watcher, context probes, default timing,
and final profiles are refreshed after the repair. The final profile checks
must prove the four identity conversions disappeared; prior timing is retained
as before-repair evidence, not used as the final headline.


### Completion-wait AutoFix and selected host mesh setup

`AUTODEBUG_profile_gap.md` derives C++ execution and completion-wait durations
from the saved Tracy host zones. A fresh test-only counter probe then compares
the normal device-bound pool with `TT_MESH_PASS_THROUGH_THREAD_POOL=1`.
`gap_counter_pool_0/1.host_gap.json` retain main-thread timing, context switches,
per-thread runnable wait and explicit measurement limitations. Both pass all
model checks. The default counter run reproduces a 2.211 ms completion wait;
the pass-through maximum is 0.311 ms. The latter profile has device span
0.316655 ms and host interval 0.387580 ms. This closes the old unexplained
accounting tail as a controlled host completion-path effect, without claiming
that snapshots identify the exact origin of every original delay.

Unmodified-runner paired results (normal pool → pass-through, ms):

| Kind | Eager prefill | Traced decode | Queued decode |
| --- | --- | --- | --- |
| Full | 1.165571 → 1.089836 | .305313 → .306680 | .293246 → .293472 |
| Linear | 1.325071 → 1.228877 | .423690 → .421215 | .408693 → .408988 |
| Stack | 2.599765 → 2.376624 | .704711 → .704641 | .691692 → .691328 |

The optimized run/profile launchers now select pass-through by default while
allowing an explicit override for controls. The inter-layer contract records
that callers set it before opening the local TP4 mesh. It does not change
weights, TP ownership, device operations or collective semantics. All final
model, stress, watcher, capacity, profile and headline runs are repeated with
that setup; older metrics are historical evidence only.


### Final selected-path validation and packaging

The final model source is
`211b70db973294f7660ba9cd3c4ef0601ffbac32dab4866e3511b4804ff76213`.
The measured runner is
`22b17036b15d2c85626b4113f397d9f2ddc2274517ba5c9f116ae0b9196fa294`.
Both are unchanged by repository formatting. Native C++ also passed the final
clang-format hook without changes; `final_installed_library.sha256` still
identifies the measured installed extensions.

`run_review_validation.py` completes every step successfully: 31 model cases
in 291.74 s; six optional prefill-trace cases; final default timing; two
100-step eager/queued stress cases; four full-watcher cases plus a B1/S128
prefill-trace stack; five maximum-context probes; validated resource plan;
six current profiles; and eight 100-sample non-aligned/long timing controls.
The final default prefill/decode pairs are 1.248232/.421990 ms linear,
1.177266/.306304 ms full, and 2.347569/.704825 ms stacked. All final strict
PCC, per-user, changed-input, trace/state and process-exit gates pass.

The final profile/accounting refresh uses `review_profile_*`. All eight
layer-kind/device comparisons remove exactly four identity BF16 casts.
Residual-norm selection in the boundary audit explicitly excludes full
attention's two Q/K head norms; all four devices show direct residual-add to
next-input-norm adjacency. Same-window host-minus-device spans range from
0.066389 to 0.080633 ms. Original and new tables are retained separately.

`verify_optimized_multichip_evidence.py` validates 66 final default reports,
current model hashes, empty policy overrides, TP4, selected host pool, strict
PCC/trace/state, watcher environment, capacity and profiler artifacts. The
summary CSV now labels diagnostic counter probes separately and identifies
`after_review_*` as the selected headline source. `final_length_summary.json`
contains the refreshed tail measurements and their source hashes.

Repository pre-commit hooks normalized generated text and supporting Python.
Byte-exact originals and hashes are preserved in `format_originals_manifest.json`;
large raw captures and counter JSON are preserved by `raw_archive_manifest.json`
and compact gzip files. This preserves the original source/report bytes while
keeping checked-in tables lint-clean. The one native test-style issue was
changed to the repository's `expect_error` fixture, retaining the exact expected
exception and message. Its final 18-case TP4/offset-TP2 reader/cache/placement
suite passes in 13.32 s with the selected pass-through runtime
(`native_lint_final.command.json`, `.log`, `.exit_status`).

The final native build/install commands were:

```bash
/usr/local/lib/python3.12/dist-packages/cmake/data/bin/cmake --build build_Release --target ttnn --parallel 4
/usr/local/lib/python3.12/dist-packages/cmake/data/bin/cmake --install build_Release --prefix /home/mvasiljevic/qwen38-full-rerun/tt-metal/build_Release --component ttnn-runtime
/usr/local/lib/python3.12/dist-packages/cmake/data/bin/cmake --install build_Release --prefix /home/mvasiljevic/qwen38-full-rerun/tt-metal/build_Release --component tt_pybinds
```

They passed using the existing toolchain. The prescribed Docker wrapper could
not run because Docker is unavailable; no dependencies were installed. No
native edit followed the successful final build. Final formatting/evidence
checks are recorded in `pre_commit.log` and `final_host_checks.log`.


All final repository hooks pass (`pre_commit.exit_status=0`). The first hook
pass also caught the expected-error fixture style, which was fixed and rerun.
An intermediate log-capture attempt wrote into the hook's own checked file;
final hook output is captured outside the checkout and copied back after exit.
The final host audit passes (`final_host_checks.exit_status=0`), including
accounting regenerated from compact gzip CSVs, all 66 default reports, and both
working/staged `git diff --check`.

Generated CSV files use LF in the committed copy. For every CRLF conversion,
the parsed CSV cells were asserted identical, and byte-exact originals were
saved in the formatting manifest. Valid unified patches retain their original
blank context lines inside gzip files (`patch_archive_manifest.json`). Neither
packaging step changes numerical evidence or implementation semantics.


### Independent stage signoff

The fresh xhigh reviewer returns `clean-pass` in `stage_review.md`, with no
required work remaining. `stage_review_initial.md` preserves the first verdict;
the identity-cast finding and final evidence refresh are closed. The reviewer
independently checked 66 acceptance reports, current source/runtime provenance,
all-rank cast removal and stack adjacency, final timing/accounting, native tests,
watcher, context, and the host-completion anomaly controls.

This completes optimized-multichip-decoder Stage 5. The local checkpoint below
contains only stage-owned changes in tt-metal. No other repository was changed.


### Local checkpoint

- Repository: `/home/mvasiljevic/qwen38-full-rerun/tt-metal`
- Branch: `mvasiljevic/qwen38-full-bringup`
- Reviewed implementation/evidence checkpoint:
  `27a45bac609334393f0ad1af51a61be8d50a7a2e`
- Independent verdict: `clean-pass` in `stage_review.md`.
- The follow-up recording commit logs this SHA and makes the two measured
  shell launchers directly executable; their bytes and measured behavior are
  unchanged. All changes are local. Nothing was pushed.
