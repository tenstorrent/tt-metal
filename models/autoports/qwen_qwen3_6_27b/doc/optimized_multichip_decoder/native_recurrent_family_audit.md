# Native recurrent GDN family: TP4 local 12-head audit

Inspection only, 2026-11-13. Checkout HEAD `8502f42bca20dcad1507fbc8cebf7032220b7d8e`. No TTNN imports, device runs, builds, or production edits were performed. This is an actionable candidate audit, **not a rejection, performance measurement, or acceptance**.

The native operation is applicable to this model. The repository explicitly tests Qwen27B TP4's four key heads and twelve value heads. Its public wrapper has substantial preparation/output movement, and its factory admits at most one head per core. Both have concrete adaptations; neither justifies dismissing the family after an API error. Compare a persistent FP32-state family through the complete local GDN boundary before deciding. Converting the entire state back to BF16 after every step measures a different family.

## Current opportunity and existing implementations

The retained `perf/integrated14_profile_l0_b32_s33/decode_perf_report.txt` identifies op4599, `GenericOpDeviceOperation`, at125us and20.9% of its instrumented decode sum. `metrics.json` pins decoder SHA `4c56cbadb7768054100f921ced79d725a57bfa477beca9648f8f2d932e9bd765`, TP4 `[1,4]`, P300c, two CCL links. These are profiler numbers, not uninstrumented final latency. Parent reports the existing cumulative-exp/column/pair/native-pack experiments give only approximately1% whole-layer gains; this audit does not remeasure them.

Relevant code already present:

- `tt/multichip_decoder.py::_gdn`, `_local12_recurrent`, `_local12_head_partition`: four local Q/K heads, twelve V heads, current BF16 persistent state, FP32 update/readout, local grouped-QK preparation, arbitrary batches through several heads per worker. `_local12_recurrent` already accepts FP32 state and provides a useful same-layout precision control.
- Frozen `tt/optimized_decoder.py::_native_recurrent`: prior single-chip public-op adapter with at most two users/96 heads per wave, external normalization, and concat/copy or sharded `slice_write` state commit. It is useful implementation precedent; copying its48-head geometry or immediate state reconstruction is not a fair TP4 persistent-state endpoint.
- `models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py::fused_recurrent_gated_delta_rule_ttnn`: an actual model wrapper, supporting device-side normalization and the native in-place state-ring interface. The docstring's already-expanded QK is its caller contract; the native public operation itself supports grouped4-to12 heads.
- `tests/ttnn/unit_tests/operations/transformers/gdn_decode_test_utils.py`: `qwen27b_tp4=Shape(4,12)`, K=V128, B1/2/4/8, T1/4/8/12. Existing native tests use PCC>=0.9999 and max absolute error/max reference<=0.01. They note TF32-class FPU operand truncation despite FP32 state/accumulation.
- Sanity and nightly `test_fused_recurrent_gated_delta_rule.py`: accuracy, T-token versus chained single-token identity, repeated determinism, eager/trace equality, and64-step FP64-reference drift. The inspected tests use a single device representing the local head shape, not this stage's four-device layer path. They do not prove B32, this decoder's state ownership, or its512-step quality gate. The ring interface is source-documented; searches of those recurrent test files did not find ring-mode tests.

## Native contracts and exact mapping

Sources below are under `ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/`.

| Boundary | Actual contract and TP4 mapping |
|---|---|
| Public inputs | `q,k=[B,1,4,128]`; `v=[B,1,12,128]`; `g,beta=[B,1,12]`. All are local to each rank. Q/K must already be L2 normalized. `use_qk_l2norm=True` rejects; the message's word “host” means outside the kernel, and does not require a CPU tensor transfer. |
| Scaling/grouping | Public `.cpp` scales normalized Q by128^-0.5, permutes/head-splits, then repeats each Q/K head three times. Value head `hv` consumes key head `floor(hv/3)`, with flattened owner `h=user*12+hv`. Keep heads adjacent in triplets, not cyclic `hv%4`. |
| Gates | Current `_decay(a)` produces log decay `g=-exp(A_log)*softplus(a+dt_bias)` in FP32. Existing sigmoid supplies beta. Public native wrapper computes `exp(g)` once; never pass already-exponentiated decay to that public argument. |
| Primitive inputs | FP32 TILE interleaved DRAM or L1; `q,k,v=[12B,1,128]`, `decay,beta=[12B,1,1]`, state `[12B,128,128]` for T1. Exact shapes validated. Public wrapper converts/splits/repeats into these. |
| State equation | `Sdecay=S*exp(g)`; `delta=beta*(v-k@Sdecay)`; `Snew=Sdecay+k^T@delta`; `out=q_scaled@Snew`. Readout uses updated FP32 state. |
| Primitive output | FP32 `[12B,1,128]` with one padded32-row tile block per head, plus FP32 state. Public wrapper untilizes, reshapes, permutes and tilizes to `[B,1,12,128]`. |
| Existing epilogue | Current recurrence returns compact FP32 `[1,12B,128]`, then BF16 cast/RMSNorm, local affine/gate, local output projection. Native output needs a real row-packing operation to this contract, especially B>1; a metadata reshape across differently padded heads/users is incorrect. |

Current conv decode output is ROW_MAJOR `[B,1,2560]`, with Q512/K512/V1536 contiguous; grouped head rows come from offsets0/512/1024. A correct adapter owns unpacking, normalization, and all tile padding. Normalize raw BF16 Q/K in FP32 with `rsqrt(sum(x*x)+1e-6)`, then scale Q exactly once. Preserve the selected baseline's normalization and gate rounding; do not silently enable the optional BF16 HF-norm graph as a speed shortcut.

The native compute uses FPU matmuls for contractions/rank-one updates. The current custom recurrence uses SFPU FP32 products/reductions. FP32 state does not make their outputs bit-identical. HiFi4 does not establish identical operand precision. Use independent HF/FP64 references, unchanged per-user/output/contribution/state gates, and long drift evidence; exact equality is appropriate for replay versus the candidate's own eager execution.

## Persistent FP32 state is already representable

Public `initial_state_block_idx` enables an in-place ring. For decode T1, the ring has exactly `[12B,128,128]` and the persistent ROW_MAJOR UINT32 index vector is `arange(12B)` (one local index vector replicated across the TP mesh). Set `output_per_token_state=True`, pass that FP32 ring and the index tensor, and the returned state aliases the original ring. Every head reads its complete state before writing that same head; there is no cross-head dependence. T1 avoids a multiple-slot ring entirely.

This is a useful public-API B1/B8 control: no full-state copy or BF16 conversion after the call. Store a `[B,12,128,128]` metadata view in the decoder's public state owner. Own the index tensor explicitly for the same lifetime as the decoder/trace. Never reassign state buffers after capture. Never leave `initial_state=None` in a traced call: the wrapper's zeros fallback is eager-only.

For the coherent FP32 family, `create_recurrent_state` must allocate FP32 once, and `_round_recurrent_state` must not round it via BF16 at public prefill completion. `flat_prefill.py` already retains FP32 across chunks and `flat_gdn_prefill_candidate.py` requires it. With an FP32 published state, preserve that precision through subsequent decode and resumed/nonaligned prefill. This is an explicit precision policy; tests must compare it against HF rather than demand equality to the old BF16 cache bytes. Preserve the unchanged numerical acceptance thresholds.

Also measure the existing `_local12_recurrent` with this same persistent FP32 state. That separates benefits/costs of persistence precision from replacing the recurrence arithmetic. Keep current BF16 custom recurrence as the shipping baseline. A public-native BF16-convert-in/convert-out variant can be a diagnostic, but cannot reject the retained-FP32 family.

## B32 and movement adaptations, ordered from simplest to lower movement

1. **Public operation correctness control.** B1 works as12 heads; B8 works as96 heads on110 cores. Prepare normalized grouped Q/K, raw V, log g and beta entirely on TTNN. Use the T1 ring for persistent FP32 state and include wrapper preparation/output conversion in timing. This establishes native math/ring behavior before broader adapter work.

2. **Public operation B32 wave control.** The factory currently asserts `BH<=grid.x*grid.y`; B32's384 heads exceed110. Use four user-aligned waves9+9+9+5 (108+108+108+60 heads). Allocate each wave's FP32 state/index owner at setup and retain it between calls. Include input wave preparation and output packing. A state slice is not assumed to be an alias: ordinary TTNN slicing can copy. If a unified public-state tensor is required, an additional persistent mirror/commit is a measured control, not the final lower-movement design.

3. **Reuse native kernels with a prepared-input, global-owner descriptor.** The reader/writer runtime ABI contains `h,T,...,BH` and addresses state/output by global `h`; it does not require that `h` equals a physical core index. An opt-in `ttnn.generic_op` adapter can reuse the native FILE_PATH reader/compute/writer kernels and CB plan, dispatch four bounded waves with global head offsets0/108/216/324 into one global FP32 state ring and one global output. Use global `BH=384`, indices0..383, `T=1`, and each wave's actual active core set. This avoids per-wave state slicing/copying and requires no changes to recurrence arithmetic. The generic operation is a test adapter for native kernels, not evidence that the public operation admits B32. Runtime tensor-address rebinding, aliases, output ownership, and all wave offsets must be tested. A minimal native factory wave-capable extension can follow if this is worth keeping.

4. **Fuse preparation and compact output packing.** One device prep kernel can read packed BF16 conv rows plus existing g/beta, normalize each of four Q/K heads once, write three normalized copies to native per-head FP32 blocks, copy V, apply Q scale/exp once, and explicitly zero every unused row/column. Do not run public wrapper on this already-prepared representation: it would expand/scale/exp twice. Invoke the native kernels/primitive descriptor. After recurrence, extract row0 from each head's output into compact `[1,12B,128]` directly. Further folding the BF16 cast/RMSNorm/affine gate into this pack is a separate measured step with its own rounding evidence.

5. **If four dispatch waves dominate**, consider a small native factory/kernel extension that assigns several heads per core. The present compute loop intentionally leaves the final `nxt_S` CB populated; a naïve outer head loop will reuse stale CB state. Pop/reset every live final-state CB and audit reader/writer iteration counts before adding a head-count loop. This is an implementation task, not a reason to reject the existing family. The four-wave descriptor is the safer first measurement.

All routes remain local4QK/12V per rank on the actual TP4 mesh. They do not gather heads, replicate the model, or add a collective. The decoder residual remains compact hidden-sharded BF16 `[1,B,1280]`; the existing output projection/reduce-scatter stays at its normal layer-internal location. No inter-layer collective is needed.

## Capacity, traffic, and context implications

The current BF16 state is `B*12*128*128*2=393216*B` bytes/device/layer. FP32 is `786432*B`, an increase of393216*B. Shapes have no partial K/V tiles and page counts are divisible by eight, so these persistent state payloads also match the eight-bank allocated page spread (allocator metadata excluded).

| State scope | Current BF16 | Retained FP32 | Increase |
|---|---:|---:|---:|
| One GDN layer, B1 |393216 B|786432 B|393216 B|
| All48 GDN layers, B1 |18874368 B|37748736 B|18874368 B (18MiB)|
| One GDN layer, B32 |12582912 B|25165824 B|12582912 B (12MiB)|
| All48 GDN layers, B32 |603979776 B|1207959552 B|603979776 B (576MiB)|

Ring indices add a small but nonzero allocation: one ROW_MAJOR page of4*12B bytes, aligned64 and spread over eight DRAM banks; B1 approximately512B/device and B32 approximately12288B/device, excluding metadata. If `from_torch`/mapper adds other physical padding, inventory the actual buffer size. Avoid retaining both BF16 and FP32 full states by accident; a mirror doubles the family delta and must be included explicitly.

The standard native prepared buffers are much larger than their logical values. At B32, Q/K/V each have384*4 FP32 tiles (6MiB each); decay/beta each have384 tiles (1.5MiB each); output has6MiB. Together these are27MiB of prepared inputs/output per layer signature before state and wrapper temporaries. Current compact FP32 output `[1,384,128]` is192KiB; native head-padded output is6MiB. This is the reason to measure direct preparation/packing, not an assumed loss. Preallocated scratch is still resident memory and trace owners may retain it across layers.

For Kt=Vt4,T1, the native CB plan totals130 FP32 tiles/core =532480B/core, plus a small ring-index CB. That is source-level feasibility, not measured L1 placement or peak fragmentation. Local decode traffic doubles the state read/write component with FP32 persistence; native contraction speed must compensate. Read-once state traffic rises from786432*B to1572864*B bytes/device/step/layer.

KV-cache dtype/layout and sequence capacity do not inherently change. The FP32 increment is independent of sequence length. Preserve the supported262144 B1 context. If selected, update `doc/context_contract.json` with recurrent dtype, per-device inventory, any persistent preparation buffers and appropriate context evidence. Existing 32K stage5 smoke plus allocation/reservation evidence must be rerun for the selected policy; if the changed inventory calls for maximum-capacity validation, run it. Neither this arithmetic nor a temporary allocation failure supports lowering the public context contract. B32 short-context coverage is not a promise of32 simultaneous262144-token contexts.

## Required evidence before keeping or materially rejecting

Run hardware only through the parent's serialized device workflow, watcher separate from profiler. Suggested matrix:

- Focused public-native B1/B8 T1 versus an independent FP64 recurrence and the current custom BF16/FP32 variants. Actual rank-distinct activations and state; all four ranks, all12 heads, per-user metrics. Include zero Q/K/V/state, extreme finite log decay/gates, and NaN-poisoned unused tile rows; adapters must own masking, not require pre-zeroed external padding.
- Ring alias/persistent state verification, changed inputs on every replay, alternating fresh owners, cache hits and address rebinding. At least50 queued replays; retain explicit state/output checkpoints. Native existing fixed-input trace tests do not prove this caller's changing-state contract.
- B32 public waves and global-owner prepared descriptor, plus B3/17/31 for partial waves and head-group boundaries. For native-kernel descriptor routing, poison inactive heads and guard adjacent state slabs to catch incorrect global offsets. Include full batches with distinct user states so one user's valid output cannot hide a misplaced wave.
- Whole decoder actual-weight TP4 B1/B32, every current canonical prefill/decode shape; prefill->decode, resumed/nonaligned prefill, list states, and multi-layer owner/trace stress. Keep full-attention controls unchanged but rerun shared gates when production state plumbing changes. Match the original512-step B32 and128-step B1 risks with fresh independent reference sequences.
- Measure warmed complete GDN boundary from packed projection/conv through preparation, recurrence, output packing, normalization/gate and output projection; then warmed traced whole decoder. Record raw op tables/CSV and allocation peaks. Pair current BF16 custom, persistent FP32 custom, public-native persistent FP32, and prepared native-kernel persistent FP32. Attribute normalization, expansion, exp, state copy, output conversion and wave dispatch separately. No immediate BF16-state restoration may be hidden outside the timer.

If the first public call fails at head count, layout, or shape, apply the bounded wave/prepared layouts above and retry. If the native arithmetic fails long-run quality, first distinguish normalization/padding/group indexing, state rounding, TF32 FPU precision and trace alias bugs; adapt or use AutoDebug/AutoFix before rejecting. If all quality gates pass but the coherent retained-FP32 family loses whole-decoder latency, preserve that measured before/after rejection and its exact source/policy. This audit leaves performance unresolved.

## Inspected source fingerprints

- `models/autoports/qwen_qwen3_6_27b/tt/multichip_decoder.py`: `4c56cbadb7768054100f921ced79d725a57bfa477beca9648f8f2d932e9bd765`
- `models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py`: `315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22`
- `models/autoports/qwen_qwen3_6_27b/tt/flat_prefill.py`: `da2d03d17e3c15461e3d8c78cbb72ad155a0d5c2d77699a6cb308cd39847ed26`
- `ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/fused_recurrent_gated_delta_rule.cpp`: `ca4059c5c2021c03069c471c4bcce2071408243e9fd2d5682dc47af13cca4964`
- `ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/fused_recurrent_gated_delta_rule_program_factory.cpp`: `d404654b8160297504b6f769a74c05ed596c555111b354d312f26f5b3fb2589d`

## Isolated public-native probe prepared

Added only `tt/native_recurrent_candidate.py` and `tests/probe_native_recurrent.py`; production code/defaults remain unchanged. This first adapter implements public-native user-aligned waves, owns copied FP32 T1 ring/index tensors, and retains them through every update. It exposes wave tensors for probe inspection; production public-state/list ownership integration is not implemented. Native normalization, grouped-head expansion, output conversion and cross-wave row packing are all inside the measured boundary. It does not immediately restore the BF16 state contract.

The probe includes custom BF16-state and custom FP32-state controls; FP64 host recurrence with explicit BF16/FP32 persistence rounding; all-rank/per-user >=0.995 output/state checks; comparison to both controls; changed-input eager and traced recurrence; fresh-owner/rebound-address cache control with the old owner kept live; optional50changed-input queued traced steps; and alternating-order synchronized trace timings of the full preparation/recurrence/output boundary. It defaults to production FP32 log-decay and BF16 beta; `--g-dtype bfloat16` explicitly tests the other gate precision. Synthetic inputs have the actual local four-QK/twelve-V packed ROW_MAJOR layout. Actual-weight whole-decoder gates and final profile evidence are still required before selection.

CPU-only preparation checks passed: Python compilation/AST, wave coverage for every B1..32, maximum108heads per wave, B32split9+9+9+5, and persistent-state memory formulas. Neither module was imported with TTNN/torch by the author. No native operation or device evidence is claimed by this preparation.

Parent commands (serialize with other device work):

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  bash "$D/run_job.sh" native_recurrent_public_b1_smoke \
  python -m models.autoports.qwen_qwen3_6_27b.tests.probe_native_recurrent \
  --batches 1 --steps 2 --bank-size 2 --warmup 1 --queued 0 --skip-timing \
  --output "$D/native_recurrent_public_b1_smoke.json"

TT_METAL_WATCHER=10 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  bash "$D/run_job.sh" native_recurrent_public_waves_watch \
  python -m models.autoports.qwen_qwen3_6_27b.tests.probe_native_recurrent \
  --batches 1 3 9 10 17 32 --steps 4 --queued 50 --skip-timing \
  --output "$D/native_recurrent_public_waves_watch.json"

TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  bash "$D/run_job.sh" native_recurrent_public_boundary_perf \
  python -m models.autoports.qwen_qwen3_6_27b.tests.probe_native_recurrent \
  --batches 1 32 --steps 4 --queued 50 --samples 5 --replays 10 \
  --output "$D/native_recurrent_public_boundary_perf.json"
```

Initial hardware compilation/API/layout behavior is unverified. The scripted public wrapper retains all its material layout costs; a loss here is a control measurement, not a rejection of the prepared-input/global-state native-kernel family described above. The full32batch wave case may need a larger bounded timeout; preserve evidence and adapt failures through the normal device workflow. The `--cpu-only` option produces a standalone32-batch plan artifact without importing torch or TTNN.

Prepared adapter fingerprints:

- `tt/native_recurrent_candidate.py`: `25d08e3042f3756cfb2407906d6052e041c6e7502785953765c0c3081e75c38e`
- `tests/probe_native_recurrent.py`: `0ca3640f0623ba49752d21b86c7e2969e82ce2fa5d99c0dc0fb2129dda1ef03f`
