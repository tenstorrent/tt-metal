# AutoDebug: fused whole-trace protocol rejects a supported GR projection

## Findings

**Confirmed cause: the profiling harness assumes the functional projector's `block` attribute exists on `FusedPreciseProjection`.** `tests/profile_trace_protocol.py:27` evaluates `down.block` before its shape check can finish. The fused class deliberately represents grouped weights with `parent_block`, `prefill_block`, and each weight's actual row count; it neither defines nor inherits `block`. The error is a stage-test contract mismatch, not evidence of a failed device operation.

**The independent all-worker coverage argument remains valid for the actual frozen fused implementation, including prefill logical lengths 1–32 and decode length 1.** Its current explanatory text is inaccurate for one-token calls, and its source manifest omits the fused implementation. Fix those alongside the predicate. No model or BinaryNg implementation change is needed for this failure.

The minimum intervention is confined to `tests/profile_trace_protocol.py`: recognize the two audited projection representations, validate their real weight geometry/layout, describe both product-shape cases, and hash the sources that establish the proof. Do not add a compatibility `block` attribute, silently substitute `getattr(..., 32)`, or obtain participating cores from observed profiler endpoints.

## Scope and direct observations

Repository: `/home/ttuser/qwen38-flash/tt-metal`; inspected HEAD: `70dc6f5dd24298741e2f108a281160677e986155`. Model paths below abbreviated `M/` mean `models/autoports/qwen_qwen3_8_flash_next/` relative to that root. The fused runtime is an untracked working-tree file, so HEAD alone does not identify it.

* `M/doc/fused_decoder/logs/final_layer0_prefill_final_v5.log:1202–1215` records the exact stack: selected probe → `profile_trace_decoder.py:155` → `prepare_protocol` → missing `block` at line 27. Lines 1218–1224 record completed device close/destruction and process exit 1.
* `M/doc/fused_decoder/tracy/final_layer0_prefill_final_v5/profile.json` records `passed=false`, the same error, and an empty replay-call list. It has no capture-begin identifier, trace ID, or completed measurement protocol. The selection sidecar records `FusedDecoder`, residual class `GatedResidual`, `completed=false`, and the fused-source hash below.
* The same stage's baseline profile records `passed=true`, prefill length 32, and a pre-capture worker grid `[11,10]`. This is a useful functional/fused contrast. The failed fused process never reached inventory creation; the baseline's grid is used below only as an illustrative recorded configuration, not as a substitute for querying the next fused run's mesh.
* SHA256 of the current `M/tt/fused_decoder.py` is **`06df08406a50a4608210cb00b99af745f29281905a9144feda928e287a09f1bd`**, matching the supplied frozen value and failure selection sidecar. The profile's probe hash also matches the inspected `profile_trace_decoder.py`.

Only source/log/artifact reads, hashes, AST inspection, and standalone integer arithmetic were performed. No TTNN import, test invocation, device operation, reset, or implementation edit was performed. This report is the sole written artifact.

## Causal chain and completion boundaries

1. `M/tests/run_fused_probe.py:234–241,282–285` substitutes the selected factory before importing the profiling module. Final policy `{}` uses `FusedDecoder.from_state_dict` defaults, not the baseline projector. `fused_decoder.py:2394–2409` replaces both GR down/up projectors independently of residual-norm fusion; therefore a residual still named `GatedResidual` correctly contains a fused projector.
2. `M/tt/precise_residual_projection.py:25–42` defines functional GR `block=32` for K=10240. `M/tt/fused_decoder.py:26–75` instead overrides initialization, copies weight metadata, and optionally groups weights. Its base is `PreciseGDNProjection`, whose implementation at `precise_gdn_projection.py:25–35` also has no `block`. Static AST inspection confirms no `self.block` assignment in fused initialization.
3. `prepare_protocol` reads the loaded-library layout, then unconditionally accesses `down.block`. The tuple construction raises before the intended `ValueError`, worker inventory, source hashes, or protocol return.
4. In `profile_trace_decoder.py`, this is after model/state setup at lines 137–145, but before prompt creation at 161, eager prefill at 210, trace capture at 227, or either replay. Setup kernels may already have executed; that does not make this a measured decoder replay.
5. Normal successful execution uses blocking warm replay at 258; measured replay at 288 is nonblocking followed by explicit synchronization at 289. Predrain sampling follows completion, and profiler drain follows sampling. None of these steps is reached by this failure. The `finally` block at 315–327 closes the mesh, consistent with the log. No device-hang or reset diagnosis is warranted.

This predicts the observed boundary precisely: the functional projector passes attribute access, while any selected fused projector fails it before measurement, regardless of layer, mode, or logical sequence length.

## Actual model shapes, including grouped weights

The public inherited call path is `FunctionalDecoder.prefill_forward` / `decode_forward` → `_chunk` → `attn_mix.read` → `precise_down`. See `M/tt/functional_decoder.py:123–190` and `decoder_components.py:77–84`. Every supported layer calls this GR down projection; attention kind and PLE processing do not bypass it. One occurrence is sufficient to establish participation; the later MLP GR is not needed for the proof.

For logical prefill Q=1..32, the public method passes Q unchanged into `_chunk` (lines 157–159). Decode requires `[1,1,1,10240]` (188–190). TILE padding of the hidden-state token dimension does **not** change `value.shape[-2]` to 32. GR flattens its normalized streams to logical `[1,1,Q,10240]`.

Original GR down is a `[320,10240]` checkpoint matrix, uploaded as ten FP32 TILE DRAM tensors `[1,1,32,10240]` (`precise_residual_projection.py:28–42`; `decoder_components.py:45–52`). Frozen fused defaults are `direct_packed_weights=True`, `projection_decode_group=64`, and `residual_decode_group=None` (`fused_decoder.py:2274–2277`). Thus:

* Fused initialization records `input_width=10240`, `output_width=320`, `parent_block=32`, `chunk_prefill=False`, and `prefill_block=32` **before** grouping (lines 38–43).
* The requested upload group is `32*64=2048` output rows. Slicing the actual 320-row checkpoint yields **one** tensor `[1,1,320,10240]`, not a padded 2048-row tensor (lines 44–65). Its dtype remains the original FP32. The non-host concat alternative also groups the ten existing blocks into 320 rows.
* `__call__` obtains Q from the logical input, casts input to FP32 and reshapes it to `[1,Q,1,10240]` (77–81). For Q=1 it uses `parent.shape[-2]`, which is now 320; for Q>1 it uses `prefill_block=32` and iterates ten weight slices (83–95).

| Public case | Q seen by projector | Weight per multiply | Product logical and padded shape | Tiles per multiply | Multiplies per GR down |
|---|---:|---|---|---:|---:|
| Functional prefill Q=1..32 or decode Q=1 | Q | `[1,1,32,10240]` | `[1,Q,32,10240]` | `320*Q` | 10 |
| Frozen fused prefill Q=1 | 1 | `[1,1,320,10240]` | `[1,1,320,10240]` | 3200 | 1 |
| Frozen fused prefill Q=2..32 | Q | `[1,1,32,10240]` slice | `[1,Q,32,10240]` | `320*Q` = 640..10240 | 10 |
| Frozen fused decode | 1 | `[1,1,320,10240]` | `[1,1,320,10240]` | 3200 | 1 |

Expanded A has logical `[1,Q,1,10240]`, padded `[1,Q,32,10240]`. Q is an outer batch/channel extent in the product; the last two dimensions, B and 10240, are the tile axes. For general positive block B, physical product tile count is

`T = (1 * Q * ceil(B/32)*32 * 10240) / (32*32) = 320 * Q * ceil(B/32)`.

For the validated grouped family, each stored row count R is a positive multiple of 32, their sum is 320, Q=1 uses B=R, and Q>1 uses B=32. Therefore every such multiply has T>=320. The **actual final configuration's** minimum over all named cases is 640 (Q=2), so retaining the existing 320-worker cap is conservative. These are per-operation counts; the proof does not add tiles from different kernel invocations to reach the grid size.

## Lowering into BinaryNg and exact work assignment

The traced operation is the multiply at `M/tt/fused_decoder.py:95`, before either reduction strategy. `balanced_tree` and `widen_in_unpack` do not alter this coverage result for GR: weights already have FP32 dtype, and the product precedes the reduction.

1. `ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp:1390–1416` routes tensor multiply through `invoke_binary_ng`, then `invoke_binary_ng_impl` at 654. Its dispatch at 809–824 and 849–852 preserves operand order. The older volume-swapping `preprocess_inputs` helper is not called by this route. Actual A is expanded input `[1,Q,1,10240]`; B is weight `[1,1,B,10240]`.
2. Both inputs are FP32, TILE, interleaved DRAM. There are no input/post activations, explicit output, explicit memory configuration, or sub-core/sub-device overrides in this call. `binary.cpp:783` defaults output dtype to A's FP32. `binary_ng_device_operation.cpp:682–688` selects TILE and A's memory configuration. Logical output broadcasting gives `[1,Q,B,10240]`; BinaryNg validates 32x32 tiles at 480–495.
3. `binary_ng_device_operation.cpp:217–246` chooses **ROW_A**, since A's logical H=1, B's H=32 or 320, and both W=10240. The factory chooses `kernels_ng/dataflow/reader_interleaved_row_bcast.cpp` with `SRC_BCAST=1`, `SRC_BCAST_B=0`, and `kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` (factory 231–240,1294–1317). FP32 multiply uses SFPU; the Blackhole FP32 ROW guard disables LLK broadcast at 1456–1466, leaving `kernels/compute/eltwise_binary_sfpu_no_bcast.cpp`. The ROW_B-only column-group optimization is not selected; runtime `col_group=0` (677–686).
4. `get_worker_grid` in `binary_ng_device_operation.cpp:131–214` uses the first active sub-device's TENSIX worker set for this interleaved call without overrides. The profiling fixture opens an ordinary 1x1 mesh (`profile_trace_decoder.py:129`) and does not install another manager. `tt_metal/distributed/mesh_device.cpp:1854–1857` constructs its single default sub-device as the full rectangle `[0,0]..[grid.x-1,grid.y-1]`; tracker initialization at `tt_metal/impl/sub_device/sub_device_manager_tracker.cpp:38–45` activates it. `mesh_device.cpp:1624–1629` returns that set. Thus the protocol's pre-capture logical enumeration matches this factory's eligible workers, under the fixture's default-manager invariant.
5. In `binary_ng_program_factory.cpp:492–509`, all operands are unsharded, so `grid={}`, `rt_has_sharding=false`, `row_major=true`, and `zero_start_grid=false`. **The actual branch is line 643, `interleaved_split(all_device_cores,T,true)`, not the rectangular overload at line 639.** Tile count is `c.physical_volume()/tile_hw` at 594. The cached split keys on grid, tile count and traversal order (439–454) and calls the **CoreRangeSet overload** of `split_work_to_cores`.
6. That overload in `tt_metal/common/work_split.cpp:407–490` takes all G eligible cores when T>=G (416–418). Let `q=T//G` and `r=T%G`: the first r row-major cores receive q+1 tiles and the remaining G-r receive q; for r=0 the single group contains all cores. Since `T>=320>=G>0`, every core has positive work. The factory transfers those assignments to reader/writer/compute runtime arguments at 705–803 and 876–910, advancing the start tile at 912. There is no minimum block-granularity condition that reduces participating cores here.

For the baseline's recorded 11x10 grid (G=110), concrete assignments are:

| Product | T | Per-core assignment | Last core's start/count |
|---|---:|---|---|
| Functional Q=1 lower-bound case | 320 | 100 cores ×3, 10 ×2 | 318 / 2 |
| Frozen fused prefill Q=2 | 640 | 90 cores ×6, 20 ×5 | 635 / 5 |
| Frozen fused Q=1, including decode | 3200 | 10 cores ×30, 100 ×29 | 3171 / 29 |
| Frozen fused prefill Q=32 | 10240 | 10 cores ×94, 100 ×93 | 10147 / 93 |

This proves coverage from the model and allocator/scheduler contracts. It does not infer the inventory from CSV rows, kernel compile messages, or endpoints that happened to be emitted.

## Minimal stage-test-only adaptation

Keep `prepare_protocol(mesh, model)` and its timing/inventory lifecycle. Replace only the assumption about projection representation, update its proof/metadata, and extend its source list. No new model operation is necessary.

Use exact recognized implementation classes, not an arbitrary object that happens to expose similar attributes. The following are the proposed predicates (to be implemented later; no patch was applied):

* Common: `down.input_width == 10240`, `down.output_width == 320`, nonempty weights; for every weight j, `shape == padded_shape == (1,1,R_j,10240)`, `R_j >= 32`, `R_j % 32 == 0`, FP32 dtype, TILE layout, interleaved DRAM memory config, and residency on the profiling mesh; `sum(R_j) == 320`.
* `type(down) is PreciseResidualProjection`: additionally require `down.block == 32` and row counts exactly `[32]*10`.
* `type(down) is FusedPreciseProjection`: additionally require `down.parent_block == 32`, `down.prefill_block == 32`, and `down.chunk_prefill is False`. The common predicates allow only complete tile-row partitions of the actual matrix, so prefill slicing has no partial block. For a guard deliberately limited to **this frozen final configuration**, further require row counts exactly `[320]`; this is the smallest strict final-stage admission rule. The broader partition rule is also proven and supports other grouped variants without inventing `block`.
* Unknown projection types or failed predicates must raise a descriptive `ValueError`. Do not accept an unknown implementation solely because its dimensions match. The baseline branch retains support for the stage's paired functional measurements.
* Retain the existing single-device, Blackhole, loaded-library-layout and `0 < grid.x*grid.y <= 320` checks. Preserve the default full-grid manager/no-sub-grid fixture contract. If this helper is later reused with custom sub-devices, require equality between the actual selected worker set and the inventory; grid dimensions alone are then insufficient.

The logical input family is already guarded by `profile_trace_decoder.py:50–51` and the public decoder methods: prefill Q=1..32 and decode Q=1 with shape `[1,1,Q,10240]`. The harness upload helper and calls at `profile_trace_decoder.py:158–162` establish device residency, BF16 input dtype and TILE layout; the public methods themselves check logical shapes/counts. Do not perform a forward call or generate synthetic tensors inside `prepare_protocol` to discover shape or coverage. The source trace plus weight metadata supplies the proof before any capture.

Suggested replacement proof text:

> Each public decoder pass invokes GR down with input width 10240 and output width 320. For logical Q=1..32, the functional projector multiplies FP32 [1,Q,1,10240] by [1,1,32,10240]. The fused projector uses each grouped R-row weight for Q=1 and 32-row slices for Q>1; R is a positive multiple of 32. Product [1,Q,B,10240] has 320*Q*(B/32) physical 32x32 tiles, at least 320. The interleaved BinaryNg CoreRangeSet split gives nonzero work to every worker in the default full grid of at most 320 cores. This inventory is established before capture independently of observed endpoints.

Record the recognized projection class, actual logical/padded weight shapes and row counts, `prefill_block` where applicable, input family, and lower-bound tile count alongside `coverage_proof`. This makes the branch and grouped representation reviewable. For frozen final `[320]`, recording 320 as the shared conservative bound is valid; recording an additional exact final minimum of 640 is also valid.

Continue constructing `known_cores` by enumerating the mesh's logical grid and calling `worker_core_from_logical_core` before capture. Leave `sample_predrain`, replay synchronization, event ordering and native CSV checks intact. `report_trace_performance.py:165–217` already checks the entire independent inventory and requires completed, non-dropped markers on every known core. No acceptance threshold or inventory relaxation follows from this diagnosis.

## Coverage-source identity

The current manifest at `profile_trace_protocol.py:40–46` hashes only the original projection, `decoder_components.py`, BinaryNg program factory, work split and the protocol itself. In particular it **does not bind `fused_decoder.py`**, which owns the actual weight grouping and forward loop. Adding a fake attribute would leave both this gap and the incorrect Q=1 shape statement in place.

At minimum retain those existing entries and add the fused implementation, inherited public decoder, selected probe/factory, binary wrapper and BinaryNg device operation. For the default-grid assertion include mesh construction and manager activation as well. The following source identities were measured from the inspected tree; paths are repository-relative. `M/` has the exact expansion given above.

| Source | SHA256 |
|---|---|
| `M/tt/fused_decoder.py` | `06df08406a50a4608210cb00b99af745f29281905a9144feda928e287a09f1bd` |
| `M/tt/functional_decoder.py` | `e775e5fd6a37ebaca8faadd9d4f5414b82e44e4b57e8265fae331edec94fb2d1` |
| `M/tt/decoder_components.py` | `7f2b95f6998237e9c5208542bc2d2147bf0899b2438d175b1aa3ff2bbaf4ad15` |
| `M/tt/precise_residual_projection.py` | `8af1023e7dcbfae6523769fbfa7245348f4bf022d4b303309417649a2e47046a` |
| `M/tests/profile_trace_protocol.py` — before repair | `cf5ed0cdf2f5bfc44e8031c7d0b6120fc34318291d04d0e7baf28f3834a948dc` |
| `M/tests/profile_trace_decoder.py` | `7e0fb31efe3a10fef59d4677108778162b429ec2622086a42c4f78aaf8c1d8a4` |
| `M/tests/run_fused_probe.py` | `22e553499ea1abfa19affe916db651d4163b5f02a7ffb8232fc6137ecb5638d4` |
| `ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp` | `b38be1797bd0e3f01fad7c93c7c60be5e8061d7b7a768e3700e94177d752434e` |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp` | `36aadfba5ef9400d72f135d72116ed6ab86a0f36f4c62cfb97c8a57c72a61f92` |
| `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp` | `d5f7392c050bfc02efc17850d040f1e31cb7c0235a96b87717edb3ba1d0fca4e` |
| `tt_metal/common/work_split.cpp` | `fbff403fcfd013f5147beade952de7bc196f2a2d3207848e6c57d189235108a8` |
| `tt_metal/distributed/mesh_device.cpp` | `2b5dbcd487255625e6e57df8fe576b6ece8f545425d27f8260ec3fd86d80f869` |
| `tt_metal/impl/sub_device/sub_device_manager_tracker.cpp` | `5ddf47d0bc234f49f34dad9b8a1199c4c5326f5c3e87f3fec27cc8fcd617c79c` |

For reproducibility of the precise kernel-selection explanation, also bind these inspected files (same repository root):

| Source under `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/` | SHA256 |
|---|---|
| `binary_ng_utils.hpp` | `b79dd0ab4aa358cfe8af9c7f1ff3e07cc21301b82f38a08474a8580221028547` |
| `binary_ng_utils.cpp` | `6f1f51327ed80bceecc2f5ed7877b3f108ae01075932a24bcf9a621ec2753597` |
| `kernels_ng/dataflow/reader_interleaved_row_bcast.cpp` | `b79f6607f2f2e3829e60b8546ed3cd961f74889bf0f272f04ff315cd061aa7a6` |
| `kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` | `8d8effcfcfd2ccb5210ce67cabb3d8ad9452459d0e04a6de75785ff9cbb55e2b` |
| `kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` | `194922145ae8a4eb244ae5343e0d4dc1ea05dedd1dab28a48fdacf5edc7b0736` |

Generate `coverage_source_sha256` from actual file bytes during preparation, using full repository-relative paths, as the existing implementation does. After repair the protocol's own hash necessarily changes; do not copy the pre-repair hash above into a new artifact. The fused runtime hash must remain unchanged for this stage-test-only intervention. These source hashes establish source provenance, not by themselves correspondence between every compiled shared library and those sources; retain the loaded-library/layout identity and build provenance too.

## Other potential issues and limits

* **Custom sub-device reuse is outside the current proof.** An interleaved operation selects the first active sub-device, not necessarily every core in a physical chip. The present fixture uses the default full rectangle, so this is not an explanation of the reported failure. A future custom-manager caller must supply/check the matching worker set.
* **The original statement's Q=1 shape and missing fused hash are evidence defects, not observed device faults.** Updating only the attribute access would allow profiling to proceed but leave those defects. Nothing inspected shows that grouping invalidates all-worker coverage.
* **Existing archived evidence has its original source identity.** `validate_trace_evidence.py:139–142` compares recorded coverage hashes to files. Editing the shared protocol changes that identity, so preserve the old baseline's source snapshot/provenance or rerun paired measurements under the updated protocol. Do not rewrite old artifacts' hashes to make them look freshly captured.
* **Runtime timing and correctness are unverified here.** Static proof establishes scheduled worker participation if the audited call executes. It cannot certify trace completion, endpoint integrity, memory sufficiency, numerical acceptance, or latency of a future run. The existing synchronization, exact eager/replay output/state comparisons and strict known-core checks remain the follow-up acceptance gates. A file existing in the failed run's profiler directory is not a valid measured replay.

## Claim review

The first draft was checked against the code, including independent shape and BinaryNg review, for the tempting alternative explanations: the final residual class name does not imply a functional projector; a group size of 64 does not produce 2048 actual rows; logical prefill length is not rounded to 32 at the projection; multiply does not take the unused legacy operand-swap helper; and this interleaved factory takes the CoreRangeSet work split rather than the rectangular overload. None of those alternatives changes the diagnosed exception or defeats the coverage proof. Review clarified that the harness upload, rather than the public shape checks, establishes device residency and TILE layout. All 18 source hashes in the report were rechecked against file bytes.

Standalone arithmetic enumerated Q=1..32 and every grid size G=1..320 using the derived final tile formula: the smallest assigned share is two tiles, at Q=2,G=320. The functional shared lower bound still gives at least one. This is a static calculation, not a hardware or test-suite result. The headline finding is the source-proven stage-test contract mismatch; potential runtime problems remain explicitly unclaimed.
