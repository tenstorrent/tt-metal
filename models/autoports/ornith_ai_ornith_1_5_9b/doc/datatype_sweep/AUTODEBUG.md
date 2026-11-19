# AutoDebug: BF16 prefill activation L1 collision

Investigation date: 2026-11-13. Inspection only: no target execution, hardware access, source changes, or AutoDebug relaunch. This report is the only artifact written by the investigation.

Paths beginning `tt/`, `tests/`, or `doc/` below are relative to `models/autoports/ornith_ai_ornith_1_5_9b/`. Other paths are relative to the tt-metal checkout. `minimal/` abbreviates `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/`.

## Finding

**The fixed prefill placement policy retains an 8192-row BF16 normalized hidden tensor in L1 while staging the BF16 attention-output projection input in the same L1 arena. Together with the existing collective buffers, these leave less room than the explicitly configured minimal matmul requires.** The complete tensor ledger reproduces the reported occupied L1 frontier exactly; the DFB payload calculation and an inferred static base account for the **142,144-byte overlap**.

The earliest incompatible model decision is keeping the normalized hidden tensor in L1 through 8192 rows without accounting for its selected dtype and its lifetime through the output projection. The final allocator guard correctly rejects the resulting overlap. This is not evidence of incorrect mixed-precision math in minimal matmul.

**The all-BF16/HiFi4 reference needs a broader intervention:** its BF16 weights enlarge the static buffers, and its BF16 collective scratch also grows. The same source calculation predicts an output-projection collision already at physical bucket **4096**. An 8192-only repair cannot cover that reference.

## Direct observations and provenance

- `doc/datatype_sweep/bfp8_prefill_act_bf16.log:55–67` logs preparation of layer 0, linear attention, slot 0, context 262144. The last entry is `bucket=8192 valid=8192`; the preceding entry is bucket 4096. The exception follows immediately.
- The stack at log lines 81–105 is `PrefillRuntime._warm → _execute → NormalizedLinearPrefill.forward → attention`, including the pooled-state recursive attention wrapper, then `d.linear(out, "linear_attn.out_proj") → minimal_matmul`.
- The exception is from `ProgramImpl::validate_dataflow_buffer_region`: program 1016, static range `[0,0]..[10,9]`, lowest occupied compute L1 address **773248**, static region end **915392**. These are dataflow buffers (DFBs) on the Metal 2 path; “CB” is convenient shorthand, but the legacy circular-buffer factory is not the selected factory.
- The JSON records `runtime_error`, context 262144, BF16 large-prefill normalized activations, BFP8 weights, BF16 output/residual, FP32 destination accumulation, and BFP8 CCL/KV. `references` is empty: the failure occurs during preparation, before scoring.
- Checkout HEAD is `f8630a5a52013134e41b38a97352a7fed9d691c2`, matching the failing JSON. At the initial inspection, all **22 recorded Python source hashes** matched. A final recheck found an external concurrent change only to `tt/cache_contract.py` (recorded `fa630ee4…`, then `16daacec…`). The other 21 files still match, including `prefill_ops.py`, `prefill_runtime.py`, `prefill_plan.py`, and `multichip_decoder.py`; the causal analysis uses those unchanged files. The investigation made no source edits. The JSON does not separately fingerprint the built C++ libraries; C++ analysis uses the prepared source, whose DFB path and sizes agree with the stack and error.
- `bfp8_mlp_lofi.json` records a pass with the same requested policy except `prefill_norm_large=bfloat8_b` and the ID. Its hashes differ in `tt/multichip_decoder.py`, `tt/generator.py`, and `tt/model.py`; therefore it is supporting passing evidence, not an exact same-source controlled run. The current source itself establishes the relevant dtype/placement contrast.

Advancing through earlier warmup entries establishes that no host allocation exception stopped those calls. It does not independently establish their numerical correctness or completion: `_warm` synchronizes only at `tt/prefill_runtime.py:372`. This particular exception is nevertheless synchronous host compilation/allocation validation, not a deferred kernel error attributed from a later synchronization.

## Exact physical profile and lowered projection

`PrefillRuntime.prepare` prepares every catalogue profile through the context-dependent maximum (`tt/prefill_runtime.py:61–72`); `_warm` prints each height and executes a tensor of that height (`:345–359`). Consequently, **8192 is the physical height actually consumed by this operation**, not a request-length estimate. Preparation reaches it even when the first scored prompt is short.

At the failing projection:

| Property | Effective value |
|---|---|
| Local activation before staging | `[1,1,8192,1024]`, BF16, tiled, DRAM |
| Local weight | `[1024,4096]`, BFP8, tiled, DRAM |
| Activation passed to minimal matmul | Same shape/dtype, interleaved L1 |
| Output | `[1,1,8192,4096]`, BF16, tiled, DRAM |
| Fusion/bias/chunks | None / none / 1 |
| Block sizes, in tiles | M=4, K=16, N=8 |
| Subblock/grid | 1×2 / 11×10 |
| Compute | HiFi2, FP32 destination, packer L1 accumulation |

Evidence: output preparation and gating at `tt/prefill_ops.py:149–153`; weight partitioning at `tt/multichip_decoder.py:315–330`; explicit config at `:394–401`; placement/output arguments at `:676–687`; compute policy at `:176–183`. The output projection activation is BF16 in **both** policies: changing the large normalized activation dtype principally changes a different, still-live tensor.

The C++ wrapper preserves activation/weight order and sends one output to `prim::minimal_matmul` (`minimal/minimal_matmul.cpp:24–54`). Compute config is resolved against the architecture (`minimal/device/minimal_matmul_device_operation.cpp:465–493`). The selected `ProgramFactory` builds `minimal_matmul_program_descriptor.cpp`, not the legacy CCL-only factory (`minimal/device/minimal_matmul_program_factory.hpp:14–20`). It selects each operand's data format separately (`descriptor:289–305`).

Instantiated work partition: M/K/N tiles = **256/32/128**. Since M>N, M spans the 11-core axis and N the 10-core axis (`descriptor:373–414`). Rounded dimensions are 264 M tiles and 130 N tiles: **24 M tiles/core, 13 N tiles/core, 6 M blocks/core, 2 N blocks/core, 2 K blocks**. The final N block contains 5 tiles. The static allocation still uses full nominal blocks.

For the passing-policy contrast at this same output projection, all these projection arguments and loop counts stay the same; the retained normalized input changes from BFP8 to BF16. At physical 4096, M=N, the orientation instead uses 10 M cores and 11 N cores: 13 M tiles/core and 12 N tiles/core, hence 4 M blocks, 2 N blocks, 2 K blocks. Neither case relies on divisibility of total rows by block or grid dimensions.

## Retained tensors and exact L1 ledger

`forward` holds `h = decoder.norm(residual, "input_layernorm")` across attention (`tt/prefill_ops.py:161–165`). At 8192, `norm` computes in DRAM, casts to the requested dtype, then explicitly returns an L1 copy (`tt/multichip_decoder.py:726–740`). The selected dtype is BF16.

There are **three Python references to this same backing allocation**, not three allocations: `forward.h`, the pooled-state wrapper's `attention.x`, and the inner `attention.x` (`tt/prefill_ops.py:81–93`). They remain live through `out_proj`. `del h` at line 165 is after the failure; deleting only the inner `x` would leave the other references. Explicitly freeing the shared input would require an ownership contract that this method does not declare.

The source-derived target has 110 interleaved L1 banks. This follows the actual 11×10 compute-grid descriptor and one bank per `ComputeAndStore` core, independently of the matmul's grid choice (`tt_metal/core_descriptors/blackhole_140_arch.yaml:60–65`; `tt_metal/impl/allocator/l1_banking_allocator.cpp:39–43,101–105,285–288`). The failing log does not directly print a bank-count memory view. The exact address reconstruction below corroborates this count.

For tiled interleaved allocations, bytes per bank are `ceil(tile_count / 110) × tile_bytes` (`tt_metal/impl/allocator/allocator.cpp:784–793`). BF16/BFP8/FP32 tiles occupy 2048/1088/4096 bytes (`tt_metal/api/tt-metalium/tt_backend_api_types.hpp:122`). Blackhole physical L1 is 1,572,864 bytes (`tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:33`).

| Allocation still live when out_proj is compiled | Bytes per affected bank/core |
|---|---:|
| Normalized `h`, BF16 `[1,1,8192,4096]`: 32768 tiles, 298 tiles/bank | 610,304 |
| Temporary projection input `to_memory_config(out,L1)`, BF16 `[1,1,8192,1024]`: 8192 tiles, 75 tiles/bank | 153,600 |
| Two persistent AR scratch buffers: each has a BF8 `[32,512]` shard on 32 cores, 16 tiles/shard | 34,816 combined |
| Fourteen persistent global semaphores, 64-byte minimum allocation apiece | 896 |
| **Total top-down tensor/semaphore reservation** | **799,616** |

Scratch/semaphore evidence: `tt/multichip_decoder.py:94–116`; inherited CCL creates six RS, four AG and two barrier semaphores at `models/demos/gpt_oss/tt/ccl.py:42–59`, plus two AR semaphores. The subclass places CCL semaphores over the actual compute grid (`tt/multichip_decoder.py:26–39`). Global semaphore storage is one four-byte shard per participating core (`tt_metal/impl/buffers/global_semaphore.cpp:84–103`), rounded by the allocator's 64-byte minimum (`bank_manager.cpp:120–128,194–196`; Blackhole `noc_parameters.h:379`). Device open supplies no L1_SMALL override (`tests/datatype_evaluate.py:57`); its default is zero (`tt_metal/hostdevcommon/api/hostdevcommon/common_values.hpp:15`). Thus these reservations occupy ordinary L1. The shared scratch is counted once, not once per decoder.

```text
1,572,864 − 610,304 − 153,600 − 34,816 − 896 = 773,248
```

This is an exact source-based reconstruction of the measured frontier. It strongly supports the staged projection input as the lowest allocation. **The exception alone does not identify a tensor owner or a particular offending physical core**: the validator queries the global lowest occupied compute L1 address; its printed core range is the static DFB region (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3063–3088`). A buffer map would independently establish ownership.

The other attention locals remain live but are DRAM: `packed` `[1,1,8192,3136]`, `z`, `a/g` FP32 and `beta` BF16, Q/K/V, projected history and history-gather storage, the new FP32 recurrent state, and the gated BF16 `out` before staging. They are not additional L1 copies to add to this ledger. Projection outputs explicitly use DRAM; convolution defaults to DRAM (`qkv_causal_conv1d_silu.cpp:38`); GDN defaults to DRAM (`chunk_gated_delta_rule.cpp:295`); the subsequent slicing, normalization, concatenate-heads and binary paths retain those DRAM placements. Profile uploads and cache/state allocations also explicitly use DRAM (`tt/prefill_ops.py:58–66`; `tt/multichip_decoder.py:514–525,547–570`).

## Static buffer sizing and allocation order

The descriptor allocates four separate DFBs with double-buffered inputs/output and one FP32 intermediate block (`minimal/device/minimal_matmul_program_descriptor.cpp:433–447,500–515`):

| DFB | Tiles | Tile bytes | Bytes |
|---|---:|---:|---:|
| Activation | `2×4×16 = 128` | 2048 | 262,144 |
| Weight | `2×16×8 = 256` | 1088 | 278,528 |
| Output | `2×4×8 = 64` | 2048 | 131,072 |
| Intermediate | `4×8 = 32` | 4096 | 131,072 |
| **Total** | | | **802,816** |

The observed end implies effective static base **112576**: `915392 − 802816`. Static allocation starts at the persistent program arena's high-water mark and appends aligned buffers (`tt_metal/impl/program/program_allocation_support.cpp:48–60,77–115`). All four sizes are alignment multiples.

```text
Available static span = 773,248 − 112,576 = 660,672
Required DFB bytes    =                       802,816
Deficit              =                       142,144
```

Allocation-time ledger:

1. Persistent CCL scratch/semaphores already exist.
2. Norm places `h` in L1; the caller retains it through attention.
3. Python stages `out` in L1 before entering minimal matmul.
4. The operation allocates its BF16 **DRAM** output (`device_operation.cpp:430–440`) before building/using the program. This adds **zero** L1 tensor bytes here.
5. The explicit block geometry determines the static DFB arena; compilation checks its end against the now-occupied L1 frontier and throws.

There is no free-L1 admission selector in this minimal path: Python supplies a fixed config and C++ accepts its structural constraints. The failure is not a stale snapshot taken by this operation. If adding adaptation, choose a deterministic dtype/role/physical-profile rule. A snapshot before staging must reserve the pending input copy; one after staging must not subtract that copy again. Aggregate free bytes or the largest free block are not substitutes for the absolute low-address static arena bound.

The baseline normalized hidden uses `298×1088 = 324224` bytes/bank, saving **286080**. With the other reservations unchanged, the reconstructed frontier is **1059328**, above the same output-projection static end by **143936**. This explains why changing the normalized-input dtype can break the BF16 output projection without changing that projection's own operands or DFB sizes. This baseline address is a prediction, not an address recorded in the passing JSON.

## Smallest interventions and the BF16 reference

### Observed BFP8-weight / BF16-normalized-activation case

Two alternatives are supported by the source budget:

1. **Smallest placement change:** keep the 8192-row BF16 normalized hidden in DRAM in `norm`, instead of copying it back to L1. The output projection then stages only its 153600-byte input; the retained 610304-byte L1 allocation disappears. Apply the same norm placement to post-attention normalization. This preserves the requested dtype and existing matmul geometry. The packed-input projection will still temporarily stage its 610304-byte activation; it fits the reconstructed frontier by only **11456 bytes**, so this is a narrow, candidate-specific bound, not a general reference solution.
2. **Small geometry alternative supported by the source budget:** use `(M,K,N)=(4,8,8)` for the 8192-row attention-output minimal matmul. DFB bytes become **532480**, predicted static end **645056**, leaving **128192 bytes** against the observed frontier. Preserve subblock 1×2, grid 11×10, BF16 output, FP32 destination and requested fidelity. A consistent per-profile helper can cover all minimal calls rather than hardcoding a policy ID. This is not a proof that K8 is the largest fitting block or a hardware-validated configuration.

Changing only `linear` to stage this projection input in DRAM releases 153600 bytes and would leave just **11456** bytes of reconstructed margin while the BF16 `h` remains live. It addresses this one collision but leaves the retained-tensor policy and other roles exposed. Merely moving `del h` after/before the attention call is not a valid lifetime repair: attention still needs its argument and has other references.

### All-BF16/HiFi4 reference: additional predicted collisions

`configs/bf16_hifi4.json` requests BF16 weights, normalized activations, CCL and KV, and HiFi4 projection math. FP32 destination accumulation remains enabled. Fidelity itself does not change these buffer byte counts; the weight dtype does.

With the original geometry, the BF16 weight DFB grows by `256×(2048−1088)=245760` bytes. Total ordinary DFB payload becomes **1048576**, with predicted static end **1161152** using the same base. Persistent BF16 CCL scratch becomes **65536** bytes/core, an extra **30720**.

| Reference case under current placements | Reconstructed L1 frontier | Predicted static end | Overlap |
|---|---:|---:|---:|
| 4096 output projection: `h=305152`, staged input `77824` | 1,123,456 | 1,161,152 | 37,696 |
| 8192 packed-input projection: only `h=610304` in L1 | 896,128 | 1,161,152 | 265,024 |
| 8192 output projection: `h=610304`, staged input `153600` | 742,528 | 1,161,152 | 418,624 |
| 8192 MLP down: staged `[8192,3072]` input `458752`, former norm no longer live | 1,047,680 | 1,161,152 | 113,472 |

Each frontier is `1572864 − 65536 − 896 − listed live activations`. Exactly 4096 uses the ordinary norm branch, which explicitly moves the requested BF16 result into L1 (`tt/multichip_decoder.py:741–751`); modifying only the `>4096` branch cannot repair that bucket. The first 4096 packed-input projection fits by 40128 bytes, so its success would not establish output-projection safety. These are **predictions**, not a recorded reference run; earlier attention operations could expose another resource limit first.

**Recommended placement repair covering both requested policies:** for affected BF16 precision combinations, keep normalized tensors and all projection inputs in DRAM starting at physical bucket **4096**. Use one coherent dtype/shape rule in both `norm` and `linear`, including the ordinary norm path at exactly 4096. Cover packed input, attention output, and MLP down; direct packed SwiGLU naturally consumes the DRAM norm result. With no large transient L1 activations, the reference frontier is **1506432**, leaving **345280** bytes above the ordinary static end under this ledger. Keep the established BFP8 policy's placements if preserving its measured performance is desired. Select from effective tensor/weight dtypes and per-layer policy, not the string config ID; weight groups and layer exceptions are independent (`tt/precision.py:53–64,83–110`).

A narrower reference placement alternative is norm DRAM at 4096 and above, plus projection-input DRAM at 8192 and above. At 4096 the remaining packed-input staging fits by 40128 bytes and down staging by 115904 bytes. The coherent 4096 rule is easier to audit and has larger margins.

Geometry alternatives, all with M=4 and subblock 1×2:

| K/N blocks | Ordinary DFB bytes, BF16 activation/BFP8 weights | Ordinary DFB bytes, BF16 activation/BF16 weights | BF16 reference static end |
|---|---:|---:|---:|
| 16/8, existing | 802,816 | 1,048,576 | 1,161,152 |
| 8/8 | 532,480 | 655,360 | 767,936 |
| 4/8 | 397,312 | 458,752 | 571,328 |
| 8/4 | 331,776 | 393,216 | 505,792 |

**K8 alone is insufficient for the unchanged 8192 reference placement:** `767936 > 742528` by **25408** bytes. K4/N8 or K8/N4 give larger analytic margins without changing dtypes or fidelity. A reference-wide geometry alternative must cover both 4096 and 8192 and every affected minimal projection, including the **direct packed-SwiGLU call in `finish`** (`tt/multichip_decoder.py:775–783`), not just the failing output projection. Fused SwiGLU has half the ordinary output-buffer size; for all-BF16 K16/N8 its payload is 983040, still too large with the retained 8192 BF16 norm.

These block choices pass the inspected structural constraints: positive sizes, subblock divisibility, destination capacity, grid bounds, and even N blocks for SwiGLU (`minimal/device/minimal_matmul_device_operation.cpp:362–397`; `descriptor:416–425`). Kernels explicitly handle partial M/N blocks and padded K (`compute_metal2.cpp:568–571,621–631`; `matmul_dataflow_common_metal2.hpp:80–163,184–199`). A K-block change changes accumulation grouping and an N-block change changes scheduling; neither has been numerically or operationally validated here. Placement preserves the current explicit matmul configuration and is the preferred first repair boundary.

## Preserve full context and non-aligned profiles

Do not reduce context, skip required warmups, shrink chunk caps, round logical lengths differently, or coalesce the 511/512 distinction to evade the failure. The contract is (`tt/prefill_plan.py:27–51`):

| Layer kind | Required physical profiles | Original chunk cap |
|---|---|---:|
| Linear | 1, 32, 64, 128, 511, 512, 640, 768, 960, 1024, 4096, 8192, 32768 | 32768 |
| Full attention | 32, 64, 128, 480, 512, 768, 1024, 4096, 8192, 16384 | 16384 |

The existing small-profile decode/short-MLP branches and 512..1024 multicast projection branches should remain as configured unless separately investigated. Linear 511 uses the general minimal path and BF16 small-profile normalization/CCL behavior even though its tile storage pads to 512; linear 512 selects different Python branches. Full attention 511 valid rows uses bucket 512. Packed MLP remains a minimal call above 128 even when projections use the multicast special case.

An exhaustive **host-only** check of every canonical whole-chunk count confirmed that both kinds map counts **4097..8192** to physical bucket 8192. Larger counts map to linear 32768 or full 16384, where the existing norm/projection policy already keeps inputs in DRAM. A complete 262144 horizon uses eight original linear chunks and sixteen original full-attention chunks, unchanged by a placement/geometry repair.

Fragment execution retains its **horizon-derived** physical bucket (`tt/prefill_plan.py:133–153`). For example, `total_length=4097, start=1, end=2` executes bucket 8192 with valid count 2 including one overlap row. A tiny valid fragment therefore still needs the large-profile repair. At `total_length=8193, start=8191, end=8193`, the physical profile is linear 32768/full 16384 with valid count 33 and overlap 31. Select placement/config from the actual physical tensor shape, never current valid count or tokenized prompt length alone.

Preserve masks, history indices, recurrent boundary snapshots and overlap replay (`tt/prefill_ops.py:102–148,172–183`; `tt/prefill_plan.py:10–22`). The host policy test explicitly hardcodes the historical placement thresholds (`tests/full_model_prefill_plan_test.py:23–35,72–80`); extend it for intentional dtype-dependent placement instead of weakening its catalogue/cap/coverage assertions.

## Other potential issues and limits

- The generic **automatic** matmul config path has a separate allocation-time concern: it chooses config before launching (`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:3525–3537`), and its capacity helpers budget CBs against current L1 without reserving a later automatic L1 output (`device/config/matmul_program_config.cpp:56,86,216`). Output allocation follows in the operation/framework. This is not needed to explain this failure: the failing call uses an explicit minimal config and a DRAM output. Reachability in other attention internals has not been established; do not headline it as this root cause.
- The exact reconstructed frontier is strong evidence, but no allocation map independently binds address 773248 to a tensor. No BF16-reference failure artifact was supplied, and no hardware result for a proposed repair exists. A source-only budget does not prove device execution, numerical agreement, performance, or complete DRAM capacity at full context.
- Ambient-free-memory config selection could change programs on partial/resumed requests despite unchanged profile tags, introducing late compilation after preparation. Prefer deterministic selection from effective dtype, role, residual/fused layout and physical bucket. Ensure the chosen config participates in the program key; minimal matmul already hashes its operation attributes (`device_operation.cpp:101–125`).

## Verification after a repair (not executed here)

1. Record effective placement/dtype/shape and a buffer map or occupied L1 frontier immediately before both ordinary and packed-SwiGLU minimal calls. Confirm the observed 8192 failure and the reference 4096/8192 predictions with unchanged context and precision settings.
2. Prepare the complete production-wrapper catalogue for baseline BFP8, BFP8 weights/BF16 normalized activations, and BF16/HiFi4 reference. Confirm synchronous completion and no new program entries on repeated or resumed profiles. Historical test adapters alone do not establish current wrapper lifetimes.
3. Validate outputs and conv/recurrent/KV state against the existing numerical contracts, including 511/512, 513, 4095/4096/4097, 8191/8192/8193; partial-profile valid counts 1, 31, 33 and bucket−1; 68→70 and other non-aligned continuation; subsequent decode; chunk-cap crossings; 262143 and 262144. Preserve accuracy gates and context/memory evidence. Re-measure latency if changing placement or geometry.

The investigation itself performed only source/artifact reads, SHA-256 comparisons, pure-host catalogue/planning calculations and buffer-size arithmetic. The first draft was revisited against the lowering, allocator, tensor lifetimes and full policy matrix; the reference 4096 prediction and the K8-only reference limitation are retained as explicit predictions rather than hardware findings.
