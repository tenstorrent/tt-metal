# Non-decode SDPA APIs and non-model compatibility audit

Scope: current checkout `ttnn/`, `tests/`, and `tt_metal/`, read-only audit on 2026-09-16. No hardware jobs or production modifications. Read root AGENTS.md. Excludes model implementations, tt-train, decode API compatibility, and `experiments/sdpa-l2/` as compatibility requirements. The pre-existing research repro under `tests/.../repro_sdpa_l2.py` is inventoried but is not an external compatibility requirement.

## Executive conclusions for the three PRs

1. **PR1 (shared streaming and presets A/B/C/D/E/G) must preserve legacy caller semantics outside explicitly migrated configurations.** Omitted compute config means HiFi2; an explicitly constructed empty config means LoFi. BH and WH config names are aliases. Existing tests exercise both approximation flags independently, HiFi3/LoFi, BFP4/BFP8 and mixed dtypes, FP32 accumulation, D other than128, causal/masked/windowed/chunked and MLA paths. A preset-only rewrite without an explicit legacy-config adapter is a behavior change.
2. **PR2 must migrate actual feature implementations, not only Python names.** Ordinary joint attention and ring-distributed SDPA are still legacy-loop consumers. Ring-joint FP32 uses its legacy branch; exp-ring-joint is already streaming-only at kernel compile time despite a host gate that can be false. Sparse APIs use streaming *primitives* in separate loops and have FP8 tilization/row-major contracts.
3. **PR3 may delete unreachable legacy loop bodies only after all dispatches are migrated. It cannot delete compute_common.hpp wholesale.** Dense streaming, sparse, decode, and CCL reduce-to-root consume its shared helpers. The Quasar namespace has separate physical copies and dispatch, not a BH/WH alias.
4. Input storage format, internal precision and output dtype are distinct API contracts. Dense output generally follows Q dtype; ring outputs may be explicitly BF16. FLOAT32 intermediate code paths are not evidence of supported FLOAT32 public Q/K/V.

## Evidence conventions

File:line references are relative to the repository root and refer to the inspected source snapshot. “Accepted” below means source validation/binding admits a configuration, **not that this audit ran it**. “Observed” means a static test/sweep call or parameter family, not a collected/passing test. Parametrized tests can skip combinations. Negative validation calls are included in inventory but are not successful usage.

[tests-api-calls.json](tests-api-calls.json) contains96 direct Python TTNN calls in25 files, with arguments and keywords. Two calls are the research repro; excluding them gives94 sites in24 files. Static AST extraction resolved direct attribute calls and simple imported/assignment aliases and excluded Torch reference calls and decode function names. Reconciliation against the root-owned [python-census.json](python-census.json) found **zero missing TTNN/indirect sites in the owned roots**. No external non-decode TTNN C++ invocation was found in `tests/` or `tt_metal/`; the C++ API implementations, wrappers and primitive calls in `ttnn/` are covered below. This is not a count of runtime calls or parametrized cases.

## Public surface and aliases

All following main-namespace Python functions are bound under `ttnn.transformer` in `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp`:

| API | Binding line | C++ entry / relationship |
|---|---:|---|
| scaled_dot_product_attention |345| `sdpa.cpp:35`; dense, causal defaulttrue; masks, sliding window, sink and cumulative-window modes |
| chunked_scaled_dot_product_attention |525| `sdpa.cpp:103,144`; scalar or device-tensor chunk start, paged K/V, optional HMA cache geometry |
| joint_scaled_dot_product_attention |573| `sdpa.cpp:184`; six Q/K/V tensors, rear joint strategy, two outputs |
| ring_joint_scaled_dot_product_attention |687| `sdpa.cpp:209`; fused communication, persistent buffers, causal/balanced/cross/cache metadata options |
| ring_mla |780| `sdpa.cpp:298`; translates latent-V semantics to ring-joint primitive, returns output/stats pair |
| exp_ring_joint_scaled_dot_product_attention |855| `ExecuteExpRingJointAttention::invoke` in `sdpa.cpp:364`; experimental fused ring |
| flash_mla_prefill |908| `sdpa.cpp:421`; shared ordinary SDPA primitive with MLA attributes |
| chunked_flash_mla_prefill |964| `sdpa.cpp:457`; paged latent-V prefill through ordinary primitive |
| ring_distributed_scaled_dot_product_attention |1022| `sdpa.cpp:492`; ring_size/ring_id, causal distributed SDPA, optional paged chunk |
| sparse_sdpa |365| `sparse_sdpa.cpp`; gathered sparse latent-V, explicit SparseKVFormat |
| sparse_sdpa_msa |418| `sparse_sdpa_msa.cpp`; block-sparse separate K/V, optional causal chunk offset |

The Python MLA binding has **two overloads**, integer `head_dim_v` and Tensor `input_tensor_v` (`sdpa_nanobind.cpp:910,929`). The latter is a valid embedding-space-V call, not a stale third-argument signature. C++ has one function with mandatory integer head_dim_v and optional explicit V (`sdpa.hpp:185`); wrappers adapt Python overloads. Chunked SDPA supports old scalar and trace-safe tensor starts; do not migrate only the scalar overload.

`ttnn.experimental.quasar.transformer` provides separate ordinary SDPA, both chunked overloads, joint SDPA, flash MLA and chunked MLA implementations: `ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa/sdpa.hpp:13` onward and its `sdpa_nanobind.cpp`. It has its own kernels/factories and is not an alias to the BH/WH implementation. No direct Quasar non-decode call was found in the owned Python census. This absence does not authorize API deletion.

`DeviceComputeKernelConfig`, `WormholeComputeKernelConfig` and `BlackholeComputeKernelConfig` are aliases of unified `ComputeKernelConfig`: `ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp:37`. No additional top-level Python SDPA alias was found in the owned roots. Documentation mapping entries are comments, not calls (`tests/ttnn/docs_examples/examples_mapping.py:419`); distributed TG's SDPA calls are decode and excluded.

## Defaults: omitted versus explicit configuration

| Family | Config omitted | Exp flag | Architecture / packer nuance |
|---|---|---|---|
| Dense, chunked, MLA, chunked MLA, ring-distributed |HiFi2, math_approx=true, fp32_dst=false, packer_l1=false, dst_full_sync=false|program config exp_approx=None/absent resolvestrue|same resolver behavior on BH/WH |
| Joint, ring-joint, exp-ring-joint |same defaults, resolved in primitive device entry|same independent program exp flag|same BH/WH config type |
| sparse_sdpa |HiFi2, math_approx=true, fp32_dst=true iff Q **or KV** is FP8_E4M3; L1false|derived from math_approx; no SDPAProgramConfig exp selector|BH-only |
| sparse_sdpa_msa |HiFi2, math_approx=true, fp32_dst=true iff **Q** is FP8_E4M3; L1false|derived from math_approx|BH-only |
| explicit empty ComputeKernelConfig() |**LoFi**, math_approx=true, fp32_dst=false, L1false, full_syncfalse|does not set program exp flag|resolver returns provided object unchanged |

Evidence: `sdpa.cpp:53,117,158,435,470,508`; `joint_sdpa_device_operation.cpp:242`; `ring_joint_sdpa_device_operation.cpp:1160`; `exp_ring_joint_sdpa_device_operation.cpp:468`; `sparse_sdpa.cpp:75`; `sparse_sdpa_msa.cpp:55`; core config header:20 and resolver `compute_kernel_config.cpp:31`. Dense exp resolution is `sdpa_program_factory.cpp:103`; joint/ring/exp-ring/ring-distributed factory lines149/1195/333/115 respectively. Sparse derives EXP_APPROX_MODE from math approximation in `sparse_sdpa_compute.cpp:19` and `sparse_sdpa_msa_program_factory.cpp:287`.

The public packer_l1_acc field must not be equated to “this loop never accumulates in L1.” Factories unpack it, but these ComputeConfigDescriptor initializers do not convey that field; kernels explicitly toggle L1 accumulation. Sparse factories explicitly discard it (`sparse_sdpa_program_factory.cpp:280`, `sparse_sdpa_msa_program_factory.cpp:251`). Preserve/document effective semantics before deciding whether a preset supersedes the flag.

`SDPAProgramConfig` constructor requires grid and Q/K chunks; exp defaultsNone and max_cores_per_head_batch defaults16 (`transformer_nanobind.cpp:27`). Omitting the entire program config is a separate API case. Dense factory has its own default chunk resolution; do not replace caller-specified chunks/buffering through preset selection.

WH caution: `compute_kernel_config.cpp:43` documents a HiFi4+FP32 hardware issue and warns to prefer HiFi3 on WH; it does not reject or rewrite the config. BH-only measured preset behavior cannot be applied to WH merely because the config classes alias.

## Accepted input / feature contracts

| Family | Source-accepted tensor formats | Important distinct semantics |
|---|---|---|
| Ordinary SDPA/chunked/MLA |Each validated operand independently BF16/BFP8_b/BFP4_b, TILE, interleaved; no equal-dtype check in common operand loop|No FLOAT32 QKV; ordinary GQA; unequal noncausal Q/K lengths; masks and generated padding; paged chunk metadata; MLA V prefix or explicit embedding V |
| Joint |all six same dtype, BF16/BFP8_b|rear joint concatenation, separate output tensors; BFP4 is not accepted here |
| Ring distributed |same dtype BF16/BFP8_b/BFP4_b|causal, GQA, optional page table/chunk; outputs follow Qdtype |
| Ring joint / ring MLA |normally same-format BF16/BFP8_b/BFP4_b; a **specific chunked sliding** exception admits BF16Q+BFP8KV|GQA/separate-V and latent-V head rules; persistent/gathered buffers; sink, partial joint tail, runtime slot/length metadata |
| Exp ring joint |same-format BF16/BFP8_b/BFP4_b|Q/K head counts must match; dedicated fused scheduling; outputs/stats explicitly BF16 |
| Sparse latent-V |Q BF16 or FP8_E4M3 row-major; KV BF16, FP8_E4M3 or packed scaled-FP8 bytes|BH-only; uint32 token indices, sentinel-tail contract; FP8 tilize through32-bitDST; outputdtypeQ |
| Sparse MSA |Q BF16 or FP8_E4M3 row-major; K and V individually BF16/BFP8_b TILE|BH-only; uint32 block indices; H/n_kv=16 or multiple32; FP8Q with causal offset explicitly rejected; outputdtypeQ |

Evidence: `sdpa/device/sdpa_device_operation.cpp:35,150,203,218,395,449,501,540`; `joint_sdpa_device_operation.cpp:37,61`; `ring_distributed_sdpa_device_operation.cpp:51,69,171,202,276`; `ring_joint_sdpa_device_operation.cpp:490,507,578,690,863`; `exp_ring_joint_sdpa_device_operation.cpp:50,71,181,373`; `sparse_sdpa_device_operation.cpp:77,84,88,188,206`; `sparse_sdpa_msa_device_operation.cpp:65,109,113,117,119,127,197`. These device filenames are under `ttnn/cpp/ttnn/operations/transformer/sdpa/device/`.

The dense FLOAT32 branches inside factories concern dataformat selection and do **not** override validation's BF16/BFP8/BFP4 whitelist. No direct FLOAT32 device-QKV usage was found in the audited tests; many `torch.float32` occurrences are generators/references/scales and must not be counted as such usage. Scaled-FP8 sparse cache contains FP32 scale bytes, not FLOAT32 QKV tensors.

## Observed test/configuration families and caller-update obligations

| Family / representative evidence | Observed configuration, not hardware qualification | Required preservation or explicit migration |
|---|---|---|
| `tests/didt/test_sdpa_op.py:51,81` |BF16 LoFi/HiFi2/HiFi3/HiFi4; mathapproxfalse, expapproxfalse, BF16dst|Do not collapse all four into one preset; retain numerical-control tests |
| `tests/didt/test_mla_sdpa.py:71,92` |HiFi2, mathapproxfalse, expfalse, BF16dst, packer_l1true|MLA/separate V and public L1 config compatibility |
| `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py:268,348,373,422,482,640` |BH HiFi2/approx/BF16dst versus other-arch HiFi4/!approx/FP32dst; BFP4 dense mask; sliding/sink; omitted-config biased attention|Defaults and architecture branches cannot silently adopt BH preset defaults |
| `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_prefill.py:274` |BF16/BFP8/BFP4, DRAM andL1 interleaved, Q128/192/256,K128/256; GQA shapes include71Qheads/1KV,D64 and D128; B4 combinations selectively skipped|Geometry, memory config, mixed mask dtype, padding/GQA and skip conditions remain test inputs |
| Same nightly file:524,784; `test_sdpa_chunked.py:242,382,522` |masks BF16/BFP8/BFP4; chunked BF16Q+BFP8KV; scalar/device chunk start, trace and HMA geometry|Cache-addressing/runtime metadata must survive selection and cache keys |
| `tests/ttnn/unit_tests/operations/sdpa/test_windowed_sdpa.py:73,89,178,278` |both DST modes; HiFi4/!mathapprox/!expapprox; packer_l1true in offset tests; scalar and sharded tensor offsets|FP32 windowed route is not covered by current restricted experimental FP32streaming guard |
| `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_joint.py:46,120` |BF16/BFP8, heads1/3, architecture-dependent HiFi4FP32 versus HiFi2BF16; exptrue|Must port joint concatenation/outputs, not redirect to ordinary attention |
| `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_ring_distributed.py:65,82,181,238` |BFP8, HiFi2, approximationtrue, BF16dst; causal GQA8Q/1KV,D128; paged and unpaged controls|Real legacy consumer even without FP32dst |
| `tests/ttnn/unit_tests/operations/sdpa/test_mla_prefill.py:11,20`; helper `mla_test_utils.py:620` |Dqk576/Dv512 and Dqk256/Dv256; BF16Q+BFP8KV and BFP8Q+BFP4KV; paged; HiFi4, bothapproximationsfalse, BF16dst|Do not equate MLA D with128 or assume Q/K/V formats equal |
| `tests/ttnn/nightly/unit_tests/operations/sdpa/test_mla_prefill_stress.py:23` |H16/32/128, KVheads1/16, latent64/512, RoPE0/32/128, paged/unpaged; mixedBF16/BFP8|Include distinct QK/PV widths and shared latent-V storage |
| `test_mla_prefill_v_embedding_space.py:123,141` |both Python MLA overloads|Tensor-third-argument overload must remain; C++wrapper adaptation is intentional |
| `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py:588,655,3517,4898,5557` |both DST modes in joint/chunked tests, expfalse, persistentND-sharded caches, cache slots, runtime lengths, sink/sliding, ringMLA|Feature-specific unsupported combinations must remain explicit, not deadlock through generic fallback |
| `tests/nightly/blackhole/sdpa/test_exp_ring_joint_sdpa.py:375,382,426` |expfalse/BF16dst; communication knobs/persistentbuffers|Streaming-only compilecontract and ringreader scheduling |
| Sparse test helpers plus `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sparse_sdpa.py:120,139,173,305` |BF16/FP8QKV, scaledFP8, determinism, indexedcache; omitted config common|Preserve automaticFP32 tilize default and actual consumed formats; do not relabel FP8_E4M3 as BFP8 |
| MSA helpers and nightly `test_sparse_sdpa_msa.py:258,369,635` |BF16/FP8Q, BF16/BFP8KV, block indices, causaloffset, SP andblockcyclic|Per-block gather/mask/GQAcontracts; retain negative validation cases |
| `tests/sweep_framework/sweeps/model_traced/scaled_dot_product_attention_model_traced.py:74,278,578,595`; chunked counterpart:52,253,400,427 |loaded model-traced configs plus BF16 fallback suite; kwargs may carry arbitrary legacy compute/program configs; LoFi-specific acceptance handling; chunked blockfloat handling|Update traced-config deserialization and manifests, not only visible constructor defaults; don't infer supported dtype from Torch generators |

No broad FLOAT32 public tensor family was observed. HiFi3/LoFi are real nondecode test parameters (DIDT), but this static audit does not assert they pass all dtypes/features. The nightly helper architecture branches are explicit user configs and must not be mistaken for API defaults.

## Dispatch and deletion audit

| Consumer | Current dispatch | Implication |
|---|---|---|
| ordinary SDPA, chunked, MLA | `sdpa_program_factory.cpp:79` streams whenever !fp32dst. Added BH FP32 experimental exception at486 requires HiFi2, both approximations, noncausal/no MLA/no chunk/no sink/window/mask, Q128,K512/1024,D128,N>=32K,BF16 QKV. Compute `sdpa.cpp` selects `sdpa_standard_v2` at159 or `sdpa_standard` at224.|Global deletion breaks WH FP32 and BH FP32 outside that narrow guard, including existing windowed/causal tests. Distinguish the worktree's experimental exception from broad production support. |
| joint | `kernels/compute/joint_sdpa.cpp:12,77` includes common and calls `sdpa_joint`; factory has no streaming selector.|Port before deleting the legacy joint loop. |
| ring distributed | `ring_distributed_sdpa_program_factory.cpp:295,339,383` explicitly sets streaming false in reader/writer/compute args; same `sdpa.cpp` kernel.|BF16 callers also depend on legacy; migration must align all three threads. |
| ring joint / ring MLA | `ring_joint_sdpa_program_factory.cpp:1324` streams iff !fp32dst; `ring_joint_sdpa.cpp:388,464` calls ring_v2 versus legacy ring.|FP32 joint route remains; latent V, kv_actual_isl and sharded joint padded tail explicitly require streaming (1326–1341). |
| exp ring joint | Host gate `exp_ring_joint_sdpa_program_factory.cpp:447` includes !fp32dst, subblock height<=2, K divisibility, >1 Q subblock; kernel `exp_ring_joint_sdpa.cpp:188` **static_asserts streaming** and calls ring_v2.|A false gate is not a working legacy fallback. Preserve/strengthen host validation rather than claiming FP32 support. |
| sparse / sparse MSA | Separate loops call `blocked_matmul_and_pack`, `reduce_c_row_group`, `sub_exp_block_bcast_cols`, recurrence/finalize helpers from streaming/common; `sparse_sdpa_compute.cpp:302,392,405,453`, MSA:151,195,207,251.|They are shared-helper clients, not ordinary sdpa_standard_v2 callers; generic loop deletion alone does not migrate them. |
| Quasar copies | `experimental/quasar/.../sdpa_program_factory.cpp:78,412` uses !fp32dst gate; joint has its own legacy kernel.|Separate namespace/source lifetime decision; BH-only migration does not remove its fallback. |

### Shared-header consumers which survive loop deletion

`ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp` is directly included by:

- dense `sdpa.cpp:12`, joint `joint_sdpa.cpp:12`, ringjoint:17, expring:13, sparse:22, sparseMSA:13;
- **decode** `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp:28`;
- **CCL** `ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/compute_kernel.cpp:20`.

Decode APIs are outside this migration scope, but their compile dependency is not. CCL actively uses `sub_exp_block`, `mul_block_inplace` and `mul_block_bcast_cols` (CCL file:67,73,82), not merely an unused include. Streaming code also relies on common declarations. Extract small shared math/CB utilities first or preserve the header while deleting only proven-unreachable loop templates. Quasar has separate common/streaming copies; its decode copy is separately referenced.

## Suggested PR acceptance checklist

- PR1: explicit preset field/config adapter; omitted-versus-empty default tests; hash key includes numerical preset and effective formats; all six preset tests on the supported BH envelope; no silent Q/K chunk or buffer changes; legacy feature/WH behavior retained; shared-helper compile tests including decode/CCL.
- PR2: migrate each dispatch row above, with per-feature reader/compute/writer tests, mixed types/output dtype, both MLA overloads, metadata/cache-hit/trace behavior and WH HiFi3 policy. Update sweep parameter extractors/config serialization as well as callers. Negative unsupported cases must fail before kernel deadlock.
- PR3: prove no factory selects legacy; delete only legacy loops and obsolete CB/CTA plumbing; rebuild all affected kernel clients; cover WH and Quasar decisions explicitly. Source grep alone is insufficient when a header contains both shared helpers and legacy loops.

## Complete direct Python callsite index

Detailed arguments are in [tests-api-calls.json](tests-api-calls.json). The repro row is research-only, and sparse negative tests are not success evidence.

| File | Call lines | APIs |
|---|---|---|
| `tests/didt/test_mla_sdpa.py` | 25 | `flash_mla_prefill` |
| `tests/didt/test_sdpa_op.py` | 29 | `scaled_dot_product_attention` |
| `tests/nightly/blackhole/sdpa/test_exp_ring_joint_sdpa.py` | 426 | `exp_ring_joint_scaled_dot_product_attention` |
| `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` | 839, 1571, 2098, 3167, 3433, 3743, 3909, 4090, 4227, 4444, 5869 | `ring_joint_scaled_dot_product_attention`, `ring_mla` |
| `tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py` | 69, 151 | `scaled_dot_product_attention` |
| `tests/nightly/blackhole/sdpa/test_sparse_sdpa_multidevice.py` | 80 | `sparse_sdpa` |
| `tests/sweep_framework/sweeps/model_traced/chunked_scaled_dot_product_attention_model_traced.py` | 400 | `chunked_scaled_dot_product_attention` |
| `tests/sweep_framework/sweeps/model_traced/scaled_dot_product_attention_model_traced.py` | 578 | `scaled_dot_product_attention` |
| `tests/ttnn/nightly/unit_tests/operations/sdpa/test_mla_prefill_chunked_vs_not.py` | 105, 161 | `flash_mla_prefill`, `chunked_flash_mla_prefill` |
| `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_chunked.py` | 151, 162, 210, 222, 361, 522, 578 | `chunked_scaled_dot_product_attention` |
| `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_joint.py` | 83 | `joint_scaled_dot_product_attention` |
| `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_prefill.py` | 132, 212, 693, 704, 752, 764, 971, 1346, 1391, 1501, 1510, 1776, 1867, 2007, 2085, 2160 | `scaled_dot_product_attention`, `chunked_scaled_dot_product_attention`, `joint_scaled_dot_product_attention` |
| `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_ring_distributed.py` | 103, 125, 238, 262 | `ring_distributed_scaled_dot_product_attention`, `scaled_dot_product_attention`, `chunked_scaled_dot_product_attention` |
| `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sparse_sdpa.py` | 198, 244, 330 | `sparse_sdpa` |
| `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sparse_sdpa_msa.py` | 258, 297, 369, 566, 635 | `sparse_sdpa_msa` |
| `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sparse_sdpa_msa_block_cyclic_multidevice.py` | 90 | `sparse_sdpa_msa` |
| `tests/ttnn/unit_tests/operations/sdpa/mla_test_utils.py` | 649, 661 | `chunked_flash_mla_prefill`, `flash_mla_prefill` |
| `tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py` | 315, 332 | `scaled_dot_product_attention` |
| `tests/ttnn/unit_tests/operations/sdpa/sparse_sdpa_msa_test_utils.py` | 186, 231 | `sparse_sdpa`, `sparse_sdpa_msa` |
| `tests/ttnn/unit_tests/operations/sdpa/sparse_sdpa_test_utils.py` | 83 | `sparse_sdpa` |
| `tests/ttnn/unit_tests/operations/sdpa/test_mla_prefill_v_embedding_space.py` | 123, 141 | `flash_mla_prefill` |
| `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py` | 299, 378, 436, 507, 640 | `scaled_dot_product_attention` |
| `tests/ttnn/unit_tests/operations/sdpa/test_sparse_sdpa.py` | 43, 108, 142, 165, 185, 200, 222, 242, 309, 337, 362, 398, 424, 463, 482, 519, 536, 542, 556, 562 | `sparse_sdpa` |
| `tests/ttnn/unit_tests/operations/sdpa/test_sparse_sdpa_msa.py` | 148 | `sparse_sdpa_msa` |
| `tests/ttnn/unit_tests/operations/sdpa/test_windowed_sdpa.py` | 110, 194, 306 | `scaled_dot_product_attention` |
