# AutoDebug: Qwen full-stack constructor exhausts L1 with layer-local constants

## Finding

**The leading diagnosis is constructor-resident L1 duplication across the decoder stack, amplified by inherited allocations and replacement-time overlap.** Each linear-attention layer retains its own identical 958,464-byte compact SiLU LUT. `QwenModel.layers` keeps every completed decoder alive. The next decoder first constructs the inherited single-chip helper, then allocates the selected TP4 helper's identical LUT before replacing the inherited helper. The reported failure occurs during that second upload, before model execution or request-cache allocation.

The immediate allocator failure is **fragmentation under high occupancy**: the request needs a contiguous 29,952-byte per-bank interval; 50,240 bytes are free in total, but the largest free block is only 18,432 bytes. The allocator is reporting an unsatisfiable placement, not demonstrating a wrong LUT size.

**Smallest proposed intervention:** in `tt/model.py`, immediately after each decoder finishes construction, adopt the first compatible linear layer's `decode_local_silu.table` tensor before constructing the next layer. Keep the selected helper instances, kernels, compute configurations, output sharding, weights, and request state intact. A model-owned immutable-tensor bundle can also deduplicate the fixed routing and mathematical constants listed below. A post-load pass over all 40 layers is too late. Constructor-time injection is an optional further reduction of the remaining transient peak.

Confidence is high in the source-level ownership/allocation diagnosis and exact failed tensor identification. Successful full-stack loading, execution headroom, trace behavior, and numerical equivalence after the proposed change require subsequent hardware validation; none was performed here. No implementation files were edited.

## Scope and evidence provenance

Inspected the prepared checkout at Git HEAD `121ab5df0f494787236fff2b371e422e6edb599d`, including its existing uncommitted full-model work. All model-relative references below are under `models/autoports/qwen_qwen3_6_35b_a3b/` unless a repository-relative path is given. This investigation used source/log inspection, AST comparisons, arithmetic, and an isolated CPU-only evaluation of the two mathematical SiLU-table functions. It did not import or execute the decoder, open devices, reset hardware, or invoke the AutoDebug launcher.

Current relevant SHA-256 values:

| File | SHA-256 |
|---|---|
| `tt/model.py` | `364cd2cca6a91bc4c74c088473db77c4c894fc8545b06ca8f6e6261c1f308978` |
| `tt/multichip_decoder.py` | `13b403518fb8b71dcf5b502bddeb9be79bcd6acf27b1bb51ab47c4ef902d952f` |
| `tt/fused_decoder.py` | `d8106611f65edccdf069baab44acc9c0088baa709090b017a6a6df75f0a7bb87` |
| `tt/mesh_decode_silu.py` | `01fb664e3ea3b90ded9bf80f56bd2769c9e53565a783f2055217b9687829268c` |

The current multichip hash matches the selected-source record in `doc/optimized_multichip_decoder/README.md:50`; the helper hash matches `doc/context_contract.json:667`. These records support preserving these selected implementations, but are not independent proof of full-model readiness.

There is a provenance limit: the full-load log has no matching source-hash sidecar in the inspected artifacts, and its traceback displays stale source text for the probe frame (`doc/full_model/logs/full_load.log:52`, showing an argparse line). The current probe hash also differs from the one recorded by `reduced_positive.json.provenance.json`, although the current model and generator hashes match that sidecar. Therefore the source interpretation is explicitly for the prepared tree; the matching decoder stack frames and allocation geometry support, but do not fully revision-pin, the historical failure.

### Direct observations

- `doc/full_model/logs/full_load.log:21–49` records successful layer-load messages for indices 0 through 28, followed by the L1 error at line 50. There is no layer-29 success message.
- The traceback at lines 55–69 names `QwenModel.__init__` → `MultichipDecoder.from_state_dict` → `install_linear_tp4` → `TP4TileSilu.__init__` → `self.table = t.from_torch(...)`.
- The allocator reports 958,464 bytes over 32 banks, 29,952 bytes per bank, capacity 1,460,288, allocated 1,410,048, free 50,240, largest free block 18,432.
- The last completed DRAM census reports 2,574,472,320 bytes free **per DRAM bank** (`full_load.log:49`). It does not measure L1 at the failure instant.
- The passing reduced artifact is `doc/full_model/reduced_probe.json`, with layers `[0,3]`, finite prefill/decode shapes, and position advanced to 5. The corresponding successful log is `logs/reduced_probe_uint.log:21–36`, including `FULL_MODEL_PROBE_OK` and device shutdown. `logs/reduced_probe.log` is an earlier failed run and must not be treated as the passing evidence.

### Source-derived interpretations

The bundled HF configuration has 40 layers: 30 linear-attention and 10 full-attention, repeating three linear layers then one full layer (`doc/functional_decoder/hf/config.json:19–75`). Thus indices 0–28 contain **22 linear and 7 full layers**, and index 29 is the 23rd linear layer. The reduced `[0,3]` case has only one linear layer and one full layer. Full-attention construction returns early from `FusedDecoder.from_state_dict` at `tt/fused_decoder.py:296`; it does not allocate the linear SiLU or Chunk64 helpers. This is duplication across a 40-layer model, not 40 copies of every linear-only tensor.

## Allocation and lifetime chain

| Phase | Source | Live allocation/lifetime implication |
|---|---|---|
| Model setup | `tt/model.py:65–78` | One shared `CollectiveWorkspace`; each completed decoder is appended to `self.layers` and remains strongly referenced. Only the host state dictionary is deleted. |
| Inherited construction | `tt/multichip_decoder.py:837`; `tt/optimized_decoder.py:657,405,43` | The entire `OptimizedDecoder` → `_ProjectionDecoder` → `_ExpertDecoder` → `FusedDecoder` setup runs before TP4 installation. |
| Inherited linear helper | `tt/fused_decoder.py:594–597,2937–2959` | Allocates layer-dependent weighted-norm gamma in L1 and a generic 32-worker `LocalSiluMultiply.table` in L1. |
| TP4 replacement | `tt/multichip_decoder.py:746–758` | Installs the TP4 decay/convolution/norm helpers, then evaluates `TP4TileSilu(ttnn, mesh)` while the inherited SiLU helper remains bound to `self.decode_local_silu`. |
| Failed upload | `tt/mesh_decode_silu.py:77–88` | Requests another identical 32-worker LUT. Python cannot replace/release the old attribute until the RHS constructor succeeds. |
| Later request setup | `tt/model.py:151–167`; probe `tests/run_full_model_probe.py:45–70` | Request cache allocation happens after the model constructor. It is not reached by this failure. Embedding/head/global RoPE construction at `tt/model.py:79–95` is also after the unfinished layer loop. |

At the reported layer-29 allocation, the current source predicts 22 retained selected LUTs plus the current decoder's inherited LUT: **23 × 29,952 = 688,896 bytes** of LUT storage on each of their common worker cores before the attempted selected upload. Success would briefly require a 24th shard, then return to 23 retained selected shards after replacement. This is not a claim that both helpers stay resident permanently in each completed layer.

The allocation is synchronous enough to throw from the host tensor-upload call; it does not require executing the SiLU kernel. Device writes may be enqueued, but that cannot remove the old helper's still-live Python reference before the replacement constructor completes. Cleanup in this checkout follows reference/destructor ownership (`ttnn/core/tensor/tensor.cpp:121–154`, `tt_metal/distributed/mesh_buffer.cpp:424–482`, `tt_metal/impl/buffers/buffer.cpp:767–810`). No delayed garbage-collection hypothesis is needed for the mandatory overlap. Additional runtime-held references cannot be excluded from this log.

The reduced probe actually reads logits to the host (`tt/model.py:206–209`; probe lines 74–85), so its passing evidence is stronger than merely reaching the end of an asynchronous forward. It still provides no full-stack capacity or accuracy guarantee.

## Exact size and capacity accounting

`SILU_COMPACT_ENTRIES = 7488` (`tt/fused_decoder.py:2882`). `TP4TileSilu` uploads a row-major FP32 tensor `[32,7488]`, height-sharded over 32 row-major worker cores, with shard `[1,7488]` (`tt/mesh_decode_silu.py:70–88`).

```text
one worker's shard = 7,488 entries × 4 bytes = 29,952 bytes
one chip's tensor  = 32 workers × 29,952    = 958,464 bytes
30 selected copies = 30 × 958,464          = 28,753,920 bytes/chip
one shared copy saves 29 × 958,464         = 27,795,456 bytes/chip
```

The mesh allocation provides this local tensor on each chip; TP4 does not divide these 32 LUT rows among four chips. This is deliberate per-worker lookup replication, multiplied unnecessarily by the number of layers. The proposed sharing removes the layer multiplicity and preserves the per-worker replication.

The allocator uses the shard count as the bank count (`tt_metal/impl/allocator/bank_manager.cpp:469–482`) and computes the per-bank extent from aligned pages (`tt_metal/impl/allocator/allocator.cpp:784–793`). The request matches the source exactly. The failure at `bank_manager.cpp:495–511` emits the allocator's actual statistics after allocation fails:

```text
29,952 required > 18,432 largest free interval
29,952 required < 50,240 total free
```

This establishes fragmentation at that instant. Coalescing adjacent freed blocks does not move live tensors separating holes (`tt_metal/impl/allocator/algorithms/free_list_opt.cpp:267–333,424–451`). The artifacts do not supply allocation identities or hole addresses, so the precise hole history and all owners of the logged 1,410,048 allocated bytes remain unaccounted for. Do not equate a source-derived tensor sum with that statistic.

There is also a stronger full-depth capacity contradiction. Even excluding almost all DRAM constants, CCL semaphores, activations, kernel CBs and constructor transients, these retained allocations all occupy worker core `(0,0)`:

| Full-40 persistent data on core `(0,0)` | Calculation | Bytes |
|---|---|---:|
| Selected compact SiLU LUTs | 30 × 29,952 | 898,560 |
| Selected weighted-norm gamma | 30 × (32 × 128 × 4) | 491,520 |
| Inherited router bias, indices, output, output IDs | 40 × (4 × 2,048) | 327,680 |
| Inherited inline-selector values and IDs | 30 × (2,048 + 32) | 62,400 |
| **Data-only minimum** | | **1,780,160** |

Gamma is allocated by `WeightedNormTail` at `tt/fused_decoder.py:2383–2398` and retained by the TP4 replacement at `tt/multichip_decoder.py:752`. Router tensors are created on core `(0,0)` at `tt/fused_decoder.py:270–290` for full layers and `:527–545` for linear layers. Inline selectors are retained only on the linear branch (`:573,1928–1957`). Tile padding makes BF16/uint16 router tensors and selector values 32×32, or 2,048 bytes; the row-major selector ID shard is 1×16 uint16, or 32 bytes. These are separate allocations, including unused inherited fields.

That minimum exceeds the logged allocatable bank capacity by **319,872 bytes**. All terms overlap on core `(0,0)`, so unused capacity on other cores cannot invalidate the bound. The current unshared full stack cannot fit in the reported L1 arena even with perfect packing. This bound explains why a one-linear-layer smoke does not establish full-depth viability; it does not predict the exact first failing index independently of the observed allocator state.

`QwenModel.census` measures only DRAM (`tt/model.py:113–118`). It has no L1 capacity/fragmentation admission check. Its healthy DRAM progression therefore misses the limiting resource.

## Exact immutable sharing candidates

Share tensor storage, preserving each tensor's existing dtype, layout, padded shape, memory configuration, per-chip contents and ownership map. A semantic constant name plus matching physical metadata is an appropriate key within one model. Shape equality alone is not.

| Candidate | Contents and placement | Construction evidence / sharing constraints |
|---|---|---|
| **`decode_local_silu.table`** | Generic compact SiLU table; FP32 row-major `[32,7488]`, 32-core height-sharded L1 | `tt/mesh_decode_silu.py:77–88`; inherited version `tt/fused_decoder.py:2948–2959`. Independent of weights, layer index and requests. Highest-priority fix. Preserve selected helper/output layout. |
| `router_bias`, `router_indices` | Zero bias and fixed transposed expert-index ordering; BF16/uint16 TILE, single-core L1 | `tt/fused_decoder.py:283–285,538–541`. Physical padding is 32×32. Both are inputs; the similarly named output tensors are mutable and excluded. |
| `silu_table` | Full mathematical SiLU over all 65,536 BF16 codes; `[1,65536]` FP32 row-major DRAM | `tt/fused_decoder.py:574–581`; consumed by gather at `tt/multichip_decoder.py:533`. Linear layers only. Distinct from the compact L1 table. |
| `combine_bias`, `constants.{eye,tril,ones,masks}` | Zero combine bias; fixed 32×32 matrices and 32×96 block masks; BF16 bias and FP32 matrices, TILE DRAM | `tt/fused_decoder.py:202–213,459–470`. Some are retained inherited data with no consumer on the selected multichip path; sharing them is still safe, while removal is a separate cleanup. |
| `attention_k_indices`, `attention_pair32_indices`, `output_pair32_indices`, `router_pair32_indices` | Fixed pair-swap permutations, uint16 DRAM; K index tiled `[1,1,2048]`; pair tables row-major `[1,64,32]` or `[1,128,32]` | `tt/fused_decoder.py:370–379,555–572`. Preserve logical shape/layout. Attention/router pair tables have equal values/geometry and may alias if their full tensor contracts match. |
| `indexed_sparsity` | All ones `[1,1,1,256]`, BF16 row-major DRAM | `tt/fused_decoder.py:291–293,546–548`; input to sparse matmul (`tt/multichip_decoder.py:1359`). |
| `expert_ids`, `ep_reduce_table`, `dispatch_table`, `local_expert_mask` | Fixed expert IDs and rank ownership maps; DRAM, with the existing mesh sharding | `tt/multichip_decoder.py:947–961`. Key on `local_expert_dispatch` and ownership geometry. IDs are uint32; tables int32; local mask FP32 TILE. Preserve the existing alias `dispatch_table is ep_reduce_table` under the default policy. |
| `mask_zero`, `mask_one`, `expert_prefix_sum` | BF16 DRAM constants: row-major zeros `[1,1,2048,256]`, row-major ones `[1,1,2048,8]`, tiled upper-triangular ones `[256,256]` | `tt/multichip_decoder.py:962–964`. Used as inputs at `:1384–1399`; `mask_zero` is not an in-place scatter destination (see audit below). |
| `chunk64_state.{lower,strict,eye}` | Fixed FP32 TILE DRAM `[1,1,64,64]` triangular/identity matrices | `tt/fused_decoder.py:2030–2032`; read by state graph at `:2070–2088`. Share fields without merging graph objects. |
| `chunk64_state.mkl_negative_exp.{high,low,one_bits,exponent_bias}` | Mathematical exp tables and uint32 bit constants, TILE DRAM | `tt/fused_decoder.py:2233–2253`. Final TP4 high/low tables use `[1,8,64,128]` FP32 (`tt/multichip_decoder.py:765–767`), together 524,288 bytes/chip. Preserve `max_shape=(1,8,64)`; default `(1,32,64)` is a different signature. Bit constants are logically `[1,1,1,1]`, physically tiled. |

The compact LUT's mathematical construction uses only BF16 code points and SiLU (`tt/fused_decoder.py:2887–2926`). Its reader accesses a `const volatile uint32_t*` and only loads table entries (`:2883`). Output/debug tensors are separately allocated and bound to output CBs; the table is an input at `tt/multichip_decoder.py:131–173`. The tile specialization changes task enumeration and loop bounds (`tt/mesh_decode_silu.py:34–56`), not table contents or writes. AST comparison found the inherited and selected 32-worker table-upload expressions identical. CPU-only extraction of the two mathematical functions passed their bitwise reconstruction assertion for **all 65,536 BF16 encodings**, including the exceptional-value encodings covered by that assertion.

`mask_zero` was checked through lowering rather than inferred from its name. `ttnn.scatter` returns a new primitive result (`ttnn/cpp/ttnn/operations/data_movement/scatter/scatter.cpp:321–331`); the device operation allocates fresh output storage (`.../device/scatter_device_operation.cpp:71–73`). On Blackhole, the reader copies the input into output scratch (`.../device/kernels/dataflow/reader_scatter.cpp:139–166`), and the writer writes the output tensor (`.../writer_scatter.cpp:18–30`). Sharing the zero template does not accumulate selections across layers.

## State that must retain its current ownership

| State | Why it is not a generic immutable constant |
|---|---|
| `decode_weighted_norm_tail.gamma`, `output_norm_bf16`, norms/sharded norms, all projection/router/expert weights and packed/DRAM-sharded representations | Immutable after loading but **weight-dependent**. `WeightedNormTail.gamma` expands the layer's actual norm weight (`tt/fused_decoder.py:2396–2398`). Identical shape is not identical data. The model's "shared expert" is shared among tokens, not necessarily among layers. |
| `gate_table`, `decay_table`, any compact decay table and its clamp/index bounds; convolution taps | Depend on each layer's `A_log`, `dt_bias` or convolution weights (`tt/multichip_decoder.py:728,741–751`). The large decay LUT is not interchangeable with the generic SiLU LUT. Retain the selected uncompressed exact per-layer decay path. |
| `router_out`, `router_out_ids`; `InlineSelector.values/ids`; `MeshFP32TopK.values/ids` | Mutable output scratch, despite zero initialization. The generalized gate receives explicit output tensors; inline selection writes values/IDs; FP32 top-k rewrites its output tensors and descriptor arguments (`tt/fused_decoder.py:1378–1379,1965–1999`; `tt/mesh_fp32_topk.py:65–73,123–134`). Sharing them would require a separately proved serial-workspace contract. Leave them alone in the minimal fix. |
| Whole Chunk64 graphs and selector objects | Graph matmul helpers update `last_plan` (`tt/fused_decoder.py:2770,3470`); selectors mutate runtime arguments. Tensor-only dedup avoids introducing shared mutable object state. `MeshFP32TopK.input_template` itself is a read-only descriptor placeholder, but its program initially stores that address (`tt/mesh_fp32_topk.py:94–102`); leave the helper intact in the first change. |
| Recurrent and convolution histories | Per-layer, per-request mutable data, created by `tt/multichip_decoder.py:598–617` and updated at `:423–435,488–508`. They cannot alias between layers or requests. |
| K/V caches, page ownership, positions, execution/trace and sampler buffers | Per-layer/request or execution state. K/V and page tables are allocated at `tt/multichip_decoder.py:1621–1657`; decode mutates caches at `:1697–1699` and positions at `tt/model.py:202–203`. Preserve existing disjoint-page request pooling within each full-attention layer (`tt/model.py:155–165`). |
| `CollectiveWorkspace` CCL state, EP outputs, direct-all-reduce buffers/semaphore | Already shared intentionally by this model (`tt/model.py:65,73`; `tt/multichip_decoder.py:770–784,900–904`). Mutable serial-CQ0 workspace whose stated contract prohibits concurrent forwards. It is not an immutable-data cache. |

## Minimal model-local intervention

1. Add a per-`QwenModel` dictionary of canonical immutable tensor references. Construct one decoder, immediately adopt compatible constants, then append/continue. The first change can be restricted to the selected compact SiLU tensor; adding the listed fixed routing/mathematical tensors reduces other duplication without changing the computation.
2. Retain each `TP4TileSilu` instance and replace only its `.table` reference. Key/validate the semantic LUT variant, mesh identity, dtype, logical/padded shape, row-major layout, L1 height-shard specification, core set and orientation. Keep alternative eight-worker `TP4LocalSiluMultiply` and different policies separate.
3. Perform all adoption before forwards, preparation, warmup or capture. Keep canonical tensors strongly owned until all dependent traces are released and work has completed. Do not force-deallocate a canonical tensor or mutate it after capture. Ordinary replacement of an unshared duplicate reference is sufficient; do not use a process-global device cache or deduplicate arbitrary uploads/weights by shape.
4. For the extra bundles, adopt individual tensor fields, preserve policy-dependent aliases/maps and MKL `max_shape`, and avoid retaining unused temporary constructor bundles in the cache. Do not merge entire layer/helper objects or redirect weight/state fields.

Illustrative control flow only; no implementation was applied:

```python
self.immutable_tensors = {}
self.layers = []
for index in self.layer_indices:
    layer = MultichipDecoder.from_state_dict(...)  # existing defaults
    adopt_compatible_immutable_tensors(layer, self.immutable_tensors)
    self.layers.append(layer)
```

Before the observed failing constructor, this changes the preceding 22 retained LUTs to one, removing **21 × 29,952 = 628,992 bytes on each common worker core**. The current decoder still temporarily creates its inherited and selected LUTs; the model-level adoption keeps that transient fixed in table count rather than growing with depth. It is therefore a strong, narrow intervention at the depth-growth boundary, with sufficient source-level headroom to merit the first hardware check. It is not proof of final placement or execution success.

If residual constructor peak remains limiting, let `TP4TileSilu` accept the inherited helper's already-compatible `.table` during replacement, avoiding the second fresh upload. To prevent even the inherited fresh upload, pass a model-owned constant bundle into the inherited constructors before allocation; that requires threading an optional argument through the factory chain. Both are broader changes than immediate adoption. Sharing the inherited **helper object** is incorrect because its output contract is `[1,1,4096]` with `[32,128]` width shards, whereas the selected tile helper produces `[1,1,1024]` with `[32,32]` width shards (`tt/fused_decoder.py:2932–2946`; `tt/mesh_decode_silu.py:5–8,72–75`).

No kernel substitution, precision reduction, LUT compression, worker reduction, L1-to-DRAM relocation, collective-policy change, or request-state sharing is needed for this first intervention.

## Claim review and competing explanations

| Candidate | Evidence and match to observations | Adjudication / smallest intervention |
|---|---|---|
| Retained generic L1 tables plus other constructor state | Exact failed tensor and size; retained layer list; 30-layer L1 lower bound exceeds capacity; one-linear-layer reduced pass | **Leading cause.** Adopt compatible constant tensors after every successful layer. Runtime fit remains to be measured. |
| Inherited replacement peak | Old helper necessarily remains live while the failed RHS upload executes | **Proved amplifier.** Reuse its table during TP4 replacement if needed. This alone does not remove the completed stack's linear growth. |
| Fragmentation / allocator defect | Largest free block is smaller than request, despite larger total free | **Observed fragmentation, no demonstrated allocator bug.** Source supplies many live allocations and allocate/replace/free activity. Exact hole owners are unavailable; no allocator rewrite justified. |
| DRAM exhaustion, KV/context size, model-head/embedding or full-model execution | Wrong memory family; unfinished decoder loop precedes request pools, terminal weights and execution | **Do not explain this failure location.** Their later capacity remains unverified. |
| Wrong SiLU LUT arithmetic, task mapping, kernel precision or numeric drift | Failure is table allocation before the new helper can execute; uploaded size matches construction; table has no weights/state | **Not supported as the cause.** Preserve selected arithmetic/kernel path. CPU reconstruction tests storage contents, not device execution. |
| Delayed cleanup / asynchronous error surfacing | Runtime-held references are not mapped by the log | **Not needed and unproven.** Mandatory overlap exists even with immediate cleanup. Capture of allocation lifetimes would test additional retention. |

The first draft's headline claims were rechecked against the early full-attention return, concrete layer counts, assignment ordering, physical shard sizes, mutable output bindings, scatter lowering and allocator statistics. Claims of an exact decomposition of the logged allocation total, proof of the exact layer-29 threshold from source alone, a hardware-proven repair, or 40 copies of each linear-only table are deliberately excluded.

## Other potential issues and follow-up checks

- **Inherited unused allocations:** selected multichip `_moe` uses `MeshFP32TopK`/native top-k (`tt/multichip_decoder.py:1308–1329`), while inherited router buffers and linear `InlineSelector` remain allocated. The `constants` dictionary and `attention_k_indices` also have no reads in these decoder implementations. These explain avoidable residency, but removing them should be a separate reachability-reviewed cleanup; the minimal sharing proposal leaves mutable scratch intact. Existing parent weight cleanup already removes some replaced representations (`tt/optimized_decoder.py:481–486`; `tt/multichip_decoder.py:925–930,709–726`), so it would be wrong to label every parent allocation permanently leaked.
- **Chunk64 construction churn:** the parent creates a graph/default MKL bundle; TP4 creates another graph/default bundle, then replaces its MKL helper with the eight-head version (`tt/fused_decoder.py:589–591,2033`; `tt/multichip_decoder.py:762–767`). These are DRAM constructor transients, not the failed L1 allocation. Final tensor adoption removes retained duplication; earlier injection could separately remove churn.
- **Missing resource visibility:** the next authorized hardware validation should record L1 allocated/free/largest-free values before/after each layer and, if available, allocation owners/addresses. Record both the post-adoption resident value and constructor peak; DRAM-only census cannot distinguish them. Exact hole attribution is the principal missing allocation evidence.
- **Discriminating repair validation:** first repeat all-40 load-only with source hashes and the same mesh/default policy. Check that all 30 selected SiLU helpers retain one compatible tensor per mesh, weight/state objects remain distinct as required, selected decoder/helper source hashes are unchanged for a model-only change, and the layer-29 peak/headroom improves. Then run the existing reduced real-weight forward check and appropriate full-stack correctness and trace/replay checks. Logit finiteness alone does not establish HF accuracy or trace-safe ownership. Check normal release after traces and model teardown.

These are proposed follow-ups, not experiments performed by AutoDebug. The only new file produced by this investigation is this report.
