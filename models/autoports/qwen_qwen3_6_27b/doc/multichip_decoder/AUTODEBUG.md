# AutoDebug: TP4 decoder inspection

## Findings

**The initial inspected revision had one verified forward blocker: distributed RMSNorm received rank 3, while its selected C++ implementation required rank 4.** Its GDN decode also changed the optimized numerical operation graph; that discrepancy was verified, but its accuracy impact was not measured. Its recurrent cache was already BF16, and the native operation did widen it for FP32 computation: a claim that the cache simply had the wrong dtype would be false.

**Revision update:** final review detected an external source edit implementing the rank fix, a mesh-topology guard, and the local-12 recurrence/postprocess. I inspected that later revision too. Its recurrence wrapper, reader, and partition match the precise specialization below; no additional concrete default-path defect was established. Hardware correctness and trace replay remain unverified. Historical findings below must not be read as unresolved defects in that later source.

The requested local-12 specialization of the optimized recurrence is specified below. Changing Python head counts alone is insufficient: the embedded reader's gate batch stride, row-major page sizes, packed byte offsets, and tiled batch strides must change together. The optimized source need not change.

Inspection only. No device execution, device health probe, reset, target-code import, or source edit was performed. This report is the only file written by this investigation.

## Evidence and revision boundary

Inspected checkout HEAD: `0915d93157bd320c7875533fba09bfbffeb2ebeb`.

| Inspected source | SHA256 |
|---|---|
| M: initial multichip source | `1700cb3835d3f42ef0c13191b5c243dc41cf1f64a6956a970ad226f5cb763e2b` |
| [M1: externally revised multichip_decoder.py](../../tt/multichip_decoder.py) | `e95f63f2477d5d38b5876ba0440d3a88abd8d5d56a2643816703035e415e69ac` |
| [O: optimized_decoder.py](../../tt/optimized_decoder.py) | `315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22` |

`M:<line>` refers to the initial multichip contents and line numbers, observed before the external edit; `M1:<line>` refers to the later contents. `O:<line>` refers to the unchanged optimized source. The initial multichip contents are not a separate archive on disk; the relevant original expressions and proposed corrections are preserved in this report. C++ paths below are relative to the repository root. The multichip source is untracked in this checkout; HEAD alone does not identify its contents.

**Direct observations from supplied artifacts:** `tp4_initial_l0.log` stopped at `open_mesh_device(physical_device_ids=None)`; `tp4_initial2_l0.log` reached model setup and stopped because the convolution exchange-core helper received a MeshDevice instead of a grid. Current runner lines 54–58 use `[]`, and M:138–140 supplies `compute_with_storage_grid_size()`, respectively. Those two reported errors are corrected in the inspected tree. Neither log reaches a forward, and neither supplies evidence of TP4 output accuracy or a trace failure. `baseline_l0.json` records a single-device B1/S33 baseline only. Its `source_sha256` field hashes the multichip file even for baseline runs (runner:243), so it is not an optimized-source provenance check.

**Interpretations:** the rank defect below predicted the initial source's next default-sharded forward failure if setup completed. It does not explain the earlier helper TypeErrors and is addressed in M1. Numerical drift and trace replay failure remain unobserved by this investigation.

### Final review of the external revision

The source changed outside this investigation while the report was being finalized. I made no source edits. M1 has:

- Rank-4 pre/post distributed norm with rank-3 collective adapters at M1:275–291, matching section 1's intervention.
- An explicit `(1,4)` mesh guard at M1:36.
- Local recurrence selection before gate sequence padding at M1:404–419; separate BF16 affine and FP32 SILU/gate boundaries are restored there.
- A local reader at M1:464–568, partition at M1:571–587, and wrapper at M1:590–727. An in-memory AST comparison proved the wrapper equals the optimized wrapper with exactly the geometry/name substitutions in section 3. An exact reader-string comparison proved all and only the specified page/offset/head edits. The partition AST likewise differs only in name and `48→12` head totals.
- Both BF16 and FP32 active state support, unchanged imported compute/writer/configuration, 16/48 raw tile strides, `user+head/32` gate pages, and trailing writer BH=`12*B`.

The actual M1 partition passed the same 2,016 coverage/grouped-QK-lifetime cases described at the end of this report. These are static checks; they do not prove kernel compilation, arithmetic accuracy, buffer allocation, completion, or trace safety. The remaining nondefault-policy/test gaps in section 5 still apply, except that its missing-topology-guard issue is repaired.

## 1. Verified blocker: rank mismatch at the first distributed norm

M:262–271 sends device DRAM-interleaved `x` directly to `ttnn.rms_norm_pre_all_gather`. Both the runner and the decoder contract supply rank-3 residuals `[B,S,1280]`. Inherited prefill calls this norm at O:1180–1183; decode calls it at M:417, before attention or its state update.

The lowering is unambiguous:

1. `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_distributed_nanobind.cpp:79` directly binds the public function. `ttnn/cpp/ttnn-nanobind/bind_function.hpp:55–57` forwards its arguments; there is no rank-expanding adapter.
2. `.../rmsnorm_distributed/rmsnorm_pre_all_gather.cpp:26–50` selects the interleaved primitive and forwards the input unchanged. M:265 explicitly makes this an interleaved input, even though it is sharded across mesh devices.
3. `.../layernorm_distributed/device/layernorm_pre_all_gather_device_operation.cpp:91–97` copies the logical shape and writes `output_shape[3] = tile_width` for RMSNorm.
4. `tt_metal/api/tt-metalium/shape_base.hpp:25–30,65–72` rejects a nonnegative index at or above the original rank. Index 3 is invalid for `[1,33,1280]`, `[1,1,1280]`, and every other rank-3 shape here.

This is a host output-sizing contract violation, before this norm dispatches or calls CCL. The replicated-residual control uses width 5120 and bypasses this branch, so it can avoid this specific defect. Both attention kinds are affected under the default sharded layout.

**Smallest local intervention:** make only the distributed norm's pre/post boundary rank 4. Keep `_collective` rank 3; it already prepends its own singleton at M:239 and always gathers dimension 3. Passing rank 4 directly into the current `_collective` would create rank 5 and select the wrong dimension.

```python
# Within MultichipDecoder._norm, after the existing width guard:
shape = tuple(x.shape)                         # [B,S,1280]
x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
x4 = ttnn.reshape(x, (1, *shape))
stats4 = ttnn.rms_norm_pre_all_gather(
    x4, compute_kernel_config=self.ckc, dtype=ttnn.bfloat16
)                                             # [1,B,S,32]
stats3 = ttnn.reshape(stats4, tuple(stats4.shape)[1:])
stats3 = self._collective(stats3)               # [B,S,128]
stats4 = ttnn.reshape(stats3, (1, *tuple(stats3.shape)))
normalized4 = ttnn.rms_norm_post_all_gather(
    x4, stats4, epsilon=self.config.rms_norm_eps,
    weight=self.shard_norms[key], compute_kernel_config=self.ckc,
)
return self._collective(ttnn.reshape(normalized4, shape))  # [B,S,5120]
```

This is a proposed adapter, not an applied or device-tested fix. Preserve rank 4 for post-norm too: its validator compares positive dimensions 0, 1, and 2, and its program setup assumes four dimensions (`.../layernorm_post_all_gather_device_operation.cpp:64–78`).

## 2. Verified numerical discrepancy: decode selects a different GDN algorithm

The optimized defaults enable compact SFPU recurrence with decay/prediction fusion (O:31–35,56,72). O:928–953 selects `_fused_delta_row_recurrent_sfpu_compact`. Initial M overrides `_gdn` and always selects the public chunk operation (M:386–402), including single-token decode. Changing policy flags alone cannot route that initial override through the optimized helper. M1 now selects its local specialization for S=1.

These are distinct finite-precision graphs:

| Phase | Optimized compact decode | Initial TP4 decode (M) |
|---|---|---|
| Q/K normalization | FP32 SFPU squares/reduction; normalize Q, then separately scale Q (O:5383–5406) | Native prep uses a matmul-based norm and folds Q scale into inverse norm before applying it |
| Update | Decay S; contract K with decayed S; form `(V - prediction) * beta`; add outer-product update | Chunk/WY preparation forms `V*beta` and `K*beta`, then scan combines matrix terms |
| Readout | Contract Q with the live FP32 updated state, rounding a separate persistence copy (O:5501–5531) | Native chunk scan combines prior-state and within-chunk matmul terms |
| Gated normalization | BF16 normalized result, distinct BF16 affine multiply, FP32 SILU/gate multiply to BF16 (O:956–983) | Weighted RMSNorm followed by a standalone BF16 SILU and multiply (M:405–412) |

Native evidence: `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gdn_math.hpp:426–487,840–844,1058–1065,1111,1143`.

These formulas implement the same real-arithmetic recurrence, but equality of their floating-point results does not follow. This is a verified departure from the optimized numerical path, **not proof of a PCC failure, incorrect cache dtype, or a trace hang**. It particularly affects decode and single-token prefill chunks: the optimized `_gdn` also routes S=1 through compact recurrence at O:830–831. Ordinary multi-token prefill uses native chunk recurrence in both implementations.

The BF16/FP32 state lifecycle is coherent:

- M:162 defaults uploads to BF16; M:290 therefore creates BF16 persistent state `[B,12,128,128]`.
- Inherited prefill widens that cache once, keeps FP32 state across all chunks in that call, and rounds back once at the end (O:1169–1171,1205–1206). M:403 writes the update into whichever dtype that active state has.
- The public chunk wrapper itself widens BF16 initial state to FP32 (`chunk_gated_delta_rule.cpp:281–290`). It computes FP32 outputs on the flat phased/fused paths. M:403 publishes a BF16 cache during decode, after computing the output.
- Same-dtype `typecast(..., output_tensor=...)` does honor its destination (`ttnn/cpp/ttnn/operations/copy/typecast/typecast.cpp:43–54`; `device/typecast_device_op.cpp:149–173`). It is not a no-op that loses prefill state updates.

**Intervention boundary:** if preserving optimized decode arithmetic is required, specialize the recurrence locally as below and preserve its post-recurrence affine/gate boundaries. Merely changing cache dtype, setting `fp32_dest_acc_en`, or enabling an inherited policy flag does not accomplish that. The retained TP4 convolution and distributed norm still need numerical comparison; recurrence specialization alone does not prove full-decoder equivalence.

## 3. Precise local-12 adaptation; optimized source remains unchanged

Create a model-local helper by copying O:5565–5702 and a local copy of its reader O:4829–4933. Reuse the unchanged optimized compute source, writer source, compute-config builder, and fusion constants. Give the local wrapper/reader/partition distinct names. Do not mutate optimized module globals or globally replace `16`/`48` in kernel text: many occurrences describe tile geometry rather than head counts.

### Python wrapper and head partition

Apply every entry in this table to the local wrapper:

| Contract | Optimized | Local TP4 rank |
|---|---|---|
| Infer batch from state | `state.shape[0] // 48` | `state.shape[0] // 12` |
| State shape | `(B*48,128,128)` | `(B*12,128,128)` |
| Aliased packed RM Q/K/V | `(B,1,10240)` | `(B,1,2560)` |
| Normalized FP32 Q/K | `(B,16,128)` | `(B,4,128)` |
| Raw BF16 Q/K width | `2048` | `512` |
| Raw BF16 V width | `6144` | `1536` |
| G/beta shape | `(B,1,48)` | `(B,1,12)` |
| FP32 output shape/assertion | `(1,B*48,128)` | `(1,B*12,128)` |
| Raw tiled Q batch stride, `rct[3]` | `64` | `16` |
| Raw tiled V batch stride, `rct[4]` | `192` | `48` |
| Writer trailing BH compile argument | `B*48` | `B*12` |

Reader compile arguments otherwise retain their exact order: normalized, G-FP32, beta-FP32, Q stride, V stride, row-major, packed-QKV, state tile bytes, then the six tensor-accessor descriptors. Runtime arguments retain all six tensor addresses followed by head start/count. Select the local reader and local partition; retain the existing compute and writer.

The exact local partition can be:

```python
def _local12_head_partition(batch, available_cores, cores=None, grouped_qk=True):
    assert batch >= 1 and available_cores >= 1
    assert cores is None or cores >= 1
    grouped = bool(grouped_qk and batch >= 3)
    units = batch * 12
    ncores = min(units, available_cores, cores or available_cores)
    result, start = [], 0
    for index in range(ncores):
        count = units // ncores + (index < units % ncores)
        result.append((start, count))
        start += count
    assert start == batch * 12
    return grouped, result
```

Do not force three-head-aligned partitions. The compute already prepares Q/K at a core's first head or a global `%3 == 0` boundary, consumes duplicate raw Q/K on the other heads, and releases the prepared pair at `%3 == 2` or the core's last head (O:5453–5461,5537–5538). Twelve heads per user is divisible by three, so those group boundaries remain valid across users. Keeping the batch>=3 heuristic is correct, even when short local partitions offer no Q/K reuse.

### Embedded C++ reader: exact changed expressions

Replace O:4847–4849 in the local reader with:

```cpp
const auto q=TensorAccessor(qa,get_arg_val<uint32_t>(0),
    PACKED?5120:(ROW_MAJOR?1024:(NORMALIZED?4096:2048)));
const auto k=TensorAccessor(ka,get_arg_val<uint32_t>(1),
    PACKED?5120:(ROW_MAJOR?1024:(NORMALIZED?4096:2048)));
const auto v=TensorAccessor(va,get_arg_val<uint32_t>(2),
    PACKED?5120:(ROW_MAJOR?3072:2048));
```

These page sizes are bytes: separate BF16 RM Q/K rows have `512*2=1024`, V has `1536*2=3072`, and packed QKV has `2560*2=5120`. Normalized FP32 tile size remains 4096, raw BF16 tile size remains 2048.

Replace the gate page calculation at O:4882 and the head map at O:4887:

```cpp
// Gates are [B,1,12], physically [B,32,32]: one tile per user.
const uint32_t page=user+head/32;       // old: user*2+head/32
// Flattened state head h is local to this chip.
const uint32_t user=h/12,hv=h%12,hk=hv/3;
```

Keep the gate face offset `((head%32)/16)*256*element_bytes` and extraction `hv%16` unchanged. Local heads 0–11 occupy the first face. **Concrete boundary:** with B=2 there are two gate tiles, IDs 0 and 1. User 1/head 11 must read page 1. Leaving `user*2` reads page 2 outside that tensor. B=1 cannot expose this missed adaptation. The current native reader is already generic and has no such defect; this warning concerns copying the optimized helper.

Replace the packed K/V offsets at O:4898–4899:

```cpp
noc.async_read(k,scratch,256,
    {.page_id=user,.offset_bytes=(PACKED?1024:0)+hk*256},
    {.offset_bytes=512});
noc.async_read(v,scratch,256,
    {.page_id=user,.offset_bytes=(PACKED?2048:0)+hv*256},
    {.offset_bytes=1024});
```

The Q read remains unchanged. Every head is still 128 BF16 elements, hence 256 bytes per read. Separate RM K/V rows still use offset zero before the head offset. Scratch offsets 0/512/1024/1536/1600 remain unchanged.

**Unchanged normalized branch:** `faces(q, user*4, hk, 4, ...)` and the analogous K read remain correct. `[B,4,128]` pads to `[B,32,128]`, still four tiles per user. Replacing that `user*4` with `user*1` would introduce another bug. Raw tiled Q/K use the new 16-tile batch stride; raw tiled V uses 48.

**Unchanged writer:** state addressing remains `h*16+i`; output row remains `h%32`, page `(h/32)*4+i`. Pass `BH=B*12` and retain its `BH%32` padding-clear branch (O:4935–4972). For B1, the writer must zero rows 12–31; for B2, rows 24–31. No 48-head assumption remains in the writer itself.

### Preserve precision, ownership, and queue transitions

Keep state as a view of the persistent `[B,12,128,128]` allocation, reshaped to `[B*12,128,128]`. BF16 state uses 2048-byte tiles and the original config's default unpack for CBs 5/11; FP32 state uses 4096-byte tiles. All other FP32 CBs retain direct FP32 unpack, full DST synchronization, and the unchanged compute arithmetic (O:4814–4825,5176–5240,5501–5531).

The live updated state supplies FP32 readout. `_gdn_store_bf16_rne_` rounds a copy for persistence; do not cast the full updated state to BF16 and reload it before Q readout. The helper mutates state in place and returns FP32 `[1,B*12,128]`; no native final-state tensor/copy is needed on this path.

| CB | Capacity, tiles | Per-head production/consumption |
|---|---:|---|
| Raw Q/K/V, 0/1/2 | 8 each, FP32 | Reader reserves/pushes 4 each; compute consumes all 4, including duplicate Q/K on cached-group heads |
| G/beta, 3/4 | 2 each, FP32 | Reader pushes 1 each; compute consumes 1 each; G is converted to decay CB18 |
| State/updated, 5/11 | 32 each, state dtype | Reader pushes 16 state tiles; compute pops 16 and pushes 16 updated; writer pops updated only after NoC completion |
| Readout, 12 | 8, FP32 | Compute pushes 4; writer waits for 4, writes live row plus final padding, barriers, then pops 4 |
| Prepared Q/K, 13/14 | 1 each, FP32 | Produced once per retained Q/K group and released at the group/core boundary |

The original allocation is two heads deep for raw/state/output rings despite older one-head comments. Keep its sizes. Zero/one CBs 6/7 are published once; scratch CB15 remains private to the reader; norm/rows/decay CBs 16/17/18 keep sizes 1/4/1. Total unchanged CB allocation is 328 KiB/core for BF16 state, 456 KiB/core for FP32 state. Head dimension 128 and sixteen state tiles per head do not shrink with TP.

At 110 available cores, local partition examples are B1: 12 cores ×1 head; B2: 24×1; B16: 110 cores with 1–2 heads; B32: 110 cores with 3–4 heads. For B32 there are 384 heads, not 1536. Individual ranges cover each head exactly once, including ranges that split triples.

### Hook it into the TP4 path

For the smallest recurrence adaptation, use the already-produced TILE Q/K/V from M:380 with `normalized=False, packed_qkv=False`. Their shapes `[B,32,512]` / `[B,32,1536]` match the specialized raw interface. Form G/beta as `[B,1,12]` and invoke the helper **before** M:384–385 pads their sequence dimension for the native chunk operation. Pass the persistent state view `[B*12,128,128]`.

Keep a separate multi-token native-prefill branch. Inherited prefill can pass an FP32 active state, and the specialized helper must retain support for both state dtypes if also used on its S=1 tail. Do not call the unchanged optimized wrapper directly: even local B1 has 12 state heads, and its `//48` inference and shape checks reject that input.

To match the optimized compact postprocess, retain flattened heads `[1,B*12,128]` through BF16 cast and unweighted RMSNorm, apply norm weight in a distinct BF16 multiply, then device-reshape to the local output width 1536. Apply SILU to FP32 Z and multiply FP32 normalized/affine output, producing BF16, before the local output projection. Local Z is `packed[...,2560:4096]`. Do not reuse the unchanged optimized affine/gate or beta/decay kernels: they separately hardcode 48 heads and global packed offsets (O:5705–5918,5922–6111). Standalone device operations suffice for this local adaptation.

## 4. Audited and cleared on the intended 1×4 default path

### Packed projection, correction, and head ownership

| Region | Local columns | Live width |
|---|---|---:|
| Q | 0–511 | 512 = 4×128 |
| K | 512–1023 | 512 |
| V | 1024–2559 | 1536 = 12×128 |
| Z | 2560–4095 | 1536 |
| Beta logits | 4096–4127 | 12 plus 20 zeros |
| Decay logits | 4128–4159 | 12 plus 20 zeros |

M:98–112 builds exactly these regions. M:195 zeros only Z correction; M:382–383 uses the live beta/decay extents. `narrow_gdn_correction=False` selects the generic FP32-primary-plus-FP32-correction → BF16 combine (O:443–445), avoiding the global narrow offsets. All local boundaries are tile aligned. Global rank r owns key heads `4r..4r+3` and value heads `12r..12r+11`; global `(12r+hv)//3 = 4r + hv//3`, so contiguous weight/tap/state partitioning preserves GQA ownership.

With eight DRAM banks/readers2/cores8, physical packed and MLP gate/up N round from 4160/4352 to 4608. `logical_n` is saved and inherited projection code crops back before packed slicing (M:208–234; O:607–609). Local K dimensions 5120/1536/4352 are 160/48/136 tiles. Prefill block4 divides all three. The apparent MLP-down shard `17 % block2` error is **not** a default-path bug: its qualifying multicore pipeline waives that divisibility rule (`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:1721–1738`; `device/utilities/matmul_utilities.cpp:470–491`).

### Convolution and native prefill

Q/K/V widths 512/512/1536, carry `[B,3,2560]`, tap volume 2560, exchange `[1,12,2560]`, and channel chunk256 satisfy the convolution validator (`ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_device_operation.cpp:75` onward). `actual_end=s` excludes padded tail from replacement carry; M:381 publishes that carry after every call.

The default sequence-parallel axis is zero (`.../qkv_causal_conv1d_silu_nanobind.cpp:156`), whose extent is one on the runner's `[1,4]` mesh. The factory selects a local, no-fabric history path (`.../device/qkv_causal_conv1d_silu_program_factory.cpp:137–138,174–175`). It does not exchange unrelated TP head histories on axis 1.

Native GDN derives H=4, HV=12, K=V=128, grouping3 from the actual inputs (`chunk_gated_delta_rule.cpp:145–199`). The adapter pads Q/K/V and gates to the same sequence extent, so the flat-input multiple-of-32 requirement is met (`:217–230,333–335`). For finite padded inputs, beta=0/g=0 makes padded recurrence steps leave state unchanged. The native gate reader derives `ceil(HV/32)` dynamically (`device/kernels/dataflow/chunk_gdn_gate_read.hpp:30–31,48–56`); it already handles twelve heads correctly.

### Distributed norm mathematics and paged KV

Distributed pre-norm produces local sums; post-norm divides by local width × gathered devices = `1280*4=5120`. There is no missing factor of four (`.../layernorm_distributed/device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp:32–34`; `.../device/layernorm_post_all_gather_program_factory.cpp:270–271`). The rank violation is independent of that valid formula.

Full-attention packed width2048 equals `(6+2*1)*256`. Q/K/V split into six Q heads and one KV head per rank. Cache shape `[blocks,1,32,256]` agrees. Inherited prefill casts K/V to cache BFP8 before paged fills (O:1383–1409), meeting the raw same-dtype tile contract (`ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_device_operation.cpp:32–45,74–95`). Nonaligned continuation performs token updates until alignment (O:1174–1199). Small tables select 32-key chunks; tables with at least sixteen pages receive finite padding to a multiple of sixteen and select 512-key chunks. No new default page-index/head-count defect was established.

## 5. Trace audit and other potential issues

No trace has been demonstrated for this TP4 revision. The direct source blocker is norm output sizing, not an observed replay deadlock. Forward calls enqueue work; completion would require a blocking read, synchronization, or blocking trace replay. The runner's `read()` performs device-to-host reads after each initial forward, timed regions explicitly synchronize, and correctness replays are blocking (runner:77–80,104–110,161–166,199–214). The supplied failing logs never reach these boundaries.

The current GDN path supplies all four constant tensors and explicit initial state. It therefore avoids native eager-only constant construction/host uploads (`chunk_gated_delta_rule.cpp:267–293`). Conv scalars, taps, exchange storage, semaphores, and state are prepared in setup. State buffers retain identity through `copy`/`output_tensor`. Sharded decode uses four gathers, two reduce-scatters, and six barrier-semaphore selections per forward; the manager's two-way indices return to the same parity. This audit found no further definite trace defect in the intended path, but does not establish allocation safety, asynchronous completion, or replay correctness.

Remaining risks and scope limits:

- **Precision:** BF16 distributed statistics add an intermediate rounding boundary relative to the baseline's FP32 hidden norm. TP output projections also round each local partial to BF16 before reducing it, whereas a single-device projection accumulates the full K dimension first (O:499–500; M:258–260,273–283). These need measurement; they are not additional proven accuracy bugs.
- **Topology guard:** M:24 checks only four devices, while collectives require axis1 to contain all four TP ranks and convolution assumes singleton axis0. A 4×1 or 2×2 mesh violates that ownership. Require mesh shape `(1,4)` locally; the current runner already uses it.
- **Policy overrides:** disabling the qualifying matmul pipeline or selecting readers1 while retaining MLP-down cores8/block2 exposes its 17-tile shard divisibility rejection. Use block1 for that geometry or reject the combination. Default pipelineTrue/readers2 is legal. M:304–305 also hardcodes BFP8 KV allocations despite accepting a `kv_dtype` policy key; a different requested cache dtype would be ignored, but the default matches.
- **Test coverage:** the current trace runner changes activation contents and restores state, but does not change `current_pos` or RoPE contents during its correctness replays. That leaves continuation/input-address freshness unproven. A B1-only run also cannot validate local-12 gate batch stride. Source SHA labeling in baseline output should be corrected before relying on it for provenance.

## Static validation and recommended order

Cheap checks parsed both Python files without importing TTNN; computed packed offsets, physical N padding, and K geometry; and extracted the pure partition function with AST. Local-12 partition coverage was checked for B=1,2,3,8,16,32 with 110 available cores. A separate coverage/QK-lifetime simulation passed 2,016 cases: B=1..32 × available cores `(1,2,3,8,16,64,110)` × requested cores `(None,1,2,3,4,8,16,32,128)`, including split triples. CB allocation byte totals were independently recomputed. These checks prove indexing/count properties, not hardware math or synchronization.

After the first draft, headline claims were rechecked against bindings, public wrappers, shape indexing, native dtype conversion, compute/readout ordering, and the proposed intervention. Suspicions about missing world-size division, a wrong recurrent-cache dtype, same-dtype state updates being dropped, default MLP-down block divisibility, and accidental TP convolution exchange were rejected or confined to explicit nondefault conditions.

Recommended order before any authorized device validation (items 1–2 are now present in the externally edited M1 and have received the static review above):

1. Repair the rank boundary in distributed norm and require the intended mesh topology.
2. If optimized decode numerics are part of the acceptance contract, add the complete local-12 recurrence specialization and preserve the separate affine/gate rounding boundaries. Keep multi-token prefill and published state ownership intact.
3. Review B2/user1 gate addressing and B1/B2 output padding statically before capture. Keep the base optimized source unchanged.
4. Once the operator authorizes hardware work, validate sharded versus replicated norm, prefill and state at S=1/31/32/33/512/513, decode at B=1/2/16/32, both attention kinds, nonidentity page tables/continuation, and restored-state/changed-input/changed-position trace replay. These are proposed follow-ups only; none was run here.
