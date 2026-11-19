# AutoDebug: the 70% gate is still unmet; additional graph fusions remain testable

## Verdict

The supplied default implementation has **not reached the user’s 70% stopping gate**. Independent arithmetic reproduces approximately **64.11% for GDN and 65.75% for full attention**. I found no defensible roofline correction that turns either result into a pass.

The stronger discrepancy is the claim of graph-fusion exhaustion. The recorded experiments do not cover several existing native fusions or structural rewrites: native matmul/SwiGLU, RMSNorm cast absorption, GDN slice/pad cancellation and flat-Z gating, fused K/V updates, and FP32 softplus/binary fusion. These are concrete hypotheses within `fused_decoder.py` and its tests/docs. **Legality or algebraic equivalence is not measured performance evidence**; none is claimed to close the remaining gap without testing.

No new model correctness defect is established. Preserve BF16 weights and exposed activation/output dtypes, HiFi4 with FP32 accumulation, PCC >= 0.995, state/cache correctness, trace replay, and the 262144-token context contract. Some native fusions move internal rounding boundaries; those changes require numerical acceptance rather than an assumption of bit equality. Do not declare completion from checklist coverage or substitute elapsed time for the user’s stopping rule.

## Scope and evidence anchor

Inspection only: no hardware, target model execution, implementation edits, reset, or launcher invocation. Three xhigh investigators independently audited GDN, attention/MLP, and timing/roofline evidence. Only this report was written.

Main implementation: `models/autoports/qwen_qwen3_6_27b/tt/fused_decoder.py` (abbreviated **F** below). Evidence directory: `models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/` (**D**). Short native filenames within a subsection are relative to the native operation directory named in that subsection.

The report anchors its runtime claims to `timing_{functional,fused}_{0,3}_512.json`, `final_profile_{0,3}.json`, and `perf/final_profile_{0,3}/{metrics.json,decode_perf_report.csv}`, checked against the raw Tracy CSVs. Source identity:

- Decoder SHA256: `9c83f10d0e7e4cb94c0f98513ebe836c37fa4c3239cc4edc702a265152712a3b`.
- Runner SHA256: `6d6c072112db566e45a7365cdf7399c5eb047e9b522ebdbdc0ecb0904183f6f6`.
- Checkout commit recorded by the evidence: `60c3c9e00911429d1ab6d4a519e8ae0c79be5c4a`; the decoder is an untracked stage artifact, so the source hash matters more than HEAD.
- The current default files match these hashes. The compressed raw CSVs decompress to their recorded metrics hashes.

Later candidate files `D/candidates/norm_width.py` and `padded_chain.py` appeared during inspection. They are separate prototypes, not evidence that the default changed or that a candidate passed. The latter already sketches part of the padded-chain opportunity below; avoid duplicating it as a supposedly new experiment. No conclusions here use unverified timings from these newer candidates.

## Direct observations versus interpretation

**Observed in artifacts/source:**

- The B1, length-512 timing JSONs contain 100 traced replay measurements. Full attention’s exact final median is **2.217937 ms**, rather than the rounded approximately 2.20 ms in the request. GDN is **2.355165 ms**.
- Final profiled GDN: **44 device ops, 2309.339 us kernel time + 50.110 us gaps = 2359.449 us**. Full attention: **43 ops, 2180.403 + 28.285 = 2208.688 us**. Reported overlap is zero.
- The five GDN dense matmuls consume **1893.311 us**; the six full-attention dense matmuls consume **1829.921 us**. Their individual measured bandwidth is roughly 79–82% of the report’s 512 GB/s reference.
- Each layer has two 5120-wide FP32 RMSNorms, each approximately **82 us on one core**. Their totals are 164.324/164.519 us. The cast-fusion proposal below does **not** establish removal of this normalization work.
- Existing correctness reports record the original 0.995 bar and full-context checks. They validate the selected old graph, not any proposed rewrite.

**Interpretation:** dense streaming plus serial norms and structural overhead explain why removing many small primitive ops improved latency without achieving 70%. The artifacts do not prove that remaining graph-only changes can achieve the gate, nor do they prove that all applicable native fusions have been exhausted.

## Roofline audit: no accounting rescue

All quantities below use decimal bytes and a **per-device 512 GB/s reference**, not aggregate card or multi-device bandwidth.

| B1 decode at 513 live KV tokens | GDN | Full attention |
|---|---:|---:|
| Active BF16 dense weights | 766771200 B | 744488960 B |
| Minimum recurrent-state / logical KV traffic | 6291456 B | 2101248 B |
| Total floor numerator | 773062656 B | 746590208 B |
| Bandwidth floor | 1.509888 ms | 1.458184 ms |
| Final 100-replay median | 2.355165081 ms | 2.217937261 ms |
| Floor / median | **64.1096%** | **65.7451%** |
| Maximum latency for 70% | 2.156982857 ms | 2.083120000 ms |
| Additional savings required | **198.182 us** | **134.817 us** |

Reproducible numerator arithmetic:

```text
GDN dense = 2*(5120*16512 + 6144*5120 + 3*5120*17408)
GDN state = 2*48*128*128*4
Full dense = 2*(5120*8192 + 5120*6144 + 6144*5120 + 3*5120*17408)
Full KV reads = 2*513*4*256*2
floor_ms = bytes / 512e9 * 1000
```

The GDN numerator already includes the real 48-to-64 projection padding for each A/B gate: an extra 327680 weight bytes. F:143–154 removes obsolete unpacked matrices after packing; counting both original and packed matrices would be wrong. Additional recurrence scratch, the final state copy, or unused cache capacity must not be added merely to improve the score. They are implementation overhead rather than indispensable algorithmic traffic.

Physical matmul M is 32 despite logical M=1. Counting those padded MACs gives dense compute floors of approximately 0.161358/0.156669 ms using the installed tool’s 110-core HiFi4 peak, still far below the memory floors. No compute/bandwidth crossover changes the conclusion.

There **is a reporting-definition discrepancy**: `tt-perf-report` displays 65.4%/67.3% by counting analyzed matmul input/output bytes (773201920/751108096 B) and dividing by kernel busy time, excluding gaps. This is not the same end-to-end weight/state floor. See `python_env/lib/python3.12/site-packages/tt_perf_report/perf_report.py:486` for Blackhole constants and `:2117` for bandwidth aggregation. Neither definition passes 70%.

The measurement boundary is sound for the reported latency: `models/autoports/qwen_qwen3_6_27b/tests/run_decoder.py:220–229` restores state and synchronizes before the timer, then uses `execute_trace(...blocking=True)`. Device completion is inside the timer; host state restoration and output readback are outside. The small profiler/wall differences do not explain a 135–198 us shortfall. Advice to enable tracing is inapplicable because decode already replays a trace.

## Ranked, minimally testable graph hypotheses

The time figures below identify baseline work touched by a rewrite. They are **not additive speedup predictions**: replacement kernels retain compute, can alter scheduling, and may be slower. Change one topology at a time before combining winners.

### 1. Native matmul + SwiGLU is different from the rejected packed-MLP experiment

**Smallest intervention:** replace F:192–198’s two gate/up linears and fused-activation binary multiply with `ttnn.experimental.minimal_matmul(..., fuse_swiglu=True)`. Keep the existing down projection. Prepare a BF16 tile-pair-interleaved gate/up matrix at setup, then remove the obsolete separate uploaded matrices.

For transposed matrices `[5120,17408]`, concatenate gate then up to `[5120,34816]`, reshape `[5120,2,544,32]`, permute `(0,2,1,3)`, then flatten the last three axes. Columns become `[gate_tile0,up_tile0,gate_tile1,up_tile1,...]`. The in-tree `prepare_for_fused_swiglu(..., ndev=1, gate_is_first=True)` demonstrates this representation; ordinary concatenation alone is incorrect. Preserve `_linear`'s existing batch/sequence fold: for b>1, reshape h to `[1,b*s,5120]` before the call and restore `[b,s,17408]` afterward. Feeding `[b,1,5120]` directly would retain per-batch tile padding and undo that earlier structural improvement.

```python
mlp = ttnn.experimental.minimal_matmul(
    h, packed_gate_up,
    fuse_swiglu=True,
    dtype=ttnn.bfloat16,
    compute_kernel_config=self.ckc,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

**Lowered contract:**

- `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/minimal_matmul.cpp:38–54` calls the local device op directly; no collective or multi-chip execution is involved.
- `device/minimal_matmul_device_operation.cpp:167–182,251,264–283` supports tiled BF16, matching K, and packed N divisible by 64. Here Ktiles=160 and packed Ntiles=1088; N=34816 satisfies the gate/up pairing constraint. `:401–424` returns width 17408.
- `device/minimal_matmul_program_descriptor.cpp:397–425` partitions output tile pairs together. Defaults are block sizes `(8,8,8)` and subblock `(2,2)` (`:61–80`), satisfying K/pair divisibility and FP32 DEST capacity. Logical M=1 is physically tiled rather than an unsupported scalar special case (`:321–339`).
- `device/kernels/compute_metal2.cpp:651–672` applies SwiGLU in the last-K output stage. `device/kernels/swiglu_sfpu.hpp:44–53` computes the FP32-DST gate/Sigmoid/up expression before final packing.
- `tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul.py:574–632` covers both gate/up layouts. Those tests do not establish this model’s B1 HiFi4 performance.

**What it removes:** three device calls become one, without materializing and rereading both gate/up outputs. Baseline gate/up plus binary cost is 897.620 us GDN / 893.235 us full; most of that is still mandatory weight streaming.

**Adjudication:** `D/candidates/packed_mlp.py` uses ordinary packed linear, slices, and a later binary op. Its rejection does not test this native epilogue. This is a real unassessed dedicated fusion. Numerical equality is not bitwise: the baseline packs each projection to BF16 before SiLU/multiply; the native path can retain FP32 accumulators until the epilogue. Keep BF16 weights/output and `self.ckc`; require contribution/HF PCC >= 0.995 and measured improvement. The native 2D schedule may lose B1 bandwidth, so speed remains uncertain.

### 2. Absorb the two RMSNorm casts while retaining FP32 intermediates and gamma

**Smallest intervention:** in F:179–187, try `ttnn.rms_norm(x, weight=self.weights[key], epsilon=..., compute_kernel_config=self.ckc)` directly on tiled BF16 `x`. Keep gamma FP32 and every compute-config setting.

The statement in `D/native_runtime_audit.md` that both explicit casts are necessary to preserve the FP32 policy is not established by the native contract:

- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.cpp:89–91,119–145` admits the BF16 input/tiled-gamma geometry; `layernorm_op_multi_core.cpp:491–493` independently obtains gamma's FP32 format.
- `layernorm_op_multi_core.cpp:474–493` obtains FP32 accumulation from `compute_kernel_config` independently of input format, then selects **Float32 intermediate format** when `fp32_dest_acc_en=True`.
- `:715–738` supplies FP32 accumulation/fusion buffers. `device/kernels/compute/layernorm.cpp:266–303,314–378` forms statistics and normalization using that configured precision, packs the normalized intermediate to the FP32 fusion buffer, applies gamma, and packs the output. The streaming alternative `device/kernels/compute/layernorm_large_tensor.cpp:325–405,435–479` also keeps FP32-configured accumulation/intermediate storage through normalization and gamma before final packing. The input format can change which plan fits; both alternatives must be considered for width5120.
- `layernorm_device_operation.cpp:539–544` makes the interleaved output inherit the input dtype, establishing why this call returns BF16.

BF16 inputs are exactly representable in FP32. The explicit widening need not establish FP32-configured statistics/intermediate storage; the narrowed final output can be the norm’s own output pack. This is cast absorption, not a lower-precision arithmetic policy. Neither graph is promised full IEEE FP32 arithmetic at every internal FPU boundary: the native unpack choices include UnpackToSrc/TF32 paths even for some Float32 buffers (`layernorm_op_multi_core.cpp:1179–1187,1233–1241`). Compare against the existing implementation's actual arithmetic rather than assuming a stronger precision guarantee from its Python cast.

**Baseline opportunity:** four casts in GDN total 13.505 us kernel time; eight in full total 26.538 us, plus gaps. The two approximately 82 us norms remain unless the changed native plan itself improves them; do not count their whole time as saved.

**Remaining uncertainty:** input format can change unpack modes and scheduling. A successful dtype validation alone is insufficient. Compare normalized values and decoder contributions to the existing path, then state/cache and long-context evidence. The public RMSNorm API has no output-dtype argument; trying to set a primitive dtype is not a substitute for tracing its output-spec behavior.

### 3. Cancel GDN conv-output slicing followed by re-padding

**Confirmed redundant structure:** F:231–233 slices all three native convolution outputs to logical `s`; F:439–440 pads those same tensors back to the next 32-token boundary. Every one-token decode takes this path.

**Smallest intervention:** retain the native padded Q/K/V tensors, carry the original logical `s` separately, pad only beta/g in the public chunk-GDN call, and trim the final output back to `s`. Keep convolution `actual_end=self.conv_ends[s]`; otherwise history advances through padding and becomes wrong.

**Native proof:**

- `ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_device_operation.cpp:190–196` gives Q/K/V the padded input sequence extent.
- Its `device/kernels/dataflow/writer_qkv_causal_conv1d_silu.cpp:117–142` writes the complete output tiles. Retained padding is not uninitialized memory.
- With finite data, a discarded token with beta=0 and g=0 has no delta update and unit state decay. Causality prevents its Q/K/V from changing earlier valid outputs. The conv tail can be nonzero; neutral gates, rather than zero Q/K/V, are the required invariant.

Final GDN profile IDs **1458–1460 and 1469–1471** total **24.294 us including their preceding gaps**. This is an exact mapped baseline cost, not all GDN slice/pad time. `padded_chain.py` now contains a related prototype; its broader padded Z/output transformations should be evaluated separately from this minimal cancellation.

**Stronger existing-native variant:** remove explicit beta/g padding too by using the already-bound prep+scan primitives with padded flat Q/K/V and original logical gates. The public wrapper requires identical T (`ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp:215–224`), but the prep primitive requires only `ceil(gate_T/32)==num_chunks` and matching g/beta lengths (`device/chunk_gdn_phased.cpp:21–39,72–79,141–155`). `device/kernels/dataflow/chunk_gdn_gate_read.hpp:61–81` explicitly zeros gate rows beyond logical T. This targets IDs1472/1473, **22.381 us including gaps**.

Prototype this stronger adapter only for **B1 decode**, preserving its phased topology and supplied constants, FP32 state, flat flags, `HV=48`, `Hk=16`, `qk_norm=True`, `scale=128**-0.5`, chunk_size=32, and compute config. Copy the public wrapper’s state reshape/output handling faithfully. Retain the public path for other regimes initially: B32 has BH=1536 and cannot use the phased scan's one-core-per-head plan; its public wrapper selects fused head waves (`chunk_gated_delta_rule.cpp:321–327`). A generalized private adapter must reproduce that dispatch and wave plan. This requires more adapter code and relies on an internal bound API; it is not a one-argument change to the public wrapper.

**Tests that can falsify the proposal:** original-state versus rewritten-state PCC after multiple successive tokens; awkward tails 1/31/33/511/513; batch 2/32; continuation across chunk boundaries. Padded finite values should also be made deliberately nonzero in a focused test. No claim of equivalence for arbitrary NaN/Inf inputs is made (`0*NaN` is not neutral).

### 4. Keep GDN Z flat and concatenate normalized heads before gating

**Smallest intervention:** after F:240–245’s unchanged per-head RMSNorm, concatenate those normalized heads to `[b,s,6144]`, then multiply by SiLU of the already-flat packed Z slice. Feed that result directly to the unchanged output projection.

For concatenation/permutation P and elementwise gating, `P(norm_out * silu(z_heads)) == P(norm_out) * silu(z_flat)`. Norm stays per 128-wide head. This eliminates the Z reshape and permutation at F:246 while retaining the dedicated head concatenation that already runs at F:248.

Final profile IDs1481/1482 cost **24.146 us including gaps** (18.795 us reshape + 4.173 us transpose). The new flat multiply may have a different schedule; this is a target, not a guaranteed saving.

`conv_carry.py` used token-major delta output and reshaped both norm/gate operands; `head_major.py` kept both operands in head-major form. Neither is this placement of head concatenation. The newer `padded_chain.py` still pads/reshapes Z, so this alternative remains distinct. Check B1 decode first, retain a separate prefill path if only decode improves, and verify arbitrary logical lengths and batch order.

### 5. Fuse both paged K/V updates with the required disjoint input core grids

**Smallest intervention:** replace F:384–389’s two `paged_update_cache` calls with one `ttnn.experimental.paged_fused_update_cache`. Preserve both caches, page table, per-row positions, BF16 dtype, tensor permutation, and shard shape `[32,256]`.

It is **not** legal to reuse the current same core grid for both inputs. The native validator requires equal-sized, nonoverlapping K/V shard grids:

`ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:180–209,214–246`.

For b<=32 on a device with an available grid of at least 8x8, allocate K to the first b row-major cores and V to the next b cores. B1 can use (0,0)/(1,0); B32 can use rows0–3/4–7. Keep both K/V inputs simultaneously resident using their distinct shard configs. The tiled factory `paged_tiled_fused_update_cache_program_factory.cpp:629–644,672–703` pairs each grid's row-major ith core and supplies the same `my_batch_idx=i` to both reader/writer groups, so relocating V does not change request ownership. No activation dtype or cache representation changes. This limited layout adaptation is required to consume a dedicated fusion, rather than a general sharding/configuration search.

Final full-attention IDs468/471 each cost approximately 6.1 us. A fused invocation still performs both writes and may use similar device time; only measurement can establish the gain. Cover shuffled page ownership, mixed row positions, page crossings, B32, cache PCC, and refreshed retained traces. Grid overlap is a precise first-attempt rejection condition, not proof the op is globally inapplicable.

### 6. Fold the FP32 softplus into a neighboring binary op

F:236–237 currently emits FP32 cast -> add dt_bias -> softplus -> multiply neg_a. The ledger’s binary-activation experiments cover SiLU/sigmoid, but leave this activation as a standalone dispatch.

**Smallest intervention:** keep the cast and use `ttnn.add(a_fp32, dt_bias, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)])`, then the existing FP32 multiply. Alternatively put that activation on the multiply’s summed input. Final ID1467 is 4.241 us plus a 0.547 us gap; the fused kernel still executes softplus.

**Source/test support:** `ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp:654–663,809–824` forwards activation lists to binary_ng; `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp:1088–1146` creates post/pre activation definitions and FP32 DST from the input/output formats. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:547–557` emits the parameterized softplus helper. `tests/ttnn/unit_tests/operations/eltwise/test_binary_fp32.py:120–148` explicitly compares FP32 add+softplus with standalone softplus. That test uses zero bias without broadcast; it proves the FP32 activation path, not this model's nonzero dt_bias broadcast equivalence.

**Optional second step:** remove the widening cast by putting the FP32 bias in the first operand slot and the BF16 projection slice in the second, requesting FP32 output. This operand order matters: post-activation definitions use the operation’s input dtype, derived from the first tensor. Keep the simple all-FP32 fusion as the initial controlled test. Confirm s=1 and s=512 broadcast paths plus recurrent-state drift before combining changes.

### 7. Preserve logical RoPE output shape to avoid the repair slices

The generic rotary op returns a padded logical sequence and F:359 repairs it with a real slice. `ttnn.experimental.rotary_embedding_hf(..., is_decode_mode=False)` implements the same rotate-half convention on interleaved head-major input and preserves the input logical shape:

- `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/rotary_embedding_hf_device_operation.cpp:44–67,111–127` admits the current 64-wide partial-rotation tensor and `[1,1,s,64]` cos/sin, including s=1 in its interleaved prefill-style layout.
- `:143–179` constructs output from `input.logical_shape()`.
- `device/kernels/compute/rotary_embedding_hf.cpp:58–84` uses half-width rotation and the sign/multiply/add chain, rather than adjacent-pair Llama rotation.

Try that API with `self.ckc`, leaving the 192 unrotated channels and per-row loop unchanged. The repair slice then becomes an identity. Final IDs458/463 account for **4.471 us including gaps**. This is different from the rejected token-index/decode-layout experiment. A zero-copy logical-shape view of the generic rotary output is another possible adapter, but the shape-preserving op is the clearer first test.

## Other Potential Issues and secondary opportunities

### In-place native recurrent-state output: plausible, more invasive

The private `chunk_gdn_fused` binding exposes `o_out` and `final_state_out` (`ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule_nanobind.cpp:812–840`). Passing the same `[b*48,128,128]` view as initial and final state can remove the separate final-state copy, final profile ID1476 (**16.577 us**, plus gap).

Static safety evidence in that operation directory: `device/chunk_gdn_device_operation.cpp:139–151,280–283` accepts the supplied output pair; each scan receiver reads its complete disjoint state slice before use (`device/kernels/dataflow/reader_chunk_gdn_scan.cpp:239–260,525`), consumes that initial state at chunk0 (`device/kernels/compute/chunk_gdn_scan.cpp:68–75`), and writes the same final slice afterward (`device/kernels/dataflow/writer_chunk_gdn_scan.cpp:63–75`). No cross-slice reread was found. Match the wrapper’s exposed wave plan before testing; do not conflate aliasing with a new wave/grid experiment. The old fused factory timing was slightly slower than phased, so gross copy time is not net gain.

This remains secondary because validation permitting alias is not a documented alias guarantee, and paired preallocated outputs, cache hits, head waves, batch shape, and trace lifetimes require a dedicated runtime test. A Python `state.recurrent = updated` pointer swap is **not** an equivalent fix: a retained trace continues to read its captured original addresses. Conv carry has no supplied-output API in the inspected wrapper; its small copy cannot simply be deleted.

### Recurrent/mono compatibility claims need narrow wording

Chunk mono really rejects flat inputs and folded Q/K normalization (`chunk_gdn_device_operation.cpp:154–165`) and uses a different output format. Reintroducing normalization and head transforms is a separate, higher-overhead topology, not a drop-in replacement.

The public fused-recurrent op also has an untried in-place ring mode: T=1, persistent `idx=arange(b*48)`, `output_per_token_state=True`, and initial state viewed as `[b*48,128,128]`, with each index selecting its own block. Its binding documents the alias (`ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/fused_recurrent_gated_delta_rule_nanobind.cpp:32–42`), and its output creation returns the initial state. Previous candidates used non-ring output plus copy. Nevertheless it cannot fold the selected flat L2/head-map work and requires `b*48<=grid_core_count` (thus not B32 here), so it is lower priority than retaining the current chunk topology. Compatibility does not imply performance, and the existing non-ring timing does not prove every adapted ring topology slower.

### Residual epilogues exist, but have small directly removable work

`ttnn.experimental.dit_minimal_matmul_addcmul_fused` can express output projection plus residual using scalar1 and a persistent ones operand; its wrapper directly invokes local minimal_matmul. This removes the separate add but also changes the BF16 projection-output rounding boundary and introduces a ones multiplication. Baseline add work is only approximately4 us per location. Keep it below native SwiGLU and simple structural cancellations unless matching-control timings justify it. The existing statement about residual RMSNorm not eliminating the separately needed residual sum remains reasonable; that statement does not exhaust matmul-residual epilogues.

### Width-sharded normalization is a bottleneck diagnosis, not proof of an in-scope fusion

The one-core norm cost is real: the interleaved native implementation partitions tile rows, and B1 has only one tile row. Width-sharded RMSNorm may address it, and a `norm_width.py` prototype now exists. A pure program-config/sharding search belongs to the excluded optimization work. Do not silently reclassify that search as graph fusion or claim that cast absorption removes the entire 164 us. This report’s primary proposals actually remove/merge operations or cancel layout work.

### More elaborate partial-RoPE changes are unnecessary first steps

The indexed DeepSeek rotary op supports a partial rotary dimension but uses an interleaved basis. Replacing HF rotate-half directly would be wrong. Permuting Q/K weights and norm weights, converting cos/sin, and consistently changing cached K can make the mathematics equivalent, but changes the state-comparison representation and is much more invasive. It is not a headline remedy for a few-microsecond slice cost.

`sigmoid_gated_rms_norm` is likewise not a direct substitute for Qwen GDN’s SiLU-gated norm. No SiLU/sigmoid equivalence is assumed.

## Follow-up experiment and acceptance contract

The investigator did not execute these experiments. The smallest useful follow-up is isolated candidates on the **anchored final default**, using the existing real-weight selector and functional control. First prove op-level shape/dtype legality and contribution/state PCC on B1 decode; then measure the same warmed 100-replay benchmark. For native SwiGLU, compare ordinary minimal_matmul with the same interleaved weights plus separate gating as a control if a factory change obscures the fusion effect; that control must deinterleave tile pairs rather than incorrectly split the output into two contiguous halves. Keep or reject from measurements, not removed-op count alone.

Before accepting a combined graph, rerun the affected cases already required by the delivery: lengths1/31/32/33/511/512/513, successive decode and continuation, batch1/2/32, mixed positions, shuffled pages, retained traces with refreshed inputs/prepared requests, determinism and allocation-tracking controls. Preserve all three functional output/contribution/state or cache comparisons at >=0.995; residual-dominated output PCC alone is insufficient. Run both 262143 (including follow-on decode at context262144) and 262144 full prefill checks for any change affecting recurrence, normalization, MLP, RoPE, or cache handling.

Recompute active-weight traffic if a packing scheme changes genuine physical padding, while preserving an algorithmic minimum numerator. Evaluate the final graph against the unchanged 70% gate. If every valid graph-only candidate is measured and rejected but the gate remains unmet, report the stage as unmet with an exact scope blocker; do not mark it passed, waive the threshold, initiate excluded stages, or consume idle time.

## Post-draft claim review

Each headline was rechecked against caller and native code rather than accepted from a log label:

- Native SwiGLU is local and callable on BF16/HiFi4, but its moved rounding boundary and new matmul schedule prevent claiming numerical/performance acceptance.
- BF16 RMSNorm retains FP32 intermediate/gamma formats; only cast work is counted, not all normalization time.
- GDN cancellation requires separate original logical length and neutral padded gates. Conv history still uses the original end position. Gate-pad removal is a lower-primitive adapter, not falsely advertised as a public-wrapper option.
- Flat-Z gating retains per-head normalization and the same element order; older token-major candidates do not test it.
- Fused cache update requires disjoint grids and preserves page ownership; a same-grid direct replacement is rejected.
- FP32 softplus fusion has a direct in-tree test; optional mixed-input absorption explicitly preserves first-operand FP32 activation dispatch.
- Shape-preserving HF RoPE uses rotate-half and the existing interleaved layout; no unsupported decode-mode sharding is assumed.
- Aliased state, residual epilogues, mono/ring alternatives, width sharding, and basis-changing RoPE remain secondary with their unresolved conditions stated.
- Roofline definitions and source revisions were kept separate. No savings estimates or newer prototype files were promoted to measured results.
