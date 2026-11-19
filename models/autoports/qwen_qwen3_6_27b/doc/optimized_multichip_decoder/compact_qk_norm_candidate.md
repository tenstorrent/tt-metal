# Compact Q/K normalization without input reshard

CPU-only opt-in prototype: `tt/compact_qk_norm_candidate.py` and `tests/probe_compact_qk_norm.py`. No main/shared/native files changed; no torch/TTNN imports or device execution occurred in this task. Compilation, numerical results, and latency are unverified.

The current full-attention compact decode path splits Q/K/V to HEIGHT_SHARDED L1 with one user per core, shard `[32,256]`, on the first B cores in eight-column row-major order. Q is logically `[1,B,6,256]`, K `[1,B,1,256]`. It then converts Q and K separately to DRAM and runs two RMSNorm operations before partial RoPE. The parent identifies this group as approximately 20 us of material decode work; this note does not claim a measured candidate improvement.

## Existing-operation audit

- `layernorm_device_operation.cpp:223` explicitly rejects HEIGHT_SHARDED input and requires non-height-sharded output for a sharded input. Passing the current shards directly to native RMSNorm is invalid.
- Existing `TestDeepSeekV3.test_q_kv_rms_norm_inline/persistent` examples use `Parallel` RMSNorm branches on disjoint WIDTH_SHARDED grids, with Q width1536 and KV width512. Their layout and reduction contract differs from the local per-user 256-wide Qwen heads. Relabeling the current height shards as width shards would incorrectly combine users and is not valid.
- In this checkout, the descriptor normalization helper still calls `factory.create_descriptor`, but the current layernorm nanobind registration exposes no such factory method: the factory now produces Metal2 ProgramSpec artifacts. Consequently the old test text is useful design evidence, not proof that this descriptor path is currently callable. It should not be blindly copied.
- GDN fused Q/K L2-normalization applies to recurrent attention and has different scaling/affine semantics; it cannot replace these learned Q/K RMS norms.
- The decoder's norm weights are **FP32**, not BF16: `OptimizedDecoder.from_state_dict` folds `1 + HF_weight` for both Q/K norms and uploads folded norms as float32. The helper preserves those exact weights. It also supports BF16 weights if explicitly supplied for a separate test, without casting default weights down.
- Native RMSNorm with the current FP32 destination configuration uses FP32 intermediates and gamma application. The candidate keeps FP32 square/sum/mean/epsilon/rsqrt/affine arithmetic and precise SFPU mode, with a final BF16 output rounding. It changes reduction/kernel instruction ordering; source review cannot establish bit identity to native FPU/row-mean arithmetic.

## Bounded candidate contract

One generic kernel covers both Q and K on each user's existing L1 owner. Its reader copies the eight local BF16 tiles directly over the local NoC and reads the original Q/K gamma tiles. The compute processes each tensor in turn:

1. Square and sum the eight channel tiles in FP32.
2. Reduce only the 32 channel columns within each head row, scale by 1/256, add the model epsilon, and apply precise reciprocal square root.
3. Multiply the original BF16 activation by that head's FP32 inverse and the unchanged gamma value in FP32, then round to BF16 output.

Head rows and users never reduce together. The local padded rows remain independent. The writer emits normalized Q/K directly into the existing DRAM layout expected by partial RoPE. This replaces two input conversions plus two normalization launches with one operation; it does not replace the downstream rotary implementation or force an extra restore of the old unnormalized DRAM contract. Outputs have exactly the old normalized shapes/layout, so prefix64 RoPE, unchanged suffix192, paged K update and SDPA remain on their existing paths.

The helper reads neither host values nor global runtime state. It has no tensor-output cache, in-place input mutation, or persistent scratch. Optional preallocated output buffers are supported. Explicit per-core CB allocation with FP32 gamma is 69,632 bytes (16KiB input +32KiB gamma +4KiB inverse +16KiB output); BF16 gamma reduces this by16KiB. Per-device transient/output allocation and actual L1 availability still need resource evidence.

The epsilon is passed from `layer.config.rms_norm_eps` (current HF value `1e-6`), rather than fixed in the kernel. Compute remains HiFi4, FP32 destination and precise math. FP32 gamma and inverse CBs use direct FP32 unpack, avoiding an implicit precision reduction. The descriptor's local owner grid is exactly the split operation's grid, including sparse final rows for non-power-of-two B.

## CPU checks

```bash
python -m models.autoports.qwen_qwen3_6_27b.tt.compact_qk_norm_candidate --cpu-check
python -m py_compile models/autoports/qwen_qwen3_6_27b/tt/compact_qk_norm_candidate.py models/autoports/qwen_qwen3_6_27b/tests/probe_compact_qk_norm.py
```

Eighteen owner/tile-address cases pass: B=1,2,3,7,8,9,17,31,32 and Q/K head counts6/1. They check channel-to-tile/face mapping, unique user owners, output page ownership and exact reciprocal constant. Both module imports and CLI help remain lazy. These checks do not compile or execute device kernels.

## Parent-serialized first probe

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 MULTICHIP_JOB_TIMEOUT=900 bash "$D/run_job.sh" compact_qk_norm_first \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_compact_qk_norm \
  --batches 1 3 32 --weight-dtype float32 --output "$D/compact_qk_norm_first.json"
```

The probe uses actual `_local_qkv_split` on real TP4, distinct rank/user inputs and FP32 weights, changed inputs, an exact zero head, and the existing native DRAM-conversion-plus-RMSNorm path as the control. Candidate/native and candidate/torch RMSNorm PCC must each be >=.995 and finite. Inputs must remain bit-exact. Each branch is traced separately and released before the next capture; all changed-input trace outputs must match that branch's eager results bit-for-bit. Five warmed samples of20 queued trace replays time the entire native group against the candidate, with endpoint synchronization. No host reads occur inside capture or timing.

Use a separate watcher run (`TT_METAL_WATCHER=120 TT_METAL_DEVICE_PROFILER=0 TT_METAL_TRACE_ALLOC_TRACKING=1`) with `--batches 1 2 3 7 8 9 17 31 32`, fresh output name, no timing. The default FP32-weight test is the model-applicable one. Optional `--weight-dtype bfloat16` only tests that separate supported contract and is not evidence for changing model weight precision.

A first compile/API error is an implementation defect to adapt, not rejection evidence. If numerical comparison fails, inspect the column-reduction output/broadcast and FP32 unpack modes first, then adapt the reduction tree or clone the native row-mean body to preserve native arithmetic. Keep full-layer quality gates unchanged.

## Minimal optional integration after focused acceptance

Immediately after `_local_qkv_split` in `_local_compact_attention_decode`, wrap the existing two conversion/norm calls:

```python
if hasattr(self, "_compact_qk_norm"):
    q, k = self._compact_qk_norm(q, k)
else:
    # Existing q/k DRAM conversion and native RMSNorm comprehension.
```

Install `compact_qk_norm_candidate.install(model)` after model construction. It fails closed if the optional hook is absent. It uses the actual model weights and epsilon and records the selected policy. This instance hook is for isolated testing; if selected, make it an owner-explicit class method for the final 64-owner resource/lifetime validation. Main integration is parent-owned and was not edited here.

Run the full layer3 TP4 harness at B1/B32 with the accepted reference, decode trace, and all existing PCC/state/cache gates. The focused probe does not establish rotary compatibility on hardware, so include changed per-user rotary positions and cache-update checks in full-layer validation. Compare final warmed/traced full-layer decode, profile operation removal, and rerun final-default watcher/stress/resource gates if kept. Inter-layer residual layout and context capacity are not intentionally changed; any measured resource impact must still be reflected in the context contract.
