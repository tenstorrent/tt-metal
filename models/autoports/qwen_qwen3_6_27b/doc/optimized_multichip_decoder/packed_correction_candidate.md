# Joint GDN primary/correction decode projection candidate

CPU-only implementation prepared in `tt/packed_correction_candidate.py`. No TTNN import, device execution, hardware correctness claim, or measured speedup was made by this investigation. Neither existing decoder source was edited. The original topology audit identified this repeated-input projection group; this candidate makes its packed alternative executable.

## Topology and numerical contract

Current local GDN primary contains QKV `[0:2560]`, Z `[2560:4096]`, and padded B/A `[4096:4160]`. The narrow correction contains QKV `[0:2560]` and B/A `[2560:2624]`. Correction scale is **one**; correction is the BFP8-quantized residual from the original setup, not a newly calculated approximation. The existing combiner adds FP32 primary and correction before its single BF16 rounding. It also explicitly adds FP32 zero to Z before rounding.

The candidate uploads `[primary_quantized, correction_quantized]` once during setup and performs one BF16-activation/BFP8-weight matmul with FP32 output and the unchanged attention compute config. Both original quantized matrices are read during setup, cropped to their logical columns, concatenated on a complete tile boundary, and uploaded as BFP8. An exact device-to-host readback comparison is mandatory at setup; it raises if repacking changes any decoded value. It does not re-quantize original HF values or merge the correction into a BF16/BFP8 primary weight. Only same-dtype BFP8/BFP8 weights and BF16 activation are accepted in this first family comparison.

The default `fused` combiner keeps the existing compute and writer kernel source. Its reader accesses both operands from the single matmul output and changes only the correction address to `row * joint_stride + 130 + rc`. QKV correction tile indices are 130–209, B/A indices 210–211, and Z uses the existing zero tile. Output remains BF16 `[1,B,4160]` in DRAM. This avoids slicing/copying two FP32 intermediate tensors. A `slice` adapter is also implemented: move the joint FP32 output to DRAM, slice its logical operands, and invoke the original combiner. That mode can isolate a new reader/addressing problem, but cannot by itself disprove the lower-movement fused family.

The current `correction_l1=True` default already converts the activation to width-sharded L1 **once** and shares it between the original two matmuls. Consequently, the candidate is credited with one fewer matmul and reduced weight padding, **not** a second input conversion removed from that default. If comparing a path with correction_l1 disabled, use the measured topology to account for movement separately.

## Geometry and memory

Inherited default: eight DRAM banks, eight activation workers, two readers per bank, K-block2, pipeline enabled. All variants retain K=5120. Width quantum is `32*lcm(banks*readers, cores)`.

| Quantity | Separate | Joint |
|---|---:|---:|
| Logical N | 4160 + 2624 | 6784 |
| Physical N, r2/c8 | 4608 + 3072 | 7168 |
| BFP8 tile count/device | 38400 | 35840 |
| Decode weight bytes/device | 41779200 | 38993920 |
| Projection matmuls | 2 | 1 |

The 2785280-byte saving is 6.67% of this projection family's weight traffic, not 6.67% of total decoder latency. Program per_core_N becomes 28 tiles, compared with separate18/12. The larger per-program N may increase circular-buffer requirements despite lower total live output bytes; a resource/API error at r2/c8 is not sufficient rejection. Retry smaller K blocks and r1/c8 or r1/c16/r2/c16 geometry through `packed_correction_options`. Candidate setup supports independent `cores`, `readers`, and `block` overrides. K/32 must remain divisible by cores and block. Some reader/core combinations have more padding; the selected-policy report records every computed width.

**Resident-memory consequence:** original separate matrices are retained for untouched prefill, so this prototype adds 38993920 bytes/device/GDN layer at r2/c8. Across 48 GDN layers, that is 1871708160 bytes/device. Account for this before any default promotion and update the cumulative context inventory. This is resident capacity, not decode traffic: the old weights are not read in candidate decode. The current generic capacity/profiler helper will need to recognize the joint key and subtract inactive separate weights from the decode inventory; it must not count all three as active decode matrices. There is no evidence requiring a reduced context limit.

## Prefill isolation and hook

`install(layer, cores=None, readers=None, block=None, combine="fused")` is setup-only. It creates one shallow Python decode view owned by the same decoder, then permanently binds `decode_forward` to that view. The view overrides only `_linear(...,"packed_gdn")`. Other decode projections delegate to the original class method. Original prefill runs on the original object, preserving `_flat_chunk`'s two direct `_linear_raw` calls, list/resume behavior, BF32-live recurrence handling, and short-prefill tails. No dynamic phase flag, output stash, runtime tensor cache, weight transfer, or monkeypatch inside forward is used.

The view shares the exact original weight dictionary, policy, CCL manager, and `_ccl_buffers` dictionary; it is not an independently instantiated layer or separate owner. Every layer still constructs its own view. Do not rebind those shared attributes independently after installation. Later debugging wrappers installed only on the original object's methods will not automatically instrument the decode view; instrument `layer._packed_correction_decode_view` too if using such wrappers. Runtime TTNN fallback patches remain global and cover both views.

The packed weight is registered under `weights["packed_gdn_correction_joint"]`, so existing matrix-allocation reporting records its shape and dtype. The existing runner's selected-policy JSON records geometry, combine adapter, exact setup readback result, original-weight retention, and candidate source hash. The module CLI preserves all ordinary full-layer harness gates and refuses the single-chip baseline option.

## CPU evidence and commands

Executed:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tt.packed_correction_candidate --cpu-check
python_env/bin/black --target-version py312 models/autoports/qwen_qwen3_6_27b/tt/packed_correction_candidate.py
```

CPU check passed B=1,2,3,17,32 and generated nine r1/r2/r3 × c4/c8/c16 geometry rows. Independent and concatenated small-integer FP32 GEMMs matched exactly; all130 output tile mappings, QKV/B/A correction offsets, and Z+0 mapped exactly. This is an address/algebra check, not a TTNN BF8 packer, FP32 matmul, SFPU-rounding, or trace validation.

Exact initial full-layer serial measurement (only while the parent hardware scheduler is free):

```bash
source python_env/bin/activate
source /home/ttuser/qwen36-27b/lane-env.sh
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
R=models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder
python -c 'import json,pathlib; d=pathlib.Path("models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder"); p=next(j["policy"] for j in json.loads((d/"profile_integrated2.json").read_text()) if j["layer"]==0); p["packed_correction_options"]={"combine":"fused"}; (d/"packed_correction_policy.json").write_text(json.dumps(p))'
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 MULTICHIP_JOB_TIMEOUT=600 bash "$D/run_job.sh" packed_correction_l0_b32_s33 \
  python -m models.autoports.qwen_qwen3_6_27b.tt.packed_correction_candidate \
  --layer 0 --batch 32 --length 33 --trace --repeats 100 \
  --policy-file "$D/packed_correction_policy.json" \
  --reference "$R/delivery_perf_ref_l0_b32_s33.pt" \
  --output "$D/packed_correction_l0_b32_s33.json"
```

Use the same policy without installation through `tests.run_multichip_decoder` for the control. Then measure B1S512 and B32S513 using the corresponding existing fixtures; prefill should retain its original numerical path. Retry resource errors using options such as `{"cores":16,"readers":1,"block":2,"combine":"fused"}`. For a reader-specific error, `{"combine":"slice"}` is a diagnostic adaptation while the fused implementation is repaired.

Before adoption, require unchanged output/contribution/per-user/state PCC, traced changed-input correctness, tail batches, separate watcher, final default measurement and device profiler attribution. The current implementation has no hardware evidence yet. A custom combine error needs a bounded diagnosis/adaptation rather than rejection of joint projection as a family. Preserve source hashes and failed evidence for each attempt.

## Optional joint prefill extension (CPU-prepared, not device-validated)

The candidate now implements `PackedCorrection.project_prefill(x)` and accepts setup options `prefill=True`, `prefill_config={...}`, and `release_separate=True`. All default options retain the previous decode-only behavior. The extension has no device evidence yet; earlier passing decode-only results do not establish prefill correctness or speed.

`project_prefill` accepts logical `[B,S,5120]`, flattens only token-independent projection rows, owns alignment to32, partitions rows with a configurable cap (default the current `matmul_row_cap`, usually512), performs one joint FP32 matmul per partition, and applies the same exact FP32 primary+correction/BF16 boundary. It concatenates in original row order, crops owned padding, and restores `[B,S,4160]`. For <=32 total rows it uses the existing DRAM-sharded decode program. For larger row partitions it uses the existing family of 2D multicast programs with the joint weight and FP32 output in DRAM; there is no extra persistent interleaved weight copy. A first program-factory error still needs geometry/layout adaptation.

2D settings inherit `prefill_projection_configs["packed_gdn"]`, then `prefill_config` overrides them. Accepted overrides: `grid`, `in0_block_w`, `subblock_h`, `subblock_w`, `input_l1`, `out_block_h`, and `row_cap`. Subblock area remains <=4 for the FP32 path, K block divides160 tiles, N subblock divides per-core N, and row cap must be a multiple of32. `prefill_plan` is a CPU-only helper exposing the resulting exact program constructor arguments. For default physical N7168/grid11x10, per-core N is21 tiles rather than the separate primary14/correction9.

Projection row partitioning never invokes convolution, recurrence, publishes a recurrent state, or touches state dtype. The original prefill scheduler still owns temporal chunk boundaries and its live FP32 recurrent state. That is structurally preserved, but the required long/chunk-boundary state PCC must still be measured on hardware.

### One required caller hook

The parent must add this small branch to `flat_prefill.py::_flat_chunk`, immediately before its current `if key in layer.weight_corrections:` branch, with the existing two-raw-matmul implementation retained as `elif`:

```python
key = "packed_gdn"
if hasattr(layer, "_prefill_packed_gdn"):
    packed = layer._prefill_packed_gdn(h)
elif key in layer.weight_corrections:
    packed = layer._combine_correction(
        layer._linear_raw(h, key),
        layer._linear_raw(h, layer.weight_corrections[key]),
        key,
    )
else:
    packed = layer._linear(h, key)
```

The remainder of `_flat_chunk`, including reshape to `(batch,seq,4160)` and `_gdn(...packed=packed,flat_output=True)`, is unchanged. This agent did not edit either caller file. Installation with `prefill=True` and `flat_prefill=True` deliberately fails at setup until the function actually references `_prefill_packed_gdn`, preventing a silent claim that flat prefill uses the new path.

At setup, installation additionally overrides the original object's `_linear(...,"packed_gdn")` to call joint prefill. This covers single-user, short tails, legacy/non-flat dispatch, and list-prefill paths. Decode still uses its separate permanent Python view and the original joint decode method. No transient phase flag or output stash is introduced.

### Removing old weights and capacity

After the caller hook is in place, `release_separate=True` explicitly removes both old matrix entries, their projection configs, and the correction mapping. It requires `prefill=True`; there is no hidden fallback to removed matrices. Both views share the weight dictionary, so the release affects both consistently. Metadata reports `scope="decode_and_prefill"` and `original_prefill_weights_retained=false`.

At r2/c8, using only the joint matrix replaces41779200 resident bytes with38993920 bytes per GDN layer, saving2785280 bytes/device/layer or133693440 bytes/device across48 GDN layers relative to original separate storage. This differs from decode-only mode, which adds the joint matrix while retaining old weights. Setup transient capacity and context accounting still require verification; update the capacity helper to distinguish these two policies. There are no per-forward weight conversions or host reads.

### Parent serial tests after wiring

Update the policy file from the earlier command (do not overwrite an active job's file), for example:

```bash
python -c 'import json,pathlib; d=pathlib.Path("models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder"); p=json.loads((d/"packed_correction_policy.json").read_text()); p["packed_correction_options"]={"combine":"fused","prefill":True,"release_separate":True,"prefill_config":{"row_cap":512}}; (d/"packed_correction_both_policy.json").write_text(json.dumps(p))'
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 MULTICHIP_JOB_TIMEOUT=600 bash "$D/run_job.sh" packed_correction_both_l0_b32_s33 \
  python -m models.autoports.qwen_qwen3_6_27b.tt.packed_correction_candidate \
  --layer 0 --batch 32 --length 33 --trace --repeats 100 \
  --policy-file "$D/packed_correction_both_policy.json" \
  --reference "$R/delivery_perf_ref_l0_b32_s33.pt" \
  --output "$D/packed_correction_both_l0_b32_s33.json"
```

Compare against a fresh decode-only joint candidate using the same other policy, then repeat B32S513 and B1S512. Check final matrix allocations contain the joint matrix and neither old matrix. Before promotion, cover B2/B3 non-aligned lengths, chunk/resume/list-prefill equivalence, BF16 public versus FP32 live recurrent-state continuity, changed-input traces, and watcher. A prefill failure can first retry `release_separate=False` for diagnostics, but material family rejection must include a repaired or adapted joint path; keep the selected implementation's actual resident-memory consequence in its report.

CPU evidence after this extension: original five batch mapping checks and nine geometries still pass; 100 B/S/cap row-order/padding/reassembly cases pass; five legal 2D example plans generated. Module syntax and Black formatting pass. These checks do not validate native 2D matmul numerical behavior or memory limits.
