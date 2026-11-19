# Decode expert reduction: remove irrelevant padding preparation

## Poison-injection diagnostic repair

The first root-owned probe reached bit-exact original/direct outputs with preserved physical input buffers for zero and32768 padding, then failed its NaN-injection assertion **before running either reduction on the NaN case**. It therefore did not establish a NaN reduction failure or NaN-padding coverage. The original [failed JSON](moe_padding_elision_probe.initial_nan_injection_failure.json), [provenance](moe_padding_elision_probe.initial_nan_injection_failure.json.provenance.json), and [probe source](moe_padding_elision_probe.initial_nan_injection_failure.py.snapshot) were preserved byte-for-byte. JSON SHA256: `1df41989d2b2df7cc85364c1e7a60b4759853a6b58198aed16f67a33b34cc837`.

The device fill API is not a raw-BF16-bit write. `fill_pad_program_factory.hpp:45–66` preserves the supplied float32 bits for BF16, but an integer argument for a floating dtype is converted numerically, so passing integer0x7fc0 would not inject BF16 NaN. `fill_pad_program_factory.cpp:95–100` disables FP32 destination for BF16; the compute kernel then runs `fill_tile_bitcast`, Float16_b `where_tile`, and BF16 pack (`fill_pad_compute.cpp:24–39`). Thus the Python/host float-to-bit step is not demonstrated to be the fault: the fill/store/where/pack path is an additional boundary. The initial run did not save the resulting padding codes, so its exact NaN transformation cannot be inferred from that artifact.

Probe v2 records physical BF16 code histograms, finite/NaN/±infinity counts and matched-element counts for every requested injection before testing reductions. It covers zero, ±32768, ±infinity and NaN. If native fill fails to produce the requested padding, it retries **setup-only** `from_torch(..., layout=TILE, pad_value=fill, preserve_nan_values=True)` on the host, verifies that host buffer, uploads it with `to_device`, and independently verifies the physical device buffer. Host tilization uses the supplied pad value in `ttnn/core/tensor/py_to_tt_tensor.cpp:299,335`, avoiding the device SFPU fill path. This is diagnostic input construction, not runtime oracle substitution.

If both injection methods fail, the probe records that category as untested and reports `exact_for_verified_padding_only`, with `nan_padding_tested_all_cases=false` when applicable. It never substitutes a failed NaN request with finite padding and labels it NaN. A verified injection whose reduction outputs differ still fails normally. The default output is now `endurance/moe_padding_elision_probe_v2.json`; the same root command below runs it. CPU tests verified all six category summaries and rejected a synthetic infinity-as-NaN injection. The root-owned v2 results are now complete; see the measured decision below.

Source/CPU-only hypothesis; no production or device edits. The inspected decoder SHA is `d9ced8ec1d70f48329d98d2b04a713b5fe8977dd6625cdddffe6fe636e6b25b2`. Its complete `_moe` AST equals the previously profiled `projection_l1_chain.py.snapshot`, so the [profiled decode sequence](../profile_projection_l1/tracy/layer_0/decode_op_sequence.csv) describes the same MoE structure despite intervening attention changes.

## One new structural candidate

The decode routed-expert result currently computes BF16 weighted expert outputs `[1,8,1,2048]`, then `ttnn.sum(..., dim=1)`. Generic sum already invokes `FastReduceNCDeviceOperation` here. Before that native reduction, however, the generic wrapper clones the tensor and fills all padded H rows with zero because logical H=1 is not tile aligned. The reduced dimension is expert axis1, not H or W; each valid row0 element depends only on row0 from the eight experts. Padded rows1–31 never contribute to a consumed output element.

In one measured replay, steps191/192 are Clone6.922µs and FillPad8.030µs, followed by FastReduce4.260µs and Slice1.693µs. This gives a concrete target of two unnecessary padding passes (14.952µs of that sampled device sequence), not a guaranteed full-trace speedup. No same-purpose direct BF16 MoE reduction trial was found in the current pattern assessment/work log. This differs from the rejected dedicated post-combine adapter, which padded token rows and converted indices/weights to another native operation's contract.

Minimal model-only control:

```python
weighted = ttnn.multiply(down, routing)  # preserve existing BF16 product boundary
total = ttnn.experimental.fast_reduce_nc(
    weighted, dims=[1], memory_config=DRAM,
    compute_kernel_config=self.compute,
)
total = total[:, :, :1, :2048]  # native output derives H from padded shape
routed = reshape(total, (1, 1, 2048))
```

Do not widen `weighted` to FP32 or omit the explicit compute config. The goal is to call the exact native primitive that generic sum already selects, with the same BF16 input/output and reduction arithmetic. Keep the final slice: native output has logical H32 because its output spec derives from the padded input shape.

## Source proof and precision boundaries

- `ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp:638–645` clones and zero-fills whenever tiled H/W has implicit padding, even for an exclusively outer-axis sum.
- The same file's `call_fast_nc` at498–504 accepts BF16 sum, and `non_height_width_reduce` at509–546 uses HiFi4/nonapproximate/FP32 destination on Blackhole, calls the existing FastReduce primitive, and crops back to the logical shape. Current `self.compute` has those settings and disables packer-L1 accumulation. An explicit config is essential because direct `fast_reduce_nc.cpp:31–36` otherwise defaults to approximate mode and enables FP32 destination only for FP32 input.
- `fast_reduce_nc/device/fast_reduce_nc_program_factory.cpp:289–292` selects `REDUCE_FP32_IN_DEST` only for FP32 inputs. BF16 remains on the same existing FPU addition branch in both wrappers; this proposal does not substitute the ordered FP32 SFPU recurrence arithmetic.
- `fast_reduce_nc/device/kernels/reduce_nc.cpp:41–89` adds corresponding elements from the expert tiles. It does not reduce neighboring H/W positions. Consequently poisoned padded rows cannot contaminate the consumed row0, provided the logical crop is retained.
- `fast_reduce_nc/device/fast_reduce_nc_device_operation.cpp:122–137` sets the reduced axis to1 but retains padded H in the output spec. Generic sum also crops using `min(input_logical, output_logical)` at `generic_reductions.cpp:66–80`; the candidate explicitly retains this crop.

A CPU address-dependency check replaced all unconsumed H rows with NaNs in `[1,8,32,2048]`, then compared row0 of the same eight-term fold against zero-filled padding: all2048 BF16 codes agreed. This proves only the independence of consumed coordinates, not installed device arithmetic or scheduling.

## Minimal device discriminator for the root owner

Capture the actual BF16 `weighted` tensor and compare original sum against direct FastReduce plus crop on identical inputs. Also test logical H1 tensors whose physical H padding is zero, large finite values and NaNs. Require exact consumed BF16 codes, output shape/dtype, and no mutation of the input. Use the same explicit native compute config in the proposed path and preserve multiplication before summation. Then measure the complete weighted-reduction adapter and full decoder; retain only if faster and all existing seeded/endurance/continuation gates remain satisfied. Recheck both layer kinds, since their singleton MoE path is shared. The preparation above preceded the root-owned device results recorded below.

## Prepared root-owned control

The actual BF16 FastReduce row in `profile_projection_l1/tracy/layer_0/ops.csv` records `HiFi4; math_approx_mode=0; fp32_dest_acc_en=1; packer_l1_acc=0; dst_full_sync_en=0`, dim1, DRAM, and `output_dtype=nullopt`. This directly verifies that `self.compute` matches the currently executed generic sum. Omitting the direct-call config would change defaults and invalidate the padding-only comparison.

[The patch](moe_padding_elision.patch) replaces only the routed expert sum wrapper; it was checked with `git apply --check` but not applied. [The device probe](../../../tests/moe_padding_elision_probe.py) creates logical `[1,8,1,2048]` BF16 tensors, fills physical H padding with zero,32768 orNaN using `fill_implicit_tile_padding`, and reads back the physical padding with `from_device(...).to_torch_with_padded_shape()` to prove the poison was present. It compares every consumed BF16 code against the original generic sum and checks the entire input buffer remains unchanged after each adapter, including NaN payload bits. Both outputs must retain BF16 TILE DRAM `[1,1,2048]`. It also times complete original and direct reduction/crop/reshape traces, beginning after weighted multiplication.

Default inputs are three deterministic BF16 expert-output/routing products. `--weighted-capture PATH --weighted-key weighted` additionally accepts an actual saved BF16 weighted tensor of that exact shape; setup-only host loading never enters capture. CPU padding-independence fixtures passed. The new probe was formatted with pinned Black23.10.1/isort5.13.2; production and frozen oracle harnesses were not edited.

Root command through the usual `wait_for` wrapper, with exclusive device ownership:

```bash
timeout 600 env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE \
  QWEN_DECODER=fused python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.moe_padding_elision_probe
```

The default output is `endurance/moe_padding_elision_probe_v2.json` plus provenance. Existing output is protected unless `--overwrite` is supplied. `--oracle-only --output /tmp/moe_padding_cpu.json` performs only the CPU coordinate-independence check. This investigation agent has not run the device probe or patch on hardware.

## Other inspected MoE/shared-expert boundaries

The router currently performs same-dtype UINT16 and BF16 typecasts that are real dispatches in the profile (steps180 and188,0.832/0.936µs). The generalized gate requires UINT16 indices and BF16 scores in its output validation, so they are narrower cleanup opportunities, but they are not bundled into the reduction control.

The shared expert already packs gate/up/scalar projection and fuses activation into its two multiplications. Merging its final weighted product with the routed add risks removing an existing BF16 product boundary. Merging router and shared projections would combine FP32 router-weight storage with BF16 shared weights and alter matmul geometry; the earlier BF16 router-storage experiment did not win. Neither is a source-proven precision-preserving improvement, so neither is proposed here. Previously rejected native sparse/SwiGLU adapters are not reopened.

## Measured decision: retain direct reduction

The root-owned [v2 probe](moe_padding_elision_probe_v2.json) reports `status=exact`, no unverified padding cases, and `nan_padding_tested_all_cases=true`. All three input cases preserve every consumed BF16 output code and every physical input code for zero, ±32768, ±infinity and NaN padding. The native NaN fill produced **0x7f80 (+infinity)** in all 507,904 padding elements per input. The setup-only host-tilization fallback instead produced **0x7fc0 (quiet NaN)** in all of them, verified again after upload. Thus NaN reduction coverage is real in v2; the original failure was an input-construction failure, preserved above, rather than a failure of the candidate reduction.

Complete warmed reduction/crop/reshape trace medians are **28.589 µs generic versus 14.237 µs direct**, from 30 samples with 20 replays each. Both timed adapters remain bit-exact against the original output. These timings begin after the existing BF16 weighted product and include the adapter conversions; they do not stand for whole-decoder latency.

Root also measured S256 decoder decode medians of **1.473297 → 1.457692 ms** for linear attention and **0.616379 → 0.601635 ms** for full attention, with unchanged PCC. The integrated 8,192-step plus 33-token continuation controls passed for [layer 0 seed 154](moe_padding_elided_layer0_seed154_results.json), [layer 0 seed 42](moe_padding_elided_layer0_seed42_results.json), [layer 0 seed 913](moe_padding_elided_layer0_seed913_results.json), and [layer 3 seed 913](moe_padding_elided_layer3_seed913_results.json). Retain the padding-elision candidate on these measured results. These candidate controls do not replace the final fresh-reference validation queue; final closure remains pending.
