# Compact singleton ordered cumsum: bounded new placement control

Base: `../selected_fused_decoder_v3.py.snapshot`, SHA256 `73ba9aabb10a2c34f3b10b5ff1fc93418470c0a624dd863e87a8c8c2af2901ba`. Prepared isolated `v3_singleton_cumsum_l1.py.snapshot`, patch/provenance, and `tests/v3_singleton_cumsum_probe.py`. Production, frozen snapshots and existing harnesses are unchanged. No hardware was executed by this investigation.

## History is retained

The K-in-tile-height projection scan is **not new**. `../cumsum_reductions.py.snapshot` used `[blocks,K,12352]`, FP32 products/operands, uncompensated cumsum, terminal-row slices and explicit block adds. Its complete decoder measured5.773ms decode /332.131ms prefill. The later last-only FastReduce control measured5.157ms decode /332.241ms prefill, preserving the8192-token gate. `AUTOFIX_ordered_reduction_optimization.md` documented why materializing every prefix was avoidable. Those results are neither erased nor claimed obsolete.

The specific untried control is today's6208-column split with compact **BF16** weights/coefficients, selected compact RM coefficient gather, **L1 product and prefix**, and a bounded terminal-row adapter. It is a placement/layout hypothesis against current v3, not a claim that cumsum was historically better.

## Exact runtime adapter

Setup uploads one additional pair-permuted BF16 matrix `[1,1,2048,6208]`,25,427,968 bytes. Singleton runtime retains the existing RM pair32 gather, then:

1. View coefficients as `[blocks,1,K,1]` and tilize. View the setup matrix as `[blocks,1,K,6208]`, with `(blocks,K)=(4,512)` or `(2,1024)` from the existing public policy.
2. Accurate BF16×BF16→FP32 BinaryNG multiply into interleaved L1.
3. `cumsum(products, -2, dtype=float32, disable_compensated_sum=True, memory_config=L1)`.
4. Release the product reference. Slice the **tile-aligned final32-row band** before slicing its row31, then release large prefix references. Native `slice.cpp:233,382` converts nonaligned tiled starts through RM; cropping first prevents untilizing the entire prefix tensor for a one-row result.
5. FastReduce the resulting FP32 `[blocks,1,1,6208]` partials along outer dim0, preserving the existing block order, then cast to the requested output dtype and view `[1,1,6208]`.

The entire adapter is inside the snapshot's actual `linear` call and inside trace timing. Multirow projection, its new v3 filled-row packing, recurrent state and public cache boundaries remain unchanged. No runtime host conversion is added.

## Arithmetic and native contracts

`accumulation_device_operation.cpp` accepts forward FP32 tiled interleaved H scans on Blackhole; both L1 and DRAM are supported. `accumulation_common.cpp` recognizes this tile-axis path without a large permutation. `accumulation_program_factory.cpp:442–545` enables32-bit destination, direct FP32 destination unpack (avoiding TF32 input narrowing), four input and four output tiles, and disables compensation only when explicitly requested.

`accumulation_tile_axis_compute.cpp` scans every column top-to-bottom and carries FP32 totals across consecutive32-row tiles. Blackhole `ckernel_sfpu_cumsum.h:69–99` emits the ordered SFPADD chain, preserving the per-product FP32 sequence. There is no tree reduction or reassociation. However, cumsum initializes at **+0**, whereas FastReduce copies the first product. Signed-zero differences are therefore possible in a general all-negative-zero stream. The diagnostic reports all FP32 differences and requires every final BF16 output code to match; it does not waive an output mismatch. The current probe conservatively refuses timing whenever either requested FP32 or BF16 final output differs.

No in-place shortcut exists in the public cumsum wrapper. `accumulation_common.cpp` passes `std::nullopt` as the primitive output, allocates a new result, then rebinds a supplied `out` tensor's storage. Passing `out=products` would not avoid the new prefix allocation. The candidate does not use that misleading shortcut. Cumsum has no last-only output flag and always packs every prefix tile.

## Capacity and tradeoffs

| Singleton payload | Current v3 | Candidate |
| --- | ---: | ---: |
| BF16 coefficient tiles | 4,194,304 B | 131,072 B |
| FP32 products | 58,720,256 B | 50,855,936 B |
| FP32 all-prefix tensor | 0 B | 50,855,936 B |
| Additional resident BF16 weight | 0 B | 25,427,968 B |

The two large L1 tensors coexist during cumsum. The source uses the device's `compute_with_storage_grid_size()`, not a fixed8x8 grid (`accumulation_program_factory.cpp:84–95`). Existing actual `../profile_projection_l1/tracy/layer_0/ops.csv` reports **110 CORE COUNT and110 AVAILABLE WORKER CORE COUNT** for BinaryNG, supporting the earlier110-worker assumption for this hardware session. The new diagnostic records the actual mesh grid again before allocations/calls.

At110 equal banks, rounding each tensor to113 FP32 tiles per bank gives904KiB for product+prefix, plus32KiB/core cumsum buffers:936KiB before other live tensors, reserved/small-L1 regions, allocator alignment and fragmentation. This is nominally viable but **not a device allocation proof**. At64 banks, the corresponding product+prefix alone would exceed1.5MiB/core; the probe records grid/capacity context so such a failure is explicit. Nothing in this source analysis establishes that all interleaved L1 banks equal available workers on a different device.

The terminal32-row band is3,178,496 bytes for K512 or1,589,248 bytes for K1024. Converting that band and repacking its final row still costs real operations and padded output traffic. The candidate trades more L1 writes/reads and a larger terminal plane for less scalar-coefficient padding, more independent cumsum column owners, fewer ordered per-tile additions per owner, and different producer geometry. No performance claim follows from byte counts alone.

## Diagnostic and evidence

`v3_singleton_cumsum_cpu.json` passes selected-column FP32 partial and final checks on actual saved BF16 inputnorm rows at128/130/7703, real checkpoint K/V/A/B weights, and both K policies. This checks the starting convention on these fixtures, not all-input equivalence. No heavy CPU work remains queued by this investigation.

The device probe compares **all6208 columns** at both FP32 and BF16 boundaries and checks first/last timed replay outputs. Warmed complete-adapter timing includes gather, coefficient tilization, product, all-prefix scan, both terminal slices, block reduction, cast and reshape. It runs physical chip0, records source/capture/weight hashes and observed compute grid, uses32MiB trace reservation, releases traces and closes the device in `finally`. Original probes and historical results remain unchanged.

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v3_singleton_cumsum_probe
```

Only a device-exact, materially faster complete adapter warrants a current-source full decoder trial. The conservative device-time and roofline stop gates remain open; this diagnostic does not establish completion.

## First device attempt and source-backed logical-output correction

The parent's first `v3_singleton_cumsum_l1_probe.json` failed at the final reshape, after constructing the cumsum/reduction path, before any candidate output comparison or timing. Device close is recorded true. The observed compute/storage grid is11x10=110, confirming that this device is not an8x8 grid. This failure establishes no numerical or timing result; its artifact and original snapshot remain unchanged.

Cause: `FastReduceNCDeviceOperation::compute_output_specs` (`fast_reduce_nc_device_operation.cpp:117–142`) derives its default output **logical** shape from the input's **padded** shape. Thus input partial `[blocks,1,1,6208]`, physically `[blocks,1,32,6208]`, produces logical `[1,1,32,6208]` when no output tensor is supplied. A direct logical reshape to `[1,1,6208]` correctly fails its volume check. The prior source plan had incorrectly assumed reduction would preserve the input's logical H1.

Corrected independent snapshot: `v3_singleton_cumsum_l1_logical_output.py.snapshot` (SHA256 `93135b1d3e7dd1d7b3d628dad87c50cdbb6cfb3787ed76828acd18488b57d273`). It creates an explicit FP32 TILE L1 output with logical shape `[1,1,1,6208]` **inside the actual linear method**, then passes `output=result` to the last FastReduce call. The final typecast/reshape remains unchanged. Its patch is relative to the preserved failed snapshot; provenance retains the original v3 base and binds the failure artifact.

This is supported by the operator contract, not an unchecked workaround:

- Validation accepts a preallocated tiled FP32 output of the same rank as input (`fast_reduce_nc_device_operation.cpp:52–66`). Both ranks are4.
- `compute_output_specs` immediately returns the supplied output's TensorSpec; `create_output_tensors` returns that same buffer instead of allocating the padded-logical default.
- The wrapper forwards `output` to the final primitive (`fast_reduce_nc.cpp:51–60`). This is a one-dimension reduction, so there is no intermediate output-spec ambiguity.
- `[1,1,1,6208]` and the failed default `[1,1,32,6208]` have **identical physical tiled shape** `[1,1,32,6208]` and194 FP32 output tiles. The writer coverage, reduction association, physical allocation size and output dtype are unchanged. Only the exposed logical height is corrected, matching the existing explicit-output pattern in v3 inputnorm.

The failed probe's exact source is archived as `v3_singleton_cumsum_probe_initial.py.snapshot`; its SHA matches `source_sha256` in the failure artifact. The live diagnostic only additionally binds `obj.mesh_device = mesh` so the corrected snapshot's explicit output allocation works with its lightweight fixture object. The original candidate default and original result path are retained; use explicit new paths to preserve failure evidence:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v3_singleton_cumsum_probe --candidate-snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_singleton_cumsum_l1_logical_output.py.snapshot --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v3_singleton_cumsum_l1_logical_output_probe.json
```

Snapshot/probe syntax was checked and the old probe hash verified. No arithmetic changed, so the earlier CPU mapping proof still applies; there was no heavy CPU rerun and no hardware execution by this investigation. Native output exactness, warmed timing and full-path allocation remain unverified until the corrected discriminator runs.

## Corrected device discriminator passed

Parent-owned `v3_singleton_cumsum_l1_logical_output_probe.json` is **exact** on all six real cases: positions128/130/7703 under K512 and K1024, all6208 columns at both FP32 and BF16 output boundaries. First and last timed replay outputs are exact, and the device closed cleanly. The observed grid remains11x10. No signed-zero exception or other output waiver was needed. The corrected provenance binds the probe, its sidecar and these source hashes; the initial failed artifact/snapshot/probe remain preserved.

The explicit output preserves **logical H1 and physical tiled H32**. Both the failed default output and corrected output occupy194 FP32 tiles, with the same physical writer coverage and reduction arithmetic. Actual final outputs, rather than only source reasoning, now validate this correction.

Complete-adapter timings below summarize each K policy by the median of its three position-specific medians (not a pooled confidence estimate):

| K | v3 adapter ms | Corrected cumsum adapter ms | Latency reduction |
| --- | ---: | ---: | ---: |
| 512 | 0.308522 | 0.279835 | 9.30% |
| 1024 | 0.445440 | 0.250575 | 43.75% |

The full snapshot benchmark also completed while this evidence update was prepared: `v3_singleton_cumsum_l1_logical_output_s256.json` records **139.800522 ms prefill / 1.360551 ms decode**. The source-matched v3 comparator `v3_movement_baseline_s256.json` records **138.982528 / 1.393293 ms**. Decode is 2.35% lower; the multirow prefill path is unchanged. All reported prefill/contribution/eager/traced PCCs match exactly and replay is deterministic. These short results support further qualification, not production selection or the overall stop gate. No hardware, production or core-harness changes were performed for this update.
