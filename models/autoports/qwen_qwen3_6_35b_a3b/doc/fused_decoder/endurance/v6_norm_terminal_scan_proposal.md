# Tiny exact-normalization terminal scan fusion: ready for device test

Source-only AutoFix hypothesis, no hardware run or speed claim. The ab28 movement audit identifies this as untried; prior projection terminal scans cover K512/K1024 tensors, not the two `(rows,1,4,8)` → `(rows,1,1,8)` → `(rows,1,1,1)` norm scans. Existing last-reduction alternatives and weight folding are different arithmetic boundaries and are not reused here.

Prepared component: `tests/v6_norm_terminal_scan_probe.py`, SHA256 `85d00325f34ab3ac803fbc9b4acb917b9bd6d3decacb4b3d15c0c52c8104408c`. CPU setup: `v6_norm_terminal_scan_cpu_setup_v2.json`, with exact source, three extracted norm methods, three C++ kernels and fixtures beside it. Pinned Black23.10.1/autoflake2.3.1/isort5.13.2 pass. No production, frozen helper, source compatibility manifest, or shared/native code changed.

## Exact boundary and native basis

`ab28` lines1002–1093 use native `fast_reduce_nc` to form 32 FP32 partials, reshape to four groups of eight lanes, scan groups along H and crop row3, then scan lanes along W and crop lane7. This component replaces only those two scan/crop assignments. Re-inserting the original assignments restores the entire extracted norm method AST exactly. The preceding multiply/reduction/reshape, following scaled epsilon, split-product square-root/reciprocal refinement, learned weight multiply and BF16 result are unchanged.

Native basis, raw hashes recorded in setup:

- `ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/compute/accumulation_tile_axis_compute.cpp`: H scans copy to DST and call `cumsum_tile<CumsumMode::Default>`; W scans transpose first, run the same scan, then transpose back.
- `ttnn/cpp/ttnn/operations/reduction/accumulation/device/accumulation_program_factory.cpp`: FP32 direct-to-DST unpack, HiFi4, precise SFPU, 32-bit destination, disabled compensated sum.
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cumsum.h`: sequential native SFPADD replay order with zero carry on each first tile.

The one-dispatch generic adapter retains two native Default scans and their FP32 pack boundaries. Between them, BRISC copies packed row3/lane0..7 bits (`48+j` in face0) into the transposed column (`16*j`). After the second scan it copies row7/column0 (`112`) into output scalar lane0. This substitutes bit movement for native crop/transpose materialization, with no added floating-point arithmetic. Unused padding is zeroed; later padded elements cannot affect either selected prefix. Both inputs unpack directly into FP32 DST. Runtime performs no Torch, host tensor conversion, RM layout conversion, reshard, or approximate reduction.

Each active core uses seven FP32 CB pages (28 KiB). Rows are partitioned contiguously across available worker cores; reader/compute/writer use bounded CB handshakes. One token uses one core. Optional debug mode emits the eight intermediate lane sums; timing disables debug. No instruction-order or FP32/subnormal equivalence is claimed until the strict device probe passes.

## Fixture and timing coverage

Nineteen complete norm cases include actual layer34 positions227/738/5799 for input and postnorm, random logical rows1/3/33/65, zeros, and alternating signs. Actual postnorm is also tested in the production 32-core WIDTH_SHARDED `(32,64)` input layout, preserving the norm's own DRAM conversion and shard restoration; DRAM variants cover the independent placement-removal candidate. Timing covers complete baseline/candidate norm calls for actual227 input, postnorm DRAM, and postnorm WIDTH_SHARDED. Input setup happens outside both timing paths. No `_forward` outer placement removal is bundled here.

Norm weights use checkpoint `weight.float()+1` uploaded FP32, exactly matching ab28 `from_state_dict`; epsilon1e-6 is checked against checkpoint config. The first CPU-only setup (`v6_norm_terminal_scan_cpu_setup.json`, source f3a74055) used an incomplete weight adapter; it remains preserved but is superseded before any device run by v2's corrected FP32 `+1` setup and producer-layout cases.

Separate scan fixtures contain768 BF16 midpoint/adjacent-FP32 totals and five signed-zero/subnormal/cancellation controls. All intermediate FP32 lane bits, terminal FP32 bits and native-cast BF16 bits must match. Actual complete norm outputs must match every BF16 bit. The native BF16 typecast is retained, so this does not repeat the rejected custom pack-rounding shortcut. Subnormal controls are compared against the actual native path without excusing a mismatch. Complete-adapter timing includes three warmups, five eager samples, and five batches of20 trace replays, followed by exact replay-output checks.

## Parent serial device command

Run only after the current chip0 job closes; host topology requires visibility unset. Preserve the fresh result path and log. The probe is device-opt-in, and CPU setup never imports TTNN.

```bash
timeout 900 env -u TT_VISIBLE_DEVICES -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE \
  PYTHONPATH=. python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_norm_terminal_scan_probe \
  --run-device --timing \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_norm_terminal_scan_device.json \
  > models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_norm_terminal_scan_device.log 2>&1
```

A component win is only eligibility for a standalone integration and full semantic/performance requalification. It does not establish decoder acceptance, a latency improvement, or the no-unnecessary-movement gate by itself.
