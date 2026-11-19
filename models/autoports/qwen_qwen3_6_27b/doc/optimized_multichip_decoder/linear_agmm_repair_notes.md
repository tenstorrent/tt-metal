# Conventional Linear AGMM receiver repair

2026-11-13. AutoFix implementation following `debug_conventional_agmm/AUTODEBUG.md`; hardware proof is pending. No device commands or native build/install were run by this implementation agent.

The prior host guard masked two independent receiver problems: the conventional factory's four directional transfers implied eight source slices, and simply reducing that count would still request unavailable endpoint credits. The repair selects an explicit `LINEAR_FULL_SLICE_V1` capability only for validated Linear conventional AGMM. The legacy Ring path and its runtime ABI are preserved.

Both interleaved operand readers use the production `linear_full_slice` helper: local source first, then increasing distance with lower before higher source, without wrapping. A waits for complete-slice publication on dir0 (distance) or dir1 (distance + local credit); B follows the identical source order without waiting. Each source contains 40 K tiles. Its operand offset is 40 A tiles or 40 * N B tiles, added to the existing tensor/core-column base. Kblock4 and Kblock8 divide each source into ten and five blocks respectively. The nine-word Linear receiver record uses the actual full-slice count in word 0; both waiting and nonwaiting readers consume all nine words. Ring serialization and parsing remain unchanged. Program-local readiness semaphores retain their ordinary initialization each launch.

Host validation bounds initial support to the complete 1x4 placement, Linear width gather, one link/two workers per direction, default subdevice, interleaved DRAM operands/scratch/output, ordinary 32x32 tiles, no bias/transposition/prefetcher, logical M fitting one physical tile, local/full K1280/5120, BF16 activation/output and BF16/BFP8_B weights. Explicit 1D mcast-in0 or 2D multicast configs must have one output h/w block per core; sharded/SINGLE, gather-in0, transposed multicast, output sweeps and insufficient compute coverage reject. The conventional factory reserves the entire configured multicast rectangle against every AG descriptor kernel, including the unused physical destination of the 31-core 1D N tail. The selected mode is a deterministic function of topology, already included in the program hash; kernel defines also distinguish the new ABI.

Companion admission changes are owned by `linear_agmm_admission`: the capability permits the prior guard exception only with exact FABRIC_1D, full-range global admission even for persistent scratch, atomic consumption of this epoch's barrier credits, and explicit worker-vector/index agreement. They must ship together with this receiver change. Existing cache-hit buffer and caller-global-semaphore rebinding remains unchanged. Native MMRS edits were not touched.

## CPU evidence

Command:

```bash
python models/autoports/qwen_qwen3_6_27b/tests/check_linear_agmm_receiver_cpu.py \
  > models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/linear_agmm_receiver_cpu.json
git diff --check
```

PASS. The test extracts the actual production `advance_start_page_idx` and `MatmulOpReceiver`, compiles them with CPU argument/semaphore stubs, then executes them. It covers 96 Linear and 192 TP8 Ring receiver combinations: all ranks, both operands, Kblock4/8, logical M1/32 (both physically one tile), and three nonzero/zero base offsets including N-tail core columns. It checks every block address, unique K coverage, matching operand source sequence, counter bounds, A waits/B nonwaits, and parser cursor. Three Linear invalid count/divisibility/empty-slice cases reject. The only compiler warning is an existing unused variable in the excluded sharded helper. This is receiver/control/address evidence, not dataflow, Fabric, allocation, host validation execution, or hardware numerical proof.

## Files

- `ttnn/cpp/ttnn/operations/ccl/ccl_op_fusion.hpp` and `.cpp`: capability metadata and nine-word serialization.
- `ttnn/cpp/ttnn/operations/ccl/shared_with_host/linear_full_slice_schedule.hpp`: shared deterministic source/credit schedule.
- `ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp`: Linear parser and interleaved receiver schedule; existing shared A/B kernels include this helper without new reader source edits.
- `ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_{1d,2d}.cpp`: actual CCL descriptor dataflow defines.
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_{device_operation,program_factory}.cpp`: bounded host gates, mode propagation and physical destination exclusion.
- `tests/check_linear_agmm_receiver_cpu.py`: CPU production receiver regression test.

## Remaining proof and risks

Owner must build/install host runtime, run host negative cases, and run the focused fresh-cache watcher probe before any performance result is accepted. Validate every output shard and gathered scratch with rank-distinct inputs, K-source-distinct/one-hot/cancellation weights; M1/32; both exact 1D31-core tail and 2D8x1 layouts; Kblock4/8; and cached replacement of all tensor/global semaphore addresses. Run at least 50 changed-generation queued launches with deliberate skew and bounded synchronized readback to test persistent admission. Keep unchanged numerical gates: source-order accumulation can differ from an unfused natural K order. Physical Fabric readiness/visibility, worker-barrier behavior, L1 feasibility, runtime compilation, and throughput are unverified here. Measure the whole hidden-sharded residual MLP family, including admission, against current separate/minimal paths. No receiver-only timing would establish a family win.
