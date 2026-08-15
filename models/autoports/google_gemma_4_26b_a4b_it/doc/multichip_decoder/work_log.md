# Multichip decoder work log

## 2026-08-15 inventory and strategy selection

- Baseline commit: `8bc06c55da5` (`optimized_decoder.py`).
- Unrelated pre-existing worktree state, excluded from this stage:
  `models/autoports/qwen_qwen3_6_27b/` and `third_party/tt-metal/`.
- `timeout 60 tt-smi -ls --local`: four Blackhole P300C chips visible.
- First `1 x 4` mesh probe opened the mesh but the diagnostic called the absent
  `MeshDevice.get_devices` binding before explicit close. This was an
  inspection-script error, not model evidence. Per `$tt-device-usage` the
  devices were reset with `timeout 180 tt-smi -r`, all four chips relisted, and
  a bounded `open_mesh_device(MeshShape(1,4))` / close smoke printed
  `MESH_SMOKE_OK`.
- Read `tech_reports/LLMs/llms.md` section 3.3 and the GPT-OSS/common-module
  references required by `$multichip`.
- Selected plan recorded in `mesh_plan.md` before implementation.

## Bringup experiments

- Target-ring hidden all-reduce smoke: passed for the exact batch-32 BF16
  payload `[1,1,32,2816]`; device inputs 1/2/3/4 reduced to 10 on every chip.
- First real-weight sliding-layer run reached TP attention, then raised
  `AttributeError: decode_dram_weights` at optimized dense-MLP dispatch.
  Hypothesis: the multichip setup path omitted empty candidate dictionaries
  normally installed by `OptimizedDecoder.from_state_dict`. Verified directly
  from the traceback and constructor paths. Fix: initialize all four empty
  decode-DRAM dictionaries. No math, dtype, or topology change was made.
- Second sliding run passed attention and dense MLP, then the expert-down
  sparse program rejected inherited `in0_block_w=11`: TP-local padded expert K
  is 192 elements = 6 tiles. The exact divisibility prediction is therefore
  verified. Both prefill and decode expert-down configs now use block width 6,
  consuming the complete local K in one block.
- A direct comparison initially attempted to create an overlapping `1x1`
  submesh under the active `1x4` fabric mesh. Host setup made no progress for
  five minutes, so the invalid test architecture was terminated. The devices
  were reset, relisted, and passed the `1x4` open/close smoke. The replacement
  workflow serially captures optimized `1x1` output artifacts and then compares
  TP4 output in a separate target-mesh run; both layer kinds passed PCC 0.995.
- Long-context routing audit found the first TP implementation always used
  square SDPA. Restored the optimized baseline's sliding/full chunk routing
  with TP-local Q/KV head shapes. Sliding non-aligned prefill at 262143 passed.
  Full initially rejected decode-only `block_size`/`num_kv_heads` kwargs on the
  chunked prefill API; removing those unsupported kwargs fixed the exact
  traceback. After reset/relist/smoke, full prefill at 262143 passed in 53.285 s.
- Advertised-context traced decode at position 262143 passed for both layer
  kinds with rolled page tables, cache sentinels, and repeat PCC 1.0.
- An attempted reuse of the HF batch-32 helper patched global KV-head constants
  too early and invalidated the host HF oracle (2 expected heads vs 8). It was
  removed rather than weakening the oracle; the stage does not advertise a
  new multichip batch-32 gate. Reset/relist/smoke passed afterward.
- The first watcher invocation instrumented Ethernet and exceeded the
  Blackhole active-ERISC kernel-config buffer (`27920 > 25600`) before model
  execution. After reset/relist/smoke, the supported
  `TT_METAL_WATCHER_DISABLE_ETH=1` configuration retained worker watcher
  coverage and passed real-weight plus traced tests for both layer kinds (4/4).

## Final evidence

- HF PCC: layer 0 prefill/decode `0.9986126386 / 0.9996529166`; layer 5
  `0.9970876597 / 0.9997855512`.
- Direct optimized-TTNN vs TP4 prefill: both layer kinds pass PCC >= 0.995.
- Warmed batch-1 trace: sliding `1.122 ms` vs `1.272 ms` (`1.133x`, 28.33%
  efficiency); full `1.220 ms` vs `1.270 ms` (`1.041x`, 26.02%).
- Trace replay is bit exact across repetitions and replicated residual ranks.
  Full KV pair ownership and distinct-head behavior are device-read verified.
- Tracy and `tt-perf-report` artifacts exist for both layer kinds. Modeled DRAM
  roofline is 6.4%/33 GB/s sliding and 5.4%/28 GB/s full. CCL appears as
  all-gather plus fast-reduce; sparse expert matmuls remain active.
- Runtime fallback audit used `throw_exception_on_fallback=true` throughout.
- Stage review and commit SHA are recorded below after their final gates.

## Stage-review repair loop

First independent review returned `more-work-needed` on six items. `$autofix`
and isolated gates resolved them as follows:

- Optimized references now contain both prefill and decode outputs. TP4 passes
  direct PCC 0.995 for both outputs and both layer kinds.
- Added identical current-source batch-32 baseline/TP harnesses with local
  page/cache layout and trace determinism. Sliding is `14.842 -> 12.691 ms`
  (`1.169x`); full is `14.908 -> 12.813 ms` (`1.163x`). Both pass optimized
  PCC 0.995 and bit-exact eager/replay/repeat checks.
- `$autofix` proved the 1D reduce-scatter alternative cannot reduce bytes:
  `[1,1,32,2816]` BF16 ring AR and RS+mandatory AG each move approximately
  270336 bytes/device. All next projections retain global K=2816; distributed
  norm adds statistics traffic, and fused matmul+RS is disabled on Blackhole
  by nondeterministic race #46181. The selected ring path is retained.
- The first TP bounded-modulo test exposed inherited global `8` KV-head slicing
  against a local `2`-head tensor. A multichip-local modulo-safe fill override
  fixes the exact traceback. After reset/relist/smoke, logical 1025 passed with
  slot 0 replaced and slots 1..1023 preserved at PCC 0.9999.
- Recomputed stack cache: five BF16 full layers use 2.5 GiB/device at maximum
  context; 25 bounded 1024-token sliding layers use 0.048828125 GiB/device.
  The full-context sliding artifact is explicitly a stress allocation, not the
  stack allocator.
- Added `artifacts/evidence_manifest.json` with actual wrapper commands and
  SHA-256 hashes. Regenerated optimized references after the decode additions.
- Added signposted batch-32 replay profiles. Decode-only modeled DRAM roofline
  is 7.4%/38 GB/s sliding and 7.3%/37 GB/s full; CSV and human reports retain
  isolated CCL, sparse expert, matmul, attention, and movement evidence.
- A failed read-only DRAM API probe opened the mesh before discovering that the
  queried Python bindings were absent. Per `$tt-device-usage`, all four devices
  were reset/relisted and a `try/finally` 1x4 open/close smoke passed before
  acceptance resumed.
- Closed the stack memory audit against 32 GiB/device: the conservative total
  is 19.411328125 GiB/device, including 6 GiB activation/workspace and 4 GiB
  program/allocator reserves, leaving 12.588671875 GiB/device.
- The signposted batch-32 report is sparse/layout dominated (~53% sparse
  matmul, ~26% transpose/unary, ~1.35% CCL). The bounded expert-down program
  sweep selected the largest legal local-K block (6 tiles); 1/2/3 add loops.
- Exact hook-clean source watcher acceptance passed 9/9 tests in 97.84 s with
  worker watcher enabled and Ethernet watcher disabled. A subsequent ordinary
  run restored uncontaminated warmed timing artifacts: batch-32 sliding
  `14.842 -> 12.691 ms` and full `14.908 -> 12.813 ms`; batch-1 sliding
  `1.272 -> 1.122 ms` and full `1.270 -> 1.220 ms`.
- Final fresh `$stage-review`: `clean-pass`; no required work, hard-check gaps,
  or blocking concerns remained.
- Stage implementation commit: `aef721434f4` (local only; never pushed).
