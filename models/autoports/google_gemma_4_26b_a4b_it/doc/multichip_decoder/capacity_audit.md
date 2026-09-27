# Full-context reservation and prefill capacity audit

2026-09-26, source/CPU only. No TTNN import or device command was run. This audit
applied the runner fixes and wrote [prefill_capacity.patch](prefill_capacity.patch)
as a proposal. The parent subsequently applied the runtime patch; reverse
application check matches runtime SHA-256
`d3dee9c50f549b6a48e657f0f8b2745a42c436a81ebecb8d8b099e1ae34f454b`.
Hardware remains owned by the parent agent. All execution claims remain pending.

**The original 262143-prefill/1-decode hybrid command is not covered by the
existing 29.085 GB bound.** Native concat retains additional full-length
intermediates. The corrected current-path bound, including the independent
2 GiB reserve and reservation rounding, exceeds decimal 32 GB. The proposed
bounded output assembly restores the three-buffer bound while retaining the
requested nonaligned length and final absolute position. An aligned probe is
useful diagnosis, not a replacement for nonaligned maximum-context acceptance.

## Applied runner fixes

Only `tests/run_multichip_decoder.py` was changed:

1. Allocate both tiled prefill and row-major decode RoPE before reservation.
   Previously `current_rope` subtracted both layouts while only prefill tables
   were resident, under-reserving during prefill by 268,435,456 bytes for sliding
   or 536,870,912 for full at maximum context.
2. Delete device `y` after the warmup host read and after each timed prefill.
   Python otherwise retains the old full output while evaluating the RHS of the
   next `y = prefill_forward(...)`, adding another 1,476,395,008-byte tensor.
   Host `prefill` and the final `outputs` dictionary hold CPU tensors, not those
   device allocations. Profiling signposts and the warmup/timing order remain
   unchanged. One timing sample still means **two full prefills**, warmup plus
   measured call.
3. Select the extra EP weight allowance by attention kind rather than
   `layer == 0`; other sliding layers previously received the smaller full-layer
   allowance.

AST parsing and Black check passed without importing the runner. Runner hash at
release was `907e8e3f7daad0b4c73320787d28a95211aa83799aca7c36f7b250bc85bfd1b0`;
the parent subsequently added its separate shared-MLP candidate flags.

## Reservation arithmetic

For the requested replicated-layout hybrid command, the existing plan gives
24,655,921,152 bytes per device of resident payload/state bounds plus the
independent 2 GiB workspace/trace/allocator reserve. Reservation subtracts that
2 GiB, the actual tested layer's weight bound, its actual cache extent and both
RoPE layouts. The remaining payload is rounded upward to 64 MiB allocations.

`ttnn.empty((1,1,32768,1024), BF16, TILE, DRAM)` allocates 67,108,864 bytes per
rank. `ttnn/cpp/ttnn/operations/creation/creation.cpp:305` calls
`create_device_tensor`; `ttnn/core/tensor/tensor_ops.cpp:76` defaults the mesh
topology to replication, so this is not a 64 MiB total divided by four. The
Python `reservations` list retains all blocks through prefill and decode.

| Maximum-context quantity, bytes/device | Sliding layer 0 | Full layer 5 |
| --- | ---: | ---: |
| Tested TP+EP layer weight/state bound | 383,254,528 | 256,737,280 |
| Tested K+V cache | 285,212,672 | 285,212,672 |
| Tested cos/sin, both layouts | 536,870,912 | 1,073,741,824 |
| Required other-resident reservation | 21,303,099,392 | 20,892,745,728 |
| Number of 64 MiB allocations | 318 | 312 |
| Actual anonymous reservation | 21,340,618,752 | 20,937,965,568 |
| Upward rounding excess | 37,519,360 | 45,219,840 |

The reservation intentionally leaves the 2 GiB allowance free for real runtime
use; it is included in every conservative peak below. This tests anonymous DRAM
capacity against a calculated payload plan, not construction/execution of all
30 layers, the real allocation size distribution, or actual allocator peak.
The tested layer's 8 MiB small-state allowance is also a bound rather than an
exact live-byte measurement.

The proposed command does not enable the parent's new `--optimized-shared`
option. That option retains the original BF16 shared-MLP closures and adds
decode-only quantized matrices. If enabled, add `3*88*17*1088 = 4,882,944`
bytes/sliding layer and `3*88*17*576 = 2,585,088` bytes/full layer, or
134,999,040 bytes for the full stack, to the resident plan. Reservation must
then subtract the tested layer's corresponding extra bytes. Do not assume those
new matrices replaced the BF16 prefill weights.

## Page, RoPE, and endpoint contracts

`length=262143, steps=1` produces `extent=262144` after 1024-token rounding,
8,192 pages of 32 tokens, and RoPE for absolute positions 0..262143. Prefill's
last chunk starts at 261120 with 1,023 valid rows and 1,024 physical rows. Its
final padded row is inside the same request's last page. Decode writes position
262143. Both the SDPA 128-token-rounded exclusive read end and the page capacity
are exactly 262144, so no additional page beyond the model context is needed.

`rt_args_common.hpp:45-65` rounds the SDPA read window; the current FP32-destination
decode factory caps dynamic chunks at four 32-token tiles. Sliding's last
logical window is [261120,262144); full attention reads [0,262144). The runner
permutes physical page IDs, exercising absolute logical-to-physical addressing.

The RoPE formula `2*(cos.numel()+sin.numel())*2` correctly represents two
layouts, cos/sin, and BF16 storage. At maximum extent the widths are 256 sliding
and 512 full; the latter follows the proportional full-attention
`global_head_dim` path in `Gemma4TextRotaryEmbedding`. The full-stack plan shares
one pair per kind/layout; anonymous reservation stands in for the other kind.
It does not validate a future stack owner's sharing implementation.

## Why the old output bound was incomplete

Let `A = 262144*2816*2 = 1,476,395,008` bytes (1.375 GiB). The inherited
`OptimizedDecoder.prefill_forward` at `tt/optimized_decoder.py:700-735` retains
256 sliced chunk outputs and then calls `ttnn.concat(outputs, dim=2)` while the
full input remains live.

- `concat/device/concat_device_operation.cpp:240-317` sets an interleaved concat
  limit of 47 input tensors. At line 338 it concatenates batches and retains all
  six intermediate results until a final concat returns. Even aligned prefill
  can therefore overlap input, original chunks, batch results and final output:
  **four full hidden equivalents**.
- `concat/concat.cpp:93-133` detects padding in the concat dimension. One final
  1,023-row tile makes it untilize **every** chunk into row-major storage before
  concatenation, then retilize the final result. The native unaligned tiled
  fast path is for width concat, not this height concat.
- `data_movement/common/common.hpp:225-232` keeps `formatted_input` alive during
  both the operation and post-transform. At final row-major concat, input,
  original chunks, untilized chunks, batch results and final row-major output
  coexist. At retilization, the batch results have left scope, but input,
  original chunks, untilized chunks, final row-major and final tiled output
  coexist. Thus a conservative bound is **five equivalents**, or 6.875 GiB.
  Counting padded maximum-size tensors slightly overestimates row-major storage
  by one logical row and is intentional.

All paths below include the 2 GiB independent reserve; all numbers are source
bounds, not measurements:

| Maximum-context execution path | Sliding bytes/device | Full bytes/device |
| --- | ---: | ---: |
| Current nonaligned inherited concat, five equivalents | 32,075,415,552 | 32,083,116,032 |
| Diagnostic aligned 262112 prefill + 32 decode, four equivalents | 30,598,299,648 | 30,606,000,128 |
| Proposed bounded concat, original 262143 + 1 command, three equivalents | 29,122,625,536 | 29,130,326,016 |
| Proposed path headroom to decimal 32 GB | 2,877,374,464 | 2,869,673,984 |

The old nonaligned path exceeds the conservative decimal capacity comparison;
physical 32 GiB would provide a different margin but is not a justification for
retaining the incomplete plan. `memory_capacity_plan.json` currently describes
three live buffers, so its peak applies only after an output-assembly repair or
another source-proven equivalent change. It remains `calculated_unvalidated`.

## Proposed model-local repair

[prefill_capacity.patch](prefill_capacity.patch) adds only a
`MultichipDecoder.prefill_forward` override, based on runtime SHA-256
`5f75aa5f250b0424b93cfdd5a9fa036f9b322fca38bb7eab40b8c2f5c4764dff`.
No import or C++ change is needed. It preserves the existing chunk/cache/RoPE
calls and changes fresh replicated multi-chunk output assembly:

1. Retain at most 32 physical, tile-aligned chunk outputs; concatenate that
   group and clear its Python input list. At maximum context there are at most
   eight groups, so neither concat call crosses the native 47-input threshold.
2. Keep the final physical tile until assembly is done. There is no logical
   1023-to-1024 reshape: `_forward` already receives and returns 1024 physical
   rows. For shorter sliding tails, an aligned slice keeps only
   `ceil(valid/32)*32` rows. This avoids the all-input untilize fallback.
3. Concatenate groups, clear their references, then slice the final tensor to
   the requested logical length. If slicing makes a full copy, input + assembled
   output + final logical output still fit within three equivalents.

The final slice starts at zero with unit steps. `data_movement/slice/slice.cpp:247`
selects the tiled path based on aligned starts, not aligned ends; lines 331-385
round the physical end and apply a logical view. This final nonaligned end does
not trigger a full-output untilize/retilize fallback. The capacity bound still
allows one full tiled output allocation for that slice.

Only ordinary reference release is used (`list.clear`, `del out`); there is no
forced tensor deallocation that could invalidate a single-input/group alias.
The 32-output group contributes at most 184,549,376 bytes of bounded pending
chunks; group assembly peaks below the final three-equivalent peak. Chunk-local
attention/MLP work remains within the separate unvalidated 2 GiB allowance.

Prefix continuation, single-chunk inputs and `sharded_residual=True` delegate
unchanged to the inherited implementation. `user_id`, per-request page tables,
cache updates, the sliding tail's final release, and batch/decode behavior are
unchanged. This patch's capacity claim is specifically fresh prefill in the
requested replicated hidden-width-2816 layout; it does not fix the inherited
long sharded-residual or long prefix-continuation assembly path.

Direct `slice_write` into a tiled interleaved full output is not a suitable
drop-in: `experimental/slice_write/slice_write.cpp:30-34,119-127` converts that
input and the full destination to row-major, writes, and converts the full
destination back for each chunk. A row-major destination held throughout or a
tiled sharded-input path is a separate implementation experiment.

## Focused verification order

The parent owns all execution and hardware recovery. Keep commands serialized,
bounded and under the stage's required Watcher settings; inspect complete exit
and closure before starting the next process.

1. After applying/reviewing the proposed runtime patch, run a short replicated
   hybrid/fused-tail correctness and cache-preservation case for both layer 0
   and layer 5, including existing prefix/heterogeneous decode contracts. Check
   the repaired setup reservation independently with a short input.
2. Exercise `length=32769, steps=1` for both kinds. This crosses the new 32-chunk
   group boundary and ends with a one-token sliding tail. Verify logical output
   shape, all-rank equality, preserved cache contents, and available reference
   correctness. A paired TP1 comparison is meaningful here; a TP4-only capacity
   run reports `passed=null`, not a PCC result.
3. Run the requested exact endpoint first on sliding and then on full attention:
   `--length 262143 --steps 1 --repeat-input --prefill-timing-samples 1 --tp 4
   --hybrid-experts --fused-tail --reserve-full-stack`, with `--layer 0` or
   `--layer 5` and a distinct output path. Add trace acceptance as a separate
   explicit gate if this run does not use `--trace`. Full attention's work grows
   with context; the two full-prefill calls need a suitable bounded timeout.
4. Record allocation/peak evidence, finite outputs, exact logical output length,
   all-rank equality, endpoint cache update and complete process closure. The
   anonymous reservation does not by itself establish full-model correctness,
   all-layer allocation topology, or a measured full-stack memory peak.

CPU verification: runner AST and Black pass; proposed candidate AST parses;
`git apply --check` passed against the recorded base, and reverse application
check passed after the parent applied it; 26 host-only chunk
geometry cases cover both kinds, 32-chunk boundaries and lengths 262112,
262143 and 262144. These checks are not TTNN execution or correctness proof.

## Follow-up audit of optimized shared reservation

The parent's updated runner SHA-256
`6088e95b3cf4aece2659ef1c816bc7e2c91cee1eecce92fb3b41b5a73e4239f7`
now adds `optimized_shared_decode.extra_full_stack_bytes` to resident bytes and
the matching per-kind extra to `current_weights` before subtraction. This is
correct: the reservation represents the other 29 layers' extra matrices, while
the tested layer actually owns its extra tensors. Both RoPE layouts are still
allocated before subtracting their bytes; warmup/timed outputs are still
released. No additional runner error was found in this source check.

For hybrid experts **and** optimized shared decode at extent 262144:

| Quantity, bytes/device | Sliding | Full |
| --- | ---: | ---: |
| Tested layer weight/state bound | 388,137,472 | 259,322,368 |
| Anonymous reservation, 320 / 314 blocks | 21,474,836,480 | 21,072,183,296 |
| Reservation rounding excess | 41,620,992 | 47,023,616 |
| Three-buffer peak including 2 GiB reserve | 29,261,726,208 | 29,267,128,832 |
| Headroom below decimal 32 GB | 2,738,273,792 | 2,732,871,168 |

These totals rely on the parent's bounded-concat override and fresh replicated
prefill. The plan's generic long-prefill assumption should identify that
implementation, since the inherited unbounded path does not satisfy the same
three-buffer bound. `memory_audit.md` records an earlier source where the dual
layout was hypothetical; this follow-up and the plan's current `implemented`
field supersede that historical implementation-status statement. No allocator
or full-stack execution claim follows from this arithmetic.
