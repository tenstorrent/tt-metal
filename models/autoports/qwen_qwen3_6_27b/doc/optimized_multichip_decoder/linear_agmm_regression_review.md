# Linear AGMM regression and admission review

2026-11-13; independent CPU-only AutoFix review of the current native diff against
HEAD `8502f42bca20dcad1507fbc8cebf7032220b7d8e`. No TTNN import, device access,
native build, or source modification. Only this report was written. This is not
a stage review, performance result, or completion verdict.

**Verdict:** no supported Ring behavior change found in the receiver/admission
diff. Two persistent-scratch checks are missing from the newly admitted Linear
contract. Fix these before treating host admission as complete. The focused
positive tests do not exercise either mismatch.

## Findings requiring repair

1. **A scratch tensor on fewer mesh coordinates can shrink the workload below
   TP4 while the collective still expects four participants.**

   `all_gather_matmul_async_device_operation.cpp:41–50` verifies complete input
   placement and equal weight coordinates, but its scratch block at lines74–82
   checks only memory/tile properties and buffer aliasing. Generic matmul
   validation checks the scratch and weight device pointers, not their coordinate
   lists (`operations/matmul/device/matmul_device_operation.cpp:36–63`). The
   delegated all-gather validation checks allocation, padded dimensions,
   dtype/layout/page config and memory config, not placement
   (`all_gather_async_device_operation.cpp:49–92,159–192`).

   This is not covered by generic launch validation:
   `ttnn/core/device_operation_detail.cpp:187–232`,
   `extract_tensor_coordinates_impl`, explicitly accepts a strict coordinate
   subset and returns that subset. `ttnn/api/ttnn/device_operation.hpp:345–358`
   uses it to construct the workload; `mesh_device_operation_utils.hpp:40–65`
   also filters returned tensors to those coordinates. The fused operation's
   `tensor_args` includes persistent scratch. Thus a same-device scratch view on
   fewer ranks is not equivalent to the proven four-rank contract; a missing-peer
   wait is a source-supported risk, not a reproduced hardware hang.

   Minimum repair, inside the Linear scratch block: require
   `scratch.device() == input_tensor.device()` and
   `std::ranges::equal(scratch.device_storage().get_coords(),
   input_tensor.device_storage().get_coords())`, with a dedicated message such as
   `Linear AGMM persistent scratch must occupy the complete operand mesh`.
   Keep the complete input/weight checks. Do not generalize this fix to the Ring
   path or generic all-gather without separate evidence.

2. **Logical scratch rows may differ from input rows while all existing shape
   gates pass.**

   Input `[1,1,32,1280]` and persistent scratch `[1,1,1,5120]` both have padded
   row count32. The all-gather validator compares only padded rows; matmul sees
   scratch logical M1 and accepts its K5120 against weight K5120. All-gather
   `create_output_tensors` returns the supplied scratch directly
   (`all_gather_async_device_operation.cpp:281–285`). The fused wrapper constructs
   matmul attributes using that scratch (`all_gather_matmul_async_device_operation.cpp:335–355`),
   and output allocation uses it too (lines221–227). The expected public logical
   result can therefore be truncated to one row; conversely input M1 plus scratch
   M32 can expose rows that are merely padding. This is a source-derived admission
   counterexample, not a device reproduction.

   Minimum repair: construct expected logical and padded shapes from input,
   multiply dimension3 by `ag.ring_size`, and require exact equality to both
   scratch shapes. Suggested message:
   `Linear AGMM persistent scratch must match the exact gathered logical and padded shapes`.
   This also states the full-K scratch bound directly instead of depending on
   delegated matmul validation. Preserve valid public M1 through M32.

Focused negative cases after rebuilding the guard: M32 input/M1 scratch;
M1 input/M32 scratch; a same-device scratch view on a strict coordinate subset.
Require each dedicated native rejection before launch, on a cold program cache
and after a successful matching call. Construct the subset by a storage view of
the full mesh, not a separate submesh device, so the coordinate check itself is
covered. Do **not** run the subset case against the current unchecked runtime.
The supported Python construction is
`subset = ttnn.get_device_tensors(scratch)[0]`; assert
`len(subset.device_coords()) == 1` and unchanged `shape`/`dtype` before passing it
as persistent scratch. The binding is
`ttnn/core/distributed/distributed_nanobind.cpp:922`; the device implementation at
`ttnn/core/distributed/api.cpp:68–87` calls `DeviceStorage::single_shard`.
`ttnn/core/tensor/storage.cpp:139–154` shares the original mesh-tensor holder and
root holder, changing coordinates/topology only. It neither opens a submesh nor
copies tensor contents. The caller must retain the original scratch owner.
Existing matching B1/B32 positive probes should still pass. A short-K scratch
case is useful to cover the exact-shape guard but is already rejected downstream
by matmul and is not an additional discovered correctness hole.

## Ring and ABI findings

- Both signaler structs initialize the new schedule to `RING`; the conventional
  factory selects `LINEAR_FULL_SLICE_V1` only when topology is Linear. Topology
  is already in its program hash. Both CCL multicast descriptor builders emit the
  receiver define only for that explicit capability. No other inspected caller
  receives an implicit new ABI.
- `push_matmul_fused_op_rt_args` retains the Ring word0 (`num_transfers`) and
  remaining eight words. With the Linear define absent, the receiver retains its
  original seven-word nonwaiting B parser, nine-word waiting A parser, alternating
  directions, address updates, and thresholds. Linear uses full count4 and consumes
  nine words for both operands, with matching source schedules and operand strides.
- One assertion, `num_blocks_per_slice > 0`, was added outside the Linear define.
  It rejects a zero divisor before the later modulo; it does not change an
  otherwise valid supported Ring schedule. The report does not claim byte identity
  for the entire receiver.
- The all-gather factory's barrier expression is algebraically unchanged when
  capability is false. Worker-index checks and the admission define are capability
  guarded. Removing the `LINEAR_FULL_SLICE_ADMISSION` conditional and retaining its
  default branch makes `minimal_default_writer.cpp` **byte-identical to HEAD**
  (independently checked with `git show` and a Python text comparison).
- New capability requires exact FABRIC_1D, Linear TP4, one link/two workers,
  MULTI, non-reversed ordering and a supplied barrier. The normal Linear fused
  guard remains for unrelated signalers. Shape/layout gates exclude sharded/SINGLE,
  extra output sweeps, transposed multicast and nonordinary tiles. The full
  configured multicast rectangle is checked against every AG kernel, including
  the 1D N-tail dummy destination. The worker-vector and runtime-index comparisons
  prevent reordered master identity. No additional placement hole was found in
  these physical-core checks.
- Persistent ownership remains a single-owner, ordered-CQ contract. The new
  admission mechanism does not authorize unrelated concurrent queues sharing the
  same scratch/semaphores. This documented caller obligation is separate from
  the two missing tensor checks above.

## CPU evidence rerun

Commands:

```sh
python models/autoports/qwen_qwen3_6_27b/tests/check_linear_agmm_receiver_cpu.py
python models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/linear_agmm_admission_cpu.py
```

Both PASS: actual extracted production receiver compiled for 96 Linear and 192
TP8 Ring cases, plus 3 invalid Linear parser cases; admission ledger checks 16
worker destinations, 3 credits each, 7 future-credit counts and both selected
worker placements. Existing unused-variable warning belongs to the sharded helper
outside this admitted path. These tests establish receiver/address/credit-model
properties, not hardware transport equivalence or numerical behavior.

Audited source SHA256:

- AGMM device-operation validation:
  `b25f4f5d6a49972235edcbc6ee28db18c9124154ad1d7894305c0ff1918c6636`.
- Shared receiver:
  `50fd87bb258be4865b56ded706494fa5447fa52f7fa9134228d32325d257aff1`.
- Admission factory:
  `d14f28181b6c559b142ba4d65932a31cf8a13698aa69586c56390e8b9e676069`.
- Admission writer:
  `f6e4c1d5a83031f40e6468e02942902a25f35d1ae95efce69005cb6d44a54313`.

Paths abbreviated above are under `ttnn/cpp/ttnn/operations/experimental/ccl/`
unless explicitly qualified. Earlier AutoDebug and implementation notes were
read; their hardware-pending statements describe their original observation time.
This review intentionally makes no new hardware or performance claims.

## CPU rereview after the parent repair

2026-11-13. **Both reported admission findings are addressed in source.** Inside
the Linear-only scratch block, the parent now checks the scratch device pointer
and exact coordinate equality against input, then exact expected logical and
padded gathered shapes. Both checks precede descriptor construction. Existing
`validate_on_program_cache_hit` calls the same complete miss validator, so these
guards apply on cache hits as well. The additions do not affect the Ring branch.
No further fix is requested for these two findings.

The focused probe correctly constructs both M32/M1 logical mismatch directions
with the same physical padding. Its subset case uses
`ttnn.get_device_tensors(scratch)[0]`, checks its singleton coordinates and
preserved shape/dtype, and keeps its parent scratch live in the same scope. It
requires the corresponding dedicated native message; construction/API errors
cannot satisfy the rejection. The first phase contains the original nine plus
three new cases (12); after a synchronized, numerically checked successful fused
call, the three new cases repeat (15 total). Valid positive cases retain their
original shapes and gates. Probe Python syntax passes CPU `ast.parse`.

The second phase proves rejection in a **warm program-cache environment** once
executed; it does not by itself prove the invalid call took a cache-hit branch,
because changed tensor shapes/placement can change the hash. Cache-hit guard
coverage here is additionally supported by the direct validator delegation
above. Native build and actual host rejection/positive device results remain
the parent's pending validation; this rereview did not run them.

Rereviewed SHA256:

- AGMM device-operation validation:
  `35a33982c76abd945828891eea2cdc13cf389737caeeb47d2b50b7df41dafd0f`.
- Focused probe:
  `70f5f4c2b9d9105de41f66884e3e00abadd07329493a74b805bf1b7a7f1ee82e`.
  No probe or native source was edited by this reviewer.
