# Compute-kernel API ordering: reviewed true positives

These are the true positives from the bounded corpus run in
`compute_api_order_corpus.md` (40 model kernels, both architectures). Each was
reviewed against the source; the report keeps the line-level evidence.

## 1. `copy_init` followed by `mul_binary_tile_init`, then `copy_tile`

**Files**

- `models/demos/blackhole/qwen38_flash_next/ttnn/fused/gdn_rows_scan/kernels/compute.cpp`
  (`recurrence`, MATH, kernel call sites 565)
- `models/demos/blackhole/qwen38_flash_next/ttnn/fused/gdn_step/kernels/compute.cpp`
  (MATH, 557)
- `models/demos/blackhole/qwen38_flash_next/ttnn/fused/router_tail/kernels/compute_router_tail.cpp`
  (MATH, 181 and 212)

**Pattern**

```cpp
copy_init(src);            // seats OperationFpuEltwiseUnaryDatacopy
mul_binary_tile_init();    // seats OperationFpuEltwiseBinary instead
...
copy_tile(src, t, 0);      // llk_math_eltwise_unary_datacopy for the unseated op
```

The FPU holds one seated operation. `mul_binary_tile_init` replaces the datacopy
init, so the later `copy_tile` runs the datacopy operation without its init. The
LLK sanitizer reports the same condition as "execute was called for an Operation
the Exu is not initialized for".

**Fix**

Re-run `copy_init(src)` after `mul_binary_tile_init()` (or move the copy loop
before the binary init, or use the binary op's destination-reuse path). The same
pattern should be checked in the four sites above; the checker reports the
kernel call line, the exact `copy_tile` line is in the inlined body.

## 2. `llk_unpack_A` acc-to-dest mismatch

**Files** — the same four sites as above (UNPACK).

The unpacker is initialized with `acc_to_dest=1` by a destination-reuse init and
then `llk_unpack_A` runs with `acc_to_dest=0`. Re-initialize the unpacker for the
copy (`copy_init`) after the destination-reuse init, or keep the unpack
configuration consistent between the init and the copy.

## 3. Missing `unary_bcast_uninit`

**Files**

- `gdn_pre_rows/kernels/compute_qkv.cpp:352,370`
- `gdn_step/kernels/compute.cpp:554`
- `gr_write/kernels/compute.cpp:43`
- `ple/kernels/gate_compute.cpp:156`
- `shared_expert/kernels/compute_shared_eltwise.cpp:103`

`unary_bcast_init` arms unpack/math state that the operation group must
uninitialize before switching to another op group. The kernels never call
`unary_bcast_uninit`; this is the curated "broadcast requires an uninit" rule
from `compute_api_order_rules.yaml`. Add the matching uninit at the end of the
broadcast scope.

## Triaged false positives

- All 302 `loop-invariant-init` warnings: the init is intentionally re-run after
  a reconfiguration or to restore per-iteration SFPU state; hoisting is not
  valid in the general case. The rule is warning-only.
- The `tilize_16x32_kernel.cpp` `missing-init`/`missing-uninit` findings: Blackhole
  `fast_tilize_*` routes through `tilize_*` in a runtime-selected branch; the
  checker's branch-correlation model does not yet follow that fallback. The
  kernel does call `fast_tilize_uninit`.
