# Opt-in native sorted64 sampling patch (CPU-only implementation)

Patch: `presorted64.patch`, relative to the live checkout. The live sampling tree is unchanged; the baseline copy and `live_hashes.json` preserve the input source. No build, install, server, plugin, or device operation was performed.

The public/primitive/nanobind keyword is `input_sorted=False`. The default sorter, writer scan, random threshold construction, and softmax helpers retain their original operations. The new bool is an operation attribute and a compile-time argument in all three kernels, so sorted and unsorted program cache entries differ.

Opted-in inputs are BF16 TILE `[1,1,H,64]` values, standard 32x32 tiles, descending per row; identically permuted UINT32/INT32 ROW_MAJOR IDs with the same shape; k/p/temp `[H]`, H=1..32. WH/BH accept UINT32 IDs; existing other-architecture INT32 restrictions remain. k must be 1..64 (a device ASSERT checks this); sortedness is a caller precondition. Output stays `[1,1,1,H]` and supports the existing preallocated destination.

Sorted mode generates the unchanged 24-bit random threshold first, copies two value tiles into the existing values buffer, applies a two-tile top-k mask, and runs the existing temperature/MAX/exp/SUM/reciprocal/normalization helpers across both tiles. The writer scans candidate ordinals using explicit face/tile mapping and selects directly from paired full-width integer IDs. It preserves the crossing token for strict cumulative `> p` and the original first-candidate numerical fallback. No host read, generic sampler, or additional RNG owner is introduced.

All sorter/index descriptors remain allocated and bound to preserve generated symbols and avoid unrelated descriptor changes. Both endpoints execute zero generated-index/output-index traffic in sorted mode; writer does not wait/read/pop the inactive output index stream. Mask/probability streams produce/consume two tiles; final ID sticks produce/consume H rows. The values producer handle lives until after softmax has consumed it. In-place helpers retain pop-before-reserve ordering.

Static validation:

```bash
python3 /tmp/qwen-stage10-presorted-native/check_static.py
git apply --check --whitespace=error /tmp/qwen-stage10-presorted-native/presorted64.patch
```

`static_checks.json` records 2,048 unique/invertible tile coordinates and 43,008 CPU scan cases covering all 32 physical rows, every k=1..64, p=0/.95/1, and seven thresholds. These validate addressing and scan arithmetic against a flat CPU formulation, not execution of the C++ kernel. Default writer body, sorter call, compute helpers, and threshold construction are checked against the baseline. Source hashes prove the live sampling tree is unchanged. clang-format was applied using the checkout style.

Required next validation: native rebuild/JIT, descriptor acceptance (including dormant streams), eager default32 regression and sorted64 H=1/32 high-ID fixtures, top-p distribution/tile boundaries/actual k40 tail support, changing persistent k/p/temp, repeated completed traces, and seeded per-request lifecycle/integration. Existing default branches are source-preserved; latency or cross-family bitwise behavior is not claimed. The parent's separate common/Qwen patches prepare and select the wide candidates and preserve request RNG ownership.
