# MoE1D TP=8 (T3K) blocker: column-sharded `ttnn.sparse_matmul` hangs

**Status:** confirmed repro on one T3K box; needs confirmation on a clean/independent box.
**Owner context:** came out of benchmarking `MoE1D` (`models/common/modules/moe/moe_1d.py`) against the
Gemma4 reference `MoEBlock` (`models/demos/gemma4/tt/moe.py`).

## TL;DR

A single `ttnn.sparse_matmul` whose **weight is column-sharded across a (1,8) mesh hangs** (no progress,
host-side stall, devices idle). The **same op with the weight replicated passes in ~3s.** This sits
*below* MoE1D — it equally hangs the Gemma4 reference, which shards gate/up/down the same way — so it is
not a MoE1D bug. It blocks every TP=8 MoE forward.

Everything else on the box is healthy: fabric inits on all 8 devices, and a bare `ttnn.all_reduce`
(both `Ring` and `Linear`, at the MoE shapes) passes. So this is **not** flaky ethernet and **not** the
collective — it is `sparse_matmul` with a sharded weight at TP=8.

## Minimal repro

```bash
# T3K (8-chip 1x8). Routes through the device; needs a working ttnn build.
MESH_DEVICE=T3K pytest models/common/tests/modules/moe/test_sparse_matmul_probe.py -rA -s
```

`test_sparse_matmul_probe.py` runs ONE `sparse_matmul` (E=8, H=256, I=256, top-k=2) three ways, ordered
so the expected-pass controls report before the suspect:

| case id            | gate weight            | program-config n      | observed on the repro box |
|--------------------|------------------------|-----------------------|---------------------------|
| `replicate-fulln`  | replicated (256-wide)  | 256                   | **PASS (~3s)**            |
| `shard-perdevn`    | col-sharded, 32/device | 32 (per-device, correct) | **HANGS**              |
| `shard-fulln`      | col-sharded, 32/device | 256                   | not reached (prior hang)  |

The key data point is **`shard-perdevn`: it hangs even with the *correct* per-device program config**
— the exact configuration the Gemma4 reference uses. So the program config is NOT the issue; the
sharded weight is. (An earlier hypothesis — that MoE1D mis-sizes the program config from the full
intermediate instead of the per-device shard — is **disproven** by this case.)

A clean box should make all three cases PASS. If `shard-perdevn` hangs there too, the bug is in
`ttnn.sparse_matmul`'s multi-device (sharded-weight) path and is reproducible/universal.

## Evidence chain (how we localized it)

1. `MoE1D` and Gemma4 `MoEBlock` both hang in the `(1,8)` forward, right after the sparsity
   `to_layout(ROW_MAJOR)` prep — i.e. at the first sharded `sparse_matmul`. (Run
   `test_moe_1d_vs_gemma4_perf.py -k 1x8`; isolate either module with `MOE_PERF_MODULES=moe1d|gemma4`.)
2. Bare `ttnn.all_reduce` on `(1,8)` PASSES — `Ring` & `Linear`, shapes `[1,1,32,4096]`,
   `[1,1,32,256]`, `[1,1,1,256]`. So fabric + collective + ring are fine.
   (`test_allreduce_probe.py`.)
3. Direct `sparse_matmul` probe: replicated weight PASSES, col-sharded weight HANGS (above).
4. `TT_METAL_WATCHER=10` is NOT usable to debug this: it throws
   `idle_erisc.elf: segment ... overflows region limit of 0x54c0 bytes` (watcher instrumentation
   overflows the ETH `cq_dispatch` kernel) before the hang. Disable watcher, or use a build with a
   larger erisc region, when chasing this.

## Files in this change

- `test_sparse_matmul_probe.py`   — **minimal repro** (the thing to run on the clean box).
- `test_allreduce_probe.py`       — control showing collectives/ring are healthy.
- `test_moe_1d_vs_gemma4_perf.py` — the end-to-end MoE1D-vs-Gemma4 perf benchmark. `(1,1)`/`(1,8)`
  parametrized; per-module env gate `MOE_PERF_MODULES=both|moe1d|gemma4`. The `(1,8)` cases are blocked
  by this hang; the `(1,1)` cases pass and show parity (below).

## Repro-box environment

- Host `wh-lb-43-special-...-for-reservation-86946`; `ttnn.get_num_devices()==8`,
  `get_num_pcie_devices()==4` (4 PCIe + 4 eth-remote), arch `wormhole_b0`.
- Mesh `(1,8)`, fabric `FABRIC_1D_RING` (auto-set by `models/common/tests/conftest.py::ttnn_mesh_device`
  for multi-device shapes).
- This box's fabric *init* was intermittently flaky (router-sync timeout on Device 4 / eth `e0-6`); a
  device reset cleared init, after which init was clean but the sharded-`sparse_matmul` hang remained
  deterministic. Worth re-confirming on a box without that init flakiness to rule the box out entirely.

## Context: the `(1,1)` result this was all for (unaffected by the hang)

`test_moe_1d_vs_gemma4_perf.py` at `(1,1)`, matched config (E=8/H=256/I=128/top_k=2, bf8 experts, bf16
router+sparse_matmul, GeGLU, softmax→topk→sum-norm), trace-captured median per forward:

| case               | MoE1D   | Gemma4 ref | ratio  |
|--------------------|---------|------------|--------|
| N150 decode        | 366.7us | 363.4us    | 1.009x |
| N150 prefill-128   | 881.6us | 880.7us    | 1.001x |
| N300 decode        | 367.9us | 364.0us    | 1.011x |
| N300 prefill-128   | 880.2us | 880.1us    | 1.000x |

→ MoE1D is at parity with the reference on a single chip. The `(1,8)` head-to-head (incl. the TP
all-reduce) is what the `sparse_matmul` hang is blocking.

## How to run everything (once the op is fixed)

```bash
# the blocker repro
MESH_DEVICE=T3K pytest models/common/tests/modules/moe/test_sparse_matmul_probe.py -rA -s
# collective control (should pass regardless)
MESH_DEVICE=T3K pytest models/common/tests/modules/moe/test_allreduce_probe.py -rA -s
# end-to-end MoE1D-vs-Gemma4 perf, TP=8
MESH_DEVICE=T3K pytest "models/common/tests/modules/moe/test_moe_1d_vs_gemma4_perf.py" -rA -s -k 1x8
# isolate one module (debug):
MOE_PERF_MODULES=gemma4 MESH_DEVICE=T3K pytest ... -k "decode and 1x8"
```

The `test_allreduce_probe.py` and `test_sparse_matmul_probe.py` files are scratch diagnostics — delete
them once the `sparse_matmul` issue is resolved.
