# MoE1D TP=8 (T3K) blocker: column-sharded `ttnn.sparse_matmul` hangs

**Status:** ✅ **RESOLVED (2026-06-19).** Root cause found and fixed in the `sparse_matmul` program
factory. All three probe cases now PASS in ~1.7s (was: `shard-perdevn`/`shard-fulln` hung forever), and
the end-to-end Gemma4 MoE block at `(1,8)` passes (`decode-1x8` + `prefill-128-1x8`, 2/2).

## Root cause & fix

The hang was **not** about sharding per se — it was about the **core-grid geometry the shard produces**.
With a column-sharded weight, per-device `N = I/num_devices = 256/8 = 32`, i.e. a single output tile, so
`_build_sparse_matmul_config` selects a **1×1 (single-core) grid**. (The replicated `N=256` case selects
an 8×1 grid, which is why it passed.)

On a single-core grid there are **no in0 mcast receivers**, so `in0_mcast_num_dests == 0` and
`in0_mcast_num_cores == 0`. The in0 sender kernel
(`reader_bmm_tile_layout_in0_sender_padding.cpp`) nonetheless ran the full mcast handshake: it issued a
`noc_async_write_multicast` to a degenerate (self-only) rectangle with 0 destinations and a semaphore
handshake against receivers that were never launched → permanent NOC/semaphore deadlock.

The **dense** `matmul_multicore_reuse_mcast_1d` factory already guards this exact case
(`if (in0_mcast_receiver_num_cores == 1) defines["SKIP_MCAST"] = "1";`). The **sparse** factory set
`SKIP_MCAST` only for the in1 sender, never for the in0 sender. The fix mirrors the dense guard:

```cpp
// ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/
//   sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp
if (in0_mcast_receiver_num_cores == 1) {
    mm_kernel_in0_sender_writer_defines["SKIP_MCAST"] = "1";
}
```

With `SKIP_MCAST` the single-core sender reads in0 into its local CB and `push_back`s it for the
co-located compute kernel without any multicast/semaphore handshake — the correct single-core behavior.

> This affects **any** TP=N sparse_matmul whose per-device `N` collapses to one output tile (≤32 wide),
> not just TP=8. Rebuild `ttnncpp` after applying. (On this box the freshly-built
> `build_Release/ttnn/_ttnncpp.so` had to be copied over `build_Release/lib/_ttnncpp.so`, which is what
> the python module RPATHs to — a local build-layout quirk, not part of the fix.)

---

## (Original investigation notes below — kept for history)

**Status:** CONFIRMED on two independent T3K boxes (2026-06-19). The "a clean box should PASS all
three" hypothesis is **disproven** — a second box, freshly `tt-smi -r` reset with healthy fabric init,
still hangs `shard-perdevn` while `replicate-fulln` passes and the bare `all_reduce` control passes
(6/6, 21.7s). The bug is in `ttnn.sparse_matmul`'s multi-device (sharded-weight) path and is
reproducible/universal.

> Repro hygiene: the box may start in the flaky ETH state (conftest *skips* all 3 with
> `ETH core ... heartbeat check failed`); `tt-smi -r` clears it. A timed-out hang leaves dispatch
> kernels running, which contaminates the next run (`dispatch kernels still running ... following a
> reset`) — **always `tt-smi -r` between hang repros.**
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

## Related upstream issue (#45943) — same deadlock shape, different trigger

`sparse_matmul` has a known mcast deadlock — [#45943](https://github.com/tenstorrent/tt-metal/issues/45943)
(CLOSED, P0) — *"deadlocks when declared `nnz` > actual `count_nonzero(sparsity)`."* The in0 sender loops
over `batchB` and multicasts only for non-zero entries while receiver/compute loop a fixed `nnz`, so an
overcounted `nnz` leaves receivers waiting on a semaphore the sender never re-sets. **It is not the cause
of this hang**, for two reasons: (1) the probe passes `nnz=None` (`get_batch_from_reader=true`, the
runtime-count path — not the baked-`nnz` path #45943 needs); (2) sparsity is replicated and identical
between the passing replicated-weight case and the hanging sharded case, so any `count_nonzero`/`nnz` bug
would fail both. What #45943 *does* tell us: the failure shape is a sender↔receiver mcast-handshake count
mismatch — so look at whether the runtime batch-count broadcast is wired correctly on the smaller N=32
sharded core grid. Its fix was validation-only ("fail loudly instead of hang") and does not touch the
sharded path. See `.claude/sparse_matmul_tp8_hang_handoff.md` §5.2 for the device-side detail.

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
