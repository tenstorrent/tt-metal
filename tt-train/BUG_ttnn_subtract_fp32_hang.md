# `ttnn::subtract` deadlocks on Blackhole TP setups with `FLOAT32` output, COL_B broadcast, and certain W-tile counts

## TL;DR

`ttnn::subtract(bf16_lhs, bf16_rhs, ttnn::DataType::FLOAT32)` deadlocks on a 1×4 Blackhole (P150_x4) mesh when the LHS is sharded across the innermost dim (TP), the RHS has `W=1` (COL_B broadcast), and `W_tiles_per_shard` ∈ {3, 5, 7, ≥8}. It does **not** hang for W ∈ {1, 2, 4, 6}, and it does not hang at all if any one of {FP32 output dtype override, COL_B broadcast, sharded LHS} is removed.

Hit in production via `ttml::ops::distributed::vocab_parallel_cross_entropy_loss` on TinyLlama (`V=32000`, `TP=4` → `V_per_shard=8000`, i.e. `W_tiles_per_shard=250`).

## Minimal reproducer

`tt-train/tests/ops/distributed/subtract_fp32_col_b_bcast_test.cpp` — four-test GoogleTest fixture, all four tests share the env var `REPRO_W_TILES_PER_SHARD` (default `2`, a passing value):

| Test | LHS layout | RHS | dtype | Hangs at W=3? |
|---|---|---|---|---|
| `ColBBroadcast_DefaultDtype_NoHang` | replicated | W=1 | BF16 out | **NO** |
| `NoBroadcast_Fp32Output_NoHang` | replicated | full | FP32 out | **NO** |
| `ColBBroadcast_Fp32Output_ReplicatedLhs_NoHang` | replicated | W=1 | FP32 out | **NO** |
| `ColBBroadcast_Fp32Output_ShardedLhs_Hangs` | **sharded dim=3** | W=1 | FP32 out | **YES** |

Reproducer command (after `tt-smi -r`):

```bash
cd tt-train
REPRO_W_TILES_PER_SHARD=3 \
TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_x4_mesh_graph_descriptor.textproto \
timeout 60 ../build_Release/tt-train/tests/ttml_tests \
    --gtest_filter='SubtractFp32ColBBcastTest.*'
# Or, equivalently: ./probe.sh 3
```

Expected: first 3 tests `[ OK ]`, last test prints `dispatch returned (W_tiles=3)` and `sync BEFORE after subtract` then hangs until the `timeout` kills it (exit 124).

## Required conditions (all of these must hold)

1. Inputs are `BFLOAT16`.
2. Output dtype override to `FLOAT32` (this triggers `binary_ng`'s auto-injected `TYPECAST(BF16, FP32)` in the post-activations chain).
3. RHS has innermost dim = 1 → `COL_B` broadcast.
4. LHS is **sharded** across the mesh on the innermost dim (dim 3 in `[B, 1, S, V_total]`).
5. `W_tiles_per_shard` ∈ {3, 5, 7, ≥8}.

Drop any one of (2)–(5) and the hang disappears.

## W sweep data

1×4 Blackhole P150_x4 mesh, `B=5, S=256`, BF16 inputs, FP32 out, RHS=`[B, 1, S, 1]` replicated, LHS=`[B, 1, S, W*32*4]` sharded on dim=3:

| W | V_per_shard | Result |
|---|---|---|
| 1 | 32 | PASS |
| 2 | 64 | PASS |
| **3** | **96** | **HANG** |
| 4 | 128 | PASS |
| **5** | **160** | **HANG** |
| 6 | 192 | PASS |
| **7** | **224** | **HANG** |
| **8** | **256** | **HANG** |
| 256 | 8192 | HANG (production-shaped) |

Each result reproduced with a fresh `tt-smi -r` between probes. The hang is **non-monotonic in W** in the 3..7 range — strong evidence of a tile-chunk / DST-register boundary effect rather than a pure size threshold.

## Diagnostic signature

Host dispatch *returns* successfully. The subsequent `tt::tt_metal::distributed::Synchronize` then never completes. Repro log around the hang:

```
[repro] sync BEFORE before subtract (FP32 out, COL_B bcast, sharded)
[repro] sync AFTER  before subtract (FP32 out, COL_B bcast, sharded)   ← upload sync OK
[repro] dispatch returned (W_tiles=3)                                   ← op enqueued
[repro] sync BEFORE after  subtract (FP32 out, COL_B bcast, sharded)
                                                                        ← hang here forever
```

After the hang the cluster is wedged: the next process opening these chips will block on the **upload** sync rather than the post-op sync. `tt-smi -r` recovers.

## Suspected location

`ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp`

When the user requests FP32 output for BF16 inputs, `binary_ng` auto-injects a `TYPECAST(BF16, FP32)` op into the post-activations chain (per the existing comment in `tt-train/sources/ttml/ops/distributed/losses.cpp`). The hang requires this typecast injection (condition 2), the COL_B broadcast path (condition 3), AND sharded LHS (condition 4). With those three combined, certain `W_tiles_per_shard` values trip an inner-loop / DST-register issue that the simpler kernels avoid.

The non-monotonicity (W=3,5,7 hang; W=2,4,6 pass) is consistent with a code-generation bug in how the per-chunk tile count is computed when FP32 output forces a smaller DST budget (FP32 tiles take 2× the DST space of BF16 tiles), combined with a partial-chunk handling path that's only valid for some remainder sizes.

## Workarounds

In `vocab_parallel_cross_entropy_loss`:

- **Recommended**: drop the explicit FP32 output dtype: `ttnn::subtract(logits, global_max)` instead of `ttnn::subtract(logits, global_max, ttnn::DataType::FLOAT32)`. Keeps the op in BF16, no typecast injection, no hang. Verified end-to-end on TinyLlama training.
- Alternatively, materialize RHS to full W (eliminates the COL_B broadcast) before subtracting.

## Environment

- Hardware: Tenstorrent Blackhole, P150_x4 (4 chips, arranged as 1×4 via locally-modified `p150_x4_mesh_graph_descriptor.textproto` — default is 2×2).
- KMD: 2.7.0
- Firmware bundle: 19.6.99
- tt-metal: `bklockiewicz` local branch
- Test mesh shape: `MeshShape(1, 4)`, fabric config `FABRIC_2D`

## Files

- Reproducer: `tt-train/tests/ops/distributed/subtract_fp32_col_b_bcast_test.cpp`
- Probe driver: `tt-train/probe.sh`
- Production site: `tt-train/sources/ttml/ops/distributed/losses.cpp` → `vocab_parallel_cross_entropy_loss`
- MGD used: `tt_metal/fabric/mesh_graph_descriptors/p150_x4_mesh_graph_descriptor.textproto` (locally edited from `[2,2]` to `[1,4]` device_topology)
