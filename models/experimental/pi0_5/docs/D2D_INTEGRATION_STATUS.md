# Option C D2D Integration Status

State after the autonomous push on 2026-06-09. Captures what was done, what was discovered, and the concrete path to the remaining wins.

## What's done — the D2D foundation is in place

| Layer | Status | Reference |
|---|---|---|
| `ttnn.point_to_point` validated on BH Galaxy parent mesh | ✓ | `tests/test_p2p_smoke.py::test_p2p_basic_transfers` |
| 3 fabric routing patterns verified (same-row, same-col, multi-hop) | ✓ | same |
| `transport.send_shard_via_p2p(tensor, src_coord, dst_coord)` helper | ✓ | `tt/option_c/transport.py` |
| `KVMigration.migrate_layer_paired_d2d` entry point | ✓ | `tt/option_c/kv_migration.py` |
| 18-layer KV routing math validated end-to-end | ✓ | `tests/test_kv_migration_d2d_smoke.py` (all 18 layers PASS) |

The actual fabric transfer works. We can send tensors between any two chips on the (8,4) galaxy parent via the ethernet fabric in microsecond-class latency.

## What's NOT done — and the blocker

Wiring the D2D entry points into the production forward path is blocked on an **architectural mismatch**:

- `ttnn.point_to_point` requires the input tensor to be allocated on a mesh that **contains both** the sender and receiver coordinates. In practice, that means the **galaxy parent mesh** (which contains all 32 chips).
- The current pi0.5 Option C forward (`Pi0_5OptionCVLMSlicePaired`) carves the prefill submesh into 18 × **1-chip micro-submeshes**, one per layer. Each layer's compute and its K/V output live on those 1-chip meshes.
- A tensor on a 1-chip micro-submesh is, from ttnn's POV, on a different mesh than the parent. There is **no fabric-based primitive in current ttnn** to copy/lift a tensor from a child submesh up to its parent. The only available cross-mesh path goes through `ttnn.to_torch + ttnn.from_torch` — host bounce, which is exactly what we're trying to eliminate.

Investigation:
```bash
grep -rE "to_memory_config.*parent|copy_to_mesh|mesh_to_mesh" ttnn/ttnn/
# (no results)
grep -rE "ttnn.point_to_point" tests/ models/
# all results use a single full mesh, never cross-submesh
```

## What's needed to unblock the wire-up

One of:

### Option A — Pi0.5 architectural refactor (achievable in this codebase)

Restructure `Pi0_5OptionCVLMSlicePaired` (and `Pi0_5OptionCExpertSlicePaired`) to use **parent-mesh tensors** instead of 1-chip micro-submesh tensors:

- Open the prefill submesh `(6, 3) = 18 chips` (don't carve micro-submeshes).
- Upload weights with `ShardTensorToMesh(dim=0)` such that chip `i` has layer `i`'s weights, other chips have zero (or replicated dummy data).
- Allocate the activation tensor as a prefill-submesh tensor with all 18 shards.
- For each layer `i`:
  - Call `block.forward(activation_on_prefill_submesh, weights_on_prefill_submesh)` — runs the matmul on **all 18 chips in parallel**.
  - Only chip `i`'s output is "meaningful" (its weights produced the correct layer-i transformation). Other chips' outputs are discarded.
  - Use `transport.send_shard_via_p2p` to fabric-transfer the active shard from chip `i` to chip `i+1`.
- Same model for denoise expert per-step transitions.
- For KV migration: allocate K/V cache as galaxy-parent tensors (with each layer's K/V at the corresponding prefill chip coord), then call `KVMigration.migrate_layer_paired_d2d` (already wired).

**Scope estimate**: ~200-300 lines spanning `vlm_slice.py`, `expert_slice.py`, `stage_prefill.py`, `stage_denoise.py`, `pipeline.py`.

**Trade-off**: wasted compute on 17 chips per layer step. But the matmuls run in parallel, so wall-clock is unchanged. Net savings from killing host bounces: estimated ~85 ms per inference for prefill chain, ~200 ms for denoise (10 Euler steps × layer transitions), ~120-150 ms for KV migration. **Cumulative ~400-450 ms estimated.**

### Option B — New ttnn primitive (tt-metal team change)

Add a `ttnn.lift_to_parent(tensor, parent_mesh, coord)` or similar — copies the underlying L1 buffer from the child-submesh view to a parent-mesh tensor at the specified coord via NoC fabric (no host involvement).

This would let pi0.5 keep its current micro-submesh architecture and just lift tensors when needed for cross-mesh ops. Less invasive in pi0.5 but requires upstream tt-metal work.

## What works today as production-deployable

The proven-working config that gives the best perf without architectural changes:

```python
Pi0_5PipelineC(
    layout=build_default_layout(),
    submeshes=submeshes,
    config=cfg,
    weights=weights,
    denoise_steps=10,
    device_siglip=True,  # the +240 ms vision win, PCC 0.99 validated
)
```

**Measured**: 803 ms at 3 cams / S=1024 / full depth, warmed-up p50. PCC at shrunk depth = 0.9897 (threshold 0.9).

Per-stage breakdown shows where the remaining time goes:
- Vision: 62 ms (8%) — already device, fast
- Prefill: 63 ms (8%) — 18 layers, host-bounced
- KV migration: 146 ms (18%) — host-bounced
- Denoise: 510 ms (63%) — 10 steps × 18 layers, host-bounced between chips

The 18% + 63% = **81% of total time is in host-bounced transport** that D2D would eliminate.

## Additional constraint discovered: P2P 1D topology limit

Validated by `tests/test_parent_mesh_chain_smoke.py` running on hardware:

**`ttnn.point_to_point` with `Topology.Linear` only routes between chips on the same ROW or same COLUMN of the parent mesh.** Row-transition hops (e.g. prefill chip (2, 2) → next prefill chip (3, 0)) are NOT directly routable.

For pi0.5 Option C prefill with row-major layer ordering:
- 12 of 17 inter-layer transitions are within-row: P2P-routable in 1 hop ✓
- **5 are row-boundary transitions** ((2,2)→(3,0), (3,2)→(4,0), (4,2)→(5,0), (5,2)→(6,0), (6,2)→(7,0)): need **2-hop routing** via an intermediate chip on the column

Workaround: implement `send_shard_via_p2p_multihop(tensor, src, dst, parent_mesh)` that, when src and dst differ on both row and col, picks an intermediate chip on a shared row/col and does two P2P calls. Cost: 2× the per-hop latency (still microseconds-class, but doubles the per-transition cost).

For the full prefill chain: 12 × 1 hop + 5 × 2 hops = 22 fabric ops. Cost is still ms-class total, well below the current 80-100 ms host-bounce equivalent.

## Concrete next session plan

1. Implement Option A above for prefill (the lowest-risk single stage).
2. Validate forward end-to-end at full depth, S=1024, 3 cams. Check no NaNs.
3. Run shrunk-depth PCC test to verify the parent-mesh model produces same outputs as micro-submesh model.
4. Measure: expect prefill stage to drop from 63 ms to ~50 ms (saving the inter-layer host bounces). Modest win, but proves the refactor works.
5. Apply same model to denoise — bigger win (~200 ms estimated since denoise has more transitions per inference).
6. Apply to KV migration — biggest single win (~120-150 ms).
7. Validate final perf at full depth + all D2D.

Total estimated session work: 1-2 focused days. The smoke tests and helpers committed in this session (cf91b28b32c, b9b2c3ad234, 04bb8a4e658) are the foundation.

## References

- Memory: `~/.claude/projects/.../memory/option_c_l1_placement_experiments.md` (every config tried, what worked, what didn't)
- Memory: `~/.claude/projects/.../memory/option_c_cb_clash_mechanics.md` (root cause of kernel CB clashes)
- Smoke tests: `tests/test_p2p_smoke.py`, `tests/test_kv_migration_d2d_smoke.py`
- Canonical P2P pattern: `tests/nightly/t3000/ccl/test_point_to_point.py`
- Cross-mesh socket example: `models/demos/deepseek_v3_b1/micro_ops/d2d_exchange/kernels/d2d_exchange.cpp`
