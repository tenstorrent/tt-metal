# WIP handoff: single snake gather for the TP-deduped KVPE prefix

Branch `ipotkonjak/snake-tp-kv-gather-wip`, off `ipotkonjak/tp-stack-work` at `2de2a24ed6a`.

| commit | contents |
|---|---|
| `afaa0d95bea` | the substantive change: single full-mesh gather + guard, and the preload topology fix |
| `3d25166f500` | **drop before PR** — the `_snake_topo` probes that localised the corruption |

## Goal

Replace the two-stage `TP-inner -> SP-outer all_gather_async` in
`MlaAttention._gather_kvpe_prefix_tp_sharded` with **one 32-rank snake-ring
`high_bw_all_gather`** (`cluster_axis=None`), for the GLM-5.2 SP x TP KV-deduped path.

`mla.py` names this as follow-up work in its own comment:

> `high_bw_all_gather` rides ONE cluster axis and lands in the single preallocated worst-case
> scratch, so it cannot rebuild an sp*tp-striped slab ... teaching the high-BW gather sp*tp stripes
> is follow-up work.

Both objections dissolve with the snake: `cluster_axis=None` removes the one-axis limit, and with a
single gather there is no TP-stage intermediate to find room for. The scratch already spans the
whole global sequence.

## Why one gather is substitutable (the correctness argument)

A TP-deduped cache is dim-2 sharded across **both** mesh axes. Per `kv_cache_utils.py` (commit
`768edf5ae97`):

> A cache sharded on dim 2 across BOTH mesh axes has the same declared distribution whether it got
> there via full_mesh (one snake ring) or via tp_axis (SP x TP dedup): **row-major over the mesh IS
> the sp*tp linearization.**

- two-stage output order = TP-inner then SP-outer = sp-major / tp-minor
- snake output order = `transport_rank -> tensor_rank` = mesh row-major

With `sp_axis=0, tp_axis=1` these are the same order. That layout is already asserted in this path
(`mla.py`: *"sparse_mla assumes sp_axis=0 (outer), tp_axis=1"*) and in `kv_cache_utils`.

Verified arithmetic on 8x4: local `[78, 1, 1760, 576]`, `1760 * 32 = 56320` = global sequence.

## What is done

### 1. The gather (`mla.py`)

- `_gather_kvpe_prefix_full_mesh` — one `high_bw_all_gather(cluster_axis=None, dim=2,
  output_tensor=self._sparse_kv_gather_buffer, input_batch_index=slot_lo, gathered_dim_size=...)`.
  No intermediate; result lands in the persistent scratch instead of being transient.
- `_can_full_mesh_gather_kvpe(cache_storage)` — three preconditions, each mirroring a hard
  `TT_FATAL` in the op so the guard **degrades to the two-stage route instead of crashing**:
  1. geometry: 2-D mesh, both axes >= 2, one even dimension, `sp_axis=0/tp_axis=1`
  2. declared dim-2 shard factor spans `sp*tp` (mirrors `tensor_dim_shard_factor`)
  3. input and output share a `MeshDevice` handle

### 2. The preload topology fix (`test_prefill_transformer_chunked.py`) — **root cause, solved**

`_preload_kvpe_prefix_from_trace` builds its host cache folded as `[layers, tp, seq/tp, D]` and maps
it with `ShardTensor2dMesh(dims=[2, 1])` — TP declared as a **dim-1** split. Then
`copy_host_to_device_tensor` propagates that declaration onto the persistent cache, overwriting the
`[Shard(2), Shard(2)]` that `init_kvpe_cache` stamps at creation.

The bytes were always correct (`_to_tp_stripe_major` lays them in linear chip order `L = s*tp + t`,
i.e. mesh row-major). Only the metadata lied. Fix restores the real distribution after the copy,
matching the idiom two functions away in `init_kvpe_cache`:

> DRAMZeroFill is an in-place generic op whose output follows the allocator's default replicated
> topology, so stamp the intended distribution after the fill.

Measured, via the probes in `3d25166f500`:

```
                 before fix              after fix
after_create     [Shard(2), Shard(2)]    [Shard(2), Shard(2)]
before_preload   [Shard(2), Shard(2)]    [Shard(2), Shard(2)]
after_preload    [Shard(2), Shard(1)]    [Shard(2), Shard(2)]   <- the single corrupting step
before_update_B  [Shard(2), Shard(1)]    [Shard(2), Shard(2)]
at_gather_entry  [Shard(2), Shard(1)]    [Shard(2), Shard(2)]
```

Note `update_padded_kv_cache` is **not** at fault — it defines `compute_output_topologies` precisely
to preserve the cache's distribution, and was faithfully preserving an already-corrupted one.

## The remaining blocker

With the declaration fixed the snake path activates and fails on:

```
TT_FATAL @ high_bw_all_gather_device_operation.cpp:322: output_tensor.device() == mesh_device
info: high_bw_all_gather input and output tensors must be on the same mesh device
  in high_bw_all_gather_build_operation_args(...)   [mesh_device = input_tensor.device()]
```

A bare `MeshDevice*` comparison. Established facts:

- **It is not a sharding requirement.** The op's only output constraints are
  `storage_type() == DEVICE` and `buffer() != nullptr` (lines 150, 151, 235). A replicated scratch is
  fine — the SP-only path passes this same buffer through the same check successfully.
- **`TT_CCL` is exonerated by inspection.** `__init__` only does `self.mesh_device = mesh_device`;
  no reshape, no submesh. `get_tt_ccl` caches by `mesh_device.id()`. If a stale instance were being
  returned, the SP-only path would fail this same check on its first gather, and it does not.
- The two tensors are allocated by **different routes**, which is the prime suspect:

  | tensor | route |
  |---|---|
  | persistent cache | `ttnn.allocate_tensor_on_device(..., mesh_device, ...)` |
  | gather scratch | `ttnn.from_torch(..., device=mesh_device, mesh_mapper=ReplicateTensorToMesh(...))` |

### Next step

Run the minimal probe (already written, needs a healthy board):
`scratchpad/mesh_probe.py` opens an 8x4 mesh, allocates one tensor each way, and prints
`.id()`, `id()`, and the `==` result. It takes ~1 min and needs no model weights.

- `ccl.mesh_device is md` **True** but `cache.device() == buf.device()` **False**
  -> the `from_torch` route is the bug; allocate the scratch with `allocate_tensor_on_device`
  (in `TT_CCL.get_mla_sparse_kv_gather_buffer`).
- `ccl.mesh_device is md` **False** -> two handles exist in the process; `get_tt_ccl`'s `id()` keying
  is implicated.

Changing that allocation is worth doing regardless: the scratch is fully overwritten every layer, so
`torch.zeros(1, 1, 56320, 576)` (~62 MB) on the host plus a 32-way H2D broadcast is wasted
construction work for bytes nobody reads.

## Validation state

Run against the merged TP+snake tree on a 32-device Blackhole Galaxy 8x4, `FABRIC_2D_TORUS_XY`.

| | result |
|---|---|
| build, `-Werror` | clean |
| GLM TP-sharded accuracy, two-stage baseline | KV min PCC 0.986807, indexer 0.998237, output 0.918142 |
| GLM TP-sharded perf, two-stage baseline | 1.782 s median/chunk (flat across all 11), 19.701 s/iter, ~2,859 tok/s |
| same, with this branch's code (guard falls back) | **bit-identical**: 0.986807 / 0.918142 |
| snake path actually engaged | blocked by the mesh-handle check above |

So the change is currently a proven no-op, safe to carry, and self-enabling once the handle issue is
fixed. The per-chunk table is in `scratchpad/glm_twostage_baseline/`.

**No perf comparison exists yet for the snake path** — the two-stage numbers above are the baseline
to beat. Worth noting the direction is not obvious: at matched rows-per-device the snake beat the
8-rank axis ring by ~5 GB/s in op-level tests, but at matched global volume (which is this case's
shape, 1/32 the rows per device) it measured lower. A regression here would not be surprising.

## Board state at handoff

The Galaxy is **wedged and not enumerating** (`tt-smi -ls` returns 0 boards; all 32
`/dev/tenstorrent` nodes exist). `tt-smi -glx_reset` cannot recover it from a non-interactive
context because the IPMI step needs `sudo` and there is no TTY:

```
sudo: a terminal is required to read the password
Error: POST_RESET failed for device 0.
```

Earlier resets in the session worked off a cached sudo credential that has since expired.
**`tt-smi` exits 0 even when `POST_RESET` fails**, so check `tt-smi -ls` for 32 boards rather than
the exit code. Recovery needs an interactive `tt-smi -glx_reset` (or `sudo -v` first).

## Related: review finding F1

F1 (*"full-mesh output declares a replicated topology while the data is sharded"*) was dropped from
the snake PR on the grounds that the framework derives output topology from inputs. That holds for
the axis path. Its full-mesh half is what blocked this work: `all_gather_async` is only
bounds-checked, so an under-declared distribution goes unnoticed, while `high_bw_all_gather`'s
full-mesh path validates strictly and refuses. Worth revising F1's disposition rather than leaving it
recorded as dismissed.
