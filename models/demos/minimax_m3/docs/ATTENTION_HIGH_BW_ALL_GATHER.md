# MiniMax-M3 prefill: MSA attention gathers on `high_bw_all_gather`

Findings from the 2026-09-07 investigation of the "new attention all-gather"
(`ttnn.experimental.high_bw_all_gather`, Pavle Josipović, #51134 / #52606 / #53368 — the collective
DeepSeek / Kimi / GLM sparse MLA moved to) for MiniMax-M3 chunked prefill on a single Blackhole galaxy
(8x4, SP=8 rows x TP=4 cols, EP=32). Everything below was measured on one box; treat absolute numbers
as that box's, the relative ones as the finding.

## What changed in the model

The MSA (sparse) layers gather the SP-sharded K / V / index_k of the accumulated context across the 8 SP
rows before the indexer + `sparse_sdpa_msa`. Before this change the cache-read path
([attention/prefill.py](../tt/attention/prefill.py)) converted the **whole** packed ND-sharded KV cache
(all `num_users*num_layers` slots, x3 tensors) from NdShard to DRAM-interleaved on **every** sparse
layer, sliced out one slot, ran `all_gather_async`, and typecast the bf8 result to bf16 — the
`cache_read/deshard` hypothesis in [tests/perf/README_profiling.md](../tests/perf/README_profiling.md).

Now ([attention/msa.py](../tt/attention/msa.py) `msa_sp_attention_cache_read`) each of K / V / index_k
is gathered **straight out of its cache slot** by `high_bw_all_gather`:

* `input_batch_index=slot` addresses the (user, layer) slot in-op — no de-shard, no slice, no copy;
* `gathered_dim_size = n_rows*sp` moves only the written prefix;
* the output is a persistent worst-case `[1, 1, max_seq_len, hd]` buffer
  (`CCLManager.get_high_bw_gather_buffer`), rank `r` landing at the fixed slot `r*seq_local`.

That fixed-slot layout is exactly the block-cyclic layout the consumers already decode in-kernel
(stride `T/sp`), so the only consumer-side change is bounding the indexer and top-k to the written
prefix (`kv_len` / `valid_length`). The consumers now read bf8 K/V directly instead of a bf16 copy,
which also halves `sparse_sdpa_msa`'s K/V traffic. The first (no-cache) chunk uses the same op on its
activations. The old path was removed with this change; the unit test below keeps it as the reference.

Numerics are unchanged: PCC vs the previous path is 1.0 in
[tests/unit/test_msa_sp_cache_read_vs_ref.py](../tests/unit/test_msa_sp_cache_read_vs_ref.py)
(bf16 and bf8 caches, multi-slot, capacity > prefix), and the 60-layer golden KV check moved from
K 0.96703 / V 0.88667 / index_k 0.97851 (baseline) to 0.96725 / 0.88743 / 0.97862.

## Knobs

| knob | where | meaning |
| --- | --- | --- |
| `M3_FABRIC=<FabricConfig>` | `tests/galaxy_prefill_kv_pcc.py`, `scripts/run_prefill_perf.sh` | fabric config (default `FABRIC_1D`, same as the production runner) |
| `FABRIC=1d\|1d_ring\|2d\|2d_torus_xy` | `scripts/run_prefill_profile.sh` (`PROFILE_FABRIC` in `tests/perf/profile_prefill.py`) | fabric config for the zone profiler (default `1d`) |
| `M3_CCL_TOPOLOGY=Linear\|Ring` | both harnesses | topology handed to the legacy CCLs (default `Linear`) |
| `TT_MESH_GRAPH_DESC_PATH` | scripts | the scripts pick `single_bh_galaxy_torus_xy_graph_descriptor.textproto` (declares both wraps) for ring / torus fabrics and the plain mesh descriptor otherwise |
| `PREFILL_FABRIC_MODE=1d_ring` | `models/demos/common/prefill/runners/runner_utils.py` | the shared production prefill runner's fabric (default `1d` for sp<=8) |

## Fabric config vs. topology (the two "rings")

* **Fabric config** (`ttnn.set_fabric_config`, process-wide) programs the Ethernet routers.
  `FABRIC_1D` = open line per axis; `FABRIC_1D_RING` = the axis may close on its wrap cable; `FABRIC_2D`
  = routers can turn corners (needed for cross-mesh D2D pipelining); `FABRIC_2D_TORUS_{X,Y,XY}` = 2D plus
  wraps. This is capability: it makes the wrap link available, it does not make any op use it.
* **Topology** (`ttnn.Topology.Linear|Ring`) is a per-op argument of the legacy CCLs
  (`all_gather_async`, `reduce_scatter_minimal_async`, via `CCLManager`; M3's `ring_joint_sdpa` calls
  hardcode Linear). It selects the algorithm. Ring on an unwrapped fabric hangs; Linear on a wrapped
  fabric simply ignores the wrap.
* `high_bw_all_gather` takes **no** topology argument: it reads the fabric config and checks the closing
  link is wired, then picks its ring or line schedule itself. It only rings on the **full 8x4 mesh**; on
  an 8x1 submesh the wrap check fails (and on this box the submesh run also returned wrong data and hung
  the mesh close — full-mesh usage is fine).

Deployed configuration after this change: unchanged fabric, i.e. `FABRIC_1D` + plain mesh descriptor and
Linear legacy CCLs (row 3 below) — the production runner's default and what CI runs. The ring fabric
(row 6) is opt-in (`M3_FABRIC=FABRIC_1D_RING` / `FABRIC=1d_ring`, which also selects the torus_xy
descriptor); it is within noise of row 3 end-to-end and needs the wrap cables declared on the box.

## Results

### Op level (real M3 shapes: one 60-slot bf8 cache, K+V+index_k, 8 ranks, 2 links, device-profiler)

| fabric / legacy topology | cache | legacy total (de-shard+slice+AG+cast) | legacy AG only | new `high_bw` | total speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| 1D / Linear | 30k | 1.33 ms | 0.61 | 0.58 | 2.3x |
| 1D / Linear | 55k | 2.36 ms | 0.88 | 0.82 | 2.9x |
| 1D ring / Ring | 30k | 1.15 ms | 0.49 | 0.45 | 2.5x |
| 1D ring / Ring | 55k | 2.11 ms | 0.64 | 0.58 | 3.6x |
| 2D torus XY / Ring | 30k | 1.27 ms | 0.50 | 0.47 | 2.7x |
| 2D torus XY / Ring | 55k | 2.14 ms | 0.64 | 0.60 | 3.6x |

At these payloads the two gather ops are the same speed; the win is dropping the de-shard. Ring adds
another 1.3-1.4x on the gather itself. Large payloads (512K rows/device, full mesh, wall-clock):
1D line 12.4 ms, 1D ring 6.5 ms, 2D torus XY 7.2 ms.

### Model level (60 layers, 3 timed iterations, whole-sequence tok/s; last column = wall time of one 5k chunk attending 55k)

| # | gather | fabric | legacy-CCL topology | 5k cold | 5k @ 25k | 5k @ 55k | last chunk @ 55k |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | legacy (main) | 1D | Linear | 6173 | 5718 | 5688 | 902 ms |
| 2 | legacy (main) | 1D ring | Ring | 6114 | 5668 | 5633 | 903 ms |
| 3 | new AG | 1D | Linear | 6153 | 6106 | 6134 | 848 ms |
| 4 | new AG | 1D ring | Ring | 6016 | 6004 | 6019 | 877 ms |
| 5 | new AG | 2D torus XY | Ring | 5612 | 5558 | 5549 | 929 ms |
| 6 | **new AG** | **1D ring** | **Linear** | **6250** | **6156** | 6079 | **836 ms** |

Row 1 is production before this change; row 3 is production after (row 6 is the opt-in ring fabric).
Golden KV PCC was identical across every fabric.

### Device kernel time only (ms per layer, 6-layer zone profile, 5k @ 25k, new AG unless noted)

| zone | legacy 1D Lin | new 1D Lin | new 1D Ring (row 4) | new 1D ring + Linear (row 6) | new 2D torus (row 5) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **dense layer total** | 4.005 | 4.012 | 3.763 | 4.010 | 3.855 |
| norm all-gathers (2x) | 0.416 | 0.421 | 0.247 | 0.424 | 0.290 |
| reduce-scatters (attn + mlp) | 0.463 | 0.459 | 0.383 | 0.462 | 0.420 |
| **sparse layer total** | 8.756 | 7.723 | 7.475 | 7.697 | 7.612 |
| attn total | 3.179 | 2.140 | 1.964 | 1.998 | 1.987 |
| sparse_sdpa | 1.691 | 0.959 | 0.959 | 0.954 | 0.957 |
| ag_kv + ag_index_k | 0.591 | 0.394 | 0.288 | 0.288 | 0.303 |
| cache_read (de-shard) | 0.089 (6 slots; ~10x at 60) | 0 | 0 | 0 | 0 |
| mlp total | 4.962 | 4.961 | 5.008 | 5.073 | 5.128 |
| combine | 2.245 | 2.256 | 2.207 | 2.206 | 2.398 |
| dispatch | 0.935 | 0.936 | 1.107 | 1.082 | 0.988 |
| moe_reduce | 1.911 | 1.917 | 1.945 | 1.989 | 2.096 |

### Where the wall time goes (same 6-layer chunk, worst device)

| | new 1D Lin | new 1D Ring | new 1D ring + Linear | new 2D torus XY |
| --- | ---: | ---: | ---: | ---: |
| device kernel time | 36.1 ms | 34.5 ms | 36.1 ms | 35.2 ms |
| op-to-op gaps | 54.6 ms | 57.7 ms | 54.9 ms | 66.8 ms |
| wall-clock | 95.0 ms | 94.2 ms | 92.8 ms | 104.0 ms |

## Interpretation

* **Ring and 2D are NOT slower on device.** Kernel time per 6 layers is 36.1 ms on 1D Linear, 34.5 ms
  with every legacy CCL on Ring, 35.2 ms on 2D torus XY. Every collective gets faster on a wrapped
  fabric. The only kernels that lose are the MoE `combine` / `moe_reduce` (+~0.3 ms per sparse layer on
  2D) and `dispatch` (+0.15 ms under the ring fabric regardless of topology).
* **What makes rows 4 and 5 slower end-to-end is host / dispatch overhead**, visible as op-to-op gaps:
  +3 ms per 6 layers for Ring topology on the legacy CCLs (plus the fused matmul+reduce-scatter path
  that Ring enables), +12 ms per 6 layers under Fabric2D. Under 2D the gap tax is uniform (~30 us per
  matmul, ~100 us per legacy all-gather) plus ~480 us per `high_bw_all_gather` call: the op re-runs its
  Fabric2D direct-neighbour route proof (~64 control-plane queries for 4 columns x 8 ranks x 2
  directions) on **every** call, before the program-cache lookup. Trace replay would remove all of this,
  at which point 2D torus should come out slightly ahead of 1D on kernel time alone.
* Row 6 works because the ring fabric costs nothing for ops that don't use it (cold chunk is not slower)
  and `high_bw_all_gather` picks the ring up by itself. It is not the default: the gain over row 3 is
  within run-to-run noise, and the ring fabric needs the torus descriptor (wrap cables) that CI and the
  production runner do not set.

## Follow-ups / tracking

* **2D fabric**: the MoE team is landing `combine` / `dispatch` kernels optimised for Fabric2D; re-run
  rows 5/6 under 2D when they land, and once prefill runs under trace (`use_trace`) the host gaps above
  disappear. Then revisit `M3_FABRIC=FABRIC_2D_TORUS_XY` (needed anyway for multi-galaxy D2D pipelining).
* **Ring topology on the legacy CCLs** (row 4): faster kernels, slower dispatch — revisit under trace.
* **For the op owner**: (a) memoize the Fabric2D route plan per (mesh, axis, num_links, fabric config)
  instead of re-proving it per call; (b) an 8x1 submesh carved from the 8x4 galaxy returned wrong data
  and hung `close_mesh_device` — the op's own 512k test uses that submesh and measures line speed on a
  torus box for the same reason.
* Cold-cache weight loads on this box read the 256 GB bf4 set from NFS at ~60 MB/s (>1 h); the runs
  above were done with the set pre-read into the 566 GB page cache (a few minutes per model load).

## Reproducing

```bash
# unit PCC (high_bw cache read vs the previous path kept as reference, ~2 min on the galaxy)
pytest models/demos/minimax_m3/tests/unit/test_msa_sp_cache_read_vs_ref.py -s
# perf sweep / zone profile (defaults: FABRIC_1D, Linear CCLs, plain mesh descriptor = row 3)
./models/demos/minimax_m3/scripts/run_prefill_perf.sh
LAYERS=6 CACHE=25600 ./models/demos/minimax_m3/scripts/run_prefill_profile.sh
# rows 4-6: M3_FABRIC=FABRIC_1D_RING [M3_CCL_TOPOLOGY=Ring] | M3_FABRIC=FABRIC_2D_TORUS_XY  (perf sweep)
#           FABRIC=1d_ring [M3_CCL_TOPOLOGY=Ring] | FABRIC=2d_torus_xy                       (profile)
```
