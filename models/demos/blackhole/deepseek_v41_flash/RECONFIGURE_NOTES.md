# One model build, many batch sizes: `Model.reconfigure`

Branch `ssinghal/dsv4p1-reconf` (worktree `/mnt/tt-data/ssinghal/wt/pf_reconf`), base `ssinghal/dsv4p1`. Default behaviour (one batch size per process) is unchanged.

## How to run a multi-batch session

```
DSV41_LAYERS=0-39 DSV41_SESSION=isl4k_b4,isl4k_b8,isl4k_b16,isl4k_b32,isl4k_b64,isl4k_b128,isl4k_b4 \
  pytest -x -s -q models/demos/blackhole/deepseek_v41_flash/demo/text_demo.py -k session
```

* The first scenario builds the model (60-90 min at 40 layers, ~15 min of that the Engram tables into host RAM). Every later scenario with ANOTHER padded batch size calls
  `Generator.reconfigure(...)` -> `Model.reconfigure(...)` instead of a second build, any order (small -> large, large -> small, repeats).
  `DSV41_SESSION_RECONFIG=0` restores the old behaviour (one build per batch size; the second one runs out of DRAM).
* Every reconfigure logs one `RECONFIGURE B 4 (U=1, ctx 8192) -> B 8 (U=2, ctx 8192): release .. s, rebuild .. s; DRAM MiB/bank before [..] | weights only [..] | after [..] | L1 ...`
  line (free DRAM / largest free block per bank before the release, with only the weights left, and after the rebuild; L1 allocated bytes per bank).
* `max_seq_len` of a batch size in a session = the largest `max_seq_len` of the session's scenarios of that batch size (KV pool and per-context tables are sized for it).
* API: `Generator.reconfigure(max_batch_size, max_seq_len, paged_attention_config, layer_ids, kv_dtype)`, `tt.common.reconfigure_tt_model(model, mesh_device, ...)`,
  `Model.reconfigure(args, max_ctx, num_pages, kv_dtype)` (returns timings + DRAM snapshots).
* Debug knobs: `DSV41_RECONFIG_DEBUG=1` (lists old-model objects that survive the release with their referrer chain), `DSV41_RECONFIG_CLEAR_PCACHE=0` (keep the program cache),
  `DSV41_L1_DUMP=<prefix>` (with `DSV41_MEMLOG=1`: per-allocation allocator dumps in generated/reports/), `DSV41_MEMLOG=1` (DRAM + L1 per phase, now with the L1 column).

## What is built where and what depends on the batch size

`Model._build` (tt/dsv41_model.py), in this order:

| piece | built by | depends on batch (U = users per mesh row) / max_ctx? | on reconfigure |
|---|---|---|---|
| routed-expert weights (moe_compute ring layout, `tt_w0_w1`, `tt_w2`, expert mapping), ~95 % of device bytes | `DSV41MoEBlock` -> `TTMoEDecode` (tensorbin cache) | no | **kept** (`keep["experts"]`, passed as `expert_state=`) |
| unified prefill MoE weights | ring mode: the same tensors read in place; copy mode (`DSV41_UNI_RING=0`, `UNI_LAYERS=auto`): own tensors | no | kept (`keep["umoe"]`) |
| embedding table, LM head (bfp8), Engram device weights | `DSV41DeviceEmbedding/Head/Engram` | no (embedding `pre` is [U,..]) | kept; `embedding.rebatch(U)` |
| Engram host tables (2 x ~95 GB in RAM, `_MappedTable`) | `HostEngramRows` | no | kept (`HostEngramRows(tables=...)`); hash cache / row buffers `[B, max_ctx+1024]` rebuilt |
| attention (wqkv, wq_b, wo_a/b, sinks, compressor), shared expert, mHC consts, gate, norms, indexer weights | layer constructors from `load_layer` | no, but they are created inside objects that also own batch state | re-read from the checkpoint (~1-2 s/layer) |
| CCL semaphores (L1) | `DSV41DecodeChain` | no | released + re-created (see L1 below) |
| KV pool `[1,1,NP*320 + n_ring*U*ring_rows, 512]`, page table, host page allocators, ring regions | `PagedKVPool` | U, num_pages, max_ctx, dtype | rebuilt |
| per-layer decode state: compressor state `cs_state`, `prev_cs`, index-key slabs (`n_alloc` by max_ctx, B users), step states | attention / indexer constructors | U, max_ctx | rebuilt |
| decode MoE buffers (dispatch output, indices / scores in L1, combine output, 2 global semaphores) + gate config (`batch_per_device`) | `DSV41MoEBlock` | U | rebuilt |
| prefill MoE front-ends (T = 32*G tokens, G by U), shared T=32 buffers `_G1_BUFFERS`, unified-MoE shared tables | `DSV41PrefillMoE`, `prefill_unified_moe._SHARED` | U | rebuilt |
| prefill model: input buffers per chunk size (`alloc_inputs`), `DynCtx`, head stash / one-hot selectors, `PagedStateSink` index tensors, prefill-sparse tables, chunk trace, head trace(s) | `GenPrefillModel` | U, chunk C, S_pad (max_ctx) | traces released, rebuilt on the next prefill |
| decode: device-loop buffers, packed Engram rows `rows_cat`, decode trace, spec runner | `Model.prepare_for_traces`, `Model._capture_decode`, `Generator.enable_spec` | U | released, rebuilt lazily (spec: `enable_spec` again) |
| module-level caches of device constants (`prefill_attention._MASKS/_TABS/_ZEROS`, `prefill_sparse._CONST/_PERSIST`, `pf_tune._P64`, `prefill_unified_moe._SHARED`, `prefill_layer._G1_BUFFERS`, `mhc_mixes2._PLANS`) | lazily | chunk / U | freed + cleared |
| program cache (programs own L1 semaphores and were specialised to the old shapes) | ttnn | yes | `mesh_device.clear_program_cache()` |

## Why the old "second build in the same process" ran out of DRAM

1. It is not (mainly) a leak: a second `create_tt_model` builds a SECOND full weight set. After the first 40-layer build only ~0.9 GiB/bank are free
   (`MEMLOG model built`: allocated 2.96 GiB/bank, free 928 MiB at U=1), the second weight set needs ~2.8 GiB/bank. In the demo session the first model also stayed referenced
   (`cache[key] = (model_args, model, generator)`), and no API released a built model's tensors.
2. Even with DRAM available the second build would fail in L1: the L1 allocator is first-fit and the persistent L1 allocations of a model (two global-semaphore groups, the
   indices / scores buffers of the decode and prefill MoE, CCL semaphores, ~39 KB/bank) are laid out so tightly that the big programs' static circular buffers (up to ~1.36 MB) just
   fit below them. In the first reconfigure prototype the 4 prefill-MoE buffers (2 x 16 KB + 2 x 2 KB) of the old model survived (garbage cycle) and the new, larger ones landed lower:
   `Statically allocated circular buffers in program ... clash with L1 buffers` in the prefill SDPA.
3. What release must therefore do (all implemented, order matters): release the traces first (they own the intermediate buffers of their capture), drop every reference of the old
   model (no local variable may keep the old prefill model alive through the gc), clear the module-level caches, `gc.collect()` x3 (the model graph is full of reference cycles: hooks,
   `functools.partial(self.sink.write, attn)`, MEMLOG wrappers), free the CCL semaphores as well (kept CCL objects fragment the first-fit L1 allocator), then clear the program cache.
   Verified: after the release L1 allocated = 0 B/bank (fresh process: 0), and DRAM allocated = the weights only, identical after every cycle.

## Implementation (files)

* `tt/dsv41_model.py`: `Model.__init__` -> `_build()`; `_keep` (batch independent state collected at the end of every build); `reconfigure`, `_release_batch_state`,
  `_release_prefill_traces`, `dram_snapshot`; MEMLOG lines now also print the L1 allocation; `DSV41_L1_DUMP`.
* `tt/reconfigure.py`: `reset_module_caches`, `free_tensors`, `debug_l1_referrers`.
* `tt/moe_block.py`, `tt/layer.py`: `expert_state=` (reuse the uploaded expert weights); `tt/moe_weights.py`, `tt/loader.py`: `load_experts=False` (no 384-expert read);
  `tt/engram.py`: `HostEngramRows(tables=)`; `tt/device_head.py`: `DSV41DeviceEmbedding.rebatch`.
* `tt/common.py` (`reconfigure_tt_model`), `tt/generator.py` (`Generator.reconfigure`), `demo/text_demo.py` (session switches batch size, `build_len` per batch size,
  scenarios `prefill_128_b8` / `prefill_128_b64`).
* Tools: `tools/devrun_rc.sh` (flock wrapper of this worktree), `rc_session.sh` (multi-batch session), `rc_baselines.sh` (fresh process per scenario), `rc_compare.py` (session vs fresh
  outputs), `rc_commit.sh`.

## Validation (see the final report for the tables)

All runs on the 4x8 Blackhole galaxy hosts of the cluster, one device job per host, logs in `/mnt/tt-data/ssinghal/dsv4-logs/pf_reconf_*.log`.
"Equal to fresh" = per scenario the FIRSTTOK_IDS line and the decoded 64-token text of EVERY user of the session log equal those of a separate fresh process running only that scenario
(`tools/rc_compare.py`; fresh logs `pf_reconf_base*_<scenario>.log`; for 40 layers the main-tree grid logs `grid_b{4,16,64,128}_G4k.log`, head a9593c4477a / 35b7afdefda).

### 1. Layers 0-3 (Engram layer 1, unified prefill MoE on, indexer on at ISL 3.7k), `isl4k_b4,b8,b16,b32,b64,b128,b4` in ONE process (`cyc4d`)
Equal to fresh for all 7 scenario runs (4, 8, 16, 32, 64, 128 users, and B=4 again after the whole cycle).

first build ('model built', U=1): alloc 459.8 MiB/bank, free 3423.9, largest block 3423.8

| reconfigure | release s | rebuild s | free MiB/bank before | weights only: alloc / free | after: alloc / free / largest block | L1 B/bank released / after |
|---|---|---|---|---|---|---|
| B 4 -> 8 (ctx 8192 -> 8192) | 0.9 | 9.8 | 3394.0 | 408.2 / 3475.5 | 463.4 / 3420.3 / 3419.3 | 0 / 38592 |
| B 8 -> 16 (ctx 8192 -> 8192) | 0.9 | 9.1 | 3378.3 | 408.2 / 3475.5 | 470.5 / 3413.1 / 3411.4 | 0 / 38848 |
| B 16 -> 32 (ctx 8192 -> 8192) | 1.1 | 9.0 | 3291.4 | 408.2 / 3475.5 | 481.7 / 3402.0 / 3396.8 | 0 / 6464 |
| B 32 -> 64 (ctx 8192 -> 8192) | 0.9 | 9.5 | 3378.3 | 408.2 / 3475.5 | 513.5 / 3370.2 / 3367.9 | 0 / 40384 |
| B 64 -> 128 (ctx 8192 -> 8192) | 1.0 | 8.8 | 3335.6 | 408.2 / 3475.5 | 570.7 / 3313.0 / 3310.4 | 0 / 42432 |
| B 128 -> 4 (ctx 8192 -> 8192) | 1.1 | 7.5 | 3259.2 | 408.2 / 3475.5 | 459.8 / 3423.9 / 3422.3 | 0 / 38464 |

Weights only = the model after the release (allocated DRAM is the weights, constant 408.2 MiB/bank over six reconfigures, L1 allocated 0 B like a fresh process before its first build);
the last "after" (B=4 again) has exactly the allocation of the first build (459.8 MiB/bank allocated, 3423.9 free): nothing leaks. (Before the `_w_rows` fix the Engram device layer's per-T
weight-row cache grew the weights-only number by ~0.2 MiB per new batch size and fragmented the largest free block by ~57 MiB at 4 layers / ~92 MiB at 40 layers; it is now cleared on release.)

### 2. Other cycles (all equal to fresh)
* layers 2-5 (no Engram layer; compressed ratio-2 / ratio-1 layers with indexer), same 7 scenarios: equal (`cyc25b`).
* layers 0-3, `gsm8k_b4,b16,b32,b64,b128,b4` (a different real prompt per user, instruct template, up to 384 tokens): equal for all users (`cycg2`). `gsm8k_b8` (U=2 x C=128) fails with
  "Tensor is not allocated" in `forward_cols` also in a FRESH process (pre-existing, not related to reconfigure; skipped).
* same batch, longer context (`DSV41_SESSION_CTX_PER_SCENARIO=1`, layers 2-5, `isl4k_b16,isl8k_b16,isl16k_b16,isl4k_b16`: ctx 8192 -> 16384 -> 32768 reconfigures at B=16): equal to fresh for isl8k / isl16k and back at isl4k (`cycctx`):

first build ('model built', U=1): alloc 465.8 MiB/bank, free 3417.9, largest block 3417.7

| reconfigure | release s | rebuild s | free MiB/bank before | weights only: alloc / free | after: alloc / free / largest block | L1 B/bank released / after |
|---|---|---|---|---|---|---|
| B 16 -> 16 (ctx 8192 -> 16384) | 1.0 | 8.9 | 3313.6 | 405.5 / 3478.2 | 481.1 / 3402.6 / 3397.3 | 0 / 38848 |
| B 16 -> 16 (ctx 16384 -> 32768) | 1.0 | 9.9 | 3297.3 | 405.5 / 3478.2 | 511.6 / 3372.1 / 3368.2 | 0 / 38848 |

* speculative decoding (DSV41_SPEC=3, layers 0-3, `gsm8k_b16,b32,b16`): the spec runner is dropped by the reconfigure and rebuilt; B=16 before and after the B=32 detour give identical spec statistics (383 rounds,
  identical first-divergence lists). (Spec exactness itself is not meaningful at 4 layers: 0/16 identical to plain also without any reconfigure; not re-validated at 40 layers.)

### 3. 40 layers (DSV41_LAYERS=0-39), one process: `isl4k_b4, isl4k_b16, isl4k_b64, isl8k_b16, isl4k_b4` (`s40a`, 81 min in total including the 55 min first build)
Equal to the grid fresh-process runs at B=4, 16, 64 (isl4k, all users, first tokens `[427, ...]`) and B=4 again after three reconfigures; isl8k_b16 has no fresh reference (self-consistent only).

first build ('model built', U=1): alloc 2955.8 MiB/bank, free 927.9, largest block 927.8

| reconfigure | release s | rebuild s | free MiB/bank before | weights only: alloc / free | after: alloc / free / largest block | L1 B/bank released / after |
|---|---|---|---|---|---|---|
| B 4 -> 16 (ctx 8192 -> 16384) | 1.8 | 95.9 | 832.6 | 2549.8 / 1333.9 | 3014.9 / 868.7 / 835.6 | 0 / 38848 |
| B 16 -> 64 (ctx 16384 -> 8192) | 2.3 | 84.4 | 670.9 | 2550.0 / 1333.7 | 3076.9 / 806.8 / 790.6 | 0 / 40384 |
| B 64 -> 16 (ctx 8192 -> 16384) | 2.3 | 69.1 | 729.5 | 2550.3 / 1333.4 | 3015.4 / 868.3 / 835.6 | 0 / 38848 |
| B 16 -> 4 (ctx 16384 -> 8192) | 3.1 | 66.9 | 658.2 | 2550.3 / 1333.4 | 2961.4 / 922.2 / 835.6 | 0 / 38464 |

Reconfigure time at 40 layers: 67-98 s (release 2-3 s) against 60-90 min for a build. The remaining DRAM difference at the end (alloc 2961.4 vs 2955.8 MiB/bank, largest block 835.6 vs 927.8) is the Engram `_w_rows`
cache described above (fixed afterwards, see 3b).

Timings of the scenarios (same session, noisy host): B=4 TTFT 5.9 s (first build) / 2.8 s (after 4 reconfigures; grid 3.4 s), B=16 11.9 s (grid 9.4 s), B=64 45.5 s (grid 39.3 s), decode 43.4 / 43.9 / 63.9 ms/token
(grid 45.0 / 42.7 / 65.5): no systematic difference after a reconfigure (the first, freshly built B=4 run of the same session was as slow as the reconfigured ones).


## Limits / not done

* The attention / shared-expert / mHC / norm / indexer weights are re-read from the checkpoint at each reconfigure (about 1-2 s per layer when the checkpoint is in the page cache), not
  kept on the device: they are uploaded inside the constructors of objects that also own batch state. A keep-everything variant (tag the host weight tensors, memoise their
  `ttnn.from_torch` uploads, replay with empty tensors) is possible but was not needed: reconfigure is minutes, not an hour.
* `kv_dtype` can change in a reconfigure (`DSV41_POOL_DTYPE`) but was not exercised; the layer set cannot change.
* A speculative runner (`enable_spec`) is dropped by `Generator.reconfigure` and must be built again after the next prefill.
* A reconfigure while a scenario failed half-way (open trace capture) is not attempted; the demo's failure hook releases the traces first.

### 3b. 40 layers with the Engram-cache fix: `isl4k_b4, isl4k_b16, isl4k_b64, isl4k_b128, isl4k_b4` (`s40b`, 58 min including a 25 min build)
Equal to the grid fresh-process runs for ALL users at B=4, 16, 64, 128 and B=4 again after four reconfigures.

first build ('model built', U=1): alloc 2955.8 MiB/bank, free 927.9, largest block 927.8

| reconfigure | release s | rebuild s | free MiB/bank before | weights only: alloc / free | after: alloc / free / largest block | L1 B/bank released / after |
|---|---|---|---|---|---|---|
| B 4 -> 16 (ctx 8192 -> 8192) | 1.7 | 86.5 | 832.6 | 2544.6 / 1339.0 | 2979.0 / 904.7 / 903.0 | 0 / 38848 |
| B 16 -> 64 (ctx 8192 -> 8192) | 2.3 | 74.1 | 691.3 | 2544.6 / 1339.0 | 3071.6 / 812.1 / 809.8 | 0 / 40384 |
| B 64 -> 128 (ctx 8192 -> 8192) | 2.3 | 69.5 | 729.8 | 2544.7 / 1339.0 | 3195.1 / 688.6 / 686.9 | 0 / 42432 |
| B 128 -> 4 (ctx 8192 -> 8192) | 2.7 | 63.6 | 628.1 | 2544.7 / 1339.0 | 2955.8 / 927.9 / 926.5 | 0 / 38464 |

After the last reconfigure (B=4) the allocation is again exactly that of the first build (2955.8 MiB/bank allocated, 927.9 free; largest block 926.5 vs 927.8), the weights-only
level is constant (2544.6 -> 2544.7 MiB/bank), L1 allocated is 0 B after each release.
