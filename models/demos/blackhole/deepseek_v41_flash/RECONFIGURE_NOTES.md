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

RESULTS_PLACEHOLDER

## Limits / not done

* The attention / shared-expert / mHC / norm / indexer weights are re-read from the checkpoint at each reconfigure (about 1-2 s per layer when the checkpoint is in the page cache), not
  kept on the device: they are uploaded inside the constructors of objects that also own batch state. A keep-everything variant (tag the host weight tensors, memoise their
  `ttnn.from_torch` uploads, replay with empty tensors) is possible but was not needed: reconfigure is minutes, not an hour.
* `kv_dtype` can change in a reconfigure (`DSV41_POOL_DTYPE`) but was not exercised; the layer set cannot change.
* A speculative runner (`enable_spec`) is dropped by `Generator.reconfigure` and must be built again after the next prefill.
* A reconfigure while a scenario failed half-way (open trace capture) is not attempted; the demo's failure hook releases the traces first.
