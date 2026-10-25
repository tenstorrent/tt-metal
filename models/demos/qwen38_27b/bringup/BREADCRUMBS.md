# qwen38_27b bring-up breadcrumbs

## C.linear.attn_norm (layer 0)

- `tt/norm.py:TtRmsNorm`: one fused `ttnn.rms_norm` on the local [1, 1, 640, 5120] bf16 rows (SP 8 on mesh axis 0,
  replicated on axis 1), no collective. The `1 + w` gamma is added in fp32 at load and replicated as a
  [1, 1, 1, 5120] fp32 TILE tensor. HiFi4 + fp32 dest (`NORM_FIDELITY`, `NORM_FP32_ACC`, `NORM_WEIGHT_DTYPE` in
  `tt/settings.py`).
- A ROW_MAJOR [1, 1, 160, 32] gamma overflowed L1 (1.85 MB of CBs): a row-major affine turns off the op's streaming
  kernel for a TILE input. A TILE gamma fixes it.
- `bringup/hooks.py`: `device_component` builds the module from the checkpoint's `input_layernorm.weight`. The host
  boundary (`_rows_to_device` / `_rows_to_host`) splits rows over mesh axis 0 and reads column 0 of each mesh row.
  `_ops_visible` runs the call with `ttnn.CONFIG.enable_fast_runtime_mode = False` for that call only, because
  FastOperation never calls ttnn's pre-operation hooks. Without it the gate's device proof counts 0 ops.
- Gate: PCC 0.999998, rel 0.0020 vs CPU (limit 0.0064), auto_margin_min 3.21, 1 device op, determinism
  bit-identical. Re-run: `bfw job check --task C.linear.attn_norm --gate-only`.

## C.linear.attention (layer 0)

- `tt/linear_attention.py:TtGatedDeltaNet`: the qwen36 SP fold (`qwen36/tt/sp.py`) plus deepseek's conv exchange
  (`deepseek_v3_d_p/tt/kda/kda.py`), copied and adapted to 8x4 (SP 8 on axis 0, TP 4 on axis 1; per column 4 k / 12 v
  heads, conv channels 2560). The ops: qkv|z linear (bf16) and a|b linear (fp32; a and b each padded to 32 columns),
  `kda.qkv_causal_conv1d_silu_with_carry` (fabric exchange on axis 0, carried conv history in, new carry out),
  g/beta in fp32, `chunk_gated_delta_rule` on v' = [0 | v] from [I | 0], `all_gather` of [A | B] (bf16) over axis 0,
  `kda.chain_affine_transforms` from the slot's carried recurrent state, [U | O0] @ [[S_entry], [I]], gated
  `rms_norm` (eps 1e-6) * silu(z), out_proj row-parallel, fp32 `all_reduce` over axis 1, typecast to bf16.
- Weight layout matters for bfp8 (one exponent per 16 values along a tile row): out_proj stays [N, K] and is read with
  `transpose_b=True`. As [K, N], output column 3993 was off by rel 0.50, because its block neighbour 3994 is about 20x
  larger at every linear layer. The in-projections stay [K, N]: with transpose_b the layer-3 input (one huge channel)
  failed rel 0.038 > 0.033.
- `GDN_DELTA_CHUNK` 64 (default) vs 32: the op decays its fp32 state once per chunk with `(dl*I) @ S` on the FPU,
  which reads fp32 operands with a short mantissa (measured: an fp32 identity matmul returns S x 0.99928, about half
  the values truncated to TF32). That compounds over the 20 chunks of one 640-row SP segment and the 8 segments of
  the chain on slow-decay heads. Post-chunk state vs golden: rel 0.016 / worst head 0.029 at 64, 0.029 / 0.057 at 32.
  At 64 the op takes rank-4 q/k/v, so q/k are l2-normalised in the module.
- The hook builds the state from `ctx.extra["state_prefix"]` (none at start 0) and writes the post-chunk state to
  `ctx.extra["state_out"]` (conv [conv_dim, 3], recurrent [48, 128, 128]), in the reference's layout.
- Gate: PCC 0.999994; vs CPU rel 0.0034 (limit 0.015); layer-3 input rel 0.0175 (limit 0.033); auto_margin_min 1.56;
  49 device ops; bit-identical determinism. Re-run: `bfw job check --task C.linear.attention --gate-only`.

## C.linear.attn_residual (layer 0)

- `tt/residual.py:TtResidualAdd`: one `ttnn.add` (bf16 out, DRAM) of two residual-stream tensors, [1, 1, 640, 5120]
  bf16 per chip (SP 8 on axis 0, replicated on axis 1). The add is local, so there is no collective.
- `bringup/hooks.py`: `device_component` returns it for `attn_residual` and for `mlp_residual` (the same add). Both
  inputs cross the harness boundary through `_rows_to_device`.
- Gate: PCC 0.999997, auto_margin_min 1.97, 1 device op, host flops 0. Re-run:
  `bfw job check --task C.linear.attn_residual --gate-only`.

## C.linear.mlp_norm (layer 0)

- Reuses `tt/norm.py:TtRmsNorm` (same layout and settings as attn_norm) with the checkpoint's
  `post_attention_layernorm.weight`; `device_component` serves `attn_norm` and `mlp_norm` from one branch.
- Gate: PCC 0.999995, rel vs CPU 0.0019 (limit 0.0063), auto_margin_min 2.71, 1 device op, host flops 0,
  bit-identical determinism. Re-run: `bfw job check --task C.linear.mlp_norm --gate-only`.

## C.linear.mlp (layer 0)

- `tt/mlp.py:TtMlp`: gate / up `ttnn.linear` column-parallel (4352 of 17408 intermediate columns per TP column,
  weights [H, I/4] bfp8, fp32 outputs) -> `silu(gate) * up` -> bf16 -> down `ttnn.linear` row-parallel (weight stored
  [H, I/4] and read with `transpose_b`, so bfp8 blocks run along K as for out_proj) with an fp32 partial ->
  `ttnn.all_reduce(cluster_axis=1)` in fp32 -> `ttnn.typecast` bf16. Token-local, no SP collective. Weights are
  replicated over the 8 SP rows.
- Settings `MLP_FIDELITY` (HiFi4) and `MLP_FP32_ACC` (True), both in MAX_PRECISION. Auto matmul program configs (no
  tuning yet; known issue: ttnn.linear's auto config is slow, a perf task can add 2D program configs).
- `device_component` serves `mlp` for any layer (the attn block's mlp is the same module).
- Gate: PCC 0.999985, rel vs CPU 0.0054 (limit 0.0113), vs golden 0.0056, ratio [0.9959, 1.0033], worst column
  0.010 (col 3812), auto_margin_min 2.11, 7 device ops, host flops 0. Re-run:
  `bfw job check --task C.linear.mlp --gate-only`.

## C.attn.attention (layer 3)

- `tt/attention.py:TtGatedAttention`: no prior; composed from gpt_oss_d_p `dense_sp.py` / `kv_cache.py` (ring SDPA
  over a block-cyclic served cache) and qwen36 `rope_tp.py` (partial RoPE). Per chip 6 q heads (+ 6 gates) and 1 kv
  head (TP column), 640 local rows (SP row). Ops: one fused q|k|v|gate `ttnn.linear` (per column q 1536 | k 256 |
  v 256 | gate 1536, the checkpoint's per-head [q | gate] rows split at load; weight [K, N] bfp8) ->
  `nlp_create_qkv_heads` -> per-head `rms_norm` with a (1 + w) fp32 gamma -> `rotary_embedding_hf` on the first 64
  dims + concat -> K/V typecast bfp8 and `update_padded_kv_cache` (kv_actual_global = start, cluster_axis 0) ->
  `ring_joint_scaled_dot_product_attention` (cluster_axis 0, Linear, causal, logical_n = start + 5120,
  kv_cache_batch_idx / num_layers / layer_idx) -> `nlp_concat_heads` * sigmoid(gate) -> o_proj (weight [N, K], read
  with transpose_b) -> fp32 `all_reduce` axis 1 -> bf16.
- RoPE: cos/sin are built at load in each SP row's cache order (`blockcyclic_positions`, equal to deepseek's helper),
  sharded over axis 0, so the forward takes one device slice at rows [start / 8, start / 8 + 640) on every chip.
- Cache: the served layout from the start (bfp8, ND-sharded [1, 1, 32, 256] ROUND_ROBIN_1D, user-major batch
  slot * layers + layer, capacity rounded up to whole chunks). The hook loads the golden prefix [0, start) into it
  once per call at the harness boundary (`cache_from_host`) and passes `start` as a host int.
- Ring SDPA at fp32 DEST (max precision, `ATTN_SDPA_FP32_ACC`): q128/k128 fails to build ("Program size (73904) too
  large for kernel config buffer (70656)", kernel text of ring_joint reader/writer/compute); q64/k64 builds
  (`ATTN_SDPA_Q_CHUNK` / `ATTN_SDPA_K_CHUNK` default 64). fp32 DEST runs the op's legacy compute, which refuses
  `kv_actual_isl` (a start that is not a whole chunk); the module asserts start % chunk == 0. The serving contract's
  pad tail (actual_end, zeroing the last record) is not built here yet.
- Gate: PCC 0.999982; vs CPU rel 0.0042 (limit 0.0142); vs golden rel 0.0059 (limit 0.0183); chunk0 / layer0 /
  mixed / small / big all pass; auto_margin_min 1.21; 27 device ops, host FLOPs 0; bit-identical determinism.
  Re-run: `bfw job check --task C.attn.attention --gate-only`.

## M.1 assemble (layers 0-3, all on device)

- `tt/model.py:TtModel`: no prior; assembled from the validated modules. Hidden state [1, 1, 640, 5120] bf16 per chip
  from embedding to final norm. embed: uint32 ids sharded over rows -> `to_layout(TILE)` -> `ttnn.minimum(ids, V - 1)`
  (0xFFFFFFFF pad id) -> ROW_MAJOR -> `ttnn.embedding` on the replicated bf16 table. Each block runs through
  `run_block` with the device steps; the block's intermediates are freed after it. lm_head bfp8, vocab split over TP,
  gathered on the host in `logits` (harness boundary).
- `TtState`: one served KV cache for all attention layers (bfp8, ND-sharded [1, 1, 32, 256] ROUND_ROBIN_1D,
  batch = slot * n_attn + attention ordinal, `KV_SLOTS` = 2) and per (slot, linear layer) a GdnState (conv bf16,
  recurrent fp32). `formats()` / `device_tensors()` for the contract; `to_torch` / `load_prefix` at the harness
  boundary. Start 0 runs the linear layers from a load-time zero state (a slot reused by a new request resets).
- actual_start / actual_end are replicated uint32 scalars built once (one per chunk-aligned position, at
  `new_state`): no host transfer inside `layer`.
- Pad tail (contract): attention writes K/V with `update_padded_kv_cache(valid_global=end)` and zeroes the last
  record with `zero_padded_kv_cache(pad_align=32)`; GDN gets `actual_end` on the conv (carry cut at the end) and
  g = beta = 0 on pad rows (a device mask `row_offset < end - start` built at load), so the recurrent state skips them.
- `bringup/hooks.py:device_model` wraps TtModel; only `layer` runs under `_ops_visible` (device-op proof).
- Re-run: `bfw job check --task M.1 --gate-only`.

## K.1 contract (adapter, served cache, table)

- `tt/runners/kv_cache.py`: the engine-owned serving state `Qwen38KvCaches` (allocate_kv_cache): one bfp8 K/V cache for
  the served attention layers (ND-sharded [1, 1, 32, 256] ROUND_ROBIN_1D, user-major slot * n_attn + ordinal, kv head
  c on TP column c) and per (slot, linear layer) the GDN conv (bf16) / recurrent (fp32) state. `TtState` now runs on
  this object (`model.new_state(max_seq, caches=...)`; it allocates one the same way by default), so the ladder and
  the runner share one cache format.
- Table (`build_table`): configs 0..3 = K heads, 4..7 = V heads (gpt_oss_d_p layout), chunk_n_tokens 32, 8704 B,
  num_layers = served layers, entries only on attention layers. One-chip device groups (head h on column h, position p
  on SP row (p // 640) % 8, local row (p // 5120) * 640 + p % 640; shard j in bank j % B at base + (j // B) * 8704).
  The walk matches `blockcyclic_positions` for every 32-token position (checked on the CPU).
- `tt/runners/runtime.py`: eager, one rank. prefill_chunk embeds the engine's [8, 1, 640] uint32 input
  (`embed_engine_input`), runs every served layer on the slot, acks after each layer (D2H record when the runner
  passes the service, else device sync + host sink). Chunk starts must be chunk multiples (prefix_reuse false).
- Adapter `tt/runners/adapter.py`, registered as `qwen38_27b`; checkpoint from the `HF_DIR` setting. The producer's
  KV read-back goes through the adapter hook `read_slot_kv_and_check_pcc` (new generic branch in prefill_producer).
- hooks: `contract_state_pcc` (GDN state vs golden snapshot), `contract_independent_pcc` (ttnn's bfp8 decoder).
- Re-run: `bfw job check --task K.1 --gate-only`.
- Attempt 1 result: the gate FAILs in the framework, not the model. `testing/contract.py:102` calls
  `get_chip_unique_id_from_fabric_node_id(mesh_id, chip_id)`; this ttnn wants `(mesh, mesh_id, chip_id)`. Its layout
  check also passes `range(L)`, so GDN layers without records fail it (owner answer hybrid_table_check). A scratch copy
  of the test with only those two changes passes: contract_checks_failed 0, acks_early 0, producer KV k 0.99987 /
  v 0.99981, GDN state conv 0.99993 / recurrent 0.99978. The runner test could not start here: on
  bh-glx-120-b07u08 the runner's H2DStreamService fails to pin host pages (IOMMU disabled on chip 15). With the
  engine feeder it would also stop at its ack-channel precheck until tt-d-gen is re-pinned (owner answer engine_pin).

### K.1 attempt 2 (2026-10-24)
No code change. The framework's `testing/contract.py` is unchanged: line 102 still calls the ttnn function without the
mesh, and line 155 still passes `range(L)`. The gate failed the same way: TypeError at contract.py:102,
layout_failures 1. Chip 15 on bh-glx-120-b07u08 still cannot pin DMA pages. Re-run with:
`bfw job check --task K.1 --gate-only`.

### K.1 attempt 3 (2026-10-24)
- Same code as attempt 1, nothing changed. I ran `bfw job check --task K.1 --gate-only` once (94 s). It failed the same way: a TypeError at framework `testing/contract.py:102`, which calls `get_chip_unique_id_from_fabric_node_id` without the mesh, and `layout_failures` = 1 because line 155 passes `range(L)`, which includes the GDN layers that have no records. Both lines are outside K.1's paths, so a person has to fix them. Chip 15's DMA pin warning is still in the log.

### K.1 (run 20261024-173440, attempt 1)
- Restored the parent's K.1 work (refs/bringup/wip/K.1) unchanged. With the framework fixes, the adapter contract test
  passes: contract_checks_failed 0, acks_early 0, layout_failures 0, producer KV k 0.99987 / v 0.99981.
- The runner test ran for the first time, fed by tt-d-gen's engine (driver attached in 57 s, a1/b1 admitted on slots
  0/1). The runner died 5 s after "entering request loop", before any chunk ran: `TT_THROW: TIMEOUT` in its own H2D
  drain (`_socket_next` -> `_decode_metadata`). The test copies the safe runner's `TT_METAL_OPERATION_TIMEOUT_SECONDS=5`
  into the runner subprocess, so waiting idle for the next chunk counts as a hang (known_issues "The safe runner's 5 s
  dispatch timeout..."). The gate reported HANG (fabric routers left mid-flight; the device reset failed).
- Fix: `prefill_runner.py` `_apply_operation_timeout_floor()`, run at import before the mesh opens, raises a positive
  timeout below `PREFILL_MIN_OPERATION_TIMEOUT_S` (default 600 s) to that floor; 0 (no timeout) stays as it is. Not
  re-run in this session (the board stays hung until the next attempt).
- Re-run: `bfw job check --task K.1 --gate-only`.

### K.1 (run 20261024-173440, attempt 2)
- Code unchanged from attempt 1 (the parent's K.1 work plus the runner's timeout floor). The gate
  (`bfw job check --task K.1 --gate-only`, which uses `--dev` because attempt 1 hung) FAILs in 77 s, before any
  model check: under the watcher, `chunk_gated_delta_rule` at chunk 64 does not build. TT_FATAL "Program size (90016)
  too large for kernel config buffer (70656)" (fused program: chunk_gdn_prep / reader / writer_chunk_gdn_fused).
  Probes of layer-0 linear attention with `--dev`: phased program at chunk 64 is 89232 B (prep trisc0-2 alone 80 KB);
  mono needs 2.3 MB of CBs; fused at chunk 32 is 1648 B over; phased at chunk 32 builds and passes (PCC 0.999994).
  I did not change the shipped GDN chunk or program: chunk 64 is the measured-better choice (GDN_DELTA_CHUNK), and the
  limit applies only to the watcher build.
- Runner test without the watcher (`scripts/run_safe_pytest.sh --run-all .../test_runner_contract.py`), fed by
  tt-d-gen's engine @ 1cf34388: the runner logs the timeout floor (5 -> 600), reaches CHUNK_START, runs 17 chunks, and
  the driver reports ok. a1/b1 PREFILL_DONE at 8017, a2 at 56320, b2 at 10240, on tt-d-gen's chunk plan. Shutdown is clean.
  The test then fails in the framework: `server_rules.harness` copies tt-d-gen's `kv_manager/tests/harness/tables.py`
  out of its package, so its `from .config import ...` raises ImportError.
- A scratch copy of the test that imports `tables` as a package module (diagnosis only; frozen test untouched): the
  table matches tt-d-gen's profile on all 8 configs, layout_failures 0, KV through the table vs golden with
  kv_dump_compare is 0.99977-0.99992 for every head (a2 [0, 56320), b2 [0, 10240)). The only failures are in "starts":
  `ran [(5120, 8017), (0, 5120)]` for a1 and b1. The test picks the last len(plan) chunks on the slot that are in the
  plan, and a2/b2 restart the same slot at 0 (prefix_reuse false), so a1's own (0, 5120) is shadowed by a2's. The
  runner ran a1 as [0,5120) then [5120,8017).
- The adapter contract test without the watcher: PASS.
- Re-run: `bfw job check --task K.1 --gate-only`.

### K.1 (run 20261024-230754, attempt 1)
- Code unchanged: the K.1 work from refs/bringup/wip/K.1 plus the runner's timeout floor. The gate ran without `--dev`
  (423 s). The adapter contract test PASSes: contract_checks_failed 0, acks_early 0, layout_failures 0, KV k 0.99987 /
  v 0.99981.
- Runner test, fed by tt-d-gen 1cf34388's engine: a1/b1/a2/b2 PREFILL_DONE at their prompt lengths, driver ok. The
  table profile and layout checks pass, and so does KV via the table vs golden: every head >= 0.99977 for a2
  [0, 56320) and b2 [0, 10240).
- Failed only "starts" for a1/b1: `ran [(5120, 8017), (0, 5120)], plan [(0, 5120), (5120, 8017)]`. The runner log shows
  the engine's plan, in order (c=0 slot0 [0,5120), c=2 slot0 [5120,8017), c=4 slot0 [0,5120) = a2 ...). The test
  filters every chunk on the slot that matches a1's plan over the whole log. With prefix_reuse false, the follow-up a2
  restarts at [0,5120) on the same slot, so its first chunk is counted as a1's. The check is in the frozen
  test_runner_contract.py (outside K.1's paths), so a person has to fix it. The chunks come from tt-d-gen, so the
  model cannot change them.
- Re-run: `bfw job check --task K.1 --gate-only`.
