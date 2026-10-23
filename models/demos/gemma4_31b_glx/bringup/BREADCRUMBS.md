# gemma4_31b_glx bring-up breadcrumbs

## Reference (R.1 / R.2 / R.3), 2026-10-17

- `reference/gemma4_ref.py`: standalone pure-torch fp32 reference (`Gemma4Reference`). It reads the HF safetensors
  directly, loading only the selected layers with 16 threads (about 1 s per layer pair from page cache), and mirrors
  transformers 5.12 `Gemma4TextModel` op for op. It also sets the KV state layout: key/value `[Hkv, max_seq, D]`, K
  after k_norm + RoPE, V after v_norm.
- Block graph (both block types): attn_norm, attention (stateful), post_attn_norm, attn_residual, ffn_norm, mlp,
  post_ffn_norm, ffn_residual (`(h_mid + mlp_post) * layer_scalar`).
- Gemma-4 31B specifics:
  - RMSNorm is `x * w`; v_norm has no weight.
  - Attention scale is 1.0.
  - Sliding layers: window 1024 (kv > q - 1024), 16 kv heads x 256, RoPE theta 1e4 on all dims.
  - Full layers: no v_proj (V = v_norm(raw k_proj)), 4 kv heads x 512, "proportional" RoPE theta 1e6 with only the
    first 64 of the 256 frequencies non-zero, rotate-half layout.
  - Tied lm_head, final logit softcap 30, embed scale sqrt(5376).
  - layer_scalar is a per-layer checkpoint tensor.
- Attention runs in 512-row query blocks against the cached K/V. The score matrix stays bounded at 56k, and sliding
  layers read only their window.
- `bringup/reference_hooks.py`: `reference`, plus `hf_layers` (text layers at `model.model.language_model.layers`).
- Speed (64-core EPYC 9354P, 32 threads): about 4.3 s per layer per 5120-token chunk at position 0.
- Re-run: `bfw tasks run R.1 R.2 R.3`. R.3's gate (`--seq 5120 --chunk 5120`) trips check_reference's "need at least
  two chunks" assert, which is a ledger/framework issue. The same check passes exactly at
  `--seq 5120 --chunk 2560` and `--seq 10240 --chunk 5120`.

## Reference retry (attempt 1.r1), 2026-10-17

- No code change. On bh-glx-120-c06u14 the python_env interpreter resolves, so `bfw tasks run` runs as written. R.1 and R.2 PASS.
- R.3 still fails: its ledger command `--seq 5120 --chunk 5120` trips check_reference's assert "need at least two chunks".
- With `--seq 5120 --chunk 2560` the same check passes: hidden and state pcc 1.0, graph replay diff 0.0.
- The R.3 ledger entry needs a two-chunk command. That is a person's or the framework's fix; this step may not edit it.

## Reference (run 20261017-130217-gemma31-glx-2, attempt 1), 2026-10-17

- No code change. The overseer's saved files (gemma4_ref.py, reference_hooks.py, BREADCRUMBS.md) were copied in as is.
- The ledger now has R.3 as `check_reference --seq 10240 --chunk 5120` (two chunks).
- `bfw tasks run R.1 R.2 R.3`: all PASS.
  - R.2: pcc_logits 0.99999999999967, min pcc_hidden 0.9999999999996, top1 1.0.
  - R.3: pcc_hidden 1.0, pcc_state_min 0.9999999999999, graph_replay_maxabs 0.0. It took 1060 s.

## Reference on the HF golden trace (run 20261017-221614-gemma31-glx-3, attempt 2), 2026-10-17

- No code change. R.1 now builds the prompt from the trace's prefix_ids and token cache (isl56320).
- `bfw tasks run R.1 R.2 R.3`: all PASS.
  - R.2: pcc_logits 0.9999999999998, top1 1.0.
  - R.3: pcc_hidden 0.99999999999993, pcc_state_min 0.9999999999999, graph_replay_maxabs 0.0.
- freeze_report.json reports L.starts as failing with the CPU reference as the module. That is not a reference bug:
  - `testing/chunk_starts.py:_rows` reshapes a state tensor to `[shape[0], -1]` and slices rows `[:n]`, so it treats
    dim 0 as positions.
  - This reference keeps key/value as `[Hkv, seq, D]`. The imported golden (`kv_cache/layer_i`, `[16, 56320, 256]`)
    uses the same layout, and `import_golden.maps_by_default` requires it for a plain-KV trace import.
  - The test therefore compares 16 head rows against 16 head rows of the whole 56320-long golden. The log shows
    `size of tensor a (32899072) must match ... b (230686720)`.
  - Making state seq-first would drop every layer from the trace import's state_layers, unless a trace_state hook
    (T.1) is written. The fix belongs in chunk_starts (slice positions on dim -2 for a 3-D KV tensor), which is
    the framework's to make.

## C.sliding.attn_norm (run 20261018-003406-gemma31-glx-3, attempt 1), 2026-10-18

- `tt/rms_norm.py`: `RMSNorm(mesh, weight, eps)`, adapted from gemma4_d_p `tt/rms_norm.py`.
  - The bf16 weight is uploaded once and replicated, as a TILE gamma `[1, 1, 1, 5376]`.
  - It runs block-sharded with gemma4_d_p's geometry. For 160 rows that is a 12 x 5 grid with 1 x 14-tile blocks.
  - Precision is HiFi4 with fp32 accumulation. Ops per call: to_memory_config, rms_norm, sharded_to_interleaved.
- `tt/weights.py` reads single tensors and the text config straight from the HF safetensors. Model code never
  imports the reference or bringup_fw.
- `tt/settings.py` gains these switches:
  - NORM_FIDELITY and NORM_FP32_ACC, both in MAX_PRECISION.
  - NORM_SHARDED.
  - COMPONENT_FAST_RUNTIME (default False, see the knowledge note).
- `bringup/hooks.py`:
  - `residual_to_device` / `residual_to_host` are the harness boundary. A chunk `[S, H]` maps to `[1, 1, S/32, H]`
    per chip, and chip (r, t) holds rows `r*S/8 + t*S/32 + [0, S/32)`. These are the plan's SP and TP row splits;
    the server's block-cyclic reshuffle is K.1's.
  - `_ResidualStep` wraps a module on that layout.
  - `device_component` handles only `attn_norm`. The other norms take the same RMSNorm with their own weight name
    (NORM_WEIGHTS) in their own tasks.
- On this tt-metal, `ttnn.ConcatMesh2dComposer` is gone. Use `ttnn.create_mesh_composer(mesh, ttnn.MeshComposerConfig(dims=[0, 1], mesh_shape_override=ttnn.MeshShape(8, 4)))`.
- Check results (`bfw job check --task C.sliding.attn_norm`): PASS.
  - pcc 0.9999962, auto_margin_min 2.87, device_ops 3, host_flops_frac 0.
  - Deterministic.

## C.sliding.attention (run 20261018-003406-gemma31-glx-3, attempt 1), 2026-10-18

- `tt/attention.py`: `SlidingAttention(mesh, w, cfg, max_seq, chunk)`, adapted from gemma4_d_p's ring_prefill.
  Everything it needs is built at load.
  - Per chunk, the forward runs: all_gather rows over TP; fused QKV linear (column-parallel,
    `[q 8 | k 4 | v 4] x 256` per TP column); nlp_create_qkv_heads; fp32 per-head RMS norms; fp32 elementwise RoPE
    with an exact 256x256 pair-rotation matmul; update_padded_kv_cache from fp32 into the served bfp8 cache; ring_joint
    SDPA (causal, window 1024, scale 1, kv_actual_isl); nlp_concat_heads; o_proj; reduce_scatter over TP.
  - Q/K weight rows and the q/k norm weights are permuted to adjacent-pair lanes (`sliding_kv_indices`), so the
    cache's K is in the served order.
  - Per-chunk positions (SP-sharded uint32) and kv_actual scalars are device tensors built once at load. The forward
    only indexes them by `start // chunk`.
  - Cache per chip: `[KV_SLOTS=2, 4, max_seq/8, 256]` bfp8, DRAM ND-sharded `[1,1,32,256]` round-robin. It is zeroed
    on device with DRAMZeroFill and has replicate topology.
- `bringup/hooks.py`: `_AttentionStep` loads `ctx.extra["state_prefix"]` into the cache one chunk at a time, through
  the same `write_kv` the forward uses. K is converted from NeoX to adjacent-pair order, and the upload is fp32.
  `device_component` handles step `attention` on sliding layers. max_seq and chunk come from `spec.get("target.seq")`
  and `spec.get("target.chunk")`; `shapes.*` returns None.
- `tt/settings.py` gains these switches:
  - ATTN_PROJ_FIDELITY, ATTN_PROJ_FP32_ACC, ATTN_SDPA_FIDELITY, ATTN_SDPA_FP32_ACC (False only), ATTN_SDPA_Q_CHUNK,
    ATTN_SDPA_K_CHUNK, ATTN_SDPA_K_SPLITS, ATTN_RING_NUM_LINKS, KV_SLOTS.
  - ATTN_HEAD_NORM (fp32 | rms_norm), ATTN_ROPE (elementwise | llama), ATTN_KV_FP32_WRITE (True).
- Norm-bias hunt. The first version had attention output median row norm +0.54% against a 0.4% limit. The sources:
  - Main cause (~+0.3% on the output): the device's bf16 -> bfp8 conversion rounds up on average. typecast raises row
    norms +0.28%; any op packing bfp8 from bf16 (clone, add, multiply with a bfp8 output) raises them +0.54%. fp32 ->
    bfp8 typecast is unbiased. Measured on unit-RMS rows with a scratch test. So K and V now stay fp32 until the cache
    write, and the prefix upload is fp32 too.
  - rotary_embedding_llama at head_dim 256 must run with bf16 dest (TT_FATAL otherwise). It raised q/k row norms
    +0.11%; elementwise fp32 RoPE gives -0.02%.
  - ttnn.rms_norm on q gave -0.08%. The fp32 eltwise norm gives 0.00%.
  - The ring SDPA itself (bf16 softmax statistics on the streaming path) is unbiased: +0.08% vs a float64 CPU SDPA on
    the same device q/k/v, insensitive to k_splits, q_chunk and packer_l1_acc. No fork was needed.
- Check results (`bfw job check --task C.sliding.attention`): PASS.
  - pcc 0.99998, auto_margin_min 1.026, device_ops 49, host_flops_frac 0. Deterministic (A,B,A x5 bit-identical).
  - Golden case: rel 0.0069 (limit 0.0084), bias +0.0023 (limit 0.004). The closest check is the column rel:
    0.0329 against 0.0338 at column 5161.

## C.sliding.post_attn_norm (run 20261018-003406-gemma31-glx-3, attempt 1), 2026-10-18

- `bringup/hooks.py`: `NORM_WEIGHTS["post_attn_norm"] = "post_attention_layernorm"`. It reuses the attn_norm
  `RMSNorm` (block-sharded, HiFi4 + fp32 acc) on the residual layout through `_ResidualStep`. No new module.
- Check (`bfw job check --task C.sliding.post_attn_norm`): PASS. Metrics: pcc 0.9999961, auto_margin_min 2.87,
  device_ops 3, host_flops_frac 0. Regressions C.sliding.attn_norm and C.sliding.attention: PASS.

## C.sliding.attn_residual (run 20261018-003406-gemma31-glx-3, attempt 1), 2026-10-18

- `tt/residual.py`: `ResidualAdd`, a single `ttnn.add` (bf16 out, DRAM) on the residual layout, as in gemma4_d_p
  `tt/layer.py`. Both inputs share the TP row split, so no collective. ffn_residual (with layer_scalar) is its own task.
- `bringup/hooks.py`: `device_component` returns `_ResidualStep(mesh, ResidualAdd())` for `attn_residual`.
- Check (`bfw job check --task C.sliding.attn_residual`): PASS. pcc 0.9999963, auto_margin_min 1.936, device_ops 1,
  host_flops_frac 0. Regressions C.sliding.attn_norm, C.sliding.attention, C.sliding.post_attn_norm: PASS.

## C.sliding.ffn_norm (run 20261018-003406-gemma31-glx-3, attempt 1), 2026-10-18

- `bringup/hooks.py`: `NORM_WEIGHTS["ffn_norm"] = "pre_feedforward_layernorm"`. Reuses the same `RMSNorm`
  (block-sharded, HiFi4 + fp32 acc) on the residual layout through `_ResidualStep`. No new module.
- Check (`bfw job check --task C.sliding.ffn_norm`): PASS. pcc 0.9999957, auto_margin_min 2.83, device_ops 3,
  host_flops_frac 0. Regressions C.sliding.attn_norm, attention, post_attn_norm, attn_residual: PASS.

## C.sliding.mlp (run 20261018-003406-gemma31-glx-3, attempt 1), 2026-10-18

- `tt/mlp.py`: `MLP(mesh, w)`, adapted from gemma4_d_p `tt/mlp.py` on this package's residual layout.
  - Per chunk: all_gather rows over TP; gate linear with fused gelu_tanh and up linear (column-parallel, 21504 / 4 =
    5376 intermediate per chip); gate * up; down linear (row-parallel); reduce_scatter rows over TP. 6 device ops.
  - Weights are bfp8 in DRAM, transposed once at load. Default ttnn.linear program configs (no explicit blocking).
- `tt/settings.py` adds MLP_FIDELITY (HiFi4), MLP_FP32_ACC (True) and MLP_FP32_INTERMEDIATE.
  - MLP_FP32_INTERMEDIATE: gate and up outputs are fp32 before the multiply. The shipped default is False (bf16, as
    in gemma4_d_p); MAX_PRECISION is True, so the component test ran with fp32.
- `bringup/hooks.py`: `device_component` returns `_ResidualStep(mesh, MLP(...))` for step `mlp`, for any block type.
  The attn block's MLP is the same module.
- Check (`bfw job check --task C.sliding.mlp`): PASS. pcc 0.9999828, auto_margin_min 1.705, device_ops 6,
  host_flops_frac 0.
  - Golden case: rel 0.0059 (limit 0.0114), bias +0.00085.
  - Regressions attn_norm, attention, post_attn_norm, attn_residual and ffn_norm: PASS.

## C.sliding.post_ffn_norm (run 20261018-003406-gemma31-glx-3, attempt 1), 2026-10-18

- `bringup/hooks.py`: added `post_ffn_norm -> post_feedforward_layernorm` to `NORM_WEIGHTS`. It reuses the existing
  `tt/rms_norm.py` RMSNorm (x * w, eps from config) on the residual's TP row split, through `_ResidualStep`.
- Check (`bfw job check --task C.sliding.post_ffn_norm`): PASS. pcc 0.9999959, auto_margin_min 2.84, device_ops 3,
  host_flops_frac 0.
  - Regressions attn_norm, attention, post_attn_norm, attn_residual, ffn_norm and mlp: PASS.

## C.sliding.ffn_residual (run 20261018-003406-gemma31-glx-3, attempt 1), 2026-10-18

- `tt/residual.py`: `ResidualAdd(scale=None)`. With a scale, the one `ttnn.add` carries a fused
  `MUL_UNARY_SFPU(scale)` activation: out = (h_mid + mlp_post) * layer_scalar, as in gemma4_d_p `tt/layer.py`.
  Without one, `activations=[]` (the binding rejects `None`; the first try broke the attn_residual regression).
- `bringup/hooks.py`: `ffn_residual` reads `layers.{layer}.layer_scalar` once at load as a Python float.
- Check (`bfw job check --task C.sliding.ffn_residual`): PASS. pcc 0.9999943, auto_margin_min 1.369 (vs_cpu rel
  0.00342 against 0.00469), bias -0.0024, device_ops 1, host_flops_frac 0. All 7 earlier sliding tasks: PASS.

## C.attn.attention (run 20261018-003406-gemma31-glx-3, attempt 1), 2026-10-18

- `tt/global_attention.py`: `GlobalAttention` (layer 5 and every full_attention layer). 62 device ops.
  - Tied QK projection (fp32 out). fp32 q / k / v head norms; V = norm(k_raw) without weight; k_norm gamma
    folded into q. Elementwise fp32 RoPE on the 128 rotary lanes.
  - Packed served cache `[2, 1, 7040, 640]` bfp8 as `[K rot 128 adjacent pairs | V value order 512]`, written with
    `deepseek_prefill.update_padded_kv_cache` from fp32 rows.
  - SDPA, `ATTN_GLOBAL_SDPA=composed`: per chip, all_gather the slot's rows over SP. QK^T in fp32 against the cache
    prefix (bfp8) and this chunk's fresh bf16 rows, plus a causal mask on the chunk. Then a composed fp32 softmax
    (max, subtract, exact exp, sum, reciprocal) and P V in fp32. q and heads stay fp32.
  - o_proj as `[out, in]` bfp8 with transpose_b (`ATTN_GLOBAL_W_LAYOUT=out_in`), fp32 partials through the TP
    reduce_scatter, and one bf16 typecast at the end.
- Why not ring_mla (kept as `ATTN_GLOBAL_SDPA=ring_mla`): with kv_actual_isl it only runs the streaming compute, which
  needs fp32_dest_acc False and keeps bf16 O / sum / max state. At head_dim 512 that put column 2524 at 1.42x its
  limit (margin 0.71).
- `bringup/hooks.py` `_GlobalAttentionStep._load_prefix` hands the prefix to `write_kv` as host-rounded bfp8, not
  fp32. The device's fp32 -> bfp8 typecast of the bf16-exact state biased prefix magnitudes by +0.28 %. That alone
  kept column 1658 at 1.02x its limit (margin 0.98 in runs 5-10).
- New settings: ATTN_GLOBAL_SDPA, ATTN_GLOBAL_FP32_REDUCE and ATTN_GLOBAL_FP32_QK, all at max precision in
  MAX_PRECISION.
- Check (`bfw job check --task C.attn.attention`): PASS. pcc 0.999991, auto_margin_min 1.133, device_ops 62,
  host_flops_frac 0.
  - vs_cpu: rel 0.0047 (limit 0.0096), column 1658 0.0338 (limit 0.0382).
  - vs golden: column 5264 0.0379 (limit 0.0475).
  - Regressions on all 8 sliding tasks and C.attn.attn_norm: PASS.

## M.1 (run 20261018-003406-gemma31-glx-3, attempt 1), 2026-10-18

- `tt/model.py` (new): the all-device Model. `bringup/hooks.py:device_model(mesh, spec, layers, lm_head=True)`
  returns it.
  - embed (bf16 table, `* sqrt(H)` in fp32) -> 60 blocks through `run_block` over the reference graph (`BLOCK`) ->
    final RMSNorm -> tied bfp8 LM head split over all 32 chips, soft-capped, fp32 logits.
  - `embed(tokens, start)` stages one `ChunkMeta` per chunk, shared by every layer:
    - the positions in device-row order (`chunk_positions`, the block-cyclic staircase of update_padded_kv_cache);
    - RoPE values looked up on device from load-time tables (`MODEL_ROPE_LOOKUP=hilo`);
    - the pad keep mask and the global causal / prefix masks;
    - the slot / kv_actual / valid scalars.
  - With no start, embed continues from the last chunk embedded since new_state (sequential ladder), so layer() runs
    no host transfer. layer() stages lazily only for a chunk-aligned start after from_host.
  - `State`: bfp8 key / value. `load_prefix`, `to_torch` and the de-block-cycled reads cover both the global packed
    cache and the sliding caches.
- Precision fixes (all Settings, all at max precision):
  - `RESIDUAL_FP32_SCALE`: the ffn_residual uses an fp32 output plus a typecast instead of the bf16 fused-scale add
    (-0.0026 norm bias per layer).
  - `NORM_WEIGHT=fp32`: unweighted fp32 rms_norm, then the weight multiply to bf16 (fused weight: -0.0005 to -0.0008
    per norm).
  - `RESIDUAL_STREAM=fp32`: the residual stream (embedding, h_mid, out) stays fp32 between blocks. With a bf16 stream,
    norm bias reached 0.0070 at layers 47-53 even with every step unbiased on golden inputs. The component steps
    (hooks) keep bf16.
- Determinism: `MATMUL_CORE_GRID=8x8` passes an explicit core grid to every model matmul (projections, MLP, RoPE
  rotation, the composed global SDPA; with only the linears, the C.attn.attention regression's A,B,A determinism
  check failed on 5 elements). On the default 12x10 grid, mesh chip (4, 0) gave run-to-run differences in a few
  output tiles, so test_pad_rows' resend was not bit-identical. 8x8 and 4x4 were bit-identical over 20 repeats.
- s10240 ladder (fp32 stream):
  - chunk 0 90 s, chunk 1 41 s;
  - worst layer pcc 0.9936, max rel 0.113, max bias 0.0015;
  - final hidden 0.9991, top1 0.989, top5 0.967;
  - worst state pcc 0.989.
  test_kv_layout and test_pad_rows: PASS.
- Final `bfw job check --task M.1` (750 s): PASS. Worst layer pcc 0.9936, max rel 0.113, max bias 0.0013,
  worst state pcc 0.989, final hidden 0.9991, top1 0.986, top5 0.967, 59 device ops / 0 host transfers per layer.
  All 32 C.*/S.* regressions PASS.
- Re-run: `bfw job check --task M.1`.

## K.1 (run 20261019-050348-gemma31-glx-3, attempt 1), 2026-10-19
- Serving adapter `tt/serving.py`, registered as `gemma4_31b_glx` in `models/demos/common/prefill/adapter.py`.
  - The adapter allocates the served caches and passes them into the Model (`caches=`). Capacity is
    `max(max_seq, 2*chunk)`: the sliding ring SDPA rejects a cache that holds only one chunk (`TT_FATAL is_chunked`
    on the 5120/5120 contract rung).
  - The runtime runs op by op (no trace), one slot scalar per slot. `ChunkMeta.slot_index` selects the global
    composed SDPA's prefix slot. `Model.embed_device` takes the H2D payload `[SP,1,C/SP]` uint32 and clamps PAD ids
    on device.
  - Acks per layer: D2H `outbound_socket_service_sync` when the runner passes d2h_service, else sync + sink.
  - Table: gemma4_d_p's 36 configs and walk, layer index = served layer. `kv_migration_base_address` = 0 because
    layer 0 is sliding and config 0 (global) is unset there.
  - Producer PCC: a generic `ADAPTER.read_slot_kv_and_check_pcc` hook in `prefill_producer.py`. The gemma
    implementation reads each head through the table (`read_dram_umd`, bfp8 decode) against the golden mapped to the
    served layout.
- Gate: `bfw job check --task K.1` PASS (1491 s).
  - contract_checks_failed 0, acks_early 0, layout_failures 0;
  - pcc_producer_kv global 0.9876, sliding_k 0.9895, sliding_v 0.9848;
  - all C.*/S.*/M.1/L.* regressions PASS.
  - Runner feeder: tt-d-gen engine.
- Re-run: `bfw job check --task K.1` (~25 min; launch detached and wait on the pid in the foreground).

## C.attn.attention (run 20261020-144235-gemma31-glx-3, attempt 1), 2026-10-20
- No code change. The previous gate run (19:28) hit pytest-timeout (300 s) inside the composed SDPA's P V matmul, after
  ~140 s of host-side setup (CPU reference cases) and a cold JIT cache (41.9 % hits), with the host load average ~12.
- Re-run: PASS, test call 169 s. pcc 0.999991, auto_margin_min 1.133 (vs_cpu column 1658 0.0338 / 0.0382),
  device_ops 62, host_flops_frac 0.
- Headroom: the frozen test runs under the repo's 300 s default timeout; the module's whole run (6 input cases plus
  A,B,A determinism) uses 170-250 s of it. A slower module (e.g. MATMUL_CORE_GRID smaller than 8x8) can time out.
- Re-run: `bfw job check --task C.attn.attention`.

## S.sliding.02 (swap: attn_norm + attention on device, sliding layer 0)
- No code change: the existing `tt/rms_norm.py` and `tt/attention.py` (wired through `device_component` in hooks.py) chain as is in the residual layout.
- Gate: pcc_swap_out 0.999975, swap_margin_min 1.445; attention vs CPU step rel 0.006748 (limit 0.01289), block out with device vs CPU attention rel 0.00692 (limit 0.01); device_ops attn_norm 3, attention 49, host_flops_frac 0.
- Re-run: `bfw job check --task S.sliding.02`.

## Z.1 settings audit (run 20261019-130132-gemma31-glx-3, attempt 1), 2026-10-20
- Moved the literals that pick a behaviour into `tt/settings.py`, each with its current value (no behaviour change).
  Env names are `GEMMA4_31B_GLX_<NAME>`. The 34 switches already in the table kept their names and defaults.
- New switches (default; where the literal was before):
  - `LM_HEAD_FP32_ACC` True: `tt/model.py` Model.head_cfg `fp32_dest_acc_en=True`.
  - `ATTN_AUX_FIDELITY` "HiFi4" / `ATTN_AUX_FP32_ACC` True: `tt/attention.py` and `tt/global_attention.py`
    `self.norm_cfg = _compute_config(mesh, "HiFi4", True)`. Used by the head-norm mean, the RoPE rotation and
    K-permutation matmuls, and the composed global SDPA (QK^T, softmax sum, P V).
  - `MATH_APPROX_MODE` False: `math_approx_mode=False` in `tt/attention.py::_compute_config`, `tt/mlp.py`,
    `tt/rms_norm.py`, `tt/model.py` (LM head).
  - `PACKER_L1_ACC` False: `packer_l1_acc=False` at the same four places.
  - `SDPA_EXP_APPROX` False: `exp_approx_mode=False` in the SDPAProgramConfig of `tt/attention.py` (ring_joint) and
    `tt/global_attention.py` (ring_mla).
  - `ELTWISE_FAST_APPROX` False: `fast_and_approximate_mode=False` of the fp32 rsqrt (`attention.py::_head_norm`,
    `global_attention.py::_norm`) and the exp (`global_attention.py::_composed_sdpa`).
  - `ATTN_GLOBAL_K_SPLITS` 1: `max_k_splits=1` in `tt/global_attention.py` (ring_mla mode only).
  - MAX_PRECISION gained: LM_HEAD_FP32_ACC True, ATTN_AUX_FIDELITY HiFi4, ATTN_AUX_FP32_ACC True, MATH_APPROX_MODE
    False, SDPA_EXP_APPROX False, ELTWISE_FAST_APPROX False. These equal the defaults, so component and swap tests
    run the same values as before. PACKER_L1_ACC is left out: it changes how partials are accumulated, not the
    precision ceiling.
- Left as code, not switches: ring CCL op structure (`topology=Linear`, `use_column_major_ccl`, `is_balanced=False`,
  `joint_strategy="rear"`, `is_causal`), the served KV layout and dtypes (contract-fixed: bfp8, ND-shard round robin),
  weight dtypes (spec: bfloat8_b), the rms_norm block-shard limits `_MAX_GRID_Y=8` / `_MAX_BLOCK_TILES=84` (L1
  capacity, measured), the shard geometry choices `(12, 2) if rows <= 512 else (8, 4)` (geometry), the global attn
  `MASK=-1e9`, and every DRAM memory config (the only memory choice the model makes).
- Environment reads: none in `tt/` or `bringup/hooks.py`. The whole-package lint (`bfw testing.settings_lint`) still
  reports 6 reads, all in frozen SC.1 contract test code outside Z.1's paths:
  `tests/bringup/contract/serving_helpers.py:255` (env_flag) and `tests/bringup/contract/test_runner_contract.py:46,
  85, 153, 168, 235` (PREFILL_MODEL, os.environ copies for subprocesses, PREFILL_H2D_SERVICE_ID,
  PREFILL_MIGRATION_TABLE_PATH). They are the runner's process interface, not model switches. SETTINGS_AUDIT.md says
  test code is not a switch, but the lint walks every .py of the package, tests/ included. So the Z.1 gate cannot reach
  settings_violations == 0 within Z.1's allowed paths. A framework fix is needed (the lint should skip
  `<model>/tests/`, as it already skips `ttnn/ttnn/bringup/**/tests/`), or a person-approved change to those tests.
- Re-run: `bfw job check --task Z.1`; ladder only:
  `BRINGUP_RUNG=last scripts/run_safe_pytest.sh --run-all generated/bringup/gemma4_31b_glx/tests/test_ladder.py`.
- Result: `bfw job check --task Z.1` FAIL on settings_violations = 6 (the contract-test reads above). The gate's ladder
  command, run alone on the edited code, passed (760 s) with numbers identical to the previous L.last run: worst layer
  pcc 0.992740, final hidden 0.999149, top1/top5 1.0, worst state pcc 0.998889.
- Attempt 2 (2026-10-20): no code change. I checked again that no switch literal or env read is left in `tt/` or
  `bringup/hooks.py`. `bfw job check --task Z.1` still FAILs on the same 6 contract-test env reads. Blocked on a
  framework/person decision (lint scope for `<model>/tests/`).

## Z.1 settings audit (run 20261021-050814-gemma31-glx-3, attempt 1), 2026-10-21
- Re-applied the earlier Z.1 work (run 20261019-130132-gemma31-glx-3 attempt-2.patch; the section above) unchanged.
  The settings lint no longer counts `<model>/tests/`, so the 6 contract-test env reads no longer block the gate.
- Model fix (`tt/model.py`, for the profile X.1): `State.load_prefix(layer, tensors, n)` now sets the model's
  `next_start = n`, so `Model.embed(tokens)` with no start stages the chunk right after the loaded prefix. Before, it
  staged at 0 and `model.layer()` saw a mismatched start and restaged the chunk inside the layer call (5 host uploads
  in `_stage`: positions, qpos, kv_actual, valid, kpos), counted by X.1 as `host_transfers_per_layer = 5`.
  Sequential embed is unchanged: new_state still resets next_start to 0, and the ladder's prefix rung, which calls
  `embed(tokens[s0:])` with s0 = the prefix length, now stages at s0 in embed (as the layer used to) instead of at 0.
- Re-run: `bfw job check --task Z.1`.
- Result: `bfw job check --task Z.1` PASS (ladder 817 s): settings_violations 0; pcc_final_hidden 0.999149 and
  pcc_state_min 0.998889 (same as the earlier Z.1 ladder run), top5 1.0; every C.*/S.*/M.1/L.*/K.1 regression PASS.

## Manual perf pass: trace (not pipeline-generated), 2026-10-23
- A manual performance pass by a Claude agent outside the prefill-stepwise flow (frozen tests and goldens untouched).
- `tt/model.py`: a chunk's host-side inputs (positions, slot / start / end scalars, pad keep mask, the composed global
  attention's query/key positions and partial-slab key mask, token ids) go into persistent device buffers
  (`Model._put`); `_derive` builds the ChunkMeta from them with device ops only. So the whole chunk
  (`Model.chunk_body`: meta, embedding, 60 layers) is one fixed op graph per `chunk_variant` (pad mask present; with the
  composed global attention also its prefix rows and partial-slab mask) and can be captured (`Model.capture`) and
  replayed (`Model.replay`) after restaging. Numerics are unchanged: L.s5120 metrics are identical to the pipeline's.
- `tt/serving.py`: `params.use_trace` (PREFILL_USE_TRACE=1) runs a variant's first chunk op by op and captures it, then
  replays it (payload and ack metadata copied into persistent buffers first). PREFILL_USE_TRACE=0 keeps the op-by-op
  path. D2H acks are recorded in the trace; the host sink acks every layer after a replay.
- Persistent inputs must be allocated before the trace that reads them is captured (a later allocation may overlap a
  trace's freed intermediates, which every replay writes); every input is rewritten before each replay.
- Benchmarks (`tests/perf/test_chunk_perf.py`, warm 60-layer chunk, zero KV prefix, random ids): op by op 459 ms at
  start 0 and 1163 ms at 51200; traced 423 / 1127 ms (bit-identical outputs). The X.1 wall time (3075 ms) was taken
  with the device profiler on; without it host dispatch is only ~35 ms of the chunk: the chunk is device-bound.
- `tests/perf/test_serving_trace.py`: op-by-op slot 0 vs traced slot 1 (capture pass, then replay pass): KV bytes of
  layers 0, 5, 59 identical, acks one per layer per chunk in order.

## Manual perf pass: global attention on ring_mla (not pipeline-generated), 2026-10-23
- `ATTN_GLOBAL_SDPA` default `ring_mla` (gemma4_d_p's op). The device model passes the ChunkMeta's on-device slot /
  start scalars (the op's metadata path: one program, one trace, any tile-aligned start); the ring_mla receive buffer
  is shared by the global layers. `composed` stays MAX_PRECISION (component tests) and its cost grew with the
  prefix: a chunk at 51200 took 1127 ms traced, 431 ms on ring_mla (start 0: 423 -> 388).
- `ATTN_GLOBAL_SDPA_FIDELITY` HiFi2: at HiFi4 the ring kernels overflow the 70656 B kernel config buffer by 32 B for
  a 210k-position cache (test_positions). LoFi (gemma4_d_p) put layer 59's median row-norm bias at 0.0126 (limit
  0.005, s56320 ladder, with other perf switches).
- `ATTN_GLOBAL_Q_CHUNK` 64: with q96 the HiFi2 kernels overflow the same buffer by 176 B at a 10240-position cache
  (test_contract's shape).
- Gates: s56320 min layer PCC 0.99167 (composed: 0.99148), max rel 0.1288 (0.1303), KV PCC by chunk
  0.9754-0.9938; test_contract producer KV PCC global 0.9894, sliding_k 0.9905, sliding_v 0.9864.

## Manual perf pass: projection and MLP matmuls (not pipeline-generated), 2026-10-23
- `tt/matmul.py` (`MATMUL_PROGRAM` 2d_all): an explicit 2D-multicast program config for every projection and MLP
  linear over the whole compute grid (gemma4_d_p's blocking: the widest column count dividing N, every core row,
  K in blocks of up to 16 tiles; the global layers' transposed [out, in] weights too). ttnn's default ran these on
  the 8x8 MATMUL_CORE_GRID. Output is still deterministic (traced == op by op, bit for bit).
- `MLP_FIDELITY` / `ATTN_PROJ_FIDELITY` HiFi2 (bfp8 weights). With the 2D config the s56320 ladder's max rel is
  0.129 (HiFi4 op by op: 0.130).
- Measured on 6-layer stacks and dropped (no gain or worse): MATMUL_CORE_GRID=auto for the remaining default matmuls
  (slower), CCL num_links 1/2, Ring topology, gemma4_d_p's short-slab reduce_scatter transport, async all_gather /
  reduce_scatter on persistent semaphores, ATTN_SDPA_FIDELITY HiFi2 / LoFi for the sliding SDPA, sliding SDPA q64
  and K splits 2/3, an uneven-N QKV config on all 12 columns.
- Switches added here, off by default (turned on in the next commit after their own gates): ACT_MEMORY,
  ATTN_HEAD_NORM=fp32_fused, NORM_FP32_SHARDED, NORM_WEIGHT=fp32_gamma. The modules also mark profile sections
  (`signpost("attention.s_qkv")` ...), free when nothing profiles.
- Warm 60-layer chunk, traced: start 0 387.9 -> 200.0 ms, start 51200 430.7 -> 242.7 ms.

## Manual perf pass: L1 activations and fewer fp32 norm ops (not pipeline-generated), 2026-10-23
- `ACT_MEMORY` l1: the short-lived activations inside attention and the MLP are L1 interleaved (gemma4_d_p's
  prefill_short_lived_memcfg). The ring SDPA operands (Q, caches) and the collectives' outputs stay in DRAM (ring
  joint SDPA asserts DRAM operands).
- `ATTN_HEAD_NORM` fp32_fused: the per-head q / k / v norm is one fp32 ttnn.rms_norm (HiFi4, fp32 acc) on the heads
  flattened to rows plus the fp32 weight multiply (3 ops instead of 7).
- `NORM_FP32_SHARDED`: the fp32 residual norms run block-sharded, one tile row per core row (60 cores); the
  interleaved fp32 rms_norm ran one core per tile row (5 cores for 160 rows, ~69 us).
- `PACKER_L1_ACC` True (gemma4_d_p for its matmuls).
- Warm 60-layer chunk, traced: start 0 200.0 -> 161.5 ms, start 51200 242.7 -> 204.3 ms. s56320 max rel 0.138
  (limit 0.15): little precision headroom is left. Measured but not taken: global ring_mla LoFi (layer 59 bias 0.0126
  with these switches), MLP / projection LoFi and fp32_gamma norms (see the next entries if taken).
