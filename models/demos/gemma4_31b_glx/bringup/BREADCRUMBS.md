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
