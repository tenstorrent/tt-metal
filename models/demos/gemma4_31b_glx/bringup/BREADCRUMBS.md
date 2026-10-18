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
