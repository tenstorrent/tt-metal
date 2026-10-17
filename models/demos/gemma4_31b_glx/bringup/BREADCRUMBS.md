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
