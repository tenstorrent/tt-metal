# allenai/Olmo-3.1-32B-Instruct on Tenstorrent Blackhole

Text-only decoder (32.2B dense, 64 layers, hidden 5120, 40 Q / 8 KV heads, head_dim 128, MLP 27648,
vocab 100278, 65k context) served through `models/tt_transformers` on p150 (1 chip), p300 (2 chips, TP2)
and p300x2 (4 chips, TP4), packaged as a tt-model-manager CONTAINER (v5.1) `kind: vllm-plugin`.

## Architecture deltas vs. Llama and where they live

| OLMo-3 feature | tt_transformers implementation |
|---|---|
| post-norm residual order (`post_attention_layernorm` on the attention output, `post_feedforward_layernorm` on the MLP output, no `input_layernorm`) | `ModelArgs.use_post_norm` (set for `model_type in (olmo2, olmo3)` in `tt/model_config.py::_set_model_specific_params`), `tt/decoder.py` sandwich branch |
| full-width QK-norm (`q_norm.weight [5120]`, `k_norm.weight [1024]`, applied before the head split) | `ModelArgs.qk_norm_full_width` → `Attention._apply_qk_norm_full_width` (`tt/attention.py`): RMSNorm over the q / k column blocks of the fused QKV activation, distributed statistics under TP; gammas permuted per head in `tt/load_checkpoints.py::reverse_permute_1d_per_head` |
| 3 sliding (4096) + 1 full attention layers (`layer_types`) | existing `layer_types` / `sliding_window` support (`tt/attention.py`), `HybridAttentionForCausalLM` on the vLLM side |
| YaRN (factor 8, original 8192, attention_factor 1.2079) on the full-attention layers ONLY, plain θ=5e5 rope on sliding layers (HF #46911, OLMo 3 report) | global `rope_setup` (scaled) + `rope_local_setup` (unscaled `rope_theta_local`) in `tt/model.py`; `tt/decoder.py` picks per layer |
| tool calls as `<function_calls>name(kwargs)</function_calls>` | vLLM ≥ 0.26 `--tool-call-parser olmo3` (and `--reasoning-parser olmo3` for Think variants) |

## Precision

`model_params/Olmo-3.1-32B-Instruct/performance_decoder_config.json`: MLP FF1/FF3 + FF2 in bfp4 (LoFi), attention
and KV cache bfp8 — the configuration that fits one 32 GB P150 (≈21 GB of weights).
`accuracy_decoder_config.json`: FF2 back to bfp8 for the multi-chip profiles.

## Running

```bash
export HF_MODEL=allenai/Olmo-3.1-32B-Instruct MESH_DEVICE=P150 TT_METAL_VISIBLE_DEVICES=0
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" --token_accuracy True
```
vLLM: `EXTRA_MODELS_DIR=models/autoports/allenai_olmo_3_1_32b_instruct/vllm_bundle vllm serve allenai/Olmo-3.1-32B-Instruct --block-size 64 --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser olmo3`.
tt-model-manager: `tt-model package --container models/autoports/allenai_olmo_3_1_32b_instruct/tt-model.yaml`.

Bring-up evidence: `doc/` (per stage), pipeline: `~/olmo3-bringup` on the author's QB2.

## Known limitations (bring-up 2026-09-09)

- Decode is validated at `max_batch_size` 1 and 32 (token accuracy 95.0% / 99.8% top-1/top-5 vs HF at both); a model
  built for max batch 8 degrades over decode steps on a single P150 (root cause not yet isolated — not the SDPA op,
  not the QK-norm). Keep `max_num_seqs: 32` in the manifest profiles.
- One P150: 16k single-chunk prefill (sliding-window layers cannot use chunked prefill); 32k does not fit next to the
  weights. KV pool capped at 48k tokens (`Olmo3ForCausalLM.get_max_tokens_all_users`). Host-side sampling: the
  device-sampling buffers make the decode MLP circular buffers clash with L1 on one chip.
- The decode SDPA runs with the op's auto program config and an interleaved Q: tt_transformers' explicit (8,8) config
  hangs (auto chunks) or mis-attends past ~500 tokens (explicit chunks) for 40/8 heads on Blackhole.
- p300 (a direct 1x2 mesh) can only be booted on a physical P300; on a QuietBox-2 the P300 shape was validated as a
  (1,2) submesh of the 1x4 parent (`TT_MESH_PARENT_SHAPE=1x4`).
