# LongCat-Video — end-to-end TTNN pipeline (`meituan-longcat/LongCat-Video`)

Real **text-to-video latent-diffusion** pipeline over the graduated TTNN stubs, placed on a
**4-chip mesh (TP=4 × DP=1, FABRIC_1D)**.

LongCat-Video is a diffusers-style *multi-component* pipeline — there is no `model.generate()`
and no HF pipeline registry entry. The real forward is three stages, each with its own Source-A
golden:

| Stage (`PIPELINE_STAGES`) | TTNN forward | Golden (Source A) | Gate | Measured e2e PCC |
|---|---|---|---|---|
| `text_encode` | prompt → T5Tokenizer → UMT5-xxl encoder → caption embeds | `transformers.UMT5EncoderModel.last_hidden_state` | ≥ 0.99 | **0.9996** |
| `denoise` | (latent, timestep, embeds) → velocity/noise prediction | vendored `LongCatVideoTransformer3DModel` | ≥ 0.95 | **0.9885** |
| `vae_decode` | video ↔ latent (encode.mode → decode) | `diffusers.AutoencoderKLWan` | ≥ 0.95 | **0.9910** |

Each stage is fed the **previous stage's real TT output** (no reference tensor is injected at any
joint). `run_t2v` chains all three over a capped diffusion horizon (flow-matching Euler; no shipped
scheduler) to emit a real decoded video latent — the behavioral proof.

## One shared pipeline

The chained forward lives once in **`tt/pipeline.py`** (`build_pipeline(device, model=None, **kwargs)`
→ resident `LongCatVideoPipeline`). Both `demo/` and `tests/e2e/` import and call it, so a green
test guarantees a working demo.

## Run it

```bash
# cheap 485MB VAE stage (fastest full on-device PCC gate)
./python_env/bin/python -m models.demos.hf_eager.longcat_video.demo.demo_vae_decode
# UMT5 text encode (~26GB)
./python_env/bin/python -m models.demos.hf_eager.longcat_video.demo.demo_text_encode --prompt "A cat playing piano"
# one DiT denoise step (~54GB, 48 blocks)
./python_env/bin/python -m models.demos.hf_eager.longcat_video.demo.demo_denoise --size 32
# full text-to-video behavioral chain
./python_env/bin/python -m models.demos.hf_eager.longcat_video.demo.demo_t2v --prompt "A cat playing piano" --steps 4

# gates
./python_env/bin/python -m pytest models/demos/hf_eager/longcat_video/tests/e2e/test_pipeline_e2e.py -s -k "gate1 or text_encode or vae"
./python_env/bin/python -m pytest models/demos/hf_eager/longcat_video/tests/e2e/test_pipeline_e2e.py -s -k "denoise or t2v"   # heavy
```

## Gates

- **Gate 1 — native ttnn.** Static scan of every routed stub: no torch host-compute op on
  activations in the hot path (constant RoPE/timestep tables via `torch.arange`/`torch.einsum` are
  permitted prep, per the trace contract), and `_runtime_fallbacks.json` is empty. The pipeline
  contains `ShardTensorToMesh` + a CCL collective (`all_reduce`/`all_gather`) → genuine TP=4.
- **Gate 2 — every graduated module invoked.** See coverage below.
- **Gate 3 — final-stage PCC ≥ 0.95** (text_encode ≥ 0.99), printed as `e2e PCC=<x>` before every
  assert.

## Gate-2 coverage (28 graduated NEW components)

The bring-up graduated the **same** stage computation at several **overlapping** granularities.
The pipeline routes the leaf stubs onto the real forward path wherever they compose cleanly:

- **DiT — 10/10 on the real forward.** The composite is fully decomposed:
  `patch_embed3_d` + `timestep_embedder` + `caption_embedder` + 48× `long_cat_single_stream_block`
  (each now calls `feed_forward_swi_g_l_u`, `r_m_s_norm_f_p32`, `layer_norm_f_p32`,
  `rotary_positional_embedding`) + `final_layer_f_p32`. Every DiT stub's output feeds the next op.
- **UMT5 — 6/6 on the real forward.** The encoder runs both as the composite
  (`u_m_t5_encoder_model` / `u_m_t5_stack`, numerically identical) and as a decomposed real chain
  (ttnn embedding + 24× `u_m_t5_block` + `u_m_t5_layer_norm`); `u_m_t5_layer_f_f` /
  `u_m_t5_dense_gated_act_dense` (the block's FF sub-region, nested inside `u_m_t5_block`) run as
  real sub-forwards on the real per-layer hidden state.
- **VAE — 3/12 on the real forward** (`autoencoder_k_l_wan` full round-trip, `wan_encoder3d`
  encoder half, `wan_decoder3d` decoder half). **Honest limitation:** the 9 remaining Wan
  sub-block stubs (`wan_residual_block`, `wan_mid_block`, `wan_up_block`, `wan_resample`,
  `wan_upsample`, `wan_attention_block`, `wan_causal_conv3d`, `wan_r_m_s`, `zero_pad2d`) are
  graduated ports of *internal sub-regions* that the composite computes via the shared `tt_dit`
  Wan-VAE library. They do **not** chain into a golden-matching VAE standalone — verified: driven
  outside their per-component harness they produce mismatched temporal/channel shapes
  (`wan_encoder3d` alone emits T=2 vs the golden encoder's T=5, and the quant/post-quant convs live
  only on the outer `AutoencoderKLWan`). They remain validated by their per-component PCC tests
  under `tests/pcc/` (Gate 1 native + PCC ≥ 0.99). Only the composite reproduces the full VAE golden
  on-device, so forcing them into the real forward would break correctness — we did not do that.

So **19/28** graduated stubs are genuinely on the real forward path; the other 9 are the VAE's
internal sub-region ports, covered per-component. This split is a direct consequence of the
bring-up over-decomposing the VAE.

## Precision note

Chaining 48 DiT blocks in bf16 lands just under the 0.95 gate; the DiT block / FFN / final-layer /
caption matmuls therefore use `MathFidelity.HiFi4` + `fp32_dest_acc_en` (a pure precision knob — no
sharding change). UMT5 uses an fp32 residual stream to clear its layer-20+ activation outlier.

## Trace + 2CQ contract (Command 3)

`tt/pipeline.py` exposes the full contract surface: `PIPELINE_STAGES`, per-stage
`<stage>_trace_setup/_trace_step/_write_inputs` hooks (variable seq/spatial dim pinned, constants
pre-uploaded outside the trace), `build_pipeline(...)` (returns the resident object — does not run
it), `trace_capture_selftest(device)` and `host_op_selftest()`.

**Honest result:** the graduated composite stubs do **in-forward host round-trips** (`ttnn.to_torch`
for DiT patch extraction/unpatchify, VAE encoder→decoder channel/HW re-padding, UMT5 id re-cast), so
they are **not host-free**. `trace_capture_selftest` therefore reports every stage as a **single-CQ
fallback** with the reason (a `begin_trace_capture` around an in-forward readback would deadlock —
and a readback is not an aten op, so `host_op_selftest` reports host ops too). This is contract-
compliant behavior (degrade + print the fallback, never silently drop). Making the stages
trace-capturable host-free requires rewriting the stubs to keep the whole forward on device (no
in-forward readback) — a follow-up beyond this correctness-first bring-up.
