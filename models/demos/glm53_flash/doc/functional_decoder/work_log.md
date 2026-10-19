# GLM-5.3-Flash functional decoder: work log

Host: `sjc3-glx-a04u07` (32 x Blackhole Galaxy, FW 19.15.0.0). Checkout: `~/ashai/glm53/tt-metal`
main @ `8cfcaf754d3` (2026-10-19), branch `arg/glm53_flash`. Design of record: `~/ashai/GLM_5.3_DESIGN.md`.
Stage: tt-model-bringup stage 1 (functional decoder, 1x1 mesh) = design Phase 1.

## 2026-10-19

- Fresh clone with submodules; `./build_metal.sh` (Release, clang-20, Tracy on) 10m12s on 64 cores.
- Checkpoint `zai-org/GLM-5.3-Flash` (sha 46a050bec13002ad29ee795c16e61cf4fa3c4a9c, 62 shards, 328.4 GB) into the
  HF cache (`hf download`, 16 workers, ~25 min).
- Verified checkpoint key names/dtypes/shapes for layers 0, 3, 4 directly from the safetensors headers:
  raw Kimi-K3 KDA naming (`q/k/v_conv1d`, `f_a/f_b_proj`, `A_log`, `dt_bias` under `self_attn`), mHC as
  `hc_attn_{fn,base,scale}` / `hc_ffn_*` (fn bf16 [24,16384], base/scale fp32), per-expert fp8 weights with
  128x128 `weight_scale_inv`; fp8 groups = experts, shared experts, dense MLP, DSA `q_a/q_b/kv_a/o_proj`.
  transformers remaps these in `conversion_mapping.py` ("glm5_next").
- Host reference (`reference/hf_layer_reference.py`, CPU venv `~/ashai/glm53/hostref_venv`: torch cpu +
  transformers 5.20.2): builds the real `Glm5NextTextDecoderLayer` per layer from the shards (fp8 dequant on
  host), runs embedding -> layers 0..4 prefill with a `DynamicCache`, then 4 single-token decode steps, and
  dumps per-layer `x_in/x_out` (mHC streams `[B,S,4,4096]`), top-k indices and cache state to
  `bringup/references/glm53_flash/<run>/` (git-excluded locally; not committed).
  - `T256`: layers 0-4, 256 tokens + 4 decode steps: done (layer 3 prefill 38.6 s, layer 4 27.7 s on CPU).
  - `T4097`: layers 0-4, 4097 tokens (sparse DSA path, > 2048) + 4 decode steps: running.
- Note: the config.json names DSA layers `deepseek_sparse_attention`; the modeling code keys its mask mapping
  on `indexed_attention`, so the layer-level harness normalizes `layer_types` (same as from_pretrained).
- `tt/config.py`: frozen config dataclass loaded from config.json (45 layers: 34 KDA, DSA at 3+4k, 42 MoE).
- Build done (10m12s); `create_venv.sh` (uv) + `transformers==5.20.2` + `tt-perf-report` added with
  `uv pip install --python python_env/bin/python`. Single-chip smoke (TT_VISIBLE_DEVICES=0, MeshShape(1,1)): OK.
- `reference/check_kda_reference.py`: the in-tree Kimi-K3 KDA torch reference == HF GLM KDA with real layer-0
  weights (prefill PCC 0.99999, state 0.99999, decode 0.99999) -> ttKDA prefill kernels + state contract reused.
- `tt/kda.py` + `tests/test_kda.py` (1x1 mesh, layer 0, real weights): prefill T=96 PCC 0.99994 / T=1000 0.99995,
  recurrent state 0.9999, conv history 0.99999; 3 decode steps from the device state PCC 0.9999 each. Commit f9ed9df261e.
- `tt/moe.py` (router = moe_grouped_topk single-group; experts = sparse_matmul per 32-token block with a per-expert
  union mask; shared expert = clamped SwiGLU). sparse_matmul needs a rectangular grid whose core count divides the
  output tile count (TT_FATAL "num_cores_with_work must equal in0_mcast_receiver_num_cores"): `pick_sparse_grid`.
  First run (layer 4, T=32, realistic MLP input from the T256 dump): output PCC 0.9964, but top-8 sets identical to
  HF on only 16/32 tokens (dense routing-weight PCC 0.985) -> quantifying whether that is near-tie noise.
- Root cause of the router mismatch: the HF layer builders cast the whole module to bf16 before restoring the
  strict-fp32 tensors (router bias ~5.5 resolves 0.03 in bf16) -> fixed (`cast_like_from_pretrained`), dumps
  regenerated. Separately, `moe_grouped_topk` ranks with bf16 resolution and swaps the 8th expert on ~6/32 real
  tokens (HF biased-score gaps 0.003-0.004) -> router is now an exact fp32 composite (iterative masked argmax).
  Result: top-8 identical 32/32 (T=32) and 255/256 (T=256); MoE output PCC 0.99993 / 0.99995.
- `tt/dsa.py`: dense causal SDPA prefill on materialized K/V (exact for context <= 2048), latent written with
  paged_fill_cache; decode = paged_update_cache (height-sharded [1,B,32,512] input) + absorbed
  paged_flash_multi_latent_attention_decode. L1 trap: the decode op's c_19 staging CB is sized
  (cores_per_user-1)*(out+2) fp32 tiles (1.6 MB at 12 cores/user with the 512-wide latent) -> 4 cores/user,
  k_chunk 32. Layer 3 vs HF: prefill PCC 0.99964 (T=96) / 0.99961 (T=1000), latent cache 0.99997, decode 0.9991-0.9999.
- Commit 25c8b1100f8: moe, dsa, decoder_layer, functional_decoder (stage contract), tests.
- Full-layer tests (`tests/test_decoder_layer.py`, HF dumps T256): layer 0 prefill 0.99984 / decode 0.9999;
  layer 3 0.99992 / 0.99996; layer 4 0.99997 / 0.99999; non-aligned S=200 vs HF built live: layer 0 0.99980,
  layer 3 0.99993. Commit 8e2723a6cea.
- Traced decode (`tests/test_decode_trace.py`): KDA decode writes its carries back in place (ttnn.copy into the
  persistent batched state; prefill writes user slots), the DSA cache is already in place; capture once on a
  dummy token, replay the 4 reference steps updating only the input/position tensors: replay PCC identical to
  eager for layers 0, 3, 4 (0.9999).
