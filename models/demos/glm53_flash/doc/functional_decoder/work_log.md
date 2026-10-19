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
