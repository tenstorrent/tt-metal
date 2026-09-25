# Mistral-Medium-3.5-128B prefill on the Blackhole Galaxy (SP=8 x TP=4)

TTNN prefill for the language model of `mistralai/Mistral-Medium-3.5-128B` (Ministral3 architecture;
the vision tower is out of scope) on one 32-chip Blackhole Galaxy, mesh 8x4: sequence parallel over the
8 rows, tensor parallel over the 4 columns. It follows `MODEL_BRINGUP_RECIPE.md` through P2: one-shot
and chunked prefill with the real FP8 checkpoint, validated per layer against a CPU golden KV trace.
Serving, KV migration, perf tuning and decode are out of scope.

**Result** (full depth 88 layers, full width 12288, real weights, 10240-token golden trace, linear
fabric): every layer's K and V clear the spec's `pcc_lower_bound` 0.85 in both modes.

| mode | chunks | worst K | worst V | layers below `pcc_target` 0.99 (K / V) |
|---|---|---|---|---|
| one-shot (`PREFILL_CHUNKED=0`) | 1 x 10240 | 0.9868 (layer 85) | 0.9360 (layer 71) | 11 / 53 |
| chunked (`PREFILL_CHUNKED=1`) | 2 x 5120 | 0.9844 (layer 85) | 0.9256 (layer 71) | 20 / 53 |

The values under `pcc_target` are explained in [Why deep layers sit below 0.99](#why-deep-layers-sit-below-099):
a CPU fp32 forward of the same weights, no device involved, is itself only V 0.964 at layer 71 against
the bf16 golden. All correctness knobs were tried; see [Correctness knobs](#correctness-knobs).

## Model and spec

From `configs/config.json` (vendored `text_config`, asserted by `config.py::MistralMediumConfig.from_json`):
88 decoder layers, hidden 12288, GQA 96 query / 8 KV heads, head_dim 128, SiLU-gated MLP 28672, vocab
131072, RMSNorm eps 1e-5, untied embeddings, no sliding window. RoPE is YaRN (theta 1e6, factor 64,
original 4096, beta_fast 4, beta_slow 1, attention factor 0.1 ln 64 + 1 = 1.4159 folded into cos/sin); the
Ministral3 llama-4 query scale has beta 0 and is the identity (asserted). Weights are `float8_e4m3fn` with
one rank-0 `weight_scale_inv` per tensor; embedding and lm_head are bf16.

Binding spec (`PREFILL_SPEC`, vendored as `configs/prefill_spec.json`): `bh_galaxy`, SP 8 x TP 4,
`chunk_size` 5120, `max_seq_len` 262144, activations bf16, KV cache bf8, all weights bf8, PCC target 0.99
and lower bound 0.85. Weights: `/mnt/models/mistralai/Mistral-Medium-3.5-128B` (shared store, step 0 of
the recipe). Golden trace: `.../golden/synthetic_10240` (10240 tokens, 88 layers, bf16, K post-RoPE HF
half-split, V raw).

## Architecture

```
tokens [S] --SP shard--> Embedding (1D: table hidden-sharded over TP)      -> residual [1,1,S/8,12288/4]
  x88 DecoderLayer:
    RMSNorm (distributed: fp32 stats, TP all-gather)                        -> [1,1,S/8,12288]
    Attention: fused QKV (24 q + 2 kv heads/chip) -> YaRN RoPE (indexed) -> KV-cache write
               -> ring-joint SDPA over SP (live K/V, or the cached prefix) -> o_proj -> TP reduce-scatter
    + residual; RMSNorm; MLP: gate/up col-parallel, SiLU gate, down row-parallel -> TP reduce-scatter; + residual
  final RMSNorm -> LM head (vocab col-parallel, 32768/chip)                (headless for the KV acceptance)
```

| file | role |
|---|---|
| `config.py` | constants class, spec loader, dataformat resolution |
| `reference/model.py` | torch-only bf16 reference (provenance lines to transformers 5.12.1) |
| `reference/golden.py` | whole-model golden cache (`ReferenceCacheKey` + shared save/load helpers) |
| `reference/checkpoint.py` | safetensors walk + per-tensor fp8 dequant |
| `tt/ccl.py`, `tt/fabric.py` | `MeshConfig`, `CCLManager`, fabric/topology knob |
| `tt/rms_norm.py`, `tt/mlp.py`, `tt/attention.py`, `tt/rope.py`, `tt/kv_cache.py` | decoder blocks |
| `tt/layer.py`, `tt/model.py`, `tt/embedding.py`, `tt/lm_head.py` | composition |
| `tt/weights.py` | weight sources (random / checkpoint / tilized cache), cache location + completion marker |
| `tt/runtime.py` | `compile` / `make_chunk_input` / `prefill_chunk` (asserts on chunk ranges) / `prefill` |
| `tt/kv_validation.py` | per-layer KV PCC vs the golden trace |
| `tt/precision.py` | correctness knobs (`MISTRAL_PRECISION`), defaults as measured below |

Layout decisions:

* **Residual** is SP x TP sharded (`[1, 1, S/8, 3072]` per chip) through every layer, M3's sharded scheme.
  Norms return full width for the column-parallel projections; attention and MLP close with a
  reduce-scatter straight back into that layout. The 1D embedding produces it directly (no CCL).
* **KV cache** (recipe section 5): 2 tensors (K post-RoPE, V raw), per chip
  `[users * 88, 2, max_seq_len / 8, 128]` bf8, slot `user * 88 + layer`, DRAM ND-shard `[1, 1, 32, 128]`
  round-robin over `get_num_dram_banks` banks, block-cyclic over SP with period = chunk size, written by
  `update_padded_kv_cache`, read by ring-joint SDPA. 8 KV heads over TP=4 put **2** heads on each chip,
  where the canonical layout carries 1 (logged as `model_quirk`). K is stored in the Meta interleaved
  RoPE layout (q/k rows permuted at load); `tt/kv_validation.py` permutes the golden's head_dim the same way.
* **RoPE**: whole-cache cos/sin tables (fp32 angles, rounded to bf16 once, matching HF/golden bit-exactly),
  block-cyclic reordered and SP-sharded once; `rotary_embedding_indexed` picks each chunk's rows on device.
* **Data formats** (spec): attention and MLP weights bf8, lm_head bf8, KV cache bf8, activations bf16.
  The embedding table stays bf16: `ttnn.embedding` only takes a bf16 ROW_MAJOR table (logged `spec_gap`),
  and bf16 is the checkpoint's own dtype, so the lookup is exact. Compute: HiFi4 + fp32 accumulation for
  every matmul, norm and RoPE; ring-joint SDPA with fp32 accumulation on the no-cache path, without it on
  the cache-read path (the op requires that).
* **DRAM**: 30.1 GiB of bf8 layer weights + 0.75 GiB embedding per chip of 31.83 GiB; 0.111 GiB/bank
  (0.89 GiB/chip) remains after the build at a 10240-token cache.

## Exploration (stage E): what was borrowed

Candidates registered in `common/prefill/adapter.py::ADAPTER_PATHS`, gated on the target mesh: `minimax_m3`
(8x4 SP8/TP4, BH Galaxy; GQA + sparse MSA, MoE) and `gemma4_d_p` (8x4 CP8/TP4, dense GQA with sliding and
packed global caches) ran on this mesh; `gpt_oss_d_p` (4x8), the DeepSeek/Kimi/GLM/Mistral-Small-4 MLA or
DSA packages and `llama_3p1_8b_d_p` (not registered) were math sources at most. Ranked by attention family
(GQA, 2 caches), MLP density (dense) and quantization (per-tensor fp8), minimax_m3's dense path is the
structural source; gemma4_d_p confirmed ring-joint SDPA with several KV heads per chip.

| part | source | measured envelope (hidden, head_dim, chunk, sp x tp) | here |
|---|---|---|---|
| weight loading | fresh | - | `reference/checkpoint.py` |
| dequant | per-tensor scheme of `deepseek_v3_d_p/utils/test_utils.py` (oracle in the loader test) | Mistral-Small-4 dense fp8 | fresh scale application |
| norm | `minimax_m3/tt/rms_norm.py` + `residual.py` | 6144, 128, 5120, 8x4 | distributed form (single-pass overflows L1 at 12288) |
| embedding | `minimax_m3/tt/parallel_embedding.py` | 6144, 128, 5120, 8x4 | 1D default, 2D kept |
| MLP | `minimax_m3/tt/dense_mlp.py` (structure) | 6144 / 12288 ffn, 5120, 8x4 | SiLU gate written fresh |
| attention | `minimax_m3/tt/attention/*` dense ring-joint path | 6144, 128 (16q/1kv per chip), 5120, 8x4 | 24q/2kv per chip |
| RoPE | M3 indexed-rope plumbing | 128 (rotary 64), 5120, 8x4 | YaRN values fresh from transformers 5.12.1 |
| KV cache | `minimax_m3/tt/attention/kv_cache.py` + `dense_sp.py` | 128, 1 kv/chip, 5120, 8x4 | 2 kv/chip |
| runtime | `minimax_m3/tt/tt_prefill_runtime.py` | 6144, 128, 5120, 8x4 | same contract |
| lm_head | M3 `tt/model.py` lm_head; `gemma4` lm-head test pattern | 6144, vocab 200064, 8x4 | vocab 32768/chip |
| CCL | `minimax_m3/tt/ccl.py`, `config.py::MeshConfig` | 6144, 128, 5120, 8x4 | local copy |

Shared code is imported: `get_num_dram_banks` (migration.py), `ReferenceCacheKey` /
`save_reference_cache` / `load_reference_cache` (deepseek transformer_helpers), `blockcyclic_positions` /
`block_cyclic_reorder` (models/common/utils.py), `KvCaches` (common/prefill/adapter.py). No torch CPU
fallback anywhere. Records are in `bringup_log.jsonl` (`source` events).

## PCC status

Every test asserts the spec's `pcc_lower_bound` (0.85) and flags values under `pcc_target` (0.99).
Random-weight module tests run on the 8x4 mesh at full width; full sequence 10240 unless noted.

| component (test) | measured PCC | note |
|---|---|---|
| RMSNorm, distributed (`test_norm_vs_ref`) | 0.999996 | |
| final norm instance (`test_final_norm_vs_ref`) | 0.999996 | |
| SiLU gate (`test_swiglu_vs_ref`) | 0.999996 | |
| dense MLP (`test_dense_mlp_vs_ref`) | 0.999899 | |
| indexed YaRN RoPE (`test_rope_vs_ref`, 3 geometries) | 0.999999 | |
| ring-joint SDPA, live K/V (`test_ring_joint_sp_vs_ref`, 10240 / 5120) | 0.999959 / 0.999966 | fp32 acc |
| ring-joint SDPA cache read (`test_ring_joint_cache_read_sp_vs_ref`) | 0.999646 / 0.999640 | chunk 1 of 2 / 2 of 4 |
| KV cache write/read, 2 kv/chip (`test_kv_cache_gqa_sp_vs_ref`) | 0.999975 | bf8 |
| KV write through Attention (`test_kv_cache_write_vs_ref`) | K 0.999943, V 0.999944 | |
| attention one-shot (`test_attention_vs_ref`) | 0.999621 | |
| attention 2 chunks vs one-shot (`test_attention_chunked_vs_ref`) | 0.998905 (chunk-1 vs ref 0.998913) | |
| decoder layer (`test_decoder_layer_vs_ref`) | 0.999951, K/V 0.99994 | |
| embedding 1D / 2D (`test_parallel_embedding_vs_ref`) | 1.0 (bit-exact) | 2D at 5120 |
| LM head (`test_lm_head_vs_ref`) | 0.999969 | |
| whole model, **reduced depth 4L** (`test_model_sp_vs_ref`) | residual 0.999460, logits 0.999255, K/V >= 0.99948 | random weights |
| runtime, **reduced depth 2L** (`test_runtime_contract`) | chunked vs one-shot KV >= 0.99966 | ragged 2nd chunk |
| real layer 0, **reduced depth 1L** (`test_real_weights_layer`) | out 0.999956, K 0.999973, V 0.999930 vs golden | real weights |
| **full model, one-shot** (`test_prefill_acceptance`) | worst K 0.9868, V 0.9360 | below target, see below |
| **full model, chunked** (`test_prefill_acceptance`) | worst K 0.9844, V 0.9256 | below target, see below |

Host-side references: the torch reference equals the HF Ministral3 modules on real weights (PCC 1.0 for
layers 0/43/87, embedding, final norm and lm_head bit-exact); at full depth over the whole 10240-token
prompt in bf16 it reproduces **every layer** of the golden trace exactly (PCC 1.000000, 88 layers);
FP8 dequant is bit-identical to the shared per-tensor helper.

Full-depth per-layer KV PCC vs golden (every 8th layer; real weights, linear fabric):

| layer | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 | 71 | 80 | 86 | 87 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one-shot K | 1.0000 | 0.9999 | 0.9999 | 0.9999 | 0.9989 | 0.9922 | 0.9922 | 0.9916 | 0.9894 | 0.9919 | 0.9903 | 0.9877 | 0.9969 |
| one-shot V | 0.9999 | 0.9997 | 0.9995 | 0.9991 | 0.9959 | 0.9722 | 0.9596 | 0.9487 | 0.9526 | 0.9360 | 0.9585 | 0.9676 | 0.9898 |
| chunked V | 0.9999 | 0.9996 | 0.9994 | 0.9989 | 0.9947 | 0.9667 | 0.9516 | 0.9394 | 0.9445 | 0.9256 | 0.9515 | 0.9617 | 0.9874 |
| CPU fp32 V (no device) | 1.0000 | 0.9999 | 0.9999 | 0.9997 | 0.9986 | 0.9870 | 0.9784 | 0.9720 | 0.9739 | 0.9636 | 0.9759 | 0.9812 | 0.9943 |

### Why deep layers sit below 0.99

The decay is not a random walk; it starts sharply at layers 28-40 and plateaus. Measured with
`scripts/golden_precision_ceiling.py` (host only, real weights, full 10240-token prompt, all 88 layers):

* the bf16 torch reference reproduces the golden trace **exactly** (PCC 1.000000 on every layer's K and V):
  the golden is one specific bf16 rounding path of HF's math;
* the **same model in fp32** (same weights, no device, no bf8) drifts off that path from layer ~28 on: worst
  K 0.9926, worst V **0.9636 at layer 71**, 47 layers with V below 0.99; its bf16-vs-fp32 hidden-state PCC
  falls to 0.984. The model amplifies rounding differences in these layers, so no computation that is not
  bit-identical to the CPU bf16 run can reach 0.99 on V against this golden;
* on top of that ceiling, the device adds the spec's bf8 weights and on-device accumulation. Attribution on
  40 layers (worst V over layers 0-39): CPU fp32 0.9912; device bf16 weights 0.9849 (diagnostic only, the
  spec binds bf8); device bf8 0.9807. The device's worst layer is the fp32 run's worst layer (71).

### Correctness knobs

Measured with `tests/diag_kv_depth.py` on the first 48 real layers, one-shot 10240 (**reduced depth**,
attribution only), worst K / V over those layers:

| knob (`MISTRAL_PRECISION`) | worst K | worst V | kept |
|---|---|---|---|
| M3 settings (SDPA bf16 accumulation) | 0.989244 | 0.952879 | no |
| SDPA fp32 accumulation, no-cache path | 0.991586 | 0.962560 | **yes (default)** |
| + fp32 residual stream (`residual_fp32`) | 0.981121 | 0.921980 | no (moves off the bf16 golden path) |
| + fp32 o_proj/down_proj partials (`proj_out_fp32`) | 0.991701 | 0.962639 | no (+0.0001, 2x CCL bytes) |
| + no packer L1 accumulation (`no_packer_l1_acc`) | 0.991473 | 0.961987 | no |
| all of the above | 0.981140 | 0.922000 | no |
| SDPA q/k chunks 64/256 (128/1024, 256/512 overflow L1) | 0.991489 | 0.962103 | no |
| bf16 attention + MLP weights (40 layers; spec-forbidden) | 0.996093 | 0.984896 | n/a |

Already at their most precise setting: HiFi4, fp32 accumulation in every matmul, norm (fp32 stats) and
RoPE, exact-exp SDPA, bf16 activations, exact embedding. The cache-read ring path cannot take fp32
accumulation (`TT_FATAL: !kv_pad_rotation_enabled || use_streaming_compute`), which is the chunked-vs-one-shot
gap (max per-layer delta V 0.010, same per-layer profile, worst layer 71 in both).

## Topology

All PCC numbers above were measured on the **linear** fabric: plain `single_bh_galaxy_mesh_graph_descriptor`,
`FABRIC_1D`, `Topology.Linear` (the default, maps on any galaxy). This pod's links also map the torus-xy
descriptor: the mesh smoke test passes with `MISTRAL_FABRIC=ring` (`FABRIC_1D_RING`, `Topology.Ring` for the
all-gather / reduce-scatter CCLs; ring-joint SDPA always runs Linear). Ring is a perf lever only and was not
used for any PCC measurement. The descriptor is exported by the package `conftest.py` before the cluster
initialises.

## Reduced runs

Diagnostics only, labelled wherever quoted: `test_model_sp_vs_ref` (4 of 88 layers, random weights),
`test_runtime_contract` (2 layers, random), `test_real_weights_layer` (layer 0, real), the knob sweep of
`tests/diag_kv_depth.py` (48 / 40 layers, real). The graded result is the full-depth acceptance test.

## Running

```bash
cd $TT_METAL_HOME   # the tt-metal checkout this package lives in
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib:$LD_LIBRARY_PATH
export PREFILL_SPEC=<prepared spec.json>
export PREFILL_HF_MODEL=/mnt/models/mistralai/Mistral-Medium-3.5-128B HF_MODEL=$PREFILL_HF_MODEL
export PREFILL_TRACE_DIR=$PREFILL_HF_MODEL/golden/synthetic_10240

# package suite (host + device, random and real weights; ~13 min)
scripts/run_safe_pytest.sh models/demos/mistral_medium_3_5_128b/tests \
  --ignore=models/demos/mistral_medium_3_5_128b/tests/test_prefill_acceptance.py

# acceptance, full depth: one-shot and 2 x 5120 chunks
for c in 0 1; do
  PREFILL_CHUNKED=$c PREFILL_ACCEPTANCE_OUT=/tmp/acceptance_$c.json \
    scripts/run_safe_pytest.sh models/demos/mistral_medium_3_5_128b/tests/test_prefill_acceptance.py::test_prefill_kv
done

# diagnostics (not collected by the suite)
MISTRAL_DIAG_LAYERS=48 MISTRAL_PRECISION=residual_fp32 \
  scripts/run_safe_pytest.sh models/demos/mistral_medium_3_5_128b/tests/diag_kv_depth.py
python models/demos/mistral_medium_3_5_128b/scripts/golden_precision_ceiling.py \
  --checkpoint $PREFILL_HF_MODEL --trace $PREFILL_TRACE_DIR --out /tmp/ceiling.json   # ~45 min, host
```

Knobs: `MISTRAL_FABRIC=linear|ring`; `MISTRAL_PRECISION` (see `tt/precision.py`); `MISTRAL_TT_CACHE` (tilized
weight cache root, default `~/.cache/ttnn/models/mistral_medium_3_5_128b/<checkpoint>_<index-hash>_mesh8x4_v1`);
`MISTRAL_FORCE_LOAD_WEIGHTS=1`; `MISTRAL_REF_CACHE` / `MISTRAL_REF_CACHE_REQUIRED=1` (host golden cache,
default `/tmp/mistral_medium_3_5_128b_transformer_ref_cache`); `MISTRAL_EMBED_2D_DIAG=1`.

Weight load: first build from the checkpoint takes ~14 min (fp8 read + dequant + bf8 tilize, ~9 s/layer)
and writes a 133 GB tilized cache on local disk plus a `COMPLETE_*` marker; later builds load the cache in
~2.5 min. Prefill itself is ~5 s for 10240 tokens (not tuned).

## Known gaps

* **KV PCC under `pcc_target`** on deep layers (worst V 0.936 one-shot / 0.926 chunked); bounded by the
  golden's own precision ceiling (CPU fp32 V 0.964), see above. KV PCC is a proxy; top-1 / logits
  agreement with HF at full depth was not measured (the trace carries no logits).
* **262144-token capacity does not fit**: the spec's `max_seq_len` needs ~1.6 GiB/chip of KV cache + ring
  gather buffers, while 0.89 GiB/chip is left after the bf8 weights and 1D embedding (estimate from the
  measured free DRAM; not run). Validated capacity is the 10240-token trace. The 2D embedding would free
  0.66 GiB/chip; bf4 MLP weights or more chips would be needed beyond that.
* **Chunked is slightly below one-shot** (max per-layer delta V 0.010): the ring-joint cache-read op rejects
  fp32 accumulation (ttnn gap).
* **2D (vocab-on-SP) embedding** is not the default. It is bit-exact at 5120 and 10240 tokens on this
  tt-metal (e77d8222); run 1 of this bring-up saw its SP reduce-scatter corrupt rows 1088..1099 per shard
  at 10240 and once hang the mesh, so the 10240 case stays opt-in (`MISTRAL_EMBED_2D_DIAG=1`).
* Ring / torus fabric measured by the smoke test only; no PCC or perf numbers on it.
* Not in scope: serving adapter / manifest, KV migration, decode, trace capture, perf tuning.
