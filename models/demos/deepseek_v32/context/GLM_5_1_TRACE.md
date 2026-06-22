# GLM-5.1 — MLA/DSA trace comparison (how to run)

GLM-5.1 (`zai-org/GLM-5.1`, HF `model_type: glm_moe_dsa`) is a **DeepSeek-V3.2-family**
model: MLA + the DSA "lightning indexer" + sparse attention. We validate our TTNN port
against a GPU golden trace (captured from vLLM's `deepseek_v2.py` `is_v32` path), reusing
the DeepSeek-V3.2 `reference_cpu` and the v32 `ttMLA` with GLM dims.

Harness: `models/demos/deepseek_v32/tests/test_vs_gpu_ref.py` — one parametrized file covering
**both** DeepSeek-V3.2 and GLM-5.1 (a `ModelSpec` per model). GLM cases carry **`glm_5_1`** in the
test id; DS cases carry `deepseek_v32`. Filter to GLM with `-k glm_5_1`.

## Status — all phases green (2026-06-17)
- **Phase 1** trace self-consistency (no weights/device): green L0/30/60/77.
- **Phase 2** host ceiling (CPU-ref vs trace): L0 indexer logits PCC ≈ 1.0, topk overlap 0.996.
- **Phase 3** device (ttMLA vs trace, Blackhole): L0 1x2 — indexer logits PCC 1.0 / topk 0.9957,
  KV latent PCC 0.9997, **MLA output PCC 1.0**. L60 1x2 — output PCC 0.9979.

## GLM ≠ DS-V3.2 (the deltas that matter)
| | DS-V3.2 | GLM-5.1 |
|---|---|---|
| hidden | 7168 | **6144** |
| MLA q heads | 128 | **64** |
| indexer heads | 64 | **32** |
| q_lora_rank | 1536 | **2048** |
| qk_nope / qk_rope / v_head | 128 / 64 / 128 | **192 / 64 / 256** |
| kv_lora_rank | 512 | 512 (same) → cache row still **576** = 512 ‖ 64 k_pe |
| index_topk | 2048 | 2048 (same) |
| MLA RoPE | interleaved | interleaved |
| **indexer RoPE** | **non-interleaved** (rotate_half) | **interleaved** |
| YaRN | yes (mscale²) | **no** → scale = `qk_head_dim**-0.5` = 256**-0.5 |
| rope_theta | 10000 | **1e6** |

The shared 576 latent + topk 2048 are why the C++ ops (`indexer_score`, `topk_large_indices`,
`sparse_sdpa`) carry over unchanged.

### Constraint: `sparse_sdpa` needs per-chip **query** heads ≥ 32 (multiple of 32)
MLA runs absorbed/MQA-style even in prefill: `q [1,H,S,576]`, `kv [1,1,T,576]` (one shared
latent; K = full 576, V = first 512). The only head dim in the op is the **query heads**.
So `q_heads / tp ≥ 32` → GLM's 64 heads ⇒ **tp ≤ 2**. This caps only the TP (head) axis;
SP (sequence) is free, so any device count works as long as tp ∈ {1,2}. Meshes are
**box-adaptive** (`mesh_utils`, filtered to tp ≤ 2 by `_glm_parametrize_mesh_device`): the box
is auto-detected from `/dev/tenstorrent/*` and run at the EXACT device count (no sub-mesh — the
runtime doesn't reliably support sub-meshes; see the mesh_utils note). LoudBox(8) → `sp8xtp1`,
`sp4xtp2`; QuietBox(4) → `sp1xtp1`, `sp2xtp2`; single → `sp1xtp1`. Run with the whole box
visible — no `TT_VISIBLE_DEVICES` juggling.

## Setup / prerequisites
1. **`transformers == 4.53.0`** (tt-metal's pin). 5.x removed `no_init_weights` /
   `is_torch_fx_available` used by the vendored `deepseek_v3/reference/modeling_deepseek.py`
   that the device path (`ttMLA`/`RotarySetup`) imports → the device/host tests won't even import
   on 5.x. `uv pip install transformers==4.53.0`. (GLM itself needs nothing newer: the config is
   built by hand, never `AutoConfig`, since transformers doesn't know `glm_moe_dsa`.)
2. **Built ttnn** with the DSA ops (`./build_metal.sh` — provides `IndexerScoreProgramConfig`,
   `topk_large_indices`, `sparse_sdpa`). Phase 3 needs **Blackhole** hardware; Phase 1/2 are CPU-only.
3. **HF access to `zai-org/GLM-5.1`** for weights, and **the trace present** — both auto/described below.

### Weights — downloaded automatically, per-layer (never the full ~700 GB)
No manual step: the first test run that needs layer `L` calls
`reference_cpu.weights.initialize_weights(mla, layer=L, repo="zai-org/GLM-5.1")`, which uses
`huggingface_hub.hf_hub_download` to pull **only the shard(s) holding that layer's `self_attn.*`**
(bf16 → no fp8 dequant; GLM's HF weight names are identical to DeepSeek-V3.2, so the `_HF_TO_MLA`
loader is unchanged). Layers 0/30/60/77 live in shards `00001/00083/00204/00271` (~6 GB each,
~24 GB for all four). Cached at `~/.cache/huggingface/hub/models--zai-org--GLM-5.1/`.
- **Auth:** the repo may be gated → `huggingface-cli login` once, or `export HF_TOKEN=...`; the
  loader forwards the token via huggingface_hub. Override the repo with `GLM51_REPO` if needed.

### Trace — the GPU golden bundle (vLLM 5120-token prefill)
The harness reads it from **`GLM51_REF_DIR`** (env), defaulting to `<repo>/bit_sculpt/results/glm-51`.
The bundle is git-LFS-tracked in the **`bit_sculpt`** repo on branch `nmilicevic/v3_2-trace`. Clone
with LFS smudge off (pointers only), then **selectively** pull just the four captured layers' streams
— the full bundle is huge (all 78 layers of `module_io`, etc.):

```bash
# from the tt-metal repo root, so the default GLM51_REF_DIR (<repo>/bit_sculpt/results/glm-51) resolves:
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/tenstorrent/bit_sculpt.git
cd bit_sculpt
GIT_LFS_SKIP_SMUDGE=1 git switch --track origin/nmilicevic/v3_2-trace   # fetches LFS pointers only
# download only layers 0/30/60/77 (the ones with indexer_logits that the device/host tests use):
RUN=results/glm-51; INC=""
for L in 0 30 60 77; do
  for s in module_io/mla_input_layer module_io/mla_output_layer module_io/indexer_input_layer \
           dsa/indexer_logits_layer dsa/dsa_topk_indices_layer; do
    INC="$INC,$RUN/${s}_${L}/*"
  done
  INC="$INC,$RUN/kv_cache/layer_${L}/*"
done
git lfs pull --include="${INC#,}"
```
(Or clone `bit_sculpt` anywhere and point `GLM51_REF_DIR` at its `results/glm-51`.) The bundle's
`results/glm-51/TRACE_MANUAL.md` documents the streams; regeneration on a GPU node is in
`bit_sculpt/docs/model_traces/zai_org/glm_5_1_gpu_node_guide.md`. If the streams are absent the
tests **skip** (not fail), printing the path they looked for.

**Streams** (`indexer_logits` only for layers 0/30/60/77; `dsa_topk_indices` + `kv_cache` for all 78):
`module_io/mla_input` = ttMLA input (post input_layernorm, == indexer input); `module_io/mla_output`
= forward output (post o_proj, pre-residual); `dsa/indexer_logits` = index_score **pre causal mask**
(compared over tril); `dsa/dsa_topk_indices` (−1 = pad); `kv_cache/layer_L` = `[latent 512 ‖ k_pe 64]`.

## Run it (pytest; GLM cases carry `glm_5_1` in the test id)
```bash
T=models/demos/deepseek_v32/tests/test_vs_gpu_ref.py

# Phase 1 — trace self-consistency (no weights, no device):
python -m pytest $T -k "glm_5_1 and not host and not device"

# Phase 2 — host ceiling (CPU ref vs trace; downloads per-layer weights; the seq5120 CPU forward
# in the kv/output host tests is slow — prefer the device tests):
python -m pytest $T -k "host and glm_5_1" -s

# Phase 3 — device (Blackhole). Box-adaptive meshes (mesh_utils); run with the whole box visible
# (no TT_VISIBLE_DEVICES masking). On a LoudBox(8) the meshes are sp8xtp1 + sp4xtp2 (both 8-chip):
python -m pytest $T -k "device and glm_5_1" --ds-kpe-layout vllm -s              # all meshes × all layers
python -m pytest $T -k "device and glm_5_1 and sp4xtp2" --ds-kpe-layout vllm -s  # one mesh, all layers
python -m pytest $T -k "device and glm_5_1 and L30" -s                           # one layer, all meshes
#   only MLA output: -k "mla_output_device and glm_5_1"
#   drop --ds-kpe-layout vllm to use the frame-invariant k_pe L2 check instead (see caveat)
#   drop "and glm_5_1" to also run the DeepSeek-V3.2 cases (the file covers both models).

# Phase 4 — WHOLE-LAYER block (MLA+DSA+MoE) vs trace. Parametrized over GLM_BLOCK_LAYERS (default [30, 60];
# any MoE layer 3-77 works — just edit that list in the test). Layers 0-2 are dense (no MoE).
#
# The whole-layer decoder_io trace exists for ALL 78 layers but is git-LFS-pointer-only, so you must
# MANUALLY `git lfs pull` each layer's two streams (L-1 = block input, L = block output) before its run:
for L in 30 60; do
  git -C bit_sculpt lfs pull --include="results/glm-51/decoder_io/decoder_output_layer_$((L-1))/*,results/glm-51/decoder_io/decoder_output_layer_$L/*"
done
# A layer whose trace isn't pulled AUTO-SKIPS (the skip message prints the exact pull cmd). The ~30 GB of
# MoE expert weights per layer download AUTOMATICALLY (JIT) on first run — there is no manual weight step.
#
# LoudBox(8)/QuietBox(4) — box-adaptive, tp<=2 (the tp=4 mesh skips). Narrow with `and L60` / `and gate_device`:
python -m pytest $T -k "block and glm_5_1 and not galaxy" -s
# Galaxy(32, Wormhole) — tp<=2 shapes (8,2)[16 chips] / (16,2)[32 chips]; native (8,4) is tp=4 → unusable for GLM:
python -m pytest $T -k "block and glm_5_1 and galaxy" -s
```

## Whole-layer block test (Phase 4)
`test_glm_block_device_vs_reference` (parametrized over `GLM_BLOCK_LAYERS`, default `[30, 60]`) runs the
full GLM decoder block on device (`TtPrefillBlock`: input_layernorm → MLA+DSA → residual →
post_attn_layernorm → MoE → residual) and PCC-compares layer L vs the trace's whole-layer output
`decoder_io/decoder_output_layer_L`. Input is `decoder_io/decoder_output_layer_{L-1}` (the layer's
pre-input_layernorm hidden; the block applies the norm internally). The block reuses `deepseek_v3_d_p`'s
`TtMoe`; **GLM MoE = Kimi single-group routing** (`NUM_EXPERT_GROUPS=1`, top-8 of all 256 experts, sigmoid
+ `e_score_correction_bias`, route_scale 2.5).
- **Layers / trace**: add any MoE layer (3–77) to `GLM_BLOCK_LAYERS`. Its `decoder_io` trace (streams
  `L-1` + `L`) must be **`git lfs pull`-ed manually, per layer** (they're LFS pointers); an unpulled layer
  **auto-skips** with the pull command in the message. MoE expert weights (~30 GB/layer) download JIT.
- **Gate** is parametrized: `gate_host` (`HOST_ALL` — routing on host) and `gate_device` (`DEVICE_FP32`
  — the on-device single-group gate kernel, Kimi path). Routed/shared experts + MLA + norms always run on
  device. **Both verified at whole-layer PCC ≈ 0.9995 on LoudBox `sp4xtp2`** (32 experts/chip, layer 30).
  Filter one: `-k "...and gate_device"`.
- **Weights**: `reference_cpu.weights.load_moe_block_weights(layer)` pulls the layer's norms + `mlp.gate`
  (+`e_score_correction_bias`) + 256 routed experts + shared expert (~30 GB across shards 00079–00083);
  MLA weights come from the existing `WEIGHT_NAME_MAP`/MLACPU path. Fails loud if experts aren't a
  contiguous 0..255 set (incomplete download).
- **tp ≤ 2** (same MLA head constraint). Two tests share one body (`_run_glm_block`):
  `test_glm_block_device_vs_reference` (box-adaptive, FABRIC_1D — LoudBox runs `sp8xtp1`+`sp4xtp2`,
  skips `sp2xtp4`) and `test_glm_block_device_vs_reference_galaxy` (explicit `(8,2)`/`(16,2)`,
  FABRIC_2D+RELAXED_INIT, `requires_mesh_topology` → auto-skips off a 16-/32-chip box). **Galaxy's
  native `(8,4)` mesh is tp=4 → unusable for GLM**; `(32,1)` can't be opened on Galaxy so it's omitted.
- Threshold `BLOCK_OUTPUT_PCC = 0.95` (bf16 MLA + `bfloat4_b` routed experts + host gate + DSA top-k
  noise stack vs the fp8 GPU trace); bump `routed_expert_weights_dtype` for a cleaner comparison.

## Caveats (all expected, not bugs)
- **topk overlap < 1.0** (~0.98 on high rows): bf16 ties at the top-2048 cutoff flip a few
  borderline (least-relevant) keys. Logits PCC ≈ 1.0 and output PCC ≈ 1.0 confirm it's benign.
- **k_pe raw PCC is low** (~0.13) while the test passes: our **interleaved** k_pe storage vs
  vLLM's **half-split** is a dim-reordering (same values) — frame-invariant for q·k. The kv test
  asserts the per-row L2 match (≤0.05) and logs raw PCC as a diagnostic. Output PCC proves the
  RoPE is correct. **`--ds-kpe-layout vllm`** reindexes our k_pe via `interleaved_to_halfsplit_perm`
  and asserts element-wise PCC — **verified for GLM (1x2 L0): k_pe PCC 0.99977** (the DS permutation
  maps GLM's k_pe to vLLM's half-split layout exactly). Keep that flag for the stricter cross-stack
  k_pe check; it only affects the kv_cache test.

## What changed in shared code (DS path unchanged by default)
- `reference_cpu/model.py`: `ModelArgs.index_rope_interleave` (default **False** = DS); `IndexerCPU`
  uses it for its RoPE.
- `tt/mla/mla.py`: `ttMLA.__init__` accepts an `index_args` kwarg (default = DS `ModelArgs`);
  `_build_index_rope_tables` / `_device_rope_pe` do interleaved (Meta-style cos/sin +
  `get_rot_transformation_mat` + `rotary_embedding_llama`) when `index_args.index_rope_interleave`,
  else the original non-interleaved `rotary_embedding_hf`. DS uses defaults → byte-identical path.
- GLM dims come from `models/demos/deepseek_v3_d_p/reference/glm_5_1_config.py` (`GLM51Config`).
- `tt/tt_prefill_block.py`: `TtPrefillBlock.__init__` gained an optional `index_args` kwarg that it
  forwards to `ttMLA` (default `None` → ttMLA's DS default → byte-identical for DS). GLM passes
  `_glm_model_args()` so the block's indexer is 32-head / interleaved / θ=1e6.
- `reference_cpu/weights.py`: added `resolve_layer_moe_shards` + `load_moe_block_weights` (load a
  layer's decoder norms + MoE gate/256-experts/shared from HF in raw `[out,in]` orientation). The
  block (`TtPrefillBlock`) and MoE (`TtMoe`) themselves are reused unchanged from `deepseek_v3_d_p`.
