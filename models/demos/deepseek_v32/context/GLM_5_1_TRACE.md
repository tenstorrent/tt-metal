# GLM-5.1 — MLA/DSA trace comparison (how to run)

GLM-5.1 (`zai-org/GLM-5.1`, HF `model_type: glm_moe_dsa`) is a **DeepSeek-V3.2-family**
model: MLA + the DSA "lightning indexer" + sparse attention. We validate our TTNN port
against a GPU golden trace (captured from vLLM's `deepseek_v2.py` `is_v32` path), reusing
the DeepSeek-V3.2 `reference_cpu` and the v32 `ttMLA` with GLM dims.

Harness: `models/demos/deepseek_v32/tests/test_vs_gpu_ref_glm.py` (mirrors `test_vs_gpu_ref.py`).

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

## Run it
```bash
# Phase 1 — trace self-consistency (no weights, no device):
python models/demos/deepseek_v32/tests/test_vs_gpu_ref_glm.py            # standalone runner
#   (Phase-1 pytest also works: pytest <file> -k "trace or topk or logits or kv_split")

# Phase 2 — host ceiling (CPU ref vs trace; downloads per-layer weights):
python models/demos/deepseek_v32/tests/test_vs_gpu_ref_glm.py --layers 0,30,60,77
#   add --full for the MLA output/kv recompute (CPU dense forward at seq5120 is very slow — prefer device)

# Phase 3 — device (Blackhole). Meshes box-adaptive (mesh_utils); run with the whole box visible
# (no TT_VISIBLE_DEVICES masking). On a LoudBox(8) the meshes are sp8xtp1 + sp4xtp2 (both 8-chip):
python -m pytest .../test_vs_gpu_ref_glm.py -k "device" --ds-kpe-layout vllm -s              # all meshes × all layers
python -m pytest .../test_vs_gpu_ref_glm.py -k "device and sp4xtp2" --ds-kpe-layout vllm -s  # one mesh, all layers
python -m pytest .../test_vs_gpu_ref_glm.py -k "device and L30" -s                           # one layer, all meshes
#   only MLA output: -k "mla_output_device and L30 and sp4xtp2"
#   drop --ds-kpe-layout vllm to use the frame-invariant k_pe L2 check instead (see caveat)
```

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
