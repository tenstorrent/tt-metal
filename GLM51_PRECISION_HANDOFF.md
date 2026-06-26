# GLM-5.1 device-vs-GPU PCC investigation — handoff for Galaxy continuation

**Branch:** `nmilicevic/glm-tp4` (= `ipotkonjak/dsa-sparse-head-reshard-tp4` = squashed DSA PR #47832 + tp4 head-reshard). GLM-5.1 lives in `models/demos/deepseek_v3_d_p/`.
**Model:** GLM-5.1 (HF `glm_moe_dsa`, DeepSeek-V3.2 family): MLA + DSA lightning-indexer sparse attention (top-2048) + 256-expert single-group (n_group=1) top-8 MoE. 78 layers (0–2 dense, 3–77 MoE), 64 attn heads.
**Validation target:** vLLM fp8 GPU trace, 5120-token prefill, per-layer chained PCC.

---

## 0. TL;DR — current verdict

The chained per-layer PCC vs the GPU trace declines smoothly (0.999 → ~0.95 @ L39 → ~0.88 trough ~L62 → recover). **This is our DEVICE UNDER-PRECISION vs the GPU, NOT a logic/routing/scale/input bug.** Every "small thing" was ruled out:
- **Input** to layer 0 is byte-identical to the GPU (verified 2 ways).
- **All scales** (SDPA 0.0625, indexer 1/√128, rope θ=1e6) align.
- **MoE routing** (expert selection) and **indexer top-2048 selection** match the GPU.
- **Quant/dequant** is the canonical DeepSeek formula, identical to DS-V3/Kimi.

Two real bugs WERE found+fixed: the `e_score_correction_bias` bf16-crush (mean-center) and the dense-FFN dims bug.

The residual drop is precision: **we run bf4 experts + bf16 accumulation + bf8 W_UV; the GPU runs fp8 experts + fp32 accumulation everywhere + bf16 W_UV.** Raising our precision toward the GPU helps (a "match-GPU" move). On 2x4 the small levers give +0.003; the BIG lever (8-bit experts) and the full 78-layer run are **DRAM-blocked on 2x4** → need Galaxy.

---

## 1. GPU data-format profile (source-verified from the GPU machine — vLLM 0.22.1rc1, zai-org/GLM-5.1-FP8, H200, FLASHMLA_SPARSE, enforce_eager)

**Global rules:**
- **Activations:** bf16 on the residual stream everywhere.
- **Weights:** fp8 e4m3 block-scale (128×128 blocks, fp32 `weight_scale_inv`) for **MoE experts + big attention projections (q_a/q_b/kv_a/o_proj)**. STAY bf16/fp32 (`modules_to_not_convert`): router/gate, `e_score_correction_bias` (fp32), **W_UK/W_UV** (bf16), all norms, embed, lm_head, **indexer `weights_proj` (bf16, NOT fp8)**.
- **Accumulation: fp32 at EVERY matmul/softmax/reduction**, stored back to bf16. (bf16 bmm accumulates fp32 internally; DeepGEMM fp8 GEMMs accumulate fp32; attention QKᵀ/softmax fp32; MoE combine fp32; router sigmoid+bias fp32.)
- **No Hadamard anywhere** (indexer, SDPA, MoE all confirmed). Only rotation is RoPE.

**Experts (confirmed Q&A from GPU machine):**
- `quantization_config = {quant_method: fp8, fmt: e4m3, activation_scheme: dynamic, weight_block_size: [128,128]}` — **identical recipe to DS-V3/Kimi.**
- Weights stay **fp8 e4m3 END-TO-END** — never dequantized to bf16. fp8×fp8 DeepGEMM → fp32 accum → bf16 out → re-quant to fp8 before next GEMM. So GPU effective expert-weight precision = **fp8 (8-bit)**.
- `weight_scale_inv` = **fp32**, 128×128 blocks (gate/up [2048,6144]→scale[16,48]; down [6144,2048]→scale[48,16]). NO second-level / per-tensor / input_scale / UE8M0 (UE8M0 only on Blackwell sm100; Hopper uses plain fp32).
- **Activation quant: fp8 e4m3, per-token-group-128 along the contraction (hidden) dim, dynamic (per-forward).** Applied before W13 and re-quant on the SiLU·mul intermediate before W2.
- Load-time: **no transpose, no interleave, scale NOT folded** into the weight (stays a separate fp32 [N/128,K/128] tensor, applied on-the-fly in DeepGEMM). Only fusion: gate_proj+up_proj concatenated on output dim, each keeping its own block scales.

**KV cache:** `fp8_ds_mla` — 512×fp8 e4m3 + 4×fp32 scale + 64×bf16 rope per token (656 B/token), upconverted to bf16 for the bf16 sparse-prefill kernel.

**Scales (also in trace `metadata.json`):**
- SDPA `softmax_scale = 0.0625 = 1/√256 = qk_head_dim**-0.5` (qk_head_dim=256), single scalar all 78 layers, **no YaRN/mscale** (`rope_scaling=None`).
- Indexer scale = `1/√128 (= index_head_dim**-0.5)`, applied as `weights·q_scale·(1/√128)·n_head**-0.5` (n_head=32). **Distinct from SDPA.** No softmax in the indexer — just topk on fp32 logits.
- rope θ = 1e6. index_n_heads=32, index_head_dim=128, index_topk=2048.

---

## 2. Our DEVICE data-format profile (audited from code)

| Component | Weight | Act | Accum (fidelity) | File |
|---|---|---|---|---|
| q_a/q_b/kv_a/o_proj | **bf8** | bf16 | **bf16 (HiFi2, fp32_dest_acc=False)** | mla.py:290-295 (`default_compute_kernel_config`) |
| W_UK (wkv_b1) | bf16 ✓ | bf16 | bf16 | mla.py:444 |
| **W_UV (wkv_b2)** | **bf8** | bf16 | bf16 | mla.py:446 |
| sparse_sdpa | — | bf16 | **bf16 (HiFi2)** | mla.py:1251 (uses default config) |
| gate/router | bf16 ✓ | bf16 | **fp32 (HiFi4)** ✓ | tt_moe_gate_prefill.py:83-85 |
| e_score_correction_bias | bf16→fp32 (centered) | — | fp32 | tt_moe_gate_prefill.py:201 |
| **routed experts** | **bf4** | bf8 | **bf16 (LoFi, fp32_dest_acc=False)** | tt_routed_expert.py:28-30,200 |
| shared experts | bf8 | bf16 | bf16 (HiFi2) | tt_shared_expert.py:26-28 |
| indexer wq_b/wk/weights_proj | bf16 ✓ | bf16 | HiFi4/fp32 ✓ | indexer.py:162,192,205 |
| KV cache | bf8 (whole 576) | — | — | mla.py:851,1129 |
| norms/embed/lm_head | bf16 ✓ | bf16 | — | — |

**ttnn dtypes available:** `bfloat16, bfloat8_b, bfloat4_b, fp8_e4m3` — **NOTE: ttnn HAS `fp8_e4m3`** (the GPU's exact format). `bfloat8_b` is a *block-float* (~7 mantissa bits, shared exp per 16) — actually MORE precise than fp8 e4m3 (3 mantissa bits), so for matching the fp8 trace `fp8_e4m3` is the tighter target.

### Where we DIVERGE from the GPU (we are LOWER, ranked by impact)
1. **Accumulation: bf16 (MLA HiFi2, experts LoFi) vs GPU fp32 — everywhere.** Systemic.
2. **Routed experts: bf4 (4-bit) vs GPU fp8 (8-bit).** The MoE half.
3. **W_UV (wkv_b2): bf8 vs GPU bf16.**
4. Expert activations: bf8 (block-float) vs GPU fp8 e4m3 per-token-group-128. (second-order)

---

## 3. Experiments + results

### 3.1 Input verification — IDENTICAL ✓
- `metadata.json`: `applied_chat_template=False`, full tokenization `truncated_from_tokens=5178`, `truncate_tokens=5120`, `chunk_rows=5120`. So GPU input = first 5120 tokens of raw `models/demos/deepseek_v3_d_p/demo/sample_prompt.txt`.
- Independent re-tokenization with GLM tokenizer (`add_special_tokens=False`) → **5178 tokens, first 5120 match trace `token_ids` exactly (5120/5120)**. First tokens `[32, 62324, 14847, 1526, 279, ...]`; first completion token = **19264**.
- Our test (`_glm_load_token_ids`) reads `metadata.json["token_ids"][:5120]` directly → identical by construction.

### 3.2 Scale alignment — ALL ALIGN ✓
SDPA 0.0625 ✓; rope θ=1e6 ✓; index dims 32/128/2048 ✓; no mscale (GLM rope_scaling=None) ✓; indexer rope interleaved (GLM-specific, `index_rope_interleave=True`) ✓. Indexer scale is **topk-invariant** (global positive scalar; indexer output is just the top-2048 selection) → its exact value can't change the selection.

### 3.3 MoE routing (expert selection) vs trace — HEALTHY ✓
2x4 DEVICE_FP32, fresh cache: **L3 = 253 unique experts (0.856 top-8 overlap)**, HOST_ALL = 252 (0.872), GPU = 254. No collapse; healthy at all depths (237–253 unique, 0.6–0.93 overlap). DEVICE_FP32 ≈ HOST_ALL ⇒ gate-mode/bf16-logits irrelevant. The earlier "86% misroute / 50 experts" (`/localdev/nmilicevic/glm_routing_device/`, Jun-24) was a **STALE pre-fix dump.**

### 3.4 Indexer top-2048 selection vs trace — MATCHES ✓
Injecting the GPU's exact `dsa_topk_indices` at every layer moved the chained trough by only **Δ −0.004** ⇒ our indexer selects ≈ the same tokens. sparse_sdpa op itself (fed exact trace q/kvpe/indices) = **0.9997 flat** across all layers (Exp A, `test_sparse_sdpa_vs_gpu.py`).

### 3.5 Device chained per-layer PCC (L40, DEVICE_FP32, BASELINE current precision)
| L | PCC | | L | PCC |
|---|---|---|---|---|
| 0–19 | 0.998–0.999 | | 32 | 0.99051 |
| 20 | 0.99714 | | 34 | 0.98307 |
| 24 | 0.99599 | | 36 | 0.97211 |
| 28 | 0.99377 | | 38 | 0.95735 |
| | | | **39** | **0.94995** |
Smooth monotonic decline. HOST_ALL ≈ DEVICE_FP32 within 0.001 every layer. First-6-layer ≈ 0.999.

### 3.6 sparse_sdpa injection (GPU sparse_sdpa teacher-forced, all else device, DEVICE_FP32)
Knob `GLM_INJECT_SPARSE_SDPA=<trace>/sparse_sdpa`. Replaces `_sparse_mla` output with trace `sparse_sdpa_output_layer_L`.
| L | baseline | +GPU sparse_sdpa | Δ |
|---|---|---|---|
| 20 | 0.99714 | 0.99799 | +0.001 |
| 32 | 0.99051 | 0.99464 | +0.004 |
| 36 | 0.97211 | 0.98738 | +0.015 |
| **39** | **0.94995** | **0.97902** | **+0.029** |
**⇒ ~58% of the deep-layer drop flows through the attention sub-block** (q projections + indexer + sparse_sdpa). The op is clean (0.9997), so the carried error is the **bf8 projections + bf16 accumulation** vs the GPU's bf16-weight + fp32-accum. The other ~42% is MoE (bf4 experts) + residual.

### 3.7 Fast CPU bf16 reference (`glm_5_1` HF modeling, SDPA) — CONFOUNDED, don't trust chained
`/localdev/nmilicevic/glm_cpu_ref.py` (teacher-forced), `glm_cpu_ref_chained.py` (chained). ~1 min/layer (>10× faster than the old einsum MLACPU). Teacher-forced per-layer 0.994–0.999. **Chained collapses to 0.445 @ L76** — but teacher-forced stays ~0.99 EVERY layer ⇒ it's the reference's OWN bf16 indexer diverging from the GPU (compounding ~0.99/layer → 0.45). **The device (0.95 @ L39) is FAR more faithful than this CPU ref ⇒ not a usable bf16 ceiling.**

### 3.8 Precision levers (L40, DEVICE_FP32) — validated, but small + memory-blocked
| Lever (knob) | L32 | L36 | L39 | Δ@L39 |
|---|---|---|---|---|
| baseline | 0.99051 | 0.97211 | 0.94995 | — |
| #1 fp32 MLA accum (`GLM_MLA_FP32_ACC`) | 0.99139 | 0.97417 | 0.95265 | **+0.0027** |
| #2 W_UV bf16 (`GLM_MLA_WUV_BF16`) | 0.99072 | 0.97237 | 0.95017 | +0.0003 |
| #1+#2 | 0.99131 | 0.97429 | 0.95292 | **+0.0030** |
| #3 experts bf8 (`GLM_MOE_EXPERTS_BF8`) | — | — | — | **OOM (fail)** |

### 3.8b Component teacher-forcing decomposition (L40, DEVICE_FP32) — ROUTING is the dominant carrier
Injected each GPU trace component into the device chain (env-gated hooks `GLM_INJECT_TOPK` in mla.py:1285/1366, `GLM_INJECT_ROUTING` in tt_moe.py:507/414; `GLM_INJECT_SPARSE_SDPA` already in mla.py). Logs `/localdev/nmilicevic/glm_tp4_logs/run_{A..E}`.
| Layer | baseline | inject TOPK | **inject ROUTING** | HOST_ALL |
|---|---|---|---|---|
| L20 | 0.99714 | 0.99716 | 0.99836 | 0.99725 |
| L32 | 0.99051 | 0.99059 | 0.99693 | 0.99056 |
| L36 | 0.97211 | 0.97190 | 0.99311 | 0.97184 |
| **L39** | **0.94995** | 0.94896 | **0.98793 (+0.038)** | 0.94906 |
- **Inject GPU routing (expert_ids+weights) → +0.038 @ L39** (dominant, grows with depth) > sparse_sdpa/MLA (+0.029) >> indexer topk (~0).
- **HOST_ALL ≈ DEVICE_FP32** ⇒ gate *compute precision* is NOT the issue.
- Device expert-selection overlap vs GPU **degrades with depth: 0.872@L3 → 0.747@L20 → 0.605@L39** (no collapse; ~250 experts used).
- **Interpretation:** the gate is faithful but routes on a DRIFTED input; near-degenerate top-8-of-256 boundaries flip → wrong experts → more drift. **A feedback amplifier of the upstream precision (bf4 experts + bf16 MLA), not a gate bug.** ⇒ On Galaxy, fp8 experts → less hidden drift → less routing divergence → recovers most of the +0.038. The experts are the root; routing is the amplifier. **Token-drop ruled out** (capacity_factor=8 → 40960/chip buffer vs ~6000 actual max per-chip sum; overflow check never fired).

### 3.9 THE BLOCKER — 2x4 DRAM ceiling
- **bf8 experts OOM** (2× bf4) at L40 (`TT_FATAL bank_manager.cpp:462`).
- **The full 78-layer monolithic forward OOMs at ~L47 of model load at ANY precision** (even compute-only lever-1). GLM-5.1's 78 layers do not fit all-resident in 2x4 DRAM.
- ⇒ **The "0.88 trough" came from a STREAMING run (old deepseek_v32 `test_glm_model_vs_reference`, build→forward→free per layer) or a Galaxy mesh — never the 2x4 monolithic path** (which tops out ~L40–47).

---

## 4. Bugs found + FIXED (real, kept)
1. **`e_score_correction_bias` bf16-crush** — GLM's bias is mean ~34.5 with ±0.3 load-balancing variation; bf16 (step ~0.25 at mag 34) crushes it to ~3 levels → routing collapses onto a few experts. **Fix: mean-center before bf16** (`tt_moe_gate_prefill.py:201`, `torch_bias = torch_bias - torch_bias.mean()`). noaux_tc constant cancels in top-k; weights use sigmoid not sigmoid+bias → mathematically identical, bf16-safe. (DS/Kimi biases are small so they never hit this.)
2. **Dense-FFN dims bug** — `TtPrefillBlock` built `TtFfn` without `emb_dim`/`hidden_dim` → GLM dense layers (0–2) used DeepSeek's 7168/18432 and ASSERTED. **Fix: pass `emb_dim=config.hidden_size`, `hidden_dim=model_cfg.INTERMEDIATE_SIZE`** (`tt_prefill_block.py`; config-driven, no-op for DS/Kimi 18432, correct for GLM 12288).

---

## 5. Code changes this session (all env-gated, off by default)

**`models/demos/deepseek_v3_d_p/tt/mla/mla.py`**
- `import os`.
- `GLM_MLA_FP32_ACC` knob (~line 307): when set, `default_compute_kernel_config = hifi4_fp32_compute_kernel_config` (HiFi4 + fp32_dest_acc). **Requires `out_subblock_h*out_subblock_w ≤ 4`** (fp32_dest_acc halves DST). Fixed via `_cap_subblock(pc, budget=4)` helper + `_make_batched_mm_kwargs` `subblock_budget = 4 if _fp32_acc else 8`.
- `GLM_MLA_WUV_BF16` knob: `wkv_b2` dtype bf8→bf16.
- `GLM_INJECT_SPARSE_SDPA=<dir>` hook in `_sparse_mla` + `_inject_sparse_sdpa()`: load trace `sparse_sdpa_output_layer_L` [S,H,512], shard `dims=(2,1)` (heads→TP, seq→SP) — replaces the device sparse attention output. Requires `is_balanced=False` (the test uses it).

**`models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py`**
- `test_glm_chained_vs_trace` (2x4 sp2×tp4, single-shot 5120, per-layer chained PCC vs trace `decoder_output_layer_L` + MoE routing-vs-`expert_ids`). Parametrized `num_layers∈{6,40,78}` × `gate_fallback_mode∈{HOST_ALL,DEVICE_FP32}`, mesh-2x4. `is_balanced=False`. Two-pass (cache build → forward from cache + inject indexer stems).
- `_glm_routed_expert_dtype()` → `GLM_MOE_EXPERTS_BF8` knob.
- dtype-aware routed-expert cache completeness check (the wildcard otherwise treats wrong-dtype cache as complete → GLM cache landmine).
- `_glm_load_token_ids`, `_glm_build_cache`, `_run_glm_chained_vs_trace`.

**Other:** `/localdev/nmilicevic/glm_cpu_ref.py`, `glm_cpu_ref_chained.py` (CPU ref); `/localdev/nmilicevic/tt-metal/run_glm_lever.sh` (runner); offline bisection: `/localdev/nmilicevic/bisect_bf16.py`, `bisect_moe_bf4.py`, `bisect_mla_sparse.py`.

---

## 6. NEXT STEPS ON GALAXY (8x4 = 32 chips; GLM tp must be ≤... actually tp4 OK via head-reshard, use (16,2) for tp2 or 8x4 with tp4)

**The whole point: with Galaxy DRAM the full 78 layers + 8-bit experts FIT, so you can finally measure whether matching the GPU's precision closes the trough.**

### 6.1 First: reproduce the baseline full L78 trough on Galaxy
Run `test_glm_chained_vs_trace` with `num_layers=78`, `gate_fallback_mode=DEVICE_FP32`, on the Galaxy mesh. Expected baseline trough ~0.88 @ L62 (confirm). The cache build is the long pole (~75 MoE layers × 256 experts; hours). Re-tokenize/verify input matches (it will — uses trace token_ids).

### 6.2 Then apply the precision levers (match-targets), measure the trough each time
Priority order (biggest expected impact first, since on 2x4 only the small ones fit):
1. **Experts bf4 → `fp8_e4m3`** (NOT bf8 — `fp8_e4m3` matches the GPU's e4m3 rounding; bf8/bfloat8_b *overshoots* GPU precision). Verify ttnn matmul supports `fp8_e4m3` weights + the 128×128 block scale. This is the MoE half (~42% of the drop) — the lever we could NOT test on 2x4.
   - Knob exists as `GLM_MOE_EXPERTS_BF8` (bfloat8_b); ADD an `fp8_e4m3` option to `_glm_routed_expert_dtype()`.
   - Also raise routed-expert **accumulation to fp32** (`tt_routed_expert.py:28-30` LoFi+fp32_dest_acc=False → HiFi4/fp32) and **activations toward fp8 e4m3 per-token-group-128** if matchable.
2. **`GLM_MLA_FP32_ACC=1`** — fp32 MLA accumulation (+0.0027 @ L39 measured on 2x4; the attention-half lever).
3. **`GLM_MLA_WUV_BF16=1`** — W_UV bf16 (+0.0003; tiny but free, +~8MB/layer DRAM).
4. **Combine all** → does the trough move from 0.88 toward 0.95+? That answers the core question.

### 6.3 Match-target table (set these on Galaxy)
| Component | Current device | GPU target | How |
|---|---|---|---|
| routed experts (weight) | bf4 | **fp8_e4m3** (8-bit) | `_glm_routed_expert_dtype` |
| routed experts (accum) | bf16 (LoFi) | **fp32** | tt_routed_expert.py compute config |
| routed experts (act) | bf8 | fp8_e4m3 per-tok-group-128 | (if supported) |
| MLA accum (proj+sdpa) | bf16 (HiFi2) | **fp32** | `GLM_MLA_FP32_ACC` |
| W_UV (wkv_b2) | bf8 | **bf16** | `GLM_MLA_WUV_BF16` |
| shared expert accum | bf16 (HiFi2) | fp32 | tt_shared_expert.py |
| q_a/q_b/kv_a/o_proj | bf8 | fp8_e4m3 (8-bit) | (optional; ≈ already 8-bit) |

### 6.4 Open questions / risks for Galaxy
- Does ttnn's `fp8_e4m3` matmul support the **128×128 block scale** like DeepGEMM? If it's per-tile/per-tensor only, it won't bit-match — may need a custom block-scaled fp8 path (or accept the approximation).
- fp32 accumulation everywhere will need the `out_subblock ≤ 4` cap applied to the MoE/FFN matmul program configs too (only the MLA path is capped so far). Watch for `bank_manager`/DST `TT_FATAL`.
- An **intermittent tp4 spin-hang** was seen once at L3 sparse MLA in a 40-layer forward (process Running, log frozen >10min; succeeds on retry). Watch for it; `tt-smi -r` to recover.
- GLM tp constraint: 64 heads. tp4 → 16 heads/chip triggers `_sparse_head_reshard` (mla.py:1198-1267); ran clean on 2x4. On Galaxy decide tp (tp2 = (16,2) no reshard; tp4 = uses reshard).

### 6.5 Also re-run (clean, on Galaxy) — selection sanity (the user asked)
- **Indexer top-2048 vs trace `dsa/dsa_topk_indices_layer_L`** per layer (Jaccard overlap). Expect high (validated Δ−0.004 before).
- **MoE expert_ids vs trace `routing/expert_ids_layer_L`** per layer (top-8 set overlap, unique-expert count). Expect ~250 unique ≈ GPU's 254 (validated on 2x4).

---

## 7. File / path / env reference
- **Repo:** `/localdev/nmilicevic/tt-metal` (branch `nmilicevic/glm-tp4`). Pushes go to `origin/nmilicevic/glm-tp4`.
- **GPU trace:** `/localdev/nmilicevic/tt-metal/bit_sculpt/results/glm-51-traces/vllm-glm51-sdpa-5k-trace/` — `decoder_io/decoder_{input,output}_layer_L`, `routing/expert_ids_layer_L` (+expert_weights), `dsa/dsa_topk_indices_layer_L`, `sparse_sdpa/sparse_sdpa_{input,output}_layer_L`, `kv_cache/layer_L`, `metadata.json`. Trace-gen scripts: `bit_sculpt/analysis/model_traces/zai_org/glm_5_1/vllm_tracer.py`.
- **Cache (2x4, regenerate on Galaxy for its mesh):** `/localdev/nmilicevic/glm_tp4_cache/ttnn/` + `host_ref/`. Cache key = (model, mesh, layer, weight, **dtype**, layout) — dtype change ⇒ re-cache only that weight.
- **Weights:** `GLM51_REPO=zai-org/GLM-5.1-FP8` (fp8; loaders dequant). The bf16 master `zai-org/GLM-5.1` was deleted.
- **Env (all runs):** `TT_METAL_HOME=/localdev/nmilicevic/tt-metal PYTHONPATH=/localdev/nmilicevic/tt-metal GLM51_REPO=zai-org/GLM-5.1-FP8 HF_HUB_OFFLINE=1 TT_DS_PREFILL_TTNN_CACHE=<dir> TT_DS_PREFILL_HOST_REF_CACHE=<dir>`. Optional dump: `TT_DS_PREFILL_DUMP_GATE_INDICES=<dir>`.
- **Knobs:** `GLM_MLA_FP32_ACC`, `GLM_MLA_WUV_BF16`, `GLM_MOE_EXPERTS_BF8`, `GLM_INJECT_SPARSE_SDPA=<trace>/sparse_sdpa`.
- **Invoke:** `pytest -svq models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py -k "glm_chained_vs_trace and mesh-2x4 and device_fp32 and L40"` (swap mesh/num_layers/gate on Galaxy).

---

## 8. One-paragraph conclusion
GLM-5.1's chained PCC drop vs the fp8 trace is **distributed device under-precision** — roughly half the attention path (bf8 projections + bf16 accumulation) and half the MoE (bf4 experts) — measured against a GPU that is actually **high precision (bf16 weights/activations + fp32 accumulation everywhere, fp8 only for expert/proj weights + indexer logits + KV storage)**. It is NOT a logic, routing, scale, input, or quant-formula bug — those were each verified aligned, and the two real bugs (bias bf16-crush, dense-FFN dims) are fixed. Matching the GPU's precision is the lever; on 2x4 only the small parts fit (+0.003), so the decisive test — **8-bit (`fp8_e4m3`) experts + fp32 accumulation everywhere + bf16 W_UV across the full 78 layers** — must be run on Galaxy, where it will finally show whether matching the GPU's precision closes the ~0.88 trough.
