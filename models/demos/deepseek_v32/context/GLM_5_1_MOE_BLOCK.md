# GLM-5.1 MoE decoder block — architecture & bring-up findings

How the GLM-5.1 **whole decoder layer** (MLA + DSA + MoE) is assembled and tested on device, and the
research that backs it. Companion to `GLM_5_1_TRACE.md` (the run guide). MLA/DSA details live there;
this doc is the MoE + full-block picture.

## TL;DR
- The device block **already existed**: `models/demos/deepseek_v32/tt/tt_prefill_block.py` `TtPrefillBlock`
  is a copy of `deepseek_v3_d_p`'s with only the `ttMLA` import pointing at the v32 DSA MLA. It composes
  `input_layernorm → MLA(+DSA) → residual → post_attention_layernorm → MoE/dense → residual`.
- MoE is the **real device** `TtMoe` from `deepseek_v3_d_p/tt/moe/` (experts/dispatch/combine on device);
  only the **gate/router** optionally runs on host (`GateComputeMode.HOST_ALL`, the default).
- **GLM MoE == Kimi routing**: top-k of **all** experts, a **single group** (`n_group=1`). DeepSeek-V3.2
  uses grouped routing (`n_group=8, topk_group=4`). GLM has 256 experts (Kimi 384, DS-V3.2 256).
- So building a GLM block needed **no new block/MoE code** — just an `index_args` passthrough, a MoE
  weight loader, and the test.

## The device block — `TtPrefillBlock`
- `tt/tt_prefill_block.py:25` `class TtPrefillBlock(LightweightModule)`. forward at `:340`:
  `attn_norm(x) → mla → x=add(x,mla_out) → ffn_norm → MoE|dense → x=add(x,ffn_out)`; returns `(x, kv)`.
- Dense-vs-MoE per layer: `self.is_moe = layer_idx >= model_cfg.NUM_DENSE_LAYERS` (`:197`). GLM
  `NUM_DENSE_LAYERS=3` (HF `first_k_dense_replace=3`) → **layers 0/1/2 dense, layer 30 is MoE**.
- `config` (HF-attribute) supplies `hidden_size`, `rms_norm_eps`, and the MLA dims (read by `ttMLA`);
  `model_cfg` (the `GLM51Config` static class) supplies all the MoE/dense scalar constants.
- Canonical tensor: `[1,1,seq/sp, emb/tp]` TILE, SP-sharded on seq (axis 0) + TP-sharded on hidden (axis 1).
- **GLM fix added here**: `__init__` gained an `index_args` kwarg forwarded to `ttMLA` (`:226`). Without
  it the block's indexer would use the DS default (64 heads, non-interleaved rope); GLM passes
  `_glm_model_args()` → 32 heads, interleaved rope, θ=1e6. Default `None` = DS → byte-identical for DS.

## The MoE — `TtMoe` (single-group = Kimi = GLM)
`deepseek_v3_d_p/tt/moe/tt_moe.py:39`. forward pipeline (`:413`):
`gate → routing_setup → all_gather(TP) → shared_expert + dispatch → routed_experts → combine → reduce
(weighted top-k sum + reduce-scatter TP) → final = routed + shared`.
- **Router** (`tt_moe_gate_prefill.py`): sigmoid scoring (`scoring_func='sigmoid'`), `noaux_tc` — a
  per-expert `e_score_correction_bias` is added **for selection only**; the **unbiased** sigmoid scores
  are gathered as weights, normalized (`norm_topk_prob`, ÷sum), then ×`route_scale`.
- **Grouped routing** (`reference modeling_deepseek.py:406`): scores reshaped `(n_group, experts/n_group)`,
  pick `topk_group` groups, mask the rest, then global top-k. **`n_group==1` collapses this to a plain
  top-k over all experts** — the GLM/Kimi path (`tt_moe_gate_prefill.py` `single_group = n_expert_groups==1`
  → fp32 `deepseek_prefill.moe_grouped_topk`; the bf16 `deepseek_grouped_gate` kernel is DS-only).
- **Expert-parallel dispatch**: `deepseek_prefill.dispatch` (`tt_dispatch.py`) with 5-field per-token
  metadata `[src_device, token_idx, topk_idx, routed_expert_id, router_weight]`; combine reconstructs
  token order from the same offsets+metadata (see [[project_deepseek_metadata_flow]]).
- **Gate compute mode** (`GateComputeMode`): `HOST_ALL` runs the gate matmul + top-k on host (torch);
  `DEVICE`/`DEVICE_FP32` run it on device (GLM single-group → the fp32 `moe_grouped_topk` path). Experts
  always run on device regardless. The block test **parametrizes both** — `gate_host` (HOST_ALL) and
  `gate_device` (DEVICE_FP32, the Kimi path); **both verified at whole-layer PCC ≈ 0.9995 on `sp4xtp2`**.

## GLM MoE config (authoritative: HF `config.json`, model_type `glm_moe_dsa`)
| field | GLM-5.1 | DeepSeek-V3.2 | Kimi-K2 |
|---|---|---|---|
| n_routed_experts | **256** | 256 | 384 |
| num_experts_per_tok (top-k) | 8 | 8 | 8 |
| n_shared_experts | 1 | 1 | 1 |
| **n_group / topk_group** | **1 / 1** | **8 / 4** | 1 / 1 |
| scoring_func / topk_method | sigmoid / noaux_tc | sigmoid / noaux_tc | sigmoid / noaux_tc |
| norm_topk_prob | true | true | true |
| routed_scaling_factor | **2.5** | 2.5 | 2.827 |
| moe_intermediate_size | 2048 | 2048 | 2048 |
| intermediate_size (dense MLP) | 12288 | 18432 | — |
| first_k_dense_replace | 3 | 3 | 1 |
| hidden_act | silu | silu | silu |

`GLM51Config` (`deepseek_v3_d_p/reference/glm_5_1_config.py`) mirrors these as static constants
(`NUM_EXPERT_GROUPS=1`, `NUM_LIMITED_GROUPS=1`, `ROUTE_SCALE=2.5`, `NUM_DENSE_LAYERS=3`,
`MOE_INTERMEDIATE_SIZE=2048`, `EMB_SIZE=6144`, `FABRIC_PAYLOAD_SIZE=6144`). GLM uses Kimi's routing
*method* but **not** its expert count or route_scale — don't copy `KimiK26Config` wholesale.

## Weights
- GLM HF MoE names are **identical** to DeepSeek-V3.2: per layer L —
  `model.layers.L.mlp.gate.weight`, `…mlp.gate.e_score_correction_bias` `[256]`,
  `…mlp.experts.{0..255}.{gate_proj,up_proj,down_proj}.weight` (gate/up `[2048,6144]`, down `[6144,2048]`),
  `…mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight`, plus the two decoder RMSNorms
  `input_layernorm.weight` / `post_attention_layernorm.weight` `[6144]`.
- Loader: `reference_cpu/weights.py` `load_moe_block_weights(layer)` (added here) returns those in the
  exact dict shape `TtPrefillBlock`/`TtMoe` consume, **raw HF `[out,in]`** (the TT modules transpose
  internally — do NOT pre-transpose); fp8 dequant branch kept (no-op for the bf16 GLM repo). MLA weights
  still come from the existing `WEIGHT_NAME_MAP`/MLACPU path.
- **Cost**: a single MoE layer's 256 experts span ~5 shards (layer 30 = `model-00079..00083`, ~6 GB each
  ≈ **30 GB**); the loader fetches them just-in-time and fails loud if the experts aren't a contiguous
  0..255 set (partial download). The full repo is ~700 GB — never pull it.

## Trace streams (bit_sculpt `results/glm-51`, 5120-tok prefill)
Whole-layer block test (per layer L; e.g. L=30) uses:
- `decoder_io/decoder_output_layer_29` `[5120,6144]` — block **input** (this layer's hidden **before**
  input_layernorm; == `decoder_input_layer_30`). The block applies the norm internally, so feed this, not
  `mla_input` (which is post-LN).
- `decoder_io/decoder_output_layer_30` `[5120,6144]` — block **output** (post-MoE, **post both residuals**).
- MoE internals (diagnostics): `module_io/gate_input_layer_30` `[5120,6144]` (post post_attn_layernorm),
  `module_io/router_logits_layer_30` `[5120,256]`, `routing/expert_ids_layer_30` `[5120,8]` int32,
  `routing/expert_weights_layer_30` `[5120,8]` (sigmoid, ÷sum, ×2.5).
- **Not present**: a standalone moe_output or shared-expert-out (only via `decoder_output - decoder_input
  - mla_output`). Layers 0/1/2 dense, 3–77 MoE; DSA on all 78.
- **Per-layer manual LFS pull.** The block test is parametrized over `GLM_BLOCK_LAYERS` (default
  `[30, 60]`; any MoE layer 3–77 — edit the list). `decoder_output_layer_{L}` exists for **all 78 layers**
  but is a **git-LFS pointer**, so pull each layer's two streams (`L-1` input, `L` output) **manually,
  per layer**, before its run:
  ```bash
  L=60; git -C bit_sculpt lfs pull --include="results/glm-51/decoder_io/decoder_output_layer_$((L-1))/*,results/glm-51/decoder_io/decoder_output_layer_$L/*"
  ```
  A layer whose trace is still a pointer **auto-skips** (the skip message prints this exact command). By
  contrast the layer's ~30 GB of MoE expert weights download **automatically (JIT)** on first run — no
  manual weight step.

## Meshes & experts-per-chip — LoudBox vs Galaxy
Routed experts are **expert-parallel across every device** in the mesh: `experts_per_chip = 256 / num_devices`.

| box | chips | mesh (GLM, tp≤2) | experts/chip | fabric |
|---|---|---|---|---|
| **LoudBox** | 8 | `sp8xtp1`, `sp4xtp2` | **32** | FABRIC_1D |
| QuietBox | 4 | `sp1xtp1`, `sp2xtp2` | 64 | FABRIC_1D |
| **Galaxy** | 32 | `(8,2)`, `(16,2)` | **8** | FABRIC_2D + RELAXED_INIT |

- **GLM needs tp ≤ 2** (64 q-heads; `sparse_sdpa` needs per-chip H = 64/tp ≥ 32). So Galaxy's native
  **`(8,4)` mesh (tp=4) is unusable for GLM** — use `(8,2)`/`(16,2)`. `(32,1)` can't be opened on Galaxy.
  On LoudBox, `sp2xtp4` (tp=4) is skipped.
- **LoudBox packs 4× the experts/chip of the Galaxy 8×4 DS layout** (32 vs 8). The routed-expert weights
  live in DRAM as `bfloat4_b` (~0.6 GB/chip at 32 experts), and the dispatch buffer is sized by
  `compute_constants(seq_len_per_chip, …, dispatch_buffer_capacity_factor=2)` — both fit on Blackhole, so
  LoudBox is a valid (and denser) place to validate the MoE dispatch/combine. It is a different stress
  point than Galaxy, not a different result.

## What we added on this branch
1. `tt/tt_prefill_block.py` — two GLM-support fixes (no-ops for DeepSeek, where the defaults are right):
   (a) `index_args` passthrough to `ttMLA` (GLM indexer: 32 heads / interleaved / θ=1e6); (b) pass
   `emb_dim=config.hidden_size` to `TtFfn` (the dense MLP) at both call sites — `TtFfn` hardcodes a
   default `EMB_DIM=7168` (DeepSeek), so dense GLM layers asserted 7168 vs 6144 until this was added.
2. `reference_cpu/weights.py` — `resolve_layer_moe_shards`, `_read_layer_ffn_tensors`, and **two** FFN
   loaders sharing it: `load_moe_block_weights` (norms + 256 experts + gate + shared, fail-loud on gaps)
   for MoE layers, and `load_dense_block_weights` (norms + plain `mlp.{gate,up,down}_proj`) for dense
   layers. Both raw HF `[out,in]` orientation.
3. `tests/test_vs_gpu_ref.py` — `_run_glm_block` body + two tests: `test_glm_block_device_vs_reference`
   (box-adaptive, FABRIC_1D — LoudBox/QuietBox) and `test_glm_block_device_vs_reference_galaxy`
   (explicit `(8,2)`/`(16,2)`, FABRIC_2D, `requires_mesh_topology` auto-skip). Parametrized over
   `GLM_BLOCK_LAYERS=[0,1,30,60]` (dense vs MoE branch in `_glm_block_state_dict`; layer-0 input =
   `decoder_input_layer_0`) × gate (`gate_host`/`gate_device`; dense skips `gate_device`) × mesh.
   `BLOCK_OUTPUT_PCC=0.95`. (`TtPrefillBlock`/`TtMoe`/`TtFfn` reused — only the GLM wiring is new.)

## Status
**Verified on LoudBox `sp4xtp2`** (8 chips) vs the GPU trace — both the MoE and dense variants of the
block run on device and match:
| layer | type | gate | whole-layer PCC |
|---|---|---|---|
| L0 | dense | host | 0.99958 |
| L1 | dense | host | 0.99963 |
| L30 | MoE (32 experts/chip) | host / device | 0.99953 / 0.99951 |
| L60 | MoE | device | 0.99820 |

So the full GLM layer — MLA+DSA + (dense MLP **or** MoE experts + single-group router) — runs on device.
The block test is parametrized over `GLM_BLOCK_LAYERS = [0, 1, 30, 60]` (0/1 dense, 30/60 MoE) × gate ×
mesh; dense layers run only `gate_host` (no MoE gate → `gate_device` skips). Still open (needs Galaxy
hardware): whether ttnn can open the non-native `(8,2)`/`(16,2)` meshes on physical Galaxy (validated
shape is `(8,4)`, which GLM can't use).

## Whole-model test — `test_glm_model_vs_reference` (streaming)
Goal: feed `decoder_io/decoder_input_layer_0` through the **whole stack** and PCC the *running* hidden
vs `decoder_output_layer_L` at each layer — i.e. measure **error accumulation**, not per-layer error.
- **Why not `TtPrefillTransformer`:** both the v3_d_p class and the v32 copy build **all layers RESIDENT**
  in `__init__` (78 GLM MoE layers ≈ 45 GB/chip — won't fit on 8 chips), and their forward embeds
  token-ids + runs the lm_head (wrong entry point for a pre-embedded hidden). The v3 transformer *test*
  only runs reduced stacks (`num_layers ∈ {5,12,61}`). So we do **not** use that class.
- **Design = test-side STREAMING block loop:** for `L in range(GLM_MODEL_LAYERS)`: build
  `TtPrefillBlock(L)` (via `_glm_block_state_dict(L)`) → forward → gather + PCC vs `decoder_output_layer_L`
  → **carry the DEVICE output into L+1** (true chaining, NOT teacher-forcing) → `del block; gc.collect()`
  to free its weights → next. One block resident at a time ⇒ memory bounded (same footprint as the block
  test). A shared rope + a single 1-layer kvpe cache are reused across layers.
- **Verified `del`+`gc.collect()` frees a MoE layer's device weights** between iterations (no OOM through
  chained MoE layers) — this is what makes the 78-layer stream feasible.
- **`GLM_MODEL_LAYERS`** env controls the chain length (default 2 = dense 0/1, quick; `78` = full stack;
  chain is contiguous from 0, stops at the first unpulled-trace layer). `GLM_MODEL_EXPERT_DTYPE` (bf4 default;
  bf8/bf16 currently `TT_THROW` — the MoE expert kernels are bf4-only).
- ⭐ **RESOLVED — the L7 cliff is a SILENT DISPATCH-BUFFER OVERFLOW that DROPS tokens; `init_zeros=True` only MASKS it.**
  (Kernel-traced over 2 subagent passes; the first pass wrongly concluded "uninitialized combine buffer, correct fix" —
  overturned by the arithmetic + clamp evidence below.) At L7 the chained massive-activation hidden imbalances routing →
  a chip's local experts receive more tokens than the flat dispatch buffer
  (`max_dispatch_buffer_token_size = max_dispatched_tokens_per_expert × dispatch_buffer_capacity_factor`,
  init_helpers.py:471; factor default **2**) → the **dispatch reader SILENTLY DROPS** the overflow tokens
  (reader_dispatch.cpp:252 bumps a counter, never writes the payload) → the **combine writer clamps** its row count to
  capacity (writer_untilize.cpp:219-226), so the dropped token's combine-output slot is **never written** = uninitialized
  DRAM. The reduce treats that slot as **LOCAL** (the token *did* select that local expert; post_combine_reduce_compute.cpp:90-93
  `is_local`, so NOT skipped) → multiplies the uninitialized DRAM × the token's **real gate weight** ≈ 3.3e38 → `inf`.
  **`init_zeros=True` (`GLM_COMBINE_INIT_ZEROS=1`) only zeros that slot → reads 0 → finite (L7 8e-05 → 0.9912) — but the
  dropped token's real expert contribution is STILL LOST** (0.991 vs the teacher-forced block's 0.998). It is **MASKING,
  not a fix** — keep it as defense-in-depth (an overflow degrades to a bounded error instead of NaN/inf propagating).
  - **REAL FIX:** eliminate the drop — raise **`dispatch_buffer_capacity_factor`** (default 2) so the max per-chip token
    sum ≤ capacity; AND promote the overflow check (tt_moe.py:~665, currently only under `return_intermediates`) to an
    ALWAYS-ON hard assert so a silently-corrupted run can't pass. The `must_zero_init` forced-read path is NOT the source
    (it multiplies by a zeroed weight → 0/NaN, never 3.3e38). v3/Kimi don't hit it because they run a single MoE layer on
    bounded random input (balanced routing, no overflow) — NOT a tp=1 thing (they DO test tp>1 (4,2)/(2,4)/(8,4)).
  - **VERIFIED + FIXED.** Run A (factor=2 + `GLM_MODEL_RETURN_INTERMEDIATES=1`): the kernel guard logged
    *"Overflow tokens were dropped - output data is corrupted"* at **L7 and only L7** — and it's the tile-padded
    **region offset (10400 > 10240)** that overflows, not the raw per-chip sum (10071, under cap). **Fix = bump
    `GLM_MODEL_CAPACITY_FACTOR` 2 → 8** (buffer 40960). **Full 78-layer factor=8 run (real-load, sp4xtp2): PASSES,
    0 overflow across all 75 MoE layers, no collapse anywhere.** PCC curve: 0.9996(L0) → smooth erosion → 0.88 trough
    (L62) → recovers to 0.9546(L77); ~45 leading layers ≥0.90. The deep erosion is **precision-limited chaining drift
    (bf4/bf16), not drops** — factor=8 ≈ masked(factor2+init_zeros) at L24/32/40 (0.9928/0.9866/0.9329 vs
    ~0.993/~0.987/0.932), so preventing the drop and zeroing it give the same deep PCC. Residual ~0.88-0.95 deep PCC
    would need higher-precision experts (bf8/fp8, kernel-blocked) — separate effort. (Caching: `GLM_USE_CACHE=1`
    writes a `.tensorbin` cache via the block's own `as_tensor` on a MISS — real-load-accurate that run — but cache
    LOAD is ~0.0006 lossy vs real-load; use real-load for accuracy, cache for fast approximate iteration.)
- **The two sections below are the DIAGNOSTIC JOURNEY — leading hypotheses that were TESTED and REFUTED before the
  combine buffer was found.** Keep for "don't re-investigate"; the bf4 over-amplification is *real but a red herring*
  for the cliff. The clamp experiment is what refuted it: `GLM_MOE_EXPERT_CLAMP=512` clamps `expert_outputs` clean to
  512 yet L7 STILL overflows (`routed_output`=3.3e38) — proving the `inf` is **not** expert magnitude. Arithmetic then
  forced the answer: gate scores can't be huge (norm_topk_prob bounds them O(1) or underflow→tiny), so the reduce must
  be summing a large value *not present in* `expert_outputs` = uninitialized buffer.
- **(REFUTED) ROOT CAUSE of the cliff = NUMERICAL OVERFLOW (`inf`), confirmed by instrumentation** (`GLM_MODEL_DIAG=1`
  logs per-layer DSA-topk overlap, MoE expert overlap, router_logits PCC, and output magnitude). NOT the
  topk (overlap stable ~0.98 through L7), NOT the MLA (router_logits PCC 0.994 at L7 → MoE input tracks),
  NOT MoE routing (expert overlap low ~0.1–0.6 but uncorrelated with the cliff — the mis-picked experts are
  low-weight; shared expert + residual absorb them). What actually happens: through L5 our magnitudes match
  the trace exactly; at **L6 a "massive activation" outlier emerges** (`max|x|` ~0.4 → ~4 in BOTH ours and
  the trace — the known attention-sink/outlier phenomenon at depth); at **L7 our hidden OVERFLOWS to `inf`**
  while the trace stays finite (`max|x|`=5.1). PCC→0 (inf vs finite) and `inf` then propagates → every layer
  after 7 is ≈0. So our reduced-precision MoE compute (bf16 activations + **`bfloat4_b` routed experts**)
  overflows on the outlier token where the fp8 GPU stays in range. It's a CHAINING artifact: teacher-forced
  block tests never hit it (each input is the bounded exact trace); only the accumulated chain reaches the
  overflow. `bfloat8_b`/`bfloat16` experts would likely add headroom but `TT_THROW` (MoE kernel is bf4-only)
  — the real blocker. A proper fix is higher-precision / clamped MoE accumulation on massive-activation tokens.
- (Earlier hypotheses — bf16 topk divergence, MoE-routing chaos — were tested and REFUTED by the diag run.)
- **PRECISE OP LOCALIZATION** (`GLM_DIAG_INF=1` magnitude probes in `tt_prefill_block.forward`, `mla._dsa_forward`,
  `tt_moe.forward`; sp4xtp2, 8-layer chain). Per-layer the routed-expert FFN output grows geometrically
  `expert_outputs` = 5 → 11 → 36 → 348 (L3→L6) then overflows; the first CLEAN inf tensor is **`routed_output`**
  (the `TtReduceModule` output) at **L7 = 3.19e38**. Decisively, at L7 the MoE INPUT (`ffn_norm_out` = 1.35) and the
  **`bfloat8_b` shared expert on that identical input (0.46) are both FINE** — only the **`bfloat4_b` routed-expert
  GEMMs** blow up. That clean shared(bf8)-vs-routed(bf4) contrast on the *same input* isolates **`bfloat4_b`'s
  shared-exponent block format** as the cause: one large weight per 16-element block forces the shared exponent up and
  craters its block-mates' precision, so the experts over-amplify the depth-accumulated outlier channel (348 already at
  L6 vs the GPU's ~O(10)). MLA is bounded throughout (`sparse_mla_out`/`after_o_proj` ~0.01–0.15). **Probe caveat:** the
  `expert_outputs`/`combined_output` probes read uninitialized dispatch/combine-buffer padding, so their `<<INF/NAN>>`
  flags fire even on good layers — trust only the dense tensors (`routed_output`, `final_output`, residual).
- **HADAMARD IS NOT THE ANSWER** (4-source research workflow, high confidence: local DS-V3.2 ref, vLLM source, canonical
  HF, DeepSeek-V3.2 deepwiki). The Walsh-Hadamard (`rotate_activation = fast_hadamard_transform(x, scale=128**-0.5)`) is
  applied to the **lightning-indexer q/k ONLY** (model.py:472-473), before fp8 `act_quant`. It is **quant-only** (absent
  from the bf16/fp32 canonical path, gated behind `use_fp8_path`; vLLM removed it from the model entirely) and
  **orthogonal** (`(Hq)·(Hk) = q·k`, L2-norm-preserving) — it spreads outliers for fp8 rounding, it does not bound any
  magnitude. So adding it would NOT fix the L7 overflow. The GPU stays finite via **fp8 e4m3 ±448 + ue8m0 per-128-block
  saturation** (`act_quant` clamp) on every quantized tensor — a magnitude bound the bf16-act + `bfloat4_b`-weight port
  has no equivalent of. There is NO qk-norm, NO attention logit soft-cap, and NO expert/swiglu clamp anywhere in
  canonical GLM math (`activation_clamp`/`swiglu_limit` exist only in `DeepseekV4MoE`, not `Glm4MoE`) — nothing to port.
- **FIX OPTIONS.** Root fix: **`bfloat8_b` routed experts** (the bf8 shared expert proves bf8 stays bounded on the same
  input) — blocked by the MoE kernel's bf4-only `TT_THROW`, i.e. kernel work. Band-aid (no kernel change):
  **`GLM_MOE_EXPERT_CLAMP=<C>`** env clamps `expert_outputs` to ±C to mimic the fp8 bound (prevents inf; may not restore
  high PCC since the bf4 experts are still numerically wrong). Note higher precision alone (fp32) does NOT help — bf16
  and fp32 share the 8-bit exponent / 3.4e38 ceiling; only *lowering* the ceiling (clamp) or *cutting the bf4 error*
  (bf8) helps.
- **The chain tracks the trace for ~7 layers, then the cliff.** Full 78-layer trend (sp4xtp2):
  `0.99958, 0.99929, 0.99926, 0.99830, 0.99634, 0.99350, 0.99270, 8e-05, …≈0… , -0.0009` (layer 77). It tracks
  beautifully for 7 layers then **collapses to ≈0 at layer 7** and never recovers (finite, just uncorrelated).
  This is **not a port bug**: layer 7 *in isolation* (teacher-forced with the exact `decoder_output_6`) scores
  0.99806 — every layer is individually correct. It is **inherent discrete-op sensitivity**: GLM has two
  discontinuous selections per layer (DSA top-2048 tokens, MoE top-8/256 experts); accumulated bf4/bf16-vs-fp8
  drift stays tiny for 6 layers, then near layer 7 flips enough selections at once that the route-scale-2.5
  MoE output swamps the residual correlation (a near-bifurcation, not gradual decay). Precision can't fix it
  (bf8 experts `TT_THROW`), so this is a property of comparing an approximate *chained* model to the *exact*
  trace — not a defect.
- **Metric (because of the above):** "all 78 ≥ PCC" is meaningless for chained MoE. The test instead counts
  **leading layers tracked at PCC ≥ `MODEL_PCC` (0.90)**, logs the trend + the divergence layer, and asserts
  `tracked ≥ min(N, MODEL_MIN_TRACK_LAYERS=5)` — proves per-layer fidelity propagates and catches a regression
  (which would move the cliff earlier). 8-layer run: `tracked 7, DIVERGED at layer 7` → PASS.
- Run the full stack: `GLM_MODEL_LAYERS=78 python -m pytest $T -k "glm_model and sp4xtp2" -s` (~5.3 h once cached).

## Key file references
- `tt/tt_prefill_block.py:25,197,226,309,340` — block class / is_moe / ttMLA(index_args) / `_build_moe` / forward
- `deepseek_v3_d_p/tt/moe/tt_moe.py:39,413,611` — TtMoe orchestrator / forward / routed+shared add
- `deepseek_v3_d_p/tt/moe/tt_moe_gate_prefill.py` — sigmoid/noaux_tc gate, single-group path, gate modes
- `deepseek_v3_d_p/utils/transformer_helpers.py:186` — canonical HF→block state_dict (MoE dict shapes)
- `deepseek_v3_d_p/reference/glm_5_1_config.py` — `GLM51Config`
- `reference_cpu/weights.py` — `load_moe_block_weights` / `resolve_layer_moe_shards`
- `tests/test_vs_gpu_ref.py` — `_run_glm_block`, the two block tests, `_load_block_trace`, `_glm_block_state_dict`
