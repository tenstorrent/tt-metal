# Non-decode SDPA usage: DiT, vision, audio, other model roots

Static audit of the current working tree on 2026-09-16. No model/kernel changes or device tests were made. Root AGENTS.md was read; this is a docs-only change.

## Scope and census reconciliation

Owns `models/tt_dit` and model roots outside the parallel language-model audit. Excludes `models/tt_transformers`, `models/common`, DeepSeek, GPT-OSS implementation roots, Gemma4, MiniMax-M3, Llama3 Galaxy, BH Qwen3.6, Falcon40b, diffusion_gemma, gated_attention_gated_deltanet, llama32_1b_quasar, and quasar GPT-OSS/Qwen3 implementation roots. The quasar **generated tests** are included here.

Reconciled against `python-census.json`: **72 direct TTNN callsites in 47 Python files**, plus **2 function references** in generated graph tests, **3 indirect `self.sdpa` wrappers**, and **15 Torch reference calls**. Direct-call families: 50 standard, 2 chunked prefill, 13 ring-joint, 4 joint, 3 experimental ring-joint. These are static source sites, not model counts, parameter combinations, or runtime invocations. The separately audited BGE custom `generic_op` implementation is additional to these public-op counts.

Tables below cover all 74 TTNN census records. Decode operator calls are excluded even when they share a file with prefill. A file whose name contains `decode` can still contain an in-scope prefill call (XTTS).

## Numeric notation and limitations

`F / D / M / E / P` means math fidelity / FP32 destination enabled / math approximation / program exp approximation / packer L1 accumulation. `T/F` means explicitly true/false; `–` means omitted at that call/config, therefore inherited from the API/config default, **not proven false**. `default` means no compute/program config supplied. Host settings do not prove every setting is consumed by every device implementation.

BF8 and BF4 mean TTNN `bfloat8_b` and `bfloat4_b`. They are block floating-point formats, not IEEE FP8. “Inherited dtype” means the call itself has no enforcing cast; it depends on its producer/runtime configuration. Weight dtype is not assumed to equal QKV activation dtype. BF16-input *typical* rows based on shared Linear/norm paths are not a guarantee that a caller cannot override module inputs.

## DiT transformer calls

Unless noted, default activations follow BF16 Linear/norm paths, not the low-bit prepared recipes E/G. Every listed standard/joint/ring path is non-causal.

| File under `models/tt_dit/` and call lines | Q/K/V | F / D / M / E / P | Contract affecting migration |
|---|---|---|---|
| `blocks/attention.py:275,308` | BF16 typical | H2/F/F/F/– | Ring or joint, rear joint tokens, logical spatial length, persistent KV buffers; Wormhole config constructor |
| `blocks/attention_opt.py:621,657` | BF16 typical | H2/F/F/F/– | Ring/joint, separately logical spatial and prompt lengths; prompt may itself be sharded; existing private experiment override hook is not public API |
| `models/transformers/attention_sd35.py:219,248` | BF16 typical | H2/F/F/F/– | Ring/joint, architecture/topology-dependent chunks; shared SDPA config defined at123–134 |
| `models/transformers/attention_mochi.py:272,301` | BF16 typical | H2/F/F/F/– | Ring/joint, separate spatial/prompt normalization and QKV, full and ring grids differ |
| `models/transformers/wan2_2/attention_wan.py:430,460,490,500` | BF16 default; ordinary BF8 all QKV in quantized **ring self-attention** | H2/F/F/F/– default; quant config can change F/D | Experimental ring, ring, standalone self, standalone cross; cross SDPA is not changed by ring quant preset; logical tail masking |
| `models/transformers/ltx/attention_ltx.py:739,775,785,798,828` | BF16 default; ordinary BF8 all QKV in quantized self-attention | H2/F/F/F/– default; profile can change F/D | Ring self, masked/non-SP self, ring cross and plain cross; rectangular Q/K, explicit masks and STG V passthrough; post-RoPE quantization |
| `models/transformers/minimax_h3/attention_minimax_h3.py:520,548,576` | BF16 typical | H2/F/F/F/– | Experimental ring, ring, standalone; host duplicates streaming eligibility and L1 estimate; experimental max-pass limit |
| `models/transformers/transformer_ideogram4.py:576,611` | BF16 typical | H2/F/F/F/– | Ring or standard masked/gather fallback; **D=256**, Q128/K256; documented ring Q256 L1 overflow |

Numerical config locations: generic `attention.py:75–85`, optimized `attention_opt.py:141–155`, Mochi126–153, SD35123–134, Wan120–162, LTX211–268, H3201–205, Ideogram4252–290. `exp_approx_mode=False` is pervasive here; this is not automatically identical to our frozen A.

Quant overrides are real existing product-level policies, not only experimental knobs:

- `pipelines/wan/quant_config.py:38–44,103–162,177–184,254`: `SDPAQuantConfig(input_dtype, math_fidelity, fp32_dest_acc)`; `all_lofi` uses raw BF16 LoFi, `all_bf8_lofi` actually leaves SDPA at **HiFi2/BF16dst with ordinary BF8 inputs**. Q/K conversion is fused into normalization; V/dummy are typecast in the ring path (`attention_wan.py:378–426`).
- `models/transformers/ltx/quant_config.py:37–106,109–125`: quant profile exposes input dtype, fidelity and dst independently. `all_bf8_lofi` also selects BF8/HiFi2 for self SDPA; `LTX_QUANT_ACTIVATIONS` gates the cast. `attention_ltx.py:726–732` casts after RoPE. Cross stays BF16/default H2.

These should not be silently renamed E: E has BF16 prepared Q, its own prescribed preparation, and compensation.

## DiT encoders and VAEs

| File under `models/tt_dit/` and call lines | Q/K/V | F / D / M / E / P | Contract |
|---|---|---|---|
| `encoders/transformer.py:596` | Inherited shared projection/norm dtype; usual BF16 | H4/T/F/F/– | Causal or explicit additive attention bias; configurable encoder model |
| `encoders/qwen25vl/model_qwen25vl.py:364` | Inherited projection/norm dtype, usual BF16 | H4/T/F/F/– | Causal or attention bias |
| `encoders/qwen3vl/model_qwen3vl.py:540` | Inherited projection/norm dtype, usual BF16 | H4/T/F/F/– | Causal or attention bias |
| `encoders/qwen3vl/vision_qwen3vl.py:539,553,581` | Inherited projection dtype, usual BF16 | H4/T/F/(–,–,F)/– | Single full SDPA, per-image sliced SDPA, or ring; ring uses **column-major CCL** and full valid local length |
| `encoders/gemma/model_gemma.py:273` | Inherited projection/norm dtype; BF16 embedding path | H2/T/F/F/T | Causal or external mask, explicit scale |
| `encoders/gemma/embeddings_connector.py:201` | Inherited input/projection output dtype; **FP32 weights do not prove FP32 QKV** | H4/T/F/F/T | Optional QK norm/RoPE, TP gathering, audio-sensitive connector; do not insert BF16 casts merely to fit a preset |
| `models/vae/vae.py:574` | Shared Linear dtype inherited, default BF16 | H2/T/F/F/– | Spatial gather plus local full attention; tuned chunks |
| `models/vae/vae_wan2_1.py:238` | **Explicit BF16 all QKV** | H2/T/F/F/– | Casts even when VAE itself operates in FP32; Q32/K256 |
| `models/vae/vae_sd35.py:322` | Inherited projection dtype, default BF16 | default | No explicit numerical knobs |
| `models/vae/minimax_h3/decoder_minimax_h3.py:184` | Constructor dtype default BF16; inherited activations | H2/F/F/F/– | Dense additive mask; **Q192/K192**, not power-of-two chunks |
| `models/audio_vae/minimax_h3/encoder_minimax_h3_audio.py:240` | **Explicit BF16 all QKV** | default | Causal, input/output audio model usually FP32 but casts before SDPA and back afterward |

Config evidence: encoder transformer472/708; Qwen2.5 270/392; Qwen3 443/568; Qwen3 vision448/462; Gemma195/201; connector42–43,93–99,195–208; VAE475–485; Wan VAE128–138; H3 decoder91–101; audio encoder238–242.

## Other production/demo model calls

| File under `models/` and call lines | Q/K/V | F / D / M / E / P | Migration notes |
|---|---|---|---|
| `demos/qwen3_vl/tt/attention.py:766,777` | Standard: explicit BF8 all; chunked: BF8 Q, cache K/V runtime-configured | H4/T/F/config/T | Chunked prefill with page table and ordinary causal/noncausal/masked prefill; config inherited from tt_transformers |
| `demos/qwen3_vl/tt/vision_attention.py:462,475` | BF8 Q/V; K `kv_cache_dtype`; chunked cache inputs runtime | Chunked H4/T/F/config/T; standard decoder-optimization configurable | Do not assume equal Q/K/V dtype. Chunked branch is copied legacy code noted in source; its presence does not establish it runs in normal vision inference |
| `demos/qwen25_vl/tt/vision_attention.py:472` | BF8 Q/V; K `kv_cache_dtype` | Decoder-optimization config; program exp false in generated example | **cu_window_seqlens** packed image-window attention; stale chunked vision branch explicitly removed |
| `demos/stable_diffusion_xl_base/tt/tt_attention.py:190` | Self QKV explicitly BF16; cross inherited projection dtype | **LoFi/F/F/F/T** | Real raw-BF16 LoFi caller, self/cross and mask; not E/G; SDXL refiner imports this implementation |
| `demos/stable_diffusion_xl_base/vae/tt/tt_attention.py:143` | Explicit BF16 projection output | H2/T/T/F/T | math approximation true but exp approximation false; config41 differs from other matmul config47 |
| `demos/vision/generative/stable_diffusion/wormhole/tt/vae/ttnn_vae_attention.py:151` | Explicit BF16 projection output | H2/T/F/F/F | Helper `ttnn_vae_configs.py:51`, Q128/K128 |
| `demos/vision/classification/vit/blackhole/tt/ttnn_optimized_sharded_vit_hiRes_bh.py:321` | **Explicit BF8 all QKV** | H4/T/F/F/T | QKV linear dtype at284; output L1, Q256/K256 |
| `demos/wormhole/owl_vit/tt/ttnn_owl_vit.py:385,652` | Explicit BF16 projection outputs | H2/F/T/**T**/T | Vision and text; text uses explicit causal additive mask but is_causal=False |
| `demos/multimodal/gemma3/tt/gemma_image_attention.py:263` | Explicit BF16 QKV projection at241 | H4/T/F/F/F inherited | Shared `configuration.compute_kernel_config_sdpa`, padded heads, optional mask |
| `demos/vision/detection/rtdetr/tt/encoder.py:599` | Runtime model dtype propagated through projections | default | Full noncausal attention, explicit scale |
| `demos/z_image_turbo/tt/dit/model_ttnn.py:291` | BF16 head outputs (source comments/producer path) | default | HiFi4 config elsewhere is **not passed to this SDPA** |
| `demos/z_image_turbo/tt/vae/model_ttnn.py:439` | Explicit BF16 QKV | default | Explicit cast despite potentially higher precision VAE intermediates |
| `demos/informer/tt/attention.py:445,622` | Runtime model dtype (non-FP32 SDPA gate); BF16/BF8 configuration dependent | default | ProbSparse selected-query rectangular SDPA with dense mask; standard full attention; **FP32 compute explicitly avoids SDPA** |
| `demos/audio/whisper/tt/ttnn_optimized_functional_whisper.py:334` | Cross Q inherited projection dtype; KV BF16 cache or produced tensors | H2/F/T/T/T | Cross attention with mask; this nondecode API can also be used for cross-attention during decode |
| same file `:372` | Encoder-self fused QKV **BF8** in nondecode path | H2/F/T/T/T | Noncausal, Q256/K256 |
| same file `:504` | Decoder-prefill fused QKV **BF8** | H2/F/T/T/T | Causal prefill; config helper249–266. The H2/F/F/F/F helper231–244 belongs to excluded decode, not this prefill call |

Qwen configuration is not inferred from variable names: `models/tt_transformers/tt/model_config.py:938–942` defines `compute_kernel_config_hifi4` as H4/FP32dst/math_approx=False/packer=True; `944–948` separately defines H4 BF16dst; `968–974` defines `compute_kernel_config_sdpa` with packer=False. `get_math_fidelity` at4980+ is runtime/config selectable, and vision callers actually use it. Existing BFP8 casts are ordinary typecasts, not five/seven-bit RNE preparation.

## BGE-M3: ordinary and custom SDPA are both live dependencies

Public call `demos/wormhole/bge_m3/tt/attention.py:226` uses runtime `score_dtype`, explicit dense mask cast to same dtype, and shape-specific config. `encoder.py:139–175` selects BF8 model dtype for B1/B16/B32 at S512 and long S8192, but deliberately uses BF16 QKV/score for B8/S512. Other shapes use BF16 score.

`tt/optimizations.py:404–424` sets:

| Shape / model dtype | Fidelity | FP32dst | Math approx | Packer |
|---|---|---|---|---|
| B1/S512 BF8 | LoFi | False | False | True |
| B1/S512 not BF8 | HiFi2 | False | False | True |
| B8/B16/B32 S512 BF8 | HiFi2 | True | False | True |
| B8/B16/B32 S512 not BF8 | HiFi4 | True | False | True |
| S8192 | LoFi | True | False | True |
| Other/default | HiFi4 | True | False | True |

Fallback config `attention.py:605` is H4/FP32dst/math_approx=False/packer=True. `_sdpa_exp_approx` at462–465 selects exp=False on Blackhole and exp=(seq_len%128==0) otherwise; `_sdpa_program_config` always forwards it through kwargs.

More importantly, `BgeM3AttentionJit._attend` at **attention.py:384** invokes `bge_encoder_sdpa_experimental`, which builds a **generic_op**, not public SDPA. `custom_ops/encoder_sdpa/op.py:595–601` fixes **LoFi**, math_approx=False and configurable dst. Its `config.py:34–45` defaults `use_streaming=False`; copied `compute_common.hpp` and `compute_streaming.hpp` plus production include fallback are dependencies (`op.py:33–42`). Live serving configurations at attention.py:363–382:

- BF4 serving: Q256/K2048, BF16dst, direct-concat output, previous-max scratch reuse, compact per-batch runtime valid lengths, single-buffered KV. Fused QKV-scatter can emit **Q/K/V all BF4** (`attention.py:305–313`); otherwise Q inherited plus BF4 KV. GQA head-fold maps 16 KV heads to32 query heads, with D64, rectangular 4096×8192 logical matrix per head.
- Other serving: Q128/K512, FP32dst, two buffers each, compact lengths optional.
- KV physical alias experiment exists but is explicitly launch-blocked as unvalidated. It must not be enabled as part of migration.

The custom baseline wrapper **`custom_ops/encoder_sdpa/op.py:672`** calls stock SDPA with **caller-supplied compute config** and program config that omits exp mode. Its supported type set includes BF16/BF8/BF4 (`op.py:84+`). It is not evidence that this exact default config runs every shape.

This custom model path cannot be mapped to G: G uses BF16 prepared Q and its own BF4 KV RNE/saturation, whereas this producer emits ordinary BF4 QKV. Deleting public legacy code does not delete this copied non-streaming algorithm, and deleting shared headers can break its include fallback.

## Experimental/model integration wrappers

| File under `models/experimental/` and call lines | Q/K/V | F / D / M / E / P | Notes |
|---|---|---|---|
| `depth_anything_v2/tt/model_def.py:751` | Explicit BF16 QKV linear at725–730 | H4/T/F/–/T, from pconfigs1126 | Dense mask optional; a program config at1082 exists but is **not passed** to this call |
| `detr3d/ttnn/multihead_attention.py:70` | Inherited runtime input/projection dtype | default | Self/cross, optional mask; no enforcing dtype cast in wrapper |
| `mistral_24b/tt/vision_attention.py:203` | Explicit BF16 projection at174 | H4/T/F/F/F inherited | Shared transformer `compute_kernel_config_sdpa`; head padding |
| `pi0/tt/ttnn_siglip.py:412` | Explicit BF16 projection at383 | H4/T/F/F/F | Config347, full noncausal |
| `pi0/tt/ttnn_gemma.py:320` | BF8 projected QKV at268; Q/K then RoPE, V retains BF8; actual RoPE output format inherited | default | GQA, explicit mask handles causality, optional concatenated KV cache; must support mixed formats rather than infer all BF16 |
| `transfuser/tt/self_attn.py:52` | Constructor dtype, default BF16 | Caller compute config or default; program default | Optimized branch only; parameters supplied externally |
| `xtts_v2/tt/ttnn_xtts_gpt_decode.py:278` | BF16 prefill input/cache path | Fidelity constructor selectable, H4 default /T/F/–/T | **Prefill**, despite filename; config helper `ttnn_xtts_gpt.py:49` |
| `tt_symbiote/modules/attention.py:84` | Wrapped caller tensors; no enforcing dtype cast | Usually H4/T/F/F/T from replacement builders; object fields overrideable | Mask unsupported, dropout0; tile/DRAM conversion but no numeric cast |
| `tt_symbiote/core/dispatchers/default_dispatcher.py:425` | TorchTTNNTensor-converted runtime input dtypes | Caller `ttnn_kwargs` or API defaults | Generic Torch-to-TT dispatch, forwarding configurable kwargs |

The three indirect census sites `tt_symbiote/modules/attention.py:161,296,513` call `self.sdpa`. At155 it starts as TorchSDPAAttention; replacement at265 installs TTNNSDPAAttention. At332 and373 other replacement builders install TTNN; numeric configs at272,339,381 use H4/FP32dst/math_approx=False/packer=True, exp=False at266/333/375. These wrappers multiply the impact of changing defaults despite having only one actual TTNN callsite.

## Tests and function-valued dispatch

| Site | Inputs/settings and coverage |
|---|---|
| `models/tt_dit/tests/unit/test_ring_joint_attention.py:264,475,1473` | BF16 standard cases, helper accepts runtime dtype; H2/math_approx=False/exp=False; first two parameterize dst (defaultFalse), sharded-prompt case False; packer explicitlyFalse in init-device configs, omitted in Wormhole branch. Covers WH/BH multi-chip grids/topologies, logical lengths, sharded joint prompts. |
| `models/tt_dit/tests/unit/test_exp_ring_joint_attention.py:191` | BF16 concrete BH-Galaxy test; helper dtype parameter; H2/BF16dst/math_approx=False/exp=False/packer=False. |
| `models/demos/qwen25_vl/tests/test_windowed_sdpa.py:198,229` | **BF8 QKV**, H4/FP32dst/math_approx=False/exp=False/packer=True. Dense **BF4 mask** versus uint32 cu_window_seqlens. |
| `models/experimental/ops/quasar/tests/gpt_oss_ops/test_scaled_dot_product_attention.py:32` | **Function reference**, invoked through graph-case harness at160–161. Captured two signatures/48 model calls; BF16 Q + BF8 KV, GQA, causal sliding window and BF16 attention sink. Compute config deliberately dropped by capture, so API defaults, expFalse Q32/K32. |
| `models/experimental/ops/quasar/tests/qwen3_vl_ops/test_scaled_dot_product_attention.py:32` | **Function reference**, graph harness at147–148. Two signatures/144 captured calls; vision BF8Q/BF16K/BF8V D64 and language BF8 QKV D128. Compute config deliberately dropped; expFalse Q256/K256. |

Generated tests therefore do **not** prove production math fidelity—the capture explicitly loses that information.

## Exclusions and interpretation

The census's 15 Torch reference calls in owned roots are golden/reference implementations, not production SDPA migration targets. These include tt_dit reference models, VAE host encoder, ring tests' golden reference, SD35 VAE golden, Qwen2.5 reference functions, and TorchSDPAAttention. `speecht5_tts` had decode-only SDPA calls. Z-image text encoder explicitly avoids SDPA for its GQA path; its comment is not a callsite. Older BERT/SD cross-attention manual matmul/softmax implementations are not users of the SDPA operator and will not disappear by replacing its internal compute implementation. No additional sparse/MLA public-op calls were found in owned roots.

## Consequences for the proposed merge

1. **Six presets cannot losslessly represent current usages.** Besides the six proposed recipes, live code uses raw LoFi/BF16, raw LoFi/BF8 FP32dst, HiFi4/BF8 FP32dst, HiFi2/FP32dst, several mixed-QKV formats, and independent exp/math approximation flags. Keep old explicit config behavior as a compatibility policy during migration; do not silently “round” it to a nearby preset. One streaming engine can still instantiate that compatibility policy.
2. **Prepared E/G must be opt-in.** Existing BF8/BF4 inputs cannot be tagged E/G merely from their dtype. Existing quantizers may be fused into QKV projections/norms, and model quant configs already expose their own accuracy/performance choices. Need clear prepared-format metadata, or a separate preparation entry point, with no hidden second quantization.
3. **Deleting non-streaming is substantially larger than enabling six recipes on D128 Blackhole.** Real users need D64/D256 and padded head dimensions, rectangular cross attention, GQA, masks, packed windows, chunked prefill caches, attention sinks, non-power-of-two chunks, joint/prompt tails, WH and BH, persistent communication buffers and column-major CCL. H3 duplicates eligibility/L1 models on host, and Ideogram already documents shape-specific L1 failures.
4. **Model performance tuning is part of compatibility.** Preserve memory placement, ring overlap, custom grids/subdevices, output dtype/layout, no-extra-allocation trace behavior, and batch/shape-specific dtype choices. Forcing every call to D/C or adding standalone E/G preprocessing can regress established bandwidth-bound models.
5. **A single core-feature PR is feasible if additive.** Combine policy definition, BF16 refactor, B/C/D, and E/G preparation in one internally staged PR, with tested explicit presets and unchanged legacy-default dispatch. A second PR can migrate public families/model usages with compatibility policies and qualification. A third can remove obsolete compute implementations **after the coverage matrix is empty**, including resolving BGE's copied-kernel dependency or explicitly excluding model-local experimental code from deletion scope.

This audit establishes source compatibility obligations, not numerical/performance equivalence. No hardware claims are made.
