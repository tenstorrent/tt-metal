# Language/shared-model non-decode SDPA audit

Date: 2026-09-16. Static source audit of the current working checkout; no device experiments. Precision settings below are caller requests, not a claim that every backend honors each setting identically. A configured single fidelity applies to both QK and PV unless the op specializes internally; no model call in this scope explicitly selects separate QK/PV fidelity.

## Scope and coverage

Audited models/tt_transformers, models/common, demos deepseek_v3 and deepseek_v3_d_p, gpt_oss and gpt_oss_d_p, gemma4, minimax_m3, llama3_70b_galaxy, blackhole/qwen36, t3000/falcon40b; experimental diffusion_gemma, gated_attention_gated_deltanet, llama32_1b_quasar, ops/quasar/gpt_oss, ops/quasar/qwen3_vl, including their tests.

The machine-readable language-call-sites.json records **62 literal TTNN calls** across dense, chunked, ring, FlashMLA and sparse families. Function-valued aliases and monkeypatches are discussed separately below. Counts are source locations, not distinct models, runtime executions, or test cases. Imported model wrappers in tests reuse these call sites.

Decode exclusions are by called API: scaled_dot_product_attention_decode, paged_scaled_dot_product_attention_decode, and flash_mla decode are excluded even when their filename lacks “decode”. Conversely diffusion denoising/commit/packed verification using ordinary SDPA is included.

## Numerical families

Notation: F=fidelity, dst=FP32 destination flag, MA=math_approx_mode, EA=exp_approx_mode, L1=packer_l1_acc. “Default” below means an explicitly constructed model default, not necessarily TTNN op default.

| Caller family | Actual Q / K / V formats | Requested numeric configuration | Important features and evidence |
|---|---|---|---|
| tt_transformers text, common Attention1D | Q=activation_dtype (legacy fallback BFP8; common resolver can choose BF16); K/V=cache dtype, default commonly BFP8; runtime model precision overrides remain possible | Default H4/dst=true/MA=false/EA=false/L1=true; layer-level model optimization can select another tuple | GQA/MQA, causal, sliding-window, paged chunked prefill, host/device chunk offsets, batch packing. attention.py:1219,1136,1146; model_config.py:938,1668,4988; common attention_1d.py:607,1683 |
| Multimodal Llama image and Mistral vision | BF16 Q/K/V from explicitly BF16 projection output | H4/dst=true/MA=false/EA=false/L1=false | Noncausal vision, image masks, padded head dims. llama_image_attention.py:219,249; mistral_24b/vision_attention.py:172,198; model_config.py:968 |
| Multimodal Llama cross attention | BF16 Q; K/V are supplied cache tensor formats (initial projected K/V BF16; paged writes cast to actual cache dtype) | H4/dst=true/MA=false/EA=false/L1=false | Rectangular cross attention and additive mask. llama_cross_attention.py:148,192,333,356 |
| GPT-OSS; Quasar GPT-OSS copy | BF16 Q (operations.py:22); K/V explicitly converted to cache dtype (prefill.py:101–102) | Explicit model default H4/dst=false/MA=false/EA=false/L1=false; config accepts LoFi/H2/H3/H4 and boolean overrides | GQA, learned attention sink, sliding window; 32/32 or 256/256 chunks. config.py:51,95,109,125. A local activation_dtype=BFP8 for long sequence is NOT Q dtype; it controls later output projection. |
| GPT-OSS d_p dense | Q BF16 projection; KV cache dtype, ring cache hard-coded BFP8 scratch | Ordinary config same H4/BF16dst defaults; ring branch explicitly H4/dst=false/MA=false/EA=false/L1=false | Ring sink currently requires BF16dst, cache-slot/layer indexing, chunked KV, sliding-window. prefill.py:190–209; dense_sp.py:106 |
| MiniMax-M3 dense | BF16 Q/K/V from projection; cached ring reads BFP8 KV, fresh ring uses BF16 KV | Ordinary default and explicit ring H4/dst=false/MA=false/EA=false/L1=false | Ring GQA, logical_n, cache-backed and cache-free, reserved CCL column. operations.py:25; prefill.py:263; dense_sp.py:78,145 |
| Gemma4 | Projection/norm activation format, normally BF16; shared-KV and supplied paged cache dtype can differ; no SDPA-call-local forced conversion | Explicit H4/dst=true/MA=false/EA=false/L1=false | Shared KV, paged cache geometry, sliding window, batch slicing, small chunks chosen for L1. compute_config.py:34,39 return caller defaults (NOT environment overrides); operations.py:341,460; prefill.py:627,789,1105 |
| Llama3-70B Galaxy | Explicit BFP8 Q/K/V for ordinary/ring; chunked K/V actual cache dtype | H4/dst=true/MA=false/EA=false/L1=true | ring_distributed SDPA, flexible device chunk offset, causal, Galaxy grid/partition. llama_attention.py:1174,1179,1216; model_config.py:757 |
| Blackhole Qwen3.6 TP dense | Variables q8/k8/v8 are aliases of Q/K/V, NOT BFP8 casts; BF16-normalized/rotated QK and projection activation V | Compute config omitted (TTNN default); EA=false explicit | D256 and L1-resident activations, 64/128 chunks; comment records 256-chunk resident-L1 clash. tp.py:444–454 |
| Blackhole Qwen3.6 TP chunked | Q explicitly BFP8 at tp.py:739; KV supplied paged cache (generated BFP8 in current setup) | H2/dst=true/MA=true/EA=false/L1=true via tpc.COMPUTE_HIFI2 | Device/scalar offset, paged cache. tp_common.py:22; tp.py:776,786 |
| Qwen3.6 vision | Q/V BFP8, K=kv_cache_dtype | Layer-configured SDPA_PREFILL (default shared H4+FP32dst tuple) | Noncausal vision; vision_attention.py:367–383 |
| Falcon40B T3000 | Projection output model-config dtype; K/V fed to SDPA are NOT the separately cache-cast tensors | Compute config OMITTED; EA omitted in SDPAProgramConfig | Stock defaults must be retained; projection COMPUTE_KERNEL_CONFIG must NOT be misattributed to SDPA. falcon_attention.py:311,339,350; model_config.py:651–682,868 |
| Diffusion Gemma denoise and commit | Normally BF16 activations/cache, wrapper forwards actual tensors | Dense denoise/commit compute config omitted; denoise EA=false by default but DG_SDPA_EXP_APPROX can enable it | Rectangular noncausal explicit-mask attention, scale=1; grid/chunks configurable. diffusion_attention.py:113,124,207; commit_batched.py:784 |
| Diffusion Gemma chunked/prefill helpers | Forwarded activation and supplied cache dtype, normally BF16 | Explicit H4/dst=true/MA=false/L1=false; EA=false for chunked config; sliding helper leaves program config omitted | Paged cache offsets, sliding-window workaround via square SDPA, padded commit batches. chunked_prefill.py:154,415; commit_batched.py:670 |
| Gated attention / Gated DeltaNet | BF16 Q/K/V explicitly retained, supplied cache dtype for paged branch | H2/dst=true/MA=false/EA=false/L1=false | D256, GQA, chunked flexible offsets, rectangular masked noncausal calls. ttnn_gated_attention.py:154,285,306,317,538 |
| Quasar Llama Attention1D | Resolved Q activation dtype (default BF16), KV default BFP8 but configurable | Explicit resolved H4/dst=true/MA=false/EA=false/L1=true unless model supplies another tuple | Separate experimental.quasar API; not automatically same backend as production. attention_1d.py:541,553,566,1639,1649 |
| Quasar Qwen3-VL | Text QKV explicit BFP8; vision Q/V BFP8, K configured cache dtype | Text fixed configuration.compute_kernel_config_hifi4; vision layer SDPA_PREFILL; shared default H4+FP32dst/MA=false/L1=true, EA=false | Text causal/noncausal + mask, vision noncausal or chunked, GQA. attention.py:708–714,766,777; vision_attention.py:450–475 |
| DeepSeek-v3 FlashMLA prefill | Q projection/concat activation dtype (no local cast; normal BF16 path); unpaged KVPE inherited activation; paged KVPE explicitly BFP8 cache | Explicit H4/dst=false/MA=false/EA=false/L1=false | MLA QKdim576, latent Vdim512, different K/V interpretation, paged MLA. mla1d.py:510–531,1407,1664,1828 |
| DeepSeek-v3 d_p dense ring_joint/ring_mla | Q default BF16; dense KVPE BFP8 cache; materialized V defaults BFP8 (tuned matmul output dtype can override) | Explicit H2/dst=false/MA=false/EA=false/L1=true | QK576 vs materialized V128 or latent V512; ring cache/metadata, logical padding/balancing. mla.py:427,704,796,1005,1503 |
| DeepSeek heavily compressed attention | Constructor dtype default BF16 Q/KV/mask; configurable activation dtype | Compute config omitted; EA=false explicit | Dense masked SDPA with attention sink and compressed+sliding+carry KV concatenation. heavily_compressed_attention.py:73,168,414,436 |
| DeepSeek sparse_sdpa | BF16 row-major Q; KV cache format selectable BF16 row-major / FP8 E4M3 row-major packed storage | Compute config omitted at call; backend format-dependent defaults | Indices, block-cyclic SP, KV-dedup TP, latent Vdim512, selected top-k. mla.py:358,1786. This FP8 is NOT BFP8_B. |
| MiniMax sparse_sdpa_msa | BF16 row-major Q, tiled K/V; BF16 fresh K/V and BFP8 cached K/V | Compute config omitted at call; backend defaults | Block indices, token-level causal mask, block-cyclic SP; BF16 Q explicitly required by causal path. msa.py:149–166 |

No statically selected HiFi3 SDPA call was found in these model roots. GPT-OSS/MiniMax config validators permit it, so it is a supported runtime setting, not a proven active default. HiFi3 in GPT top-k/router and Diffusion Gemma DG_HIFI linear/sparse_matmul experiments is not SDPA.

Likewise LoFi in DeepSeek prefill projection config is not FlashMLA fidelity: the immediately following FlashMLA config explicitly selects HiFi4. GPT-OSS/MiniMax accept LoFi via runtime config, but the scanned defaults select HiFi4.

## Direct test specializations

- DeepSeek_d_p test_ring_joint_mla.py:350,802: Q BF16, KV BFP8; QKdim576/V128; explicit H2 default, dst=false, MA=false, L1=false, EA=false. Includes causal/balanced and multiple ring meshes.
- MiniMax test_ring_joint_sp_vs_ref.py:105: all BF16, explicit H4/BF16dst, both approx flags false, L1=false.
- MiniMax test_msa_prefill_vs_ref.py:76: Q row-major BF16, KV tiled BF16; sparse op compute default.
- Diffusion Gemma test_attention.py:894: BF16 QKV/mask, explicit H2/BF16dst, MA=false, EA=false, L1=false.
- Diffusion Gemma test_commit.py:276: BF16 cache/query, compute config omitted, EA=false, 8x1 grid, Q=CANVAS/K32.
- Quasar Llama ops and prototype_ops dense/chunked tests: BF16 Q/K/V (op_utils.py:88), explicit H4/FP32dst/MA=false/EA=false/L1=false; tests are source duplicates and not four independent model recipes.
- Quasar Llama graph_ops/test_scaled_dot_product_attention.py:32 stores the API as _OP; G.run_case executes it at :95. Captured case is Q BF16, KV BFP8, Qheads32/KVheads8/D64/S1024; compute config omitted, EA=false, Q/K64. This extra aliased invocation is not in the 62 literal-call count.
- Common attention test architecture defaults pin H4/FP32dst/MA=false/L1=true (test_attention_1d_arch_config.py:135); mock chunked function at test_attention_1d.py:862 is not a real device call.

## Aliases, wrappers, references and exclusions

Diffusion Gemma prefill_moe.py:70 saves _original_sdpa; :389 forwards through that alias when its contextual condition is false. :399 monkeypatches the public API; causal scale=1 calls under the active DG context can execute staged GQA matmuls instead. The source documents a QB2 causal SDPA deadlock. This is an important real integration workaround, not evidence that every lexical dense-SDPA call runs the kernel.

Torch F.scaled_dot_product_attention and torch.nn.functional.scaled_dot_product_attention in reference files/tests are golden implementations only; Falcon's TT_functional.scaled_dot_product_attention is a Torch implementation despite its name. These are not TTNN migration sites. Documentation strings and markdown examples are likewise excluded from executable counts.

Cross-checked against the root agent's independent python-census.json. Its additional TTNN references in these roots are the Diffusion Gemma saved original above and the Quasar graph-test function-valued alias above; no additional literal TTNN call was missing. DeepSeek-v3_b1 fused attention_block/decoder_block/post_sdpa `sdpa` names are runtime-context dictionaries, not API invocations; attention_block/op.py:1580 instantiates FlashMLADecode.ProgramConfig, so these custom decode micro-op paths are excluded. Falcon's reference implementation is locally defined at hf_modeling_falcon.py:224 and consists of Torch matmul/softmax/dropout.

Sparse_sdpa and sparse_sdpa_msa already use compute_streaming primitives but include compute_common.hpp for shared declarations/helpers (sparse_sdpa_compute.cpp:20–23 and sparse_sdpa_msa_compute.cpp:11–14). Therefore deleting the entire compute_common.hpp is not safe even after deleting its old dense algorithm. Non-decode FlashMLA and ring_mla are actual migration requirements, not dismissible decode exceptions.

## Merge implications

1. **The six user presets cannot be the only representable internal recipes.** Existing H4/BF16dst and H2/FP32dst have no exact point among A/B/C/D/E/G. Preserve legacy compute_kernel_config semantics when precision is omitted; expose presets as a recommended layer. Do not map H4/BF16dst to A or D silently.
2. **Support preexisting low-precision inputs independently from E/G.** Existing all-BFP8 QKV, mixed BF16-Q/BFP8-KV and FP8-E4M3 sparse caches did not use our prescribed preparation. Treating them as prepared E/G is numerically wrong. Automatic preparation cannot recover bits discarded in upstream projection/cache.
3. **Coverage is much wider than D128 noncausal BF16.** Mandatory migration axes include D64/D256/MLA576, V dimension differing from QK, GQA/MQA, additive masks/sinks, sliding window, flexible paged offsets, cache geometry, ring logical padding, and non-Blackhole/Quasar backends.
4. **L1 admission must account for resident model tensors.** Qwen3.6 explicitly limits chunks because an isolated winning 256 chunk clashes with live model L1 allocations. New FP32/compensated states and preparatory tensors change that budget.
5. **Keep numerical policy distinct from cache/transport format and algorithm semantics.** Ring SP cache dtype and sparse row-major FP8 have storage/communication contracts; presets must not unexpectedly rewrite persistent cache ownership.
6. **Suggested three-PR staging matches user preference:** combined recipe/state-policy/API/preparation implementation while preserving compatibility; then migrate remaining caller/transport/features; then delete obsolete non-streaming implementations once capability parity is measured. First PR need not bulk-edit every model call just to preserve behavior.

## Complete literal call-site inventory

| Source location | API | Class |
|---|---|---|
| `models/tt_transformers/tt/attention.py:1250` | `scaled_dot_product_attention` | model |
| `models/tt_transformers/tt/attention.py:1226` | `chunked_scaled_dot_product_attention` | model |
| `models/tt_transformers/tt/attention.py:1237` | `chunked_scaled_dot_product_attention` | model |
| `models/tt_transformers/tt/multimodal/llama_cross_attention.py:356` | `scaled_dot_product_attention` | model |
| `models/tt_transformers/tt/multimodal/llama_image_attention.py:249` | `scaled_dot_product_attention` | model |
| `models/tt_transformers/tt/multimodal/mistral_24b/vision_attention.py:198` | `scaled_dot_product_attention` | model |
| `models/common/modules/attention/attention_1d.py:647` | `scaled_dot_product_attention` | model |
| `models/common/modules/attention/attention_1d.py:625` | `chunked_scaled_dot_product_attention` | model |
| `models/common/modules/attention/attention_1d.py:636` | `chunked_scaled_dot_product_attention` | model |
| `models/demos/deepseek_v3/tt/mla/mla1d.py:1664` | `flash_mla_prefill` | model |
| `models/demos/deepseek_v3/tt/mla/mla1d.py:1828` | `chunked_flash_mla_prefill` | model |
| `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py:802` | `ring_joint_scaled_dot_product_attention` | direct test |
| `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py:350` | `ring_joint_scaled_dot_product_attention` | direct test |
| `models/demos/deepseek_v3_d_p/tt/mla/heavily_compressed_attention.py:436` | `scaled_dot_product_attention` | model |
| `models/demos/deepseek_v3_d_p/tt/mla/mla.py:1005` | `ring_mla` | model |
| `models/demos/deepseek_v3_d_p/tt/mla/mla.py:1503` | `ring_joint_scaled_dot_product_attention` | model |
| `models/demos/deepseek_v3_d_p/tt/mla/mla.py:1786` | `sparse_sdpa` | model |
| `models/demos/gpt_oss/tt/attention/prefill.py:147` | `scaled_dot_product_attention` | model |
| `models/demos/gpt_oss_d_p/tt/attention/dense_sp.py:106` | `ring_joint_scaled_dot_product_attention` | model |
| `models/demos/gpt_oss_d_p/tt/attention/prefill.py:36` | `scaled_dot_product_attention` | model |
| `models/demos/gemma4/tt/attention/operations.py:483` | `scaled_dot_product_attention` | model |
| `models/demos/gemma4/tt/attention/operations.py:385` | `chunked_scaled_dot_product_attention` | model |
| `models/demos/gemma4/tt/attention/operations.py:399` | `chunked_scaled_dot_product_attention` | model |
| `models/demos/gemma4/tt/attention/prefill.py:1114` | `scaled_dot_product_attention` | model |
| `models/demos/gemma4/tt/attention/prefill.py:670` | `scaled_dot_product_attention` | model |
| `models/demos/gemma4/tt/attention/prefill.py:706` | `scaled_dot_product_attention` | model |
| `models/demos/gemma4/tt/attention/prefill.py:796` | `scaled_dot_product_attention` | model |
| `models/demos/minimax_m3/tests/unit/test_msa_prefill_vs_ref.py:76` | `sparse_sdpa_msa` | direct test |
| `models/demos/minimax_m3/tests/unit/test_ring_joint_sp_vs_ref.py:105` | `ring_joint_scaled_dot_product_attention` | direct test |
| `models/demos/minimax_m3/tt/attention/dense_sp.py:78` | `ring_joint_scaled_dot_product_attention` | model |
| `models/demos/minimax_m3/tt/attention/dense_sp.py:145` | `ring_joint_scaled_dot_product_attention` | model |
| `models/demos/minimax_m3/tt/attention/msa.py:153` | `sparse_sdpa_msa` | model |
| `models/demos/minimax_m3/tt/attention/prefill.py:327` | `scaled_dot_product_attention` | model |
| `models/demos/llama3_70b_galaxy/tt/llama_attention.py:1225` | `ring_distributed_scaled_dot_product_attention` | model |
| `models/demos/llama3_70b_galaxy/tt/llama_attention.py:1245` | `chunked_scaled_dot_product_attention` | model |
| `models/demos/llama3_70b_galaxy/tt/llama_attention.py:1275` | `scaled_dot_product_attention` | model |
| `models/demos/blackhole/qwen36/tt/attention/tp.py:453` | `scaled_dot_product_attention` | model |
| `models/demos/blackhole/qwen36/tt/attention/tp.py:776` | `chunked_scaled_dot_product_attention` | model |
| `models/demos/blackhole/qwen36/tt/attention/tp.py:786` | `chunked_scaled_dot_product_attention` | model |
| `models/demos/blackhole/qwen36/tt/vision/vision_attention.py:377` | `scaled_dot_product_attention` | model |
| `models/demos/t3000/falcon40b/tt/falcon_attention.py:350` | `scaled_dot_product_attention` | model |
| `models/experimental/diffusion_gemma/tests/test_attention.py:894` | `scaled_dot_product_attention` | direct test |
| `models/experimental/diffusion_gemma/tests/test_commit.py:276` | `scaled_dot_product_attention` | direct test |
| `models/experimental/diffusion_gemma/tt/chunked_prefill.py:161` | `scaled_dot_product_attention` | model |
| `models/experimental/diffusion_gemma/tt/chunked_prefill.py:422` | `chunked_scaled_dot_product_attention` | model |
| `models/experimental/diffusion_gemma/tt/commit_batched.py:662` | `chunked_scaled_dot_product_attention` | model |
| `models/experimental/diffusion_gemma/tt/commit_batched.py:784` | `scaled_dot_product_attention` | model |
| `models/experimental/diffusion_gemma/tt/diffusion_attention.py:222` | `scaled_dot_product_attention` | model |
| `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py:306` | `chunked_scaled_dot_product_attention` | model |
| `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py:317` | `chunked_scaled_dot_product_attention` | model |
| `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py:538` | `scaled_dot_product_attention` | model |
| `models/experimental/llama32_1b_quasar/modules/attention/attention_1d.py:553` | `quasar.chunked_scaled_dot_product_attention` | model |
| `models/experimental/llama32_1b_quasar/modules/attention/attention_1d.py:566` | `quasar.scaled_dot_product_attention` | model |
| `models/experimental/llama32_1b_quasar/tests/ops/test_chunked_scaled_dot_product_attention.py:89` | `quasar.chunked_scaled_dot_product_attention` | direct test |
| `models/experimental/llama32_1b_quasar/tests/ops/test_scaled_dot_product_attention.py:72` | `quasar.scaled_dot_product_attention` | direct test |
| `models/experimental/llama32_1b_quasar/tests/prototype_ops/test_chunked_scaled_dot_product_attention.py:89` | `quasar.chunked_scaled_dot_product_attention` | direct test |
| `models/experimental/llama32_1b_quasar/tests/prototype_ops/test_scaled_dot_product_attention.py:72` | `quasar.scaled_dot_product_attention` | direct test |
| `models/experimental/ops/quasar/gpt_oss/tt/attention/prefill.py:147` | `scaled_dot_product_attention` | model |
| `models/experimental/ops/quasar/qwen3_vl/tt/attention.py:766` | `chunked_scaled_dot_product_attention` | model |
| `models/experimental/ops/quasar/qwen3_vl/tt/attention.py:777` | `scaled_dot_product_attention` | model |
| `models/experimental/ops/quasar/qwen3_vl/tt/vision_attention.py:462` | `chunked_scaled_dot_product_attention` | model |
| `models/experimental/ops/quasar/qwen3_vl/tt/vision_attention.py:475` | `scaled_dot_product_attention` | model |

## Verification

Read-only source search plus Python AST census; JSON parses successfully. Documentation-only artifacts added with apply_patch; no production code, weights, reservation state, or kernels changed. No accelerator tests or build were necessary for this audit.
