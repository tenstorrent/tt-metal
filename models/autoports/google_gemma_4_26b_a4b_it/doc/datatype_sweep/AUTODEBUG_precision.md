# AutoDebug: datatype policy construction

Source-only diagnosis, 2026-09-27. No devices were opened. Baseline measurement is owned by the main agent.

## Verified cause

`Gemma4Generator.__init__` calls `Gemma4Model` without a precision policy (`tt/generator.py:25-36`); the model constructs every `MultichipDecoder` with defaults (`tt/model.py:59-61`). It fixes BF16 embeddings/head, HiFi4 head compute, BF16 logits, and BFP8 cache allocation (`tt/model.py:74-83,111-141`). Writing a selected JSON alone cannot change those tensors or kernels.

## Minimal construction boundary

Accept a strict JSON path or dictionary as `precision_config` through `build_generator` → `Gemma4Generator` → `Gemma4Model`. Resolve defaults, attention-kind defaults and layer exceptions before allocating any layer. Load the selected artifact by default when present; permit an explicit empty dictionary for the exact baseline. Reject unknown fields and unsupported fixed assumptions. Pass resolved per-layer policy to `MultichipDecoder.from_state_dict`; use the same cache dtype in its validation and the full-model allocator. Read model-head policy before uploading the original tied embedding checkpoint.

Existing legal knobs are QKV/output decode fidelity; attention CCL FP32/BF16/BFP8; MoE paired CCL BF16/BFP8; `attention_precision` baseline/qkv/output/both; and expert gate BFP4/BFP8 (`tt/multichip_decoder.py:652-713`). They are insufficient for complete weight groups, shared/expert fidelity, terminal head, activation and cache selection. Preserve the accepted geometry, hybrid EP-prefill/TP-decode choice, and prefill policy while exposing those missing decode settings.

## Baseline and runtime audit locations

| Group | Accepted policy | Actual runtime attributes |
| --- | --- | --- |
| QKV | Prefill BFP8/HiFi4; decode sliding BFP8, full BFP4; decode LoFi, FP32 accumulator | `layer.layer.self_attn.source.weights.wqkv.{weight,decode_compute,prefill.weight,prefill.compute}` |
| Attention output | Prefill BFP8/LoFi; decode BFP8/LoFi, FP32 accumulator | `self_attn.source.weights.o_proj`, `self_attn.output_compute`, `self_attn.prefill_minimal_output.{weight,compute}` |
| Routed experts | TP decode gate sliding BFP8/full BFP4; down BFP4; LoFi; BFP8 input; EP prefill gate sliding BFP8/full BFP4, down BFP4, LoFi | `layer.layer.moe.experts.decode.{gate_up,down,decode_compute,decode_activation_dtype}` and `.prefill.{prefill_gate,prefill_down,prefill_compute}` |
| Shared MLP | BF16 prefill; decode gate BFP4/down sliding BFP8/full BFP4, LoFi | `layer.layer.shared_mlp.decode_weights`, `.decode_compute`; prefill `.gate_up` and `.down` wrappers |
| Attention math | Norm/head arithmetic HiFi4 FP32; native decode SDPA sliding HiFi4/full LoFi; prefill SDPA sliding LoFi/full HiFi2 | `self_attn.compute`, `.decode_sdpa.compute`, `.prefill_attention_compute` |
| Router | BF16 checkpoint projection; direct fidelity sliding HiFi4/full LoFi; FP32 score arithmetic | `layer.layer.moe.router.projection_weight`, `.projection_compute` (verify exact wrapper names) |
| Activation/residual | Layer input/output BF16; post-attention residual FP32; expert/shared GEMM output BF16; QKV/output projection FP32 | Assert actual forward output dtypes and projection kwargs; decode QKV/shared input BFP8 switches default off |
| CCL | Attention sliding BF16/full BFP8; paired MoE sliding BFP8/full BF16 | `layer.attention_ccl_dtype`, `.moe_ccl_bfp8`; persistent pool keys include role/shape/dtype |
| KV | BFP8, 32-token pages, 128-token allocation read padding | `layer.kv_cache_dtype`, actual K/V tensors returned by `model.allocate_cache` |
| Terminal | BF16 embedding/head/logits and sampler input; FP32 final norm; HiFi4 head with FP32 accumulator | `model.{embedding,head,norm,compute}`, logits dtype; common sampler boundary |

`MultichipDecoder._forward` and `_fused_tail` fix BF16 layer outputs and FP32 intermediate residual (`tt/multichip_decoder.py:1312-1397`). Record these as validated fixed assumptions; a general “activation dtype” field would be misleading. QKV/output/shared input casts and expert decode input dtype are independent knobs.

## Quantization and mode traps

1. `_Projection.prefill` captures the initial BFP8 weight before decode narrows `.weight` to BFP4 (`tt/multichip_decoder.py:814-818`). Preserve that split when changing decode precision.
2. Attention prefill output likewise captures BFP8 before optional BFP4 decode (`:854-862`). Changing only `source.weights.o_proj` must not be reported as a prefill change.
3. Hybrid expert prefill and decode are separate objects (`:969-974`). A TP decode-only candidate must audit both, and cannot claim to change EP prefill automatically.
4. `OptimizedExperts` casts from its supplied `PackedExperts` source (`tt/optimized_decoder.py:104-112`). The current demo loader receives `dtype=BF16` and defaults each module override to that dtype (`models/demos/gemma4/tt/layer.py:74-83`), so this constructor source is clean BF16. Keep it available until candidate tensors are built. Never implement recovery by casting an existing BFP4/BFP8 candidate upward. Shared decode already packs original checkpoint tensors (`tt/multichip_decoder.py:318-340`), and sliding TP expert gate explicitly reloads the checkpoint (`:914-933`).
5. BF16/BFP8 KV are already supported by optimized cache validation (`tt/optimized_decoder.py:648-654`), but the subclass and allocator currently hardcode BFP8. Require configured dtype equality for externally supplied caches too.
6. Full-model runtime summaries must read tensor dtypes and compute objects, compare them to resolved requests, and report fixed assumptions separately. Echoing JSON proves nothing. Capture fresh execution/accuracy evidence after reconstruction; old traces and caches belong to the old policy.

## Discriminating checks

Host checks can prove strict schema rejection, per-kind/per-layer merge, selected-file loading, and exact constructor argument propagation with stand-ins; syntax compilation catches edit errors. The main agent must subsequently construct a reduced model with intentionally changed groups and compare actual attributes, execute BF16/BFP8 cache paths, and run each candidate's full-model accuracy and warmed timing. No performance or hardware correctness claim is made here.
