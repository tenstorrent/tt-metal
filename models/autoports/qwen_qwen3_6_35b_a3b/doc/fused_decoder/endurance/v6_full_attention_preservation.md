# Preserve V5 full attention beside corrected V6 linear attention

Candidate `v6_corrected_fma_explicit_norm_full_v5.py.snapshot` is frozen at SHA256 `52df1b042d445630ef9bdce0d100245fa2faad9dd801042dc638b24bf3fe50d9`. It embeds the original **fused** V5 full-attention setup/runtime while preserving the corrected linear-attention source `8391717476ae224ee9b82c203d7dec21ceab69632c5edd0abd89bf8cc4daeb03`. It imports neither a decoder snapshot nor functional_decoder and has no functional fallback.

The old full-attention reference is `../selected_fused_decoder_v5.py.snapshot`, SHA256 `8ac733159befba919abf705603f3aada9d105195c7b495d8cd25aa65e531769f`, identical to current production at construction time. The candidate is for new full-kind stress/performance validation; historical qualification or 0.583570916-ms traced decode does not automatically qualify this new source.

The audit found three material full-kind changes in corrected V6 despite the linear-specific repair objective:

- Ordered router projection ran at full-kind decode boundaries because its condition did not restrict layer kind.
- Singleton MoE replaced native `generalized_moe_gate` with the inline nth-element selector for both kinds.
- `_forward` changed the first residual from the original add to `fast_and_approximate_mode=False` for both kinds.

V6 setup also allocated ordered router weights/indices, inline-selector buffers, and the SiLU table for full attention. The new candidate returns from exact V5 full-kind setup before those allocations, preserving V5's native generalized-gate resources instead.

`tests/v6_build_full_attention_preservation.py` embeds exact V5 bodies behind full-attention guards in seven methods: `from_state_dict`, `linear`, `norm`, `_moe`, `_forward`, `prepare_prefill`, and `prefill_forward`. Setup chooses the branch from `hf_config.layer_types[layer_idx]`; runtime uses immutable `self.kind`. Each full branch is proven AST-identical to the corresponding V5 method after normalizing the two shared methods' extra optional False keywords, which V5 full call sites do not supply. Removing the guards restores the **entire corrected V6 module AST exactly**, including setup, all linear methods, every helper, and the static corrected-FMA compute source.

Every other shared method is independently proven AST-identical to V5; `_linear_attention` is the intentional exception and remains exactly corrected V6, unreachable from full kind. In particular `_full_attention`, paged KV setup, positional setup, decode/decode_batch, and the shared tensor reshape helper remain unchanged. No new device operations execute in a false full-kind guard, so the linear mathematical graph is preserved; actual tracing/latency still requires measurement.

Source-only checks passed: emitted Python compiles, builder formatting/compilation passes, no snapshot/test imports or runtime exec/compile/eval are introduced, and actual emitted planners match their respective references in **440 full-kind and 440 linear-kind cases**, including nonaligned starts/lengths and chunk-size variations. The adjacent `.provenance.json` contains method-by-method AST hashes and the exact builder snapshot. No Torch/TTNN or hardware was used in preparation; all previously frozen sources remain unchanged.

Reproduction with fresh artifact paths:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_build_full_attention_preservation --output NEW_CANDIDATE_SNAPSHOT --report NEW_PROVENANCE
```

Next gates are parent-owned full-attention stress, paged/nonaligned continuation, deterministic traced replay, and complete warmed latency against the qualified V5 full-kind control. The candidate is not selected until those checks establish correctness and performance under the unchanged stage contract.
