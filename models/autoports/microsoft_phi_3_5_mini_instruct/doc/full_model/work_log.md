# Full Model Work Log

## 2026-06-15

Implemented:

- Added `tt/model.py` with the full CausalLM path: embedding, 32 optimized multichip decoder layers, final RMSNorm, split prefill/decode LM-head weights, logits padding mask, paged KV cache allocation, and page-table creation.
- Added `tt/generator.py` with the readiness `build_generator` contract, prefill/decode methods, traced autoregressive generation, teacher forcing, split sampling, trace counters, perf counters, reset, and teardown.
- Updated shared readiness helpers to handle Phi chat-template tokenization, reference provenance metadata, pinned revisions, native HF model loading, and multidevice trace-region defaults.

Reference generation:

```bash
python -m models.common.readiness_check.generate --hf-model microsoft/Phi-3.5-mini-instruct --revision 2fe192450127e6a83f7441aef6e3ca586c338b77 --no-model-trust-remote-code --prompt-source aime24 --chat-template --gen-len 100 --top-k 100 --output models/autoports/microsoft_phi_3_5_mini_instruct/readiness_aime24_chat_100.refpt
```

Artifact:

- `models/autoports/microsoft_phi_3_5_mini_instruct/readiness_aime24_chat_100.refpt`

Notes:

- Initial remote-code HF generation was rejected after it produced repeated `uter` tokens and inconsistent logits under the current Transformers cache API.
- Native Transformers Phi3 produced coherent AIME reasoning and is recorded in the reference metadata as `model_trust_remote_code=false`.

Validation:

```bash
python -m py_compile models/autoports/microsoft_phi_3_5_mini_instruct/tt/model.py models/autoports/microsoft_phi_3_5_mini_instruct/tt/generator.py
python -m py_compile models/common/readiness_check/schema.py models/common/readiness_check/generate.py models/common/readiness_check/mesh_device.py models/common/readiness_check/run_prefill_check.py models/common/readiness_check/run_teacher_forcing.py models/common/readiness_check/run_autoregressive.py
```

```bash
python -m models.common.readiness_check.run_prefill_check --model-dir models/autoports/microsoft_phi_3_5_mini_instruct --reference models/autoports/microsoft_phi_3_5_mini_instruct/readiness_aime24_chat_100.refpt --mesh-device T3K --fabric-config FABRIC_1D_RING
```

Result: top1 96/100, top5 100/100, top100 100/100.

```bash
python -m models.common.readiness_check.run_teacher_forcing --model-dir models/autoports/microsoft_phi_3_5_mini_instruct --reference models/autoports/microsoft_phi_3_5_mini_instruct/readiness_aime24_chat_100.refpt --mesh-device T3K --fabric-config FABRIC_1D_RING
```

Result: top1 91/100, top5 100/100, top100 100/100, TTFT 221.54 ms, traced decode 36.88 t/s/u.

Autoregressive:

```bash
python -m models.common.readiness_check.run_autoregressive --model-dir models/autoports/microsoft_phi_3_5_mini_instruct --hf-model microsoft/Phi-3.5-mini-instruct --hf-revision 2fe192450127e6a83f7441aef6e3ca586c338b77 --no-model-trust-remote-code --mesh-device T3K --fabric-config FABRIC_1D_RING --max-new-tokens 100 --output-dir models/autoports/microsoft_phi_3_5_mini_instruct/readiness_autoregressive
python models/common/readiness_check/check_degenerate_output.py --model-dir models/autoports/microsoft_phi_3_5_mini_instruct --scope autoregressive --missing-artifacts critical --json models/autoports/microsoft_phi_3_5_mini_instruct/readiness_autoregressive/degenerate_report.json
```

Verdict: HF and TT completions are coherent English; TT diverges early but has no repetition, wrong-language drift, or early stop. Degeneracy check found no issues.

Token-out perf:

```text
TTFT 265.98 ms
decode 38.37 t/s/u
e2e 35.14 t/s/u
trace counters: model_trace_replays=99, device_token_feedbacks=99, full_logits_decode_readbacks=0
```

Trace and fallback audit:

- Greedy split sampling is semantically greedy through the canonical sampler path, not force-argmax.
- Top-k/top-p-capable sampling was exercised with `temperature=0.7`, `top_k=8`, `top_p=0.9`; output was coherent for the short prompt.
- Same-page trace reuse did not recapture; changed page-table identity recaptured model and sampling traces.
- No single-chip, host-side decode, full-logit decode readback, or full-vocab decode all-gather fallback was found in the model/generator paths.
