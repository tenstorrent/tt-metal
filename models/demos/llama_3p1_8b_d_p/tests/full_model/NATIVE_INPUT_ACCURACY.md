# Full-prefill accuracy coverage

Full 2K acceptance requires both the boundary test and the all-layer native-input test for the selected cache dtypes and prompt cases. Neither test alone establishes the full milestone.

## Why there are two numerical comparisons

The raw-FP32 full-model reference starts at the prompt and follows its own 32-layer trajectory. A comparison at a later layer includes all preceding rounding error. The native-input reference instead gives each independent reference layer the exact input used by the corresponding device layer.

Independent CPU BF16 and stock-HF BF16 controls exceeded some intermediate raw-global hidden/V limits while preserving all sampled final token predictions. The native diagnostics passed all 9,792 same-input comparisons in each cache dtype and preserved final logits, token agreement, cache integrity and replay equality. These findings motivated the explicit split below. They do not justify wider numerical constants.

## Enforced checks

| Check | PCC minimum | NL2 maximum |
|---|---:|---:|
| Local hidden, BF16 cache | .999 | .025 |
| Local hidden, BFP8_B cache | .999 | .05 |
| Local K/V, BF16 cache | .9999 | .01 |
| Local K/V, BFP8_B cache | .999 | .02 |
| Final raw-global logits | .99 | .15 |

The local limits are the existing decoder limits. They account for activation/cache precision and do not grow with layer depth. Final token agreement stays top-1 >= .90 and top-5 >= .99 over the original selected positions and all 128,256 vocabulary entries.

Raw layer0 hidden and K/V remain hard because there is no preceding decoder-layer error. Later raw-global hidden/K/V, including final hidden, retain the original limits and every miss as explicit characterization. Constants are unchanged. No failure is hidden through xfail or an expected-failure marker.

The boundary test passes enforce_accumulated=False explicitly. Its first-layer checks, final logits/token checks, finite checks and structural checks remain hard. The native-input test enforces every layer against the independent reference fed that layer's exact native input. There is no label-based metric monkeypatch.

## Execution and coordinates

Layer0 uses the captured native embedding, checked exactly against the checkpoint embedding lookup. Later layers use the preceding native layer output. Both 1,024-token chunks are assembled in natural sequence order before the independent causal 2,048-token reference runs. The reference never injects values into the device.

Each complete 2K hidden tensor and each K/V head is checked both as a whole and in each 256-token SP stripe. Chunk PCCs are not averaged. Raw-global characterization stays available alongside local metrics.

Each phase interleaves slot0 and slot1 through [0,1024), then slot0 and slot1 through [1024,2048). A baseline has no capture hooks, a capture phase records native values, and a replay again has no hooks. Every phase starts with a fresh sentinel cache. Final hidden values, selected logits and decoded cache values must match exactly across phases.

All eight TP replicas per SP are compared. Raw checkpoint layer loading is checked by tensor identity. Layer order must be 0..31, and original unchanged-token, other-slot, other-plane and future-cache-row checks remain hard.

For BFP8_B, exact equality compares host-decoded tensor values, not physical packed exponent/mantissa bytes. Readbacks change synchronization; before/after replay checks reduce this blind spot but do not cover every asynchronous schedule.

The boundary companion retains 1,033/1,537-token tails, padding and the overlapping restart at 1,536. Complete chunks in the native-input test do not replace that coverage.

## Prompt and checkpoint configuration

Set LLAMA31_8B_CHECKPOINT to a local Llama-3.1-8B-Instruct checkpoint containing model.safetensors.index.json. The original boundary test honors this setting and retains its existing path fallback. The native-input test requires an explicit setting.

The native-input test has prompt_case=baseline and prompt_case=held_out. The held-out case uses fixtures/held_out_operations_chat.txt only for slot0; slot1 stays the original tea fixture. The override is supplied as one user message, rendered through the chat template once, and truncated to 2,048 tokens. A shorter rendering fails instead of silently repeating content.

The registered fixture SHA256 is afb7cc0eca086571dba6fd5fa2d1c08f93d319fd01763e486bdbe60bd5e432f6. Initial validation with the checkpoint tokenizer produced 2,081 rendered tokens; the 2,048-token prefix had 668 unique tokens and one BOS. Each report records the source hash and exact token IDs. The device launcher remains responsible for enforcing its registered source/fixture pins.

## Running and reports

Device selectors before parameter expansion:

- test_prefill_model_vs_ref.py::test_full_prefill_context_boundaries
- test_prefill_native_input_accuracy.py::test_prefill_all_layers_from_native_inputs

Collect exact parameter IDs on the allocated node. The second test has BF16/BFP8_B and baseline/held_out parameters. Run the selected cases under the normal 4x8 Blackhole mesh and ring-fabric fixtures. CPU thread counts are a runner policy; four threads are recommended for bounded oracle work.

Artifacts default to pytest's temporary directory. LLAMA_PREFILL_EVIDENCE_DIR can select another parent. New children <prompt_case>/<dtype> keep plugin/session metadata separate and prevent overwriting a prior run.

Reports separate local_misses, raw_global_misses and final_global_misses. The latter two are characterization. case_passed describes only the current test case; result_scope explains the required companion coverage. Partial counts and misses remain available when a hard check fails.

CPU-only checks run with unittest from the repository root without importing the device test:

```bash
python -m unittest -v \
  models.demos.llama_3p1_8b_d_p.tests.full_model.test_native_input_mapping \
  models.demos.llama_3p1_8b_d_p.tests.full_model.test_native_input_mutations \
  models.demos.llama_3p1_8b_d_p.tests.full_model.test_native_input_metric_policy
```

## Interpretation and cost

The known BFP8_B raw-final-hidden stripe PCC about .987577 / NL2 about .157919 remains visible as characterization. Its local layer checks, final semantics and exact replay checks are separate enforced criteria. The optional host RNE quantizer is not a native-equivalent oracle.

Teacher-forced agreement on sampled positions is not free-generation quality or coverage of all prompt distributions. Same-input checks alone can miss coherent upstream errors; exact embedding/weights, full-model semantic checks and cache invariants remain essential. Small CPU mutation probes show sensitivity to deliberate skipped/wrong-weight layers, not every arbitrarily small defect.

Per dtype and prompt case, the native-input test runs 12 native forwards, two raw-global 32-layer 2K references and 64 local reference-layer calls. Captured hidden payload is about 1 GiB plus 32 MiB embeddings. Decoded cache payload is 512 MiB in BF16 and can be 1 GiB for BFP8_B decoded to FP32. Two raw references retain about 3 GiB. These are payload estimates, not peak-memory or performance claims.
