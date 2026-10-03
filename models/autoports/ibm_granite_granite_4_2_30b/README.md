# Granite 4.2 30B autoport

This publication packages the runtime validated locally at
`f3d9ecd733822daf7a6380958532dd9400234680`, on tt-metal base `e476514c3d3`.
Runtime files retain their validated bytes except the generator import fallback.
The selected precision JSON differs only by its final newline.
The exact Apache-2.0 abstract contract from model-bringup 0.1.20 is included
as `tt/generator_contract.py`, used only when the installed readiness package
is absent. Its SHA-256 is `2656dad613acad8df8a392cbfa336a722181138a6a755a4a0b9535fc1ea8494a`.
The existing skill contract remains preferred when installed.

Pipeline evidence and failed attempts remain in the original bringup checkout;
this branch excludes tensor dumps, private workflow logs and raw evaluation payloads.

The checkpoint and tokenizer are `ibm-granite/granite-4.2-30b` at
`9e668ce1c538387ef24d3644e9b0606647762636`. The implementation runs all 64
layers on four Blackhole P300c chips (P300x2, TP4, logical 1x4 ring).
Supported context is 131072 shared tokens with physical batch buckets 1/8/16.
Use the committed selected precision config and a 134217728-byte trace region.

Serving requires the validated vLLM plugin commit
`dedcc6af8c628ca8fa7a5fed0fd803d33f038ae2` and inference-server commit
`ee3d121cc01aea418fde203041992d0d9931c47d`. The TTI dev selector is
`granite-4.2-30b`, implementation `granite-autoport`. Its catalog sets
`TT_MODEL_CLASS_OVERRIDES` to this autoport's `GraniteForCausalLM`, disables
prefix caching, pins both HF revisions and selects the native `qwen3_xml`
tool parser with the IBM reasoning parser shipped here.

Local Stage 1–11 and the separate bounded agentic qualification completed.
The final full Actions benchmark, accuracy and agentic suites have not run;
publication alone establishes no final-suite readiness. The preselected local
agentic pilots scored 1/2; task 022's reward 0 remains a capability failure.
A separate official Harbor container trial scored 1/1 after a preserved
cancelled transport attempt. These scopes are not interchangeable.

The warmed 4096-input/128-output primary benchmark baseline was 544.420 ms
TTFT and 46.939 output tokens/s. Three post-pass repeats were 544.949–548.381 ms
and 46.795–46.905 tokens/s; no speed improvement is claimed. Local sampling,
protocol and context validation do not replace full standard/agentic scores.
