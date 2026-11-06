# Parser protocol investigation

Goal: determine from pinned checkpoint template and stock harness source whether missing reasoning/tool parser registration blocks requested suite protocols. Zero device-hour box. No source edits or dispatch.

Inspect resolved autoport serving args, eval suite configuration and agent API payloads. Compare expected tool-call/response representation with pinned HF chat template. Report necessary registration correction, distinguishing static protocol compatibility from live output quality.

## AutoFix plan after static diagnosis

Validity-first: restore the native mini-swe-agent tool response contract. Candidate patch adds only three explicit vllm_args to the new model entry and a focused registration test. Original entry is preserved in original-serving-entry.txt. Host resolution of candidate YAML passes; active TTI checkout remains unmodified by this effort.

Coordinator confirmation after restart: C2/B8 API probes with actual suite protocols and release generation settings (temperature1/top_p0.95/top_k20, thinking enabled, output32768). SWE must return a bash tool_call with JSON command arguments; Terminal must expose content accepted by the real TerminusJSONPlainParser. No returned commands execute. Device-hour box0.25, dispatched only by coordinator. A quality verdict would additionally require same-prompt HF control; these probes establish API acceptance/parseability only.
