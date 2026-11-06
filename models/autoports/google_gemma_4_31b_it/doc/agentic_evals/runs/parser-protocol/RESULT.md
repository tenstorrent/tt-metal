# Parser protocol repair accepted

Goal: restore mini-swe-agent2.2.8 native tool calls and separate native reasoning from final content. The original server returned HTTP400 for the real SWE tools payload. Original CI run42166151590 was canceled for this proven protocol defect, not queue delay; no valid score was possible.

The coordinator applied only the three native parser flags in the new TTI model entry and local run_serving.sh: tool_call_parser=gemma4, enable_auto_tool_choice=true, reasoning_parser=gemma4. The stock plugin runtime is unchanged. Published TTI commit: da219dd56a241bb5314ae1da4b87104b84b0da15. Replacement SWE CI run:42173587147.

Prediction:0.1 device hours, a validity-work estimate, not a speedup prediction. Confirmation box:0.25 device hours. Candidate queue actual:0.105796 hours; baseline:0.063706 hours; combined:0.169502 hours. Queue evidence:device_queue/local/parser-protocol-{baseline,candidate}/job.json. Candidate client exit0; server exit1 reflects deliberate idle shutdown.

Confirmation passed at C2/B8: actual mini-swe-agent parser accepted the returned bash action printf protocol_ok; actual TerminusJSONPlainParser accepted final JSON with separate reasoning. Fenced-JSON extra-text warnings remain. Two deterministic paired requests each produced168 identical generated token IDs and the same finish reason across baseline/candidate. This exactness check plus API protocol confirmation accepts the registration change.30 focused host tests and full pytest/Ruff/format/copyright precommit hooks passed.

Evidence:candidate-audit.json, swe-baseline-response.json, paired-baseline-{0,1}.json, paired-candidate-{0,1}.json; hook log bringup/artifacts/stage12-parser-tti-hooks.log. Diagnosis and original configuration remain in AUTODEBUG.md and original-serving-entry.txt.

Change kept; parser-registration problem solved. CI scored suite validity remains pending. These checks establish API parseability and paired token equivalence, not model quality or an eval score; no same-prompt HF control was run.
