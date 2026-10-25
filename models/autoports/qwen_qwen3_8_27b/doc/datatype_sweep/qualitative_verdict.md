# Selected precision qualitative verdict

Policy `head_bfp8_hifi2` passes the prompt-correct quality checks. All64 layers
use the normal selected-artifact construction path with trace allocation
tracking enabled. Actual outputs are in selected_validation/qualitative and
selected_validation/precision_long_quality.json; exact prompt hashes and HF
control hashes/caps are in qualitative_comparison.json.

The structured short response returns exact JSON with4 Alice apples and3 Bob
apples, reaching EOS at114 tokens. Explanation and code use the shared128-token
cap and truncate, as do the exact-prompt HF controls. The explanation correctly
attributes blue sky to scattering; the code draft uses a set and ordered output
list to remove duplicates. These bounded samples do not prove completed code
execution or full instruction compliance. They show no new failure beyond the
controlled budget truncation.

Both long narratives were read in full. The4096-token prompt ends with five
coherent paragraphs about recording observations, checking anomalies, verifying
evidence and drawing qualified conclusions; it reaches EOS at1024 tokens. The
4095-token prompt gives five coherent paragraphs with the same relevant process
and reaches EOS at760. Both leave thinking and provide final-answer prose, with
no mechanical repetition or wrong-language output. HF narrative controls use512
tokens, explicitly shorter than the TT1024 cap; the prior TT control uses1024.
No exact free-running token equality is required.

Factual4096-token retrieval returns R0101,1459,scheduled exactly and EOS at47.
The32769-token agent-history retrieval returns Cedar,7319,Mira exactly and EOS
at33. The latter HF CPU control uses the identical input tokens and completed
before any accepted timed candidate runs. These cases cover an awkward prompt
tail, but are qualitative examples rather than dataset accuracy estimates.

Five faster measured policies were rejected using unchanged prompts and caps;
selection_state.json and quality_candidates/ retain the exact reasons, outputs
and matched controls. Post-selection autoregressive, native-context and memory checks now pass; see
performance.json, memory_capacity_plan.json and precision_propagation_check.json.
