# Benchmark integration repairs

## Idle metrics parser

Failing contract: select/stop only a verified idle owned server. Successful startup was followed by `float("Number")` in the idle parser. AutoDebug independently established that unanchored matching included Prometheus HELP and TYPE comments; see `AUTODEBUG.md`. The live response is now preserved in `metrics_parser_evidence.txt` and confirms the same input with both gauges zero.

Minimal fix: use full-line exact metric matching in `tools/benchmark_server.py`, retaining all-series zero and two-observation checks. No server, precision, template, or model-execution change. Discriminating checks cover comments with zero samples, nonzero samples, missing samples, multiple labeled series and two live idle observations; all pass (`metrics_parser_tests.log`).

AutoDebug's child sandbox preflight failed because this host disallows unprivileged namespaces. The explicitly authorized inspection-only child was retried with `AUTODEBUG_SKIP_CHILD_SANDBOX=1`; both launcher logs are retained. No device reset, server signal or target execution occurred in the child.

The initial setup stopped before any performance/accuracy request, and is preserved in `setup_failed_metrics/` plus `setup_probe_driver.log`. This is pre-invocation setup, not a timed benchmark attempt. The ready owned server is reused after live idle verification.

## Native-template host probe

The first local parity assertion compared a token list with the Transformers5 BatchEncoding mapping. The mapping's `input_ids` already matched the server exactly. Corrected host comparison passes; `native_template_parity.json` retains all tokens, structured messages, template hash and the repair note. No template or serving change was needed.

## Fixed-length performance transport

AutoDebug independently confirmed that upstream random text normalization and first-prompt server alignment do not guarantee exact input length; see `AUTODEBUG_TRANSPORT.md`. The failed text pilot completed all requests but sent variable-length inputs and triggered an unobservable chunked-prefill boundary. Its measurements and phase snapshot are preserved in `setup_failed_transport/`; they are invalid setup evidence, not final performance results. `AUTODEBUG_PHASE.md` explains why replacing the missing completion boundary with Python return time would be incorrect.

The model-owned `tools/benchmark_client.py` now adapts only upstream random sample preparation: original allowed-token sequences become integer-ID completion prompts, with immutable tokenizer revision, exact lengths, no special IDs and distinct hashes asserted for every request. The upstream CLI, request function, concurrency, streaming, usage parsing and metric calculations remain unchanged. Every input ID is retained beside its result. This avoids text retokenization and therefore excludes server text-tokenization cost from these synthetic performance rows. Accuracy still uses structured chat and the native template.

A host smoke initially rejected the CLI's string-valued range ratio; normalizing it with `float` fixed the assertion before any request. The repaired setup warmup completed32/32 requests with actual server usage131072 input tokens (all4096) and4096 output tokens (all128). Full profile and phase evidence will be checked before final invocation. The collector independently matches actual scheduler prompt lengths to client totals; it also rejects any non4096 input in a batch profile.

Repaired full setup evidence: all96 measured requests have4096/128 actual tokens; the collector maps96 scheduler IDs and541 completed phase records over65.956 seconds. Full-phase prefill28.253s and decode37.703s sum to the complete interval. Seven host phase tests pass (`phase_tests_transport.log`). Accuracy pilot then completed all18 responses with ordinary EOS; no blank-at-cap responses.

## Commit-hook formatting and evidence links

The repository end-of-file hook appended newlines to12 upstream performance JSON files, after roofline collection had hashed their original bytes. The mandatory evidence check then correctly failed its byte-level performance link. `run/hook_normalization.json` records each original and normalized SHA and proves that the index bytes differ only by one appended newline and all parsed JSON values are identical. `run/post_hook_failed_check.log` preserves the failure. Relinked only the performance-file hashes in run/pilot roofline metadata; model work, intervals, raw values and scores are unchanged. The client wrapper now appends the final newline after upstream benchmarking and before collector hashing, preventing recurrence. This post-measurement output-format repair requires no new hardware measurement; the source commit used for measured execution remains explicitly recorded as af2350b.
