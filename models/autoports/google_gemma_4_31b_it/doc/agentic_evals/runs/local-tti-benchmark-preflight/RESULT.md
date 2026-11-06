# Stock TTI benchmark preflight

The direct stock launcher completed the unchanged stage11 matrix: eight128/128 requests at concurrency1 in27.078 seconds. MeanTTFT111.967ms; meanTPOT25.767ms. This is a short-input route preflight, not the4096/128 headline or frozen release. Full60 autoport B1 server; tokenizer resolved from pinned snapshot842da3794eaa0b77d5f08bae87a17459d91ff475. No device timing or profiling.

Initial top-level custom-weights invocation failed before requests because its downstream parser rejects snapshot model paths. The supported launcher accepts canonical model identity plus the existing runtime spec, which supplies pinned weights after parsing. No source changes. See ../ci-release-route/tokenizer-AUTOFIX.md and client-retry.log for exact argv. Client exit0; server wrapper exit1 followed deliberate shutdown with zero running/waiting requests. Devices closed before next job.
