# B1 real-history capacity result

Passed capacity-only confirmation on source `11a387692d2b434198e79dd92fbb8ed308c3ca47` (all13 launched runtime file hashes match publication). C1/B1, full60 layers, context262144, APC and chunked prefill retained. Three unchanged native Django prefixes had prompt counts103509/105535/107474, each with128 diagnostic outputs; maximum actual total107602. No tools executed or original trajectory reproduction claimed.

Returning prefix reuse was 0.999797119 and 0.999403042, above0.9. This derives from skipped phase prefixes over previous prompt lengths, not unavailable API cache counters. All381 decode-produced native IDs matched exactly; three first tokens produced by prefill are outside decode-retention scope. No extra decode samples, preemptions, diagnostic loss/errors or pending steps.

Calls took 49.698/6.957/6.451s. Actual device cost0.076087344h including startup/closure versus forecast0.10h, box0.18h. Coordinator confirmed idle then graceful closure, remaining owners empty. Initial shutdown filename guard mismatch caused no signal; corrected guard passed.

This supports the measured singleton history capacity only; it does not accept a scored singleton route, rehabilitate quarantined scores, prove full262144/32768 capacity, or establish score/trajectory equality. Subsequent requests replay recorded real returns rather than these new diagnostic outputs. Private evidence and exact measured fields are referenced in RESULT.json; no raw text/token values copied here.
