# Valid SWE smoke workload and runtime

All91 recorded calls have native usage and assistant/tool timestamps (100% coverage), reconciling to1,884,700 prompt tokens and17,939 completion tokens, including10,684 reported reasoning tokens. This evidence characterizes the valid activation smoke only. Earlier contaminated workload/stress evidence is preserved separately and not relabeled as valid workload.

| Task | Calls | Agent seconds | Query-window sum | Tool-window sum | Final host tail |
|---|---:|---:|---:|---:|---:|
| Django |47|307.649815|303.539791|3.174954|0.935070|
| Pylint |44|342.937653|339.915476|2.338226|0.683951|

Query windows run from the previous tool timestamp to the next assistant timestamp; the first begins at Harbor agent start. mini-swe-agent2.2.8 timestamps assistant messages after API completion, cost calculation and action parsing, so windows include interstep host/request-preparation/parsing overhead and bound API duration from above. Exact client API latency is absent, not synthesized from integer server response.created. Tool windows include message persistence, subprocess execution and observation formatting; final submission is control flow. These are measured client timeline windows, never generation-only/device times.

Across91 windows, query median3.622076s, p9521.824737s, max44.074397s, sum643.455266s. Tool median0.004757s, p950.344659s, max0.428716s, sum5.513180s. Percentiles use linear interpolation over ordered observations. Detailed per-task statistics and per-call observations are in result.json/calls.json without raw transcripts.

Prompt min/median/p95/max=2206/22420/34072.5/35925; completion=30/98/652.5/1272. Largest single prompt+completion35998 tokens. Every recorded finish reason is tool_calls, with no observed cap/timeout; configured output32768/context262144/request3600s/agent21600s/tool30s remain unchanged. These observations do not establish full-cap capacity.

The650.587468 summed agent-seconds contain307.649815s overlap, yielding342.937653s agent union. Another32.050442s outside that union gives374.988095s suite wall. Each query+tool+tail partition reconciles within1microsecond. Never treat summed windows as device hours or add concurrent task times into suite wall.

Only two genuinely distinct task histories currently exist: django__django-11265 and pylint-dev__pylint-7080. Four-history availability must wait for validated nightly artifacts; no cloning or counting multiple prefixes of one task as distinct histories. Native responses retain reasoning and tool data, but serving-template reconstruction/replay admission is separate work. No added concurrency claim follows.

Phase evidence remains owned by the separate effort: ../swe-phase-accounting-confirmation/result.json; no duplicate calculations or evidence copied. Missing exact API/tool start/end, TTFT/intertoken timings and device-generation split remain explicitly null. Analysis used no devices or launches, made no central edits, claimed no speed credit, and leaves release savings unknown.
