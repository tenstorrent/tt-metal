# GPQA smoke workload extraction

Coordinator goal: extract defensible real smoke workload inputs for next-rung forecasts from the completed GPQA job. Zero device hours; no device commands, source changes or central-document edits.

Use only the final Docker server-log section of bringup/artifacts/stage12-gpqa-job.log, excluding duplicated live console output. Parse JSON events after the post-health06:37:44 first prefill through final response; count sampling submissions, row occupancy and startup/workload trace events separately. Preserve source hash and line boundaries. Identify prompt sizes from zero-start prefill chunks, and distinguish physical padded rows from logical prompt tokens. Do not equate async decode submissions with emitted output tokens or infer final working-set size without exact usage. Record unavailable values and forecast limitations explicitly. Write RESULT.json once after inspection.
