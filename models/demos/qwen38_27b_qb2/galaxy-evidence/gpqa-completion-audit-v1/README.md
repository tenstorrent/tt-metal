# Complete-response GPQA qualification audit

Source inspection found that the existing harness scores final-answer text even
when generation ends with `finish_reason=length`. A correct parser match can
therefore receive credit before an output cutoff. The completed runs all had
zero such credits: the current native score remains 170/198, with its sole
cutoff incorrect. The previous statements about incorrect cutoffs describe
those observed runs; they were not an unconditional harness rule.

The audit now preserves the original `full_score` and separately reports
`completed_response_score`. Both retain **all 198 questions** in the denominator;
the latter assigns zero credit unless the response stopped naturally. Neither
the official harness nor any saved answer or reward was modified. The release
check requires 177 complete correct answers and cross-checks the original full
64K-output summary against the hashes and metrics of all saved responses.

Twenty-two CPU tests passed, including an apparent 177/198 harness pass that
contains one credited cutoff and must fail the completed-response gate at
176/198. Other cases cover the unchanged 177 threshold, missing questions,
source/response mismatches and replaced or incomplete predecessor jobs.

`qwen38-gpqa-completion-audit-v1-20261009.service` is persistent on .98, bounded
to 19 hours, one CPU and 1 GiB RAM. PID 1345569 and invocation
`7172e4b27dbd47868376e12d70fa2815` were observed live, waiting for the exact
head-control invocation to finish and stop its workers. The read-only audit
will then run for at most 180 seconds. It opens no devices and makes no model
calls. [launch.json](launch.json) and [source-manifest.json](source-manifest.json)
retain the command and exact source hashes; `status.json` is a waiting snapshot.

This follower does not mutate the already frozen evaluation or HF-diagnostic
controllers. Their original harness pass fields remain original results; any
release decision must also inspect the new completed-response audit. No head
score or audit result exists yet. This job survives disconnect, not reboot.
