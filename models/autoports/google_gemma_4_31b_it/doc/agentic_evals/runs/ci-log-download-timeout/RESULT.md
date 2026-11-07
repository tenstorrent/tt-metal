# CI log download recovery

Both unchanged observers recovered: run 42323448662 downloaded 61,058,717 bytes in 1.916680 seconds; run 42332785769 downloaded 21,054,778 bytes in 1.088183 seconds. Both retrieval and observer exit codes were zero. Exact timestamps, SHA-256 hashes and checks are in RESULT.json and the private recovery-check.json.

These are successful complete blob downloads, not proof that the archive contains every runner event or is current. The latest events lagged fetch completion by approximately 140 and 176 seconds. Matching hashes, offsets, UTF-8 and terminal newlines establish local snapshot consistency only.

Bundled AutoDebug independently verified recovery and the stock observer control flow. Its whole-command 90-second timeout does not identify whether delay occurred at the API redirect, blob transfer or client. Failure preserves the previous successful current.log. No source defect was established; attribution remains unresolved. Prior cache experiments do not justify another cache query or policy change.

Change: none. Device hours: 0; host box: 0.25 hours; credited release saving: 0. No additional network probe, observer restart/edit, source/workflow modification, credential/network-policy change, job dispatch/cancellation, device operation or reset was performed. Existing monitoring recovered; archive freshness remains a separate limitation.
