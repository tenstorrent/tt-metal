# Final measurement result and cleanup limitation

All87 cases and348 requests completed and passed the saved-evidence audit. The harness and measurement report are published and remote-verified (source1dec2641c4a8f54ce22286becf472349ed2bfddf; measurementse09bd1a0de971e532f04733473137217df3e4f78).

Final prefill closure passed at07:11:07UTC. The controller and publisher exited; their locks were free. A reviewed cleanup preview at07:27 showed only the expected retained decoder and base services. The next read-only ownership check failed with a login connection reset before any cleanup mutation. No base-service stop attempt or decoder owner-stop file exists.

The next observed execution time was08:49UTC, after services07:55, owners08:00, work08:05 and allocation expiries08:31/08:36. The reason for this execution gap is not established. No compute host was accessed after expiry. Shared saved evidence records decoder wrapper exit124 at07:59:59.947335. Login-only scheduler queries show an empty queue; exact terminal records are preserved in expired-job-records.json. Accounting storage is disabled.

Final process and device-holder absence was not verified before allocation loss. This is a cleanup limitation, not a measurement failure. Full completion is not claimed. Never reconnect to either expired allocation, retry final_control.py, send owner-stop, reset devices or launch models. Any further work is limited to saved evidence and publication unless the user explicitly defines a new authorized scope.
