# AutoFix: transient CI log download timeouts

Diagnosis: see this effort's AUTODEBUG.md, produced by the bundled inspection-only launcher with personal configured login, gpt-6-astra / medium. Root reports were preserved separately. Source verification confirms the observer's 90-second timeout covers the entire gh request, failed downloads do not replace current.log, and only successful retrieval invokes the downstream observer.

Confirmation: both affected unchanged observers had recovered before additional network probes were needed. Saved successful records and independent local hash/size/UTF-8/newline/offset checks agree (private recovery-check.json). The first recovered snapshots took 1.916680 and 1.088183 seconds; subsequent saved successes corroborate recovery. The exact failing transport phase is not measured. A partial coordinator probe is not sequence-matched evidence for those timeout failures.

Repair decision: no speculative patch or retry-policy adjustment. Stop on observed transient recovery as assigned. No new network probes or changes to observer, workflow, serving source, credentials, proxy/mirrors or running jobs. No device/reset actions; device cost zero.

Outcome: retrieval recovered, cause unresolved. Successful HTTP blob transfer and local framing consistency do not establish full/current runner-log coverage, upstream fragment completeness, or future reliability. The existing archive freshness limitation remains. RESULT.json records exact recovery metadata and evidence; raw logs stay private.
