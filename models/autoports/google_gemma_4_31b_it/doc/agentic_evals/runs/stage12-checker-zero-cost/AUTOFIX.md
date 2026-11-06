# Local candidate repair

AUTODEBUG.md identifies unconditional >0 validation as the defect. The candidate changes only the ranking device-hours predicate: allow zero if unblocks declares a validity problem; retain >0 for speed rows and existing ordering/estimate checks. Negative hours still fail before any reason-based ordering exception. No schema relaxation for null savings or invalid basis fields.

Local source: candidate/scripts/check_agentic_serving.py; test: candidate/tests/test_zero_device_validity.py. Source-only patch: checker-zero-cost.patch. Installed package remains untouched; installed_source.sha256 records its original hash.

Validation: seven unittest cases pass on candidate. The principal zero-device validity test fails against the installed original, confirming regression reproduction. Tests cover malformed rankings, negative hours, zero-speed denominator rejection, speed-before-validity without reason, ratio ordering, and missing measured estimates/verbose invalid basis. No device time, real pipeline gate execution or evidence edits.

Limitation: checker cannot independently verify that a provided numeric estimate labeled measured has real evidence. Existing provenance review remains required. Host-only speed efforts with zero cost are not enabled by this narrowly scoped repair and need separate ratio semantics if required. Installed-plugin update requires operator review; this candidate does not authorize updating it or declare existing stage evidence complete.
