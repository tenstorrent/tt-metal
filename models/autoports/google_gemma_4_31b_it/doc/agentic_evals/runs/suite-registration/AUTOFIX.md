# Focused configuration correction

Contract: both stock smoke-test and ci-nightly routes must select the frozen two-task lists.

Initial generated-config test failed for smoke-test because the new entry only supplied a CI_NIGHTLY map. Inspection of stock route and generated config established that smoke-test does not inherit that map. Minimal correction: populate explicit identical SMOKE_TEST and CI_NIGHTLY maps in this model entry. No harness fix needed. The confirming generated-config tests pass for both suites in both modes.

A separate test expected declared max_length262144 verbatim; stock eval_command reserves 64 tokens. The expectation was corrected to262080 after inspecting `_HARNESS_CONTEXT_RESERVE`; no implementation or release output limit was reduced. All seven focused tests passed after these corrections.
