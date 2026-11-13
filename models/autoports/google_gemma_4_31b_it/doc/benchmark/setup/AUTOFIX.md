# AutoFix: current-container launch wiring

Contract: use enabled workflow roots, the pinned local checkpoint, and the actually imported stock plugin; preserve serving capability and runtime.
Diagnosis: AUTODEBUG.md identifies overwritten enabled roots and missing historical snapshot/plugin paths. The current HF cache and editable import origin exist; no download or runtime change is required.
Repair: serving_paths resolves the revision through HF_HUB_CACHE and the plugin through its import spec. Shell retains enabled roots; benchmark identity uses a command-scoped trust exception for this verified plugin checkout. Benchmark extras retain the chat/tool parsers. Response logging uses the documented true value.
Accounting: collector resolves its live socket from the observed server identity, allowing row evidence directories to differ from the server phase directory.
Validation: host syntax/import checks and benchmark accounting tests before launch; actual launch and identities are recorded in run/. No hardware result is claimed by this setup record alone.
