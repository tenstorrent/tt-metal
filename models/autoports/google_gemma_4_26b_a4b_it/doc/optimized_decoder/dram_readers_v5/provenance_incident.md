# Reader probe formatting provenance

The probe file was formatted at **18:20:03.992988UTC on2026-09-26**, while the parent-owned serialized run was active. Old probe hash `363222fac9947e216fa8045d891bb17987462acc8b7a84df06adb1cd72e71df9` became `d2da0336d4529b997b68d85773dfc5c89aa8bf9ec216f9f598e2648f5ba23266`. At18:20:04.054367UTC, an accidental `--plan-only` invocation rewrote `plan.json`. No runtime, driver or measured per-run JSON/CSV was changed.

Black26.3.1 ran in default safe mode (no `--fast`) and returned0, preserving AST equivalence; only the probe file was reformatted. An independent pre-format AST dump was not retained. The accompanying CPU tests still passed. This establishes a formatting-only incident; it does not justify claiming every process loaded the new file.

| Layer/phase | File hash observation |
| --- | --- |
| Layer0 capture entry and post-close report | Old |
| Layer0 isolated profile | Old |
| Layer5 capture entry | Old |
| Layer5 capture post-close report | New; the process had already loaded the old code |
| Layer5 isolated profile | New |

The runtime remained `169c0d97d7d0e9f35d97633f133305f1088b987ef625693d3faa100d25c3e67b`; the driver remained `71c24f1a69962f6239628fe1d138a4e6cf4827b8f60337244b54ca9d8faab5b1`. Capture-entry provenance is stored inside `capture_metadata`; the top-level capture report reads the file hash after capture closes. The layer5 distinction is retained verbatim in its artifacts.

All six capture/profile/summary commands completed with returncode0. The running driver finally rewrote `plan.json` using its original in-memory plan and start-time old probe hash, so its final status is complete again. That single hash is not a claim about all phases. [Reconstructed execution plan](reconstructed_execution_plan.json) instead binds the actual command journal, each phase's observed hashes and the native CSV hashes. [Machine-readable incident](provenance_incident.json) records exact timestamps and the transient overwritten-plan hash.

No benchmark rerun is justified solely by this formatting change. Published probes and drivers are now frozen during parent-owned execution; future changes require confirmed idle status. The measured records remain unedited.
