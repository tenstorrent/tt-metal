# Final29 seed adaptation audit

CPU-only; no hardware, TTNN imports, runtime/test edits, capacity updater execution or context-contract mutation. The seed's result/command/log/watcher paths were checked against `build_jobs(prefix="final_validation29", phases=all7, context_lengths=[32768], reserve_gib=20, stress_runs=2)`: canonical68-job mapping passes.

Updated `final29_seed_template.json` points to:

- `final29_runtime_audit.json` and existing `final29_runtime_scope.md`; scope now explicitly says final29 static audit and runtime sentinel attachment must be regenerated, not inherited from final28.
- Gated, not-yet-claimed `final29_capacity.json` and planned `final_validation29` manifest/summary/context/resources.
- Original `topology_audit.md`; new `final29_interlayer_contract.md`, `final29_selection.md`, and `final29_handoff.md` retain meaningful historical evidence and identify GDN HiFi4 repair scope.
- The actual `user_wrapup28.json`, plus `user_wrapup28_runner.txt` in stop evidence. No invented final29 instruction.
- Existing native build27 receipt, because this selection change does not rebuild native code. Fresh final29 native/source snapshots remain required by the assembler/checker.

Review verdict/reports and local commit SHA/path inventory remain explicitly pending. Planned file paths are not gate-pass assertions. The repaired contract diagnostic remains diagnostic, with original source/provenance; the seed uses only planned new default-path final29 HF contract bundles.

`final29_capacity.py` syntax passes and already emits exact context status `validated` only after its artifact gates. It uses final29 resource/context/manifest/output/interlayer paths and dynamically compares installed `_STAGE5_DEFAULTS` against actual final performance policies. The installed overlay equals `final29_selected_overrides.json`; compared with final28, only GDN attention fidelity adds HiFi4 and full attention is unchanged. No stored dtype/layout or supported-context reduction is implied. The updater was not run.
