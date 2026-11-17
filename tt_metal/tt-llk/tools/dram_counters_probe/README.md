# DRAM counter probes

Read-only scripts for a Blackhole card; see `docs/performance_counters/dram_counters.md` section 6.

- `dram_idle_drift.py`: idle drift of the DRAM tile NIU counters per reader (tt_umd, exalens, exalens re-initialised).
- `dram_validate.py`: known-bytes workload with `TT_METAL_PROFILE_DRAM_COUNTERS` off and on.

Run each only on an idle card nothing else has open. The scripts never write a device register.
