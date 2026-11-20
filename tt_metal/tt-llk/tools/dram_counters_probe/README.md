# DRAM counter probes

Read-only scripts for a Blackhole card; see `docs/performance_counters/dram_counters.md` section 6.

- `dram_idle_drift.py`: idle drift of the DRAM tile NIU counters per reader (tt_umd, exalens, exalens re-initialised).
- `dram_niu_probe.py`: do the NIU read counters of the DRAM tiles move at the header address (known host read), any architecture.
- `dram_validate.py`: known-bytes workload with `TT_METAL_PROFILE_DRAM_COUNTERS` off and on.

Run each only on an idle card nothing else has open. The scripts never write a device register. `--device` picks the card: the busy check, the exalens readers and the example `dram_validate.py` runs all use only that one.
