# Compute-only frontier optimization sprint

Started 2026-09-18. Variants D/C/B/A/E/G retain the numerical definitions in
`../flux2-frontier-v1/device_attention.py`: fidelity per matmul, destination,
CB formats, recurrence/compensation, exponential/subtraction/reciprocal and
input preparation. The private Q256 correction fix remains enabled where
required. Historical benchmark mode names are not numerical authorities.

## Independent tracks

- `fp32/`: D and C, state transfer batching / pack-unpack overlap.
- `bf16/`: B and A, redundant packer configuration / compute synchronization.
- `lowp/`: E and G, compensated-state setup and LoFi compute scheduling.

Each label gets its own baseline, candidates and outcome; grouping shares code
expertise, not measurements. Root coordinates hardware, reproducibility and
the consolidated report. Canonical sources and existing dirty files are not
modified. Agent candidate trees are disjoint and uploaded independently.

## Measurement and acceptance

1. Fixed Q256/K512/D128 and original input-buffer depth. No production reader
   or writer changes. Reuse resident-input harness for compute-bound timing.
2. Device preparation uses the exact canonical recipes (especially G's native
   RNE plus saturation, not an ordinary BFP4 typecast). Preparation is outside
   compute-only timing and explicitly labeled as such.
3. Compare same-device baseline/candidate with warmups and alternating order;
   record full timing samples, source hashes, numerical defines, CB specs,
   replay equality and hardware/clock context. No extrapolated chip throughput
   presented as measured chip throughput. Useful FLOPs are the original two
   attention matmuls, excluding compensation/preprocessing overhead work.
4. Repeated resident KV is a timing instrument, not general numerical
   qualification. Distinct KV, changing maxima, common modes, outliers and
   constant-V checks compare candidate against canonical output. Pure
   scheduling candidates target bitwise equality. Any mismatch must be
   investigated and disclosed; no silent numerical redesign.
5. Profiler runs attribute waiting/setup; uninstrumented timings decide wins.
   Long-loop scaling checks distinguish dispatch/setup from sustained compute.
6. Keep failed/slower candidates as evidence. Only retained verified changes
   may be described as improvements. Confirm promising candidates with
   distinct-input/full-chip checks using unchanged data movement.

## Device safety

One IRD machine. Initially all accelerator runs serialize on the same
`/tmp/tt-device.lock` used by safe_pytest. Its automatic machine-wide reset
means concurrent chip jobs are unsafe without additional coordination.
`run_locked.sh` runs standalone benchmark scripts under that lock with a
bounded timeout and persistent dirty marker on failure; coordinator inspects
and recovers before more jobs. Never wrap safe_pytest inside this lock.
No source upload while that same candidate is being compiled or measured.

Hardware allocation and verified measurements will be added to STATUS.md.
