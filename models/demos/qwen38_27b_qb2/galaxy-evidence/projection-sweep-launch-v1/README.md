# Persistent B16 projection tuning

The complete fusion candidate sweep finished cleanly: B16/32K 16.5502 TSU,
B16/16K 18.3693 TSU, B32/32K 11.8927 TSU, B32/16K 13.8374 TSU. Every cell
has three measured repetitions; raw results and passing JUnit are retained
under `candidate-completed`. Final after-control and GPQA remain pending.

The new sweep is an independent task-owned source snapshot on 10.228.203.98.
Its systemd user service waits for the exact B16-priority invocation and clean
completion before using `/tmp/tt-device.lock`. It survives client disconnects,
but does not resume after a host reboot. It makes no serving promotion.

- 44 hardware cases: B16 first, then B32; layer-0 GDN output and MLP-down.
- Ten settings per projection plus an after-control: 1/2/3 readers per DRAM
  bank, input/storage cores, and K blocks including supported multi-shard blocks.
- Actual BFP8 weights are checked unchanged after bank repacking. Activations
  are synthetic and distinct per rank/user; this is a projection diagnostic.
- Rank-local FP32 dense references and TP-reduced references test every user.
  Candidate-to-baseline gates are PCC >= 0.99999 and relative RMS <= 0.003;
  dense-reference gates are PCC >= 0.999 and relative RMS <= 0.015. These
  projection screens do not qualify model accuracy or changed accumulation.
- Changed-input A/B/A trace replay, repeated-output hashes, and before/after
  controls reject incorrect or drifting comparisons. Timings include the
  complete current projection layout boundary and all-reduce. No host readback,
  weight upload, or compilation is included in measured repetitions.
- 30-90 minutes is an accelerator-time planning allowance after preceding jobs;
  the physical test has a 90-minute timeout and the controller a 100-minute bound.
  The service has 64 GiB memory, eight CPU cores and a 28-hour total limit.
- Preflight: 16 CPU tests passed; one hardware test collected. No hardware
  correctness or speed result is claimed for the new sweep yet.

`launch.json`, the exact source manifest, staging helper, raw test output and
verified live unit identity are preserved alongside this report. Full-model
B16/32K controls and GPQA are required before selecting any winning settings.
The 1.5-3.5 ms end-to-end saving is a target, not a measurement.
