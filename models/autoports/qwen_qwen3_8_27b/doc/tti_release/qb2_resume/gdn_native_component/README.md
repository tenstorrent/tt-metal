# Guarded GDN inverse component result

Component correctness and paired timing pass. This is the generated autoport's base-checkpoint layer0 B8 complete delta operation on TP4, not a full-model or vLLM benchmark. Six alternating AB/BA pairs, each64 warmed trace replays per arm, measured median646.5804 µs for the existing path and637.5754 µs for the guarded inverse path. Median paired saving:9.05743 µs. Both arms produced identical final output and persistent state on every pair. The separate correctness run covers evolving state, native finite/nonfinite guards, B4 fallback and trace replay.

Both actual children and wrappers exited0, with151/109 passive100ms ownership samples containing no unknown owner. Sampled observation cannot rule out shorter events. Devices were empty after process exit. Exact runtime, build, command, source, timing and ownership records are included; `verified_summary.json` links their original external paths/hashes.

The required Docker build wrapper failed because Docker is unavailable. The narrow generated three-compile/two-link closure built successfully and both runtime components installed. During inspection, system Ninja1.11 removed an incompatible newer command log; retained objects/dependencies were audited and build history was not reconstructed. The original build receipt predates device validation, so its then-unverified wording is superseded only by the separately recorded actual component receipts.

The seven native source changes are applied with a default-false option. The production model does not opt in. No serving speedup, QB2 CSV acceptance or original Stage11 completion is claimed.
