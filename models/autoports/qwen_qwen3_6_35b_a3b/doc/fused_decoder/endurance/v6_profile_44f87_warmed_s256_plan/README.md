# Selected 44f87 standard warmed S256 profiling plan

Prepared host-only; **not executed**. Parent owns serial physical chip0. Run separately from repository root, only with no active device user:

```bash
bash models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_profile_44f87_warmed_s256_plan/capture.sh 0
bash models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_profile_44f87_warmed_s256_plan/capture.sh 3
```

Output directories are fresh `v6_profile_44f87_warmed_s256_layer0` and `v6_profile_44f87_warmed_s256_layer3`. The unchanged standard `profile_fused_snapshot.py` harness runs one complete correctness warmup, one complete warmed S256/context512 prefill, then20total/16measured warmed traced decode passes. Four signposts, correctness/state/PCC/determinism, strict executed durations, complete identical replay topology, zero overlap and tt-perf-report active-experts8 gates are retained. No partial profile or clean-latency claim is accepted. Capture timeout1800seconds; Watcher/allocation tracking are off, source bindings checked before/after. `TT_VISIBLE_DEVICES` remains unset; harness selects physical chip0.

Selected snapshot: `models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_transport_conv_norm_decay_lookup_integration.py.snapshot` SHA256 `44f87c04349836a4ad8b09c7a6c573fe3afe1e558af285e0482cce2c0fe5b2b3`. `capture_contract.json` and `pinned_inputs.sha256` bind exact current harness, exporter, validator, source and plan. The validator differs from the old ab28 strict validator only in human-facing source labels. The old plan and captures are untouched.

**Warnings require fresh review.** The earlier four to_layout warnings arose from an intermediate already-ROW_MAJOR conversion, followed by slices that correctly write DRAM. The full-kind runtime branch is source-identical, but new warnings are not automatically accepted. `warning_review_contract.json` and `historical_ab28_warning_validator.py.snapshot` preserve the exact prior logic and its native source pins. The new strict validator will stop on that warning if it recurs; preserve the failure, inspect/pin fresh log lines and CSV operation mappings, verify all44score/indexsliceoutputs, run the same three metadata-negative controls, and create a new capture-specific reviewed validator. Never edit the old evidence or reuse its raw/log hashes as current proof.

Host preparation runs only shell syntax and `--preflight` for both layers, with no TTNN import or device calls. Successful captures must produce text/CSV reports for both phases, topology/conclusions, completeness and artifact hashes. Clean warmed benchmarks remain separate ranking evidence. Current-source profiling and no-unnecessary-movement reconciliation remain pending until these captures complete and are reviewed.
