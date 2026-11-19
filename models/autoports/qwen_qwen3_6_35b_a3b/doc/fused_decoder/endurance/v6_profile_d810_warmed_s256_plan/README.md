# d810 standard warmed S256 profiling plan

Prepared host-only; **no device capture executed**. Parent owns serial physical chip0. Run separately from repository root after the device slot closes:

```bash
bash models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_profile_d810_warmed_s256_plan/capture.sh 0
bash models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_profile_d810_warmed_s256_plan/capture.sh 3
```

Fresh output directories: `v6_profile_d810_warmed_s256_layer0` and `v6_profile_d810_warmed_s256_layer3`. Exact candidate `models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_transport_conv_norm_decay_transpose_interleaved_tail_integration.py.snapshot` SHA256 `d8106611f65edccdf069baab44acc9c0088baa709090b017a6a6df75f0a7bb87`. This plan makes no promotion or whole-stage acceptance claim.

The unchanged standard harness performs one complete correctness prefill warmup, one complete warmed S256/context512 measured prefill, then20 total traced decode replays (3correctness,1warm,16measured). The strict validator retains all4signposts, complete identical replay topology, full correctness/state/PCC/determinism gates, finite positive executed durations, no overlap/drop/truncated-row acceptance, rawTracy provenance, and complete tt-perf-report CSV/text outputs for both phases (active experts8). Clean warmed latency measurements remain separate ranking evidence. Timeout1800s; source pins checked before and after; Watcher/allocation tracking disabled. `TT_VISIBLE_DEVICES` must stay unset; harness selects physicalchip0.

`plan_delta_audit.json` and `host_preparation_checks.json` prove capture-script changes are paths only and validator changes are human-facing labels only. Reversing those exact replacements restores the44f87 plan byte-for-byte. Contracts change only target/path/hash/command provenance; validation logic and warning gates are unchanged. All harness/exporter/native-source pins remain exact. Both CPU preflights and shell syntax passed; neitherTorch norTTNN was imported, no device call occurred, and no capture directory was created. `pinned_inputs.sha256` binds plan/source/evidence dependencies.

## Full-layer warning review remains capture-specific

The historical full-kind capture emitted four `to_layout` warnings because composite score/index slices asked for DRAM while their intermediate was alreadyROW_MAJOR; subsequent slice operations produced the correctDRAM outputs. The entire `FusedDecoder` AST remains identical from44f87 to d810. That source identity is necessary evidence, **not authorization to accept a fresh warning**.

The strict validator will stop if this warning recurs. Preserve its failure, `capture.log`, raw exporterCSV and rawTracy files. Then separately:

1. Pin fresh candidate, native to_layout/slice source, strict validator, capture log and final rawCSV hashes.
2. Map exactly four observed warning lines/timestamps to eager score/index slices and trace-construction score/index slices; classify every other warning separately.
3. Verify all22gate instances and44score/index outputs retain original dtype, ROW_MAJOR logical geometry and requestedDRAM placement.
4. Run the three metadata-negative controls: incorrectL1 placement, wrongdtype, wronglogicalwidth must each reject.
5. Create a new capture-specific reviewed validator/evidence artifact after those checks. Keep initial validation failure intact; never reuse old raw/log hashes or blanket-allow the warning phrase.

`warning_review_contract.json` explicitly remains `pending_new_capture_review_no_warning_authorized`; fresh capture hashes are null. `historical_ab28_warning_validator.py.snapshot` is copied byte-exactly as reference only. No fresh warning exemption was introduced. Current-source topology conclusions and no-unnecessary-movement assessment remain pending actual capture, strict validation and any needed review.
