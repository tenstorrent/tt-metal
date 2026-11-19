# AutoFix: grouped-projection whole-trace coverage protocol

The failure is a pre-measurement harness exception, not a runtime/device failure.
`AUTODEBUG.md` independently proves the actual interleaved BinaryNg worker split
for grouped GR-down weights. The owner verified the grouped forward loop and
CoreRangeSet split against the current source. No reset or runtime change occurs.

Repair is stage-test-only: `tests/profile_fused_trace_decoder.py` uses
`profile_fused_trace_protocol.py`. The original functional harness remains
unchanged so historical coverage-source hashes stay valid. The new protocol
recognizes the exact functional and fused projector classes, validates complete
FP32 TILE DRAM weight geometry/residency, and binds the actual fused, public
forward, BinaryNg and default-subdevice sources. Fused selected weights must be
one320-row group with32-row prefill slices. Every required product has at least320
tiles; the unchanged grid cap is320 workers. Inventory remains independent of
observed profiler endpoints. All synchronization, predrain, repeated-output/state
and report checks are retained.

Host syntax and formatting checks pass. Hardware validation is pending in
`final_v5b_native_profile_plan.json`, using fresh paths for matched baseline/fused
prefill/decode captures and official reports. The earlier failed fused capture
is excluded and preserved. Runtime SHA remains
`06df08406a50a4608210cb00b99af745f29281905a9144feda928e287a09f1bd`.


Device validation: `trace_protocol_repair_control_plan_progress.json` completed all four fresh functional/fused prefill captures and strict official reports on 2026-11-13 14:14:50 UTC. No reset. Production runtime SHA was 06df08406a50a4608210cb00b99af745f29281905a9144feda928e287a09f1bd. Remaining layer/mode captures use the same repaired harness.
