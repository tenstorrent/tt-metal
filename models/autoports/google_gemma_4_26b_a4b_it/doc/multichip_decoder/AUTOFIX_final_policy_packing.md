# AutoFix: selected BFP4 packing

The integrated default (1683b8a4) failed sliding decode PCC at 0.9912606698 despite passing cache checks and 1,024 exact replay comparisons. See sliding_final_policy_stress.json and AUTODEBUG_final_policy_packing.md.

Hypothesis: device BF16-to-BFP4 conversion did not reproduce the raw checkpoint host pack used by the passing candidate. The raw-only control sliding_final_raw_gate_control.json changed only that packing boundary and restored the entire 129-value candidate PCC sequence exactly (minimum 0.9976112404200184). Padding, ordering, effective geometry and BF16 checkpoint source were audited. Host conversion uses nearest-even; the exact device rounding rule was not measured. This proves the required loader contract without attributing an unmeasured native rounding error.

Fix: construct the sliding TP decode gate/up BFP4 tensor from the padded raw checkpoint at load time, and alias its unused TP prefill slot under the selected hybrid policy. EP prefill retains its separate existing weights. No host work was added to runtime forward or trace.

Verification: a12a913cf752b765338736dc71f71151ab972af1529a0098455755dd4f499255 default sliding_final_policy_raw_stress.json passes all 129 output comparisons, paged cache checks, all four replicas, and 128 positions times eight duplicate replays. Minimum PCC is 0.9976112404200184. full_final_policy_stress.json passes the unchanged full policy at minimum PCC 0.9994477563467595 with the same stress coverage. The failed multichip run was followed by successful bounded reset/list/mesh smoke (final_policy_reset.log and associated artifacts).

Status: fixed. Maximum-context, batch/stack, Watcher and native profile acceptance of this final policy are tracked separately.
