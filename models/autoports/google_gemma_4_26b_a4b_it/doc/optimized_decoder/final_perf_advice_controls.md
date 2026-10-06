# Current prefill advice controls

Advice controls on recorded real activations: original fidelity/shared/output controls on b585 and subsequent full-QKV geometry/placement/producer controls on daa82. Exact runtime hashes remain attached to each report. Separate-process warmed host screens and same-process alternating pairs are distinct. Selected integration/native validation is separate; this generator runs no hardware.

| Layer | Candidate | Host prefill median µs | Delta from screen baseline µs | Prefill PCC | Minimum decode PCC |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | baseline | 221091.133 | +0.000 | 0.999156094275 | 0.995257485930 |
| 0 | hifi2 | 220671.713 | -419.420 | 0.999151383685 | 0.995432064235 |
| 0 | grid110 | 220954.147 | -136.986 | 0.999156094275 | 0.995257485930 |
| 5 | baseline | 187705.330 | +0.000 | 0.999123430874 | 0.995071732300 |
| 5 | hifi2 | 187269.488 | -435.842 | 0.999122336917 | 0.995105687605 |
| 5 | grid110 | 188225.224 | +519.894 | 0.999123430874 | 0.995071732300 |
| 0 | shared_l1 | 221296.830 | +205.697 | 0.999156094275 | 0.995257485930 |
| 5 | output_l1 | 188397.882 | +692.552 | 0.999123430874 | 0.995071732300 |
| 5 | shared_l1 | 188075.827 | +370.497 | 0.999123430874 | 0.995071732300 |
| 0 | shared_grid110 | 221857.644 | +766.511 | 0.999156094275 | 0.995257485930 |
| 5 | shared_grid110 | 188432.310 | +726.980 | 0.999123430874 | 0.995071732300 |

Eight same-process alternating pairs confirm the HiFi2 gains before the long-context gate. The JSON links exact commands, fixtures, configs, all timing samples and source hashes. See [paired results](minimal_pairs_v6_summary.md).

- Retain minimalK8 HiFi4. HiFi2 paired7/8 faster but actual maximum-context position32 PCC0.9949930133358956 fails the0.995 gate. This is actual activation evidence, not a synthetic veto.
- Select minimal M2/K16/N8 HiFi2 with direct input-norm gamma output to L1, subject to selected integration validation. HiFi2 passes headline,512-step andall291maximum-context rows. M2+copiedL1 beats DRAM/M4 in8/8 pairs (-809.1405us); matchedM2 L1 beats DRAM in8/8 (-664.9255us), establishing the coupled input/output placement benefit (unspecified output memory inherits input placement). Direct producer then beats copiedL1 in22/32 pairs (-128.295us), with unchanged headline PCC. All are median paired whole-prefill host deltas, not isolated device claims.
- Retain11x8: sliding M4 grid110 paired4/8 faster (+4.3035us), full M4 loses. The adapted full M2 L1 family also tests110 directly:4/8 faster with+2.8355us median paired delta, no resolved gain. Its L1 input/output tensor specs and matmul program match the direct-producer path.
- Retain DRAM inputs for shared MLP and full output projection: allthree L1 screens pass but cost+205.697us sliding shared,+692.552us full minimal-output,+370.497us full shared versus screening baselines. These are screens, not paired gain estimates. FullQKV independently selects producerL1 after the adapted paired controls.
- Retain88active schedule: legal110-active transpose controls pass but screen at221857.644us sliding/188432.310us full, slower by766.511/726.980us than screening baselines. No source/API blocker or absolute optimum is claimed.

The [full QKV L1 family](qkv_l1_results.md) preserves the original M4 circular-buffer overlap without rejecting placement. Two legal geometry adaptations were measured; matched M2 comparisons isolate placement, grid and producer changes. The65-token case checks three tiled M rows with M2. All successful reports retain exact fixture/source hashes and program-cache guards. Candidate controls are complete; selected integration validation and native advice parsing are separate gates.

The sliding long run executes291 sampled rows and fails only position32. Aggregate PCC0.9992968877546407, end-query and decode checks pass; aggregate success does not override the individual-row failure. The identical fixture’s v5 HiFi4 control gives0.9951303778312409 at position32. The source-delta manifest connects the unchanged math path to v6. Full HiFi2 passes all291 rows with minimum0.996029643684647.

The parent acceptance driver crashed while formatting the long-run decode list as a dictionary after the child had finished. Its frozen source and partial journal are preserved; metadata_failure.json records that incident. The strict long harness and raw report retain the real failure, and the full-layer continuation has a separate command/result journal. No failure is converted to a passing return code.

Shared grid110 uses transpose multicast with perM3, gateperN14/sub3×2 anddownperN9/sub3×1. It increases padded output-tile work9.375%/5.469% but activates110 workers; the original schedule is88 active on an already configured110 grid. This schedule is tested, not dismissed using grid metadata.

Remaining: current selected integration validation and native profiles are owned by the main stage campaign.
