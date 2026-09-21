# E/G compute-only sprint: final handoff

Measured on Blackhole `bh-lb-08`, reservation 223862, device 0 under the coordinator's exclusive lock, 2026-09-18. Existing host libraries were reused; every accepted private kernel was JIT-compiled and executed. No production or canonical sources were changed.

## Result

Fixed Q256/K512/D128, LoFi QK/PV, BF16 DST, canonical preparation and full compensated recurrence. The unchanged resident reader/writer runs one core, 16 Q repetitions and 512 K repetitions. Five warmups precede ten alternating AB/BA paired trace measurements; G starts BA. Useful work is 549,755,813,888 FLOPs. Upload/preparation is excluded. These are neither chip-level nor end-to-end measurements.

| Recipe | Baseline ms | Final ms | Time reduction | Baseline TF/core | Final TF/core |
|---|---:|---:|---:|---:|---:|
| E: Q7 BF16, KV RNE5/B8 | 293.062800 | 286.965899 | 2.0804% | 1.875898 | 1.915753 |
| G: Q7 BF16, KV group-RNE/saturating B4 | 293.369801 | 287.001600 | 2.1707% | 1.873935 | 1.915515 |

Sources: [E paired record](e-final-resident-v1.json), [G paired record](g-final-resident-v1.json). E baseline/candidate ranges are 293.027834–293.210297 / 286.906933–287.123186 ms; G ranges are 293.320210–293.396691 / 286.947423–287.092106 ms. Both use fresh post-recovery controls, not historical cross-machine timings.

The final candidate changes only scheduling: two-tile numerator plane copies/packs, reuse of an identical row-broadcast correction across even/odd-column SFPU vectors, and removal of one redundant balanced PACK_DONE triplet. Arithmetic order, BF16 rounding points, live FP32 residuals, exp/reciprocal, fidelity, CB formats/capacities, Q/K chunks, and double buffering remain unchanged. The denominator remains a full 32-column partial-sum vector.

## Bitwise qualification

All **14/14** final distinct-input cases pass, seven each in [E qualification](qualification-e-v1/) and [G qualification](qualification-g-v1/). Each uses Q2048/K8192/D128 on one core: eight Q jobs, sixteen distinct K blocks, and normal, common-Q, common-K, common-V, constant-V, uniform, or outlier inputs. Normal and outlier cases additionally force changing maxima; they are not plain IID-normal accuracy measurements.

Each case checks canonical adapter versus baseline, candidate versus baseline **raw bytes including signed zero**, two actual trace replays per kernel, finite outputs, exact decoded preparation oracles, unchanged original CPU/device and prepared device inputs, and selected source hashes before/after. Preparation's oracle explicitly treats signed zeros as equivalent; output equality does not. Shorter isolated fence cases additionally cover two and three K blocks with multiple Q jobs. This is bounded empirical qualification, not proof for all shapes, masks, architectures, or exceptional inputs.

Numerical weaknesses are preserved, not repaired: final E common-Q/common-K L2 is 17.8303%/46.2077%; G is 37.2678%/78.2264% on this suite. G common-V global L2 0.5406% does not establish good residual accuracy. The sprint acceptance is bitwise non-regression, not a new accuracy qualification.

## Counters and the remaining utilization gap

Separate instrumented runs use eight Q repetitions to avoid 32-bit counter overflow. The table averages the two mandatory traced observations, not eager warmups. Useful utilization divides original two-matmul FLOPs by measured zone cycles and the LoFi 4096-FLOP/core/cycle roof. It therefore does not require assuming the configured clock was the instantaneous clock. The CSV's 1350 MHz is a reported/configured value, not a sampled active-clock measurement.

| Recipe | Useful own-roof utilization, before → after | FPU active | SFPU active | Both active | Neither active |
|---|---:|---:|---:|---:|---:|
| E | 33.9244% → 34.6033% | 39.9186% → 40.7176% | 44.5259% → 44.6076% | 21.2872% → 21.5505% | 36.8426% → 36.2253% |
| G | 33.8872% → 34.6466% | 39.8749% → 40.7685% | 44.4771% → 44.6634% | 21.3970% → 21.7186% | 37.0449% → 36.2867% |

For **both** E/G, FPU-active cycles remain exactly **78,966,784**, while SFPU-active cycles decrease **88,080,864 → 86,511,072**. The reduction **1,569,792 = 384 × 511 × 8** exactly matches the eliminated correction loads per recurrent K chunk. SFPU activity percentage rises slightly because elapsed cycles fall faster than SFPU work. This independently corroborates the optimization; activity is not equivalent to useful-matmul roof utilization.

Evidence: [E profile gates](e-final-profile-v1.json), [G profile gates](g-final-profile-v1.json), raw CSVs under [E profile](profile-e-final/) and [G profile](profile-g-final/). Reproduce the standard-library analysis with:

```sh
python3 experiments/sdpa-l2/compute-sprint-v1/lowp/analyze_profile.py \
  experiments/sdpa-l2/compute-sprint-v1/lowp/profile-e-final/.logs/profile_log_device.csv
```

Why is useful LoFi utilization still only about 35%?

- LoFi reduces QK/PV arithmetic time, but does not reduce the score-exp, max/reduction, high/low compensation, unpack/copy, or pack/publication work. E/G have the same FPU and SFPU instruction workload here; G's narrower KV storage is not an extra narrow-datapath matmul speedup. This resident experiment removes recurring external KV traffic, so it cannot measure G's potential bandwidth advantage.
- The final trace still spends about 23% of cycles with SFPU active and FPU inactive. This is consistent with the substantial exact compensated recurrence and softmax work in the source, but these aggregate counters do not attribute every cycle to a specific function.
- About 36% has neither arithmetic engine active. This includes dependencies and thread/control/unpack/pack work; it is **not** proof of free overlap or a removable 36% gap. Shared DST halves and CB publication dependencies constrain scheduling. The proven extra fence was a narrow case, not permission to remove the rest.
- FPU-active time exceeds useful roof time because activity is not a count of ideal, fully productive QK/PV operations. Auxiliary reductions/scaling do not count as additional useful attention FLOPs; the activity counter alone does not establish peak issue efficiency.

The highest-information next step is bounded phase-level profiling of score exp, state correction and pack/unpack waits, followed by a source-proven scheduling change that preserves every rounding point and producer ordering. Additional vector batching or overlap may help, but no large gain is established. Changing compensation frequency/precision would be a separate numerical algorithm, outside this sprint. Safe-rescale fidelity configuration is only final-normalization work on these full-compensated paths, so it is not an attractive long-context hot-loop target.

## Rejected work and integration boundary

`setup_cache` failed G distinct-input correctness with 147,367 mismatches after the baseline matched canonical; [failure log](g-cache-distinct-v1.log). Do not integrate it. First-column-only denominator compensation was rejected before testing because all columns carry partial sums. Keep the final winner's per-group macro/replay restoration and remaining barriers.

The plane layout is presently a fixed even-D128 research specialization. General shapes, tails, masks, GQA/paged/ring paths and other architectures require explicit dispatch/generalization and qualification. No full-chip performance gain is asserted. Integrate only the three winner changes after this broader review, not the private copied headers and rejected experiment switches wholesale.

Frozen winner files:

| File | SHA256 |
|---|---|
| `combined_fence/compute_streaming.hpp` | `11f7cb8f4c508a515687702f8d8ee0916cb604f3bcd8012111e5c2a4537b5f12` |
| `block_state/compensated_block.hpp` | `875923d51d836ff3b855c6c572d789eb09daeac82646576a672f2ba21e40b224` |
| `correction_reuse/compensated_reuse.hpp` | `50087cbaaa273b5e88aa5358b2aa74bae17ca794c908039bee88dc26d049e6cc` |

Per-record manifests pin canonical recipe/preparation, private sources and selected transitive project/API headers. They are not a full toolchain/firmware dependency closure. Existing dirty user work was preserved. No device job remains active from this track.
