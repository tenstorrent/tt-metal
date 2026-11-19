# Whole-QSA hidden residual candidate

`tests/multichip_hidden_candidate.py` is opt-in. It does not modify defaults.
The initial scope is representative QSA layer3; it explicitly rejects mixed-kind
`--stack`, GDN/PLE construction and component diagnostics. This is a measurement
candidate, not an accepted all-layer layout contract.

The public local tensor remains BF16 `[1,1,T,2560]`, but each rank packs four
640-wide hidden slices, one from every hyperconnection stream. The setup
adapter permutes the global host reference from `[stream,rank,640]` to
`[rank,stream,640]` before the ordinary mesh upload. Readback reverses that
permutation on the CPU. Those transformations are outside measured execution.
They are not device gathers or boundary restorations. The packed layout allows
the existing bounded chunk writer and nonaligned-length interface to remain
unchanged. CPU coordinate checks pass at T1,2,31,32,33,65 for all four ranks.

Both GR reads unpack this local packing into `[1,4,T,640]`, then perform the
existing probe's distributed per-stream normalization, exact down/up and mixed
block calculation. Residual writes consume local640 block columns directly and
repack locally. The injection projection computes all four required scalars
with the same ten1024-wide K-partial decomposition as the current default's
stream injection; use `--hidden-injection-splits 0` for the original hidden
injection control. No hidden block is immediately gathered back to2560.

QSA's existing attention, indexer, cache and selector remain intact. Only its
final output projection is intercepted to use the chosen native/fused family.
The post-projection sum is removed because the output is already hidden-sharded.
The MoE still executes gate-selected grouped experts with unchanged routing;
only its final partial sum becomes FP32 reduce-scatter into640 columns. The
router's original all-gather remains unchanged. Inputs and outputs retain the
same packed hidden layout, so two similarly adapted QSA layers could stack
without a boundary collective. Mixed kinds are not yet implemented.

Candidate families: `rs_fp32`, `fused_rs_fp32`, `ag_mm_fp32`,
`fused_ag_mm_fp32`. The initial whole-layer plan compares native RS and fused
AG-MM under HiFi4 against current-default stream ownership. The candidate keeps
only the selected row/column weight representation after setup. Fused buffers
are allocated for logical1..32 before execution so awkward tails cannot allocate
persistent state after capture. Each physical row arena has32 rows. AG families
reserve32×393216 =12582912 bytes/device; fused RS reserves
32×(327680+81920) =13107200 bytes/device, excluding allocator bank rounding.
Native RS has no extra producer persistent arena. Both use the existing expert
workspace. These are one-layer candidate overheads; adoption across all QSA
layers must update the context ledger and account for all persistent arenas.

`plans/hidden_qsa.json` contains three serialized clean timing jobs using the
same real prompt fixture and real layer3 reference: current-default control,
hidden native RS, hidden fused AG-MM. It includes prefill33, changed-input B1
trace checks and diverse B32 through the existing evidence harness. The initial
plans do not claim allocation-tracked acceptance; rerun the kept candidate with
tracking and the missing tail/context/watcher gates if its whole-layer latency
wins. Its command begins:

```bash
timeout 900 python -m models.autoports.qwen_qwen3_8_flash_next.tests.multichip_hidden_candidate \
  --family fused_ag_mm_fp32 --family-fidelity HiFi4 --hidden-injection-splits 10 \
  --mode mesh --layer 3 --reference /home/ttuser/qwen38-flash/stage5_real_l3.pt \
  --batch-fixture /home/ttuser/qwen38-flash/stage5_real_prompt_fixture.pt \
  --length 33 --capacity 128 --trace --batch 32 --repeats 32 --clean-timing \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/optimized_multichip_decoder/hidden_qsa_agmm.json
```

The current fused-family component controls automatically inherit the newly
selected stream injection10. `plans/fused_families_bestcontrol.json` reruns both
HiFi4 and LoFi matrices with fresh names. Earlier family trace results used the
old slower stream injection and cannot establish a win over the new control.
Also note that the component `--trace` currently replays an unchanged input and
records untracked timing; its limitations are listed in
`profile_and_probe_review.md`. The whole-decoder harness supplies stronger
changed-input trace checks for the integrated candidate.

No hardware experiment was run by the author of this candidate. Python syntax
compilation, formatting and host ownership round-trip checks passed. Initial
device failures require shape/layout adaptation and retry, not a rejection of
the entire coherent family.
