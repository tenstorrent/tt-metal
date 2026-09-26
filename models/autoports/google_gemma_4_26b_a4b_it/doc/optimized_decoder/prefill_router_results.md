# Prefill router advice controls

Four separate phase-matched controls, exact real4096/128 inputs, 32 alternating same-process pairs each. Whole-prefill host timing includes final synchronization; selection, setup, warmup and output deallocation are excluded. Device-native metrics remain separate.

| Layer | Change | Baseline µs | Candidate µs | Median paired delta µs (%) | Faster pairs | Paired IQR µs | Prefill PCC | Min decode PCC |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 0 | hifi2 | 220355.035 | 220321.671 | -67.093 (-0.0304%) | 20/32 | [-188.863, +90.879] | 0.999151434476 | 0.995257485930 |
| 0 | producer_l1 | 220321.030 | 220347.970 | +3.872 (+0.0018%) | 16/32 | [-147.574, +122.030] | 0.999156094275 | 0.995257485930 |
| 5 | hifi2 | 185755.389 | 185831.685 | +111.637 (+0.0601%) | 10/32 | [-45.284, +250.849] | 0.999117846135 | 0.995105687605 |
| 5 | producer_l1 | 185652.307 | 185625.822 | -32.762 (-0.0176%) | 20/32 | [-210.268, +50.764] | 0.999122336917 | 0.995105687605 |

Retain HiFi4 and DRAM prefill-router input for both kinds. Apparent savings are small relative to overlapping paired variation; full HiFi2 trends slower. Both source factors were executed independently with actual HF gates and stable program caches. No current-runtime edit follows from these trials.

Every control passes prefill and all128 decode PCC checks, exact repeated trace outputs and program-cache guards. The original top-k/softmax/scatter selection tail and per-expert scaling remain unchanged. The explicit native program is identical on both sides, preventing input placement from changing automatic geometry. Both variants keep FP32 outputs in DRAM.

HiFi2 changes only fidelity. Producer-L1 changes only the existing scaled-input multiply output placement and retains HiFi4, including the fused scalar-root activation. The report’s generic BFP8 fidelity label is inaccurate for this BF16 weight; the independent candidate was nevertheless measured.

Quartiles, pair counts and all samples are primary descriptive evidence. The exploratory sign statistic assumes independent pair signs and is not a proof of equivalence or a global optimum.

The JSON retains exact commands, runtime/helper/probe hashes, fixture hashes, all128 pairs, compute/program metadata, full timing samples and before/after memory observations. These runs bind the formatted paired helper; prior QKV runs resolve to its separately preserved historical snapshot.

The probe serialized compute configs with str(), which yields opaque Python object identities. Requested fidelities are established by its hash-bound constructor, baseline HiFi4 guard and selected object identity; those strings are not native config dumps. Current selected native v8 rows independently prove HiFi4/FP32 destination. Historical reports are preserved unchanged.

Remaining applicable prefill-router advice: none. Integrated default validation and native profiling use the unchanged selected runtime.
