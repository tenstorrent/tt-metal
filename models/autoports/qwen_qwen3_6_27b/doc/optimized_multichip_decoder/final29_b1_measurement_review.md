# Final29 B1 prefill measurement review

CPU-only assessment; no source edits, device work, new tuning or formal stage-review verdict. The original prompt requires before/after warmed performance; `$optimize` says do not regress B1. Neither prescribes this statistical method. The project explicitly chose `exact_paired_median_upper_bound`,95% confidence, zero margin in its final plan/checker before final measurements. Preserve that result; do not silently replace its estimator after seeing data.

All five GDN pairs now exist. From `final29_pair_l0_p{0..4}_{control,final}.json`, final-minus-control warmed prefill differences in milliseconds are:

`[-0.396156, -0.261216, +0.037668, -0.171224, -0.216594]`.

Mean control/final = **2.395564/2.194060ms** (8.41% lower sample mean); median paired difference = **−0.216594ms**. Four pairs favor final; one favors control. Calling the positive difference “noise” is plausible but not established by these data alone.

`check_optimized_multichip_evidence.py:177–190` computes a valid distribution-free one-sided median bound under independent, comparable paired differences. For n5, the least-conservative qualifying order statistic is the maximum, with96.875% coverage. Its bound is **+0.037668ms**, so the current zero-margin checker **must fail to establish nonregression**. This is not evidence that population latency regressed. Four negative differences out of five have one-sided sign-test p=0.1875; neither the faster point estimate nor the observed37.7us positive difference settles the population claim. Process-level pairing avoids call-level pseudoreplication but does not by itself eliminate time drift; retain order/timestamp/sample provenance.

## Recommended handling under the communicated cap

Finish the already scheduled independent five pairs for both kinds once. Keep the exact checker outcome and all raw samples. Report GDN as **sample estimate faster, statistical nonregression inconclusive under the predeclared test**. If the user's subsequent cap permits nonblocking carried performance uncertainty, carry this specific uncertainty explicitly; do not describe the original gate as passed, invent a tolerance, relabel a quality failure, or change any PCC/trace/watcher/context gate. Parent/reviewer must assess that scope authorization and record the carried item. Without such authorization, this remains an unmet performance-evidence gate.

Do not retrospectively switch to a paired t-test, drop the positive pair, use within-process calls as independent replicates, combine older source/harness measurements, or append pairs until the bound crosses zero. Those actions change the decision rule in response to the result.

## Only if a fixed confirmatory run is separately authorized

Pre-register one independent follow-up block before any new samples: frozen source/build/policies, identical workload/harness/three-warmed-call estimator, **24 new process pairs**, alternating arm order with its starting arm fixed in advance, all24 retained, no interim pass/stop decisions, and no production tuning. Keep the first study reported as inconclusive. Use the same zero-margin population-median target, with a prespecified stronger follow-up cutoff to account for the previous opportunity to pass.

One conservative concrete rule is accept follow-up only if the18th sorted new paired difference is<=0 (at least18/24 nonpositive). Its null upper-tail probability is `sum(C(24,j), j=18..24)/2^24 = 0.0113279223`, giving98.8672% one-sided coverage. Combined with the first n5 test's actual0.03125 false-pass bound, a union bound is **0.0425779223 < 0.05** across these two opportunities. This design is conservative; it is an explicit new confirmation plan, not a reinterpretation of the failed n5 bound. Ties should be treated conservatively. The same stationarity/comparability assumptions remain necessary.

Under the instruction to run remaining gates once and carry allowed nonblocking limitations, additional24-pair work is **not recommended by default**. The rigorous immediate conclusion is honest uncertainty plus the observed faster point estimate, not another tuning cycle or a manufactured pass. This document recommends no automatic checker amendment and makes no stage-completion claim.
