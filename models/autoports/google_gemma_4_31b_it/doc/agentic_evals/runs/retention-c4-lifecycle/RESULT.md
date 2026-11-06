# C4 retention lifecycle result

Passed the observed synthetic C4/B8 window: all eight native completions finished through returning long histories and slot reuse. All632 expected emitted decode IDs matched retained IDs at their exact processed positions, with zero missing, mismatched or duplicate positions. Eight first tokens are explicitly excluded because they originate in prefill/host sampling. Native output total640; zero unassigned active samples. The656 inactive physical-row samples are not emitted/request tokens; no async extras are claimed.

Four actual active decode rows were observed on149steps, with7nonidentity remaps. Prompts span130048–132864tokens and outputs32–128, with each returning input preserving its preceding actual output plus deterministic padding. Disjoint native position bands provide the association; no request IDs are invented. This confirms retention under observed row churn, not release-cap output duration or original-loop reproduction.

The171 paired submit/completion steps include161decode and10prefill;167device and4host-compatibility sampling steps. No preemptions, errors, pending steps, diagnostic loss/disable/export failure were reported. Captures observed: [3]. The generic recorder still reports client_emission_verified=false; this separate offline native-response audit proves only its specified632-ID window, not all future recorder output.

All13 runtime source hashes match accepted publication `11a387692d2b434198e79dd92fbb8ed308c3ca47` and remain unchanged after the run. Full60 selected runtime, B8, mesh devices0–3, pinned model/tokenizer and unchanged native serving configuration were used. No runtime changes or profiling occurred.

Measured lifecycle window:21.833349s; cold priming:262.106957s. Total queue cost **0.148381473deviceh (534.173303s)** versus0.15h forecast (0.9892×), within0.30h box. Fresh idle metrics were0running/0waiting; shutdown left no owners. No causal speed or release-hours savings claim follows. Private source, responses and phase evidence are referenced in RESULT.json; no raw tokens are published.
