# Singleton batch comparison

B1 completed the three calls in63.106163s versus B867.288444s: observed saving4.182281s (6.215% of B8 time). **Exact native output confirmation failed for call1**; calls0/2 match. All request objects and prompt IDs match. Call1 has90 differing token positions, first at index23; digests, not raw tokens, are in RESULT.json. No numerically exact batch substitution or scored acceptance follows.

Both arms completed128outputs per call with prompt counts103509/105535/107474. B8 retention matched381 emitted decode IDs with three first tokens excluded; no missing/extra active IDs, diagnostic loss, pending work or preemption. Captures observed[3]. Returning reuse and executed prefill work are recorded per call. Native output difference prevents treating equal token counts as identical numerical work.

Runtime hashes bind to11a387692d2b434198e79dd92fbb8ed308c3ca47. One sequential host-wall comparison supports only the measured call timings; no stochastic score parity or release forecast. Conditional same-three-call saving0.001161745h is not a release-hours prediction. SWE is not banked/go, and representative frequency/distribution remain unknown.

B8 queue cost0.079457698deviceh versus0.10h forecast, inside0.18h box. The optional-env prelaunch guard cost0deviceh; AUTODEBUG/AUTOFIX record static diagnosis, two focused host checks and successful orchestration repair. Idle shutdown left no owners. No runtime edit, profiler, hardware action or central edit by this analysis.
