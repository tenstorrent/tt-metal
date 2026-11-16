# head_bf16_hifi4 fixed-108 candidate

This is an unadmitted host-only candidate. It references the frozen P common-cap4096 manifest (108 jobs; seeds 42/43/44; 15 target pairs) and changes only lm_head weights from BFP8 to BF16 and lm_head compute fidelity from HiFi2 to HiFi4 versus the matched baseline.

The runtime policy accepts the BF16/HiFi4 pair and the model routes both dtype and fidelity through policy fields. The default selected precision is untouched. `precision_repro_batch --validate-only` passed with this candidate build-kwargs file. No TTNN import, device command, score change, launch, or admission was performed.

`runtime-estimate.json` gives a wide planning proxy from the frozen HiFi2 estimate and observed BFP8 load/preparation proxies. It does not establish BF16/HiFi4 load time, per-token speed, capacity, or fit. `command.json` is reference-only and must not be run before root admission and serialized device preflight.

P2 remains unadmitted and has no `admission.json`. Its prepared serial driver checks the real P1 and X admission/proposal/driver bytes, each predecessor's frozen inputs, terminal zero exits, and P1 plus per-mode X forecast ranges. A forecast miss blocks until root writes an acceptance JSON tied to that completion SHA, elapsed seconds, exact range, and admission SHA. Before the one 7200-second panel command, the driver checks the STOP file, duplicate-launch guard, empty output directory, no active HF reference process, and zero device owners. The complete source inventory is SHA-pinned in `source-hashes.json`; host-only self-tests cover tampered pins and forecast acceptance fields.


Forecast rebasing: P1 is now complete (108 trials, 2018.039 s wall; 1874.391 s case generation, 138.389 s load, 5.260 s outside-load work). `runtime-estimate.json` replaces the earlier47-row hybrid proxy with these all108 measured P1 components. The BF16/HiFi4 planning scenarios apply explicit 1.0/1.5/3.0 generation and 1.0/1.5/2.0 load multipliers (outside-load1.0/1.0/2.0). Even the high scenario is a proxy rather than a bound: BF16/HiFi4 output tokens may diverge and memory fit remains unmeasured. The P1 low forecast miss is accepted by root in the SHA-bound P1 record; P2 remains unadmitted.
