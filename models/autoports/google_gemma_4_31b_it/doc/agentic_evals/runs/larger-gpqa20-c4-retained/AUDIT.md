# Retained GPQA20 C4 run audit

Functional_at remains **10**. Raw score **16/20 (80%)**, Wilson95 **58.40–91.93%**, is reported but not banked. Evaluator **1.5120h**, device-job wall **1.6492h**, workflow **1.6553h**, against6.75h forecast. No device-time or causal speed claim follows.

The completed collector has no phase JSONL or tt_triage directory. Full console line128930 reports tar's “file changed as we read it” for gemma4-phase-147-1793984275796374080.jsonl, then “No tt-triage reports.” Token retention was enabled, but its artifact and final integrity state were not delivered. The console has no retained-token/schema snapshots; model export writes a file rather than stdout, so physical chains cannot be reconstructed from this console.

All20 exact question/prompt hashes and visible responses match prior20. Actual indices7 and20 again have empty answers; the other18 are conclusive. Harness samples retain no native completion IDs, reasoning, usage or finish reason. Consequently this run cannot establish their cap lengths or distinguish persistent looping from unfinished useful reasoning. Prior local replay does not replace missing CI trajectories. No sample-to-slot assignment is made.

Startup confirms this autoport and all60 layers at B8, C4/maxseq8/context262144 with frozen pins and release limits. Sampling seed42, temperature1, top_k20, top_p.95, output126976 and request14400s are preserved. GPQA uses HTTP requests and no task sandboxes. Raw scores are not gated; representativeness remains blocked by missing failure evidence. Recover the original private phase artifact if available, or repair collection before the required rerun. Original collection and logs remain preserved.
