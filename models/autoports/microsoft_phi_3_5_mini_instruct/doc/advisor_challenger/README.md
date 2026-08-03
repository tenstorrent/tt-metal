# Advisor contribution at decode batch 32

Full-model decode-layer estimate: **23,205.6 us before -> 21,938.1 us after, ±34.0 us**. The measured advisor contribution is **1,267.5 us/model (5.46%)**, well outside the uncertainty band. This is a 32-layer estimate from one measured dense decoder layer, not a timed full-model execution.

The frozen incumbent measured 0.807152 ms/layer (five means of 50 traced replays, after 10 warm-ups). The shipped query+key L1 RoPE chain measured 0.768065 ms/layer and confirmed at 0.767542 ms/layer in a fresh process. Every candidate and confirmation repeat beat every incumbent repeat. A real-weight differential oracle against the frozen incumbent produced PCC 1.0.

Reconciliation accounted for 100% of the 725.175 us profiled window. The advisor dropped 71.637 us/layer of shipped conversions; 1.967 us/layer of conversions were advisor agreements and are reported but not credited. `model_estimate.layer_handoff` found no layer-boundary round trip to screen.

The advised 11-core norm was measured, not assumed: 11/12/24 cores produced 0.745906/0.749010/0.748513 ms. Its combination with the RoPE winner reached 0.700267 ms, but real-weight differential PCC moved to 0.9999910667, so the entire norm knob remains default-off. Direct sharded SDPA output was attempted and rejected by `TT_FATAL: Sharded output not supported for GQA`. Sub-floor position/page-table chains were recorded as below threshold. All other losing knobs remain default-off in `rejected_knobs`.

Artifacts include the frozen control (`incumbent.json`), pinned batch-32 capture (`shard_advise/dense`), tool-generated reconciliation (`reconciliation_dense.json`), fresh-process measurements (`measurements`), bounded one-replay incumbent/winner profiles (`profiles/dense_{incumbent,winner}.csv`), and real-weight oracle logs.
