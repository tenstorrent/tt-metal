# Stage 07 Evidence Provenance

- Checkout root: `/localdev/odjuricic/tt-metal`
- Branch: `odjuricic/agentic-research/graph-rewrite-skill`
- Starting HEAD: `203c4f909d9c124ed987c125ae3e90a3ddcea600`
- Completed Stage 06 implementation: `cc5b46623f0`
- HF model: `google/gemma-4-31B`
- HF revision/snapshot: `d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3`
- Mesh: four Blackhole P150b, shape `[1,4]`, TP4, `FABRIC_1D`
- Python: 3.10.19
- Run date/timezone: 2026-07-14 through 2026-07-15, UTC
- Runtime library path: checkout `build/lib`
- Matplotlib scratch: `/tmp/mplconfig`

Hardware runs were serialized. Normal functional/performance evidence, scoped worker-watcher evidence, and profiler evidence were collected in separate processes. Full Ethernet watcher instrumentation was not used because its instrumented fabric binary exceeds the active ETH configuration-buffer capacity before model execution; `watcher_reduced_functional.xml` is the passing worker-watcher control with `TT_METAL_WATCHER_DISABLE_ETH=1`.

`profiler_raw_ops.csv.gz` is the losslessly compressed retained source CSV for `tt_perf_report.csv`, `tt_perf_summary.csv`, and `tt_perf_report.txt`; `gzip -dc` restores the exact report input. `profiler_sha256.txt` binds those selected block-2 files and prevents the rejected block-3 summary from being reused accidentally. The compact summaries are derived from the signposted `GEMMA4_FULL_MODEL_TOKEN_OUT_STEADY` window. `candidate_results.csv` maps every selected/rejected terminal geometry to its JSON or JUnit source.

The top-level profiler capture is source-current after the multi-row readiness repair and the qualitative repair selected split-8,192/four-input-shard/block-2. Its M=1 window covers the exact measured token-out operations in `_project_sharded_lm_head_tile`; the 33-row focused hardware regression and 249-row full readiness run cover the M>32 branch. `profiler_block3_rejected/` preserves the earlier eight-input-shard/block-3 capture and is not used for final claims.

The primary end-to-end throughput sources are the five-sample matched `full_token_out_matched_baseline.json` and `full_token_out_matched_selected.json`, collected without profiler or watcher after one discarded warmup on each constructed generator. The older `token_out_no_readback.json` is retained as the source-current 100-token qualitative-run benchmark. The profiler's merged chronology contains a 667.9 ms inter-stream gap and is used only for device-op topology, dtype/fidelity, bandwidth, and relative kernel contribution. `perf_summary.json` records this separation explicitly.
