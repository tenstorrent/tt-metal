# GPT-OSS-120B: 2,880 versus 4,352-byte prefill fabric payload

All 29 matched P/D cases finished with one warmup and three measured requests each. The saved audit reports zero errors. No goldens or PCC checks were performed. Different host pairs prevent a packet-only causal conclusion.

- [Full comparison and setup timing report](REPORT.md)
- [Exact pins, commands, methodology and obstacles](PAYLOAD4352_MEASUREMENTS.md)
- [All 116 requests, including29 warmups](PAYLOAD4352_RUNS.csv)
- [493 mean/variance/standard-deviation records](PAYLOAD4352_STATISTICS.csv)
- [Matched old/new throughput differences](payload4352-comparison.csv)
- [Measurement audit](sweep-audit.json)
- [Measurement accounting summary](measurement-summary.json)
- [Source and raw-evidence archive metadata](source-evidence-archive.json)
- [Service cleanup, including decoder shutdown warning](service-cleanup.json)
- [Verified reservation release](allocation-release.json)

The tested model change is commit 2898a1b47b7411611e9ab41b2873e16cf5aa9ac7 on the same branch. Blaze and D-gen are unchanged. Both machines were released after process/device-holder checks. The decoder launcher returned 1 during the requested stop; this preserved teardown warning does not invalidate the earlier completed measurement audit, and is not reported as a clean shutdown pass.

The interactive paired charts, synchronized chunk toggles and full per-run JSON are served locally at http://127.0.0.1:8768/payload4352-comparison.html. A portable report ZIP and raw archive path/hash are available in the final handoff. The tar archive captures measurement/source evidence before this result-publication commit; it intentionally excludes weight caches, environments and native build trees.
