# GPT-OSS-120B fabric payload comparison: 2,880 / 4,352 / 8,704 bytes

All 29 valid ISL/chunk pairs at 8,704 bytes completed and passed the saved-evidence audit on October 21, 2026. Each case has one warmup and three measured requests, with a 32-token output cap and EOS enabled. The earlier baselines are frozen.

Compared with 4,352 bytes, 8,704 has 10 higher and 19 lower measured means, ranging from -3.745% to +7.561%. All 16K–64K chunk cases improved by 6.17–7.56%; smaller chunks were mostly slower. These are descriptive observations. Prefill/decoder hosts differ across cohorts, and three measurements per case do not establish packet-only causality or numerical accuracy.

## Files

- [Report](REPORT.md): all matched means, differences, slower samples, setup/reload timing and limitations.
- [Configuration and commands](PAYLOAD8704_MEASUREMENTS.md): exact pins, settings, topology, lifecycle and incident history.
- [All 116 requests](PAYLOAD8704_RUNS.csv): warmups and measurements, tokens/reuse/padding, TTFT, transfer accounting and native decode rate.
- [493 metric summaries](PAYLOAD8704_STATISTICS.csv): means, sample variances and standard deviations.
- [Matched comparison](payload-comparison.csv): all 29 pairs and differences against both references.
- [Final audit](sweep-audit.json), [performance review](performance-review.json), [cleanup](service-cleanup.json), [release](allocation-release.json) and [raw archive metadata](source-evidence-archive.json).

The portable report package also contains an offline interactive `index.html`, all three datasets, the previous 4,352-byte package, publication/hook receipts and report generators. Open `index.html` without a server. The live local report is http://127.0.0.1:8768/payload4352-comparison.html.

## Tested source

- Metal configuration commit: `076d79443138613e9e1a96b948d02db69858274a`, on `divanovic/mg-gptoss-pd-payload8704-20261021-metal`.
- Parent / completed 4,352 results: `9f7df82859f1b5e2e7d7f91e6fb3b1e69e9d628f`.
- Only model delta: `FABRIC_PAYLOAD_SIZE = 4352` to `8704`; imported configuration asserted for all six groups.
- Blaze: `d6d0b09f22f60a161d66fe4db90b9abed81ef82d`; nested Metal: `44d66500520fda9f2c7060c0f6b41ec48f7ab37e`.
- D-gen: `5ca79e3076f193c092962d34a3b69731e39aff91`.

The raw archive captures tested source references/patches, runtime commands, stopped controllers, request/output evidence and all failed attempts after cleanup, before this results commit. It excludes weights, environments and build trees. Its member hashes and full archive SHA256 are verified; see `source-evidence-archive.json`. The result commit is the Git commit containing this directory; the portable package records its verified remote hash in `publication.json`.

No goldens, PCC, standalone reruns or selective exclusions were used. Request span includes the decoder-preemption/replacement gap. Concurrent reload stages and overlapping transfer intervals are never added as wall time.
