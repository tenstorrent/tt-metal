# GPT-OSS-120B b3b5 packet comparison

## Cleanup outcome: unverified after allocation expiry

All87 cases are published and audited. Final process/device-holder absence was not verified before ownership expired. The decoder wrapper records timeout exit124; no supported owner-stop or manual base-service stop was sent. Slurm now has no active task allocations; terminal records are unavailable and accounting is disabled. No compute host was accessed after expiry. Full task completion is not claimed. See [DEADLINE-OUTCOME.md](DEADLINE-OUTCOME.md) and the [verified9928-file evidence archive receipt](source-evidence-archive.json).


87 audited cases across payloads2880/4352/8704. Start with [REPORT.md](REPORT.md), [interactive tables](b3b5-packet-comparison.html), or [exact commands and timing definitions](MEASUREMENTS.md). Full requests and metrics are in the CSV/JSON exports. Publication and resource cleanup are separate verified receipts.

Full exports are stored losslessly as gzip files to satisfy the repository file-size limit. `export-packaging.json` records compressed and original SHA256 values; decompress with `gzip -dk <file.gz>`. The interactive view uses compact table JSON; the full export retains every request and field.
