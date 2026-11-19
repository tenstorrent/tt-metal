# Evidence storage and replay

Each recorded job has a `<name>_provenance.json` with exact argv, environment, start/end UTC, elapsed command wall time, exit status, source hashes and console-log hash. Plans freeze the fused runtime hash and stop on the first failed command. A failed plan remains failed even after a later repaired plan passes.

Large or format-sensitive historical evidence is archived only after device jobs finish. `tests/archive_fused_evidence.py --root models/autoports/qwen_qwen3_8_flash_next/doc/fused_decoder --finalize` produces byte-identical XZ copies and `evidence_archives.json`. If the index exceeds the repository file-size gate, its `archive_parts` entries identify smaller JSON arrays under `evidence_index_parts/`; their hashes are in the root index. Each archive row gives original path, original hash/size, archive path and archive hash. Verify both hashes when restoring, and never format historical Python or generated C++ snapshots. Raw originals remain available locally.

Multi-GB raw Tracy exports remain local and are identified by SHA256/size in `local_raw_only`. Accepted report directories contain the official filtered CSV, report table, advice and compact provenance needed to inspect the reported measurements. A raw capture is not itself accepted evidence: the corresponding strict report must pass.

The `tracy/*/report/latency_summary.json` whole-trace intervals come from native firmware endpoints and include internal operation gaps. The per-op reports diagnose kernels and data movement; segmented prefill drains make their host time unsuitable for candidate ranking. Clean same-harness candidate JSON supplies warmed host prefill and traced decode ranking. Watcher runs are separate from all profiler runs.

Long-context evidence must be read with its inherited contract: reduced attention/state prefix over actual262143 rows followed by a complete public final-token decoder and traced replay. It is not a claim of a complete MoE decoder pass over every prefix row. Prepared public PLE4768/4767 probes separately exercise the complete public prefill.
