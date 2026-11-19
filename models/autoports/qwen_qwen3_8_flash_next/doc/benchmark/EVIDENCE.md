# Benchmark evidence

The final report is [run/REPORT.md](run/REPORT.md). Identity and the stage work log
are [identity.json](identity.json) and [RUN_NOTES.md](RUN_NOTES.md). The report
distinguishes the launched implementation commit from later host reporting and
evidence commits.

Raw evidence remains unmodified in `run/` and `setup/` in the working checkout.
At stage closure, each finished scope is saved as one lossless tar.xz stream,
split into pieces below the repository's per-file size limit. The local evidence
commit includes these pieces, their path/size inventories and bundle integrity
records. Large original files are represented by the bundle rather than added
individually. Setup includes unsuccessful diagnostic attempts and the completed
pilot; these are separate from the final timed run.

From the tt-metal repository root, restore all evidence with:

```bash
python models/autoports/qwen_qwen3_8_flash_next/tools/archive_benchmark_evidence.py restore \
  --manifest models/autoports/qwen_qwen3_8_flash_next/doc/benchmark/run/evidence_archives/manifest.json
python models/autoports/qwen_qwen3_8_flash_next/tools/archive_benchmark_evidence.py restore \
  --manifest models/autoports/qwen_qwen3_8_flash_next/doc/benchmark/setup/evidence_archives/manifest.json
```

Restoration validates the complete bundle before extracting. Identical existing
files are accepted; conflicting files, unsafe paths and damaged bundles are
rejected. Originals are never overwritten. The configuration preserves the
actual absolute paths used by this run, so reproduce checks in the recorded
workspace or explicitly account for relocation.

The frozen manifests retain selected document IDs and hashes. Task directories
retain all questions, upstream scored samples/results, request links and raw
responses including reasoning. Performance JSON and `.input-token-ids.json`
companions retain timings and exact distinct prompts. `phase-raw-b2.json`,
`phase-accounting-b2.json` and `roofline.json` retain the measured batch's native
completion observations, work model and request mapping. Server launch and
runtime identity evidence is linked from the per-profile configuration.

The upstream evidence check and context-contract check must be run against the
restored originals, not just the archive inventory. Their outputs and the
independent final review are retained with the completed run. No score threshold
or model accuracy verdict is part of this benchmark stage.
