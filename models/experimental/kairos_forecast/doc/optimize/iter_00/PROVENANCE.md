# iter_00 provenance notes (D5 R1-0d)

- MANIFEST: job 1102 started about 97 s before bd966218ee3 was committed. It was re-run on the committed tree
  (880bc6f440f, clean) as cpu job 1139 with the same `KAIROS_GOLDEN_JOBS`; it wrote 19 golden hashes, and
  `sha256sum -c iter_00/golden_sha256_after.txt` passes on the golden dir.
- gates.json `_doc` now names the fixed records' jobs (suite 1105, bench 1107, re-recorded for R1-0a in the jobs
  listed in candidates.md) instead of job 1051 only.
