# Closed evidence packaging update

Packaged four newly completed passing runs since the prior archive manifest
update at 2026-11-12 20:53:49 UTC:

- `ple_final_batch4_controlled`
- `public_continuation_layer0_v2`
- `public_continuation_layer1_v2`
- `public_continuation_layer3_v2`

Their watcher logs total 404,228 original bytes and 15,116 archived bytes.
The four XZ files are under `watcher_archives/`; `evidence_archives.json` now
has 47 entries. Each new entry records original/compressed hashes and sizes,
exact result/provenance identities, command exit status, watcher scan results,
and recorded/current runtime hashes. Compression uses deterministic XZ CRC64,
preset 6, with a 450,000-byte part limit; no new archive needed splitting.
Round-trip restoration and a second identical compression were verified.
The established watcher error scan found no matching lines in these four logs.

No newly completed result or console log exceeded 500,000 bytes since that
prior manifest update. Older already packaged diagnostics, report artifacts,
raw Tracy directories, and all originals remain untouched. The active
`ple_capacity_alltails_6144` log/results were excluded entirely.

These are historical passing runs, **not current-runtime final passes**.
All four recorded a different `tt/precise_residual_projection.py` from the
latest GR lifetime repair. At packaging time the current file SHA256 was
`8af1023e7dcbfae6523769fbfa7245348f4bf022d4b303309417649a2e47046a`.
The manifest preserves each prior hash and marks `historical_source_mismatch`;
no runtime-equivalence inference is made. The controlled PLE batch also retains
`literal_hf_strict_passed: false`; its successful matched-control classification
is not relabeled as a literal strict pass.
