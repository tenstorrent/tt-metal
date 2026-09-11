# Export a recorded eval run

`tools.generic_op_to_factory.export_run` fetches the currently recorded source and evidence for an
explicit `runs.id` from PostgreSQL. It does not run an LLM, reconstruct missing
files, select a best run/refinement, execute source, or run golden tests.

From the tt-metal repository root, using a Python environment that already
has `psycopg2`:

```bash
export EVAL_DATABASE_URL='postgresql://eval_ro@bgdepyc01/eval'
python3 -m tools.generic_op_to_factory.export_run --run-id "$EVAL_RUN_ID" --output "$EXPORT_DIRECTORY"
python3 -m tools.generic_op_to_factory.export_run --verify "$EXPORT_DIRECTORY"
```

Set `EVAL_RUN_ID` to the selected DB primary key and `EXPORT_DIRECTORY` to a new
directory. The output directory must not already exist, even if empty.
`--verify` is offline and does not require a DB URL or driver. Credentials are
not written to the manifest or printed in connection errors. Prefer the
`eval_ro` database role; use the environment rather than a URL command argument
for credential-bearing connection strings.

## Selection and transaction contract

- `--run-id` selects exactly one run. Only terminal `complete` or `failed` runs
  are accepted. A failed run remains failed in the exported provenance; a
  successful export says nothing about operation correctness.
- There is no SQLite fallback or implicit default database. An absent URL,
  connection error, missing run/table, missing host-source rows, non-text
  artifact, or ambiguous/unsafe path fails explicitly.
- The connection starts with `default_transaction_read_only=on` and uses one
  `REPEATABLE READ`, read-only transaction for all tables. It never calls
  `eval.db.connect()`, which can initialize/migrate the schema. The transaction
  is rolled back and the connection closed after the read.
- `host_code`, `kernels`, `artifacts`, and `kw_breadcrumbs` are replaced by
  re-ingestion (see `eval.db.delete_ingest_children`). Their rows are not a
  versioned candidate history. A refinement's result rows do not prove that its
  historical source was preserved. The selection is therefore named
  `current-recorded-run-snapshot`, not "best candidate" or "source at phase X".
- All result phases and recorded refinement/phase/timing metadata are exported.
  The exporter does not apply dashboard aggregation or canonical-phase rules.

## Package layout

```text
manifest.json
source/
    <host_code.filename>
    kernels/<kernels.filename>
artifacts/
    <artifacts.name>
records/
    run.json
    host_code.jsonl
    kernels.jsonl
    artifacts.jsonl
    kw_breadcrumbs.jsonl
    test_results.jsonl
    score_criteria.jsonl
    refinement_snapshots.jsonl
    phases.jsonl
    device_timings.jsonl
    device_phases.jsonl
```

`records/` preserves every selected row and column, including raw metadata,
configuration fields, result history, and source text. Child rows are ordered
by DB row ID; empty tables have empty JSONL files. The large result tables are
serialized in batches. `source/` and `artifacts/` provide convenient exact-text
copies without requiring a consumer to extract source from JSON.

Names are preserved under the explicit table namespaces above. The ingester
normally records host and kernel basenames; the exporter cannot recover a
directory hierarchy that ingestion discarded. It rejects duplicate output
paths (even identical duplicate content), file/directory conflicts, absolute
paths, traversal, noncanonical separators, Windows-style paths, and control
characters rather than renaming or choosing a row. Stored empty files are
preserved. Zero kernel rows are allowed and noted because kernels can be
inline; the exporter does not invent or extract inline kernels.

Text is written as UTF-8 without newline conversion or an added final newline.
This preserves the database text, not necessarily original filesystem bytes:
ingestion may already have normalized newlines or skipped files.

## Manifest and repeatability

Format version 1 records:

- The database host/port/name, selected run ID, status, source branch/commit,
  eval branch/commit, golden-suite name, architecture, and selection semantics.
- Counts for every exported table and an index of every payload file with its
  byte size and SHA-256. Materialized files also identify their DB table, row
  ID, and original stored name.
- `source_sha256`: a digest of the sorted host/kernel path, size, and content
  hashes. It identifies recorded source independently of mutable run annotations
  and re-ingestion row IDs.
- `snapshot_sha256`: a digest of the complete manifest excluding that field
  itself. Through the payload index it covers all exported rows and artifacts.
- `migration_ready: false` and explicit unresolved inputs, described below.

JSON uses sorted keys, UTF-8, compact separators, and a final newline. Values
outside JSON's native types use single-key tagged objects: `$float` for
non-finite floats (`nan`, `inf`, `-inf`), `$decimal` for exact decimals, and
`$date`/`$datetime` for ISO-formatted temporal values. Existing JSON/string
columns remain as returned by the driver; the exporter does not interpret
configuration strings or deserialize source.

No export timestamp, destination path, or random staging name enters the
package. Repeated exports of an unchanged DB snapshot have identical file
bytes. If the DB is re-ingested or annotated between exports, the full snapshot
digest can change even when the source digest stays the same. Save the package
to pin a migration input; a run ID alone is not an immutable artifact version.

Files are prepared in a temporary sibling directory and published only after
the complete manifest is written. Existing output is never overwritten.
Ordinary failures clean up staging. An interrupted process can leave staging
or an empty output reservation; only a successfully verified package is usable.

Offline verification rejects changed/missing/unexpected files, symlinks, unsafe
manifest paths, and mismatched manifest/source digests. These checks detect
accidental drift; they are not a signed attestation of DB origin.

## Boundary before C++ migration

The package is complete for the exported tables, not necessarily a runnable,
dependency-closed operation. The manifest explicitly leaves these unresolved:

1. **Golden-suite source.** Results and generated test artifacts are not the
   matching golden suite. Resolve `eval_commit` and `golden_name` against the
   eval repository; do not silently substitute its current checkout or another
   previously exported suite.
2. **External dependencies.** Resolve canonical helpers and other files not
   captured in the run tables, using recorded provenance and an audited
   dependency inventory. Do not reconstruct them with an LLM.
3. **Runtime configuration.** Preserve the recorded metadata/artifacts, then
   establish the exact invocation and tuning settings required by the baseline.
   The DB does not guarantee a complete environment snapshot.

Missing references remain explicit `null` values. Resolving these inputs and
reproducing the original baseline is the next workflow stage; see
[baseline preparation](PREPARE_BASELINE.md). This exporter
does not import artifacts into a tt-metal checkout or claim migration success.

## Tests

From the tt-metal repository root, after activating its existing environment:

```bash
./scripts/run_safe_pytest.sh --run-all --no-precompile tools/generic_op_to_factory/tests/test_export_run.py -q
```

Tests use a synthetic SQLite fixture solely behind an injected read-only
connection adapter, plus mocked PostgreSQL connection setup/failures. The CLI
supports PostgreSQL only. Tests contain no production run IDs or operation
fixtures. For integration validation, supply any explicit remote run ID, export
to two new directories, compare checksums/file bytes, verify offline, and
compare the materialized text and raw table rows against read-only DB queries.
