<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Deterministic input export

The first migration stage uses `tools.generic_op_to_factory.export_run` in this
tt-metal repository. Its implementation and synthetic tests live with the rest
of the migration flow. See [EXPORT_RUN.md](EXPORT_RUN.md) for the package format
and complete command contract.

From the tt-metal repository root, with `psycopg2` available for live DB reads:

```bash
export EVAL_DATABASE_URL='postgresql://eval_ro@bgdepyc01/eval'
python3 -m tools.generic_op_to_factory.export_run --run-id "$EVAL_RUN_ID" --output "$EXPORT_DIRECTORY"
python3 -m tools.generic_op_to_factory.export_run --verify "$EXPORT_DIRECTORY"
```

Supply an explicit DB primary key and a new output directory. No LLM participates
in selection, retrieval, path handling, or file materialization. No remote DB
writes, schema initialization, or SQLite fallback occur. Exported source is
not executed.

The package contains exact UTF-8 encodings of stored host/kernel/artifact text,
all recorded result phases and run metadata, and a deterministic checksummed
manifest. Unsafe paths and duplicate output names fail instead of being
guessed or renamed. Generated tests are preserved as artifacts; they are not
silently treated as golden-suite source.

The selection is the **currently recorded run snapshot**. The eval DB replaces
source rows on re-ingestion; it does not expose a versioned source candidate
for each refinement. Freeze the exported package and its `source_sha256` and
`snapshot_sha256` before applying [MAPPING.md](MAPPING.md).

`tools.generic_op_to_factory.prepare_baseline` now resolves the recorded golden revision and static
eval imports, preserves canonical helpers at the source revision, and installs
unchanged operation files only into an explicitly selected pinned runtime.
See [PREPARE_BASELINE.md](PREPARE_BASELINE.md) for preparation, verification
and installation commands, plus the remaining runtime gates.

After execution, `tools.generic_op_to_factory.compare_baseline` compares JUnit case identities and
outcome classes against one explicitly selected DB phase (or unphased rows).
Missing/added cases and changed outcomes remain visible. Matching a recorded
failure does not make an operation correct or mark it migration-ready.

The manifest deliberately reports `migration_ready: false`. Before translation,
resolve the recorded golden-suite revision, external helper dependencies, and
runtime configuration, then reproduce the baseline. Those inputs are not
guaranteed by a successful DB export. In particular, a failed eval run can be
exported successfully without becoming a validated operation.

The exporter runs from this tt-metal checkout. The evaluator repository supplies
recorded Git objects, not migration-tool code. This flow does not update the
tt-metal evaluator submodule pin. Production run IDs belong in invocation
arguments and validation reports, not exporter logic or automated test fixtures.
