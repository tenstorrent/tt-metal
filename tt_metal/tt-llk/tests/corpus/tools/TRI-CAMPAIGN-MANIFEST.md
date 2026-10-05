# Tri campaign manifest builder

`tri_campaign_manifest.py` turns reviewed planner and producer inputs into the
three small files needed to stage a tri-arm campaign. It does not compile,
discover revisions, or modify an existing campaign directory.

```sh
python3 tri_campaign_manifest.py \
  --roster roster.txt \
  --profiles tri-profiles.tsv \
  --idmap TRI-IDENTITY-MAP.tsv \
  --search search.json \
  --producer-tt-metal-head "$PRODUCER_TT_METAL_HEAD" \
  --runner-tt-metal-head "$RUNNER_TT_METAL_HEAD" \
  --sfpi-head "$SFPI_HEAD" \
  --sfpi-gcc-head "$SFPI_GCC_HEAD" \
  --compiler-sha256 "$COMPILER_SHA256" \
  --out campaign
```

The SFPI, SFPI-GCC, and compiler identities are optional. Git heads are
lowercase 40- or 64-hex object IDs; the compiler identity is a lowercase
SHA-256. The two tt-metal heads are required and explicit so producer and
runner identity cannot be inferred from whichever checkout runs the command.

The roster is one operation name per line. `tri-profiles.tsv` has the exact
10-column schema emitted by `exhaustive_campaign_plan.py`; its operation set
must equal the roster. `TRI-IDENTITY-MAP.tsv` is the headerless seven-column
producer map and may contain other operations. `search.json` is hashed as
supplied and is also checked against every selected and baseline flag string.

The output path must not exist. On success it contains:

- `flags.tsv`: sorted `op<TAB>selected_flags` rows;
- `idmap.tsv`: the sorted seven-column identity rows for roster operations;
- `manifest.json`: numeric `schema_version`, schema name, status `READY`,
  integer `eligible_ops`, sorted `ops`, input/output SHA-256 values, and the
  supplied producer/runner identities. The assembler authority hashes are the
  flat `search_sha256`, `flags_tsv_sha256`, and `idmap_sha256` fields.

Every validation is completed before the output directory is created. The
builder rejects duplicate or malformed rows, a roster/profile set mismatch,
missing identity/search rows, invalid hashes, A/B node disagreement, B/C flag
disagreement, and any disagreement with the pinned search.

Run the hardware-free checks with:

```sh
python3 selftest_tri_campaign_manifest.py
```
