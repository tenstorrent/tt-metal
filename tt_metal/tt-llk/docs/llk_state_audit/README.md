# LLK state audit

This directory is a generated, source-only map of persistent LLK state effects. Code and the checked-in effect model are authoritative; these artifacts are a reviewable projection, not a hardware-semantic specification.

## Scope and boundary

The audit scans `_llk_*` definitions in Wormhole B0, Blackhole, and Quasar header trees. It records only reviewed parser/model matches. Source parsing does not instantiate templates or resolve every template branch, and MOP instruction encodings are represented only where the reviewed model recognizes them. Empty cells mean unknown or not applicable; no description is inferred from a name.

Definition discovery is exhaustive, but per-parameter effect coverage is bounded by the reviewed effect model rather than the full instruction set. Instructions the model does not recognize — including opcode-form (`TT_OP_*`) operands assembled into MOPs and some architecture-specific addressing macros — produce no effect rows and surface only through an enclosing MOP effect or a classification-only entry. A missing effect row therefore means "not modeled," not "no state change."

## Published data

`llk_state_map.json` is canonical machine-readable data: inventory, classifications, effects, reviewed restore contracts, and summary. `llk_state_map.csv` is RFC 4180 CSV for spreadsheets. Its columns identify the function and canonical wrapper target, parameter/type/kind, value expression, architecture/thread/stage/stability/lifecycle, classification and reason, condition, state resource/operation/persistence/retention, restore contract, evidence/source identity, commit, mapping status, and notes.

Rows with `Mapped Status=effect` are parameter-to-state mappings. `Evidence Kind=direct` means the sink occurs in that body; `transitive` means the effect reaches the LLK through the listed recursive helper chain. Sink source columns anchor the deepest recognized state primitive, while the ordinary source columns anchor the public LLK call site. `classification-only` rows ensure definitions without effects remain discoverable. Notes mark excluded definitions, validation-only/unknown lifecycle ownership, no-op teardown, and reviewed restore/retention contracts.

## Architecture and confidence

Wormhole B0 (WH) and Blackhole (BH) resource names are typically register-letter/address based; Quasar (QSR) uses more semantic helper/resource naming. Names therefore are not guaranteed to be cross-architecture equivalents. Confidence is `high` for directly observed parameter flow and directly observed fixed sinks, `medium` for local flow or fixed/transitive effects, and `low` when a transitive mapping follows a local flow.

Stability tiers are `stable`, `experimental`, `debug`, and `test_only`, inferred from source paths. Every mapped effect includes a reviewed retention contract describing how long the affected state remains meaningful. Restore contract kinds are `snapshot_restore`, `canonical_reset`, `no_op_transient`, and `retained_until_reconfigure`; explicit restore contracts supplement retention and are populated only when reviewed.

## Generated summary

- Definitions: 553
- Effect rows: 5463
- Classification: {"excluded": 15, "included": 526, "wrapper": 12}
- Architectures: {"blackhole": {"definitions": 226, "effects": 2300}, "quasar": {"definitions": 140, "effects": 847}, "wormhole_b0": {"definitions": 187, "effects": 2316}}
- Analyzed source fingerprint: `04031199f2c69d9ea3618623b6af84b958a52f55499883bc3cd97a544f1a0f38`
- Generator base commit (informational): `cb734903f80375e835ba038828abe0e233ce7230`

## Verification matrix

The reviewed verification manifest links 11 existing LLK tests to audited functions and state domains. Covered validation scopes: canonical_baseline, restore, runtime_reconfigure, synchronization. These links identify relevant checks; they do not claim that hardware tests ran during artifact generation. See `tools/llk_state_audit/verification_manifest.json` for the exact architecture, function, family, domain, and test-path mappings.

## Regeneration and drift

From `tt_metal/tt-llk`, run:

```sh
python3 -m tools.llk_state_audit generate --root .
python3 -m tools.llk_state_audit verify --root .
python3 -m tools.llk_state_audit check --root .
```

Regenerate twice and compare bytes when reviewing renderer changes. A changed source fingerprint, classification, or reviewed restore anchor is audit drift and should be reviewed in code and the effect model before updating artifacts.
