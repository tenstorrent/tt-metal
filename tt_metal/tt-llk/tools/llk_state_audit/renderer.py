"""Canonical JSON and spreadsheet CSV rendering for the LLK state audit."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any

from .classification import audit
from .inventory import load_effect_model
from .verification import (
    load_verification_manifest,
    validate_verification_manifest,
)

GENERATOR_VERSION = "2.0"
SCHEMA_VERSION = 2
CSV_HEADERS = [
    "Function",
    "Canonical Function",
    "Parameter",
    "Type",
    "Parameter Kind",
    "Description/Value Expression",
    "Architecture",
    "Thread",
    "Stage",
    "Stability",
    "Lifecycle",
    "Classification",
    "Classification Reason",
    "Condition",
    "Condition Kind",
    "State Domain",
    "Register/Configuration Resource",
    "Operation",
    "Persistence",
    "Retention Contract",
    "Activation",
    "Restore Contract Kind",
    "Restore Owner/Pair",
    "Confidence",
    "Evidence Kind",
    "Helper Chain",
    "Sink Source Path",
    "Sink Source Line",
    "Sink Body Fingerprint",
    "Source Path",
    "Source Line",
    "Body Fingerprint",
    "Generator Base Commit",
    "Analyzed Source Fingerprint",
    "Mapped Status",
    "Notes",
]


def render(
    audit_data: dict[str, Any],
    *,
    revision: str,
    source_fingerprint: str | None = None,
    command: str = "python3 -m tools.llk_state_audit generate --root .",
    scope: dict[str, Any] | None = None,
    restore_contracts: list[dict[str, Any]] | None = None,
    verification_manifest: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Render an audit payload into canonical JSON and RFC 4180 CSV strings."""
    scope = scope or {
        "architectures": ["blackhole", "quasar", "wormhole_b0"],
        "source": "tt_llk_*/*.h",
    }
    source_fingerprint = source_fingerprint or _audit_fingerprint(audit_data)
    document = {
        "analyzed_source_fingerprint": source_fingerprint,
        "audit": audit_data,
        "generation_command": command,
        "generator_base_commit": revision,
        "generator": {"schema_version": SCHEMA_VERSION, "version": GENERATOR_VERSION},
        "reviewed_restore_contracts": restore_contracts or [],
        "verification_manifest": verification_manifest
        or {"schema_version": 1, "tests": []},
        "scope": scope,
    }
    json_text = (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return {
        "json": json_text,
        "csv": _render_csv(
            audit_data,
            revision,
            source_fingerprint,
            restore_contracts or [],
        ),
    }


def generate(
    root: Path | str,
    *,
    output_dir: Path | str | None = None,
    revision: str | None = None,
    audit_data: dict[str, Any] | None = None,
    effect_model: Path | str | None = None,
) -> dict[str, Any]:
    """Run the approved audit and write canonical JSON, CSV, and README artifacts."""
    root_path = Path(root).resolve()
    revision = revision or _revision(root_path)
    supplied_audit_data = audit_data is not None
    audit_data = audit_data or audit(root_path, effect_model)
    source_fingerprint = (
        _audit_fingerprint(audit_data)
        if supplied_audit_data
        else _source_fingerprint(root_path, effect_model)
    )
    contracts = (
        []
        if supplied_audit_data
        else load_effect_model(effect_model, root=root_path)["restore_contracts"]
    )
    verification_manifest = (
        {"schema_version": 1, "tests": []}
        if supplied_audit_data
        else load_verification_manifest(root_path)
    )
    validate_verification_manifest(verification_manifest, audit_data)
    artifacts = _render_artifacts(
        audit_data,
        revision,
        source_fingerprint,
        contracts,
        verification_manifest,
    )
    destination = (
        Path(output_dir)
        if output_dir is not None
        else root_path / "docs" / "llk_state_audit"
    )
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "llk_state_map.json"
    csv_path = destination / "llk_state_map.csv"
    readme_path = destination / "README.md"
    json_path.write_text(
        artifacts["llk_state_map.json"], encoding="utf-8", newline="\n"
    )
    csv_path.write_text(artifacts["llk_state_map.csv"], encoding="utf-8", newline="")
    readme_path.write_text(artifacts["README.md"], encoding="utf-8", newline="\n")
    return {
        "json_path": json_path,
        "csv_path": csv_path,
        "readme_path": readme_path,
        "summary": audit_data["summary"],
    }


def verify(
    root: Path | str,
    *,
    output_dir: Path | str | None = None,
    revision: str | None = None,
    audit_data: dict[str, Any] | None = None,
    effect_model: Path | str | None = None,
) -> dict[str, Any]:
    """Check generated artifacts against the current in-memory audit without writing."""
    root_path = Path(root).resolve()
    destination = (
        Path(output_dir)
        if output_dir is not None
        else root_path / "docs" / "llk_state_audit"
    )
    revision = revision or _recorded_revision(destination) or _revision(root_path)
    supplied_audit_data = audit_data is not None
    audit_data = audit_data or audit(root_path, effect_model)
    source_fingerprint = (
        _audit_fingerprint(audit_data)
        if supplied_audit_data
        else _source_fingerprint(root_path, effect_model)
    )
    contracts = (
        []
        if supplied_audit_data
        else load_effect_model(effect_model, root=root_path)["restore_contracts"]
    )
    verification_manifest = (
        {"schema_version": 1, "tests": []}
        if supplied_audit_data
        else load_verification_manifest(root_path)
    )
    validate_verification_manifest(verification_manifest, audit_data)
    expected = _render_artifacts(
        audit_data,
        revision,
        source_fingerprint,
        contracts,
        verification_manifest,
    )
    mismatches = [
        name
        for name, content in expected.items()
        if not (destination / name).is_file()
        or (destination / name).read_bytes() != content.encode("utf-8")
    ]
    return {
        "valid": not mismatches,
        "mismatches": mismatches,
        "summary": audit_data["summary"],
    }


def _render_artifacts(
    audit_data: dict[str, Any],
    revision: str,
    source_fingerprint: str,
    restore_contracts: list[dict[str, Any]],
    verification_manifest: dict[str, Any],
) -> dict[str, str]:
    rendered = render(
        audit_data,
        revision=revision,
        source_fingerprint=source_fingerprint,
        restore_contracts=restore_contracts,
        verification_manifest=verification_manifest,
    )
    return {
        "llk_state_map.json": rendered["json"],
        "llk_state_map.csv": rendered["csv"],
        "README.md": _readme(
            audit_data,
            revision,
            source_fingerprint,
            verification_manifest,
        ),
    }


def _render_csv(
    audit_data: dict[str, Any],
    revision: str,
    source_fingerprint: str,
    restore_contracts: list[dict[str, Any]],
) -> str:
    inventory = audit_data["inventory"]["functions"]
    definitions = audit_data["classification"]
    definition_by_identity = {
        _identity(definition): definition for definition in definitions
    }
    contract_by_identity = {
        (
            contract["architecture"],
            contract["function"],
            contract["source"]["path"],
            contract["source"]["body_fingerprint"],
        ): contract
        for contract in restore_contracts
    }
    functions_by_effect = {}
    for function in inventory:
        functions_by_effect.setdefault(
            (
                function["architecture"],
                function["name"],
                function["source"]["path"],
                function["body_fingerprint"],
            ),
            [],
        ).append(function)

    rows: list[tuple[tuple[Any, ...], dict[str, str]]] = []
    covered: set[tuple[str, str, str, int, str]] = set()
    for effect in audit_data["effects"]:
        candidates = functions_by_effect.get(
            (
                effect["architecture"],
                effect["function"],
                effect["evidence"]["source_path"],
                effect["evidence"]["body_fingerprint"],
            ),
            [],
        )
        function = next(
            candidate
            for candidate in candidates
            if candidate["source"]["start_line"]
            <= effect["evidence"]["line"]
            <= candidate["source"]["end_line"]
        )
        identity = _identity(function)
        definition = definition_by_identity[identity]
        covered.add(identity)
        rows.append(
            (
                _row_sort_key(function, effect),
                _effect_row(
                    effect,
                    function,
                    definition,
                    revision,
                    source_fingerprint,
                ),
            )
        )

    for function in inventory:
        identity = _identity(function)
        if identity not in covered:
            definition = definition_by_identity[identity]
            contract = contract_by_identity.get(
                (
                    function["architecture"],
                    function["name"],
                    function["source"]["path"],
                    function["body_fingerprint"],
                )
            )
            rows.append(
                (
                    _row_sort_key(function),
                    _classification_row(
                        function,
                        definition,
                        revision,
                        source_fingerprint,
                        contract,
                    ),
                )
            )

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=CSV_HEADERS, lineterminator="\r\n", extrasaction="raise"
    )
    writer.writeheader()
    for _, row in sorted(rows, key=lambda item: item[0]):
        writer.writerow(row)
    return output.getvalue()


def _effect_row(
    effect: dict[str, Any],
    function: dict[str, Any],
    definition: dict[str, Any],
    revision: str,
    source_fingerprint: str,
) -> dict[str, str]:
    restore = effect["restore"] or {}
    sink = effect["evidence"]["sink"]
    owner_pair = _restore_owner_pair(restore)
    notes = f"{effect['evidence']['kind']} source evidence"
    if effect["evidence"]["kind"] == "transitive":
        notes = f"transitive via {effect['evidence']['via'] or 'unmapped'}"
    if restore:
        notes += f"; reviewed restore contract: {restore['kind']}"
    return _row(
        {
            "Function": effect["function"],
            "Canonical Function": effect["alias_of"]
            or definition["canonical_target"]
            or "",
            "Parameter": effect["parameter"]["name"],
            "Type": effect["parameter"]["type"],
            "Parameter Kind": effect["parameter"]["kind"],
            "Description/Value Expression": effect["value_expr"] or "",
            "Architecture": effect["architecture"],
            "Thread": effect["thread"],
            "Stage": effect["stage"],
            "Stability": effect["stability"],
            "Lifecycle": effect["lifecycle"],
            "Classification": definition["classification"],
            "Classification Reason": definition["reason"],
            "Condition": effect["condition"] or "",
            "Condition Kind": effect["condition_kind"] or "",
            "State Domain": effect["domain"],
            "Register/Configuration Resource": (
                effect["resource"] if effect["resource"] != "-" else ""
            ),
            "Operation": effect["operation"],
            "Persistence": effect["persistence"],
            "Retention Contract": effect["retention_contract"],
            "Activation": effect["activation"],
            "Restore Contract Kind": restore.get("kind", ""),
            "Restore Owner/Pair": owner_pair,
            "Confidence": effect["confidence"],
            "Evidence Kind": effect["evidence"]["kind"],
            "Helper Chain": " -> ".join(
                hop["function"] for hop in effect["evidence"]["via_chain"]
            ),
            "Sink Source Path": sink["source_path"],
            "Sink Source Line": str(sink["line"]),
            "Sink Body Fingerprint": sink["body_fingerprint"],
            "Source Path": effect["evidence"]["source_path"],
            "Source Line": str(effect["evidence"]["line"]),
            "Body Fingerprint": effect["evidence"]["body_fingerprint"],
            "Generator Base Commit": revision,
            "Analyzed Source Fingerprint": source_fingerprint,
            "Mapped Status": "effect",
            "Notes": notes,
        }
    )


def _classification_row(
    function: dict[str, Any],
    definition: dict[str, Any],
    revision: str,
    source_fingerprint: str,
    restore: dict[str, Any] | None,
) -> dict[str, str]:
    lifecycle = function["lifecycle"]
    if definition["classification"] == "excluded":
        notes = f"excluded: {definition['reason']}"
    elif definition["classification"] == "wrapper":
        notes = "classification-only wrapper; no independently mapped effect"
    elif lifecycle == "uninit":
        notes = "no-op teardown or validation-only/unknown; no mapped effect"
    else:
        notes = f"validation-only/unknown lifecycle: {lifecycle}; no mapped effect"
    if restore:
        notes += f"; reviewed restore contract: {restore['kind']}"
    return _row(
        {
            "Function": function["name"],
            "Canonical Function": definition["canonical_target"] or "",
            "Architecture": function["architecture"],
            "Thread": function["thread"],
            "Stage": function["stage"],
            "Stability": function["stability_tier"],
            "Lifecycle": lifecycle,
            "Classification": definition["classification"],
            "Classification Reason": definition["reason"],
            "Restore Contract Kind": restore["kind"] if restore else "",
            "Restore Owner/Pair": _restore_owner_pair(restore),
            "Source Path": function["source"]["path"],
            "Source Line": str(function["source"]["start_line"]),
            "Body Fingerprint": function["body_fingerprint"],
            "Generator Base Commit": revision,
            "Analyzed Source Fingerprint": source_fingerprint,
            "Mapped Status": "classification-only",
            "Notes": notes,
        }
    )


def _restore_owner_pair(restore: dict[str, Any] | None) -> str:
    return " / ".join(
        dict.fromkeys(
            value
            for value in (
                (restore or {}).get("owner"),
                (restore or {}).get("pair"),
            )
            if value
        )
    )


def _row(values: dict[str, str]) -> dict[str, str]:
    return {header: str(values.get(header) or "") for header in CSV_HEADERS}


def _identity(item: dict[str, Any]) -> tuple[str, str, str, int, str]:
    return (
        item["architecture"],
        item["name"],
        item["source"]["path"],
        item["source"]["start_line"],
        item["body_fingerprint"],
    )


def _row_sort_key(
    function: dict[str, Any], effect: dict[str, Any] | None = None
) -> tuple[Any, ...]:
    return (
        function["architecture"],
        function["source"]["path"],
        function["source"]["start_line"],
        function["name"],
        effect["evidence"]["line"] if effect else -1,
        effect["domain"] if effect else "",
        effect["resource"] if effect else "",
        effect["parameter"]["name"] if effect else "",
    )


def _revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _audit_fingerprint(audit_data: dict[str, Any]) -> str:
    payload = json.dumps(
        audit_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_fingerprint(
    root: Path,
    effect_model: Path | str | None,
) -> str:
    """Hash every source/model input that determines the generated state map."""
    inputs: list[tuple[str, Path]] = []
    for architecture in ("wormhole_b0", "blackhole", "quasar"):
        architecture_root = root / f"tt_llk_{architecture}"
        if architecture_root.is_dir():
            inputs.extend(
                (source.relative_to(root).as_posix(), source)
                for source in sorted(architecture_root.rglob("*.h"))
            )
    model_path = (
        Path(effect_model)
        if effect_model is not None
        else Path(__file__).with_name("state_effects.json")
    )
    for label, path in (
        ("effect_model", model_path),
        ("candidate_model", Path(__file__).with_name("candidate_model.json")),
        (
            "verification_manifest",
            Path(__file__).with_name("verification_manifest.json"),
        ),
    ):
        if path.is_file():
            inputs.append((label, path))
    digest = hashlib.sha256()
    for label, path in sorted(inputs, key=lambda item: item[0]):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _recorded_revision(destination: Path) -> str | None:
    """Return the informational generator base stamped into an existing map.

    Drift detection must compare audit *content*, not the provenance stamp.
    ``HEAD`` differs from the commit an artifact was generated at on every
    pull-request merge ref and even in the artifact's own commit, so re-stamping
    with the recorded revision keeps that movement from reading as false drift;
    a real source change still surfaces because the regenerated content diverges.
    """
    try:
        recorded = json.loads(
            (Path(destination) / "llk_state_map.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    revision = (
        recorded.get("generator_base_commit") if isinstance(recorded, dict) else None
    )
    return revision if isinstance(revision, str) and revision else None


def _readme(
    audit_data: dict[str, Any],
    revision: str,
    source_fingerprint: str,
    verification_manifest: dict[str, Any],
) -> str:
    summary = audit_data["summary"]
    verification_count = len(verification_manifest["tests"])
    verification_scopes = sorted(
        {entry["validation_scope"] for entry in verification_manifest["tests"]}
    )
    return f"""# LLK state audit

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

- Definitions: {summary["definition_count"]}
- Effect rows: {summary["effect_count"]}
- Classification: {json.dumps(summary["by_classification"], sort_keys=True)}
- Architectures: {json.dumps(summary["by_architecture"], sort_keys=True)}
- Analyzed source fingerprint: `{source_fingerprint}`
- Generator base commit (informational): `{revision}`

## Verification matrix

The reviewed verification manifest links {verification_count} existing LLK tests to audited functions and state domains. Covered validation scopes: {", ".join(verification_scopes) or "none"}. These links identify relevant checks; they do not claim that hardware tests ran during artifact generation. See `tools/llk_state_audit/verification_manifest.json` for the exact architecture, function, family, domain, and test-path mappings.

## Regeneration and drift

From `tt_metal/tt-llk`, run:

```sh
python3 -m tools.llk_state_audit generate --root .
python3 -m tools.llk_state_audit verify --root .
python3 -m tools.llk_state_audit check --root .
```

Regenerate twice and compare bytes when reviewing renderer changes. A changed source fingerprint, classification, or reviewed restore anchor is audit drift and should be reviewed in code and the effect model before updating artifacts.
"""
