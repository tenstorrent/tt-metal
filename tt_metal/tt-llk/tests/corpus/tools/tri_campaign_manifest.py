#!/usr/bin/env python3
"""Assemble a reviewed tri-arm campaign into a small immutable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


SCHEMA = "tt-llk-tri-campaign-v1"
PROFILE_FIELDS = (
    "op", "category", "full_space", "state",
    "a_selected_sem_node", "selected_flags",
    "b_baseline_sem_node", "baseline_flags",
    "c_baseline_hand_node", "c_baseline_flags",
)
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
OP = re.compile(r"[A-Za-z0-9_.+-]+")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_roster(path: Path) -> list[str]:
    text = path.read_text()
    if not text:
        raise ValueError("roster is empty")
    ops: list[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line or line.strip() != line or not OP.fullmatch(line):
            raise ValueError(f"roster line {number} is malformed")
        if line in ops:
            raise ValueError(f"duplicate roster op: {line}")
        ops.append(line)
    return ops


def tsv_lines(path: Path, label: str) -> list[str]:
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"{label} is empty")
    for number, line in enumerate(lines, 1):
        if not line:
            raise ValueError(f"{label} line {number} is blank")
    return lines


def read_profiles(path: Path) -> dict[str, dict[str, str]]:
    lines = tsv_lines(path, "profiles")
    header = tuple(lines[0].split("\t"))
    if header != PROFILE_FIELDS:
        raise ValueError("profiles header must be the exact 10-column tri profile schema")
    rows: dict[str, dict[str, str]] = {}
    for number, line in enumerate(lines[1:], 2):
        values = line.split("\t")
        if len(values) != len(PROFILE_FIELDS):
            raise ValueError(f"profiles line {number} does not have 10 columns")
        row = dict(zip(PROFILE_FIELDS, values))
        op = row["op"]
        if not OP.fullmatch(op):
            raise ValueError(f"profiles line {number} has malformed op")
        if op in rows:
            raise ValueError(f"duplicate profile op: {op}")
        if any(row[field] == "" for field in PROFILE_FIELDS if field != "selected_flags"):
            raise ValueError(f"profiles line {number} has an empty required field")
        try:
            if int(row["full_space"]) <= 0:
                raise ValueError
        except ValueError as error:
            raise ValueError(f"{op}: full_space must be a positive decimal integer") from error
        if row["state"] != "GAP":
            raise ValueError(f"{op}: profile state is not GAP")
        if row["a_selected_sem_node"] != row["b_baseline_sem_node"]:
            raise ValueError(f"{op}: A/B semantic nodes differ")
        if row["baseline_flags"] != row["c_baseline_flags"]:
            raise ValueError(f"{op}: B/C baseline flags differ")
        rows[op] = row
    if not rows:
        raise ValueError("profiles has no rows")
    return rows


def read_idmap(path: Path) -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}
    for number, line in enumerate(tsv_lines(path, "identity map"), 1):
        values = tuple(line.split("\t"))
        if len(values) != 7:
            raise ValueError(f"identity map line {number} does not have 7 columns")
        op = values[0]
        if not OP.fullmatch(op):
            raise ValueError(f"identity map line {number} has malformed op")
        if op in rows:
            raise ValueError(f"duplicate identity-map op: {op}")
        if any(not HEX64.fullmatch(value) for value in values[1:]):
            raise ValueError(f"{op}: identity map contains an invalid SHA-256")
        rows[op] = values
    return rows


def read_search(path: Path) -> tuple[dict, str]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid search JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("search root is not an object")
    operations = data.get("operations")
    settings = data.get("settings")
    if not isinstance(operations, dict) or not isinstance(settings, dict):
        raise ValueError("search needs operations and settings objects")
    baseline = settings.get("baseline_flags")
    if not isinstance(baseline, str) or not baseline:
        raise ValueError("search settings.baseline_flags is missing or empty")
    return operations, baseline


def search_flags(operations: dict, op: str, baseline: str) -> str:
    record = operations.get(op)
    if not isinstance(record, dict):
        raise ValueError(f"{op}: missing from search operations")
    proposal = record.get("proposal")
    selection = proposal.get("selection") if isinstance(proposal, dict) else None
    flags = selection.get("flags") if isinstance(selection, dict) else None
    if not isinstance(flags, str):
        raise ValueError(f"{op}: search proposal.selection.flags is missing or malformed")
    if proposal.get("frozen_baseline_flags") != baseline:
        raise ValueError(f"{op}: search frozen baseline disagrees with settings")
    return flags


def git_hash(value: str, name: str) -> str:
    if not (HEX40.fullmatch(value) or HEX64.fullmatch(value)):
        raise ValueError(f"{name} must be a lowercase 40- or 64-digit hexadecimal hash")
    return value


def optional_hash(value: str | None, name: str, *, sha256_only: bool = False) -> str | None:
    if value is None:
        return None
    pattern = HEX64 if sha256_only else None
    if pattern is not None:
        if not pattern.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase 64-digit SHA-256")
        return value
    return git_hash(value, name)


def build(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise ValueError(f"output path already exists (a fresh path is required): {args.out}")

    roster = read_roster(args.roster)
    profiles = read_profiles(args.profiles)
    idmap = read_idmap(args.idmap)
    operations, global_baseline = read_search(args.search)
    roster_set = set(roster)
    profile_set = set(profiles)
    if roster_set != profile_set:
        missing = sorted(roster_set - profile_set)
        extra = sorted(profile_set - roster_set)
        raise ValueError(f"roster/profile op mismatch: missing={missing}, extra={extra}")
    missing_id = sorted(roster_set - set(idmap))
    if missing_id:
        raise ValueError(f"identity map is missing roster ops: {missing_id}")

    producer = git_hash(args.producer_tt_metal_head, "producer tt-metal head")
    runner = git_hash(args.runner_tt_metal_head, "runner tt-metal head")
    identities = {
        "producer_tt_metal_head": producer,
        "runner_tt_metal_head": runner,
    }
    optional = (
        ("sfpi_head", optional_hash(args.sfpi_head, "sfpi head")),
        ("sfpi_gcc_head", optional_hash(args.sfpi_gcc_head, "sfpi-gcc head")),
        ("compiler_sha256", optional_hash(args.compiler_sha256, "compiler sha256", sha256_only=True)),
    )
    identities.update((key, value) for key, value in optional if value is not None)

    eligible = sorted(roster_set)
    for op in eligible:
        row = profiles[op]
        selected = search_flags(operations, op, global_baseline)
        if row["selected_flags"] != selected:
            raise ValueError(f"{op}: profile selected flags disagree with pinned search")
        if row["baseline_flags"] != global_baseline:
            raise ValueError(f"{op}: profile baseline flags disagree with pinned search")

    flags_data = "".join(
        f"{op}\t{profiles[op]['selected_flags']}\n" for op in eligible
    ).encode()
    idmap_data = "".join("\t".join(idmap[op]) + "\n" for op in eligible).encode()
    manifest = {
        "schema_version": 1,
        "schema": SCHEMA,
        "status": "READY",
        # These flat fields are the craq-sfpi assembler authority contract.
        "eligible_ops": len(eligible),
        "search_sha256": file_sha256(args.search),
        "flags_tsv_sha256": sha256_bytes(flags_data),
        "idmap_sha256": sha256_bytes(idmap_data),
        # The operation names and source-table hashes remain review metadata.
        "ops": eligible,
        "roster_sha256": file_sha256(args.roster),
        "profiles_sha256": file_sha256(args.profiles),
        "identities": identities,
    }

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "flags.tsv").write_bytes(flags_data)
    (args.out / "idmap.tsv").write_bytes(idmap_data)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--profiles", "--tri-profiles", type=Path, required=True)
    parser.add_argument("--idmap", "--tri-idmap", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--producer-tt-metal-head", required=True)
    parser.add_argument("--runner-tt-metal-head", required=True)
    parser.add_argument("--sfpi-head")
    parser.add_argument("--sfpi-gcc-head")
    parser.add_argument("--compiler-sha256", "--compiler-sha", dest="compiler_sha256")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser_args = parse_args(argv)
    try:
        build(parser_args)
    except (OSError, ValueError) as error:
        print(f"tri_campaign_manifest: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
