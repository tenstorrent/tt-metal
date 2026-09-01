#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Emit one portable cluster health record (stdout JSON line; optional file).

Sibling to analyze_*.py. Call after analyze on the same host. Does not change
analyze pass/fail: this process exits 0 after printing JSON even when status is
failed. Store I/O errors warn on stderr and still print stdout JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cluster_health_schema import SCHEMA_ID, TEST_TYPES, validate_record
from report_adapters import reason_for, status_for
from report_backfill import Leftover, discover_leftovers, filter_leftovers, leftover_key, parse_window_date
from analyze_host_health_results import parse_diag_report
from resolve_host_ring_order import (
    parse_textproto,
    read_descriptor_text,
    resolve_leaf_host_ids,
)

STORE_ROOT_ENV = "CLUSTER_HEALTH_STORE_ROOT"
# setgid + sticky + owner/group rwx. mkdir is umask-masked (often 0755), so the
# first writer of the day would otherwise lock out the store's group. Sticky
# keeps non-owner writers from unlinking each other's records; as usual, the
# directory owner can still unlink any entry. No other-write unless the store
# root is already other-writable (then date dirs follow that and stay sticky).
STORE_DIR_MODE = 0o3770
STORE_DIR_MODE_WORLD = 0o1777


def date_dir_mode_for_root(root_mode: int) -> int:
    """Match a world-writable store root so mixed uids can share the date dir."""
    if root_mode & 0o002:
        return STORE_DIR_MODE_WORLD
    return STORE_DIR_MODE


def dumps_compact(obj: dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def parse_hosts(hosts_arg: str) -> list[str]:
    return [h.strip() for h in hosts_arg.split(",") if h.strip()]


def parse_labels(label_args: list[str] | None) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in label_args or []:
        if "=" not in item:
            raise ValueError(f"label: expected key=value, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError("label: key must be a non-empty string")
        labels[key] = value
    return labels


def format_ts_utc(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError("ts: must include a UTC offset")
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def artifact_uri_from_dir(artifact_dir: str) -> str:
    path = Path(artifact_dir)
    if path.is_absolute():
        return os.path.abspath(artifact_dir)
    return artifact_dir


def compute_record_id(
    test_type: str,
    hosts: list[str],
    artifact_uri: str,
    analyzer_code: int | None,
    ts: str,
) -> str:
    payload = json.dumps(
        {
            "analyzer_code": analyzer_code,
            "artifact_uri": artifact_uri,
            "hosts": sorted(hosts),
            "test_type": test_type,
            "ts": ts,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _host_matches(name: str, run_hosts: list[str]) -> bool:
    wanted = {(h.lower(), h.split(".")[0].lower()) for h in run_hosts}
    n = name.lower()
    short = n.split(".")[0]
    for full, wshort in wanted:
        if n == full or short == wshort or n == wshort or short == full:
            return True
    return False


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _int_or_str(value: Any) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def _physical_from_host_obj(obj: dict[str, Any]) -> dict[str, Any] | None:
    hostname = obj.get("hostname") or obj.get("host")
    if not hostname:
        return None
    item: dict[str, Any] = {"hostname": str(hostname)}
    if "aisle" in obj:
        item["aisle"] = str(obj["aisle"])
    rack = _int_or_str(obj.get("rack"))
    if rack is not None and "rack" in obj:
        item["rack"] = rack
    shelf = _int_or_str(obj.get("shelf_u"))
    if shelf is not None and "shelf_u" in obj:
        item["shelf_u"] = shelf
    return item


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def _read_optional(path: str | None) -> str | None:
    """Return file text, or None after warning if the path is missing or unreadable.

    Catches Exception so optional descriptor I/O cannot fail the report if
    read_descriptor_text changes which error types it raises.
    """
    if not path:
        return None
    try:
        return read_descriptor_text(path)
    except Exception as exc:
        _warn(f"could not read {path}: {exc}")
        return None


def topology_from_cabling_deployment(
    cabling_text: str,
    deployment_text: str,
    run_hosts: list[str],
) -> dict[str, Any]:
    cabling = parse_textproto(cabling_text)
    deployment = parse_textproto(deployment_text)
    root = cabling.get("root_instance", {})
    if not isinstance(root, dict):
        root = {}
    path_to_hid = resolve_leaf_host_ids(root, {})

    dep_hosts = [h for h in _as_list(deployment.get("hosts")) if isinstance(h, dict)]
    hid_to_name: dict[int, str] = {}
    physical: list[dict[str, Any]] = []
    for idx, host_obj in enumerate(dep_hosts):
        item = _physical_from_host_obj(host_obj)
        if item is None:
            continue
        hid_to_name[idx] = item["hostname"]
        if _host_matches(item["hostname"], run_hosts):
            physical.append(item)

    instance_paths: list[str] = []
    for path, hid in path_to_hid.items():
        name = hid_to_name.get(hid)
        if name is None or _host_matches(name, run_hosts):
            instance_paths.append(path)
    instance_paths.sort()

    out: dict[str, Any] = {}
    if instance_paths:
        out["instance_paths"] = instance_paths
    if physical:
        out["physical"] = physical
    return out


def topology_from_fsd(fsd_text: str, run_hosts: list[str]) -> dict[str, Any]:
    fsd = parse_textproto(fsd_text)
    physical: list[dict[str, Any]] = []
    for host_obj in _as_list(fsd.get("hosts")):
        if not isinstance(host_obj, dict):
            continue
        item = _physical_from_host_obj(host_obj)
        if item is None:
            continue
        if _host_matches(item["hostname"], run_hosts):
            physical.append(item)
    if physical:
        return {"physical": physical}
    return {}


def parse_gsd_hostnames(text: str) -> list[str]:
    """Extract ``compute_node_specs`` mapping keys without a YAML library."""
    hostnames: list[str] = []
    in_section = False
    for line in text.splitlines():
        if line.startswith("compute_node_specs:"):
            in_section = True
            continue
        if not in_section:
            continue
        if line and not line[0].isspace():
            break
        match = re.match(r"^  ([^:\s][^:]*):", line)
        if match:
            hostnames.append(match.group(1).strip())
    return hostnames


def topology_from_gsd(gsd_text: str, run_hosts: list[str]) -> dict[str, Any]:
    physical: list[dict[str, Any]] = []
    for name in parse_gsd_hostnames(gsd_text):
        if _host_matches(name, run_hosts):
            physical.append({"hostname": name})
    if physical:
        return {"physical": physical}
    return {}


def parse_rankfile(text: str) -> dict[int, str]:
    ranks: dict[int, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*rank\s+(\d+)\s*=\s*(\S+)", line)
        if match:
            ranks[int(match.group(1))] = match.group(2)
    return ranks


def parse_rank_bindings_yaml(text: str) -> list[dict[str, Any]]:
    """Parse the small rank_bindings YAML subset (stdlib only)."""
    bindings: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_list = False
    skip_nested = 0
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if stripped.startswith("rank_bindings:"):
            in_list = True
            continue
        if not in_list:
            continue
        if skip_nested and indent > skip_nested:
            continue
        skip_nested = 0
        if stripped.startswith("- "):
            if current:
                bindings.append(current)
            current = {}
            rest = stripped[2:]
            if ":" in rest:
                key, value = rest.split(":", 1)
                _assign_rank_field(current, key.strip(), value.strip())
            continue
        if current is None:
            continue
        if (
            stripped.endswith(":")
            and ":" == stripped[-1]
            and stripped[:-1].strip()
            not in {
                "rank",
                "mesh_id",
                "mesh_host_rank",
                "host",
            }
        ):
            skip_nested = indent
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            _assign_rank_field(current, key.strip(), value.strip().strip("'\""))
    if current:
        bindings.append(current)
    return bindings


def _assign_rank_field(current: dict[str, Any], key: str, value: str) -> None:
    if key in ("rank", "mesh_id", "mesh_host_rank"):
        text = value.strip().strip("'\"")
        # An absent or null mesh_host_rank is legal in tt-run bindings; leave the key unset.
        if text.lstrip("-").isdigit():
            current[key] = int(text)
    elif key == "host" and value:
        current["host"] = value


def topology_from_rank(
    rankfile_text: str | None,
    bindings_text: str | None,
) -> dict[str, Any]:
    if not bindings_text:
        return {}
    bindings = parse_rank_bindings_yaml(bindings_text)
    hosts_by_rank = parse_rankfile(rankfile_text) if rankfile_text else {}
    out: list[dict[str, Any]] = []
    for item in bindings:
        if not all(k in item for k in ("rank", "mesh_id")):
            continue
        entry: dict[str, Any] = {
            "rank": item["rank"],
            "mesh_id": item["mesh_id"],
        }
        if "mesh_host_rank" in item:
            entry["mesh_host_rank"] = item["mesh_host_rank"]
        host = item.get("host") or hosts_by_rank.get(item["rank"])
        if host:
            entry["host"] = host
        out.append(entry)
    if out:
        return {"rank_bindings": out}
    return {}


def merge_topology(parts: list[dict[str, Any]]) -> dict[str, Any]:
    instance_paths: list[str] = []
    physical_by_host: dict[str, dict[str, Any]] = {}
    rank_bindings: list[dict[str, Any]] = []
    for part in parts:
        for path in part.get("instance_paths", []):
            if path not in instance_paths:
                instance_paths.append(path)
        for item in part.get("physical", []):
            key = item["hostname"]
            existing = physical_by_host.get(key, {"hostname": key})
            merged = dict(existing)
            merged.update(item)
            physical_by_host[key] = merged
        rank_bindings.extend(part.get("rank_bindings", []))
    out: dict[str, Any] = {}
    if instance_paths:
        out["instance_paths"] = instance_paths
    if physical_by_host:
        out["physical"] = list(physical_by_host.values())
    if rank_bindings:
        out["rank_bindings"] = rank_bindings
    return out


def build_topology(
    *,
    hosts: list[str],
    cabling: str | None,
    deployment: str | None,
    fsd: str | None,
    gsd: str | None,
    rankfile: str | None,
    rank_bindings: str | None,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    cabling_text = _read_optional(cabling)
    deployment_text = _read_optional(deployment)
    if cabling_text and deployment_text:
        parts.append(topology_from_cabling_deployment(cabling_text, deployment_text, hosts))
    elif cabling or deployment:
        _warn("cabling and deployment must both be readable to derive topology from descriptors")

    fsd_text = _read_optional(fsd)
    if fsd_text:
        parts.append(topology_from_fsd(fsd_text, hosts))

    gsd_text = _read_optional(gsd)
    if gsd_text:
        parts.append(topology_from_gsd(gsd_text, hosts))

    rankfile_text = _read_optional(rankfile)
    bindings_text = _read_optional(rank_bindings)
    if bindings_text:
        parts.append(topology_from_rank(rankfile_text, bindings_text))

    return merge_topology(parts)


@dataclass
class RecordRequest:
    """Inputs consumed by ``build_record``.

    CLI, leftover backfill, and ``--from-diag-report`` must construct this type
    (see ``record_request_from_cli`` / ``leftover_namespace``). A new field
    without a default fails at construction instead of as a backfill-only
    ``AttributeError``.
    """

    test_type: str
    hosts: str
    analyzer_code: int | None
    artifact_dir: str
    cabling: str | None
    deployment: str | None
    fsd: str | None
    gsd: str | None
    rankfile: str | None
    rank_bindings: str | None
    source: str | None
    triggered_by: str | None
    trigger_kind: str | None
    orchestrator_id: str | None
    cluster: str | None
    label: list[str]
    duration_s: float | None
    ts: str | None
    incomplete: bool = False
    incomplete_reason: str = ""


def _cli_overlay(args: argparse.Namespace) -> dict[str, Any]:
    """CLI fields shared by live, leftover, and diag-report record requests.

    Keys must match ``RecordRequest`` overlay attributes. Adding a field here
    without the matching dataclass field (or the reverse) raises TypeError.
    """
    return {
        "cabling": args.cabling,
        "deployment": args.deployment,
        "fsd": args.fsd,
        "gsd": args.gsd,
        "rankfile": args.rankfile,
        "rank_bindings": args.rank_bindings,
        "source": args.source,
        "triggered_by": args.triggered_by,
        "trigger_kind": args.trigger_kind,
        "orchestrator_id": args.orchestrator_id,
        "cluster": args.cluster,
        "label": list(args.label or []),
    }


def record_request_from_cli(args: argparse.Namespace) -> RecordRequest:
    return RecordRequest(
        test_type=args.test_type,
        hosts=args.hosts,
        analyzer_code=args.analyzer_code,
        artifact_dir=args.artifact_dir,
        duration_s=args.duration_s,
        ts=args.ts,
        incomplete=bool(getattr(args, "incomplete", False)),
        incomplete_reason=getattr(args, "incomplete_reason", "") or "",
        **_cli_overlay(args),
    )


def build_record(args: RecordRequest) -> dict[str, Any]:
    hosts = parse_hosts(args.hosts)
    if not hosts:
        raise ValueError("hosts: must be a non-empty array")

    analyzer_code: int | None = args.analyzer_code
    incomplete = bool(args.incomplete)
    if incomplete:
        status = "degraded"
    else:
        if args.test_type != "recover" and analyzer_code is None:
            raise ValueError("analyzer_code: required except for recover")
        status = status_for(args.test_type, analyzer_code)
    ts = format_ts_utc(args.ts)
    record: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "ts": ts,
        "test_type": args.test_type,
        "status": status,
        "hosts": hosts,
    }
    if args.test_type != "recover" and analyzer_code is not None and not incomplete:
        record["analyzer_code"] = analyzer_code

    record["artifact_uri"] = artifact_uri_from_dir(args.artifact_dir)

    topology = build_topology(
        hosts=hosts,
        cabling=args.cabling,
        deployment=args.deployment,
        fsd=args.fsd,
        gsd=args.gsd,
        rankfile=args.rankfile,
        rank_bindings=args.rank_bindings,
    )
    if topology:
        record["topology"] = topology

    for field in ("cluster", "source", "triggered_by", "trigger_kind", "orchestrator_id"):
        value = getattr(args, field.replace("-", "_"), None)
        if value:
            record[field] = value

    if args.duration_s is not None:
        record["duration_s"] = args.duration_s

    labels = parse_labels(args.label)
    if status != "passed" and "failure_reason" not in labels:
        if incomplete:
            detail = args.incomplete_reason or "missing_terminal_outcome"
            reason = f"Incomplete run ({detail.replace('_', ' ')})"
        else:
            reason = reason_for(args.test_type, analyzer_code)
        if reason:
            labels["failure_reason"] = reason
    if incomplete:
        labels["incomplete"] = "true"
        labels["incomplete_reason"] = args.incomplete_reason or "missing_terminal_outcome"
    if labels:
        record["labels"] = labels

    return record


def _payload_matches(existing: bytes, payload_bytes: bytes) -> bool:
    try:
        return json.loads(existing) == json.loads(payload_bytes)
    except json.JSONDecodeError:
        return existing == payload_bytes


def _existing_or_conflict(
    date_dir_fd: int,
    dest_name: str,
    dest: Path,
    payload_bytes: bytes,
    record: dict[str, Any],
    published: dict[str, Any],
) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    existing_fd = os.open(dest_name, flags, dir_fd=date_dir_fd)
    with os.fdopen(existing_fd, "rb") as handle:
        existing = handle.read()
    if _payload_matches(existing, payload_bytes):
        return published
    _warn(f"refusing to overwrite different content at {dest}")
    return record


def _ensure_date_dir(root: Path, date_name: str) -> int:
    """Open today's directory securely and return a caller-owned descriptor.

    Only the date directory is chmod'd. It is the one directory this tool owns
    per day, so a mistyped --store-root cannot loosen an unrelated tree.
    ``mkdir`` cannot do this itself: its mode applies to the leaf only
    and is masked by umask, which is how the first writer of the day used to
    leave a 0755 directory owned by their uid.

    If the store root is other-writable, date dirs use STORE_DIR_MODE_WORLD
    (sticky + rwxrwxrwx) instead of STORE_DIR_MODE so mixed uids can share a
    0777 tree. Descriptor-relative operations prevent a shared-root writer
    from replacing the date path with a symlink or swapping it while a record
    is published. chown runs before chmod because a non-privileged chown may
    drop setgid.
    """
    root.mkdir(parents=True, exist_ok=True)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(root, directory_flags)
    try:
        root_stat = os.fstat(root_fd)
        root_gid = root_stat.st_gid
        dir_mode = date_dir_mode_for_root(root_stat.st_mode)
        try:
            os.mkdir(date_name, mode=dir_mode, dir_fd=root_fd)
        except FileExistsError:
            pass
        date_dir_fd = os.open(date_name, directory_flags, dir_fd=root_fd)
    finally:
        os.close(root_fd)

    try:
        try:
            os.fchown(date_dir_fd, -1, root_gid)
        except OSError:
            if os.fstat(date_dir_fd).st_gid != root_gid:
                _warn(f"date directory group does not match store root group ({root_gid})")
        try:
            os.fchmod(date_dir_fd, dir_mode)
        except OSError:
            pass
        return date_dir_fd
    except BaseException:
        os.close(date_dir_fd)
        raise


def publish_record(record: dict[str, Any], store_root: str) -> dict[str, Any]:
    """Atomically write one file under store_root. Returns the record to print.

    Uses an exclusive link (no-clobber). If dest already exists, identical
    content is treated as success; different content is left in place and the
    stdout-only record is returned. On I/O failure, warns and returns the
    stdout-only record (no record_id). The date directory is chmod'd to
    STORE_DIR_MODE (setgid + sticky + group write), or STORE_DIR_MODE_WORLD
    when the store root is already other-writable, so a shared store stays
    writable for every uid that can write DIR; record files themselves follow
    the caller's umask.
    """
    record_id = compute_record_id(
        record["test_type"],
        record["hosts"],
        record.get("artifact_uri", ""),
        record.get("analyzer_code"),
        record["ts"],
    )
    date_dir = record["ts"][:10]
    root = Path(store_root).resolve()
    dest_dir = root / date_dir
    dest = dest_dir / f"{record_id}.json"
    published = dict(record)
    published["record_id"] = record_id
    published["record_uri"] = str(dest)
    try:
        validate_record(published, file_written=True)
        payload = dumps_compact(published) + "\n"
        payload_bytes = payload.encode("utf-8")
        date_dir_fd = _ensure_date_dir(root, date_dir)
        dest_name = f"{record_id}.json"
        tmp_name = f".{record_id}.{os.getpid()}.tmp"
        try:
            tmp_fd = os.open(
                tmp_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o666,
                dir_fd=date_dir_fd,
            )
            with os.fdopen(tmp_fd, "wb") as handle:
                handle.write(payload_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(
                    tmp_name,
                    dest_name,
                    src_dir_fd=date_dir_fd,
                    dst_dir_fd=date_dir_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                return _existing_or_conflict(date_dir_fd, dest_name, dest, payload_bytes, record, published)
        finally:
            try:
                os.unlink(tmp_name, dir_fd=date_dir_fd)
            except OSError:
                pass
            os.close(date_dir_fd)
        return published
    except (OSError, ValueError) as exc:
        _warn(f"store write failed: {exc}")
        return record


def resolve_store_root(args: argparse.Namespace) -> str | None:
    if args.dry_run:
        return None
    if args.store_root:
        return args.store_root
    env = os.environ.get(STORE_ROOT_ENV, "").strip()
    return env or None


def emit_record(record: dict[str, Any], store_root: str | None) -> dict[str, Any]:
    validate_record(record, file_written=False)
    if store_root:
        record = publish_record(record, store_root)
        file_written = "record_id" in record
        try:
            validate_record(record, file_written=file_written)
        except ValueError as exc:
            _warn(str(exc))
            record.pop("record_id", None)
            record.pop("record_uri", None)
            validate_record(record, file_written=False)
    print(dumps_compact(record), flush=True)
    return record


def leftover_namespace(leftover: Leftover, args: argparse.Namespace) -> RecordRequest:
    overlay = _cli_overlay(args)
    labels = list(overlay["label"])
    for key, value in leftover.labels.items():
        labels.append(f"{key}={value}")
    overlay["label"] = labels
    overlay["source"] = args.source or "backfill"
    overlay["trigger_kind"] = args.trigger_kind or "backfill"
    return RecordRequest(
        test_type=leftover.test_type,
        hosts=leftover.hosts,
        analyzer_code=leftover.analyzer_code,
        artifact_dir=leftover.artifact_dir,
        duration_s=leftover.duration_s,
        ts=leftover.ts,
        incomplete=leftover.incomplete,
        incomplete_reason=leftover.incomplete_reason,
        **overlay,
    )


def run_backfill(args: argparse.Namespace) -> int:
    if not args.triggered_by:
        print("Error: --triggered-by is required for --from-artifact-dir", file=sys.stderr)
        return 1
    store_root = resolve_store_root(args)
    if store_root is None and not args.dry_run:
        print(
            "Error: --store-root (or CLUSTER_HEALTH_STORE_ROOT) is required unless --dry-run",
            file=sys.stderr,
        )
        return 1
    root = Path(args.from_artifact_dir)
    if not root.is_dir():
        print(f"Error: --from-artifact-dir is not a directory: {root}", file=sys.stderr)
        return 1
    try:
        from_date = parse_window_date(args.from_date, "--from")
        to_date = parse_window_date(args.to_date, "--to")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    leftovers = filter_leftovers(
        discover_leftovers(root, recursive=bool(args.recursive)),
        from_date=from_date,
        to_date=to_date,
    )
    counts = {
        "discovered": len(leftovers),
        "emitted": 0,
        "degraded": 0,
        "skipped": 0,
        "duplicate": 0,
    }
    seen_keys: set[tuple[str, str, str, str, int | None]] = set()
    for leftover in leftovers:
        key = leftover_key(leftover)
        if key in seen_keys:
            counts["duplicate"] += 1
            _warn(f"duplicate leftover {leftover.source}; skipping")
            continue
        seen_keys.add(key)
        try:
            record = build_record(leftover_namespace(leftover, args))
            emit_record(record, store_root)
        except ValueError as exc:
            counts["skipped"] += 1
            _warn(f"skipping {leftover.source}: {exc}")
            continue
        counts["emitted"] += 1
        if leftover.incomplete:
            counts["degraded"] += 1
    print(
        "backfill summary: "
        f"discovered={counts['discovered']} emitted={counts['emitted']} "
        f"degraded={counts['degraded']} skipped={counts['skipped']} "
        f"duplicate={counts['duplicate']}",
        file=sys.stderr,
    )
    return 0


def run_from_diag_report(args: argparse.Namespace) -> int:
    path = Path(args.from_diag_report)
    if not path.is_file():
        print(f"Error: --from-diag-report is not a file: {path}", file=sys.stderr)
        return 1
    try:
        extract = parse_diag_report(path, artifact_dir=args.artifact_dir)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if extract.dry_run:
        print("Error: refusing dry-run diag report", file=sys.stderr)
        return 1
    ts = args.ts or extract.ts
    if not ts:
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    duration_s = args.duration_s if args.duration_s is not None else extract.duration_s
    overlay = _cli_overlay(args)
    labels = list(overlay["label"])
    for key, value in extract.labels.items():
        labels.append(f"{key}={value}")
    overlay["label"] = labels
    request = RecordRequest(
        test_type="host",
        hosts=extract.hosts,
        analyzer_code=extract.analyzer_code,
        artifact_dir=extract.artifact_dir,
        duration_s=duration_s,
        ts=ts,
        incomplete=extract.incomplete,
        incomplete_reason=extract.incomplete_reason,
        **overlay,
    )
    try:
        record = build_record(request)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    store_root = resolve_store_root(args)
    try:
        emit_record(record, store_root)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emit a portable cluster health record after analyze (stdout JSON line)."
    )
    parser.add_argument("--test-type", choices=sorted(TEST_TYPES))
    parser.add_argument("--hosts", help="Comma-separated hostnames for this run")
    parser.add_argument("--analyzer-code", type=int, default=None, help="Analyzer / recover exit code")
    parser.add_argument("--artifact-dir", help="Analyze / MPI dump path as known to the caller")
    parser.add_argument("--from-artifact-dir", dest="from_artifact_dir", help="Backfill leftover runs under this tree")
    parser.add_argument(
        "--from-diag-report",
        dest="from_diag_report",
        help="Emit one host record from an existing diag_report.json",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="With --from-artifact-dir, discover wrapper logs under nested logs/ dirs and diag_report.json files",
    )
    parser.add_argument("--from", dest="from_date", help="Inclusive UTC start date YYYY-MM-DD (artifact mtime)")
    parser.add_argument("--to", dest="to_date", help="Inclusive UTC end date YYYY-MM-DD (artifact mtime)")
    parser.add_argument("--cabling", help="Cabling descriptor textproto")
    parser.add_argument("--deployment", help="Deployment descriptor textproto")
    parser.add_argument("--fsd", help="Factory system descriptor textproto")
    parser.add_argument("--gsd", help="Discovered global system descriptor YAML")
    parser.add_argument("--rankfile", help="OpenMPI rankfile")
    parser.add_argument("--rank-bindings", dest="rank_bindings", help="rank_bindings YAML")
    parser.add_argument("--source", help="Opaque launcher id (cli, backfill, etc.)")
    parser.add_argument("--triggered-by", dest="triggered_by")
    parser.add_argument("--trigger-kind", dest="trigger_kind")
    parser.add_argument("--orchestrator-id", dest="orchestrator_id")
    parser.add_argument("--cluster", help="Opaque caller cluster string")
    parser.add_argument("--label", action="append", default=[], help="Opaque key=value; repeatable")
    parser.add_argument("--store-root", dest="store_root", help="If set, publish one file under DATE/record_id.json")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON only; never write")
    parser.add_argument("--duration-s", dest="duration_s", type=float)
    parser.add_argument("--ts", help="RFC3339 UTC timestamp (default: now)")
    parser.add_argument("--topology", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.topology is not None:
        parser.error("--topology is not accepted")
    if args.from_artifact_dir and args.from_diag_report:
        parser.error("--from-artifact-dir cannot be combined with --from-diag-report")
    if args.from_diag_report:
        if args.test_type or args.hosts:
            parser.error("--from-diag-report cannot be combined with --test-type / --hosts")
        return run_from_diag_report(args)
    if args.from_artifact_dir:
        if args.test_type or args.hosts or args.artifact_dir:
            parser.error("--from-artifact-dir cannot be combined with --test-type / --hosts / --artifact-dir")
        return run_backfill(args)
    if not args.test_type or not args.hosts or not args.artifact_dir:
        parser.error(
            "--test-type, --hosts, and --artifact-dir are required "
            "(or pass --from-artifact-dir / --from-diag-report)"
        )
    try:
        record = build_record(record_request_from_cli(args))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    store_root = resolve_store_root(args)
    try:
        emit_record(record, store_root)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
