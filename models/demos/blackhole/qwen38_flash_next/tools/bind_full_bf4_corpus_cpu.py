# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Bind the retained Qwen3.8 BF4 corpus with no device access."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
from datetime import datetime, timezone
from pathlib import Path

from models.demos.blackhole.qwen38_flash_next.diagnostic_bf4 import bind_diagnostic_bf4_corpus


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _device_descriptors() -> list[str]:
    descriptors = []
    for entry in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target.startswith("/dev/tenstorrent"):
            descriptors.append(target)
    return sorted(descriptors)


def _write_json_exclusive(path: Path, document: dict) -> None:
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError("output parent must be an existing absolute directory")
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--verification-result", type=Path, required=True)
    parser.add_argument("--binder-source-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.binder_source_head) is None:
        raise ValueError("binder source head must be exact lowercase 40-hex")
    if os.environ.get("TT_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU binder requires TT_VISIBLE_DEVICES to be the exact empty string")
    before = _device_descriptors()
    if before:
        raise RuntimeError(f"CPU binder inherited device descriptors: {before}")

    corpus = bind_diagnostic_bf4_corpus(args.artifact_root, args.verification_result)
    after = _device_descriptors()
    if after:
        raise RuntimeError(f"CPU binder opened device descriptors: {after}")
    report = {
        "mode": "diagnostic_non_promoting_read_only_bf4_binding_result",
        "status": "pass",
        "production_qualification": False,
        "device_opened": False,
        "device_locks_acquired": False,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "completed_utc": _utc_now(),
        "binder_source_head": args.binder_source_head,
        "identity": corpus.identity.as_dict(),
        "corpus": corpus.summary(),
        "records": [corpus.records[slot].as_dict() for slot in corpus.records],
    }
    _write_json_exclusive(args.output, report)
    print(json.dumps({"status": "pass", "corpus": corpus.summary()}, sort_keys=True))


if __name__ == "__main__":
    main()
