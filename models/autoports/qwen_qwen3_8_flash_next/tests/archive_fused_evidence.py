# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Archive immutable stage evidence without altering any original bytes.

Run after all jobs finish. Raw files remain locally available; tracked XZ copies
and the index reconstruct the same paths/bytes on a fresh checkout. In particular,
historical Python snapshots must never be reformatted with current runtime code.
"""

import argparse
import hashlib
import json
import lzma
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    root = args.root
    paths = []
    local_only = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix == ".xz" or ".logs" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root)
        is_snapshot = relative.parts[0] == "sources" and path.suffix == ".py"
        is_log = path.suffix == ".log"
        is_kernel = relative.parts[0] == "tracy" and "kernels" in relative.parts and path.suffix == ".cpp"
        is_large = path.stat().st_size > 500 * 1024
        if is_snapshot or is_log or is_large or is_kernel:
            paths.append(path)
    if not args.finalize:
        print(json.dumps({"files": len(paths), "raw_bytes": sum(p.stat().st_size for p in paths)}))
        return
    index = []
    ignores = [
        "# Raw evidence remains local; byte-identical XZ copies are tracked.",
        "# Historical snapshots are intentionally excluded from code formatters.",
        "/sources/**/*.py",
        "# Raw Tracy captures are retained locally and identified by capture provenance.",
        "/tracy/**/*.tracy",
        "# Content-addressed profiler kernel bytes must not be reformatted.",
        "/tracy/**/kernels/*.cpp",
    ]
    for path in paths:
        relative = path.relative_to(root)
        # Multi-GB raw profiler exports are retained locally. Their official
        # filtered tables/CSVs are archived separately; do not load/compress the
        # entire raw export merely to rediscover that it exceeds the git gate.
        if relative.parts[0] == "tracy" and path.stat().st_size > 64 * 1024 * 1024:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            local_only.append(dict(path=str(relative), sha256=digest, bytes=path.stat().st_size, storage="local_raw"))
            ignores.append("/" + str(relative))
            continue
        raw = path.read_bytes()
        packed = lzma.compress(raw, preset=6, check=lzma.CHECK_CRC64)
        archive = path.with_suffix(path.suffix + ".xz")
        # Large raw Tracy exports are already excluded by repository rules; don't
        # place oversized substitutes in git. Raw device CSVs get compact archives
        # from the official reporting harness independently.
        if len(packed) > 500 * 1024:
            row = {
                "path": str(relative),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "storage": "local_raw",
            }
            local_only.append(row)
            ignores.append("/" + str(relative))
            continue
        archive.write_bytes(packed)
        index.append(
            dict(
                path=str(relative),
                sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw),
                archive=str(archive.relative_to(root)),
                archive_sha256=hashlib.sha256(packed).hexdigest(),
            )
        )
        if path.stat().st_size > 500 * 1024:
            ignores.append("/" + str(relative))
    manifest = {"format": "xz; decompress to original path", "archives": index, "local_raw_only": local_only}
    # Keep the index itself below the repository artifact size gate. Parts
    # preserve the same row schema and carry hashes in the small root manifest.
    if len(json.dumps(manifest, indent=2).encode()) > 480 * 1024:
        directory = root / "evidence_index_parts"
        directory.mkdir(exist_ok=True)
        manifest["archives"] = []
        manifest["archive_parts"] = []
        for offset in range(0, len(index), 400):
            part = directory / f"archives_{offset // 400:03d}.json"
            payload = (json.dumps(index[offset : offset + 400], indent=2) + "\n").encode()
            part.write_bytes(payload)
            manifest["archive_parts"].append(
                dict(
                    path=str(part.relative_to(root)),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    entries=len(index[offset : offset + 400]),
                )
            )
    (root / "evidence_archives.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / ".gitignore").write_text("\n".join(ignores) + "\n")
    print(json.dumps({"archived": len(index), "local_raw_only": len(local_only)}))


if __name__ == "__main__":
    main()
