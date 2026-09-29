"""Preserve exact raw evidence within the repository's 500KB/file policy."""

import argparse
import hashlib
import json
import lzma
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, required=True)
    args = parser.parse_args()
    root = args.doc
    archive_root = root / "evidence_archives"
    archive_root.mkdir(exist_ok=True)
    records = []
    ignore = ["# Large original evidence stays local; exact compact archives are indexed.", "*.tracy"]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or archive_root in path.parents:
            continue
        size = path.stat().st_size
        archive = path.suffix in (".log", ".csv", ".txt") or (path.suffix == ".json" and size > 500_000)
        local_binary = path.suffix in (".pt", ".tracy")
        if not archive and not local_binary:
            continue
        relative = str(path.relative_to(root))
        checksum = digest(path)
        row = {"original": relative, "original_bytes": size, "original_sha256": checksum}
        if archive and size < 100_000_000:
            target = archive_root / f"{checksum[:16]}-{path.name}.xz"
            if not target.exists():
                with path.open("rb") as source, lzma.open(target, "wb", preset=6) as sink:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        sink.write(block)
            archive_size = target.stat().st_size
            row.update(
                archive=str(target.relative_to(root)),
                archive_bytes=archive_size,
                archive_sha256=digest(target),
                archive_committed=archive_size <= 500_000,
            )
            if archive_size > 500_000:
                ignore.append("/" + str(target.relative_to(root)))
                row[
                    "local_only_reason"
                ] = "Exact archive exceeds repository500KB/file limit; compact derived evidence is retained separately."
        else:
            row[
                "local_only_reason"
            ] = "Large raw tensor or profiler capture; compact diagnostics and op CSV retain reviewable evidence."
        row["plain_copy_committed"] = path.suffix in (".json", ".txt") and size <= 500_000
        if path.suffix == ".json" and size > 500_000:
            ignore.append("/" + relative)
        records.append(row)
    result = {
        "scope": "Byte-exact archived logs, op CSVs, tables and oversized diagnostic JSON. Raw tensors and multi-GB profiler captures are preserved locally with hashes; no files are deleted.",
        "restore": "xz -dc ARCHIVE > ORIGINAL, then verify original_sha256. Plain text tables can be whitespace-normalized by repository hooks; archives preserve original bytes.",
        "files": records,
    }
    (root / "evidence_archives.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / ".gitignore").write_text("\n".join(sorted(set(ignore))) + "\n")
    print(json.dumps({"files": len(records), "archives": sum(bool(row.get("archive_committed")) for row in records)}))


if __name__ == "__main__":
    main()
