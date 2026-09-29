"""Keep byte-exact benchmark artifacts in bounded, restorable archive parts."""

import hashlib
import json
import lzma
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "doc/benchmark"
ARCHIVES = ROOT / "run/archives"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    rows, ignored = [], ["# Originals remain locally; byte-exact archives below are committed.", "/server-state.json"]
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "archives" in path.relative_to(ROOT).parts:
            continue
        raw_source = (
            # Third-party source/cache files carry frozen content hashes and
            # must not be rewritten by repository formatting hooks.
            (
                "runtime_repair" in path.parts
                and any(part in path.parts for part in ("sources", "hf-hub", "hf-datasets"))
            )
            or "roofline-sources" in path.parts
            or ("run" in path.parts and path.suffix == ".py")
            or ("runtime_repair" in path.parts and path.suffix in (".py", ".cpp", ".hpp"))
        )
        raw_json = path.suffix == ".json" and not path.read_bytes().endswith(b"\n")
        if (
            path.suffix not in (".log", ".jinja", ".html", ".txt", ".patch")
            and not raw_source
            and not raw_json
            and path.stat().st_size <= 480000
        ):
            continue
        data = path.read_bytes()
        compressed = lzma.compress(data, preset=6)
        checksum = digest(data)
        parts = []
        for index, offset in enumerate(range(0, len(compressed), 400000)):
            target = ARCHIVES / f"{checksum[:16]}-{path.name}.xz.part{index:03}"
            block = compressed[offset : offset + 400000]
            target.write_bytes(block)
            parts.append({"path": str(target.relative_to(ROOT)), "bytes": len(block), "sha256": digest(block)})
        rows.append(
            {
                "original": str(path.relative_to(ROOT)),
                "bytes": len(data),
                "sha256": checksum,
                "compression": "xz",
                "compressed_sha256": digest(compressed),
                "parts": parts,
            }
        )
        ignored.append("/" + str(path.relative_to(ROOT)))
    (ARCHIVES / "manifest.json").write_text(
        json.dumps(
            {
                "files": rows,
                "restore": "For each file, concatenate listed parts in order, verify compressed_sha256, decompress xz, "
                "verify original sha256, and write original path relative to doc/benchmark. Originals are retained locally.",
            },
            indent=2,
        )
        + "\n"
    )
    (ROOT / ".gitignore").write_text("\n".join(ignored) + "\n")
    print(json.dumps({"archived_files": len(rows), "parts": sum(len(row["parts"]) for row in rows)}))


if __name__ == "__main__":
    main()
