"""Keep lossless stage logs in Git without adding ignored raw log files."""

import hashlib
import json
import lzma
from pathlib import Path


def main():
    model = Path(__file__).resolve().parents[1]
    repo = Path(__file__).resolve().parents[4]
    root = model / "doc/optimized_vllm"
    archive = root / "evidence_archives"
    archive.mkdir(exist_ok=True)
    entries = []
    paths = sorted(root.rglob("*.log")) + sorted((model / "readiness_vllm").glob("*.log"))
    for path in paths:
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        target = archive / f"{sha[:16]}-{path.name}.xz"
        target.write_bytes(lzma.compress(raw))
        assert lzma.decompress(target.read_bytes()) == raw
        entries.append(
            dict(source=str(path.relative_to(repo)), archive=str(target.relative_to(root)), sha256=sha, bytes=len(raw))
        )
    (archive / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Archived {len(entries)} logs losslessly")


if __name__ == "__main__":
    main()
