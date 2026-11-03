"""Keep exact raw evidence in bounded, locally committed compressed chunks."""
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    out = Path(__file__).resolve().parents[1] / "doc/benchmark/run"
    archive = out / "archive"
    archive.mkdir(exist_ok=True)
    keep = {"REPORT.md", "summary.json", "run_config.json", "manifest.json", "stage_timeline.json", ".gitignore"}
    paths = sorted(
        path
        for path in out.rglob("*")
        if path.is_file() and archive not in path.parents and path.relative_to(out).as_posix() not in keep
    )
    contents = io.BytesIO()
    records = []
    with tarfile.open(fileobj=contents, mode="w:gz", dereference=True) as bundle:
        for path in paths:
            raw = path.read_bytes()
            relative = path.relative_to(out).as_posix()
            records.append(dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            info = tarfile.TarInfo(relative)
            info.size = len(raw)
            info.mode = 0o640
            info.mtime = int(path.stat().st_mtime)
            bundle.addfile(info, io.BytesIO(raw))
    compressed = contents.getvalue()
    chunks = []
    for index, start in enumerate(range(0, len(compressed), 450 * 1024)):
        name = f"raw.tar.gz.part{index:03d}"
        data = compressed[start : start + 450 * 1024]
        (archive / name).write_bytes(data)
        chunks.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    for path in archive.glob("raw.tar.gz.part*"):
        if path.name not in {row["path"] for row in chunks}:
            path.unlink()
    index = dict(
        format="concatenated gzip tar, exact source bytes",
        archive_sha256=hashlib.sha256(compressed).hexdigest(),
        chunks=chunks,
        files=records,
    )
    (archive / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    (out / ".gitignore").write_text(
        "# Exact raw evidence is committed in archive/; keep local expanded files.\n*\n!archive/\n!archive/**\n!REPORT.md\n!summary.json\n!run_config.json\n!manifest.json\n!stage_timeline.json\n!.gitignore\n"
    )
    print(json.dumps(dict(files=len(records), compressed_bytes=len(compressed), chunks=len(chunks))))


if __name__ == "__main__":
    main()
