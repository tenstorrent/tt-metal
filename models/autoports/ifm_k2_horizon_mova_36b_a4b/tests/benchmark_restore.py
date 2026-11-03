"""Restore exact raw benchmark evidence after a checkout, without device access."""
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    out = Path(__file__).resolve().parents[1] / "doc/benchmark/run"
    archive = out / "archive"
    index = json.loads((archive / "index.json").read_text())
    chunks = []
    for row in index["chunks"]:
        data = (archive / row["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"Corrupt archive chunk: {row['path']}")
        chunks.append(data)
    data = b"".join(chunks)
    if hashlib.sha256(data).hexdigest() != index["archive_sha256"]:
        raise ValueError("Corrupt combined archive")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as bundle:
        for member in bundle:
            relative = Path(member.name)
            if not member.isfile() or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Remove only the previously-created evidence symlink, not its target.
            if target.is_symlink():
                target.unlink()
            target.write_bytes(bundle.extractfile(member).read())
    for row in index["files"]:
        if hashlib.sha256((out / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Restored file checksum mismatch: {row['path']}")
    print(f"Verified {len(index['files'])} exact raw evidence files")


if __name__ == "__main__":
    main()
