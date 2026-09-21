"""Verify the portable reproducer using Python's standard library only."""
import hashlib
import json
from pathlib import Path


def verify(root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    manifest = json.loads((root / "SHA256SUMS.json").read_text())
    for name, expected in manifest.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Manifest path escapes the reproducer")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
    return dict(files=len(manifest), all_sha256_verified=True, hardware_imported=False)


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
