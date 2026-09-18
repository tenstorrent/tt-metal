"""Temporary synthetic IDs for CPU contracts only; never a device tokenization receipt."""

import hashlib
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def token_manifest(capacities=(4096,)):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = []
        for value in capacities:
            for slot in (0, 1):
                path = root / (str(value) + "-" + str(slot) + ".json")
                path.write_text(json.dumps([(i + 37 * slot) % 128256 for i in range(value)]))
                rows.append(
                    dict(
                        slot=slot,
                        context_length=value,
                        token_ids_file=path.name,
                        token_ids_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
        path = root / "manifest.json"
        path.write_text(json.dumps(dict(validation_passed=True, scope="synthetic host tests only", fixtures=rows)))
        yield path, hashlib.sha256(path.read_bytes()).hexdigest()
