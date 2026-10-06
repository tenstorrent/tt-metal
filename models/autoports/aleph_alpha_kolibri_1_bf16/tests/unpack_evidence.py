# SPDX-License-Identifier: Apache-2.0
"""Restore exact readable artifacts from repository-sized compressed copies."""

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for row in json.loads((ROOT / "doc/functional_decoder/compressed_evidence.json").read_text()):
    data = gzip.decompress((ROOT / row["gzip"]).read_bytes())
    assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], row["path"]
    target = ROOT / row["path"]
    if target.exists():
        assert target.read_bytes() == data, f"Refusing to replace a different artifact: {target}"
    else:
        target.write_bytes(data)
print("Exact evidence copies verified/restored.")
