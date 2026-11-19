# Preserved evidence

These gzip files preserve the exact original bytes of reports that exceed the repository size limit or contain native whitespace. `manifest.json` maps each archive to its original path and SHA256. The originals remain available in this workspace; no measured output was edited.

To restore them from `doc/full_model` in a fresh checkout:

```python
import gzip, hashlib, json
from pathlib import Path
for entry in json.loads(Path("checkpoint_evidence/manifest.json").read_text())["entries"]:
    data = gzip.decompress(Path(entry["archive"]).read_bytes())
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    target = Path(entry["original"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
```

Expanded native captures, Tracy traces and the full ops CSV remain local-only at their documented paths.
