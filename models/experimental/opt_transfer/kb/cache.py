import hashlib
import json
from pathlib import Path
from typing import Any, Callable


class ContentCache:
    """Persistent cache keyed by (key, sha256(content)). Used to make KB
    mining incremental: unchanged source files are not re-mined."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str, content: str) -> Path:
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        safe = key.replace("/", "__")
        return self.root / f"{safe}.{h}.json"

    def get_or_compute(self, key: str, content: str, compute: Callable[[], Any]) -> Any:
        p = self._path(key, content)
        if p.exists():
            return json.loads(p.read_text())
        value = compute()
        p.write_text(json.dumps(value))
        return value
