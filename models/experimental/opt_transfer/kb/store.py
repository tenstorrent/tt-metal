import json
from pathlib import Path

from models.experimental.opt_transfer.schema import KBEntry


class KBStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, entries: list[KBEntry]) -> None:
        for e in entries:
            (self.root / f"{e.id}.json").write_text(json.dumps(e.to_dict(), indent=2))

    def load(self) -> list[KBEntry]:
        return [KBEntry.from_dict(json.loads(p.read_text())) for p in sorted(self.root.glob("*.json"))]

    def retrieve(self, categories: list[str] | None = None) -> list[KBEntry]:
        entries = self.load()
        if categories is None:
            return entries
        cats = set(categories)
        return [e for e in entries if e.category in cats]
