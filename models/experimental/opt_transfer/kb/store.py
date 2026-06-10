import json
import re
from pathlib import Path

from models.experimental.opt_transfer.schema import KBEntry

INDEX_FILE = "_index.json"

_NORM_RE = re.compile(r"[^a-z0-9]+")


def op_token(name: str) -> str:
    """Normalize an op name for matching: 'LayerNorm' / 'layer_norm' -> 'layernorm'."""
    return _NORM_RE.sub("", str(name).lower())


def _entry_ops(e: KBEntry) -> list[str]:
    toks = {op_token(p) for p in (e.torch_pattern or [])}
    # Mined fused_op may be a list of op names rather than one string.
    fused = e.fused_op if isinstance(e.fused_op, (list, tuple)) else [e.fused_op]
    toks.update(op_token(str(f).rsplit(".", 1)[-1]) for f in fused if f)
    return sorted(t for t in toks if t)


class KBStore:
    """One JSON file per entry plus a compact _index.json (id -> category/ops/confidence)
    so retrieval can score without loading every record."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _entry_files(self):
        return [p for p in sorted(self.root.glob("*.json")) if p.name != INDEX_FILE]

    def save(self, entries: list[KBEntry]) -> None:
        for e in entries:
            (self.root / f"{e.id}.json").write_text(json.dumps(e.to_dict(), indent=2))
        self._write_index(self.load())

    def _write_index(self, entries: list[KBEntry]) -> None:
        index = [
            {
                "id": e.id,
                "category": e.category,
                "ops": _entry_ops(e),
                "confidence": e.confidence,
                "status": e.status,
            }
            for e in entries
        ]
        (self.root / INDEX_FILE).write_text(json.dumps(index, indent=2))

    def load(self) -> list[KBEntry]:
        return [KBEntry.from_dict(json.loads(p.read_text())) for p in self._entry_files()]

    def load_one(self, entry_id: str) -> KBEntry:
        return KBEntry.from_dict(json.loads((self.root / f"{entry_id}.json").read_text()))

    def _index(self) -> list[dict]:
        p = self.root / INDEX_FILE
        if p.exists():
            return json.loads(p.read_text())
        entries = self.load()
        if entries:
            self._write_index(entries)
        return json.loads(p.read_text()) if p.exists() else []

    def retrieve(self, categories: list[str] | None = None) -> list[KBEntry]:
        entries = self.load()
        if categories is None:
            return entries
        cats = set(categories)
        return [e for e in entries if e.category in cats]

    def lookup(self, ops: list[str], top_k: int | None = None, categories: list[str] | None = None) -> list[KBEntry]:
        """Index-backed retrieval: entries whose op tokens overlap `ops`, best first.

        Scoring = |overlap|; high-confidence entries win ties. Only matching
        entry files are loaded from disk. Returns [] when `ops` is empty or
        nothing overlaps — callers decide their own fallback.
        """
        query = {op_token(o) for o in ops if op_token(o)}
        if not query:
            return []
        cats = set(categories) if categories else None
        scored = []
        for row in self._index():
            if cats and row["category"] not in cats:
                continue
            overlap = query & set(row["ops"])
            if overlap:
                scored.append((-len(overlap), 0 if row.get("confidence") == "high" else 1, row["id"]))
        scored.sort()
        if top_k is not None:
            scored = scored[:top_k]
        return [self.load_one(entry_id) for _, _, entry_id in scored]
