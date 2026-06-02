import re
from pathlib import Path

# Matches ANY ttnn op call: ttnn.foo( or ttnn.experimental.bar( or ttnn.transformer.bar( — no hardcoded allowlist.
OP_CALL_RE = re.compile(r"ttnn\.(?:experimental\.|transformer\.)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

# canonical fused-op tests live in BOTH places
TEST_ROOTS = (
    "tests/ttnn/unit_tests/operations",
    "tests/tt_eager/python_api_testing/unit_testing/misc",
)


def _iter_py(base: Path):
    for p in base.rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        try:
            yield p, p.read_text()
        except (UnicodeDecodeError, OSError):
            continue


def _scan_calls(text: str):
    """Yield (op_name, context_snippet) for every ttnn op call in text."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        for m in OP_CALL_RE.finditer(line):
            lo, hi = max(0, i - 4), min(len(lines), i + 10)
            yield m.group(1), "".join(lines[lo:hi])


def inventory_ops(config) -> dict:
    """Available/supported set: every ttnn op exercised by the unit tests, with test
    provenance + example call snippets. The op's test path(s) become unit_test_refs."""
    inv: dict[str, dict] = {}
    for root in TEST_ROOTS:
        base = config.repo_root / root
        for p, text in _iter_py(base):
            rel = str(p.relative_to(config.repo_root))
            for op, snippet in _scan_calls(text):
                e = inv.setdefault(op, {"tests": set(), "examples": []})
                e["tests"].add(rel)
                if len(e["examples"]) < 5:
                    e["examples"].append(snippet)
    for op in inv:
        inv[op]["tests"] = sorted(inv[op]["tests"])
    return inv


USAGE_ROOTS = ("models/tt_transformers", "models/tt_dit", "models/demos")


def scan_usage(config) -> dict:
    """Used set: every ttnn op call in the model source roots, with config context
    + provenance."""
    usage: dict[str, list] = {}
    for root in USAGE_ROOTS:
        base = config.repo_root / root
        for p, text in _iter_py(base):
            rel = str(p.relative_to(config.repo_root))
            for op, snippet in _scan_calls(text):
                usage.setdefault(op, []).append({"source": rel, "snippet": snippet})
    return usage


from models.experimental.opt_transfer.kb.cache import ContentCache
from models.experimental.opt_transfer.kb.store import KBStore
from models.experimental.opt_transfer.schema import KBEntry
from models.experimental.opt_transfer.config import CONFIG


def _golden_source(op_name: str):
    """Tier-1 opportunity source: the op's registered torch golden, or None. Lazy ttnn
    import (device-free); any failure (no op / no golden / RuntimeError) -> None."""
    import inspect

    try:
        import ttnn

        op = (
            getattr(ttnn, op_name, None)
            or getattr(getattr(ttnn, "experimental", None), op_name, None)
            or getattr(getattr(ttnn, "transformer", None), op_name, None)
        )
        if op is None:
            return None
        g = ttnn.get_golden_function(op)
        return inspect.getsource(g) if g is not None else None
    except Exception:
        return None


def build_kb(client, cache_root=None, kb_root=None, config=CONFIG, limit_ops=None) -> list[KBEntry]:
    cache = ContentCache(cache_root or config.cache_dir)
    store = KBStore(kb_root or config.kb_dir)
    inv = inventory_ops(config)
    usage = scan_usage(config)
    ops = sorted(set(inv) | set(usage))  # union: available + used
    if limit_ops:
        ops = ops[:limit_ops]
    entries: dict[str, KBEntry] = {}
    for op in ops:
        available = inv.get(op, {"tests": [], "examples": []})
        used = usage.get(op, [])
        golden_src = _golden_source(op)
        content = repr(golden_src) + repr(available["examples"]) + repr([u["snippet"] for u in used])
        raw = cache.get_or_compute(
            key=f"op::{op}",
            content=content,
            compute=lambda op=op, a=available, u=used, g=golden_src: client.extract_entries(op, a, u, g),
        )
        for d in raw:
            e = KBEntry.from_dict(d)
            if not e.unit_test_refs:
                e.unit_test_refs = list(available["tests"])
            e.status = "in_use" if used else "supported_unused"
            if e.pattern_source == "unit_test":
                e.pattern_source = "golden" if golden_src else ("unit_test" if available["examples"] else "llm")
                e.confidence = "low" if e.pattern_source == "llm" else "high"
            entries[e.id] = e
    out = list(entries.values())
    store.save(out)
    return out
