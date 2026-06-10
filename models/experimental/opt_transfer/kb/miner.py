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


# KBEntry field names the extractor must produce; anything else is an alias or noise.
_KB_FIELDS = {f.name for f in __import__("dataclasses").fields(KBEntry)}
# Common aliases the LLM emits for KBEntry fields.
_ALIASES = {"op": "fused_op", "notes": "applicability_notes"}
# Off-schema keys whose content is still useful — folded into applicability_notes.
_FOLD_INTO_NOTES = ("constraints", "fusion_notes", "ttnn_replacement", "torch_pattern_notes", "torch_pattern_source")


def _normalize(d: dict, op: str, idx: int) -> dict:
    """Map the LLM's near-miss output shape onto KBEntry fields.

    The extractor reliably emits {op, torch_pattern, pattern_kind, category,
    config_template, weight_transform, placement_observations, notes?, ...}.
    Rename aliases, synthesize required fields (id/signature/source), fold
    informative extras into applicability_notes, drop the rest. Validation
    (e.g. pattern_kind enum) still happens in KBEntry.from_dict.
    """
    out = {}
    notes_extra = []
    for k, v in d.items():
        k = _ALIASES.get(k, k)
        if k in _KB_FIELDS:
            out[k] = v
        elif k in _FOLD_INTO_NOTES and v:
            notes_extra.append(f"{k}: {v}")
    base = re.sub(r"\W+", "_", str(out.get("fused_op") or op)).strip("_").lower()
    out.setdefault("id", base if idx == 0 else f"{base}_{idx}")
    out.setdefault("fused_op", op)
    out.setdefault("signature", {})
    out.setdefault("source", "mined")
    out.setdefault("category", "uncategorized")
    if out.get("torch_pattern") is None:
        out["torch_pattern"] = []
    if notes_extra:
        out["applicability_notes"] = "; ".join(filter(None, [out.get("applicability_notes", ""), *notes_extra]))
    return out


def build_kb(client, cache_root=None, kb_root=None, config=CONFIG, limit_ops=None) -> list[KBEntry]:
    cache = ContentCache(cache_root or config.cache_dir)
    store = KBStore(kb_root or config.kb_dir)
    inv = inventory_ops(config)
    usage = scan_usage(config)
    ops = sorted(set(inv) | set(usage))  # union: available + used
    if limit_ops:
        ops = ops[:limit_ops]
    entries: dict[str, KBEntry] = {}
    from concurrent.futures import ThreadPoolExecutor

    from models.experimental.opt_transfer.matcher import EXTRACT_SYSTEM

    def _extract(op):
        available = inv.get(op, {"tests": [], "examples": []})
        used = usage.get(op, [])
        golden_src = _golden_source(op)
        # Prompt text participates in the cache key so prompt fixes invalidate stale outputs.
        content = (
            repr(EXTRACT_SYSTEM) + repr(golden_src) + repr(available["examples"]) + repr([u["snippet"] for u in used])
        )
        raw = cache.get_or_compute(
            key=f"op::{op}",
            content=content,
            compute=lambda: client.extract_entries(op, available, used, golden_src),
        )
        return op, available, used, golden_src, raw

    # Extractions are independent LLM calls — fan out; cache writes are per-file.
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_extract, ops))

    skipped = 0
    for op, available, used, golden_src, raw in results:
        for i, d in enumerate(raw):
            try:
                e = KBEntry.from_dict(_normalize(dict(d), op, i))
            except (ValueError, KeyError, TypeError) as err:
                # LLM output that doesn't fit the schema (e.g. pattern_kind outside the
                # enum for non-fusable symbols) is noise, not a reason to abort the mine.
                skipped += 1
                print(f"KB_MINE_SKIP op={op} err={err}", flush=True)
                continue
            if not e.unit_test_refs:
                e.unit_test_refs = list(available["tests"])
            e.status = "in_use" if used else "supported_unused"
            if e.pattern_source == "unit_test":
                e.pattern_source = "golden" if golden_src else ("unit_test" if available["examples"] else "llm")
                e.confidence = "low" if e.pattern_source == "llm" else "high"
            entries[e.id] = e
    if skipped:
        print(f"KB_MINE_SKIPPED_TOTAL {skipped}", flush=True)
    out = list(entries.values())
    store.save(out)
    return out
