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
