import re
from typing import List, Optional

LOGFILE = "qwen_accuracy.log"


def clean(tok: Optional[str]) -> Optional[str]:
    if tok is None:
        return None
    tok = tok.strip()
    if tok in {"", "-", "N/A"}:
        return None
    # ignore pure newline tokens
    if tok.replace("\\n", "").strip() == "":
        return None
    return tok


top1 = 0
top5 = 0
total = 0

examples = []

# Matches the part AFTER the logger prefix
line_re = re.compile(
    r"""
    ^\s*
    (?P<idx>\d+/\d+)
    \s+
    (?P<marker>[x!\- ])?
    \s+
    (?P<true>\S+)
    \s+
    (?P<actual>\S+)
    \s+
    (?P<preds>.+)
    $
    """,
    re.VERBOSE,
)

with open(LOGFILE, "r", encoding="utf-8", errors="ignore") as f:
    for raw in f:
        if " - " not in raw:
            continue

        # Strip loguru prefix
        try:
            payload = raw.split(" - ", 1)[1]
        except IndexError:
            continue

        m = line_re.match(payload)
        if not m:
            continue

        true = clean(m.group("true"))
        actual = clean(m.group("actual"))
        preds_blob = m.group("preds")

        if true is None or actual is None or preds_blob is None:
            continue

        # Split predictions by column spacing (2+ spaces)
        preds: List[str] = [p for p in (clean(p) for p in re.split(r"\s{2,}", preds_blob)) if p]

        if not preds:
            continue

        # Sanity: Top-1 printed == first Top-5
        if preds[0] != actual:
            # This should never happen given your test
            continue

        total += 1

        is_top1 = actual == true
        is_top5 = true in preds[:5]

        top1 += int(is_top1)
        top5 += int(is_top5)

        if total <= 5:
            examples.append((true, actual, preds[:5]))

print(f"Valid samples: {total}")
print(f"Top-1 accuracy: {top1 / total:.4%}")
print(f"Top-5 accuracy: {top5 / total:.4%}")

print("\nExamples:")
for gt, act, p in examples:
    print(f"GT={gt!r} | Top1={act!r} | Top5={p}")
