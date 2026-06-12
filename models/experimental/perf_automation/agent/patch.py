"""Deterministic content-anchored patch apply (PLAN -> APPLY).

PLAN emits edits as {file, old_string, new_string}. We apply them with a plain,
SELF-VALIDATING string replace: `old_string` must occur EXACTLY ONCE in the file
(a stale or ambiguous anchor fails loudly instead of silently corrupting code),
and nothing is keyed on line numbers (which drift the moment anything above moves).
No LLM, no file-reading by an editor agent.

All-or-nothing: every edit is computed against in-memory text first; if ANY edit
fails to match, NOTHING is written and the failures are returned so APPLY can route
to REPAIR_CODE (the self-heal editor reads the real file and fixes it).
"""

from __future__ import annotations

from pathlib import Path


def apply_edits(model_root, edits: list[dict]) -> tuple[list[str], list[dict]]:
    """Apply [{file, old_string, new_string}] edits. -> (changed_files, failures).

    `new_string` "" deletes the matched text. Edits to the same file apply in order
    against its evolving text. On any failure, returns ([], [failure, ...]) and writes
    nothing.
    """
    by_file: dict[str, list[dict]] = {}
    for e in edits:
        by_file.setdefault(e.get("file", ""), []).append(e)

    new_texts: dict[str, str] = {}
    failures: list[dict] = []
    for rel, file_edits in by_file.items():
        if not rel:
            failures.append({"file": rel, "reason": "edit missing 'file'"})
            continue
        p = Path(model_root) / rel
        if not p.exists():
            failures.append({"file": rel, "reason": "file not found"})
            continue
        text = p.read_text()
        for e in file_edits:
            old = e.get("old_string") or ""
            new = e.get("new_string") or ""
            if not old:
                failures.append({"file": rel, "reason": "empty old_string anchor"})
                break
            n = text.count(old)
            if n != 1:
                where = "not found" if n == 0 else f"matched {n}x (not unique)"
                failures.append({"file": rel, "reason": f"old_string {where}: {old[:60]!r}"})
                break
            text = text.replace(old, new, 1)
        else:
            new_texts[rel] = text  # every edit for this file matched uniquely

    if failures:  # all-or-nothing: don't write a partial patch
        return [], failures
    for rel, text in new_texts.items():
        (Path(model_root) / rel).write_text(text)
    return sorted(new_texts), []
