"""Self-updating sibling map.

When `up --auto` successfully brings a brand-new model up on
tt-metal (either via the per-component iterate loop or the cold-start
`prepare --execute` path), this module writes the newly-learned
mapping back into the tool's own source so the NEXT similar model
gets an `exact` backend match without going through inline auto-
onboard again.

Three writes happen per successful bring-up:

1. ``family_backends.py``: the matched backend's ``model_type_keys``
   is extended to include the new ``model_type`` -- so the next time
   the same arch shows up, ``pick_backend_with_quality`` returns
   ``"exact"`` instead of ``"category-default"``.

2. ``compatibility.py``: ``closest_supported_model()``'s candidates
   dict is extended with ``new_model_type -> new_model_id`` so the
   scaffold step finds a sibling immediately instead of falling
   through to ``ColdStartScaffoldError``.

3. ``learned_bringups.json`` (next to this file): a history log of
   every learned bring-up, with timestamps + PCC details + the
   commit-style "what changed" diff. Auditable, replay-friendly,
   never read back by the tool itself -- it's purely a record.

All three writes are best-effort: if any of them fails (file locked,
permission denied, repo-state changed), we log the failure and
proceed. A bring-up that succeeded but couldn't be persisted into
the registry is still a successful bring-up; the user just doesn't
get the next-time-it's-faster benefit.

Concurrency:
- The functions here are NOT safe across concurrent writers. The
  expected caller is a single ``up --auto`` invocation per checkout.
- We use a lockfile next to ``learned_bringups.json`` to prevent two
  parallel ``up --auto`` runs from clobbering each other.

Idempotency:
- All three writes are idempotent. Re-running a successful bring-up
  for the same model is a no-op (the existing key / dict entry is
  detected and the function returns early).
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_BACKENDS_FILE = _THIS_DIR / "family_backends.py"
_COMPAT_FILE = _THIS_DIR / "compatibility.py"
_LEARNING_LOG = _THIS_DIR / "learned_bringups.json"
_LEARNING_LOCK = _THIS_DIR / ".learned_bringups.lock"


@dataclass
class LearnedBringup:
    """A single audit record of a successful bring-up."""

    model_id: str
    model_type: str
    category: str
    backend_name: str
    path: str
    timestamp: float
    notes: str = ""
    diffs: List[Dict[str, str]] = field(default_factory=list)


@contextmanager
def _filelock(lock_path: Path, timeout_s: float = 30.0):
    """Naive file-based lock. Good enough for the
    single-host-single-checkout case this module is designed for; not
    intended for cross-host concurrency."""
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"could not acquire learning lock at {lock_path} " f"after {timeout_s:.0f}s")
            time.sleep(0.1)
    try:
        yield
    finally:
        try:
            os.unlink(str(lock_path))
        except FileNotFoundError:
            pass


def _normalize_model_type(mt: str) -> str:
    """Lowercase + strip; HF model_type strings can have inconsistent
    casing across configs. Match the convention used elsewhere in
    the registry."""
    return (mt or "").strip().lower()


def _extend_backend_model_type_keys(
    *,
    backend_name: str,
    new_model_type: str,
    backends_file: Path = _BACKENDS_FILE,
) -> Tuple[bool, str]:
    """Find the ``FamilyBackend(name=<backend_name>, ...)`` block in
    ``backends_file`` and append ``new_model_type`` to its
    ``model_type_keys=[...]`` list (creating the field if missing).

    Returns ``(ok, message)``. Idempotent: if the key is already
    present, returns ``(True, "(no-op: ...)")``.
    """
    mt = _normalize_model_type(new_model_type)
    if not mt:
        return (False, "new_model_type is empty / falsy")
    if not backends_file.is_file():
        return (False, f"{backends_file} does not exist")
    text = backends_file.read_text()

    block_start_re = re.compile(
        r"FamilyBackend\s*\(\s*\n" r"(?:[^()]*?\bname\s*=\s*[\"']" + re.escape(backend_name) + r"[\"'])",
        re.DOTALL,
    )
    m = block_start_re.search(text)
    if m is None:
        return (False, f"could not find FamilyBackend(name={backend_name!r}) block")
    block_open = m.start()

    depth = 0
    block_close = -1
    for i in range(block_open, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                block_close = i
                break
    if block_close < 0:
        return (False, "could not find matching close-paren for block")
    block = text[block_open : block_close + 1]

    keys_re = re.compile(r"model_type_keys\s*=\s*\[([^\]]*)\]", re.DOTALL)
    km = keys_re.search(block)
    if km is None:
        insertion = f"        model_type_keys=[{mt!r}],  # auto-learned\n    "
        new_block = block[:-1] + insertion + ")"
        new_text = text[:block_open] + new_block + text[block_close + 1 :]
        backends_file.write_text(new_text)
        return (
            True,
            f"added model_type_keys=[{mt!r}] to backend {backend_name!r}",
        )

    inner = km.group(1)
    existing = re.findall(r"[\"']([^\"']+)[\"']", inner)
    if mt in {k.lower() for k in existing}:
        return (True, f"(no-op: {mt!r} already in {backend_name!r}.model_type_keys)")

    if existing:
        new_inner = inner.rstrip()
        if not new_inner.endswith(","):
            new_inner = new_inner + ","
        new_inner = new_inner + f" {mt!r}"
    else:
        new_inner = f"{mt!r}"
    new_block = block[: km.start()] + f"model_type_keys=[{new_inner}]" + block[km.end() :]
    new_text = text[:block_open] + new_block + text[block_close + 1 :]
    backends_file.write_text(new_text)
    return (
        True,
        f"appended {mt!r} to backend {backend_name!r}.model_type_keys",
    )


def _add_to_closest_supported_model_map(
    *,
    new_model_type: str,
    sibling_model_id: str,
    compat_file: Path = _COMPAT_FILE,
) -> Tuple[bool, str]:
    """Find the ``candidates = { ... }`` dict in
    ``closest_supported_model`` in ``compat_file`` and inject
    ``"<new_model_type>": "<sibling_model_id>"``.

    Returns ``(ok, message)``. Idempotent.
    """
    mt = _normalize_model_type(new_model_type)
    if not mt:
        return (False, "new_model_type is empty / falsy")
    if not sibling_model_id:
        return (False, "sibling_model_id is empty / falsy")
    if not compat_file.is_file():
        return (False, f"{compat_file} does not exist")
    text = compat_file.read_text()

    fn_idx = text.find("def closest_supported_model(")
    if fn_idx < 0:
        return (False, "could not find closest_supported_model()")
    open_idx = text.find("candidates = {", fn_idx)
    if open_idx < 0:
        return (False, "could not find `candidates = {` in closest_supported_model")

    depth = 0
    close_idx = -1
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    if close_idx < 0:
        return (False, "could not find matching `}` for candidates dict")
    dict_body = text[open_idx + len("candidates = {") : close_idx]

    key_re = re.compile(r'^\s*"' + re.escape(mt) + r'"\s*:', re.MULTILINE)
    if key_re.search(dict_body):
        return (True, f"(no-op: {mt!r} already in closest_supported_model)")

    insertion = f'        "{mt}": "{sibling_model_id}",  # auto-learned\n    '
    new_text = text[:close_idx] + insertion + text[close_idx:]
    compat_file.write_text(new_text)
    return (
        True,
        f"added {mt!r} -> {sibling_model_id!r} to closest_supported_model()",
    )


def _append_to_learning_log(
    entry: LearnedBringup,
    *,
    log_file: Path = _LEARNING_LOG,
) -> Tuple[bool, str]:
    """Append `entry` to the learning log JSON file. Creates the file
    if missing. Idempotent: if an entry for the same `model_id`
    already exists, skip."""
    data: List[Dict[str, Any]] = []
    if log_file.is_file():
        try:
            data = json.loads(log_file.read_text())
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:
            data = []
    for existing in data:
        if isinstance(existing, dict) and existing.get("model_id") == entry.model_id:
            return (True, f"(no-op: {entry.model_id!r} already logged)")
    data.append(asdict(entry))
    log_file.write_text(json.dumps(data, indent=2))
    return (True, f"logged successful bring-up of {entry.model_id!r}")


def register_successful_bringup(
    *,
    model_id: str,
    model_type: str,
    category: str,
    backend_name: str,
    sibling_model_id: Optional[str] = None,
    path: str = "(unspecified)",
    notes: str = "",
) -> List[str]:
    """Persist a learned successful bring-up. Returns a list of
    human-readable messages describing what changed. Never raises;
    individual write failures are appended as "FAIL: ..." messages
    and logged but do not propagate.

    Arguments:
      - ``model_id``: the new model brought up successfully (e.g.
        ``"openai/whisper-large-v3"``).
      - ``model_type``: HF ``config.model_type`` (e.g. ``"whisper"``).
      - ``category``: probe-classified category (e.g. ``"STT"``).
      - ``backend_name``: the FamilyBackend.name that was used to
        bring this model up (e.g. ``"Whisper (distil-large-v3)"``).
      - ``sibling_model_id``: optional canonical sibling HF id to
        register for next-time lookups; defaults to ``model_id``
        itself (i.e. THIS bring-up becomes the new sibling).
      - ``path``: which Pattern was used ("A. Template + iterate" /
        "B. Generic cold-start" / etc). Audit only.
      - ``notes``: free-form audit text.
    """
    msgs: List[str] = []
    sibling = sibling_model_id or model_id
    entry = LearnedBringup(
        model_id=model_id,
        model_type=_normalize_model_type(model_type),
        category=category,
        backend_name=backend_name,
        path=path,
        timestamp=time.time(),
        notes=notes,
    )

    try:
        with _filelock(_LEARNING_LOCK):
            try:
                ok, msg = _extend_backend_model_type_keys(
                    backend_name=backend_name,
                    new_model_type=model_type,
                )
                msgs.append(("OK   " if ok else "FAIL ") + msg)
                if ok:
                    entry.diffs.append({"file": "family_backends.py", "change": msg})
            except Exception as exc:
                msgs.append(f"FAIL extending backend keys " f"({type(exc).__name__}: {exc})")

            try:
                ok, msg = _add_to_closest_supported_model_map(
                    new_model_type=model_type,
                    sibling_model_id=sibling,
                )
                msgs.append(("OK   " if ok else "FAIL ") + msg)
                if ok:
                    entry.diffs.append({"file": "compatibility.py", "change": msg})
            except Exception as exc:
                msgs.append(f"FAIL adding to closest_supported_model " f"({type(exc).__name__}: {exc})")

            try:
                ok, msg = _append_to_learning_log(entry)
                msgs.append(("OK   " if ok else "FAIL ") + msg)
            except Exception as exc:
                msgs.append(f"FAIL appending to learning log " f"({type(exc).__name__}: {exc})")

    except TimeoutError as exc:
        msgs.append(f"FAIL learning-lock timeout: {exc}")
    except Exception as exc:
        msgs.append(f"FAIL unexpected error in register_successful_bringup " f"({type(exc).__name__}: {exc})")

    return msgs
