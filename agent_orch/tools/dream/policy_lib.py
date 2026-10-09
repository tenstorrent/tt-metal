"""The policy library: named, exported exploration policies a new campaign can start from.

    agent_orch/policies/library/<name>/policy.py   the Policy class (dream.policy_api interface)
    agent_orch/policies/library/<name>/meta.json   name, description, origin (campaign, version, replay V), caution

Policies are about where to spend attempts, not about any one op, so a policy learned on one
campaign is a reasonable start for another. `dream policies` lists them; `dream export-policy`
adds a campaign's dreamed version to the library.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .campaign import LIBRARY_DIR

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def list_policies(lib: Path = LIBRARY_DIR) -> list[dict]:
    out = []
    for d in sorted(p for p in lib.glob("*") if (p / "policy.py").exists()):
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        out.append(
            {
                "name": d.name,
                "description": meta.get("description", ""),
                "origin": meta.get("origin", {}),
                "aliases": meta.get("aliases", []),
                "caution": meta.get("caution", ""),
                "path": str(d),
            }
        )
    return out


def resolve(name: str | None, lib: Path = LIBRARY_DIR) -> Path:
    name = name or "fresh"
    for p in list_policies(lib):
        if p["name"] == name or name in p["aliases"]:
            return Path(p["path"])
    raise FileNotFoundError(f"no policy '{name}' in {lib}; see `dream policies`")


def write_policy(name: str, policy_py: str, meta: dict, lib: Path = LIBRARY_DIR, force: bool = False) -> Path:
    if not NAME_RE.match(name):
        raise ValueError(f"policy name '{name}': use lowercase letters, digits, '.', '_' and '-'")
    d = lib / name
    if d.exists() and not force:
        raise FileExistsError(f"{d} exists; pick another name or pass --force")
    d.mkdir(parents=True, exist_ok=True)
    (d / "policy.py").write_text(policy_py)
    (d / "meta.json").write_text(json.dumps({"name": name, **meta}, indent=2) + "\n")
    return d
