# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Model spec: the one file that says what is being brought up, on which box, and where everything lives.

A spec is YAML (see templates/spec.yaml). Only ``model`` is needed to run gates. ``validate()`` checks
the full schema that the pipeline steps rely on.

Paths:
    repo            git checkout that gates run in and commit to (default: this checkout)
    bringup_dir     ledger dir in the repo: tasks.yaml, state.json, results/, BREADCRUMBS.md (default <model_dir>/bringup)
    art             artifact root outside git (default /localdev/$USER/bringup/<model>)
      hf/ golden/ tt_cache/<mesh>/<version>/ profiles/ runs/<run>/
"""

from __future__ import annotations

import getpass
import importlib
import os
from pathlib import Path

import yaml

FRAMEWORK = Path(__file__).resolve().parents[1]
CODE_ROOT = FRAMEWORK.parents[3]

REQUIRED = {
    "model": str,
    "hf_id": str,
    "model_dir": str,
    "hooks": str,
    "box": dict,
    "target": dict,
    "ladder": list,
    "block_types": dict,
    "state": dict,
}


class SpecError(ValueError):
    pass


def _expand(v):
    if isinstance(v, str):
        return os.path.expandvars(os.path.expanduser(v))
    return v


def parse_layers(v, num_layers: int | None = None) -> list[int]:
    """[0, 1, "3-5"] or "0,1,3-5" -> [0, 1, 3, 4, 5]. "all" needs num_layers."""
    if v in (None, "all"):
        if num_layers is None:
            raise SpecError("layer list 'all' needs num_layers")
        return list(range(num_layers))
    items = v.split(",") if isinstance(v, str) else v
    out = []
    for it in items:
        if isinstance(it, int):
            out.append(it)
            continue
        it = str(it).strip()
        if "-" in it:
            a, b = it.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(it))
    return sorted(set(out))


class Spec:
    def __init__(self, data: dict, path: Path | None = None):
        if "model" not in data:
            raise SpecError("spec has no 'model'")
        self.data = data
        self.path = Path(path).resolve() if path else None

    @classmethod
    def load(cls, path) -> "Spec":
        path = Path(path)
        return cls(yaml.safe_load(path.read_text()) or {}, path)

    def get(self, key: str, default=None):
        cur = self.data
        for k in key.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    # ---- identity and paths
    @property
    def model(self) -> str:
        return self.data["model"]

    @property
    def tag(self) -> str:
        """Commit-message tag. Gate commits are '[<tag>][<task id>] <title>'."""
        return self.data.get("tag", self.model)

    @property
    def repo(self) -> Path:
        return Path(_expand(self.get("paths.repo") or CODE_ROOT)).resolve()

    @property
    def model_dir(self) -> Path:
        return self.repo / self.data.get("model_dir", f"models/demos/{self.model}")

    @property
    def bringup_dir(self) -> Path:
        d = self.get("paths.bringup_dir")
        return (self.repo / d) if d else self.model_dir / "bringup"

    @property
    def art(self) -> Path:
        root = self.get("paths.art") or f"/localdev/{getpass.getuser()}/bringup"
        return Path(_expand(root)) / self.model

    def _art_sub(self, key: str, default: str) -> Path:
        v = self.get(f"paths.{key}")
        return Path(_expand(v)) if v else self.art / default

    @property
    def hf_dir(self) -> Path:
        return self._art_sub("hf", "hf")

    @property
    def golden_root(self) -> Path:
        return self._art_sub("golden", "golden")

    @property
    def profiles_dir(self) -> Path:
        return self._art_sub("profiles", "profiles")

    def tt_cache(self, variant: str = "") -> Path:
        """Weight cache dir. The .tensorbin name only holds name/dtype/layout, so the path carries mesh and version."""
        base = self._art_sub("tt_cache", "tt_cache")
        mesh = "x".join(str(x) for x in self.mesh)
        version = str(self.get("cache_version", "v1"))
        return base / mesh / (f"{version}-{variant}" if variant else version)

    def run_dir(self, run: str) -> Path:
        return self.art / "runs" / run

    # ---- model facts
    @property
    def mesh(self) -> list[int]:
        return list(self.get("box.mesh", [1, 1]))

    @property
    def num_layers(self) -> int | None:
        return self.get("num_layers")

    def layers(self) -> list[int]:
        """Layers brought up on device: the subset if given, else all."""
        return parse_layers(self.data.get("layers", "all"), self.num_layers)

    def block_type_of(self, layer: int) -> str:
        for bt, info in self.data.get("block_types", {}).items():
            if layer in parse_layers(info["layers"], self.num_layers):
                return bt
        raise SpecError(f"layer {layer} has no block type")

    def representative_layer(self, block_type: str) -> int:
        info = self.data["block_types"][block_type]
        if "representative" in info:
            return int(info["representative"])
        sel = set(self.layers())
        return next(i for i in parse_layers(info["layers"], self.num_layers) if i in sel)

    def rung(self, name: str) -> dict:
        for r in self.data.get("ladder", []):
            if r["name"] == name:
                return r
        raise SpecError(f"no ladder rung {name!r}")

    def threshold(self, key: str, default: float) -> float:
        return float(self.get(f"thresholds.{key}", default))

    def hooks(self):
        return importlib.import_module(self.data["hooks"])

    # ---- validation
    def validate(self) -> list[str]:
        """Schema errors for a model spec (empty list = valid)."""
        errs = []
        for k, t in REQUIRED.items():
            if k not in self.data:
                errs.append(f"missing '{k}'")
            elif not isinstance(self.data[k], t):
                errs.append(f"'{k}' must be a {t.__name__}")
        if errs:
            return errs
        if not isinstance(self.num_layers, int):
            errs.append("missing 'num_layers' (int)")
            return errs
        mesh = self.get("box.mesh")
        if not (isinstance(mesh, list) and len(mesh) == 2 and all(isinstance(x, int) and x > 0 for x in mesh)):
            errs.append("box.mesh must be [rows, cols]")
        for k in ("seq", "chunk"):
            if not isinstance(self.get(f"target.{k}"), int):
                errs.append(f"target.{k} must be an int")
        names = set()
        for r in self.data["ladder"]:
            for k in ("name", "seq", "chunk"):
                if k not in r:
                    errs.append(f"ladder rung {r} misses '{k}'")
            if "seq" in r and "chunk" in r and r["seq"] % r["chunk"]:
                errs.append(f"ladder rung {r['name']}: seq is not a multiple of chunk")
            if r.get("name") in names:
                errs.append(f"duplicate ladder rung {r['name']}")
            names.add(r.get("name"))
            if r.get("golden") and r["golden"] not in names:
                errs.append(f"ladder rung {r['name']}: golden rung {r['golden']} must come earlier")
        covered = []
        for bt, info in self.data["block_types"].items():
            if "layers" not in info:
                errs.append(f"block type {bt} has no 'layers'")
                continue
            covered += parse_layers(info["layers"], self.num_layers)
        if sorted(covered) != list(range(self.num_layers)):
            errs.append("block_types must cover every layer exactly once")
        try:
            sel = self.layers()
            if not sel or min(sel) < 0 or max(sel) >= self.num_layers:
                errs.append("layers subset out of range")
        except (SpecError, ValueError) as e:
            errs.append(str(e))
        if not errs:
            for bt in self.data["block_types"]:
                try:
                    self.representative_layer(bt)
                except StopIteration:
                    errs.append(f"block type {bt} has no layer in the selected subset")
        if not self.get("state.tensors"):
            errs.append("state.tensors must list the per-layer state tensor names (e.g. [key, value])")
        return errs
