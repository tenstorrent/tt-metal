# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Discover candidate donor models already present in tt-metal.

A donor is any bundled HF config.json for a decoder LLM. Each gets a maturity
tier, because "most similar" is worthless if the nearest match is a 7B model
that was never run on a Galaxy.

Tiers:
  proven    - has an ACTIVE entry under a *galaxy* SKU in models/model_targets.yaml
  in-flight - lives in a bespoke models/demos/<model>/ bring-up dir, no galaxy target yet
  reference - supported elsewhere (tt_transformers model_params); mechanism reference only

Orthogonal to tier: `galaxy_class`, i.e. big enough to need a Galaxy at all.
"""

from __future__ import annotations

import glob
import json
import os
import re

from models.tooling.arch_donor import signature as S

# models/tooling/arch_donor/corpus.py -> repo root is three levels up.
_DEFAULT_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REPO = os.environ.get("TT_METAL_HOME") or _DEFAULT_REPO
GALAXY_CLASS_MIN_B = 50.0  # below this, a model is not a Galaxy sizing donor

SEARCH_GLOBS = [
    "models/demos/*/configs/*/config.json",
    "models/demos/*/reference/config.json",
    "models/demos/*/reference/*/config.json",
    "models/demos/*/model_params/*/config.json",
    "models/tt_transformers/model_params/*/config.json",
]
EXCLUDE = ("vllm_test_utils", "decoder_config")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("instruct", "").replace("-it", ""))


def galaxy_targets() -> set[str]:
    """Normalized names of models with an ACTIVE galaxy SKU entry."""
    path = os.path.join(REPO, "models/model_targets.yaml")
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml ships with the repo venv
        return set()
    with open(path) as f:
        data = yaml.safe_load(f)
    out: set[str] = set()
    for name, spec in (data.get("targets") or {}).items():
        for sku, body in (spec.get("skus") or {}).items():
            if "galaxy" not in sku:
                continue
            if any(e.get("status") == "active" for e in (body.get("entries") or [])):
                out.add(_norm(name))
                for a in spec.get("aliases") or []:
                    out.add(_norm(a.split("/")[-1]))
    return out


def _is_decoder_llm(cfg: dict) -> bool:
    t, _ = S.unwrap(cfg)
    return all(S._get(t, *k) for k in (S.N_LAYERS, S.HIDDEN, S.N_HEADS))


def build_corpus(verbose: bool = False) -> list[S.Signature]:
    gal = galaxy_targets()
    seen: set[str] = set()
    out: list[S.Signature] = []
    for pattern in SEARCH_GLOBS:
        for path in sorted(glob.glob(os.path.join(REPO, pattern))):
            if any(x in path for x in EXCLUDE) or path in seen:
                continue
            seen.add(path)
            try:
                cfg = json.load(open(path))
            except Exception as e:
                if verbose:
                    print(f"  skip {path}: {e}")
                continue
            if not _is_decoder_llm(cfg):
                continue
            name = os.path.basename(os.path.dirname(path))
            if name == "reference":  # models/demos/<model>/reference/config.json
                name = path.split("/models/demos/")[1].split("/")[0]
            sig = S.build(cfg, name=name, source=os.path.relpath(path, REPO))
            # where to go read the TT recipe if this donor wins
            rel = os.path.relpath(path, REPO)
            sig.impl_dir = "/".join(rel.split("/")[:3]) if rel.startswith("models/demos/") else "models/tt_transformers"
            n = _norm(name)
            if any(n == g or n.startswith(g) or g.startswith(n) for g in gal if g):
                sig.tier = "proven"
            elif rel.startswith("models/demos/"):
                sig.tier = "in-flight"
            else:
                sig.tier = "reference"
            sig.galaxy_class = (sig.params.get("total_B") or 0) >= GALAXY_CLASS_MIN_B
            out.append(sig)
    return out


TIER_RANK = {"proven": 0, "in-flight": 1, "reference": 2}

if __name__ == "__main__":
    corpus = build_corpus(verbose=True)
    print(f"\n{len(corpus)} donors\n")
    print(f"{'model':26s} {'tier':10s} {'total_B':>8s} {'glx':>4s}  {'kind':5s} {'mlp':22s} impl")
    for s in sorted(corpus, key=lambda s: (TIER_RANK[s.tier], -(s.params.get("total_B") or 0))):
        mlp = (
            ("MoE %d/%d" % (s.shape["mlp"]["n_experts"], s.shape["mlp"]["top_k"])) if s.mech["mlp"]["moe"] else "dense"
        )
        print(
            f"{s.name:26s} {s.tier:10s} {s.params.get('total_B', 0):8.1f} "
            f"{'yes' if s.galaxy_class else 'no':>4s}  {s.mech['attention']['kind']:5s} "
            f"{mlp + ' ' + s.mech['mlp']['glu']:22s} {s.impl_dir}"
        )
