# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The reduced checkpoint of a scaled run (F64): the model with fewer heads and experts, so that each chip of a smaller
box holds what a chip of the deployment box holds.

    python -m models.demos.common.bringup.plan.scale --spec S

A partial bring-up on a box smaller than the deployment box runs twice: at real size (accuracy) and scaled (the
deployment's per-chip shapes and code paths, and per-chip performance that carries over). The scale factor of a role
is the box's chips for that role over the deployment box's (plan/boxes.py, the plan's ``roles``). ``scaling.yaml``
next to the plan (written with the plan; not part of the approved files, so the scale step may fix it) says how to cut:

    counts:                            # config key (dotted for nested, e.g. linear_attn_config.num_heads) -> role
      num_attention_heads: heads
      n_routed_experts: experts
    tensors:                           # first match wins; tensors no rule matches are copied unchanged
      - {pattern: "model.layers.*.self_attn.q_proj.*", count: num_attention_heads, dim: 0}
      - {pattern: "model.layers.*.mlp.experts.*", count: n_routed_experts, index: "experts\\.(\\d+)\\."}
      - {pattern: "model.layers.*.mlp.gate.*", count: n_routed_experts, dim: 0}

A ``dim`` rule keeps the first kept/count of the tensor along that dimension (the dimension must hold whole units of
size / count, e.g. one head's rows, or one 128-row block scale per head pair); an ``index`` rule keeps the per-unit
tensors whose index (the regex group) is below the kept count. Every count key is scaled in config.json (also under
``text_config``). The rest of the checkpoint dir (tokenizer, generation config, modeling code) is copied.

Writes ``<art>/hf_scaled/<run box>/`` (spec.scaled_hf_dir) from the real checkpoint (spec.real_hf_dir), with
``bringup_scale.json`` (factors, kept counts, per-rule tensor counts). Idempotent: an unchanged plan and source is not
rewritten. Records scale_errors, scale_tensors_cut, scale_tensors_dropped, scale_tensors_total.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shutil
from pathlib import Path

import yaml

from models.demos.common.bringup.plan.boxes import factors
from models.demos.common.bringup.plan.memory import load_plan

MARKER = "bringup_scale.json"
INDEX = "model.safetensors.index.json"
SKIP_FILES = {INDEX, "model.safetensors.index.full.json", "bringup_trim.json", MARKER}


def _cfg_get(cfg: dict, key: str):
    """A dotted config key, at the top level or under text_config; (holder, last key) or (None, None)."""
    for root in (cfg, cfg.get("text_config")):
        cur, parts = root, key.split(".")
        for k in parts[:-1]:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if isinstance(cur, dict) and parts[-1] in cur:
            return cur, parts[-1]
    return None, None


def load_scaling(spec) -> dict:
    p = spec.bringup_dir / "scaling.yaml"
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def kept_counts(spec, plan: dict, cfg: dict, scaling: dict) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """Per count key: (original, kept), from the run box's factor of its role."""
    f = factors(spec, plan, spec.run_box)
    out, errs = {}, []
    for key, role in (scaling.get("counts") or {}).items():
        holder, k = _cfg_get(cfg, key)
        if holder is None:
            errs.append(f"counts: config has no {key!r}")
            continue
        if role not in f:
            errs.append(f"counts.{key}: {role!r} is not one of the plan's roles {sorted(f)}")
            continue
        n = int(holder[k])
        kept = n * f[role]
        if kept != int(kept) or kept < 1:
            errs.append(f"counts.{key}: {n} x {f[role]:g} ({role}) is not a whole count")
            continue
        out[key] = (n, int(kept))
    return out, errs


def plan_cut(names_shapes: dict[str, list[int]], rules: list[dict], counts: dict) -> tuple[dict, list[str]]:
    """name -> ("keep",) | ("drop",) | ("slice", dim, length), and errors."""
    out, errs, used = {}, [], [0] * len(rules)
    for name, shape in names_shapes.items():
        k = next((i for i, r in enumerate(rules) if fnmatch.fnmatchcase(name, r["pattern"])), None)
        if k is None:
            out[name] = ("keep",)
            continue
        used[k] += 1
        r = rules[k]
        if r.get("count") not in counts:
            errs.append(f"rule {r['pattern']!r}: count {r.get('count')!r} is not in counts")
            continue
        n, kept = counts[r["count"]]
        if "index" in r:
            m = re.search(r["index"], name)
            if not m:
                errs.append(f"rule {r['pattern']!r}: {name} has no index for {r['index']!r}")
                continue
            out[name] = ("keep",) if int(m.group(1)) < kept else ("drop",)
            continue
        d = int(r.get("dim", 0))
        if d >= len(shape) or shape[d] % n:
            errs.append(f"rule {r['pattern']!r}: {name} {shape} dim {d} is not whole units of {n}")
            continue
        out[name] = ("slice", d, shape[d] // n * kept) if kept < n else ("keep",)
    for i, r in enumerate(rules):
        if not used[i]:
            errs.append(f"scaling rule {r['pattern']!r} matches no checkpoint tensor")
    return out, errs


def _headers(hf: Path) -> tuple[dict[str, str], dict[str, list[int]]]:
    """name -> shard, and name -> shape, from the safetensors headers."""
    from safetensors import safe_open

    idx = hf / INDEX
    shards = (
        sorted(set(json.loads(idx.read_text())["weight_map"].values()))
        if idx.exists()
        else sorted(p.name for p in hf.glob("*.safetensors"))
    )
    wm, shapes = {}, {}
    for f in shards:
        with safe_open(str(hf / f), framework="pt") as h:
            for n in h.keys():
                wm[n] = f
                shapes[n] = list(h.get_slice(n).get_shape())
    return wm, shapes


def scale(spec, plan: dict | None = None, sc: dict | None = None) -> dict:
    plan = plan if plan is not None else load_plan(spec)
    sc = sc if sc is not None else load_scaling(spec)
    src, dst = Path(spec.real_hf_dir), Path(spec.scaled_hf_dir)
    cfg = json.loads((src / "config.json").read_text())
    counts, errs = kept_counts(spec, plan, cfg, sc)
    if not sc.get("counts"):
        errs.append("scaling.yaml has no counts (which config counts the scaled run cuts)")
    wm, shapes = _headers(src)
    cut, e2 = plan_cut(shapes, list(sc.get("tensors") or []), counts)
    errs += e2
    res = {
        "factors": factors(spec, plan, spec.run_box),
        "counts": {k: list(v) for k, v in counts.items()},
        "errors": errs,
        "tensors_total": len(shapes),
        "tensors_cut": sum(v[0] == "slice" for v in cut.values()),
        "tensors_dropped": sum(v[0] == "drop" for v in cut.values()),
    }
    if errs:
        return res
    key = hashlib.sha256(
        json.dumps([sc, res["factors"], sorted(wm.items()), (src / "config.json").read_text()]).encode()
    ).hexdigest()
    marker = dst / MARKER
    if marker.exists() and json.loads(marker.read_text()).get("key") == key:
        return {**res, "reused": True}
    from safetensors import safe_open
    from safetensors.torch import save_file

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    new_wm = {}
    for shard in sorted(set(wm.values())):
        tensors = {}
        with safe_open(str(src / shard), framework="pt") as h:
            for n in h.keys():
                c = cut.get(n, ("keep",))
                if c[0] == "drop":
                    continue
                t = h.get_tensor(n)
                if c[0] == "slice":
                    t = t.narrow(c[1], 0, c[2]).contiguous()
                tensors[n] = t
        if tensors:
            save_file(tensors, str(dst / shard), metadata={"format": "pt"})
            new_wm.update({n: shard for n in tensors})
    (dst / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": dict(sorted(new_wm.items()))}, indent=2))
    for key_, (_, kept) in counts.items():
        holder, k = _cfg_get(cfg, key_)
        holder[k] = kept
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for p in src.iterdir():
        if p.is_file() and p.suffix != ".safetensors" and p.name not in SKIP_FILES and p.name != "config.json":
            shutil.copy2(p, dst / p.name)
    marker.write_text(json.dumps({**res, "key": key, "source": str(src)}, indent=1) + "\n")
    return res


def main(argv=None):
    from models.demos.common.bringup.core import metrics
    from models.demos.common.bringup.reference.golden import load_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    if not spec.has_deploy or spec.mode != "partial":
        raise SystemExit("a scaled run is for a partial bring-up with a deployment box in the spec")
    res = scale(spec)
    for e in res["errors"]:
        print(f"SCALE ERROR {e}")
    print(json.dumps({k: v for k, v in res.items() if k != "errors"}))
    for k in ("tensors_cut", "tensors_dropped", "tensors_total"):
        metrics.record(f"scale_{k}", res[k])
    metrics.record("scale_errors", len(res["errors"]))
    return 1 if res["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
