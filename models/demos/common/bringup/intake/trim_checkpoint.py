# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Trim a layer-subset bring-up's checkpoint once the whole model is no longer needed (F47).

    python -m models.demos.common.bringup.intake.trim_checkpoint --spec S

The whole checkpoint is needed only for the one-time HF sanity check (usage-example smoke, accuracy floor). Once that
check and the reference's HF parity have passed (task R.4 depends on R.2), everything later reads only layers
0..K-1, K = max(last subset layer + 1, hf.parity_layers). This keeps those layers and every non-layer tensor except
``checkpoint.trim_drop`` globs (e.g. MTP or vision towers), and frees the rest:

- a shard holding only dropped tensors is deleted; a shard holding only kept tensors stays as it is;
- a mixed shard (e.g. expert-parallel shards that hold a slice of every layer) is rewritten as ``subset-<shard>``
  with the kept tensors, every tensor checked byte for byte against the original before the original is deleted;
- ``model.safetensors.index.json`` then lists only the kept tensors; the original is kept as
  ``model.safetensors.index.full.json``.

``bringup_trim.json`` in the checkpoint dir records the full tensor map (name -> shape) and the sanity metrics. After
the trim, ``plan.memory.checkpoint_tensors`` returns that map (so the checkpoint gate and the plan's memory check see
the whole model) and ``check_hf_sanity`` replays the recorded metrics, so R.1 and R.2 still rerun. To run the
whole-model sanity for real again, re-download the pinned revision.

Only for a spec that owns its checkpoint (no ``prior``), whose sanity metrics pass, and unless
``checkpoint.trim: false``. Idempotent; an interrupted trim resumes. Records trim_done, trim_verify_errors,
trim_kept_tensors, trim_dropped_tensors, trim_freed_gb.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import time
from pathlib import Path

MARKER = "bringup_trim.json"
INDEX = "model.safetensors.index.json"
FULL_INDEX = "model.safetensors.index.full.json"
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def keep_layers(spec) -> int:
    """Layers 0..K-1 stay: the subset's prefix (the reference runs the layers before a subset layer too) and parity."""
    return max(max(spec.layers()) + 1, int(spec.get("hf.parity_layers") or 0))


def applies(spec) -> bool:
    """A trim can free something: the kept prefix stops before the last layer, or trim_drop names tensors."""
    if spec.get("prior") or spec.get("checkpoint.trim") is False:
        return False
    return keep_layers(spec) < spec.num_layers or bool(spec.get("checkpoint.trim_drop"))


def kept(name: str, k: int, drop: list[str]) -> bool:
    if any(fnmatch.fnmatchcase(name, g) for g in drop):
        return False
    m = _LAYER.search(name)
    return m is None or int(m.group(1)) < k


def classify(weight_map: dict[str, str], k: int, drop: list[str]) -> dict[str, str]:
    """shard -> keep | drop | rewrite."""
    by_shard: dict[str, list[bool]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(kept(name, k, drop))
    return {s: "keep" if all(v) else "drop" if not any(v) else "rewrite" for s, v in by_shard.items()}


def _same_bytes(a, b) -> bool:
    import torch

    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8))
    )


def _rewrite(hf: Path, shard: str, names: list[str]) -> int:
    """Write the kept tensors of one shard to subset-<shard>, verified byte for byte; returns the mismatch count."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    out = hf / f"subset-{shard}"
    with safe_open(str(hf / shard), framework="pt") as src:
        if out.exists():  # an interrupted run: keep it only if it verifies
            try:
                with safe_open(str(out), framework="pt") as h:
                    if set(h.keys()) == set(names) and all(
                        _same_bytes(src.get_tensor(n), h.get_tensor(n)) for n in names
                    ):
                        return 0
            except Exception:  # a truncated file
                pass
        tensors = {n: src.get_tensor(n) for n in names}
        tmp = out.with_name(out.name + ".tmp")
        save_file(tensors, str(tmp), metadata={"format": "pt"})
        with safe_open(str(tmp), framework="pt") as h:
            bad = sum(not _same_bytes(tensors[n], h.get_tensor(n)) for n in names)
    if bad:
        tmp.unlink()
        return bad
    os.replace(tmp, out)
    return 0


def _drop_download_meta(hf: Path, shard: str) -> None:
    meta = hf / ".cache" / "huggingface" / "download"
    for suffix in (".metadata", ".lock"):
        (meta / f"{shard}{suffix}").unlink(missing_ok=True)


def trim(spec, hf: Path, sanity: dict | None = None) -> dict:
    """Trim hf in place; returns the marker. ``sanity`` is recorded in the marker for check_hf_sanity to replay."""
    from models.demos.common.bringup.plan.memory import checkpoint_tensors

    marker_path = hf / MARKER
    marker = json.loads(marker_path.read_text()) if marker_path.exists() else None
    if marker and marker.get("status") == "done":
        return marker
    if not (hf / FULL_INDEX).exists():
        if not (hf / INDEX).exists():
            raise SystemExit(f"{hf} has no {INDEX}: a single-file checkpoint is not trimmed")
        shutil.copyfile(hf / INDEX, hf / FULL_INDEX)
    full = json.loads((hf / FULL_INDEX).read_text())
    wm = full["weight_map"]
    k, drop = keep_layers(spec), list(spec.get("checkpoint.trim_drop") or [])
    if marker is None:
        marker = {
            "status": "trimming",
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": spec.data["hf_id"],
            "revision": spec.get("hf.revision"),
            "keep_layers": k,
            "drop_globs": drop,
            "tensors": checkpoint_tensors(hf),  # read before anything is removed
            "sanity": sanity or {},
        }
        marker_path.write_text(json.dumps(marker))
    kinds = classify(wm, k, drop)
    need = sum((hf / s).stat().st_size for s, c in kinds.items() if c == "rewrite" and (hf / s).exists())
    if need and shutil.disk_usage(hf).free < need:  # upper bound: the kept part of the mixed shards
        raise SystemExit(f"not enough free disk to rewrite the mixed shards ({need / 1e9:.1f} GB)")

    errors, new_wm, freed = 0, {}, 0
    for shard, c in sorted(kinds.items()):
        names = sorted(n for n, s in wm.items() if s == shard and kept(n, k, drop))
        if c == "keep":
            new_wm.update({n: shard for n in names})
        elif c == "rewrite":
            if (hf / shard).exists():
                bad = _rewrite(hf, shard, names)
                errors += bad
                if bad:
                    print(f"VERIFY   {shard}: {bad} tensors differ after rewrite; shard kept")
                    new_wm.update({n: shard for n in names})
                    continue
            new_wm.update({n: f"subset-{shard}" for n in names})
    missing = [s for s in set(new_wm.values()) if not (hf / s).exists()]
    if missing:
        raise SystemExit(f"shards the trimmed index needs are missing: {sorted(missing)[:5]}")
    index = {
        "metadata": {**(full.get("metadata") or {}), "bringup_trim": f"layers 0-{k - 1}, see {MARKER}"},
        "weight_map": dict(sorted(new_wm.items())),
    }
    tmp = hf / (INDEX + ".tmp")
    tmp.write_text(json.dumps(index, indent=2))
    os.replace(tmp, hf / INDEX)
    used = set(new_wm.values())
    for shard in sorted(kinds):
        if shard not in used and (hf / shard).exists():
            freed += (hf / shard).stat().st_size
            (hf / shard).unlink()
            _drop_download_meta(hf, shard)
    freed -= sum((hf / s).stat().st_size for s in used if s.startswith("subset-"))
    marker.update(
        status="done" if errors == 0 else "partial",
        t_done=time.strftime("%Y-%m-%dT%H:%M:%S"),
        kept_tensors=len(new_wm),
        dropped_tensors=len(wm) - len(new_wm),
        freed_bytes=max(freed, 0) + marker.get("freed_bytes", 0),
        verify_errors=errors,
    )
    marker_path.write_text(json.dumps(marker))
    return marker


def load_marker(hf) -> dict | None:
    p = Path(hf) / MARKER
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return m if m.get("status") in ("done", "partial", "trimming") else None


def sanity_task(spec) -> str:
    return "R.2" if spec.get("hf.custom_loader") else "R.1"


def main(argv=None):
    from models.demos.common.bringup.core import metrics
    from models.demos.common.bringup.core.gate import check_metrics
    from models.demos.common.bringup.plan.ledger_gen import sanity_metrics
    from models.demos.common.bringup.reference.golden import hf_path, load_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    hf = Path(hf_path(spec))
    if not applies(spec):
        print("nothing to trim (full model, a prior's checkpoint, or checkpoint.trim: false)")
        metrics.record("trim_done", 1)
        metrics.record("trim_verify_errors", 0)
        metrics.record("trim_freed_gb", 0.0)
        return
    marker = load_marker(hf)
    if not (marker and marker.get("status") == "done"):
        # Only after the whole-model sanity passed: the trim removes what it needs.
        want, got = sanity_metrics(spec), metrics.load(sanity_task(spec))
        ok, lines = check_metrics(want, got)
        print("\n".join(lines))
        if not ok:
            raise SystemExit(f"the HF sanity metrics of {sanity_task(spec)} do not pass: not trimming")
        marker = trim(spec, hf, {n: v["value"] for n, v in got.items() if n in want})
    gb = marker.get("freed_bytes", 0) / 1e9
    print(
        f"checkpoint {hf}: layers 0-{marker['keep_layers'] - 1} kept, {marker.get('kept_tensors')} tensors kept, "
        f"{marker.get('dropped_tensors')} dropped, {gb:.1f} GB freed, verify errors {marker.get('verify_errors')}"
    )
    metrics.record("trim_done", int(marker.get("status") == "done"))
    metrics.record("trim_verify_errors", marker.get("verify_errors", 0))
    metrics.record("trim_kept_tensors", marker.get("kept_tensors", 0))
    metrics.record("trim_dropped_tensors", marker.get("dropped_tensors", 0))
    metrics.record("trim_freed_gb", round(gb, 2))


if __name__ == "__main__":
    main()
