# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-chip DRAM check of a sharding plan, computed from the real checkpoint, not from the planner's own numbers.

The plan agent writes ``<bringup_dir>/plan.yaml``:

    headroom_frac: 0.15                 # fraction of chip DRAM kept free (default 0.15)
    users: 1                            # KV-cache user slots at the target context (default 1)
    placements:                         # every checkpoint tensor must match one pattern; first match wins
      - {pattern: "model.embed_tokens.weight", placement: replicate, dtype: bf16}
      - {pattern: "model.layers.*.self_attn.q_proj.weight", placement: shard, dtype: bf16}
      - {pattern: "model.layers.*.experts.*", placement: expert, dtype: bfp8}
      - {pattern: "model.vision_tower.*", placement: skip, why: text only}
    state:                              # per-layer state at the target context, one entry per layer group
      - {layers: "0-29", what: KV, heads_per_chip: 2, head_dim: 256, tensors: 2, dtype: bf16, window: 1024}
    extra_gb_per_chip:                  # anything else resident: second caches, trace buffers, lookup tables
      attention bf16 cache: 1.2
    activations_gb_per_chip: 3.0        # peak transient activations for one chunk (planner's estimate, > 0)

placement: replicate (every chip holds it), shard (split over all chips), expert (an expert-stacked tensor or an
expert's weights split over all chips), shard_rows / shard_cols (split over mesh rows / columns only), split (over the
mesh axes of a role, ``over: <role>``; plan ``roles``, see plan/boxes.py), skip.
dtype: fp32, bf16, bfp8 (1088 B per 1024 values), bfp4 (576 B per 1024 values), int32, uint16, uint8.
State bytes per chip = heads_per_chip * head_dim * tensors * dtype bytes * length * users, where length is
ceil(min(target seq, window) / seq_stride / seq_divisor) (seq_stride: tokens per stored row, e.g. pooled keys;
seq_divisor: chips the sequence axis is split over), or 1 with ``per_token: false`` (a fixed-size state such as a
linear-attention recurrent state: head_dim is then the per-head size, e.g. 128 * 128, plus any conv tail).
A state entry may give ``heads`` (the total) with ``split_heads: <role>`` and ``split_seq: <role>`` instead of
heads_per_chip and seq_divisor: the per-chip numbers then follow the mesh being checked (F64, required with a
deployment box). The check also requires heads_per_chip * chips >= the config's KV heads (or the entry's ``kv_heads``) unless the entry
says ``replicated: true``; entries with ``per_token: false`` are checked only against their own ``kv_heads``.
"""

from __future__ import annotations

import fnmatch
import json
import re
import struct
from pathlib import Path

import yaml

from models.demos.common.bringup.core.spec import Spec, parse_layers

DTYPE_BYTES = {
    "fp32": 4.0,
    "bf16": 2.0,
    "bfp8": 1088 / 1024,
    "bfp4": 576 / 1024,
    "int32": 4.0,
    "uint16": 2.0,
    "uint8": 1.0,
}
PLACEMENTS = ("replicate", "shard", "expert", "shard_rows", "shard_cols", "split", "skip")
GB = 2**30
LAYER_RE = re.compile(r"(?:^|\.)(?:model|language_model)\.layers\.(\d+)\.")


def checkpoint_tensors(hf_dir: str | Path) -> dict[str, list[int]]:
    """Name -> shape of every tensor, read from the safetensors headers only (no tensor data is loaded)."""
    hf_dir = Path(hf_dir)
    trimmed = hf_dir / "bringup_trim.json"  # F47: a trimmed checkpoint keeps the whole model's map
    if trimmed.exists():
        return dict(json.loads(trimmed.read_text())["tensors"])
    files = sorted(hf_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {hf_dir}")
    out = {}
    for f in files:
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        for name, info in header.items():
            if name != "__metadata__":
                out[name] = info["shape"]
    return out


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def divisor(placement: str, mesh: list[int]) -> int:
    rows, cols = mesh
    return {"replicate": 1, "shard": rows * cols, "expert": rows * cols, "shard_rows": rows, "shard_cols": cols}[
        placement
    ]


def ways(role: str | None, mesh: list[int], roles: dict) -> int:
    """Chips a role is split over on this mesh (roles: role -> mesh axes); 1 for no role."""
    if not role:
        return 1
    n = 1
    for a in roles[role]:
        n *= mesh[a]
    return n


def layer_of(name: str) -> int | None:
    """The decoder layer a checkpoint tensor belongs to, or None (embedding, final norm, LM head, vision, ...)."""
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def check_plan(
    spec: Spec,
    plan: dict,
    tensors: dict[str, list[int]],
    hf_config: dict | None = None,
    *,
    mesh: list[int] | None = None,
    cap_gb: float | None = None,
    seq: int | None = None,
    layers: list[int] | None = None,
    box: str | None = None,
) -> dict:
    """Returns {errors, per_chip_bytes (by group), total_gb, capacity_gb, fits, unplaced, rows}.

    Defaults check the run's own box at ``target.seq`` (``plan.boxes.<box>`` overrides apply). The keywords check
    another box, context or layer set (F64): ``layers`` keeps the decoder-layer tensors and state of those layers only.
    """
    mesh = list(mesh or spec.mesh)
    chips = mesh[0] * mesh[1]
    cap_gb = float(cap_gb if cap_gb is not None else spec.get("box.chip_dram_gb"))
    over = (plan.get("boxes") or {}).get(box or spec.run_box) or {}
    roles = plan.get("roles") or {}
    headroom = float(over.get("headroom_frac", plan.get("headroom_frac", 0.15)))
    errs = []
    placements = plan.get("placements") or []
    for p in placements:
        if p.get("placement") not in PLACEMENTS:
            errs.append(f"placement {p.get('pattern')!r}: unknown placement {p.get('placement')!r}")
        if p.get("placement") == "split" and p.get("over") not in roles:
            errs.append(
                f"placement {p.get('pattern')!r}: split over {p.get('over')!r}, not one of roles {sorted(roles)}"
            )
        if p.get("placement") != "skip" and p.get("dtype") not in DTYPE_BYTES:
            errs.append(f"placement {p.get('pattern')!r}: unknown dtype {p.get('dtype')!r}")
    for pat, dt in (over.get("dtypes") or {}).items():
        if dt not in DTYPE_BYTES:
            errs.append(f"boxes.{box or spec.run_box}.dtypes {pat!r}: unknown dtype {dt!r}")
    if errs:
        return {"errors": errs, "fits": False}

    sel = set(layers) if layers is not None else None
    groups, rows_out, unplaced, used = {}, [], [], [0] * len(placements)
    for name, shape in sorted(tensors.items()):
        k = next((k for k, p in enumerate(placements) if fnmatch.fnmatchcase(name, p["pattern"])), None)
        if k is None:
            unplaced.append(name)
            continue
        used[k] += 1
        p = placements[k]
        if p["placement"] == "skip":
            continue
        if sel is not None and layer_of(name) is not None and layer_of(name) not in sel:
            continue
        dtype = next((d for pat, d in (over.get("dtypes") or {}).items() if fnmatch.fnmatchcase(name, pat)), p["dtype"])
        div = ways(p["over"], mesh, roles) if p["placement"] == "split" else divisor(p["placement"], mesh)
        b = _numel(shape) * DTYPE_BYTES[dtype] / div
        key = p.get("group") or p["pattern"]
        groups[key] = groups.get(key, 0.0) + b
    for k, p in enumerate(placements):
        if not used[k]:
            errs.append(f"placement {p['pattern']!r} matches no checkpoint tensor")
    if unplaced:
        errs.append(f"{len(unplaced)} checkpoint tensors match no placement, e.g. {unplaced[:5]}")

    seq = int(seq if seq is not None else spec.get("target.seq"))
    users = int(plan.get("users", 1))
    kv_heads = (hf_config or {}).get("num_key_value_heads")
    covered = []
    for st in plan.get("state") or []:
        try:
            layers_st = parse_layers(st["layers"], spec.num_layers)
            if "heads" in st:
                hpc = -(-int(st["heads"]) // ways(st.get("split_heads"), mesh, roles))
            else:
                hpc = st["heads_per_chip"]
            b_tok = hpc * st["head_dim"] * st.get("tensors", 2) * DTYPE_BYTES[st["dtype"]]
        except (KeyError, TypeError) as e:
            errs.append(f"state entry {st}: missing or bad field {e}")
            continue
        if sel is not None:
            layers_st = [i for i in layers_st if i in sel]
        per_token = st.get("per_token", True)
        heads_total = st.get("kv_heads", kv_heads if per_token else None)
        if heads_total and not st.get("replicated"):
            if "heads" in st and int(st["heads"]) < heads_total:
                errs.append(f"state {st['layers']}: heads {st['heads']} < {heads_total} KV heads")
            elif "heads" not in st and st["heads_per_chip"] * chips < heads_total:
                errs.append(
                    f"state {st['layers']}: {st['heads_per_chip']} heads/chip x {chips} chips < {heads_total} KV heads"
                )
        div = int(st.get("seq_stride", 1)) * (
            ways(st["split_seq"], mesh, roles) if st.get("split_seq") else int(st.get("seq_divisor", 1))
        )
        length = -(-min(seq, st.get("window") or seq) // div) if per_token else 1
        groups[f"state: {st.get('what', 'state')} layers {st['layers']}"] = len(layers_st) * b_tok * length * users
        covered += layers_st
    missing_state = sorted(set(spec.layers() if sel is None else sel) - set(covered))
    if missing_state:
        errs.append(f"no state entry covers layers {missing_state}")
    for k, v in (plan.get("extra_gb_per_chip") or {}).items():
        groups[f"extra: {k}"] = float(v) * GB
    act = float(over.get("activations_gb_per_chip", plan.get("activations_gb_per_chip")) or 0)
    if act <= 0:
        errs.append("activations_gb_per_chip must be a positive estimate")
    groups["activations (planner estimate)"] = act * GB

    total_gb = sum(groups.values()) / GB
    budget = cap_gb * (1 - headroom)
    fits = total_gb <= budget
    if not fits:
        errs.append(
            f"per-chip total {total_gb:.2f} GB exceeds {budget:.2f} GB ({cap_gb:.0f} GB minus {headroom:.0%} headroom)"
        )
    rows_out = [{"group": k, "gb": v / GB} for k, v in sorted(groups.items(), key=lambda x: -x[1])]
    return {
        "errors": errs,
        "rows": rows_out,
        "total_gb": total_gb,
        "capacity_gb": cap_gb,
        "budget_gb": budget,
        "headroom_frac": headroom,
        "fits": fits and not errs,
        "unplaced": unplaced,
        "tensors": len(tensors),
        "mesh": mesh,
        "seq": seq,
        "users": users,
    }


def load_plan(spec: Spec) -> dict:
    p = spec.bringup_dir / "plan.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no plan: {p}")
    return yaml.safe_load(p.read_text()) or {}
