# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Weight-format variant of a finished tt weight cache: same tensors, some stored in another dtype.

    python -m models.demos.inkling2_mini.convert.variant --base cache/v1-bank --out cache/v1-bank-attn4 \\
        --set '*/attn.w_*=BFP4' [--set 'global/unembed=BFP4'] [--take cache/other:'*/experts.w_down']

The new cache hard-links every file of ``--base`` and rewrites only the tensors a ``--set`` pattern picks (fnmatch
on ``Lxx/<name>`` or ``global/<name>``), converted again from the checkpoint in the new dtype, in the base
tensor's layout (TILE, or the BANK / STICK expert layouts). ``--take`` links matching tensors (file + manifest
entry) from another cache instead, e.g. to combine variants without converting twice. The base cache is never
written: a rewritten file is unlinked first. The model loads a variant like any cache (``root=`` /
``$INKLING_CACHE``) and takes each tensor's dtype from it; the default cache is unchanged.

``--experts-from lotto:DIR | codes:DIR --layers 0-23`` rewrites both expert tensors of those layers as BFP4 from
GPTQ-calibrated values instead of the checkpoint (convert/gptq_codes.py); each written file is read back and
checked value for value against its source (``gptq_exact`` in the manifest entry).

Dtypes per tensor class the model reads: experts BFP4 / BFP8; attention w_qgrs / w_k / w_v / w_o BFP4 / BFP8;
router.w FP32 / BF16 / BFP8; msg_gate.*.forget / gain FP32 / BFP8 (FP32 is uploaded as bf16, exact);
unembed BFP4 / BFP8.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
from pathlib import Path

import torch

import ttnn
from models.demos.inkling2_mini.convert import convert as C
from models.demos.inkling2_mini.convert import gptq_codes
from models.demos.inkling2_mini.convert.bank_layout import write_bank
from models.demos.inkling2_mini.convert.source import Checkpoint
from models.demos.inkling2_mini.convert.stick import write_sticks


def key_of(e: dict) -> str:
    return "global/" + e["name"] if e["layer"] is None else f"L{e['layer']:02d}/{e['name']}"


class _Only(C.Writer):
    """Writer that keeps only the tensors in ``want`` ({name: dtype}) and writes them in that dtype."""

    def __init__(self, out: Path, layer, want: dict[str, str], check: bool = False):
        super().__init__(out)
        self.layer, self.want, self.check = layer, want, check

    def write(self, name, layer, tensor, dtype, layout, src, **kw):
        if name not in self.want:
            return
        rel = f"global/{name}.tensorbin" if layer is None else f"layers/L{layer:02d}/{name}.tensorbin"
        (self.out / rel).unlink(missing_ok=True)  # a hard link to the base: never write through it
        super().write(name, layer, tensor, self.want[name], layout, src, **kw)
        if self.check:  # GPTQ source: the stored BFP4 values must be the source values, bit for bit
            got = ttnn.to_torch(ttnn.load_tensor(str(self.out / rel))).to(torch.bfloat16).reshape(tensor.shape)
            self.entries[-1]["gptq_exact"] = bool(torch.equal(got, tensor.to(torch.bfloat16)))


def link_tree(base: Path, out: Path):
    for d, _, files in os.walk(base):
        rel = Path(d).relative_to(base)
        (out / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            if f != "manifest.json":
                os.link(Path(d) / f, out / rel / f)


def rewrite(ck: Checkpoint, out: Path, entries: list[dict], layer, want: dict[str, str], store=None) -> list[dict]:
    """Convert the tensors ``want`` of one unit again in their new dtype (experts from ``store`` if given); returns
    their new manifest entries."""
    w = _Only(out, layer, want, check=store is not None)
    if layer is None:
        C.convert_global(ck, w)
    else:
        if not set(want) <= {"experts.w_gate_up", "experts.w_down"}:
            C.convert_layer_dense(ck, w, layer)
        p = f"{C.P}layers.{layer}."
        for name, src, is_w13 in (("experts.w_gate_up", "w13", True), ("experts.w_down", "w2", False)):
            if name in want:
                C.convert_experts(ck, w, layer, name, f"{p}mlp.experts.{src}_weight", is_w13, want[name], store)
    old = {e["name"]: e for e in entries if e["layer"] == layer}
    new = []
    for e in w.entries:
        layout = old[e["name"]]["layout"]
        f = out / e["file"]
        if layout == "BANK":
            e = write_bank(f, f, e)
        elif layout == "STICK":
            e = write_sticks(f, f, e)
        elif layout != e["layout"]:
            raise ValueError(f"{key_of(e)}: base layout {layout}, converter wrote {e['layout']}")
        new.append(e)
    if missing := set(want) - {e["name"] for e in new}:
        raise ValueError(f"unit {layer}: converter wrote no {sorted(missing)}")
    return new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=os.environ.get("INKLING_CKPT"))
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--set", action="append", default=[], help="PATTERN=DTYPE, e.g. '*/experts.w_down=BFP4'")
    ap.add_argument("--take", action="append", default=[], help="CACHE:PATTERN, link matching tensors from CACHE")
    ap.add_argument("--experts-from", help="lotto:DIR or codes:DIR: GPTQ BFP4 experts (convert/gptq_codes.py)")
    ap.add_argument("--layers", default="0-23", help="layers for --experts-from, e.g. 0-19 or 20-23,5")
    ap.add_argument("--exact-only", action="store_true", help="refuse layers whose values are not BFP4-exact")
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; pick a new --out")
    m = json.loads((args.base / "manifest.json").read_text())
    if not m.get("complete"):
        raise SystemExit(f"{args.base}: manifest not complete")
    sets = [s.rsplit("=", 1) for s in args.set]
    if bad := [d for _, d in sets if d not in C.DTYPES]:
        raise SystemExit(f"--set: unknown dtypes {bad} (one of {sorted(C.DTYPES)})")

    tmp = args.out.with_name(args.out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    link_tree(args.base, tmp)
    entries = m["tensors"]
    changes = {}  # key -> source
    for spec in args.take:
        src, pat = spec.split(":", 1)
        sm = json.loads((Path(src) / "manifest.json").read_text())
        for e in sm["tensors"]:
            if fnmatch.fnmatch(key_of(e), pat):
                (tmp / e["file"]).unlink()
                os.link(Path(src) / e["file"], tmp / e["file"])
                entries = [x for x in entries if key_of(x) != key_of(e)] + [e]
                changes[key_of(e)] = f"{src} ({e['dtype']})"
    units: dict = {}
    for e in entries:
        dt = next((d for p, d in sets if fnmatch.fnmatch(key_of(e), p)), None)
        if dt and dt != e["dtype"]:
            units.setdefault(e["layer"], {})[e["name"]] = dt
    gptq_layers = set()
    if args.experts_from:
        for part in args.layers.split(","):
            a, _, b = part.partition("-")
            gptq_layers |= set(range(int(a), int(b or a) + 1))
        for layer in gptq_layers:
            units.setdefault(layer, {}).update({"experts.w_gate_up": "BFP4", "experts.w_down": "BFP4"})
    if args.set and not units:
        raise SystemExit("--set picked no tensor whose dtype changes")
    ck = Checkpoint(args.ckpt) if units else None
    if ck and ck.header_sha256() != m["source_header_sha256"]:
        raise SystemExit(f"{args.ckpt} is not the checkpoint {args.base} was converted from")
    for layer, want in sorted(units.items(), key=lambda x: -1 if x[0] is None else x[0]):
        print(f"{'global' if layer is None else f'L{layer:02d}'}: {want}", flush=True)
        store, exact = None, None
        if layer in gptq_layers:
            store, exact = gptq_codes.layer_store(args.experts_from, layer, args.exact_only)
        new = rewrite(ck, tmp, entries, layer, want, store)
        del store
        for e in new:
            if layer in gptq_layers and e["name"].startswith("experts."):
                e["gptq_source"] = args.experts_from + ("" if exact else " (factor folded, re-rounded)")
                if exact and not e.get("gptq_exact"):
                    raise SystemExit(f"L{layer:02d}/{e['name']}: stored values differ from the GPTQ source")
        names = {e["name"] for e in new}
        entries = [e for e in entries if not (e["layer"] == layer and e["name"] in names)] + new
        changes |= {key_of(e): f"{e['dtype']} pcc {e['pcc']}" for e in new}
    m["tensors"] = entries
    m["variant"] = {"base": str(args.base), "set": args.set, "take": args.take, "changed": changes}
    if args.experts_from:
        m["variant"] |= {"experts_from": args.experts_from, "gptq_layers": sorted(gptq_layers)}
    m["expert_dtype"] = {key_of(e): e["dtype"] for e in entries if e["name"].startswith("experts.")}
    (tmp / "manifest.json").write_text(json.dumps(m, indent=1))
    tmp.rename(args.out)
    print(f"{args.out}: {len(changes)} tensors changed")


if __name__ == "__main__":
    main()
