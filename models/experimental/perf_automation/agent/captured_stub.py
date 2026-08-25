# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Stand in for a captured reference tensor the model declares but does not ship.

WHY THIS EXISTS. A pipeline's <stage>_trace_inputs() hooks read the golden tensors a bring-up capture
wrote into `_captured/`. Those tensors are large and are NOT committed -- only the manifest beside
them is -- so on any tree that has not run that capture, which is every model being optimised for the
first time, every stage raises FileNotFoundError and the per-stage split is lost. voxtral run 35:

    [perf-adapter] stage 'encode'  cannot prepare its own inputs (FileNotFoundError ... args.pt)
    [perf-adapter] stage 'prefill' cannot prepare its own inputs (FileNotFoundError ... output.pt)
    [perf-adapter] stage 'decode'  cannot prepare its own inputs (FileNotFoundError ... output.pt)
    STAGE_MARKS_RESULT=0

Nobody should have to hand over a data file to optimise a new model, and nobody has to: the manifest
that IS shipped declares the shape and dtype of everything the hook would have loaded.

    _captured/voxtral_encoder/manifest.json
      args   : tuple -> tensor [1, 128, 3000] bfloat16
      output : dict  -> last_hidden_state [1, 1500, 1280], pooler_output [375, 3072]

The VALUES are irrelevant here: this is a timing measurement, not a correctness one -- the same
reasoning a generated test used when it fed the hooks torch.randn instead of the golden tensors. So
the shape is enough, and the shape is declared.

NOTHING IS INFERRED ABOUT THE MODEL. The path comes from the exception the hook itself raised, the
manifest sits beside that path, and the structure is read from the manifest. No stage names, no
component names, no mapping table, and no dependence on how a generated test is written -- which is
what every earlier attempt at this rested on, and why each one broke on the next generated file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# What a manifest can describe. Containers nest; a tensor is the leaf that needs materialising.
_TENSOR, _TUPLE, _LIST, _DICT = "tensor", "tuple", "list", "dict"


def _dtype(name):
    """torch dtype from the manifest's spelling, or the default. Never raises."""
    import torch

    txt = str(name or "").strip()
    if txt.startswith("torch."):
        txt = txt[len("torch.") :]
    return getattr(torch, txt, None) or torch.float32


def build(spec):
    """Materialise one manifest node: a tensor of the declared shape, or the container holding them.

    Zeros rather than random: a perf measurement does not read the values, and a deterministic buffer
    makes two runs of the same stage comparable. Anything the manifest does not describe as a tensor
    or a container comes back as its literal value, which is what a scalar field already is.
    """
    import torch

    if not isinstance(spec, dict):
        return spec
    kind = spec.get("kind")
    if kind == _TENSOR:
        shape = [int(d) for d in (spec.get("shape") or [])]
        return torch.zeros(*shape, dtype=_dtype(spec.get("dtype"))) if shape else torch.zeros(1)
    if kind == _TUPLE:
        return tuple(build(i) for i in (spec.get("items") or []))
    if kind == _LIST:
        return [build(i) for i in (spec.get("items") or [])]
    if kind == _DICT:
        return {k: build(v) for k, v in (spec.get("items") or {}).items()}
    return spec.get("value")


def for_missing_file(path) -> tuple:
    """(object, why) to stand in for the missing capture at `path`, or (None, reason).

    The manifest is looked up BESIDE the file, because that is where the capture wrote it: the hook
    asked for `<dir>/args.pt`, so `<dir>/manifest.json` describes it. The stem selects the field --
    args.pt is the manifest's `args`, output.pt its `output` -- which is the capture's own naming,
    not a guess about this model.
    """
    p = Path(path)
    man = p.parent / "manifest.json"
    if not man.is_file():
        return None, "no manifest beside %s to describe it" % p.name
    try:
        doc = json.loads(man.read_text())
    except Exception as exc:  # noqa: BLE001
        return None, "%s is not readable (%s)" % (man.name, type(exc).__name__)
    spec = doc.get(p.stem)
    if spec is None:
        return None, "%s describes %s, not %r" % (man.name, sorted(k for k in doc if k not in ("component", "submodule_path")), p.stem)
    try:
        return build(spec), "synthesised %s from %s" % (p.name, man.parent.name + "/" + man.name)
    except Exception as exc:  # noqa: BLE001
        return None, "could not build %s from its manifest (%s: %s)" % (p.name, type(exc).__name__, str(exc)[:80])


def missing_path(exc):
    """The file a FileNotFoundError was raised for, or None. Never raises."""
    try:
        name = getattr(exc, "filename", None)
        if name:
            return Path(name)
        # Some loaders re-raise with the path only in the message.
        text = str(exc)
        for tok in text.replace("'", " ").replace('"', " ").split():
            if tok.endswith(".pt") or tok.endswith(".pth"):
                return Path(tok)
    except Exception:  # noqa: BLE001
        pass
    return None


def install(monkey_target=None):
    """Make torch.load fall back to the manifest when the file is absent. Returns a restore callable.

    Patched at torch.load rather than at each hook: the hooks are the model's code and there are as
    many shapes of them as there are models, while torch.load is the one call all of them make. A
    file that EXISTS is loaded normally -- this only supplies what was never shipped.
    """
    import torch

    target = monkey_target or torch
    original = target.load

    def loading(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except FileNotFoundError as exc:
            path = missing_path(exc) or (Path(args[0]) if args else None)
            if path is None:
                raise
            obj, why = for_missing_file(path)
            if obj is None:
                print("  [captured-stub] %s: %s" % (path.name, why), file=sys.stderr, flush=True)
                raise
            print("  [captured-stub] %s" % why, file=sys.stderr, flush=True)
            return obj

    target.load = loading

    def restore():
        target.load = original

    return restore
