# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import importlib

import ttnn
from models.demos.llama31_8b_p100.tt import p100

_mods = {}


def op(name):
    if name not in _mods:
        full = f"{__package__}.ops_{name}"
        try:
            _mods[name] = importlib.import_module(full)
        except ModuleNotFoundError as e:
            if e.name != full:
                raise
            _mods[name] = None
    return _mods[name]


def reset():
    for m in _mods.values():
        if m is not None and hasattr(m, "reset"):
            m.reset()


def on(name, args):
    m = op(name)
    return m if m is not None and p100.enabled(args) and p100.has_ops(*m.OPS) else None


def linear(mm, *args, **kwargs):
    return mm.linear(*args, **kwargs) if mm is not None else ttnn.linear(*args, **kwargs)


def minimal_matmul(mm, *args, **kwargs):
    return mm.minimal_matmul(*args, **kwargs) if mm is not None else ttnn.experimental.minimal_matmul(*args, **kwargs)


def qkv_heads(mm, *args, **kwargs):
    return mm.qkv_heads(*args, **kwargs) if mm is not None else None


def swiglu(mm, *args, **kwargs):
    return mm.swiglu(*args, **kwargs) if mm is not None else None
