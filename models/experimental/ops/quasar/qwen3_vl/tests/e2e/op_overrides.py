# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Named workarounds (and, for bisecting, host fallbacks) installed around ttnn ops for one test."""
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class Workaround:
    name: str
    target: str
    reason: str
    remove_when: str
    applies: Callable
    rewrite: Callable


def resolve(target):
    import ttnn

    parts = target.split(".")
    assert parts[0] == "ttnn", target
    parent = ttnn
    for p in parts[1:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _fp32_to_bf16_on_device(args, kwargs):
    import ttnn

    src = args[0] if args else kwargs.get("tensor")
    return (
        isinstance(src, torch.Tensor)
        and src.dtype == torch.float32
        and kwargs.get("dtype") == ttnn.bfloat16
        and kwargs.get("device") is not None
    )


def _cast_source_to_bf16(original, args, kwargs):
    if args:
        return original(args[0].to(torch.bfloat16), *args[1:], **kwargs)
    return original(**{**kwargs, "tensor": kwargs["tensor"].to(torch.bfloat16)})


WORKAROUNDS = [
    Workaround(
        name="host_cast_fp32_upload",
        target="ttnn.from_torch",
        reason="fp32->bf16 uploads tilize in fp32 on device, unpacking fp32 to SrcA (#57780; QUASAR_GAPS Q3/S2)",
        remove_when="#57780 fixed: fp32 tilize unpacks to DEST on Quasar (and ttsim WH accepts it)",
        applies=_fp32_to_bf16_on_device,
        rewrite=_cast_source_to_bf16,
    ),
]


class OverrideSession:
    def __init__(self, mesh_device, host_ops, disable_wa):
        self.mesh_device = mesh_device
        self.host_ops = tuple(host_ops)
        self.disable_wa = set(disable_wa)
        self.hits = Counter()
        self.host_ops_active = []

    def install(self, monkeypatch):
        unknown = self.disable_wa - {w.name for w in WORKAROUNDS}
        if unknown:
            raise KeyError(f"unknown workaround(s): {sorted(unknown)}")
        for wa in WORKAROUNDS:
            if wa.name not in self.disable_wa:
                self._install_workaround(monkeypatch, wa)

    def _install_workaround(self, monkeypatch, wa):
        parent, attr = resolve(wa.target)
        original = getattr(parent, attr)

        def wrapper(*args, **kwargs):
            if wa.applies(args, kwargs):
                self.hits[f"wa:{wa.name}"] += 1
                return wa.rewrite(original, args, kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(parent, attr, wrapper)
