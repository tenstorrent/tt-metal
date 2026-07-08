# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from typing import Any, Dict, List

BENIGN_OPS = (
    "detach",
    "view",
    "_unsafe_view",
    "size",
    "sym_size",
    "stride",
    "sym_stride",
    "is_contiguous",
    "contiguous",
    "_to_copy",
    "to",
    "clone",
    "alias",
    "resolve_conj",
    "resolve_neg",
    "lift_fresh",
    "_local_scalar_dense",
)


@contextlib.contextmanager
def observe_host_ops():
    from torch.utils._python_dispatch import TorchDispatchMode

    seen: List[str] = []

    class _Recorder(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            try:
                seen.append(str(func))
            except Exception:  # noqa: BLE001
                pass
            return func(*args, **(kwargs or {}))

    with _Recorder():
        yield seen


def observe_forward(forward_fn) -> List[str]:
    with observe_host_ops() as ops:
        forward_fn()
    return list(ops)


def _is_benign(op: str, benign) -> bool:
    tail = op.split("aten.", 1)[-1].split(".")[0] if "aten." in op else op
    return any(b == tail or b in op for b in benign)


def verdict(ops: List[str], benign=BENIGN_OPS) -> Dict[str, Any]:
    host = sorted({o for o in ops if not _is_benign(o, benign)})
    return {
        "on_device": len(host) == 0,
        "host_ops": host,
        "n_host_ops": len(host),
        "reason": (
            "fully on device: no host aten ops fired in the forward"
            if not host
            else "host compute in the forward — aten ops fired on host: " + ", ".join(host[:12])
        ),
    }
