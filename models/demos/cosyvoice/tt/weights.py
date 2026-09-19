# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Load exported CosyVoice weights without importing the CosyVoice package.

`scripts/export_weights.py` runs once in the reference venv and emits a flat
`.npz` with weight_norm already folded. This module turns that into TTNN modules.

The point is the boundary: **tt-metal's environment never imports cosyvoice,
hyperpyyaml, or the reference's torch pin.** That is the same rule the demo README
states and the Docker image enforces the hard way -- installing whisper into
tt-metal's `python_env` pulled a triton that broke `import torch` outright.

Everything a conv needs beyond its tensors -- stride, padding, dilation -- is
either derivable from the weight shape or an architectural constant in
`model_config.py`, so nothing has to be carried in the checkpoint.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch


class WeightBag:
    """Prefix-scoped view over the exported arrays.

    `bag["conv_pre"].tensor("weight")` and `bag.sub("resblocks.0")` keep call
    sites readable when the names are three levels deep.
    """

    def __init__(self, arrays: dict[str, np.ndarray], meta: dict, prefix: str = ""):
        self._arrays = arrays
        self.meta = meta
        self._prefix = prefix

    @classmethod
    def load(cls, path: str) -> "WeightBag":
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files if k != "__meta__"}
            meta = json.loads(bytes(z["__meta__"]).decode()) if "__meta__" in z.files else {}
        return cls(arrays, meta)

    def sub(self, prefix: str) -> "WeightBag":
        full = f"{self._prefix}.{prefix}" if self._prefix else prefix
        return WeightBag(self._arrays, self.meta, full)

    def _key(self, name: str) -> str:
        return f"{self._prefix}.{name}" if self._prefix else name

    def has(self, name: str) -> bool:
        return self._key(name) in self._arrays

    def tensor(self, name: str, dtype=torch.float32) -> torch.Tensor:
        key = self._key(name)
        if key not in self._arrays:
            raise KeyError(f"{key} not in exported weights")
        # Exported large arrays are fp16; widen so the TTNN side decides its own dtype.
        return torch.from_numpy(np.ascontiguousarray(self._arrays[key])).to(dtype)

    def optional(self, name: str, dtype=torch.float32) -> torch.Tensor | None:
        return self.tensor(name, dtype) if self.has(name) else None

    def children(self) -> int:
        """How many indexed children sit directly under this prefix.

        `count("x")` asks about `x.N`; this asks about `N` itself, which is what
        an `nn.ModuleList` reached via `.sub()` looks like.
        """
        base = f"{self._prefix}." if self._prefix else ""
        idx = set()
        for k in self._arrays:
            if base and not k.startswith(base):
                continue
            head = k[len(base) :].split(".", 1)[0]
            if head.isdigit():
                idx.add(int(head))
        return len(idx)

    def count(self, pattern: str) -> int:
        """How many indexed children exist under `pattern.N`."""
        base = self._key(pattern)
        idx = set()
        for k in self._arrays:
            if k.startswith(base + "."):
                head = k[len(base) + 1 :].split(".", 1)[0]
                if head.isdigit():
                    idx.add(int(head))
        return len(idx)

    @property
    def window(self) -> np.ndarray | None:
        w = self.meta.get("stft_window")
        return np.asarray(w, dtype=np.float32) if w else None


def default_weights_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden", "hift_weights.npz")


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
def build_conv1d(device, bag: WeightBag, *, stride=1, padding=0, dilation=1, groups=1, dtype=None):
    import ttnn

    from .hifigan.conv import TtConv1d

    dtype = dtype or ttnn.bfloat16
    return TtConv1d(
        device,
        bag.tensor("weight"),
        bag.optional("bias"),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dtype=dtype,
    )


def build_conv_transpose1d(device, bag: WeightBag, *, stride, padding, dtype=None):
    import ttnn

    from .hifigan.upsample import TtConvTranspose1d

    dtype = dtype or ttnn.bfloat16
    return TtConvTranspose1d(
        device, bag.tensor("weight"), bag.optional("bias"), stride=stride, padding=padding, dtype=dtype
    )


def build_resblock(device, bag: WeightBag, *, dilations=(1, 3, 5), dtype=None):
    """Kernel size and channel count come from the weight shape; only the
    dilation schedule is architectural."""
    import ttnn

    from .hifigan.conv import TtConv1d
    from .hifigan.resblock import TtResBlock, get_padding

    dtype = dtype or ttnn.bfloat16
    n = bag.count("convs1")
    dilations = tuple(dilations)[:n]
    w0 = bag.sub("convs1.0").tensor("weight")
    channels, _, kernel_size = w0.shape

    block = TtResBlock.__new__(TtResBlock)
    block.device, block.channels = device, channels
    block.kernel_size, block.dilations, block.n = kernel_size, dilations, n
    block.convs1 = [
        TtConv1d(
            device,
            bag.sub(f"convs1.{i}").tensor("weight"),
            bag.sub(f"convs1.{i}").optional("bias"),
            padding=get_padding(kernel_size, d),
            dilation=d,
            dtype=dtype,
        )
        for i, d in enumerate(dilations)
    ]
    block.convs2 = [
        TtConv1d(
            device,
            bag.sub(f"convs2.{i}").tensor("weight"),
            bag.sub(f"convs2.{i}").optional("bias"),
            padding=get_padding(kernel_size, 1),
            dilation=1,
            dtype=dtype,
        )
        for i in range(n)
    ]
    from .hifigan.snake import TtSnake

    block.act1 = [TtSnake(device, bag.sub(f"activations1.{i}").tensor("alpha"), dtype=dtype) for i in range(n)]
    block.act2 = [TtSnake(device, bag.sub(f"activations2.{i}").tensor("alpha"), dtype=dtype) for i in range(n)]
    return block


def build_f0_predictor(device, bag: WeightBag, dtype=None, weights_dtype=None):
    """condnet is Sequential(Conv, ELU, Conv, ELU, ...) so the convs sit at even
    indices; they are found by which entries actually carry a weight.

    `weights_dtype` is separate from `dtype` and matters here: f0 error integrates
    into the excitation phase over the whole utterance, so bfloat16 *weights* alone
    are enough to ruin it even when the activations are fp32. See TtSineGen.
    """
    import ttnn

    from .hifigan.conv import TtConv1d
    from .hifigan.f0 import TtF0Predictor

    dtype = dtype or ttnn.bfloat16
    weights_dtype = weights_dtype or dtype
    cond = bag.sub("condnet")
    idx = sorted(i for i in range(0, 32) if cond.sub(str(i)).has("weight"))

    obj = TtF0Predictor.__new__(TtF0Predictor)
    obj.device = device
    obj.convs = [
        TtConv1d(
            device,
            cond.sub(str(i)).tensor("weight"),
            cond.sub(str(i)).optional("bias"),
            padding=(cond.sub(str(i)).tensor("weight").shape[-1] - 1) // 2,
            dtype=dtype,
            weights_dtype=weights_dtype,
        )
        for i in idx
    ]
    cls = bag.sub("classifier")
    obj.weight = ttnn.from_torch(
        cls.tensor("weight").t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    obj.bias = ttnn.from_torch(
        cls.tensor("bias").reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    return obj
