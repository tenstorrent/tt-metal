# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Op routing for the Quasar port of the SDXL base UNet.

Every device op the model calls goes through this module as ``qsr.<name>(...)`` and is resolved,
at call time, to ``ttnn.experimental.quasar.<name>``. Ops that are not yet ported raise a
``NotImplementedError`` naming the op and its per-op unit test
(``models/demos/stable_diffusion_xl_base/quasar/tests/ops/test_<name>.py``), so the model code
never needs to change while ops are ported one by one.

Set ``SDXL_QSR_FALLBACK_MAINLINE=1`` to route not-yet-ported ops to the mainline ``ttnn.*``
implementation instead of raising (useful to bring up a module while its ops are in flight).

Host-side helpers (``ttnn.from_torch`` without a device, ``ttnn.to_torch``, ``ttnn.deallocate``,
``ttnn.create_sharded_memory_config``, the group-norm mask/weight helpers, ...) are not device
ops and keep being called as ``ttnn.*`` from the model.
"""

import os

import ttnn

QUASAR = ttnn.experimental.quasar

# model-facing name -> (path under ttnn.experimental.quasar, path under ttnn)
OPS = {
    # convolution / matmul
    "conv2d": ("conv2d", "conv2d"),
    "linear": ("linear", "linear"),
    "matmul": ("matmul", "matmul"),
    # elementwise binary
    "add": ("add", "add"),
    "add_": ("add_", "add_"),
    "multiply": ("multiply", "multiply"),
    "mul_": ("multiply_", "mul_"),
    "div": ("div", "div"),
    # elementwise unary
    "silu": ("silu", "silu"),
    "sin": ("sin", "sin"),
    "cos": ("cos", "cos"),
    "reciprocal": ("reciprocal", "reciprocal"),
    # normalization
    "group_norm": ("group_norm", "group_norm"),
    "layer_norm": ("layer_norm", "layer_norm"),
    # data movement / layout
    "concat": ("concat", "concat"),
    "reshape": ("reshape", "reshape"),
    "permute": ("permute", "permute"),
    "squeeze": ("squeeze", "squeeze"),
    "unsqueeze": ("unsqueeze", "unsqueeze"),
    "slice": ("slice", "slice"),
    "to_layout": ("to_layout", "to_layout"),
    "to_memory_config": ("to_memory_config", "to_memory_config"),
    "sharded_to_interleaved": ("sharded_to_interleaved", "sharded_to_interleaved"),
    "interleaved_to_sharded": ("interleaved_to_sharded", "interleaved_to_sharded"),
    "move": ("move", "move"),
    "to_device": ("to_device", "to_device"),
    "upsample": ("upsample", "upsample"),
    # transformer helpers
    "nlp_create_qkv_heads": ("nlp_create_qkv_heads", "experimental.nlp_create_qkv_heads"),
    "nlp_concat_heads": ("nlp_concat_heads", "experimental.nlp_concat_heads"),
    "scaled_dot_product_attention": (
        "transformer.scaled_dot_product_attention",
        "transformer.scaled_dot_product_attention",
    ),
}

TEST_DIR = "models/demos/stable_diffusion_xl_base/quasar/tests/ops"


def _lookup(root, dotted):
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def fallback_to_mainline():
    return os.environ.get("SDXL_QSR_FALLBACK_MAINLINE", "0") not in ("", "0", "false", "False")


def resolve(name):
    """Return the callable for model op ``name`` or raise NotImplementedError."""
    if name not in OPS:
        raise AttributeError(f"qsr: '{name}' is not an op the SDXL UNet port routes; add it to qsr.OPS")
    quasar_path, mainline_path = OPS[name]
    fn = _lookup(QUASAR, quasar_path)
    if fn is not None:
        return fn
    if fallback_to_mainline():
        fn = _lookup(ttnn, mainline_path)
        if fn is not None:
            return fn
    raise NotImplementedError(
        f"ttnn.experimental.quasar.{quasar_path} is not ported yet (needed by the SDXL UNet as qsr.{name}). "
        f"Unit test: {TEST_DIR}/test_{name}.py. "
        f"Set SDXL_QSR_FALLBACK_MAINLINE=1 to use mainline ttnn.{mainline_path} meanwhile."
    )


def available():
    """{op name: True if ttnn.experimental.quasar has it}."""
    return {name: _lookup(QUASAR, path[0]) is not None for name, path in OPS.items()}


def missing():
    return sorted(name for name, ok in available().items() if not ok)


class _LazyOp:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        return resolve(self.name)(*args, **kwargs)

    def __repr__(self):
        return f"<qsr op {self.name} -> ttnn.experimental.quasar.{OPS[self.name][0]}>"


def __getattr__(name):
    if name in OPS:
        return _LazyOp(name)
    raise AttributeError(f"module 'qsr' has no op '{name}'")


def from_torch(tensor, dtype=None, *, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, memory_config=None, **kwargs):
    """``ttnn.from_torch`` for the Quasar port.

    The tensor is always built on host (tilization happens on host, never through the mainline
    device tilize that is not Quasar-safe) and then moved with ``quasar.to_device`` when a device
    is given.
    """
    host = ttnn.from_torch(tensor, dtype, layout=layout, **kwargs)
    if device is None:
        return host
    return resolve("to_device")(host, device, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)
