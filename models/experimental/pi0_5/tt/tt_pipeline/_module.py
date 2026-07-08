# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native, collapsed TTNNModule lifecycle for the standalone pi0.5 streamed-denoise port.

Distillation of ``tt_symbiote.core.module`` (602 L -> ~190 L). DROPPED relative to source:
  * the module-level ``get_tensor_run_implementation()`` eager global (the import-time
    coupling that would break the hardware-free import gate -- we never import run modes).
  * ``call``/``__call__`` dispatch indirection -> direct ``forward``.
  * ``DistributedConfig`` / ``set_distributed_tensor_config`` / ``tree_map`` /
    ``device_state`` / ``set_output_tensors_config`` (1x1 submeshes -> num_devices==1).
  * ``TTNNLayerStack``, ``deallocate_weights_after``.
KEPT: the from_torch -> preprocess -> move_weights lifecycle, the deferred-_TRACE_RUNNING
weight-prep guards, the ``_``-prefix-skipping recursion over ``__dict__`` children, the
direct-subclass ban, the StatefulTTNNModule own-``reset_trace_state`` requirement,
``run_on_devices`` (MESH_DEVICE-first -> live-introspection), DeviceArch (P150/P150x4/
P150x8/BHGLX) + MeshShapeToDeviceArch.

ZERO tt_symbiote imports. Imports with tt_symbiote NOT installed.
"""
from __future__ import annotations

import os
from enum import Enum
from functools import wraps


from ._trace import trace_enabled  # noqa: F401  (re-exported for convenience)

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"


def set_module_name_recursively(module, prefix=""):
    """Override children's module names based on this module's name."""
    for name, child in module.__dict__.items():
        if isinstance(child, TTNNModule):
            child._unique_name = f"{prefix}.{name}"
            child.override_children_module_names()
        elif isinstance(child, dict):
            for k, v in child.items():
                if isinstance(v, TTNNModule):
                    v._unique_name = f"{prefix}.{name}[{k}]"
                    v.override_children_module_names()
        elif isinstance(child, (list, tuple)):
            for i, v in enumerate(child):
                if isinstance(v, TTNNModule):
                    v._unique_name = f"{prefix}.{name}[{i}]"
                    v.override_children_module_names()


class TTNNModule:
    """Base for TTNN-accelerated modules. NOT directly subclassable: extend
    :class:`StatelessTTNNModule` or :class:`StatefulTTNNModule`."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if TTNNModule in cls.__bases__ and not cls.__dict__.get("_allow_direct_ttnnmodule_subclass", False):
            raise TypeError(
                f"{cls.__module__}.{cls.__qualname__} extends TTNNModule directly, which is "
                f"disallowed. Extend StatelessTTNNModule or StatefulTTNNModule instead."
            )

    def __init__(self):
        self._device = None
        self._preprocessed_weight = False
        self._weights_on_device = False
        self._fallback_torch_layer = None
        self._unique_name = None
        self._model_config = {}
        self._bypass_tensor_wrapping = False

    def set_model_config(self, model_config):
        self._model_config = model_config if model_config is not None else {}

    # No dispatch indirection: the streamed path calls stage.forward(x)/block(x) directly.
    def call(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def preprocess_weights(self):
        from ._trace import _TRACE_RUNNING  # deferred -> sees the live module-global

        if _TRACE_RUNNING:
            assert (
                self._preprocessed_weight
            ), f"Weights must be preprocessed for {self.module_name} before traced execution."
            return
        if not self._preprocessed_weight:
            self._preprocessed_weight = True
        else:
            return
        self.preprocess_weights_impl()

    def move_weights_to_device(self):
        from ._trace import _TRACE_RUNNING  # deferred -> sees the live module-global

        if _TRACE_RUNNING:
            assert self._weights_on_device, f"Weights must be on device for {self.module_name} before traced execution."
            return
        assert (
            self._preprocessed_weight
        ), f"Weights must be preprocessed for {self.module_name} before moving to device."
        assert self.device is not None, f"Device must be set for {self.module_name} before moving weights to device."
        if not self._weights_on_device:
            self._weights_on_device = True
        else:
            return
        self.move_weights_to_device_impl()

    def preprocess_weights_impl(self):
        for child in self.__dict__.values():
            if isinstance(child, TTNNModule):
                child.preprocess_weights()
        return self

    def move_weights_to_device_impl(self):
        for child in self.__dict__.values():
            if isinstance(child, TTNNModule):
                child.move_weights_to_device()
        return self

    def to_device(self, device):
        self._device = device
        return self

    def set_device_state(self, device_state=None):
        # 1x1 submeshes -> get_num_devices()==1, no DistributedConfig. No-op.
        return self

    @property
    def model_config(self):
        return self._model_config

    @property
    def device(self):
        return self._device

    @classmethod
    def from_torch(cls, torch_layer, *args, **kwargs):
        new_layer = cls(*args, **kwargs)
        new_layer._fallback_torch_layer = torch_layer
        return new_layer

    @property
    def module_name(self):
        if self._unique_name is None:
            self._unique_name = f"{self.__class__.__name__}_{id(self)}"
        return self._unique_name

    @property
    def torch_layer(self):
        return self._fallback_torch_layer

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    def reset_trace_state(self):
        """Hook the trace setup calls before each forward of a stateful module (see source
        contract). Base no-op; StatefulTTNNModule subclasses must supply their own."""

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        if memo is None:
            memo = set()
        if remove_duplicate:
            if self in memo:
                return
            memo.add(self)
        yield prefix, self
        for name, child in self.__dict__.items():
            if isinstance(child, TTNNModule):
                child_prefix = prefix + ("." if prefix else "") + name
                yield from child.named_modules(memo, child_prefix, remove_duplicate)

    def override_children_module_names(self):
        set_module_name_recursively(self, self.module_name)

    def named_children(self):
        for name, child in self.__dict__.items():
            if name in ["_fallback_torch_layer", "torch_layer"]:
                continue
            if isinstance(child, TTNNModule):
                yield name, child
            elif isinstance(child, dict):
                for k, v in child.items():
                    if isinstance(v, TTNNModule):
                        yield f"{name}[{k}]", v
            elif isinstance(child, (list, tuple)):
                for i, v in enumerate(child):
                    if isinstance(v, TTNNModule):
                        yield f"{name}[{i}]", v


class StatelessTTNNModule(TTNNModule):
    """Base for modules whose ``forward`` mutates NO persistent state under the trace
    double-run. Supplies a no-op ``reset_trace_state``."""

    _allow_direct_ttnnmodule_subclass = True

    def reset_trace_state(self):
        return None


class StatefulTTNNModule(TTNNModule):
    """Base for modules whose ``forward`` mutates persistent state under the trace
    double-run. Subclasses MUST implement ``reset_trace_state`` (enforced here)."""

    _allow_direct_ttnnmodule_subclass = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for klass in cls.__mro__:
            if klass is StatefulTTNNModule:
                raise TypeError(
                    f"{cls.__module__}.{cls.__qualname__} extends StatefulTTNNModule but does not "
                    f"implement reset_trace_state()."
                )
            if "reset_trace_state" in vars(klass):
                break


class DeviceArch(Enum):
    """Supported device architectures (port of the source enum members the port uses)."""

    N150 = "n150"
    N300 = "n300"
    T3K = "t3k_wh"
    TG = "gx_wh"
    P150 = "p150"
    P300 = "p300"
    P150x4 = "p150x4"
    P150x8 = "p150x8"
    BHGLX = "bhglx"


MeshShapeToDeviceArch = {
    "N150": DeviceArch.N150,
    "N300": DeviceArch.N300,
    "T3K": DeviceArch.T3K,
    "TG": DeviceArch.TG,
    "P150": DeviceArch.P150,
    "P300": DeviceArch.P300,
    "P150x4": DeviceArch.P150x4,
    "P150x8": DeviceArch.P150x8,
    "BHGLX": DeviceArch.BHGLX,
}


def run_on_devices(*allowed_archs: DeviceArch):
    """Decorator restricting a TTNNModule ``forward`` to specific device architectures.

    Resolution order: ``MESH_DEVICE`` env FIRST, else
    ``MeshShapeToDeviceArch.get(determine_device_name(self.device))`` -- keyed on the
    LIVE submesh (``self.device``), not the parent. Raises if the resolved arch is not
    in ``allowed_archs``. (Source ``mesh_shape`` enforcement is dropped -- call sites
    pass none.)
    """
    if not allowed_archs:
        raise ValueError("Must specify at least one allowed device architecture")
    allowed_set = frozenset(allowed_archs)

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, "device") or self.device is None:
                raise RuntimeError(f"{self.__class__.__name__}: No device set. ")
            mesh_device = MeshShapeToDeviceArch.get(os.environ.get("MESH_DEVICE"))
            if mesh_device is None:
                try:
                    from ._arch import determine_device_name

                    mesh_device = MeshShapeToDeviceArch.get(determine_device_name(self.device))
                except Exception:
                    mesh_device = None
            if mesh_device is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}: Unable to determine device architecture "
                    "from MESH_DEVICE environment variable or the live mesh device."
                )
            if mesh_device not in allowed_set:
                raise RuntimeError(
                    f"{self.__class__.__name__}: Device architecture {mesh_device} for device "
                    f"{self.device} not supported. Allowed architectures: {allowed_set}"
                )
            return func(self, *args, **kwargs)

        wrapper.__tt_allowed_archs__ = allowed_set
        return wrapper

    return decorator
