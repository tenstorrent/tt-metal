# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python module base inheriting from C++ ModuleBase with auto-registration."""

import re
from typing import Any, Iterable, Iterator, Optional, Union, overload
from _ttml.modules import ModuleBase as CppModuleBase
from .parameter import Buffer, Parameter, TensorMetadata


def _match_policy(path: str, tp_plan: Optional[dict]) -> Any:
    """Return the Layout for the first matching pattern in tp_plan, or None."""
    if tp_plan is None:
        return None
    for pattern, layout in tp_plan.items():
        if re.search(pattern, path):
            return layout
    return None


class AbstractModuleBase(CppModuleBase):
    """Module base with PyTorch-like auto-registration via __setattr__.

    Subclasses create ``TensorMetadata`` Parameters in their constructor
    **before** calling ``super().__init__()``.  No materialization happens
    here — use ``TransformerBase`` as the root to materialize the full tree.
    """

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_buffers", {})
        self.create_name(self.__class__.__name__)

        # Retroactively register modules/params assigned before super().__init__().
        for name, value in list(self.__dict__.items()):
            if name.startswith("_"):
                continue
            if isinstance(value, CppModuleBase):
                self._bind_module(value, name)
            elif isinstance(value, Parameter) and not isinstance(value.tensor, TensorMetadata):
                self._bind_parameter(value.tensor, name)
            elif isinstance(value, Buffer):
                self._bind_buffer(value.tensor, name)

    # ------------------------------------------------------------------
    # Attribute registration
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

        # Skip if not initialized yet.
        if "_buffers" not in self.__dict__:
            return

        if isinstance(value, CppModuleBase):
            self._bind_module(value, name)
            return

        if isinstance(value, Parameter):
            # TensorMetadata Parameters are not bound to C++ until materialized.
            if not isinstance(value.tensor, TensorMetadata):
                self._bind_parameter(value.tensor, name)
            return

        if isinstance(value, Buffer):
            self._bind_buffer(value.tensor, name)
            return

        if hasattr(value, "get_value"):
            self._bind_parameter(value, name)

    def __delattr__(self, name: str) -> None:
        self._buffers.pop(name, None)
        object.__delattr__(self, name)

    # ------------------------------------------------------------------
    # C++ registration helpers
    # ------------------------------------------------------------------

    def _bind_module(self, module, name: str) -> None:
        """Register or override a child module by name."""
        try:
            self.register_module(module, name)
        except RuntimeError:
            self.override_module(module, name)

    def _bind_parameter(self, tensor, name: str) -> None:
        """Register or override a tensor parameter by name."""
        try:
            self.register_tensor(tensor, name)
        except RuntimeError:
            self.override_tensor(tensor, name)

    def _bind_buffer(self, buffer, name: str) -> None:
        """Register a buffer by name."""
        self._buffers[name] = buffer

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        """Yield (name, parameter) pairs."""
        for name, tensor in self.parameters().items():
            short = name.split("/", 1)[-1].replace("/", ".")
            yield (f"{prefix}.{short}" if prefix else short, tensor)

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        """Yield (name, module) pairs."""
        yield (prefix or "", self)
        for name, attr in self.__dict__.items():
            if isinstance(attr, CppModuleBase):
                child = f"{prefix}.{name}" if prefix else name
                if isinstance(attr, AbstractModuleBase):
                    yield from attr.named_modules(child)
                else:
                    yield (child, attr)

    def named_children(self) -> Iterator[tuple[str, Any]]:
        """Yield (name, module) pairs for direct children."""
        for name, attr in self.__dict__.items():
            if isinstance(attr, CppModuleBase):
                yield (name, attr)

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        """Yield (name, buffer) pairs."""
        for name, buf in self._buffers.items():
            yield (f"{prefix}.{name}" if prefix else name, buf)
        for name, attr in self.__dict__.items():
            if isinstance(attr, AbstractModuleBase):
                yield from attr.named_buffers(f"{prefix}.{name}" if prefix else name)

    def modules(self) -> Iterator[Any]:
        """Yield all modules."""
        return (m for _, m in self.named_modules())

    def children(self) -> Iterator[Any]:
        """Yield direct child modules."""
        return (c for _, c in self.named_children())

    def buffers(self) -> Iterator[Any]:
        """Yield all buffers."""
        return (b for _, b in self.named_buffers())

    def state_dict(self) -> dict[str, Any]:
        """Return state dictionary."""
        return {**dict(self.named_parameters()), **dict(self.named_buffers())}

    def parallelize(self, mesh_device, tp_axis: int, cp_axis: Optional[int] = None) -> None:
        """Hook for modules to adjust themselves for tensor/context parallelism.

        Override in subclasses that need TP/CP adjustments (e.g. head counts, rope).
        Default is no-op.
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke forward(). Subclasses implement forward()."""
        from ttml.distributed.debug import dispatch_trace

        if dispatch_trace.enabled:
            name = getattr(self, "get_name", lambda: self.__class__.__name__)()
            if callable(name):
                name = name()
            dispatch_trace.push_module(str(name))
        result = self.forward(*args, **kwargs)
        if dispatch_trace.enabled:
            dispatch_trace.pop_module()
        return result


class ParallelizationPlan:
    """Parallelization plan: style patterns + TP/CP axes.

    Bundles a ``{pattern: ParallelStyle}`` dict with the mesh axes
    so callers pass a single object to ``TransformerBase``.

    Usage::

        plan = ParallelizationPlan({
            r".*\\.(q_linear|kv_linear)": ColwiseParallel(),
            r".*\\.out_linear": RowwiseParallel(),
        }, tp_axis=1, cp_axis=0)
        model = Llama(config, mesh_device=mesh, parallelization_plan=plan)
    """

    __slots__ = ("styles", "tp_axis", "cp_axis")

    def __init__(self, styles: dict, tp_axis: int = 0, cp_axis: int | None = None) -> None:
        self.styles = styles
        self.tp_axis = tp_axis
        self.cp_axis = cp_axis

    def resolve(self, mesh_device) -> dict:
        """Convert styles to a ``{pattern: Layout}`` dict for materialization."""
        resolved = {}
        for pattern, style in self.styles.items():
            for param_name, layout in style.get_layouts(mesh_device, self.tp_axis).items():
                resolved[pattern + r"\." + param_name] = layout
        return resolved

    def __len__(self) -> int:
        return len(self.styles)


class TransformerBase(AbstractModuleBase):
    """Root-level transformer base that materializes the full module tree.

    Subclasses just create child modules in ``__init__`` — no
    ``super().__init__()`` call needed.  ``mesh_device`` and ``parallelization_plan``
    are popped from kwargs automatically, and ``__post_init__``
    materializes every ``TensorMetadata`` Parameter after ``__init__``
    returns.

    Usage::

        class Llama(TransformerBase):
            def __init__(self, config):
                self.fc = LinearLayer(...)
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "__init__" in cls.__dict__:
            import functools

            orig = cls.__dict__["__init__"]

            @functools.wraps(orig)
            def wrapped(self, *args, _orig=orig, **kw):
                mesh_device = kw.pop("mesh_device", None)
                tp_plan = kw.pop("parallelization_plan", None)
                on_device_init = kw.pop("on_device_init", False)
                object.__setattr__(self, "_mesh_device", mesh_device)
                object.__setattr__(self, "_parallelization_plan", tp_plan)
                object.__setattr__(self, "_on_device_init", on_device_init)
                _orig(self, *args, **kw)
                if not hasattr(self, "_buffers"):
                    AbstractModuleBase.__init__(self)
                self._materialize_tree()
                self._parallelize_modules()

            cls.__init__ = wrapped

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def _materialize_tree(self) -> None:
        """Walk the full module tree and materialize all TensorMetadata Parameters."""
        resolved = (
            self._parallelization_plan.resolve(self._mesh_device) if self._parallelization_plan is not None else None
        )
        for prefix, module in self.named_modules():
            if isinstance(module, AbstractModuleBase):
                self._materialize_module_params(module, prefix, resolved)

    def _materialize_module_params(self, module: AbstractModuleBase, prefix: str, resolved) -> None:
        """Materialize a single module's own TensorMetadata Parameters."""
        for name in list(vars(module)):
            attr = getattr(module, name, None)
            if isinstance(attr, Parameter) and isinstance(attr.tensor, TensorMetadata):
                full_path = f"{prefix}.{name}" if prefix else name
                self._materialize_param(module, name, attr, full_path, resolved)

    def _materialize_param(
        self, module: AbstractModuleBase, name: str, param: Parameter, full_path: str, resolved
    ) -> None:
        """Convert a single TensorMetadata Parameter into a device tensor."""
        metadata = param.tensor
        layout = _match_policy(full_path, resolved)

        tensor = metadata.init_fn(
            metadata.shape,
            layout=layout,
            mesh_device=self._mesh_device,
            on_device_init=self._on_device_init,
        )
        tensor.set_requires_grad(metadata.requires_grad)
        param.tensor = tensor
        module._bind_parameter(tensor, name)

    # ------------------------------------------------------------------
    # Parallelization
    # ------------------------------------------------------------------

    def _parallelize_modules(self) -> None:
        """Walk the module tree and apply parallelization after materialization."""
        if self._parallelization_plan is None and self._mesh_device is None:
            return

        tp_plan = self._parallelization_plan
        if tp_plan is None:
            return

        from ttml.distributed.style import ParallelStyle
        from ttml.distributed.mesh_runtime import MeshRuntime, set_runtime
        from ttml.distributed._register_ops import init_ops

        tp_axis = tp_plan.tp_axis
        cp_axis = tp_plan.cp_axis
        mesh_device = self._mesh_device

        runtime = MeshRuntime(mesh_device=mesh_device, tp_axis=tp_axis, cp_axis=cp_axis)
        init_ops()
        set_runtime(runtime)

        styles = tp_plan.styles

        def _apply_recursive(module: AbstractModuleBase, prefix: str) -> None:
            # Let the module adjust itself for TP/CP
            module.parallelize(mesh_device, tp_axis, cp_axis)

            # Match a style for forward hooks (broadcast/all_reduce)
            style = _match_style(prefix, styles)
            if style is not None:
                style._apply(module, mesh_device, tp_axis)
                return

            for name, child in module.named_children():
                if isinstance(child, AbstractModuleBase):
                    child_prefix = f"{prefix}.{name}" if prefix else name
                    _apply_recursive(child, child_prefix)

        def _match_style(name: str, plan: dict) -> Optional[ParallelStyle]:
            if name in plan:
                return plan[name]
            for pattern, style in plan.items():
                try:
                    if re.fullmatch(pattern, name):
                        return style
                except re.error:
                    continue
            return None

        _apply_recursive(self, "")


class ModuleList(AbstractModuleBase):
    """A list of modules with automatic registration (PyTorch-compatible).

    ModuleList can be indexed like a regular Python list and properly registers
    all contained modules so their parameters are tracked.

    Example:
        class MyModel(AbstractModuleBase):
            def __init__(self, num_layers):
                super().__init__()
                self.layers = ModuleList([LinearLayer(64, 64) for _ in range(num_layers)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
    """

    def __init__(
        self,
        modules: Optional[Iterable[CppModuleBase]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._modules_list: list[CppModuleBase] = []
        if modules is not None:
            for module in modules:
                self.append(module)

    def _get_abs_idx(self, idx: int) -> int:
        """Convert potentially negative index to absolute index."""
        if idx < 0:
            idx = len(self._modules_list) + idx
        if idx < 0 or idx >= len(self._modules_list):
            raise IndexError(f"index {idx} is out of range")
        return idx

    @overload
    def __getitem__(self, idx: int) -> CppModuleBase:
        pass

    @overload
    def __getitem__(self, idx: slice) -> "ModuleList":
        pass

    def __getitem__(self, idx: Union[int, slice]) -> Union[CppModuleBase, "ModuleList"]:
        """Get module(s) by index or slice."""
        if isinstance(idx, slice):
            return ModuleList(self._modules_list[idx])
        return self._modules_list[self._get_abs_idx(idx)]

    def __setitem__(self, idx: int, module: CppModuleBase) -> None:
        """Set module at index, updating registration."""
        abs_idx = self._get_abs_idx(idx)
        self._modules_list[abs_idx] = module
        # Override the existing registration
        self.override_module(module, str(abs_idx))

    def __delitem__(self, idx: int) -> None:
        """Delete module at index (rebuilds registrations)."""
        abs_idx = self._get_abs_idx(idx)
        del self._modules_list[abs_idx]
        # Need to rebuild registrations since indices shift
        self._rebuild_registrations()

    def __len__(self) -> int:
        """Return number of modules."""
        return len(self._modules_list)

    def __iter__(self) -> Iterator[CppModuleBase]:
        """Iterate over modules."""
        return iter(self._modules_list)

    def __contains__(self, module: CppModuleBase) -> bool:
        """Check if module is in list."""
        return module in self._modules_list

    def __repr__(self) -> str:
        """Return string representation."""
        lines = [f"{self.__class__.__name__}("]
        for idx, module in enumerate(self._modules_list):
            module_name = module.get_name() if hasattr(module, "get_name") else type(module).__name__
            lines.append(f"  ({idx}): {module_name}")
        lines.append(")")
        return "\n".join(lines)

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        """Yield (name, module) pairs for all modules, indexed by position."""
        yield (prefix or "", self)
        for idx, module in enumerate(self._modules_list):
            child = f"{prefix}.{idx}" if prefix else str(idx)
            if isinstance(module, AbstractModuleBase):
                yield from module.named_modules(child)
            else:
                yield (child, module)

    def named_children(self) -> Iterator[tuple[str, Any]]:
        """Yield (name, module) pairs for direct children, indexed by position."""
        for idx, module in enumerate(self._modules_list):
            yield (str(idx), module)

    def _rebuild_registrations(self) -> None:
        """Rebuild all module registrations after structural change."""
        # This is needed after delete/insert operations that shift indices
        # We need to clear and re-register all modules
        # Since the C++ side doesn't have a clear method, we work around by
        # creating a new internal state - but we can't easily clear m_named_modules
        # For now, we just re-register with override for existing indices
        # and register new ones. This works because we maintain index-based names.
        pass  # Current implementation doesn't support full rebuild
        # Users should prefer append() over insert()/del operations

    def append(self, module: CppModuleBase) -> "ModuleList":
        """Append a module to the list.

        Args:
            module: Module to append.

        Returns:
            self, for chaining.
        """
        idx = len(self._modules_list)
        self._modules_list.append(module)
        self.register_module(module, str(idx))
        return self

    def extend(self, modules: Iterable[CppModuleBase]) -> "ModuleList":
        """Extend list with modules from an iterable.

        Args:
            modules: Iterable of modules to add.

        Returns:
            self, for chaining.
        """
        for module in modules:
            self.append(module)
        return self

    def insert(self, idx: int, module: CppModuleBase) -> None:
        """Insert a module at given index.

        Note: This operation is O(n) as it requires re-registering modules
        after the insertion point. Prefer append() when possible.

        Args:
            idx: Index to insert at.
            module: Module to insert.
        """
        if idx < 0:
            idx = max(0, len(self._modules_list) + idx + 1)
        idx = min(idx, len(self._modules_list))

        # Insert into list
        self._modules_list.insert(idx, module)

        # Re-register all modules from idx onwards
        for i in range(idx, len(self._modules_list)):
            if i == idx:
                self.register_module(self._modules_list[i], str(i))
            else:
                # Need to override existing registration with new index
                try:
                    self.register_module(self._modules_list[i], str(i))
                except RuntimeError:
                    self.override_module(self._modules_list[i], str(i))

    def pop(self, idx: int = -1) -> CppModuleBase:
        """Remove and return module at index.

        Note: Due to C++ backend limitations, the removed module's registration
        remains but points to the old module. Prefer creating a new ModuleList
        if you need to remove modules frequently.

        Args:
            idx: Index of module to remove (default: -1, last module).

        Returns:
            The removed module.
        """
        abs_idx = self._get_abs_idx(idx)
        module = self._modules_list.pop(abs_idx)
        return module

    def __iadd__(self, modules: Iterable[CppModuleBase]) -> "ModuleList":
        """Support += operator for extending."""
        return self.extend(modules)


class ModuleDict(AbstractModuleBase):
    """A dictionary of modules with automatic registration (PyTorch-compatible).

    ModuleDict can be indexed like a regular Python dict and properly registers
    all contained modules so their parameters are tracked.

    Example:
        class MyModel(AbstractModuleBase):
            def __init__(self):
                super().__init__()
                self.layers = ModuleDict({
                    'encoder': LinearLayer(64, 128),
                    'decoder': LinearLayer(128, 64),
                })

            def forward(self, x, layer_name):
                return self.layers[layer_name](x)
    """

    def __init__(
        self,
        modules: Optional[Union[dict[str, CppModuleBase], Iterable[tuple[str, CppModuleBase]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._modules_dict: dict[str, CppModuleBase] = {}
        if modules is not None:
            if isinstance(modules, dict):
                for key, module in modules.items():
                    self[key] = module
            else:
                for key, module in modules:
                    self[key] = module

    def __getitem__(self, key: str) -> CppModuleBase:
        """Get module by key."""
        return self._modules_dict[key]

    def __setitem__(self, key: str, module: CppModuleBase) -> None:
        """Set module at key, updating registration."""
        if key in self._modules_dict:
            self.override_module(module, key)
        else:
            self.register_module(module, key)
        self._modules_dict[key] = module

    def __delitem__(self, key: str) -> None:
        """Delete module at key.

        Note: Due to C++ backend limitations, the registration remains.
        """
        del self._modules_dict[key]

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        """Yield (name, module) pairs for all modules, indexed by key."""
        yield (prefix or "", self)
        for key, module in self._modules_dict.items():
            child = f"{prefix}.{key}" if prefix else key
            if isinstance(module, AbstractModuleBase):
                yield from module.named_modules(child)
            else:
                yield (child, module)

    def named_children(self) -> Iterator[tuple[str, Any]]:
        """Yield (name, module) pairs for direct children, indexed by key."""
        yield from self._modules_dict.items()

    def __len__(self) -> int:
        """Return number of modules."""
        return len(self._modules_dict)

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self._modules_dict)

    def __contains__(self, key: str) -> bool:
        """Check if key is in dict."""
        return key in self._modules_dict

    def __repr__(self) -> str:
        """Return string representation."""
        lines = [f"{self.__class__.__name__}("]
        for key, module in self._modules_dict.items():
            module_name = module.get_name() if hasattr(module, "get_name") else type(module).__name__
            lines.append(f"  ({key}): {module_name}")
        lines.append(")")
        return "\n".join(lines)

    def keys(self) -> Iterator[str]:
        """Return iterator over keys."""
        return iter(self._modules_dict.keys())

    def values(self) -> Iterator[CppModuleBase]:
        """Return iterator over modules."""
        return iter(self._modules_dict.values())

    def items(self) -> Iterator[tuple[str, CppModuleBase]]:
        """Return iterator over (key, module) pairs."""
        return iter(self._modules_dict.items())

    def get(self, key: str, default: Optional[CppModuleBase] = None) -> Optional[CppModuleBase]:
        """Get module by key with optional default."""
        return self._modules_dict.get(key, default)

    def update(
        self,
        modules: Union[dict[str, CppModuleBase], Iterable[tuple[str, CppModuleBase]]],
    ) -> None:
        """Update dict with modules from another dict or iterable."""
        if isinstance(modules, dict):
            for key, module in modules.items():
                self[key] = module
        else:
            for key, module in modules:
                self[key] = module

    def pop(self, key: str, default: Optional[CppModuleBase] = None) -> Optional[CppModuleBase]:
        """Remove and return module at key.

        Note: Due to C++ backend limitations, the registration remains.
        """
        return self._modules_dict.pop(key, default)

    def clear(self) -> None:
        """Clear all modules.

        Note: Due to C++ backend limitations, registrations remain.
        """
        self._modules_dict.clear()
