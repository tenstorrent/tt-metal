# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import functools
import inspect
import weakref
from typing import TYPE_CHECKING, Any

from loguru import logger

import ttnn

from ..utils import tracing

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ..layers.module import Module
    from .events import PipelineEventCallback


# The part of the eviction warning that tests look for in a log.
TRACE_RELEASE_NOTICE = "is evicting. Releasing traces"


def owns_captures(method: Callable[..., Any]) -> Callable[..., Any]:
    """For a pipeline method that captures traces outside a call, such as a warm-up: the pipeline owns them."""

    @functools.wraps(method)
    def wrapper(self: Pipeline, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        with self._owning_captures():  # noqa: SLF001
            return method(self, *args, **kwargs)

    return wrapper


class Pipeline:
    """Base of the tt_dit pipelines: owns whether the pipeline traces and whether it offloads weights.

    ``trace`` says that calls capture ttnn traces and replay them. ``offload_weights`` says that
    the pipeline evicts and reloads module weights between or within calls, because they do not
    all fit in device memory together. A captured trace replays the weight addresses it saw at
    capture, and a reload does not preserve them. So before loading one module evicts another, a
    pipeline that traces releases its traces, with a warning, and captures them again on its next
    traced calls: one release and one capture for every eviction.

    ``offload_weights`` is about weight residency only. A pipeline that serves varying input
    sizes eagerly (MiniMax-H3 ``dynamic_length``) keeps its weights resident and does not set it.

    A subclass calls ``Pipeline.__init__`` before it builds modules or runs a warm-up, registers
    every eviction through ``register_coresident_exclusions``, and implements ``_run``. It does
    not define ``__call__``: the pipeline would then not own the traces its calls capture.

    Tracing is decided at construction. A call that still passes the former per-call ``traced``
    argument runs as the pipeline was constructed, whatever the value, and logs a warning.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init_subclass__(**kwargs)
        if "__call__" in cls.__dict__:
            msg = f"{cls.__name__} must implement _run, not __call__: Pipeline.__call__ owns the traces a call captures"
            raise TypeError(msg)
        # The base only sees the trace flags a caller passes, so `_run` must not default one to on.
        for name, parameter in inspect.signature(cls._run).parameters.items():
            if (name == "traced" or name.endswith("_traced")) and parameter.default is True:
                msg = f"{cls.__name__}._run must not default {name} to True: tracing follows the pipeline's trace"
                raise TypeError(msg)

    def __init__(self, *, offload_weights: bool, trace: bool) -> None:
        self._offload_weights_requested = bool(offload_weights)
        self._evictable: weakref.WeakSet[Module] = weakref.WeakSet()
        self.trace = bool(trace)
        if not self._offload_weights_requested and not self.trace:
            # A resident pipeline could trace; operators look for this line when a serving config forgot to ask.
            logger.info(f"{type(self).__name__}: static mesh, running untraced (offload_weights=False, trace=False)")

    @property
    def offload_weights(self) -> bool:
        """Whether weights are evicted and reloaded: requested, or any module is registered as evictable."""
        # getattr: an instance built without __init__ has neither attribute.
        return bool(getattr(self, "_offload_weights_requested", False)) or bool(getattr(self, "_evictable", ()))

    @offload_weights.setter
    def offload_weights(self, value: bool) -> None:
        self._offload_weights_requested = bool(value)

    def register_coresident_exclusions(self, module: Module, *others: Module | None) -> None:
        """Make loading ``module`` evict ``others``, which thereby become evictable. ``None`` entries are skipped."""
        present = [m for m in others if m is not None]
        module._register_coresident_exclusions(*present)  # noqa: SLF001
        # hasattr: an instance built without __init__ has no set yet.
        if not hasattr(self, "_evictable"):
            self._evictable = weakref.WeakSet()
        self._evictable.update(present)
        # Weak: a module must not keep its pipeline alive.
        module._exclusion_evictor = weakref.WeakMethod(self._evict)  # noqa: SLF001

    def _evict(self, module: Module) -> None:
        """Free the weights of ``module``, an exclusion of a module that is being loaded.

        A captured trace replays the weight addresses it saw. So on a pipeline that traces, the
        pipeline's traces are released before the weights go, with a warning; each one is captured
        again on its next traced call. A module that holds no weights frees nothing, and no trace is
        released for it.
        """
        # `is_loaded` can be false on a module that holds weights (a load that raised part way). Then
        # nothing is released here, and the trace that read those weights refuses to replay instead.
        # An untraced pipeline evicts exactly as before; a traced one does the release work only when
        # this eviction frees weights and the pipeline holds a trace.
        if (
            getattr(self, "trace", False)
            and module.is_loaded()
            and any(tracer.trace_captured for tracer in self._owned_traces())
        ):
            before = tracing.captured_trace_count()
            self.release_traces()
            # A subclass releases the traces it knows of; this takes the rest of what the pipeline owns.
            self._release_owned_traces()
            logger.warning(
                f"{type(self).__name__}: module {type(module).__name__} {TRACE_RELEASE_NOTICE} "
                f"({before - tracing.captured_trace_count()} released)."
            )
        module.deallocate_weights()

    def release_traces(self) -> None:
        """Release the traces this pipeline captured. A subclass that keeps state beside its traces resets it too."""
        self._release_owned_traces()

    def _owned_traces(self) -> weakref.WeakSet[tracing.Tracer]:
        # setdefault: an instance built without __init__ has no set yet.
        return self.__dict__.setdefault("_owned_tracers", weakref.WeakSet())

    def _release_owned_traces(self) -> None:
        for tracer in tuple(self._owned_traces()):
            tracer.release_trace()

    @contextlib.contextmanager
    def _owning_captures(self) -> Iterator[None]:
        """Every trace captured inside the block is this pipeline's, whichever component holds its tracer."""
        owned = self._owned_traces()
        previous = tracing.current_capture_owner()
        # One owner at a time on a thread, for a whole call: a second pipeline starting inside it is
        # refused, not given the slot. A warm-up of the same pipeline may call it again.
        if previous is not None and previous is not owned:
            msg = f"{type(self).__name__} cannot run inside another pipeline's call: captures have one owner"
            raise RuntimeError(msg)
        tracing._capture_owner.traces = owned  # noqa: SLF001
        try:
            yield
        finally:
            tracing._capture_owner.traces = previous  # noqa: SLF001

    def _check_component_flags(self, kwargs: dict[str, Any]) -> None:
        """Per-component flags (``vae_traced``, ``encoder_traced``, ...) narrow ``trace``; they cannot widen it."""
        if getattr(self, "trace", False):
            return
        widened = sorted(name for name, value in kwargs.items() if name.endswith("_traced") and value)
        if widened:
            msg = f"{', '.join(widened)} cannot be set on a pipeline constructed with trace=False"
            raise ValueError(msg)

    def warn_if_no_trace_region(self, mesh_device: ttnn.MeshDevice, *, by_default: bool) -> None:
        """Log when the pipeline traces on a mesh opened without a trace region.

        ttnn then takes the trace buffers from DRAM, which works but shrinks what the model has.
        ``by_default`` says the caller did not ask for tracing: the pipeline's default turned it on.
        """
        # Device-free tests construct pipelines with no device, or a stand-in for one.
        if not self.trace or not isinstance(mesh_device, ttnn.MeshDevice):
            return
        # Advice only: a mesh whose trace region cannot be read must still construct.
        try:
            view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.TRACE)
            reserved = view.total_bytes_per_bank * view.num_banks
        except Exception as error:  # noqa: BLE001
            logger.debug(f"{type(self).__name__}: could not read the trace region size: {error!r}")
            return
        if reserved > 0:
            return
        name = type(self).__name__
        if by_default:
            logger.warning(
                f"{name}: tracing is on by default on this mesh and no trace region is reserved, so trace "
                "buffers are taken from DRAM. Pass trace=False to run eager, or open the mesh with a "
                "trace_region_size."
            )
        else:
            logger.warning(f"{name}: trace=True with no trace region reserved; trace buffers are taken from DRAM.")

    def _drop_legacy_traced(self, kwargs: dict[str, Any]) -> None:
        """Remove a per-call ``traced`` from ``kwargs``; it never changes how the call runs."""
        if "traced" not in kwargs:
            return
        value = kwargs.pop("traced")
        trace = bool(getattr(self, "trace", False))
        name = type(self).__name__
        # Two flags: a disagreement must be reported even after an agreeing call was.
        if value is None or bool(value) == trace:
            if not getattr(self, "_legacy_traced_warned", False):
                self._legacy_traced_warned = True
                logger.warning(
                    f"{name}: the per-call `traced` argument is deprecated and ignored; it will be removed "
                    "after tt-inference-server passes trace at construction (TDM-130 follow-up). Tracing is "
                    "set at construction (trace=...)."
                )
            return
        if getattr(self, "_legacy_traced_disagreement_warned", False):
            return
        self._legacy_traced_disagreement_warned = True
        if trace:
            logger.warning(
                f"{name}: traced=False IGNORED: the pipeline was constructed with trace=True, so this call "
                "runs traced and its trace buffers stay in use (in DRAM when the mesh has no trace region). "
                "To run eager, construct the pipeline with trace=False. run_untraced() gives one eager call "
                "where the pipeline supports it, on most pipelines only before the first traced call."
            )
        else:
            cause = " because it offloads weights (TDM-130)" if self.offload_weights else ""
            logger.warning(
                f"{name}: traced=True IGNORED: the pipeline was constructed with trace=False and runs "
                f"eager{cause}. To trace, construct the pipeline with trace=True."
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        self._drop_legacy_traced(kwargs)
        self._check_component_flags(kwargs)
        with self._owning_captures():
            return self._run(*args, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """One generation. Runs traced when ``self.trace``, unless given ``traced=False``."""
        raise NotImplementedError

    def run_untraced(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """One eager generation, whatever ``trace`` is: a warm-up, or the reference of a traced run.

        Where the eager path runs through the pipeline's ``Tracer`` objects, this is only possible
        before they capture: afterwards the ``Tracer`` raises ``RuntimeError``. Where the eager path
        bypasses them, it works at any time. A pipeline whose traces are fixed at construction
        overrides this to raise.
        """
        self._check_component_flags(kwargs)
        with self._owning_captures():
            return self._run(*args, traced=False, **kwargs)

    def run_single_prompt(
        self,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        num_inference_steps: int,
        seed: int = 0,
        on_event: PipelineEventCallback | None = None,
    ) -> Any:  # noqa: ANN401
        return self(
            prompts=[prompt],
            negative_prompts=[negative_prompt] if negative_prompt is not None else None,
            num_inference_steps=num_inference_steps,
            seed=seed,
            on_event=on_event,
        )
