# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..layers.module import Module
    from .events import PipelineEventCallback


_R_co = TypeVar("_R_co", covariant=True)


class _Pipeline(Protocol[_R_co]):
    def __call__(
        self,
        *,
        prompts: Sequence[str],
        negative_prompts: Sequence[str] | None,
        num_inference_steps: int,
        seed: int,
        traced: bool,
        on_event: PipelineEventCallback | None,
    ) -> _R_co:
        ...


class PipelineAPIMixin:
    def run_single_prompt(
        self: _Pipeline[_R_co],
        *,
        prompt: str,
        negative_prompt: str | None = None,
        num_inference_steps: int,
        seed: int = 0,
        on_event: PipelineEventCallback | None = None,
    ) -> _R_co:
        return self(
            prompts=[prompt],
            negative_prompts=[negative_prompt] if negative_prompt is not None else None,
            num_inference_steps=num_inference_steps,
            seed=seed,
            traced=True,
            on_event=on_event,
        )


class Pipeline:
    """Base of the tt_dit pipelines: owns whether the pipeline traces and whether it offloads weights.

    ``trace`` says that calls capture ttnn traces and replay them. ``offload_weights`` says that
    the pipeline evicts and reloads module weights between or within calls, because they do not
    all fit in device memory together. A captured trace replays the weight addresses it saw at
    capture, and a reload does not preserve them, so a pipeline cannot do both.

    ``offload_weights`` is about weight residency only. A pipeline that serves varying input
    sizes eagerly (MiniMax-H3 ``dynamic_length``) keeps its weights resident and does not set it.

    A subclass calls ``Pipeline.__init__`` before it builds modules or runs a warm-up, registers
    every eviction through ``register_coresident_exclusions`` or ``register_evictable``, and
    implements ``_run``. It does not define ``__call__``: that would skip the rule.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init_subclass__(**kwargs)
        if "__call__" in cls.__dict__:
            msg = f"{cls.__name__} must implement _run, not __call__: Pipeline.__call__ enforces the trace rule"
            raise TypeError(msg)

    def __init__(self, *, offload_weights: bool, trace: bool) -> None:
        self._offload_weights_requested = bool(offload_weights)
        self._evictable: weakref.WeakSet[Module] = weakref.WeakSet()
        self.trace = bool(trace)
        self._check_trace_rule()
        if not self._offload_weights_requested and not self.trace:
            # A resident pipeline could trace; operators look for this line when a serving config forgot to ask.
            logger.info(f"{type(self).__name__}: static mesh, running untraced (offload_weights=False, trace=False)")

    @property
    def offload_weights(self) -> bool:
        """Whether weights are evicted and reloaded: requested, or any module is registered as evictable."""
        # getattr: the rule must also hold on an instance built without __init__.
        return bool(getattr(self, "_offload_weights_requested", False)) or bool(getattr(self, "_evictable", ()))

    @offload_weights.setter
    def offload_weights(self, value: bool) -> None:
        self._offload_weights_requested = bool(value)

    def register_coresident_exclusions(self, module: Module, *others: Module | None) -> None:
        """Make loading ``module`` evict ``others``, which thereby become evictable."""
        present = [m for m in others if m is not None]
        # Checked first: a refused exclusion must not stay live on the module.
        if present and getattr(self, "trace", False):
            raise self._trace_rule_error()
        module.register_coresident_exclusions(*present)
        self.register_evictable(*present)

    def register_evictable(self, *modules: Module | None) -> None:
        """Declare modules whose weights the pipeline deallocates itself. ``None`` entries are skipped."""
        if not hasattr(self, "_evictable"):
            self._evictable = weakref.WeakSet()
        for module in modules:
            if module is not None:
                self._evictable.add(module)
        self._check_trace_rule()

    @staticmethod
    def _trace_rule_error() -> ValueError:
        return ValueError(
            "TDM-130: trace=True is unsupported with offload_weights=True: a captured trace replays "
            "the weight addresses it saw at capture, and offloading evicts and reloads weights. "
            "Construct the pipeline with trace=False, or keep every module resident "
            "(offload_weights=False) on a mesh with enough DRAM."
        )

    def _check_trace_rule(self) -> None:
        if getattr(self, "trace", False) and self.offload_weights:
            raise self._trace_rule_error()

    def _check_component_flags(self, kwargs: dict[str, Any]) -> None:
        """Per-component flags (``vae_traced``, ``encoder_traced``, ...) narrow ``trace``; they cannot widen it."""
        if getattr(self, "trace", False):
            return
        widened = sorted(name for name, value in kwargs.items() if name.endswith("_traced") and value)
        if widened:
            msg = f"{', '.join(widened)} cannot be set on a pipeline constructed with trace=False"
            raise ValueError(msg)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if "traced" in kwargs:
            msg = (
                "tracing is fixed when the pipeline is constructed (trace=...); a call takes no `traced` "
                "argument. For one eager run, use run_untraced()"
            )
            raise TypeError(msg)
        self._check_trace_rule()
        self._check_component_flags(kwargs)
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
