# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Protocol, TypeVar

from ..utils.tracing import Tracer
from .residency import Residency

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    @cached_property
    def residency(self) -> Residency:
        return Residency(self)

    def release_traces(self) -> None:
        """Release every captured trace.

        Override only to also release traces that are not `Tracer`s; releasing less than every `Tracer` is unsound,
        since a load after any capture can land in any live trace's scratch.
        """
        Tracer.release_all()

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
