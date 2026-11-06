# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Readiness ABC when installed, with its interface for standalone serving."""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Sequence

import torch

try:
    from readiness_check.contract import Generator
except ModuleNotFoundError as error:
    if error.name != "readiness_check":
        raise

    # Standalone deployments omit the readiness runner. Keep its abstract
    # interface without suppressing failures in an installed readiness package.
    NextInputFn = Callable[[int, int], int]

    class Generator(ABC):
        tokenizer: Any = None

        @abstractmethod
        def prefill_forward(
            self,
            tokens: torch.Tensor,
            *,
            page_table: torch.Tensor,
            kv_cache: Any,
            prompt_lens: List[int],
            start_pos: Optional[torch.Tensor] = None,
            return_all_logits: bool = False,
            **kwargs: Any,
        ) -> torch.Tensor:
            pass

        @abstractmethod
        def decode_forward(
            self,
            tokens: torch.Tensor,
            start_pos: torch.Tensor,
            *,
            page_table: torch.Tensor,
            kv_cache: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            pass

        @abstractmethod
        def prefill_logits(self, prompt_token_ids: List[int]) -> torch.Tensor:
            pass

        def prefill_logits_continuation(
            self,
            prompt_token_ids: List[int],
            start: int,
            *,
            positions: Optional[Sequence[int]] = None,
            prefill_prefix: bool = True,
        ) -> torch.Tensor:
            raise NotImplementedError("prefill_logits_continuation is not implemented for this generator")

        @abstractmethod
        def generate(
            self,
            prompt_token_ids: List[int],
            max_new_tokens: int,
            *,
            next_input: Optional[NextInputFn] = None,
            enable_trace: bool = True,
            **kwargs: Any,
        ) -> List[int]:
            pass

        @abstractmethod
        def reset(self) -> None:
            pass
