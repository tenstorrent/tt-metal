# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


class GenerativeTestModelBase:
    """Mixin/base for TT test models to be classified as *generative* by vLLM.

    vLLM's ModelRegistry determines whether a model is "text generation" capable
    via a structural interface check (see `vllm.model_executor.models.interfaces_base`).
    The minimum required pieces are:

    - `__init__` supports `vllm_config` (having `**kwargs` is sufficient)
    - `embed_input_ids(input_ids)`
    - `forward(input_ids=..., positions=...)`
    - `compute_logits(hidden_states)`

    TT test models typically expose TT-specific entrypoints
    (`initialize_vllm_model`, `prefill_forward`, `decode_forward`) used by the
    TT runner, and may never execute these methods at runtime. They exist so
    the registry classifies the model correctly and the rest of vLLM can route
    it through the "generate" runner path.
    """

    # The concrete model is expected to define `self.vocab_size`.

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # No-op embedding: return a float tensor shaped like an embedding.
        return input_ids.to(torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Produce dummy "hidden states".
        _ = positions
        _ = kwargs
        return self.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Produce dummy logits with the expected final vocab dimension.
        if hidden_states.ndim == 0:
            batch, seq = 1, 1
        elif hidden_states.ndim == 1:
            batch, seq = hidden_states.shape[0], 1
        else:
            batch, seq = hidden_states.shape[0], hidden_states.shape[1]

        vocab_size = int(getattr(self, "vocab_size"))
        return torch.zeros(
            batch,
            seq,
            vocab_size,
            dtype=torch.float32,
            device=hidden_states.device,
        )

    # Common functions needed to satisfy TT vLLM interface requirements
    # (dummy test models usually don't need to implement these unlike real models)

    def allocate_kv_cache(self, *args, **kwargs):
        return None

    def warmup_model_prefill(self, *args, **kwargs):
        pass

    def warmup_model_decode(self, *args, **kwargs):
        pass
