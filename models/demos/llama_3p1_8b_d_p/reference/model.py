# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Torch reference modules for Llama-3.1-8B prefill.

Graded against by the unit tests under ``tests/unit/``. Only the dense MLP lands here with
tt-blaze#4140; the rest of the reference stack (RMSNorm, GQA attention, RoPE, decoder layer, and the
golden-KV harness) arrives with tt-blaze#4147.

The module layout deliberately mirrors HuggingFace ``LlamaMLP`` rather than this repo's
``TorchExpert``: the parameters are ``nn.Linear`` submodules, so a state dict loaded straight out of
the Llama-3.1-8B checkpoint (``model.layers.N.mlp.*.weight``) drops in with no key translation, and
#4147 can grade the whole layer against HF without a second weight convention.

Imports torch, so nothing on the adapter import path may import this file — see the import-light
contract asserted by ``tests/unit/test_scaffold.py``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig


class Llama31MLP(nn.Module):
    """Llama-3.1-8B dense SwiGLU MLP: ``down(silu(gate(x)) * up(x))``.

    No biases and no clamp on the gate. Both are load-bearing differences from neighbouring models
    in this fleet and are asserted by the unit test rather than left to the reader: gpt-oss carries
    per-projection biases and clamps the gate at 7.0 (``SwiGLUConfigGPTOSS``), while Llama's SwiGLU
    is unclamped (``Llama31_8BConfig.SWIGLU_LIMIT is None``). Grading the device MLP against a
    clamped or biased reference would pass at small activation magnitudes and diverge exactly where
    the clamp starts to bind.
    """

    def __init__(
        self,
        emb_dim: int = Llama31_8BConfig.EMB_SIZE,
        hidden_dim: int = Llama31_8BConfig.INTERMEDIATE_SIZE,
    ):
        super().__init__()
        assert Llama31_8BConfig.SWIGLU_LIMIT is None, (
            "Llama-3.1-8B's SwiGLU is unclamped; a non-None SWIGLU_LIMIT means the dim SSOT changed "
            "and this reference no longer matches the model it grades."
        )
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # HF ``nn.Linear`` weights are (out_features, in_features); the device side transposes to
        # (in, out) for ``ttnn.matmul``. Keeping HF's orientation here means the test feeds the
        # device module the checkpoint's own layout.
        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[..., emb_dim]`` -> ``[..., emb_dim]``."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def torch_weights(self) -> dict:
        """The three projections in HF (out, in) orientation, keyed as the device module expects."""
        return {
            "gate_proj": self.gate_proj.weight.data,
            "up_proj": self.up_proj.weight.data,
            "down_proj": self.down_proj.weight.data,
        }
