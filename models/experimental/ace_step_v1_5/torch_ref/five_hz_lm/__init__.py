# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch-only **reference** copy of the ACE-Step 5 Hz LM stack for PCC / parity tests.

Use :class:`local_five_hz_llm.LocalFiveHzLMHandler` from this package when comparing against
``models.experimental.ace_step_v1_5.ttnn_impl.five_hz_lm`` with TTNN assist enabled. This tree mirrors
``ttnn_impl/five_hz_lm/`` (constants, paths, GPU config, etc.) but
``LocalFiveHzLMHandler.set_ttnn_logits_device`` is a no-op and experimental TTNN causal LM
loading is disabled so weights always load via ``transformers.AutoModelForCausalLM``.

See ``README.md`` in this folder and ``../README_LM_PCC_TORCH_REF.md`` for how to run the
tt-metal-style demo entrypoint against this handler.
"""

from .local_five_hz_llm import LocalFiveHzLMHandler

__all__ = ["LocalFiveHzLMHandler"]
