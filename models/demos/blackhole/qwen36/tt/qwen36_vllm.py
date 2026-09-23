# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim: the implementation moved to models.demos.qwen36.tt.qwen36_vllm.

The vLLM TT plugin registers TTQwen3_5ForConditionalGeneration against this old path
(vllm_tt_plugin/platform.py). Keep this re-export until the plugin points at the shared core.
"""

from models.demos.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

__all__ = ["Qwen36ForCausalLM"]
