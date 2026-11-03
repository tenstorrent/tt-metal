# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How the single-user speculative serving classes of generator_vllm drive their decoders, without vLLM: the direct
references (serve/mtp_ref.py, serve/dflash_ref.py) run in the tt-metal python, which has no vLLM, and must decode
exactly as the served classes do."""
import os

MTP_BLOCK = int(os.environ.get("QWEN36_MTP_BLOCK", "4"))  # tokens per decode step
MTP_DEPTH = int(os.environ.get("QWEN36_MTP_DEPTH", "4"))

DFLASH_BLOCK = int(os.environ.get("QWEN36_DFLASH_BLOCK", "8"))  # tokens per decode step
DFLASH_VERIFY = 8  # the resident DFlash2 verify rows (R)
# ResidentDFlash.max_prompt_tokens = max_pos - 2 * R - 64: built this much past max_model_len, every prompt vLLM
# admits fits (a prompt over the ceiling would fail the engine core)
DFLASH_POS_MARGIN = 2 * DFLASH_VERIFY + 64
