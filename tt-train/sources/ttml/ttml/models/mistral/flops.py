# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Useful FLOPs per training step for Mistral under LoRA, the MFU numerator of ``doc/plan.md`` §3.

Counted the way ``torch.utils.flop_counter`` would count the reference with ``requires_grad`` mirroring
the freeze map: a frozen linear costs forward + dX, a LoRA adapter forward + dX + dW (3x its forward),
attention is the standard full-square ``4·H·D·T²`` forward and twice that backward, the frozen LM head
forward + dX, and no elementwise work. This is neither ``6·N·T`` (full fine-tuning) nor
``6 x trainable x T`` (adapter parameters only); see the plan for why.
"""

from __future__ import annotations

from .config import MistralConfig

__all__ = ["useful_flops_per_step", "useful_flops_per_token"]


def _linears(config: MistralConfig) -> dict[str, tuple[int, int]]:
    """(in, out) of every base linear of one block."""
    head_dim = config.attention_head_dim
    q, kv = config.num_attention_heads * head_dim, config.num_key_value_heads * head_dim
    inter = config.intermediate_size
    return {
        "qkv_linear": (config.hidden_size, q + 2 * kv),
        "out_linear": (q, config.hidden_size),
        "w_gate_up": (config.hidden_size, 2 * inter),
        "w2": (inter, config.hidden_size),
    }


def useful_flops_per_step(
    config: MistralConfig,
    tokens: int,
    seq_len: int,
    lora_rank: int = 0,
    lora_targets: tuple[str, ...] = ("qkv_linear", "out_linear", "w_gate_up", "w2"),
    num_layers: int | None = None,
) -> int:
    """FLOPs one optimiser step performs on ``tokens`` tokens of sequences ``seq_len`` long.

    ``num_layers`` truncates the stack (profiling depths); the default is the config's depth.
    """
    layers = config.num_hidden_layers if num_layers is None else num_layers
    linears = _linears(config)
    base = sum(i * o for i, o in linears.values())
    per_layer = 2 * 2 * tokens * base  # forward + dX of the frozen base weights
    if lora_rank:
        adapters = sum(lora_rank * (i + o) for name, (i, o) in linears.items() if name in lora_targets)
        per_layer += 3 * 2 * tokens * adapters  # forward, dX and dW of the adapters
    head_dim = config.attention_head_dim
    attention_fwd = 4 * config.num_attention_heads * head_dim * seq_len * tokens  # QK^T and PV, full square
    per_layer += 3 * attention_fwd  # forward + twice that backward
    head = 2 * 2 * tokens * config.vocab_size * config.hidden_size  # frozen LM head forward + dX
    return int(layers * per_layer + head)


def useful_flops_per_token(config: MistralConfig, seq_len: int, **kwargs) -> float:
    return useful_flops_per_step(config, seq_len, seq_len, **kwargs) / seq_len
