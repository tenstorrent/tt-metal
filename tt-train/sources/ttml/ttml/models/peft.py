# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA adapters between ttml's layers and HF PEFT's. NumPy only.

A ttml layer may hold several HF projections as row blocks of one weight (Llama's ``qkv_linear`` holds q, k
and v), with some blocks' rows permuted for RoPE (``W = P @ W_hf``). Its adapter ``(A, B)`` is exactly one
PEFT adapter per projection:

- the projections share ``A``, since they share the layer's input;
- ``B`` splits into the same row blocks, and a permuted block's rows are permuted back, since
  ``P @ W_hf + s * B @ A == P @ (W_hf + s * P^-1 @ B @ A)``.

The way back needs the projections' ``A`` to be identical: independent adapters have no rank-r fused
equivalent. The blocks come from the model's layout (``ttml.models.llama.hf_layout``).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

import numpy as np

from ttml.common.utils import permute_proj_rows, unpermute_proj_rows

from .provider import UnsupportedTrainingConfig

PEFT_PREFIX = "base_model.model."


class HfBlock(Protocol):
    """One HF tensor held as a row block of a ttml weight, e.g. ``ttml.models.llama.HfTensor``."""

    name: str
    shape: tuple[int, ...]
    rope_heads: Optional[int]


def peft_keys(block: HfBlock) -> tuple[str, str]:
    """The PEFT names of the A and B weights adapting ``block``."""
    module = PEFT_PREFIX + block.name.removesuffix(".weight")
    return f"{module}.lora_A.weight", f"{module}.lora_B.weight"


def lora_to_peft(a: np.ndarray, b: np.ndarray, blocks: Sequence[HfBlock]) -> dict[str, np.ndarray]:
    """One ttml LoRA layer (``a`` ``[r, in]``, ``b`` ``[out, r]``) -> the PEFT tensors of the HF projections it holds."""
    rows = [block.shape[0] for block in blocks]
    if b.shape[0] != sum(rows):
        raise ValueError(f"lora_B has {b.shape[0]} rows; {[block.name for block in blocks]} have {sum(rows)}")
    tensors = {}
    start = 0
    for block, count in zip(blocks, rows):
        b_block = b[start : start + count]
        start += count
        if block.rope_heads is not None:
            b_block = permute_proj_rows(b_block, block.rope_heads)
        key_a, key_b = peft_keys(block)
        tensors[key_a] = np.ascontiguousarray(a)
        tensors[key_b] = np.ascontiguousarray(b_block)
    return tensors


def peft_to_lora(tensors: Mapping[str, np.ndarray], blocks: Sequence[HfBlock]) -> tuple[np.ndarray, np.ndarray]:
    """The PEFT tensors of the HF projections one ttml layer holds -> that layer's ``(a, b)``."""
    a_values, b_blocks = [], []
    for block in blocks:
        key_a, key_b = peft_keys(block)
        a_values.append(np.asarray(tensors[key_a]))
        b_block = np.asarray(tensors[key_b])
        if block.rope_heads is not None:
            b_block = unpermute_proj_rows(b_block, block.rope_heads)
        b_blocks.append(b_block)
    for block, a in zip(blocks[1:], a_values[1:]):
        if a.shape != a_values[0].shape or not np.array_equal(a, a_values[0]):
            raise UnsupportedTrainingConfig(
                f"checkpoint: {block.name} has another lora_A than {blocks[0].name}; ttml holds "
                f"{', '.join(b.name for b in blocks)} in one layer, which loads only adapters that share lora_A "
                "(as ttml saves them)"
            )
    return a_values[0], np.concatenate(b_blocks, axis=0)


def check_peft_keys(tensors: Mapping[str, Any], layers: Iterable[Sequence[HfBlock]]) -> None:
    """Raise unless ``tensors`` are exactly the adapters of ``layers`` (the HF blocks of each LoRA layer)."""
    expected = {key for blocks in layers for block in blocks for key in peft_keys(block)}
    missing, unexpected = sorted(expected - tensors.keys()), sorted(tensors.keys() - expected)
    if missing or unexpected:

        def first(names: list[str]) -> str:
            return f"{names[:4]}{'...' if len(names) > 4 else ''}"

        raise UnsupportedTrainingConfig(
            "checkpoint: the adapter's tensors don't match the model's LoRA layers "
            f"(missing {first(missing)}, unexpected {first(unexpected)})"
        )


def with_peft_prefix(tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``tensors`` named as ``save_pretrained`` names them: some tools leave off the ``base_model.model.`` prefix."""
    return {key if key.startswith(PEFT_PREFIX) else PEFT_PREFIX + key: value for key, value in tensors.items()}


def peft_adapter_config(
    base_model: str,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str],
    task_type: str,
) -> dict[str, Any]:
    """The ``adapter_config.json`` ``PeftModel.save_pretrained`` writes for this adapter."""
    return {
        "peft_type": "LORA",
        "task_type": task_type,
        "base_model_name_or_path": base_model,
        "revision": None,
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": sorted(target_modules),
        "bias": "none",
        "fan_in_fan_out": False,
        "use_rslora": False,
        "use_dora": False,
        "modules_to_save": None,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "inference_mode": True,
    }
