# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA adapters between ttml's layers and HF PEFT's. NumPy only.

A ttml layer may hold several HF projections as row blocks of one weight (Llama's ``qkv_linear`` holds q, k
and v), with some blocks' rows permuted for RoPE (``W = P @ W_hf``). Each adapted projection has its own adapter
(``ttml.modules.LoraBlocks``), so the layer's ``(A, B)`` is exactly one PEFT adapter per adapted projection:

- ``A`` (``[n*r, in]``) stacks the ``A_i`` of the ``n`` adapted projections, in order;
- ``B`` (``[out, r]``) holds each ``B_i`` in its projection's rows, zero for a projection that isn't adapted,
  and a permuted block's rows are permuted back, since ``P @ W_hf + s * B @ A == P @ (W_hf + s * P^-1 @ B @ A)``.

The blocks come from the model's layout (``ttml.models.llama.hf_layout``).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

import numpy as np

from ttml.common.utils import permute_proj_rows, unpermute_proj_rows

from .errors import UnsupportedTrainingConfig

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


def lora_to_peft(
    a: np.ndarray, b: np.ndarray, blocks: Sequence[HfBlock], adapted: Sequence[bool]
) -> dict[str, np.ndarray]:
    """One ttml LoRA layer (``a`` ``[n*r, in]``, ``b`` ``[out, r]``) -> the PEFT tensors of the HF projections it
    adapts: the ``n`` of ``blocks`` that ``adapted`` flags."""
    rows = [block.shape[0] for block in blocks]
    rank = b.shape[1]
    if b.shape[0] != sum(rows):
        raise ValueError(f"lora_B has {b.shape[0]} rows; {[block.name for block in blocks]} have {sum(rows)}")
    if a.shape[0] != sum(adapted) * rank:
        raise ValueError(f"lora_A has {a.shape[0]} rows for {sum(adapted)} adapted blocks of rank {rank}")
    tensors = {}
    start, slot = 0, 0
    for block, count, has_adapter in zip(blocks, rows, adapted):
        b_block = b[start : start + count]
        start += count
        if not has_adapter:
            continue
        if block.rope_heads is not None:
            b_block = permute_proj_rows(b_block, block.rope_heads)
        key_a, key_b = peft_keys(block)
        tensors[key_a] = np.ascontiguousarray(a[slot * rank : (slot + 1) * rank])
        tensors[key_b] = np.ascontiguousarray(b_block)
        slot += 1
    return tensors


def peft_to_lora(
    tensors: Mapping[str, np.ndarray], blocks: Sequence[HfBlock], adapted: Sequence[bool], rank: int
) -> tuple[np.ndarray, np.ndarray]:
    """The PEFT tensors of the HF projections one ttml layer adapts -> that layer's ``(a, b)`` (float32).

    ``adapted`` flags the ``blocks`` that have an adapter, of rank ``rank``; the rows of the others are zero in ``b``.
    """
    in_features = blocks[0].shape[1]
    a = np.zeros((sum(adapted) * rank, in_features), dtype=np.float32)
    b = np.zeros((sum(block.shape[0] for block in blocks), rank), dtype=np.float32)
    start, slot = 0, 0
    for block, has_adapter in zip(blocks, adapted):
        count = block.shape[0]
        if has_adapter:
            key_a, key_b = peft_keys(block)
            a_block, b_block = np.asarray(tensors[key_a]), np.asarray(tensors[key_b])
            for key, value, shape in ((key_a, a_block, (rank, in_features)), (key_b, b_block, (count, rank))):
                if value.shape != shape:
                    raise UnsupportedTrainingConfig(f"checkpoint: {key} has shape {value.shape}, expected {shape}")
            if block.rope_heads is not None:
                b_block = unpermute_proj_rows(b_block, block.rope_heads)
            a[slot * rank : (slot + 1) * rank] = a_block
            b[start : start + count] = b_block
            slot += 1
        start += count
    return a, b


def check_peft_keys(tensors: Mapping[str, Any], layers: Iterable[tuple[Sequence[HfBlock], Sequence[bool]]]) -> None:
    """Raise unless ``tensors`` are exactly the adapters of ``layers``: the HF blocks of each LoRA layer, with the
    flags of the ones it adapts."""
    expected = {
        key
        for blocks, adapted in layers
        for block, has_adapter in zip(blocks, adapted)
        if has_adapter
        for key in peft_keys(block)
    }
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
    """The ``adapter_config.json`` ``PeftModel.save_pretrained`` writes for this adapter of ``base_model``."""
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
