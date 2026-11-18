# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA adapters between ttml's layers and HF PEFT's. NumPy only.

A ttml layer may hold several HF projections as row blocks of one weight (Llama's ``qkv_linear`` holds q, k
and v), with the blocks' rows mapped from HF's by the layout's ``transform`` (Llama permutes q's and k's for RoPE:
``W = M @ W_hf``). Each adapted projection has its own adapter (``ttml.modules.LoraBlocks``), so the layer's
``(A, B)`` is exactly one PEFT adapter per adapted projection:

- ``A`` (``[n*r, in]``) stacks the ``A_i`` of the ``n`` adapted projections, in order;
- ``B`` (``[out, r]``) holds each ``B_i`` in its projection's rows, zero for a projection that isn't adapted,
  with its rows mapped as the weight's are, since ``M @ W_hf + s * B @ A == M @ (W_hf + s * M^-1 @ B @ A)``: the
  layout's ``inverse`` takes ``B`` to PEFT's, and its ``transform`` back. So a layout that maps more than rows
  can't hold an adapter.

The layouts come from the model (``ttml.models.llama.hf_layout``).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

import numpy as np

from .errors import UnsupportedTrainingConfig

PEFT_PREFIX = "base_model.model."


class HfBlock(Protocol):
    """One HF tensor held as a row block of a ttml weight, e.g. ``ttml.models.llama.HfTensor``."""

    name: str
    shape: tuple[int, ...]


class HfBlocks(Protocol):
    """The HF tensors a ttml weight holds as row blocks, in order, and how their rows map to the weight's, e.g.
    ``ttml.models.llama.HfTensors``."""

    tensors: Sequence[HfBlock]
    # The HF blocks -> the weight's, and back; None when the weight holds them as HF does.
    transform: Optional[Callable[..., list[np.ndarray]]]
    inverse: Optional[Callable[..., list[np.ndarray]]]


def peft_keys(block: HfBlock) -> tuple[str, str]:
    """The PEFT names of the A and B weights adapting ``block``."""
    module = PEFT_PREFIX + block.name.removesuffix(".weight")
    return f"{module}.lora_A.weight", f"{module}.lora_B.weight"


def lora_to_peft(a: np.ndarray, b: np.ndarray, layout: HfBlocks, adapted: Sequence[bool]) -> dict[str, np.ndarray]:
    """One ttml LoRA layer (``a`` ``[n*r, in]``, ``b`` ``[out, r]``) -> the PEFT tensors of the HF projections it
    adapts: the ``n`` of ``layout.tensors`` that ``adapted`` flags."""
    blocks = layout.tensors
    rows = [block.shape[0] for block in blocks]
    rank = b.shape[1]
    if b.shape[0] != sum(rows):
        raise ValueError(f"lora_B has {b.shape[0]} rows; {[block.name for block in blocks]} have {sum(rows)}")
    if a.shape[0] != sum(adapted) * rank:
        raise ValueError(f"lora_A has {a.shape[0]} rows for {sum(adapted)} adapted blocks of rank {rank}")
    b_blocks = np.split(b, np.cumsum(rows)[:-1])
    if layout.inverse is not None:
        b_blocks = layout.inverse(*b_blocks)
    tensors = {}
    slot = 0
    for block, b_block, has_adapter in zip(blocks, b_blocks, adapted):
        if not has_adapter:
            continue
        key_a, key_b = peft_keys(block)
        tensors[key_a] = np.ascontiguousarray(a[slot * rank : (slot + 1) * rank])
        tensors[key_b] = np.ascontiguousarray(b_block)
        slot += 1
    return tensors


def peft_to_lora(
    tensors: Mapping[str, np.ndarray], layout: HfBlocks, adapted: Sequence[bool], rank: int
) -> tuple[np.ndarray, np.ndarray]:
    """The PEFT tensors of the HF projections one ttml layer adapts -> that layer's ``(a, b)`` (float32).

    ``adapted`` flags the ``layout.tensors`` that have an adapter, of rank ``rank``; the rows of the others are zero
    in ``b``.
    """
    blocks = layout.tensors
    in_features = blocks[0].shape[1]
    a = np.zeros((sum(adapted) * rank, in_features), dtype=np.float32)
    b_blocks = [np.zeros((block.shape[0], rank), dtype=np.float32) for block in blocks]
    slot = 0
    for i, (block, has_adapter) in enumerate(zip(blocks, adapted)):
        if not has_adapter:
            continue
        key_a, key_b = peft_keys(block)
        a_block, b_block = np.asarray(tensors[key_a]), np.asarray(tensors[key_b])
        for key, value, shape in ((key_a, a_block, (rank, in_features)), (key_b, b_block, (block.shape[0], rank))):
            if value.shape != shape:
                raise UnsupportedTrainingConfig(f"checkpoint: {key} has shape {value.shape}, expected {shape}")
        a[slot * rank : (slot + 1) * rank] = a_block
        b_blocks[i] = b_block.astype(np.float32)
        slot += 1
    if layout.transform is not None:
        b_blocks = layout.transform(*b_blocks)
    return a, np.concatenate(b_blocks, axis=0)


def check_peft_keys(tensors: Mapping[str, Any], layers: Iterable[tuple[HfBlocks, Sequence[bool]]]) -> None:
    """Raise unless ``tensors`` are exactly the adapters of ``layers``: the HF layout of each LoRA layer, with the
    flags of the blocks it adapts."""
    expected = {
        key
        for layout, adapted in layers
        for block, has_adapter in zip(layout.tensors, adapted)
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
