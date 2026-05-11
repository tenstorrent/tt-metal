# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 HF weight loading and re-exports.

The model implementation lives in ``ttml.models.qwen3``.  This module provides
``load_weights_from_hf`` for loading HuggingFace checkpoints, and re-exports
symbols consumed by ``model_qwen3_distributed.py`` and ``model_factory.py``.
"""

import torch
from tqdm import tqdm

import ttml

# Re-export shared components so existing callers (model_qwen3_distributed,
# model_factory, etc.) continue to work with ``from model_qwen3 import ...``
from ttml.models.qwen3 import Qwen3, Qwen3Config  # noqa: F401

from utils.tensor_utils import torch_to_ttml
from utils.param_utils import (  # noqa: F401 — re-exported for callers
    unpermute_proj_rows,
    unpermute_norm_weights,
    build_weight_mapping_single,
)

# Backwards-compat alias: callers that created Qwen3ForCausalLM now get Qwen3
Qwen3ForCausalLM = Qwen3


def linear(x, weight, bias=None):
    return ttml.ops.linear.linear(x, weight, bias)


# =====================================================================
# Weight loading from HuggingFace
# =====================================================================


def load_weights_from_hf(
    ttml_model,
    hf_state_dict: dict,
    config: Qwen3Config,
    tie_word_embeddings: bool = False,
    verbose: bool = False,
) -> None:
    """Load HF weights into single-device ttml Qwen3 model."""
    ttml_params = ttml_model.parameters()

    if verbose:
        print("\n  TTML parameter names:")
        for name in sorted(ttml_params.keys()):
            shape = ttml_params[name].shape()
            print(f"    {name}: {list(shape)}")

    any_key = next(iter(ttml_params))
    root_prefix = any_key.split("/")[0]

    mapping, transforms = build_weight_mapping_single(config, root_prefix, tie_word_embeddings)

    ttml_shapes = {name: list(ttml_params[name].shape()) for name in ttml_params}

    def _prepare_and_transfer(hf_name, ttml_name):
        """CPU prep + host-side conversion + device transfer (pipelined)."""
        if hf_name not in hf_state_dict:
            return None
        if ttml_name not in ttml_shapes:
            return None

        weight = hf_state_dict[hf_name].float()

        if hf_name in transforms:
            tr = transforms[hf_name]
            if tr[0] == "unpermute_proj":
                weight = unpermute_proj_rows(weight, num_heads=tr[1])
            elif tr[0] == "unpermute_norm":
                weight = unpermute_norm_weights(weight)

        ttml_shape = ttml_shapes[ttml_name]

        if weight.dim() == 2:
            rows, cols = weight.shape
            tgt_rows, tgt_cols = ttml_shape[2], ttml_shape[3]
            if rows != tgt_rows or cols != tgt_cols:
                padded = torch.zeros(tgt_rows, tgt_cols, dtype=weight.dtype)
                padded[: min(rows, tgt_rows), : min(cols, tgt_cols)] = weight[
                    : min(rows, tgt_rows), : min(cols, tgt_cols)
                ]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0)
        elif weight.dim() == 1:
            dim = weight.shape[0]
            tgt_dim = ttml_shape[-1]
            if dim != tgt_dim:
                padded = torch.zeros(tgt_dim, dtype=weight.dtype)
                padded[: min(dim, tgt_dim)] = weight[: min(dim, tgt_dim)]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return torch_to_ttml(weight)

    from concurrent.futures import ThreadPoolExecutor

    items = list(mapping.items())
    loaded = 0
    skipped = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            (hf_name, ttml_name, pool.submit(_prepare_and_transfer, hf_name, ttml_name)) for hf_name, ttml_name in items
        ]

        for hf_name, ttml_name, future in tqdm(
            futures,
            total=len(items),
            desc="  Loading weights",
            unit="w",
        ):
            new_tensor = future.result()
            if new_tensor is None:
                if ttml_name not in ttml_shapes:
                    print(f"  WARNING: ttml param '{ttml_name}' not found for HF '{hf_name}'")
                skipped.append(hf_name)
                continue
            ttml_params[ttml_name].assign(new_tensor)
            loaded += 1

    print(f"\n  Weight loading: {loaded} loaded, {len(skipped)} skipped")
    if skipped:
        print(f"  Skipped: {skipped}")
