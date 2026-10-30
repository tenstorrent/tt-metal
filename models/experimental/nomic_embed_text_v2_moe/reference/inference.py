# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage 2 of 3: run the model.

    input_ids      (B, S) int64
    attention_mask (B, S) int64
      -> forward   -> last_hidden_state (B, S, 768) fp32, one vector per token

The only stage whose work is done by the model itself, so this file is thin: it calls the
backbone and normalizes the return type, because upstream returns an output object while the
vendored reference returns a bare tensor. embedding.py chains this between preprocessing and
postprocessing.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def forward(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Run the backbone to get one vector per token.

    B and S are unchanged; the token ids become 768-wide contextual vectors.

    Args:
        model: The vendored NomicBertModel or the upstream HF model.
        input_ids: (B, S) int64 token ids.
        attention_mask: (B, S) int64, 1 for real tokens and 0 for padding.

    Returns:
        torch.Tensor: (B, S, 768) fp32 last hidden state.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out if isinstance(out, torch.Tensor) else out.last_hidden_state
