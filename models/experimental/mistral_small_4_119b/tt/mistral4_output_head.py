# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Final text norm + ``lm_head`` (prefill) for Mistral Small 4.

Weights are read from the hub map; **RMSNorm + linear run on host** in torch for HF parity.
``forward`` accepts TTNN hidden states ``[batch, 1, seq, hidden]`` and returns **CPU**
``[batch, seq, vocab]`` logits (avoids placing the full logits tensor on every mesh device).

Device ``lm_head`` / fused norm is a later optimization.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import ttnn

from models.common.auto_compose import to_torch_auto_compose, trim_torch_compose_to_reference_shape
from models.common.lightweightmodule import LightweightModule
from models.experimental.mistral_small_4_119b.constants import (
    EXPECTED_RMS_NORM_EPS,
    TEXT_MODEL_LM_HEAD_WEIGHT_KEY,
    TEXT_MODEL_NORM_WEIGHT_KEY,
)
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import _torch_for_ttnn_upload


class TtMistral4FinalNormLmHeadPrefill(LightweightModule):
    """
    ``model.norm`` + ``lm_head`` on host; input is TTNN decoder output ``[batch, 1, seq, hidden]``, output is
    torch logits ``[batch, seq, vocab]``.

    ``forward`` takes ``logical_batch`` (HF batch size) so that after :func:`~models.common.auto_compose.to_torch_auto_compose`
    replicated 2D mesh shards are trimmed to one logical batch (see :func:`~models.common.auto_compose.trim_torch_compose_to_reference_shape`).
    """

    def __init__(self, state_dict: dict, text_config):
        super().__init__()
        try:
            from transformers.models.mistral4.modeling_mistral4 import Mistral4RMSNorm
        except ImportError as exc:
            raise ImportError(
                "TtMistral4FinalNormLmHeadPrefill requires ``transformers`` with ``Mistral4RMSNorm``."
            ) from exc

        if TEXT_MODEL_NORM_WEIGHT_KEY not in state_dict or TEXT_MODEL_LM_HEAD_WEIGHT_KEY not in state_dict:
            raise KeyError(
                f"state_dict must contain {TEXT_MODEL_NORM_WEIGHT_KEY!r} and {TEXT_MODEL_LM_HEAD_WEIGHT_KEY!r}"
            )
        norm_w = _torch_for_ttnn_upload(state_dict[TEXT_MODEL_NORM_WEIGHT_KEY])
        self.lm_w = _torch_for_ttnn_upload(state_dict[TEXT_MODEL_LM_HEAD_WEIGHT_KEY])
        eps = float(getattr(text_config, "rms_norm_eps", EXPECTED_RMS_NORM_EPS))
        self.norm = Mistral4RMSNorm(int(text_config.hidden_size), eps=eps).eval()
        self.norm.weight.data = norm_w

    def forward(self, hidden_11SH: ttnn.Tensor, *, mesh_device, logical_batch: int) -> torch.Tensor:
        nt = to_torch_auto_compose(hidden_11SH, device=mesh_device)
        seq_len = int(nt.shape[2])
        hidden = int(nt.shape[3])
        nt = trim_torch_compose_to_reference_shape(nt, (int(logical_batch), 1, seq_len, hidden))
        h = nt[:, 0, :seq_len, :].contiguous().to(torch.bfloat16)
        normed = self.norm(h)
        return F.linear(normed, self.lm_w)
