# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN port of ACE-Step ``encoder.text_projector`` (no bias).

Torch reference::
    encoder_hidden_states = text_hidden_states @ text_projector.weight.T

Checkpoint key: ``encoder.text_projector.weight`` with shape ``[D_dec, D_text]``.
"""

from __future__ import annotations

import numpy as np

import ttnn

WEIGHT_KEY = "encoder.text_projector.weight"


def load_text_projector_weight_numpy(checkpoint_path: str, *, weight_key: str = WEIGHT_KEY) -> np.ndarray:
    """Load only the projector row from ``model.safetensors`` into float32 host numpy."""
    import torch
    from safetensors import safe_open

    with safe_open(checkpoint_path, framework="pt", device="cpu") as sf:
        if weight_key not in sf.keys():
            raise KeyError(f"{weight_key} not found in {checkpoint_path}")
        w = sf.get_tensor(weight_key)
        return w.detach().to(torch.float32).cpu().numpy()


class TtAceStepTextProjector:
    """
    Maps Qwen embedding hidden states to decoder conditioner width on device::

        [B, S, D_text] -> [B, S, D_dec]   (ROW_MAJOR activation, same convention as encoder path)
    """

    def __init__(
        self,
        *,
        device,
        weight_f32_numpy: np.ndarray,
        weights_dtype=None,
        weight_memory_config=None,
    ) -> None:
        weights_dtype = weights_dtype or getattr(ttnn, "bfloat16", None)
        if weights_dtype is None:
            raise RuntimeError("TTNN build missing bfloat16; pass weights_dtype explicitly.")

        w = np.asarray(weight_f32_numpy, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError(f"{WEIGHT_KEY} must be rank-2 [D_dec, D_text], got {w.shape}")

        self.d_dec = int(w.shape[0])
        self.d_text = int(w.shape[1])
        self.device = device
        self.weight_tt = ttnn.as_tensor(
            w,
            device=device,
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
        )

    def forward(self, text_hidden_f32_numpy: np.ndarray, *, activation_dtype):
        """Host float32 ``[B, S, D_text]`` → device-encoded states for ``AceStepV15TTNNPipeline``.

        Mirrors ``Torch.matmul(text_h, W.T)`` via ``ttnn.linear(..., transpose_b=True)``.
        """
        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)

        x = np.asarray(text_hidden_f32_numpy, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"text_hidden_states must be [B,S,D], got {x.shape}")
        if int(x.shape[-1]) != self.d_text:
            raise ValueError(f"Expected last dim D_text={self.d_text}, got {x.shape}")

        xh = ttnn.as_tensor(
            x,
            device=self.device,
            dtype=activation_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=mem,
        )
        out = ttnn.linear(xh, self.weight_tt, bias=None, transpose_b=True)
        try:
            ttnn.deallocate(xh)
        except Exception:
            pass
        return out

    def forward_from_hidden(self, hidden_b1sh: ttnn.Tensor, *, activation_dtype) -> ttnn.Tensor:
        """``hidden_b1sh`` ``[B,1,S,D_text]`` → ``[B,1,S,D_dec]`` (ROW-major activations). ``activation_dtype`` reserved."""
        _ = activation_dtype
        b, _one, s, d = (
            int(hidden_b1sh.shape[0]),
            int(hidden_b1sh.shape[1]),
            int(hidden_b1sh.shape[2]),
            int(hidden_b1sh.shape[3]),
        )
        if d != self.d_text:
            raise ValueError(f"Expected D_text={self.d_text}, got {d}")
        x = ttnn.to_layout(hidden_b1sh, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (b, s, self.d_text))
        out = ttnn.linear(x, self.weight_tt, bias=None, transpose_b=True)
        # ``[B,S,D_dec]`` ROW_MAJOR — matches prior host staging for ``AceStepV15TTNNPipeline``.
        return out
