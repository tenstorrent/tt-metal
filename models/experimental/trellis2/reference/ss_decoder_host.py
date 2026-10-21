# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side pieces of the TRELLIS.2 sparse-structure decoder (`ss_dec_conv3d_16l8`, a dense 3D CNN, 16^3 -> 64^3).

The reference is `trellis2/models/sparse_structure_vae.py` `SparseStructureDecoder` (microsoft/TRELLIS.2 75fbf018) with
the TRELLIS-image-large config: latent 8, channels [512, 128, 32], 2 res blocks per level and 2 in the middle,
channel LayerNorm, out 1 channel; the pipeline keeps `logits > 0`, max-pools 64^3 -> 32^3 and takes `argwhere`.

The TT decoder works channels-last (N, D, H, W, C), the layout `ttnn.experimental.conv3d` wants. Two steps differ from
the reference's code but not its math, and `functional_forward` restates both so they can be checked on the CPU:
- `pixel_shuffle_3d` (channel c = c_ * 8 + i * 4 + j * 2 + k, then interleave) becomes a reorder of the upsample conv's
  output channels to (i, j, k, c_) order plus two rank-4 permutes (`shuffle_ndhwc`);
- the channel LayerNorm is a LayerNorm over the last dim of the token-major tensor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

LATENT_CHANNELS = 8
CHANNELS = (512, 128, 32)
NUM_RES_BLOCKS = 2
NUM_RES_BLOCKS_MIDDLE = 2
OUT_CHANNELS = 1
NORM_EPS = 1e-5
SCALE = 2
CKPT = "TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(path)


def block_layout() -> list[tuple[str, int, int]]:
    """(kind, in_channels, out_channels) of `blocks.<i>` in checkpoint order: 'res' or 'up'."""
    layout = []
    for i, ch in enumerate(CHANNELS):
        layout += [("res", ch, ch)] * NUM_RES_BLOCKS
        if i < len(CHANNELS) - 1:
            layout.append(("up", ch, CHANNELS[i + 1]))
    return layout


def shuffle_weight_order(w: torch.Tensor, b: torch.Tensor, s: int = SCALE) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder the upsample conv's output channels from (c_, i, j, k) to (i, j, k, c_) so that a channels-last output
    [.., s^3 * C_] reads as [.., i, j, k, c_]."""
    c = w.shape[0] // s**3
    w = w.reshape(c, s**3, *w.shape[1:]).transpose(0, 1).reshape(s**3 * c, *w.shape[1:])
    b = b.reshape(c, s**3).transpose(0, 1).reshape(-1)
    return w, b


def shuffle_ndhwc(x: torch.Tensor, s: int = SCALE) -> torch.Tensor:
    """[N, D, H, W, s^3 * C] with channels in (i, j, k, c) order -> [N, sD, sH, sW, C], as two rank-4 permutes:
    x[d, h, w, i, j, k, c] -> [d, h, i, j, (w k), c] -> [d, i, h, j, (w k), c]."""
    n, d, h, w, cc = x.shape
    c = cc // s**3
    x = x.reshape(n * d * h, w, s * s, s * c).permute(0, 2, 1, 3)  # [dh, (i j), w, (k c)]
    x = x.reshape(n * d, h, s, s * w * s * c).permute(0, 2, 1, 3)  # [d, i, h, (j w k c)]
    return x.reshape(n, d * s, h * s, w * s, c)


def _conv(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Conv3d on channels-last x with symmetric zero padding (k // 2), as the TT path pads D then convolves."""
    y = F.conv3d(x.permute(0, 4, 1, 2, 3), w, b, padding=w.shape[-1] // 2)
    return y.permute(0, 2, 3, 4, 1)


def _norm_silu(x: torch.Tensor, s: dict[str, torch.Tensor], p: str) -> torch.Tensor:
    return F.silu(F.layer_norm(x, (x.shape[-1],), s[p + ".weight"], s[p + ".bias"], eps=NORM_EPS))


def _res(x: torch.Tensor, s: dict[str, torch.Tensor], p: str) -> torch.Tensor:
    h = _conv(_norm_silu(x, s, p + "norm1"), s[p + "conv1.weight"], s[p + "conv1.bias"])
    h = _conv(_norm_silu(h, s, p + "norm2"), s[p + "conv2.weight"], s[p + "conv2.bias"])
    return x + h


def functional_forward(state: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
    """Torch mirror of the TT decoder in fp32: z [1, 8, 16, 16, 16] -> logits [1, 1, 64, 64, 64] (reference layout)."""
    s = {k: v.float() for k, v in state.items()}
    h = _conv(z.float().permute(0, 2, 3, 4, 1), s["input_layer.weight"], s["input_layer.bias"])
    for i in range(NUM_RES_BLOCKS_MIDDLE):
        h = _res(h, s, f"middle_block.{i}.")
    for i, (kind, _, _) in enumerate(block_layout()):
        if kind == "res":
            h = _res(h, s, f"blocks.{i}.")
        else:
            w, b = shuffle_weight_order(s[f"blocks.{i}.conv.weight"], s[f"blocks.{i}.conv.bias"])
            h = shuffle_ndhwc(_conv(h, w, b))
    h = _conv(_norm_silu(h, s, "out_layer.0"), s["out_layer.2.weight"], s["out_layer.2.bias"])
    return h.permute(0, 4, 1, 2, 3)


def occupancy_coords(logits: torch.Tensor, resolution: int = 32) -> torch.Tensor:
    """The pipeline's step after the decoder (`trellis2_image_to_3d.py:229-233`): > 0, max-pool to `resolution`, argwhere."""
    decoded = logits > 0
    if resolution != decoded.shape[2]:
        ratio = decoded.shape[2] // resolution
        decoded = F.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    return torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
