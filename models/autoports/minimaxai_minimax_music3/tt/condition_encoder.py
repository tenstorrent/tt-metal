# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-Music3's condition encoder (``MiniMaxMusic3ConditionEncoder``) with the conv on device.

Reference: diffusers ``models/condition_embedders/condition_embedder_minimax_music3.py`` (vendored in
``reference/flow_transformer_ref.py``). Per chunk (<= 200 frames) it maps the AR stage's per-frame
hidden states ``[1, F, 8 * 4096]`` (layer-major: the backbone hidden followed by the seven depth-step
hiddens) to the latent-aligned conditioning ``[1, L, 2048]`` with ``L = int(F * 44100 / 24000 * 960 / 512)``:

1. softmax-weighted mix of the 8 hidden groups (``layer_weight_logits``) times ``layer_scale`` - host fp32
   (a 6.5 M-element weighted sum, well under a millisecond; keeping it on the host also keeps the
   32768-wide input off the device);
2. ``Conv1d(4096, 2048, k=3, padding=1)`` as ONE matmul over the 3-tap unfold: rows
   ``[x[f-1], x[f], x[f+1]]`` (zero padded) ``[F_pad, 12288] @ [12288, 2048] + bias`` on device
   (bf16 weights, HiFi2 with fp32 accumulation; this is the only heavy op, 25 M parameters);
3. nearest-neighbour resampling of the F conv outputs onto the L latent slots - a host
   ``index_select`` with the index map taken from ``torch.nn.functional.interpolate`` itself on an index
   ramp, so the rounding matches the reference bit for bit.

The result is returned as host fp32 ``[1, L, 2048]`` because the chunk loop (``tt/denoiser.py``) splices the
previous window's conditioning into it and carries a window of it to the next chunk on the host; the DiT
then folds it once per chunk with ``FlowTransformer.prepare_condition``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32
CONDITION_HIDDEN = 4096
NUM_CONDITION_LAYERS = 8
OUT_DIM = 2048
KERNEL = 3
INPUT_SAMPLING_RATE = 24000
INPUT_HOP = 960
OUTPUT_SAMPLING_RATE = 44100
OUTPUT_HOP = 512


def latent_length(num_frames: int) -> int:
    """Latent frames for ``num_frames`` AR frames (the reference's ``int(...)`` truncation, at least 1)."""
    return max(1, int(num_frames * OUTPUT_SAMPLING_RATE / INPUT_SAMPLING_RATE * INPUT_HOP / OUTPUT_HOP))


def nearest_index_map(num_frames: int, num_latents: int) -> torch.Tensor:
    """``[num_latents]`` source frame per latent slot, exactly as ``F.interpolate(mode="nearest")`` picks it."""
    ramp = torch.arange(num_frames, dtype=torch.float32).reshape(1, 1, num_frames)
    return F.interpolate(ramp, size=num_latents, mode="nearest").reshape(-1).round().long()


class ConditionEncoder(LightweightModule):
    def __init__(self, mesh_device, state_dict: Dict[str, torch.Tensor], *, dtype=ttnn.bfloat16):
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.mem = ttnn.DRAM_MEMORY_CONFIG
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        t0 = time.time()
        sd = {k: v.float() for k, v in state_dict.items()}
        assert tuple(sd["proj.weight"].shape) == (OUT_DIM, CONDITION_HIDDEN, KERNEL), tuple(sd["proj.weight"].shape)
        self.layer_weights = torch.softmax(sd["layer_weight_logits"], dim=0)  # [8] fp32, host
        self.layer_scale = float(sd["layer_scale"].reshape(-1)[0])
        # Unfolded conv weight: block k (rows k*4096 .. (k+1)*4096) = W[:, :, k]^T, so that
        # cat(x[f-1], x[f], x[f+1]) @ w_unf == sum_k W[:, :, k] @ x[f + k - 1].
        w_unf = torch.cat([sd["proj.weight"][:, :, k].t() for k in range(KERNEL)], dim=0).contiguous()
        self.weight = ttnn.from_torch(
            w_unf, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=self.mem
        )
        self.bias = ttnn.from_torch(
            sd["proj.bias"].reshape(1, OUT_DIM),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=self.mem,
        )
        ttnn.synchronize_device(mesh_device)
        self.load_seconds = time.time() - t0

    @classmethod
    def from_pretrained(
        cls, mesh_device, weights_dir: Optional[Path] = None, *, dtype=ttnn.bfloat16
    ) -> "ConditionEncoder":
        from models.autoports.minimaxai_minimax_music3.reference.flow_transformer_ref import (
            load_condition_encoder_state_dict,
        )

        return cls(mesh_device, load_condition_encoder_state_dict(weights_dir), dtype=dtype)

    # ------------------------------------------------------------------ pieces
    def mix_layers(self, frame_hiddens: torch.Tensor) -> torch.Tensor:
        """``[1, F, 8 * 4096]`` -> ``layer_scale * sum_l softmax(logits)_l * h_l`` as ``[F, 4096]`` (host fp32)."""
        assert frame_hiddens.dim() == 3 and frame_hiddens.shape[0] == 1, tuple(frame_hiddens.shape)
        assert frame_hiddens.shape[-1] == NUM_CONDITION_LAYERS * CONDITION_HIDDEN, tuple(frame_hiddens.shape)
        h = frame_hiddens[0].float().reshape(-1, NUM_CONDITION_LAYERS, CONDITION_HIDDEN)
        return self.layer_scale * torch.einsum("flh,l->fh", h, self.layer_weights)

    def conv(self, mixed: torch.Tensor) -> torch.Tensor:
        """``Conv1d(k=3, padding=1)`` over the frame axis of ``[F, 4096]`` -> ``[F, 2048]`` (host fp32 in/out, device matmul)."""
        num_frames = mixed.shape[0]
        f_pad = -(-num_frames // TILE) * TILE
        padded = torch.zeros(num_frames + 2, CONDITION_HIDDEN)
        padded[1 : num_frames + 1] = mixed
        unf = torch.zeros(f_pad, KERNEL * CONDITION_HIDDEN)
        unf[:num_frames] = torch.cat([padded[k : k + num_frames] for k in range(KERNEL)], dim=-1)
        x = ttnn.from_torch(
            unf.reshape(1, 1, f_pad, KERNEL * CONDITION_HIDDEN).to(torch.bfloat16),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.mem,
        )
        y = ttnn.linear(
            x,
            self.weight,
            bias=self.bias,
            compute_kernel_config=self.compute_config,
            memory_config=self.mem,
            dtype=self.dtype,
        )
        ttnn.deallocate(x)
        out = ttnn.to_torch(y).float().reshape(f_pad, OUT_DIM)[:num_frames]
        ttnn.deallocate(y)
        return out

    def forward(self, frame_hiddens: torch.Tensor) -> torch.Tensor:
        """``[1, F, 32768]`` frame hiddens of one chunk -> ``[1, L, 2048]`` latent-aligned conditioning (host fp32)."""
        num_frames = frame_hiddens.shape[1]
        conv_out = self.conv(self.mix_layers(frame_hiddens))  # [F, 2048]
        idx = nearest_index_map(num_frames, latent_length(num_frames))
        return conv_out.index_select(0, idx).unsqueeze(0)

    def release(self) -> None:
        ttnn.deallocate(self.weight)
        ttnn.deallocate(self.bias)
