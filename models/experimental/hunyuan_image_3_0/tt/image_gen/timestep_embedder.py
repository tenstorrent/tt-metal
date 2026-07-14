# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN implementation of the HunyuanImage-3.0 TimestepEmbedder.
# Mirrors ref/image_gen/timestep_embedder.py:
#     t_freq = timestep_embedding(t, frequency_embedding_size)   # sinusoidal
#     t_emb  = down(gelu(up(t_freq)))                            # two-layer MLP
#
# PURE TTNN: every per-call operation runs on device. Only the data-independent
# frequency factor  exp(-log(max_period) * arange(half) / half)  is precomputed
# once on host and uploaded as a constant tensor — exactly the canonical pattern
# in models/tt_dit/layers/embeddings.py (Timesteps._create_time_proj_factor +
# on-device cos/sin/concat). The sinusoidal multiply/cos/sin/concat are done in
# float32 on device (sinusoidal embeddings are precision-sensitive), then cast
# to bf16 before the MLP — matching the reference, which casts t_freq to the
# (bf16) MLP weight dtype.
#
# Weight layout in the checkpoint (PyTorch nn.Linear -> [out, in]):
#     mlp.0.weight : [hidden, frequency_embedding_size]   mlp.0.bias : [hidden]
#     mlp.2.weight : [out, hidden]                        mlp.2.bias : [out]
# ttnn.linear computes x @ W, so we store the transposes [in, out].
#
# Matmuls use act_width_sharded_linear: WIDTH_SHARDED L1 activations with
# interleaved DRAM weights (resnet50-linear pattern) so Tracy shows
# in0:width_sharded without the PCC loss seen on WIDTH_SHARDED weight upload.
#
# The final MLP output is left WIDTH_SHARDED with M padded to 32 so patch_embed /
# final_layer ResBlocks can silu + emb-linear without repeating FillPad + I2S.

import math

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule

from ..matmul_utils import act_width_sharded_linear, reshard_width_act_for_next_linear


class HunyuanTtTimestepEmbedder(LightweightModule):
    """
    Single-device TTNN TimestepEmbedder for HunyuanImage-3.0.

    Args:
        device:      TTNN device.
        hidden_size: Inner MLP width (e.g. 4096).
        state_dict:  Model state_dict (plain torch tensors).
        prefix:      Module prefix, e.g. ``timestep_emb``, ``time_embed`` or
                     ``time_embed_2``. The weights ``{prefix}.mlp.0.{weight,bias}``
                     and ``{prefix}.mlp.2.{weight,bias}`` are read.
        frequency_embedding_size: Sinusoidal feature dim fed to the first Linear
                     (must be even; default 256).
        max_period:  Sinusoidal max period.
        weight_dtype: TTNN dtype for the linear weights (default bfloat16).
    """

    def __init__(
        self,
        device,
        hidden_size: int,
        state_dict: dict,
        prefix: str,
        *,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        weight_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        assert frequency_embedding_size % 2 == 0, "frequency_embedding_size must be even"
        self.device = device
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

        # --- constant frequency factor, built ON DEVICE once -----------------
        #   freqs = exp(-log(max_period) * arange(half) / half)
        # arange/exp run in TTNN; only the scalar coefficient is a Python float.
        half = frequency_embedding_size // 2
        idx = ttnn.arange(0, half, 1, dtype=ttnn.float32, device=device)  # [half] ROW_MAJOR
        idx = ttnn.reshape(idx, (1, 1, 1, half))  # broadcasts against timestep [1,1,N,1]
        idx = ttnn.to_layout(idx, ttnn.TILE_LAYOUT)
        exponent = ttnn.multiply(idx, -math.log(max_period) / half)
        ttnn.deallocate(idx)
        self.freqs = ttnn.exp(exponent)
        ttnn.deallocate(exponent)

        # --- MLP weights (interleaved DRAM — sharded at activation for matmul) -
        w0 = state_dict[f"{prefix}.mlp.0.weight"]  # [hidden, freq]
        b0 = state_dict[f"{prefix}.mlp.0.bias"]  # [hidden]
        w2 = state_dict[f"{prefix}.mlp.2.weight"]  # [out, hidden]
        b2 = state_dict[f"{prefix}.mlp.2.bias"]  # [out]

        self.w_up = ttnn.from_torch(
            w0.transpose(0, 1).contiguous(),  # [freq, hidden]
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.b_up = ttnn.from_torch(
            b0.reshape(1, -1).contiguous(),  # [1, hidden]
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.w_down = ttnn.from_torch(
            w2.transpose(0, 1).contiguous(),  # [hidden, out]
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.b_down = ttnn.from_torch(
            b2.reshape(1, -1).contiguous(),  # [1, out]
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def deallocate(self):
        ttnn.deallocate(self.freqs)
        ttnn.deallocate(self.w_up)
        ttnn.deallocate(self.b_up)
        ttnn.deallocate(self.w_down)
        ttnn.deallocate(self.b_down)

    def _timestep_embedding(self, timestep: ttnn.Tensor) -> ttnn.Tensor:
        """On-device sinusoidal featurization (fp32): [1,1,N,1] -> [1,1,N,freq].

        Mirrors ref timestep_embedding: cat([cos(t*freqs), sin(t*freqs)], -1).
        """
        args = ttnn.multiply(timestep, self.freqs)  # [1,1,N,half] (broadcast)
        c = ttnn.cos(args)
        s = ttnn.sin(args)
        ttnn.deallocate(args)
        freq = ttnn.concat([c, s], dim=-1)  # cos first, matching the reference
        ttnn.deallocate(c)
        ttnn.deallocate(s)
        return freq

    def forward(self, t, *, keep_resident: bool = False, resident_next_n: int | None = None) -> ttnn.Tensor:
        """
        Args:
            t: timesteps. Either a torch 1-D tensor [N] (host scalars — uploaded
               here as raw data, no host math) or a TTNN tensor shaped [1,1,N,1]
               in float32 TILE_LAYOUT on device.
            keep_resident: If True, return WIDTH_SHARDED L1 with the batch dim
               padded to 32 so ResBlock emb_layers can skip FillPad + I2S.
               Logical rows remain the leading ``N`` entries; denoise callers
               must pass ``batch_rows=N`` / slice with real B. Default False
               keeps interleaved DRAM ``[1,1,N,out]`` for I2I / scatter / PCC.
            resident_next_n: When keeping resident, optionally reshard to the
               activation layout expected by the next linear with weight
               ``[out, resident_next_n]`` (ResBlock emb_layers uses ``2*C_out``).
        Returns:
            TTNN tensor on device: interleaved ``[1,1,N,out]`` by default, or
            resident WIDTH_SHARDED ``[1,1,32,out]`` when ``keep_resident``.
        """
        if isinstance(t, torch.Tensor):
            n = t.numel()
            t = ttnn.from_torch(
                t.reshape(1, 1, n, 1).float(),  # raw upload — sinusoidal math runs on device
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            rank = len(t.shape)
            if rank == 1:
                n = int(t.shape[0])
                t = ttnn.reshape(t, (1, 1, n, 1))
            elif rank == 2:
                n = int(t.shape[-1])
                t = ttnn.reshape(t, (1, 1, n, 1))
            elif rank == 4:
                n = int(t.shape[2])
                if int(t.shape[-1]) != 1:
                    t = ttnn.reshape(t, (1, 1, n, 1))
            else:
                raise ValueError(f"unsupported timestep rank {rank}")
            if t.layout != ttnn.TILE_LAYOUT:
                t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
            if t.dtype != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)

        freq = self._timestep_embedding(t)  # [1,1,N,freq] fp32
        x = ttnn.typecast(freq, ttnn.bfloat16)  # match ref cast to MLP weight dtype
        ttnn.deallocate(freq)

        h = act_width_sharded_linear(
            x,
            self.w_up,
            bias=self.b_up,
            batch_rows=n,
            compute_kernel_config=self.compute_kernel_config,
            device=self.device,
        )  # [1,1,N,hidden]
        ttnn.deallocate(x)

        act = ttnn.gelu(h, fast_and_approximate_mode=False)  # exact (erf) GELU
        ttnn.deallocate(h)

        out = act_width_sharded_linear(
            act,
            self.w_down,
            bias=self.b_down,
            batch_rows=n,
            compute_kernel_config=self.compute_kernel_config,
            device=self.device,
            keep_sharded_output=keep_resident,
        )  # resident [1,1,32,out] WIDTH_SHARDED, or interleaved [1,1,N,out]
        ttnn.deallocate(act)
        if keep_resident and resident_next_n is not None:
            out = reshard_width_act_for_next_linear(out, next_n=resident_next_n, device=self.device)
        return out
