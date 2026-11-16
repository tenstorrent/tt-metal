# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# LM head for the HunyuanImage-3.0 text-generation (autoregressive) path.
#
# The backbone (tt/model.py) emits final hidden states [B, S, H] (ln_f applied when
# apply_final_norm=True). For text generation the reference projects those to
# vocabulary logits via `lm_head` (modeling_hunyuan_image_3.py forward: `logits =
# self.lm_head(self.model.ln_f(hidden_states))`). `tie_word_embeddings=False`, so
# `lm_head.weight` ([V, H]) is a distinct tensor in the checkpoint.
#
# Only the diffusion (image-gen) path was needed until now, so no head existed; this
# adds the missing text-logits projection for the Instruct recaption/think loop.

import ttnn
from models.common.lightweightmodule import LightweightModule

from .matmul_utils import TILE_SIZE, small_m_split_n_linear, to_interleaved_if_sharded


class HunyuanTtLMHead(LightweightModule):
    """Vocabulary projection: hidden [B, S, H] -> logits [B, S, V] (no bias).

    Mirrors the Linear-weight convention used across this port (e.g.
    `HunyuanTtLightProjector`): the checkpoint stores `lm_head.weight` as [V, H]
    (out, in); `ttnn.linear` wants [in, out], so we transpose at load.

    Tensor-parallel vocab: on a multi-device mesh the huge [H, V] weight (~545 MB
    BFP8 at V=133120) was previously *replicated*, so every device redundantly
    streamed the whole matrix and computed identical logits — the D2H read then
    concatenated the replicas and discarded all but one. Instead we shard V across
    the mesh (each device holds [H, V/num_devices]) and concat the vocab slices on
    the host read (see ``vocab_parallel``). The hidden input is already fully
    replicated after the backbone's SP exit-gather, so this needs no on-device
    collective — it is embarrassingly parallel and cuts the per-device DRAM read
    (the op's bottleneck — it runs at ~60% DRAM BW) by ``num_devices``x.
    """

    def __init__(self, device, state_dict: dict, *, key: str = "lm_head.weight", weight_dtype=ttnn.bfloat8_b):
        super().__init__()
        self.device = device
        w = state_dict[key].transpose(0, 1).contiguous()  # [H, V]
        self.vocab_size = w.shape[1]

        num_devices = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
        # Shard the vocab (dim 1 of [H, V]) across the mesh when it divides evenly
        # into tile-aligned per-device slices; otherwise fall back to replication.
        self.vocab_parallel = (
            num_devices > 1 and self.vocab_size % num_devices == 0 and (self.vocab_size // num_devices) % TILE_SIZE == 0
        )
        if self.vocab_parallel:
            mesh_mapper = ttnn.ShardTensorToMesh(device, dim=1)
        elif num_devices > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        else:
            mesh_mapper = None
        self.weight = ttnn.from_torch(
            w,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, hidden: ttnn.Tensor, *, last_token_only: bool = False) -> ttnn.Tensor:
        """Project hidden states to logits.

        Args:
            hidden:          ttnn [B, S, H] (post-ln_f) final hidden states.
            last_token_only: if True, project only the final sequence position
                             ([B, 1, H] -> [B, 1, V]) — the cheap path for AR decode,
                             where only the next-token distribution is needed and the
                             133k-wide projection over the full sequence is wasteful.

        Returns:
            ttnn [B, S, V] (or [B, 1, V] when last_token_only).
        """
        x = hidden
        sliced = False
        if last_token_only:
            B, S, H = hidden.shape
            x = ttnn.slice(hidden, [0, S - 1, 0], [B, S, H])
            sliced = True
        # Vocab projection (huge N, per device = V/num_devices when vocab-parallel).
        # small_m_split_n_linear pads M→32 and splits N across the widest core grid
        # (decode_mm, ~104 cores on BH) so the weight streams from more DRAM readers
        # in parallel — this op is DRAM-bandwidth-bound (see tests/perf/
        # test_lmhead_sweep.py). At last_token_only the M is a single tile (Mt=1), the
        # only shape decode_mm tolerates at this N without a CB/kernel-arg overflow.
        logits = small_m_split_n_linear(
            x,
            self.weight,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        logits = to_interleaved_if_sharded(logits)
        if sliced:
            ttnn.deallocate(x)
        return logits
