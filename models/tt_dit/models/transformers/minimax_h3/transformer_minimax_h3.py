# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 packed-token DiT.

A 33B single-stream transformer -- not an MMDiT -- that denoises one packed
sequence holding text, keyframe conditioning, target audio and target video at
once. Modality-specific parameters live only in the input/output layers and the
AdaLN branches; attention and the FFN are modality-agnostic.

**The AdaLN projections are deliberately absent from this module.** They are 13B
parameters, 40% of the checkpoint, and their output depends only on the request's
sigma schedule, so ``pipelines/minimax_h3/adaln_precompute.py`` evaluates them
once on host and this module consumes the resulting table. The model card
licenses exactly that. ``time_embedder`` and ``rope.inv_freq`` are likewise host
concerns and not loaded here.

Sharding follows the invariant in the port plan: weights shard on the TP axis and
replicate along SP; activations shard the sequence on SP and hidden on TP. Hidden
therefore stays column-fractured through a block, which is why the pre-norms are
distributed.
"""

from __future__ import annotations

import torch

import ttnn

from ....layers.linear import ColParallelLinear, RowParallelLinear
from ....layers.module import Module, ModuleList
from ....layers.normalization import DistributedRMSNorm, RMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from .attention_minimax_h3 import MiniMaxH3Attention

# Keys the device model deliberately does not own. Dropping them is not
# leniency: adaln_proj is precomputed on host, and the time embedder and rope
# frequencies feed that precompute rather than any device op.
MINIMAX_H3_HOST_OWNED_PREFIXES = (
    "time_embedder.",
    "rope.",
)
MINIMAX_H3_HOST_OWNED_SUFFIXES = ("adaln_proj.linear.weight", "adaln_proj.linear.bias")


class MiniMaxH3MLP(Module):
    """SwiGLU feed-forward with a fused ``fc1``."""

    def __init__(
        self,
        *,
        hidden_size: int,
        ffn_hidden_size: int,
        mesh_device,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
    ) -> None:
        super().__init__()
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        self.fc1 = ColParallelLinear(
            hidden_size,
            ffn_hidden_size,
            bias=False,
            activation_fn="swiglu",
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
        )
        self.fc2 = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight = state.get("fc1.weight")
        if weight is not None:
            # The checkpoint stores [gate; value] and the reference computes
            # silu(gate) * value, but tt_dit's swiglu is `t, gate = chunk(t, 2);
            # t * silu(gate)` -- it reads [value; gate]. Swap here; the
            # cross-device interleave is then ColParallelLinear's job.
            gate, value = weight.chunk(2, dim=0)
            state["fc1.weight"] = torch.cat([value, gate], dim=0).contiguous()

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.fc2(self.fc1(x))


class MiniMaxH3TransformerBlock(Module):
    """Pre-norm self-attention and FFN, each modulated per row by AdaLN."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        ffn_hidden_size: int,
        norm_eps: float,
        qk_norm_eps: float,
        mesh_device,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        distributed_norm: bool = True,
    ) -> None:
        super().__init__()
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        # H3 RMSNorms are weight-only; tt_dit defaults bias=True.
        norm_kwargs = dict(norm_eps=norm_eps, mesh_device=mesh_device)
        if distributed_norm:
            self.norm1 = DistributedRMSNorm(hidden_size, mesh_axis=tp_axis, ccl_manager=ccl_manager, **norm_kwargs)
            self.norm2 = DistributedRMSNorm(hidden_size, mesh_axis=tp_axis, ccl_manager=ccl_manager, **norm_kwargs)
        else:
            self.norm1 = RMSNorm(hidden_size, bias=False, **norm_kwargs)
            self.norm2 = RMSNorm(hidden_size, bias=False, **norm_kwargs)
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            qk_norm_eps=qk_norm_eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )
        self.mlp = MiniMaxH3MLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )
        self.ccl_manager = ccl_manager
        self.tp_axis = tp_axis

    def forward(self, x: ttnn.Tensor, *, modulation=None, rope=None, logical_n: int | None = None) -> ttnn.Tensor:
        raise NotImplementedError("block forward lands with the M5 attention gate")


class _LoadOnly(Module):
    """Base for modules that exist to own parameters before their forward lands.

    ``Module`` is an ABC with ``forward`` abstract, so a container cannot be
    instantiated for a weight-load gate without declaring one.
    """

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__}.forward lands with the M5/M6 gates")


class MiniMaxH3TokenRefiner(_LoadOnly):
    """Refines the projected text stream once per request.

    Its blocks carry the same parameters as the denoiser's -- norms, attention,
    MLP -- because the only thing a refiner block lacks is AdaLN, and AdaLN is
    host-side here. They differ only in how ``forward`` drives them: no modulation
    and no RoPE.
    """

    def __init__(self, *, num_layers: int, final_norm_eps: float, mesh_device, **block_kwargs) -> None:
        super().__init__()
        self.blocks = ModuleList(
            [MiniMaxH3TransformerBlock(mesh_device=mesh_device, **block_kwargs) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(
            block_kwargs["hidden_size"], norm_eps=final_norm_eps, bias=False, mesh_device=mesh_device
        )


class MiniMaxH3FinalLayer(_LoadOnly):
    """Final norm plus the two fp32 output heads.

    ``adaln_proj`` is absent by design -- its shift/scale come from the
    precomputed table. The heads stay fp32 and replicated: 96 and 32 columns do
    not shard across TP=4 without going sub-tile, and replicating removes the
    output gather entirely.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        video_out_features: int,
        audio_out_features: int,
        final_norm_eps: float,
        mesh_device,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size, norm_eps=final_norm_eps, bias=False, mesh_device=mesh_device)
        head_kwargs = dict(bias=True, dtype=ttnn.float32, mesh_device=mesh_device, mesh_axis=None)
        self.video_out = ColParallelLinear(hidden_size, video_out_features, **head_kwargs)
        self.audio_out = ColParallelLinear(hidden_size, audio_out_features, **head_kwargs)


class MiniMaxH3Transformer3DModel(_LoadOnly):
    """The FL2VA denoiser, minus the host-side AdaLN and time embedder."""

    def __init__(
        self,
        *,
        hidden_size: int = 5376,
        num_layers: int = 50,
        token_refiner_num_layers: int = 2,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        ffn_hidden_size: int = 14336,
        latents_dim: int = 24,
        audio_latents_dim: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        mesh_device=None,
        ccl_manager: CCLManager = None,
        parallel_config: DiTParallelConfig = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        tp_axis = parallel_config.tensor_parallel.mesh_axis

        video_patch_dim = latents_dim * patch_size[0] * patch_size[1] * patch_size[2]
        # Patch projections are fp32 and tiny; replicating avoids a gather right
        # after them, which is what the reference does with gather_output=True.
        patch_kwargs = dict(bias=True, dtype=ttnn.float32, mesh_device=mesh_device, mesh_axis=None)
        self.video_patch_proj = ColParallelLinear(video_patch_dim, hidden_size, **patch_kwargs)
        self.audio_patch_proj = ColParallelLinear(audio_latents_dim, hidden_size, **patch_kwargs)
        self.condition_proj = ColParallelLinear(
            text_dim,
            hidden_size,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
        )

        block_kwargs = dict(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            ffn_hidden_size=ffn_hidden_size,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )
        self.token_refiner = MiniMaxH3TokenRefiner(
            num_layers=token_refiner_num_layers,
            final_norm_eps=final_norm_eps,
            mesh_device=mesh_device,
            **block_kwargs,
        )
        self.blocks = ModuleList(
            [MiniMaxH3TransformerBlock(mesh_device=mesh_device, **block_kwargs) for _ in range(num_layers)]
        )
        self.final_layer = MiniMaxH3FinalLayer(
            hidden_size=hidden_size,
            video_out_features=video_patch_dim,
            audio_out_features=audio_latents_dim,
            final_norm_eps=final_norm_eps,
            mesh_device=mesh_device,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for key in list(state):
            if key.startswith(MINIMAX_H3_HOST_OWNED_PREFIXES) or key.endswith(MINIMAX_H3_HOST_OWNED_SUFFIXES):
                del state[key]


class MiniMaxH3Checkpoint:
    """Reads ``FL2VA/transformer`` and drives build plus load.

    Weights are read straight from safetensors rather than through diffusers: the
    released FL2VA layout is the original one, whose keys this module mirrors.
    """

    def __init__(self, checkpoint_dir, hidden_size: int = 5376, num_layers: int = 50) -> None:
        from pathlib import Path

        self.checkpoint_dir = Path(checkpoint_dir)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Every tensor the device model owns, skipping the host-side ones.

        The 13B of ``adaln_proj`` is never materialized here, so this reads about
        40 GB rather than the full 66 GB.
        """
        from safetensors import safe_open

        shards = sorted(self.checkpoint_dir.glob("model-*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no model-*.safetensors under {self.checkpoint_dir}")

        state: dict[str, torch.Tensor] = {}
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key.startswith(MINIMAX_H3_HOST_OWNED_PREFIXES) or key.endswith(MINIMAX_H3_HOST_OWNED_SUFFIXES):
                        continue
                    state[key] = handle.get_tensor(key)
        return state

    def build(self, *, mesh_device, ccl_manager, parallel_config, **kwargs) -> MiniMaxH3Transformer3DModel:
        return MiniMaxH3Transformer3DModel(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            **kwargs,
        )

    def load(self, model: MiniMaxH3Transformer3DModel, *, mesh_device, parallel_config, mesh_shape, dtype="bf16"):
        from ....utils import cache

        return cache.load_model(
            model,
            model_name="MiniMax-H3-FL2VA",
            subfolder="transformer",
            parallel_config=parallel_config,
            mesh_shape=mesh_shape,
            dtype=dtype,
            get_torch_state_dict=self.state_dict,
        )
