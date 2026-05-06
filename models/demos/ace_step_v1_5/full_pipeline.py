from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import ttnn
from models.demos.ace_step_v1_5.ttnn_impl.dit_decoder_core import (
    AceStepDecoderConfigTTNN,
    TtAceStepDiTCore,
    TtTimestepEmbedding,
)
from models.demos.ace_step_v1_5.ttnn_impl.output_head import TtAceStepDiTOutputHead
from models.demos.ace_step_v1_5.ttnn_impl.patchify import TtAceStepPatchEmbed1D
from models.demos.ace_step_v1_5.ttnn_impl.safetensors_loader import load_safetensors_state_dict

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


@dataclass(frozen=True)
class AceStepV15LikeConfig:
    """
    Subset of `AceStepConfig` needed by the TTNN patchify/output head.
    """

    patch_size: int
    in_channels: int
    hidden_size: int
    audio_acoustic_hidden_dim: int
    rms_norm_eps: float


class AceStepV15TTNNPipeline:
    """
    Minimal TTNN-only pipeline scaffold.

    This currently wires:
      input acoustic/context tensor -> patch embed -> (TODO: DiT blocks) -> output head

    Notes on strict rules:
    - Host->device occurs in __init__ for weights.
    - Device->host is not performed here; callers should do it once at the end.
    """

    def __init__(
        self,
        *,
        device: ttnn.Device,
        checkpoint_safetensors_path: str,
        decoder_prefix: str = "decoder.",
        activation_dtype: ttnn.DataType | None = None,
        weights_dtype: ttnn.DataType | None = None,
        timesteps_host: Optional["np.ndarray"] = None,
        expected_input_length: int | None = None,
    ) -> None:
        self.device = device
        if activation_dtype is None:
            activation_dtype = getattr(ttnn, "bfloat16", None)
        if weights_dtype is None:
            weights_dtype = getattr(ttnn, "bfloat16", None)
        if activation_dtype is None or weights_dtype is None:
            raise RuntimeError("TTNN build missing bfloat16 dtype; pass activation_dtype/weights_dtype explicitly.")
        self.activation_dtype = activation_dtype
        self.weights_dtype = weights_dtype

        # Load weights on host, then transfer once per weight tensor during module init.
        sd = load_safetensors_state_dict(checkpoint_safetensors_path, prefix=decoder_prefix).tensors

        # Config values are read from config.json in ACE-Step; for now infer minimal subset from tensors where possible.
        # proj_in weight: [hidden_size, in_channels, patch_size]
        proj_in_w = sd.get("proj_in.1.weight")
        if proj_in_w is None:
            proj_in_w = sd.get("proj_in.weight")
        if proj_in_w is None:
            raise KeyError(
                "Missing decoder proj_in weight in safetensors (expected proj_in.1.weight or proj_in.weight)"
            )
        hidden_size, in_channels, patch_size = map(int, proj_in_w.shape)

        # proj_out convtranspose weight: [hidden_size, audio_acoustic_hidden_dim, patch_size]
        proj_out_w = sd.get("proj_out.1.weight")
        if proj_out_w is None:
            proj_out_w = sd.get("proj_out.weight")
        if proj_out_w is None:
            raise KeyError(
                "Missing decoder proj_out weight in safetensors (expected proj_out.1.weight or proj_out.weight)"
            )
        _in_ch, audio_acoustic_hidden_dim, patch_size2 = map(int, proj_out_w.shape)
        if _in_ch != hidden_size or patch_size2 != patch_size:
            raise ValueError(
                f"Unexpected proj_out shape {_in_ch,audio_acoustic_hidden_dim,patch_size2} vs expected "
                f"({hidden_size}, *, {patch_size})"
            )

        norm_w = sd.get("norm_out.weight")
        if norm_w is None:
            raise KeyError("Missing decoder norm_out.weight in safetensors")

        # eps is in config.json; default to 1e-6 to match HF config default.
        cfg = AceStepV15LikeConfig(
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            audio_acoustic_hidden_dim=audio_acoustic_hidden_dim,
            rms_norm_eps=1e-6,
        )

        self.patch_embed = TtAceStepPatchEmbed1D(
            config=cfg,
            state_dict=sd,
            base_address="proj_in",
            device=device,
            expected_input_length=expected_input_length,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
        )

        self.output_head = TtAceStepDiTOutputHead(
            config=cfg,
            state_dict=sd,
            base_address="",  # decoder.* prefix was stripped
            device=device,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
        )

        # Decoder core config. Values are derived from checkpoint tensors.
        import numpy as np

        # q_proj weight: [H*Dh, D]
        q_w = sd["layers.0.self_attn.q_proj.weight"]
        hidden = int(q_w.shape[1])
        head_dim = 128  # AceStepConfig default; also stored in config.json but not loaded here
        num_attention_heads = int(q_w.shape[0]) // head_dim
        kv_w = sd["layers.0.self_attn.k_proj.weight"]
        num_kv_heads = int(kv_w.shape[0]) // head_dim
        num_layers = sum(1 for k in sd.keys() if k.startswith("layers.") and k.endswith(".self_attn.q_proj.weight"))

        core_cfg = AceStepDecoderConfigTTNN(
            hidden_size=hidden,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
            sliding_window=None,
        )

        # Default to 8-step schedule if not provided. Caller should pass the exact sampler schedule.
        if timesteps_host is None:
            timesteps_host = np.linspace(1.0, 0.0, num=8, dtype=np.float32)
        else:
            timesteps_host = np.asarray(timesteps_host, dtype=np.float32)
        # Ensure we have an explicit 0.0 entry for `time_embed_r(t - r)` where r==t -> 0.
        if timesteps_host.size == 0 or float(timesteps_host[-1]) != 0.0:
            timesteps_host = np.concatenate([timesteps_host, np.asarray([0.0], dtype=np.float32)], axis=0)
        self._zero_timestep_index = int(timesteps_host.size - 1)

        self.time_embed = TtTimestepEmbedding(
            cfg=core_cfg,
            state_dict=sd,
            base_address="time_embed",
            mesh_device=device,
            timesteps_host=timesteps_host,
            dtype=self.activation_dtype,
        )
        self.time_embed_r = TtTimestepEmbedding(
            cfg=core_cfg,
            state_dict=sd,
            base_address="time_embed_r",
            mesh_device=device,
            timesteps_host=timesteps_host,
            dtype=self.activation_dtype,
        )

        self.core = TtAceStepDiTCore(cfg=core_cfg, state_dict=sd, mesh_device=device, dtype=self.activation_dtype)

    def forward(
        self,
        *,
        hidden_states_btC: ttnn.Tensor,
        timestep_index: int,
        encoder_hidden_states_btd: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Args:
            hidden_states_btC: [B, T, in_channels] device tensor (ROW_MAJOR preferred).
            temb_bD: [B, hidden_size] device tensor.
        Returns:
            acoustic features [B, T, audio_acoustic_hidden_dim] on device.
        """
        patches, meta = self.patch_embed.forward(hidden_states_btC)

        # timestep embeddings
        temb_t, tp_t = self.time_embed(int(timestep_index))
        temb_r, tp_r = self.time_embed_r(self._zero_timestep_index)
        temb = ttnn.add(temb_t, temb_r)  # [1,D]
        timestep_proj = ttnn.add(tp_t, tp_r)  # [1,6,D]

        # Expand to batch: current bring-up assumes B==1; for B>1, replicate temb/timestep_proj.
        # (This keeps device purity; user can extend with per-batch slicing later.)
        patches_out = self.core(patches, timestep_proj, encoder_hidden_states_btd)

        acoustic = self.output_head.forward(patches_out, temb, meta)
        return acoustic
