from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from models.demos.ace_step_v1_5.torch_ref.dit_decoder_core import TorchAceStepDiTCoreRef, TorchTimestepEmbeddingRef
from models.demos.ace_step_v1_5.torch_ref.output_head import OutputHeadConfig, TorchAceStepDiTOutputHead
from models.demos.ace_step_v1_5.torch_ref.output_head import _maybe_get_state_dict_key as _maybe_key
from models.demos.ace_step_v1_5.torch_ref.patchify import patchify_1d
from models.demos.ace_step_v1_5.torch_ref.safetensors_loader import load_safetensors_state_dict
from models.demos.ace_step_v1_5.ttnn_impl.dit_decoder_core import AceStepDecoderConfigTTNN


def _state_dict_to_numpy_float32(sd: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().contiguous().to(torch.float32).numpy()
        else:
            out[k] = np.asarray(v, dtype=np.float32)
    return out


@dataclass(frozen=True)
class AceStepV15LikeConfig:
    patch_size: int
    in_channels: int
    hidden_size: int
    audio_acoustic_hidden_dim: int
    rms_norm_eps: float


def _to_torch_weight(x: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=dtype)
    return torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype)


def _build_proj_in_conv(
    *, state_dict: dict[str, Any], base_address: str, dtype: torch.dtype, device: torch.device
) -> tuple[nn.Conv1d, int, int, int]:
    w_key = _maybe_key(
        state_dict,
        (
            f"{base_address}.1.weight",
            f"{base_address}.weight",
        ),
    )
    b_key = _maybe_key(
        state_dict,
        (
            f"{base_address}.1.bias",
            f"{base_address}.bias",
        ),
    )
    w = state_dict[w_key]
    b = state_dict[b_key]
    if hasattr(w, "shape"):
        oc, ic, ks = int(w.shape[0]), int(w.shape[1]), int(w.shape[2])
    else:
        w = np.asarray(w)
        oc, ic, ks = int(w.shape[0]), int(w.shape[1]), int(w.shape[2])
    conv = nn.Conv1d(ic, oc, kernel_size=ks, stride=ks, padding=0, bias=True)
    conv.weight.data.copy_(_to_torch_weight(w, dtype=dtype, device=device))
    conv.bias.data.copy_(_to_torch_weight(b, dtype=dtype, device=device))
    conv.weight.requires_grad_(False)
    conv.bias.requires_grad_(False)
    return conv, oc, ic, ks


class AceStepV15TorchPipeline(nn.Module):
    """
    Full ACE-Step v1.5 **decoder** path in PyTorch (patch embed → DiT core → output head).

    Uses the same weight layout as the TTNN demo (``decoder.*`` keys stripped by
    :func:`load_safetensors_state_dict`) and the torch reference modules under
    ``torch_ref/``.
    """

    def __init__(
        self,
        *,
        checkpoint_safetensors_path: str,
        decoder_prefix: str = "decoder.",
        timesteps_host: Optional[np.ndarray] = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self._dtype = dtype
        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self._device = device

        sd_wrap = load_safetensors_state_dict(checkpoint_safetensors_path, prefix=decoder_prefix)
        sd = sd_wrap.tensors

        proj_in_conv, hidden_size, in_channels, patch_size = _build_proj_in_conv(
            state_dict=sd, base_address="proj_in", dtype=dtype, device=device
        )

        proj_out_w = sd.get("proj_out.1.weight")
        if proj_out_w is None:
            proj_out_w = sd.get("proj_out.weight")
        if proj_out_w is None:
            raise KeyError("Missing decoder proj_out weight")
        _in_ch, audio_acoustic_hidden_dim, patch_size2 = (
            int(proj_out_w.shape[0]),
            int(proj_out_w.shape[1]),
            int(proj_out_w.shape[2]),
        )
        if _in_ch != hidden_size or patch_size2 != patch_size:
            raise ValueError(
                f"proj_out shape mismatch vs proj_in: got ({_in_ch},{audio_acoustic_hidden_dim},{patch_size2}) "
                f"vs hidden={hidden_size}, patch={patch_size}"
            )

        if "norm_out.weight" not in sd:
            raise KeyError("Missing decoder norm_out.weight")

        ckpt_parent = Path(checkpoint_safetensors_path).resolve().parent
        hf_cfg_path = ckpt_parent / "config.json"
        hf_cfg: dict = {}
        if hf_cfg_path.is_file():
            hf_cfg = json.loads(hf_cfg_path.read_text())
        rms_eps = float(hf_cfg.get("rms_norm_eps", 1e-6))
        head_dim = int(hf_cfg.get("head_dim", 128))
        sw = hf_cfg.get("sliding_window", None)
        sliding_window = int(sw) if sw is not None else None
        layer_types = hf_cfg.get("layer_types", None)
        max_position_embeddings = int(hf_cfg.get("max_position_embeddings", 4096))
        rope_theta = float(hf_cfg.get("rope_theta", 10000.0))

        self.like_cfg = AceStepV15LikeConfig(
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            audio_acoustic_hidden_dim=audio_acoustic_hidden_dim,
            rms_norm_eps=rms_eps,
        )
        self.proj_in = proj_in_conv.to(device=device, dtype=dtype)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        self.output_head = TorchAceStepDiTOutputHead(
            config=OutputHeadConfig(
                hidden_size=hidden_size,
                audio_acoustic_hidden_dim=audio_acoustic_hidden_dim,
                patch_size=patch_size,
                rms_norm_eps=rms_eps,
            ),
            state_dict=sd,
            base_address="",
            device=device,
            dtype=dtype,
        )

        q_w = sd["layers.0.self_attn.q_proj.weight"]
        hidden = int(q_w.shape[1])
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
            rms_norm_eps=float(rms_eps),
            sliding_window=sliding_window,
            layer_types=layer_types,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
        )

        if timesteps_host is None:
            timesteps_host = np.linspace(1.0, 0.0, num=8, dtype=np.float32)
        else:
            timesteps_host = np.asarray(timesteps_host, dtype=np.float32)
        if timesteps_host.size == 0 or float(timesteps_host[-1]) != 0.0:
            timesteps_host = np.concatenate([timesteps_host, np.asarray([0.0], dtype=np.float32)], axis=0)
        self._timesteps_np = timesteps_host

        sd_np = _state_dict_to_numpy_float32(sd)
        self.core = TorchAceStepDiTCoreRef(cfg=core_cfg, state_dict=sd_np)
        self.time_embed = TorchTimestepEmbeddingRef(
            hidden_size=hidden, state_dict=sd_np, base="time_embed", timesteps_host=timesteps_host
        )
        self.time_embed_r = TorchTimestepEmbeddingRef(
            hidden_size=hidden, state_dict=sd_np, base="time_embed_r", timesteps_host=timesteps_host
        )
        self.cond_dim = int(sd["condition_embedder.weight"].shape[1])

    @property
    def timesteps_host(self) -> np.ndarray:
        return self._timesteps_np

    def forward(
        self,
        *,
        xt_bt64: torch.Tensor,
        context_latents_bt128: torch.Tensor,
        timestep_index: int,
        encoder_hidden_states_btd: torch.Tensor,
        timestep_r_index: int | None = None,
        debug_intermediates: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Returns velocity / acoustic prediction [B, T, audio_acoustic_hidden_dim].

        Cross-attention currently assumes an all-valid encoder sequence (matches the common
        text-to-audio path with a full text mask).
        """
        if xt_bt64.ndim != 3 or context_latents_bt128.ndim != 3:
            raise ValueError("xt and context_latents must be [B,T,*]")
        hidden = torch.cat([context_latents_bt128, xt_bt64], dim=-1)
        if int(hidden.shape[-1]) != int(self.in_channels):
            raise ValueError(f"concat last dim {hidden.shape[-1]} != in_channels {self.in_channels}")

        dtype = self._dtype
        device = self._device
        hidden = hidden.to(device=device, dtype=dtype)
        enc = encoder_hidden_states_btd.to(device=device, dtype=dtype)

        patches, meta = patchify_1d(hidden, conv=self.proj_in, patch_size=int(self.patch_size))
        if debug_intermediates is not None and debug_intermediates.get("enabled", False):
            debug_intermediates["pipe.patches"] = patches.detach().float().cpu().clone()

        t_idx = int(timestep_index)
        temb_t, tp_t = self.time_embed(t_idx)
        temb_t = temb_t.to(device=device, dtype=dtype)
        tp_t = tp_t.to(device=device, dtype=dtype)

        if timestep_r_index is None:
            delta_tr = 0.0
        else:
            th = self.timesteps_host
            delta_tr = float(th[t_idx]) - float(th[int(timestep_r_index)])
        temb_r, tp_r = self.time_embed_r.from_timestep_value(delta_tr)
        temb_r = temb_r.to(device=device, dtype=dtype)
        tp_r = tp_r.to(device=device, dtype=dtype)

        temb = temb_t + temb_r
        tp = tp_t + tp_r

        b = int(hidden.shape[0])
        if b != 1:
            if int(temb.shape[0]) == 1:
                temb = temb.expand(b, -1)
            if int(tp.shape[0]) == 1:
                tp = tp.expand(b, -1, -1)

        if debug_intermediates is not None and debug_intermediates.get("enabled", False):
            debug_intermediates["pipe.temb"] = temb.detach().float().cpu().clone()
            debug_intermediates["pipe.timestep_proj_b6d"] = tp.detach().float().cpu().clone()

        # Run DiT core on the same device as patch embed (weights are cached per-device).
        patch_tokens = self.core.forward(
            x_patches=patches,
            timestep_proj_b6d=tp,
            encoder_hidden_states=enc,
            debug=debug_intermediates,
        )
        if debug_intermediates is not None and debug_intermediates.get("enabled", False):
            debug_intermediates["pipe.core_out"] = patch_tokens.detach().float().cpu().clone()

        return self.output_head.forward(patch_tokens, temb, meta, debug=debug_intermediates)
