# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from models.experimental.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print


@dataclass(frozen=True)
class AceCfg:
    patch_size: int
    in_channels: int
    hidden_size: int
    audio_acoustic_hidden_dim: int
    rms_norm_eps: float


def _strip_prefix(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not prefix:
        return dict(sd)
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
    return out


def _download_base_checkpoint() -> tuple[Path, Path]:
    """
    Returns:
      (model_dir, model_safetensors_path) for ACE-Step v1.5 Base.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required for HF parity tests") from e

    snap = Path(
        snapshot_download(
            "ACE-Step/acestep-v15-base", local_files_only=bool(__import__("os").environ.get("HF_HUB_OFFLINE"))
        )
    )
    model_path = snap / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model.safetensors in base repo snapshot: {model_path}")
    return snap, model_path


def _load_cfg(model_dir: Path) -> AceCfg:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json in model dir: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    return AceCfg(
        patch_size=int(cfg["patch_size"]),
        in_channels=int(cfg["in_channels"]),
        hidden_size=int(cfg["hidden_size"]),
        audio_acoustic_hidden_dim=int(cfg["audio_acoustic_hidden_dim"]),
        rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
    )


def _torch_patch_embed(x: torch.Tensor, *, conv: nn.Conv1d, patch_size: int) -> tuple[torch.Tensor, int]:
    b, t, _c = x.shape
    pad_len = 0 if (t % patch_size == 0) else (patch_size - (t % patch_size))
    if pad_len:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_len), mode="constant", value=0.0)
    x = x.transpose(1, 2)  # [B,C,T]
    y = conv(x).transpose(1, 2)  # [B,T_p,H]
    return y, pad_len


def _torch_rmsnorm_qwen3(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f * weight.float()).to(x.dtype)


@pytest.mark.parametrize("seq_len", [257])
def test_hf_base_proj_in_patch_embed_pcc(device, seq_len: int):
    """
    HF parity test for `proj_in` only (Torch Conv1d vs TTNN TtAceStepPatchEmbed1D) using Base weights.
    """
    ttnn = pytest.importorskip("ttnn")

    from safetensors.torch import load_file as torch_load_safetensors

    from models.experimental.ace_step_v1_5.ttnn_impl.patchify import TtAceStepPatchEmbed1D

    model_dir, model_path = _download_base_checkpoint()
    cfg = _load_cfg(model_dir)

    sd_full = torch_load_safetensors(str(model_path), device="cpu")
    sd_torch = _strip_prefix(sd_full, "decoder.")
    # TTNN path expects numpy-like host weights (no torch bf16 -> numpy conversions).
    from models.experimental.ace_step_v1_5.ttnn_impl.safetensors_loader import load_safetensors_state_dict

    sd_np = load_safetensors_state_dict(str(model_path), prefix="decoder.").tensors

    conv = nn.Conv1d(
        in_channels=cfg.in_channels,
        out_channels=cfg.hidden_size,
        kernel_size=cfg.patch_size,
        stride=cfg.patch_size,
        padding=0,
        bias=True,
    ).to(torch.bfloat16)
    with torch.no_grad():
        conv.weight.copy_(sd_torch["proj_in.1.weight"].to(torch.bfloat16))
        conv.bias.copy_(sd_torch["proj_in.1.bias"].to(torch.bfloat16))
    conv.eval()

    torch.manual_seed(0)
    x = torch.randn((1, seq_len, cfg.in_channels), dtype=torch.bfloat16)
    y_ref, _pad_len = _torch_patch_embed(x, conv=conv, patch_size=cfg.patch_size)

    tt_proj_in = TtAceStepPatchEmbed1D(
        config=cfg, state_dict=sd_np, base_address="proj_in", device=device, activation_dtype=ttnn.bfloat16
    )
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y_tt, _meta = tt_proj_in.forward(x_tt)
    y_tt_torch = ttnn.to_torch(y_tt).float()

    assert_pcc_print("hf_proj_in_patch_embed", y_ref, y_tt_torch)


@pytest.mark.parametrize("seq_len", [257])
def test_hf_base_output_head_pcc(device, seq_len: int):
    """
    HF parity test for output head: `norm_out + scale_shift_table + proj_out`.

    This compares a Torch implementation vs TTNN `TtAceStepDiTOutputHead` using Base weights.
    """
    ttnn = pytest.importorskip("ttnn")

    from safetensors.torch import load_file as torch_load_safetensors

    from models.experimental.ace_step_v1_5.ttnn_impl.output_head import TtAceStepDiTOutputHead
    from models.experimental.ace_step_v1_5.ttnn_impl.patchify import PatchifyMetadata

    model_dir, model_path = _download_base_checkpoint()
    cfg = _load_cfg(model_dir)

    sd_full = torch_load_safetensors(str(model_path), device="cpu")
    sd_torch = _strip_prefix(sd_full, "decoder.")
    from models.experimental.ace_step_v1_5.ttnn_impl.safetensors_loader import load_safetensors_state_dict

    sd_np = load_safetensors_state_dict(str(model_path), prefix="decoder.").tensors

    # Build patch-token input x_patches [B,T_p,H] to feed output head.
    torch.manual_seed(0)
    pad_len = 0 if (seq_len % cfg.patch_size == 0) else (cfg.patch_size - (seq_len % cfg.patch_size))
    t_p = (seq_len + pad_len) // cfg.patch_size
    meta = PatchifyMetadata(original_seq_len=seq_len, pad_length=pad_len, patch_size=cfg.patch_size)

    x_patches = torch.randn((1, t_p, cfg.hidden_size), dtype=torch.bfloat16)
    temb = torch.randn((1, cfg.hidden_size), dtype=torch.bfloat16)

    # Torch output-head reference
    norm_w = sd_torch["norm_out.weight"].to(torch.bfloat16)
    sst = sd_torch["scale_shift_table"].to(torch.bfloat16)

    convt = nn.ConvTranspose1d(
        in_channels=cfg.hidden_size,
        out_channels=cfg.audio_acoustic_hidden_dim,
        kernel_size=cfg.patch_size,
        stride=cfg.patch_size,
        padding=0,
        bias=True,
    ).to(torch.bfloat16)
    with torch.no_grad():
        convt.weight.copy_(sd_torch["proj_out.1.weight"].to(torch.bfloat16))
        convt.bias.copy_(sd_torch["proj_out.1.bias"].to(torch.bfloat16))
    convt.eval()

    with torch.no_grad():
        normed = _torch_rmsnorm_qwen3(x_patches, norm_w, cfg.rms_norm_eps)
        shift = sst[:, 0:1, :] + temb.unsqueeze(1)
        scale = sst[:, 1:2, :] + temb.unsqueeze(1)
        modulated = (normed * (1 + scale) + shift).type_as(x_patches)
        y_ref = convt(modulated.transpose(1, 2)).transpose(1, 2)[:, :seq_len, :]

    # TTNN output head
    tt_head = TtAceStepDiTOutputHead(
        config=cfg,
        state_dict=sd_np,
        base_address="",
        device=device,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
    )
    x_tt = ttnn.from_torch(x_patches, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    temb_tt = ttnn.from_torch(temb, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y_tt = tt_head.forward(x_tt, temb_tt, meta)
    y_tt_torch = ttnn.to_torch(y_tt).float()

    assert_pcc_print("hf_output_head", y_ref, y_tt_torch)
