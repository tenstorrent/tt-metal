# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from models.demos.dots_ocr.tt.dots_vision_tt import DotsVisionTransformerTT
from models.demos.dots_ocr.tt.mesh import close_dots_mesh_device, open_mesh_device
from models.demos.dots_ocr.tt.vision_config_dataclass import DotsVisionConfig as TTDotsVisionConfig
from models.tt_dit.utils.check import assert_quality

try:
    import ttnn  # type: ignore

    _HAS_TTNN_RUNTIME = hasattr(ttnn, "open_mesh_device")
except Exception:
    ttnn = None  # type: ignore
    _HAS_TTNN_RUNTIME = False

if not _HAS_TTNN_RUNTIME:
    pytest.skip("TTNN runtime not available (skipping TTNN PCC tests)", allow_module_level=True)


VISION_PCC_REQUIRED = 0.98


def _make_demo_like_hf_config() -> types.SimpleNamespace:
    vision_config = types.SimpleNamespace(
        embed_dim=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        init_merger_std=0.02,
        use_bias=False,
        post_norm=True,
        attn_implementation="eager",
    )
    return types.SimpleNamespace(vision_config=vision_config, _name_or_path="dots_vision_pcc_local")


def _to_tt_vision_config(hf_config: types.SimpleNamespace) -> TTDotsVisionConfig:
    cfg = hf_config.vision_config
    tt_keys = set(TTDotsVisionConfig.__dataclass_fields__.keys())
    return TTDotsVisionConfig(**{k: getattr(cfg, k) for k in tt_keys if hasattr(cfg, k)})


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _get_pos_ids_by_grid(grid_thw: torch.Tensor, spatial_merge_size: int) -> tuple[list[int], list[int]]:
    h_all: list[int] = []
    w_all: list[int] = []
    for t, h, w in grid_thw.tolist():
        h_block: list[int] = []
        w_block: list[int] = []
        for hb in range(0, h, spatial_merge_size):
            for wb in range(0, w, spatial_merge_size):
                for hi in range(spatial_merge_size):
                    for wi in range(spatial_merge_size):
                        h_block.append(hb + hi)
                        w_block.append(wb + wi)
        for _ in range(t):
            h_all.extend(h_block)
            w_all.extend(w_block)
    return h_all, w_all


def _rotary_pos_emb(grid_thw: torch.Tensor, num_heads: int, hidden_size: int, spatial_merge_size: int) -> torch.Tensor:
    head_dim = hidden_size // num_heads
    rotary_dim = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    h_ids, w_ids = _get_pos_ids_by_grid(grid_thw, spatial_merge_size)
    h_freqs = torch.outer(torch.tensor(h_ids, dtype=torch.float32), inv_freq)
    w_freqs = torch.outer(torch.tensor(w_ids, dtype=torch.float32), inv_freq)
    return torch.cat([h_freqs, w_freqs], dim=-1)


def _apply_rotary_pos_emb(q_or_k: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1).unsqueeze(1)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1).unsqueeze(1)
    return (q_or_k * cos) + (_rotate_half(q_or_k) * sin)


def _block_attention_mask(grid_thw: torch.Tensor) -> torch.Tensor:
    cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0], dim=0).cumsum(0, dtype=torch.int32)
    s = int(cu[-1].item())
    mask = torch.full((1, s, s), -1e9, dtype=torch.float32)
    start = 0
    for end in cu.tolist():
        mask[:, start:end, start:end] = 0.0
        start = end
    return mask


class _TorchVisionBlock(nn.Module):
    def __init__(self, cfg: types.SimpleNamespace):
        super().__init__()
        hidden_size = cfg.hidden_size
        intermediate_size = cfg.intermediate_size
        self.num_heads = cfg.num_attention_heads
        self.head_dim = hidden_size // self.num_heads
        self.eps = cfg.rms_norm_eps
        self.norm1_weight = nn.Parameter(torch.ones(hidden_size))
        self.norm2_weight = nn.Parameter(torch.ones(hidden_size))
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=cfg.use_bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=cfg.use_bias)
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=cfg.use_bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=cfg.use_bias)
        self.fc3 = nn.Linear(hidden_size, intermediate_size, bias=cfg.use_bias)

    def forward(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        s, d = x.shape
        n1 = _rms_norm(x, self.norm1_weight, self.eps)
        qkv = self.qkv(n1).view(s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)
        q = _apply_rotary_pos_emb(q, rotary_pos_emb)
        k = _apply_rotary_pos_emb(k, rotary_pos_emb)
        qh = q.permute(1, 0, 2)
        kh = k.permute(1, 0, 2)
        vh = v.permute(1, 0, 2)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * (self.head_dim**-0.5)
        probs = torch.softmax(scores + attn_mask, dim=-1)
        attn = torch.matmul(probs, vh).permute(1, 0, 2).reshape(s, d)
        t1 = x + self.proj(attn)
        n2 = _rms_norm(t1, self.norm2_weight, self.eps)
        mlp = F.silu(self.fc1(n2)) * self.fc3(n2)
        return t1 + self.fc2(mlp)


class _TorchDotsVisionTransformer(nn.Module):
    def __init__(self, hf_config: types.SimpleNamespace):
        super().__init__()
        cfg = hf_config.vision_config
        self.cfg = cfg
        in_dim = cfg.num_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        self.patch_proj = nn.Linear(in_dim, cfg.hidden_size, bias=cfg.use_bias)
        self.patch_norm_weight = nn.Parameter(torch.ones(cfg.hidden_size))
        self.blocks = nn.ModuleList([_TorchVisionBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.post_trunk_norm = nn.Parameter(torch.ones(cfg.hidden_size))
        merge_in = (cfg.spatial_merge_size**2) * cfg.hidden_size
        self.merger_ln_q = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.merger_fc1 = nn.Linear(merge_in, cfg.hidden_size, bias=True)
        self.merger_fc2 = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        x = self.patch_proj(pixel_values)
        x = _rms_norm(x, self.patch_norm_weight, self.cfg.rms_norm_eps)
        rotary_pos_emb = _rotary_pos_emb(
            grid_thw,
            num_heads=self.cfg.num_attention_heads,
            hidden_size=self.cfg.hidden_size,
            spatial_merge_size=self.cfg.spatial_merge_size,
        ).to(dtype=x.dtype, device=x.device)
        attn_mask = _block_attention_mask(grid_thw).to(dtype=x.dtype, device=x.device)
        for block in self.blocks:
            x = block(x, rotary_pos_emb, attn_mask)
        if self.cfg.post_norm:
            x = _rms_norm(x, self.post_trunk_norm, self.cfg.rms_norm_eps)
        merge2 = self.cfg.spatial_merge_size**2
        x = self.merger_ln_q(x)
        x = x.reshape(x.shape[0] // merge2, merge2 * x.shape[1])
        x = F.gelu(self.merger_fc1(x))
        x = self.merger_fc2(x)
        return x


def _torch_to_tt_vision_state_dict(
    torch_model: _TorchDotsVisionTransformer, hf_config: types.SimpleNamespace
) -> dict[str, torch.Tensor]:
    cfg = hf_config.vision_config
    state_dict = {}
    prefix = "vision_tower."
    state_dict[f"{prefix}patch_embed.patchifier.proj.weight"] = (
        torch_model.patch_proj.weight.detach()
        .clone()
        .reshape(
            cfg.hidden_size,
            cfg.num_channels,
            cfg.patch_size,
            cfg.patch_size,
        )
    )
    state_dict[f"{prefix}patch_embed.patchifier.norm.weight"] = torch_model.patch_norm_weight.detach().clone()
    for i, block in enumerate(torch_model.blocks):
        block_prefix = f"{prefix}blocks.{i}."
        state_dict[f"{block_prefix}norm1.weight"] = block.norm1_weight.detach().clone()
        state_dict[f"{block_prefix}norm2.weight"] = block.norm2_weight.detach().clone()
        state_dict[f"{block_prefix}attn.qkv.weight"] = block.qkv.weight.detach().clone()
        state_dict[f"{block_prefix}attn.proj.weight"] = block.proj.weight.detach().clone()
        state_dict[f"{block_prefix}mlp.fc1.weight"] = block.fc1.weight.detach().clone()
        state_dict[f"{block_prefix}mlp.fc2.weight"] = block.fc2.weight.detach().clone()
        state_dict[f"{block_prefix}mlp.fc3.weight"] = block.fc3.weight.detach().clone()
    state_dict[f"{prefix}post_trunk_norm.weight"] = torch_model.post_trunk_norm.detach().clone()
    state_dict[f"{prefix}merger.ln_q.weight"] = torch_model.merger_ln_q.weight.detach().clone()
    state_dict[f"{prefix}merger.ln_q.bias"] = torch_model.merger_ln_q.bias.detach().clone()
    state_dict[f"{prefix}merger.mlp.0.weight"] = torch_model.merger_fc1.weight.detach().clone()
    state_dict[f"{prefix}merger.mlp.0.bias"] = torch_model.merger_fc1.bias.detach().clone()
    state_dict[f"{prefix}merger.mlp.2.weight"] = torch_model.merger_fc2.weight.detach().clone()
    state_dict[f"{prefix}merger.mlp.2.bias"] = torch_model.merger_fc2.bias.detach().clone()
    return state_dict


def _tt_output_to_torch(tt_output: ttnn.Tensor, expected_shape: tuple[int, int]) -> torch.Tensor:
    tt_host = ttnn.to_torch(tt_output)
    if tt_host.dim() == 4:
        tt_host = tt_host[0, 0]
    elif tt_host.dim() == 3:
        tt_host = tt_host[0]
    elif tt_host.dim() != 2:
        raise RuntimeError(f"Unexpected TT output rank: {tt_host.dim()} shape={tuple(tt_host.shape)}")

    n_tokens, hidden = expected_shape
    return tt_host[:n_tokens, :hidden].contiguous()


def test_vision_transformer_pcc_dots_torch_vs_tt():
    torch.manual_seed(0)

    device = None
    try:
        device = open_mesh_device()

        hf_config = _make_demo_like_hf_config()
        torch_vision_cfg = hf_config.vision_config
        tt_vision_cfg = _to_tt_vision_config(hf_config)

        torch_model = _TorchDotsVisionTransformer(hf_config).eval()
        tt_model = DotsVisionTransformerTT(
            vision_config=tt_vision_cfg,
            mesh_device=device,
            state_dict=_torch_to_tt_vision_state_dict(torch_model, hf_config),
            dtype=ttnn.bfloat16,
        )

        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32)
        n_patches = int(grid_thw[0, 0] * grid_thw[0, 1] * grid_thw[0, 2])
        patch_flat_dim = (
            torch_vision_cfg.num_channels
            * torch_vision_cfg.temporal_patch_size
            * torch_vision_cfg.patch_size
            * torch_vision_cfg.patch_size
        )
        pixel_values = torch.randn(n_patches, patch_flat_dim, dtype=torch.float32)

        with torch.no_grad():
            torch_output = torch_model(pixel_values, grid_thw).float().cpu()

        mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        pixel_tt = ttnn.from_torch(
            pixel_values,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        grid_tt = ttnn.from_torch(
            grid_thw,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        tt_output = tt_model(pixel_tt, grid_tt)
        tt_output_torch = _tt_output_to_torch(tt_output, tuple(torch_output.shape)).float().cpu()

        assert torch_output.shape == tt_output_torch.shape
        assert torch_output.dtype == tt_output_torch.dtype
        assert_quality(torch_output, tt_output_torch, pcc=VISION_PCC_REQUIRED)
    finally:
        if ttnn is not None:
            if "pixel_tt" in locals():
                ttnn.deallocate(pixel_tt)
            if "grid_tt" in locals():
                ttnn.deallocate(grid_tt)
            if "tt_output" in locals() and isinstance(tt_output, ttnn.Tensor):
                ttnn.deallocate(tt_output)
        if device is not None:
            close_dots_mesh_device(device)
