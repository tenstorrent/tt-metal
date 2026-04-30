# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.demos.dots_ocr.reference.dots_ocr_weights.configuration_dots import DotsVisionConfig as HFDotsVisionConfig
from models.demos.dots_ocr.reference.dots_ocr_weights.modeling_dots_vision import DotsVisionTransformer
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


def _torch_to_tt_vision_state_dict(torch_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # TT vision stack defaults to "vision_tower." key prefix.
    return {f"vision_tower.{k}": v.detach().clone() for k, v in torch_state_dict.items()}


def _to_tt_vision_config(cfg) -> TTDotsVisionConfig:
    """Convert HF-style vision config object/dict into TT dataclass config."""
    if isinstance(cfg, TTDotsVisionConfig):
        return cfg
    if hasattr(cfg, "to_dict"):
        cfg_dict = cfg.to_dict()
    elif isinstance(cfg, dict):
        cfg_dict = dict(cfg)
    else:
        cfg_dict = dict(getattr(cfg, "__dict__", {}))
    tt_keys = set(TTDotsVisionConfig.__dataclass_fields__.keys())
    return TTDotsVisionConfig(**{k: v for k, v in cfg_dict.items() if k in tt_keys})


def _tt_output_to_torch(tt_output: ttnn.Tensor, expected_shape: tuple[int, int]) -> torch.Tensor:
    tt_host = ttnn.to_torch(tt_output)
    if tt_host.dim() == 4:
        tt_host = tt_host[0, 0]
    elif tt_host.dim() == 3:
        tt_host = tt_host[0]
    elif tt_host.dim() != 2:
        raise RuntimeError(f"Unexpected TT output rank: {tt_host.dim()} shape={tuple(tt_host.shape)}")

    # Some TT tensors can be physically tile-padded; slice back to the logical shape.
    n_tokens, hidden = expected_shape
    return tt_host[:n_tokens, :hidden].contiguous()


def test_vision_transformer_pcc_dots_torch_vs_tt():
    torch.manual_seed(0)

    device = None
    try:
        device = open_mesh_device()

        # Small deterministic config for a fast yet meaningful parity test.
        # Keep this test small, but use TT-safe attention geometry:
        # head_dim = embed_dim / num_heads should avoid SDPA head-dim padding paths.
        vision_cfg = HFDotsVisionConfig(
            embed_dim=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            use_bias=False,
            post_norm=True,
            attn_implementation="eager",
        )
        tt_vision_cfg = _to_tt_vision_config(vision_cfg)

        torch_model = DotsVisionTransformer(vision_cfg).eval()
        tt_model = DotsVisionTransformerTT(
            vision_config=tt_vision_cfg,
            mesh_device=device,
            state_dict=_torch_to_tt_vision_state_dict(torch_model.state_dict()),
            dtype=ttnn.bfloat16,
        )

        # One image grid with 16 patch tokens before merge -> 4 rows after merger.
        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int32)
        n_patches = int(grid_thw[0, 0] * grid_thw[0, 1] * grid_thw[0, 2])
        patch_flat_dim = (
            vision_cfg.num_channels * vision_cfg.temporal_patch_size * vision_cfg.patch_size * vision_cfg.patch_size
        )
        pixel_values = torch.randn(n_patches, patch_flat_dim, dtype=torch.float32)

        with torch.no_grad():
            # Run torch in fp32 on host for stable CPU behavior.
            torch_output = torch_model(pixel_values, grid_thw, bf16=False).float().cpu()

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

        # Explicitly align before quality assertion.
        assert torch_output.shape == tt_output_torch.shape
        assert torch_output.dtype == tt_output_torch.dtype

        assert_quality(
            torch_output,
            tt_output_torch,
            pcc=VISION_PCC_REQUIRED,
        )
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
