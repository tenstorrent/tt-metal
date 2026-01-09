# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import ExitStack

import numpy as np
import pytest
import torch
import math

from models.experimental.maskformer_swin.tt.backbone_swin import MaskFormerSwinBackbone
from models.experimental.maskformer_swin.tt.parity import ParityConfig, compare_tensors
from models.experimental.maskformer_swin.tt.ttnn_compat import get_default_dtype, ttnn
from models.experimental.maskformer_swin.tt.weights import (
    WeightConversionConfig,
    convert_state_dict_to_tt,
    download_reference_weights,
)

try:
    from transformers.models.maskformer.modeling_maskformer_swin import window_partition
except ModuleNotFoundError:  # pragma: no cover - transformers is optional for these tests.
    window_partition = None  # type: ignore[assignment]


def _skip_reason() -> str | None:
    if os.environ.get("MASKFORMER_RUN_WEIGHT_TESTS") != "1":
        return "Set MASKFORMER_RUN_WEIGHT_TESTS=1 to enable weight-dependent parity tests."
    if window_partition is None:
        return "transformers is required for the MaskFormer Swin parity harness."
    if ttnn is None or not hasattr(ttnn, "open_device"):
        return "TTNN runtime with open_device is required for the window attention parity harness."
    return None


@pytest.mark.skipif(_skip_reason() is not None, reason="TTNN window attention parity requires optional dependencies.")
def test_stage1_shifted_window_attention_parity():
    """Validate the manual TTNN window attention path against HuggingFace for a single shifted window."""

    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    # Deterministic input for reproducibility.
    torch.manual_seed(0)

    weight_cfg = WeightConversionConfig()
    reference = download_reference_weights(weight_cfg)
    tt_state = convert_state_dict_to_tt(reference.state_dict, weight_cfg)
    backbone_cfg = reference.config.get("backbone_config", {})

    device = ttnn.open_device(device_id=0)
    dtype = get_default_dtype()

    with ExitStack() as stack:
        stack.callback(lambda: ttnn.close_device(device))
        backbone = MaskFormerSwinBackbone.from_huggingface(tt_state, device=device, config_dict=backbone_cfg)

        hf_model = backbone._hf_backbone_model.model  # type: ignore[assignment]

        # Run patch embedding + stage0 in PyTorch to capture the window inputs for block 1 (shifted block).
        height, width = backbone.config.image_size
        images = torch.randn(1, 3, height, width)

        embeddings, dims = hf_model.embeddings(images, interpolate_pos_encoding=False)
        stage = hf_model.encoder.layers[0]
        _, _, hidden_states_seq = stage(embeddings, dims, output_hidden_states=True)

        block_index = 1  # shifted block in stage 1
        block = stage.blocks[block_index]
        hidden_states = hidden_states_seq[block_index * 2]  # pre-attention hidden states for this block

        seq_len = hidden_states.shape[1]
        channels = hidden_states.shape[2]
        window_size = block.window_size
        shift_size = block.shift_size

        hidden_states = block.layernorm_before(hidden_states)
        hidden_states = hidden_states.view(1, dims[0], dims[1], channels)
        padded, pad_values = block.maybe_pad(hidden_states, dims[0], dims[1])
        padded_height = padded.shape[1]
        padded_width = padded.shape[2]
        if shift_size > 0:
            padded = torch.roll(padded, shifts=(-shift_size, -shift_size), dims=(1, 2))
        windows = window_partition(padded, window_size)
        windows = windows.view(-1, window_size * window_size, channels)

        # HuggingFace reference attention for the first window (captures shift mask application).
        debug_idx = 93
        attn_mask = block.get_attn_mask((padded_height, padded_width))
        if attn_mask is not None:
            attn_mask = attn_mask.to(windows.device)
            print("[hf_mask_debug] mask_shape:", tuple(attn_mask.shape))
            try:
                print("[hf_mask_debug] row93 first10:", attn_mask[:1, :, debug_idx, :10].detach().cpu().numpy())
            except Exception as e:
                print("[hf_mask_debug] failed to slice row93:", e)
        hf_attn = block.attention(windows, attention_mask=attn_mask)[0]
        # Quantize HF reference to BF16 to match TTNN compute dtype before comparison.
        hf_window = (hf_attn[:1].detach().to(dtype=torch.bfloat16).to(dtype=torch.float32)).cpu().numpy()

        tt_windows = ttnn.from_torch(
            windows[:1],
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        params = backbone._prepare_stage_blocks(stage_index=0)[block_index]

        query_weight_torch = ttnn.to_torch(params.query_weight)
        query_bias_torch = ttnn.to_torch(params.query_bias)
        print(
            "[debug] query_weight",
            tuple(query_weight_torch.shape),
            "query_bias",
            tuple(query_bias_torch.shape),
        )
        key_weight_torch = ttnn.to_torch(params.key_weight)
        key_bias_torch = ttnn.to_torch(params.key_bias)
        value_weight_torch = ttnn.to_torch(params.value_weight)
        value_bias_torch = ttnn.to_torch(params.value_bias)

        hf_q = block.attention.self.query(windows[:1])
        hf_k = block.attention.self.key(windows[:1])
        hf_v = block.attention.self.value(windows[:1])

        def _matmul_bias(x, weight, bias):
            weight_2d = weight
            if weight_2d.ndim == 3:
                weight_2d = weight_2d.squeeze(0)
            weight_2d = weight_2d.to(dtype=torch.float32)
            bias_1d = bias.view(-1)
            bias_1d = bias_1d.to(dtype=torch.float32)
            return torch.matmul(x, weight_2d.t()) + bias_1d

        tt_q = _matmul_bias(windows[:1], query_weight_torch, query_bias_torch)
        tt_k = _matmul_bias(windows[:1], key_weight_torch, key_bias_torch)
        tt_v = _matmul_bias(windows[:1], value_weight_torch, value_bias_torch)
        print(
            "[debug] linear delta",
            float((tt_q - hf_q).abs().max()),
            float((tt_k - hf_k).abs().max()),
            float((tt_v - hf_v).abs().max()),
        )

        # Debug: compare relative position bias row against TT's BF16 bias injection
        def _compute_rel_bias(table: torch.Tensor, window_size: int, num_heads: int) -> torch.Tensor:
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            try:
                coord_mesh = torch.meshgrid(coords_h, coords_w, indexing="ij")
            except TypeError:
                coord_mesh = torch.meshgrid(coords_h, coords_w)
            coords = torch.stack(coord_mesh)
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size - 1
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)
            bias = table[relative_position_index.view(-1)]
            bias = bias.view(window_size * window_size, window_size * window_size, num_heads)
            return bias.permute(2, 0, 1).contiguous()

        rel_bias = _compute_rel_bias(
            block.attention.self.relative_position_bias_table,
            window_size,
            params.num_heads,
        )
        head0_bias_row = rel_bias[0, debug_idx, :10].detach().cpu().numpy()
        head0_bias_row_bf16 = (
            rel_bias[0, debug_idx, :10].detach().to(dtype=torch.bfloat16).to(dtype=torch.float32).cpu().numpy()
        )
        print("[hf_bias_debug] head0 row93 first10 fp32:", head0_bias_row)
        print("[hf_bias_debug] head0 row93 first10 bf16->fp32:", head0_bias_row_bf16)

        tt_output = backbone._run_window_attention_streaming(
            tt_windows,
            None,
            params,
            window_size,
            shift_size,
            stage_index=0,
            block_index=block_index,
            padded_height=padded_height,
            padded_width=padded_width,
            batch=1,
        )
        torch_tt_window = ttnn.to_torch(tt_output).to(dtype=torch.float32)
        tt_window = torch_tt_window.detach().cpu().numpy()

        # Compute HF per-head logits and softmax for row 93 to compare numerically
        with torch.no_grad():
            num_heads = params.num_heads
            head_dim = channels // num_heads
            scale = 1.0 / math.sqrt(head_dim)
            hf_qh = hf_q.view(1, window_size * window_size, num_heads, head_dim).permute(0, 2, 1, 3)
            hf_kh = hf_k.view(1, window_size * window_size, num_heads, head_dim).permute(0, 2, 1, 3)
            logits0 = (hf_qh[0, 0] @ hf_kh[0, 0].transpose(0, 1)) * scale
            rel_bias = rel_bias[0]
            logits0 = logits0 + rel_bias
            if attn_mask is not None:
                logits0 = logits0 + attn_mask[:1, :, :]
            probs0 = torch.softmax(logits0, dim=-1)
            print("[hf_row_debug] logits0_row93_first20:", logits0[0, 93, :20].detach().cpu().numpy())
            print("[hf_row_debug] probs0_row93_first20:", probs0[0, 93, :20].detach().cpu().numpy())

        # Compare TT output against HuggingFace reference using PCC / max-abs metrics.
        config = ParityConfig()
        print("[window_debug] hf_row93_first20:", hf_window[0, 93, :20])
        print("[window_debug] tt_row93_first20:", tt_window[0, 93, :20])
        proj_bias = ttnn.to_torch(params.proj_bias).view(-1)
        print("[window_debug] proj_bias_first20:", proj_bias[:20].detach().cpu().to(dtype=torch.float32).numpy())
        diff = tt_window - hf_window
        max_idx = np.unravel_index(np.abs(diff).argmax(), diff.shape)
        max_diff = diff[max_idx]
        ref_at_idx = hf_window[max_idx]
        test_at_idx = tt_window[max_idx]
        pcc, max_abs = compare_tensors(
            ref=hf_window.reshape(-1),
            test=tt_window.reshape(-1),
            pcc_threshold=config.pcc_threshold,
            max_abs_threshold=config.max_abs_threshold,
        )
        print(
            "[window_parity] PCC="
            f"{pcc:.6f} max_abs={max_abs:.6f} "
            f"max_idx={max_idx} "
            f"ref={ref_at_idx:.6f} tt={test_at_idx:.6f} delta={max_diff:.6f}"
        )
        assert pcc >= config.pcc_threshold
        assert max_abs <= config.max_abs_threshold
