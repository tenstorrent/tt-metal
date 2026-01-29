# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DPTLargeConfig:
    """
    Configuration holder for DPT-Large bring-up.

    The defaults mirror Hugging Face's  checkpoint, but tests
    can override them with smaller shapes to avoid heavy downloads.
    """

    model_name: str = "Intel/dpt-large"
    image_size: int = 384
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-6
    dropout: float = 0.0
    output_layers: List[int] = field(default_factory=lambda: [5, 11, 17, 23])
    # DPT decoder/neck/head hyperparams (mirror HF DPT defaults)
    fusion_hidden_size: int = 256
    neck_hidden_sizes: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    reassemble_factors: List[float] = field(default_factory=lambda: [4.0, 2.0, 1.0, 0.5])
    readout_type: str = "project"
    hidden_act: str = "gelu"
    use_batch_norm_in_fusion_residual: bool = True
    add_projection: bool = False
    head_in_index: int = -1

    # TT specific knobs
    device: str = "wormhole_n300"
    dtype: str = "bfloat16"
    # When True the TT pipeline falls back to CPU to keep local/dev envs working.
    allow_cpu_fallback: bool = True
    # Whether to actually allocate TT device. Tests can set False to skip.
    enable_tt_device: bool = False
    # Prefer device-native execution for neck and head where possible (guarded for parity)
    tt_device_reassembly: bool = False
    tt_device_fusion: bool = False
    # Fast/perf encoder flag (sharded, L1, fused ops)
    tt_perf_encoder: bool = False
    # Fast/perf neck+head flag (device-first convs/upsample)
    tt_perf_neck: bool = False

    tt_force_default_attention_programs: bool = False

    def to_hf_kwargs(self) -> Dict:
        """Return kwargs to build a DPTConfig without downloading weights.

        Important: for small-layer test configs (e.g., num_hidden_layers=2),
        Hugging Face's DPT neck still expects four feature maps by default. We
        keep the decoder/neck shapes at HF defaults, but we clamp the
        out_indices to the valid range of encoder layers so that HFF can
        always gather the requested number of hidden states (allowing
        duplicates when the encoder has fewer than four blocks).
        """
        # Preferred output layers (clamped to encoder depth for HF)
        base_out_indices = list(self.output_layers)
        # Ensure HF backbone has enough layers for these taps
        needed_layers = max(base_out_indices) + 1 if base_out_indices else self.num_hidden_layers
        max_layer_idx = max(0, max(self.num_hidden_layers, needed_layers) - 1)
        safe_out_indices = [min(i, max_layer_idx) for i in base_out_indices]

        return dict(
            use_batch_norm_in_fusion_residual=self.use_batch_norm_in_fusion_residual,
            add_projection=self.add_projection,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=max_layer_idx + 1,
            num_attention_heads=self.num_attention_heads,
            qkv_bias=self.qkv_bias,
            layer_norm_eps=self.layer_norm_eps,
            dropout=self.dropout,
            backbone_out_indices=safe_out_indices,
            # Note: keep decoder params at HF defaults for stability
        )


# Convenience default instance used by runner/tests.
DEFAULT_CONFIG = DPTLargeConfig()
