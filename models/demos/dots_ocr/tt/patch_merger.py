# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from models.demos.dots_ocr.tt._ttnn_import import get_ttnn

ttnn = get_ttnn()
_HAS_TTNN = ttnn is not None

from models.common.lightweightmodule import LightweightModule
from models.demos.dots_ocr.tt.vision_rmsnorm import RMSNorm


def dots_merger_state_dict_for_patch_merger_tt(hf_sd: dict) -> dict:
    """
    Subset of HF weights for TT `PatchMerger`, with `mlp.*` mapped to `feed_forward.*`.

    This is a small compatibility shim for checkpoints that name the merger MLP as
    `vision_tower.(patch_)merger.mlp.{0,2}.*` instead of `feed_forward.{0,2}.*`.
    """

    def _pick(*candidates: str):
        for c in candidates:
            if c in hf_sd:
                return c
        return None

    state_dict: dict = {}
    for k, v in hf_sd.items():
        if k.startswith("vision_tower.merger.") or k.startswith("vision_tower.patch_merger."):
            state_dict[k] = v
    for i in (0, 2):
        for suffix in ("weight", "bias"):
            src = _pick(
                f"vision_tower.merger.mlp.{i}.{suffix}",
                f"vision_tower.patch_merger.mlp.{i}.{suffix}",
                f"model.vision_tower.merger.mlp.{i}.{suffix}",
                f"model.vision_tower.patch_merger.mlp.{i}.{suffix}",
            )
            if src is not None:
                state_dict[f"vision_tower.merger.feed_forward.{i}.{suffix}"] = hf_sd[src]
    return state_dict


# Backward-compatible private name (older tests / imports).
_dots_merger_state_dict_for_patch_merger_tt = dots_merger_state_dict_for_patch_merger_tt


class PatchMerger(LightweightModule):
    """
    TTNN PatchMerger (HF ``vision_tower`` patch-merger layout) for Dots OCR.

    Input:  x [B, 1, S_patch, H_v]
    Output: y [B, 1, S_img, H_out], where S_img = S_patch / (spatial_merge_size^2)
    """

    def __init__(
        self,
        mesh_device,
        *,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int,
        state_dict,
        state_dict_prefix: str,
        weight_cache_path,
        dtype,
    ):
        super().__init__()
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.hidden_size = hidden_size
        self.out_hidden_size = out_hidden_size
        self.spatial_merge_size = spatial_merge_size

        self.mlp_size = hidden_size * (spatial_merge_size**2)

        # HF `PatchMerger` defaults to `nn.LayerNorm` on `context_dim` (`ln_q`), not RMSNorm.
        # Using RMSNorm here tanked PCC vs HF. Prefer LayerNorm when checkpoint has bias.
        weight_dtype = ttnn.bfloat16 if _HAS_TTNN and hasattr(ttnn, "bfloat16") else torch.bfloat16
        ln_w = f"{state_dict_prefix}.ln_q.weight"
        ln_b = f"{state_dict_prefix}.ln_q.bias"
        if ln_w in state_dict and ln_b in state_dict:
            self._merge_pre_norm = "layernorm"
            w_bf = state_dict[ln_w].clone().to(torch.bfloat16)
            b_bf = state_dict[ln_b].clone().to(torch.bfloat16)
            # Host reference only (e.g. HF golden); TT path uses ``_ln_q_*_tt`` below.
            self._ln_q_weight = w_bf
            self._ln_q_bias = b_bf
            self.norm = None
            self._ln_q_w_ttnn = self._ln_q_b_ttnn = None
            if mesh_device is not None and ttnn is not None and hasattr(ttnn, "as_tensor"):
                mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
                mm = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
                self._ln_q_w_ttnn = ttnn.as_tensor(
                    w_bf,
                    dtype=weight_dtype,
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mem,
                    mesh_mapper=mm,
                )
                self._ln_q_b_ttnn = ttnn.as_tensor(
                    b_bf,
                    dtype=weight_dtype,
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mem,
                    mesh_mapper=mm,
                )
        else:
            self._merge_pre_norm = "rmsnorm"
            self.norm = RMSNorm(
                device=mesh_device,
                dim=hidden_size,
                state_dict=state_dict,
                state_dict_prefix=state_dict_prefix + ".",
                weight_key="ln_q",
                weight_dtype=weight_dtype,
                eps=1e-6,
            )

        def _pick_key(*candidates: str) -> str:
            for c in candidates:
                if c in self.state_dict:
                    return c
            raise KeyError(f"None of the candidate keys exist in state_dict: {candidates}")

        def torch_weight(name: str):
            # HF checkpoints have used both:
            # - {prefix}.feed_forward.{0,2}.weight  (our expected layout)
            # - {prefix}.mlp.{0,2}.weight          (HF module layout)
            prefixes = [state_dict_prefix]
            # Checkpoints sometimes use "vision_tower.patch_merger.*" while callers pass "vision_tower.merger".
            if state_dict_prefix.endswith(".merger"):
                prefixes.append(state_dict_prefix[: -len(".merger")] + ".patch_merger")
            elif state_dict_prefix.endswith(".patch_merger"):
                prefixes.append(state_dict_prefix[: -len(".patch_merger")] + ".merger")

            candidates: list[str] = []
            for p in prefixes:
                candidates.append(f"{p}.{name}.weight")
                if name.startswith("feed_forward."):
                    candidates.append(f"{p}.mlp.{name.split('.')[-1]}.weight")

            key = _pick_key(*candidates)
            return torch.transpose(self.state_dict[key], -2, -1)

        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        def as_weight_tensor(name: str, weight_dtype):
            # Use DRAM_MEMORY_CONFIG if available, otherwise fallback for test environments
            memory_config = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
            mesh_mapper = (
                ttnn.ReplicateTensorToMesh(self.mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
            )

            return ttnn.as_tensor(
                torch_weight(name),
                dtype=weight_dtype,
                device=self.mesh_device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
                cache_file_name=cache_name(name),
            )

        # feed_forward.0: mlp_size -> mlp_size
        # feed_forward.2: mlp_size -> out_hidden_size
        self.w1 = as_weight_tensor("feed_forward.0", dtype)
        self.w2 = as_weight_tensor("feed_forward.2", dtype)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if not _HAS_TTNN or ttnn is None:
            raise RuntimeError("PatchMerger TT path requires ttnn")
        if getattr(self, "_merge_pre_norm", "rmsnorm") == "layernorm":
            if self._ln_q_w_ttnn is None or self._ln_q_b_ttnn is None:
                raise ValueError(
                    "ln_q is LayerNorm in checkpoint: upload ln_q weight/bias to device (requires mesh_device)."
                )
            x = ttnn.layer_norm(
                x,
                weight=self._ln_q_w_ttnn,
                bias=self._ln_q_b_ttnn,
                epsilon=1e-6,
            )
        else:
            x = self.norm(x)
        # Merge spatial dimensions into channel dim: [B, 1, S_patch, H] -> [B, 1, S_img, H*m^2]
        # Workaround for reshape tilized hangs: convert to row-major first.
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (x.shape[0], x.shape[1], -1, self.mlp_size))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Use DRAM_MEMORY_CONFIG if available, otherwise fallback for test environments
        memory_config = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)

        x = ttnn.linear(x, self.w1, compute_kernel_config=None, memory_config=memory_config)
        x = ttnn.gelu(x)
        x = ttnn.linear(x, self.w2, compute_kernel_config=None, memory_config=memory_config)
        return x
