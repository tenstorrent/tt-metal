# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference for DINO-5scale Swin-L: loads mmdet model and exposes
per-stage forward so TTNN submodules can be compared (PCC) stage by stage.

Requires: mmdet, mmengine, mmcv installed (e.g. mim install mmdet).
Config and checkpoint paths are passed at init; checkpoint can be from
  mim download mmdet --config dino-5scale_swin-l_8xb2-36e_coco --dest checkpoints/dino_5scale_swin_l
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def _import_mmdet():
    try:
        from mmdet.apis import init_detector
        from mmengine.config import Config

        return init_detector, Config
    except ImportError as e:
        raise ImportError("DINO reference needs mmdet. Install with: pip install openmim && mim install mmdet") from e


class DINOStagedForward:
    """
    Wraps mmdet DINO model and exposes staged forward for PCC:
      - forward_backbone(x) -> list of 4 feature maps
      - forward_neck(feats) -> memory, spatial_shapes, level_start_index
      - forward_encoder(memory, ...) -> encoder_out
      - forward_decoder(...) -> decoder_out
      - forward_heads(...) -> logits, boxes
    """

    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cpu"):
        init_detector, Config = _import_mmdet()
        config_path = Path(config_path)
        checkpoint_path = Path(checkpoint_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.cfg = Config.fromfile(str(config_path))
        self.model = init_detector(str(config_path), str(checkpoint_path), device=device)
        self.model.eval()
        self.device = device

    def forward_backbone(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Input x: [B, 3, H, W]. Returns list of 4 feature maps (C2, C3, C4, C5)."""
        with torch.no_grad():
            feats = self.model.backbone(x)
        return list(feats)

    def forward_neck(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        feats: list of 4 tensors from backbone.
        Returns:
          memory: [B, N, 256] flattened multi-scale features
          spatial_shapes: [num_levels, 2] (H, W) per level
          level_start_index: [num_levels+1] cumulative start indices
        """
        with torch.no_grad():
            neck_out = self.model.neck(feats)
        # mmdet ChannelMapper returns tuple of 5 tensors; flatten to memory
        if isinstance(neck_out, (list, tuple)):
            all_feats = neck_out
        else:
            all_feats = [neck_out]
        spatial_shapes_list = []
        level_start_index_list = [0]
        flat_list = []
        for f in all_feats:
            b, c, h, w = f.shape
            spatial_shapes_list.append([h, w])
            n = h * w
            level_start_index_list.append(level_start_index_list[-1] + n)
            flat_list.append(f.flatten(2).permute(0, 2, 1))  # [B, n, C]
        memory = torch.cat(flat_list, dim=1)
        spatial_shapes = torch.tensor(spatial_shapes_list, dtype=torch.long, device=memory.device)
        level_start_index = torch.tensor(level_start_index_list, dtype=torch.long, device=memory.device)
        return memory, spatial_shapes, level_start_index

    def forward_encoder(
        self,
        img_feats: List[torch.Tensor],
        batch_input_shape: Tuple[int, int] = (800, 1333),
        img_shape: Tuple[int, int] = (800, 1333),
    ) -> Dict[str, torch.Tensor]:
        """
        Run pre_transformer + encoder on multi-scale feature maps from neck.

        Args:
            img_feats: list of 5 tensors [B, 256, H_i, W_i] from neck.
            batch_input_shape: (H, W) of padded input image.
            img_shape: (H, W) of the actual image (before padding).

        Returns dict with:
            memory: [B, N, 256] encoder output
            memory_mask: [B, N] or None
            spatial_shapes: [num_levels, 2] (H, W)
            level_start_index: [num_levels]
            valid_ratios: [B, num_levels, 2]
            feat_pos: [B, N, 256] positional encoding + level embed
        """
        from mmdet.structures import DetDataSample

        data_sample = DetDataSample()
        data_sample.set_metainfo(
            dict(
                batch_input_shape=batch_input_shape,
                img_shape=img_shape,
            )
        )
        batch_data_samples = [data_sample]

        with torch.no_grad():
            det = self.model
            encoder_inputs_dict, decoder_inputs_dict = det.pre_transformer(img_feats, batch_data_samples)

            encoder_outputs_dict = det.forward_encoder(**encoder_inputs_dict)

        return {
            "memory": encoder_outputs_dict["memory"],
            "memory_mask": encoder_outputs_dict["memory_mask"],
            "spatial_shapes": encoder_outputs_dict["spatial_shapes"],
            "valid_ratios": encoder_inputs_dict["valid_ratios"],
            "level_start_index": encoder_inputs_dict["level_start_index"],
            "feat_pos": encoder_inputs_dict["feat_pos"],
        }

    def forward_decoder(
        self,
        img_feats: List[torch.Tensor],
        batch_input_shape: Tuple[int, int] = (800, 1333),
        img_shape: Tuple[int, int] = (800, 1333),
    ) -> Dict[str, torch.Tensor]:
        """
        Run backbone -> neck -> pre_transformer -> encoder -> pre_decoder -> decoder.

        Returns dict with:
            hidden_states: [num_layers, B, num_queries, 256]
            references: list of [B, num_queries, 4] reference points per layer
            memory: [B, N, 256] encoder output
            query: [B, num_queries, 256] initial query content
            reference_points_init: [B, num_queries, 4] initial reference points
        """
        from mmdet.structures import DetDataSample

        data_sample = DetDataSample()
        data_sample.set_metainfo(
            dict(
                batch_input_shape=batch_input_shape,
                img_shape=img_shape,
            )
        )
        batch_data_samples = [data_sample]

        with torch.no_grad():
            det = self.model
            encoder_inputs_dict, decoder_inputs_dict = det.pre_transformer(img_feats, batch_data_samples)
            encoder_outputs_dict = det.forward_encoder(**encoder_inputs_dict)

            tmp_dec_in, head_inputs_dict = det.pre_decoder(
                **encoder_outputs_dict, batch_data_samples=batch_data_samples
            )
            decoder_inputs_dict.update(tmp_dec_in)

            decoder_outputs_dict = det.forward_decoder(**decoder_inputs_dict)

        return {
            "hidden_states": decoder_outputs_dict["hidden_states"],
            "references": decoder_outputs_dict["references"],
            "memory": encoder_outputs_dict["memory"],
            "query": decoder_inputs_dict["query"],
            "reference_points_init": decoder_inputs_dict["reference_points"],
            "spatial_shapes": encoder_outputs_dict["spatial_shapes"],
            "valid_ratios": encoder_inputs_dict["valid_ratios"],
            "level_start_index": encoder_inputs_dict["level_start_index"],
        }

    def forward_full(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Single image forward; returns dict with pred_instances (bboxes, scores, labels)
        and optionally intermediates if we hook later.
        """
        with torch.no_grad():
            from mmdet.apis import inference_detector

            result = inference_detector(self.model, x)
        return {
            "pred_instances": result.pred_instances,
            "data_sample": result,
        }
