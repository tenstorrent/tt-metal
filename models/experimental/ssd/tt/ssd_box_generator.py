# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import ttnn
import math
from typing import (
    List,
    Optional,
)

from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor_rm


class TtDefaultBoxGenerator(nn.Module):
    def __init__(
        self,
        aspect_ratios: List[List[int]],
        min_ratio: float = 0.15,
        max_ratio: float = 0.9,
        scales: Optional[List[float]] = None,
        steps: Optional[List[int]] = None,
        clip: bool = True,
        device=None,
    ):
        super().__init__()
        if steps is not None and len(aspect_ratios) != len(steps):
            raise ValueError("aspect_ratios and steps should have the same length")
        self.aspect_ratios = aspect_ratios
        self.steps = steps
        self.clip = clip
        self.device = device
        num_outputs = len(aspect_ratios)

        # Estimation of default boxes scales
        if scales is None:
            if num_outputs > 1:
                range_ratio = max_ratio - min_ratio
                self.scales = [min_ratio + range_ratio * k / (num_outputs - 1.0) for k in range(num_outputs)]
                self.scales.append(1.0)
            else:
                self.scales = [min_ratio, max_ratio]
        else:
            self.scales = scales

        self._wh_pairs = self._generate_wh_pairs(num_outputs)

    def _generate_wh_pairs(
        self,
        num_outputs: int,
    ) -> List[ttnn.Tensor]:
        _wh_pairs: List[ttnn.Tensor] = []
        for k in range(num_outputs):
            # Adding the 2 default width-height pairs for aspect ratio 1 and scale s'k
            s_k = self.scales[k]
            s_prime_k = math.sqrt(self.scales[k] * self.scales[k + 1])
            wh_pairs = [[s_k, s_k], [s_prime_k, s_prime_k]]

            # Adding 2 pairs for each aspect ratio of the feature map k
            for ar in self.aspect_ratios[k]:
                sq_ar = math.sqrt(ar)
                w = self.scales[k] * sq_ar
                h = self.scales[k] / sq_ar
                wh_pairs.extend([[w, h], [h, w]])
            wh_pairs_list = []
            for row in wh_pairs:
                for element in row:
                    wh_pairs_list.append(element)
            tt_wh_pairs = ttnn.Tensor(
                wh_pairs_list,
                [1, 1, len(wh_pairs), len(wh_pairs[0])],
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
                self.device,
            )
            _wh_pairs.append(tt_wh_pairs)
        return _wh_pairs

    def num_anchors_per_location(self) -> List[int]:
        # Estimate num of anchors based on aspect ratios: 2 default boxes + 2 * ratios of feaure map.
        return [2 + 2 * len(r) for r in self.aspect_ratios]

    # Default Boxes calculation based on page 6 of SSD paper
    def _grid_default_boxes(
        self,
        grid_sizes: List[List[int]],
        image_size: List[int],
    ) -> ttnn.Tensor:
        default_boxes = []
        for k, f_k in enumerate(grid_sizes):
            # Now add the default boxes for each width-height pair
            if self.steps is not None:
                x_f_k = image_size[1] / self.steps[k]
                y_f_k = image_size[0] / self.steps[k]
            else:
                y_f_k, x_f_k = f_k

            shifts_x = (torch.arange(0, f_k[1]) + 0.5) / x_f_k
            shifts_y = (torch.arange(0, f_k[0]) + 0.5) / y_f_k
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack((shift_x, shift_y) * 6, dim=-1).reshape(-1, 2)
            # Clipping the default boxes while the boxes are encoded in format (cx, cy, w, h)
            if self.clip:
                pt_tensor = tt_to_torch_tensor(self._wh_pairs[k]).squeeze(0).squeeze(0)
                _wh_pair = pt_tensor.clamp(min=0, max=1)
            else:
                pt_tensor = tt_to_torch_tensor(self._wh_pairs[k]).squeeze(0).squeeze(0)
                _wh_pair = pt_tensor

            wh_pairs = _wh_pair.repeat((f_k[0] * f_k[1]), 1)
            # Since Both inputs are pytorch tensor, torch op is used
            default_box = torch.cat((shifts, wh_pairs), dim=1)
            default_boxes.append(default_box)

        default_boxes_tensor = torch.cat(default_boxes, dim=0)
        default_boxes_tensor = torch_to_tt_tensor_rm(default_boxes_tensor, self.device)
        return default_boxes_tensor

    def forward(self, image: ttnn.Tensor, feature_maps: List[ttnn.Tensor]) -> List[ttnn.Tensor]:
        grid_sizes = [feature_map.get_legacy_shape()[-2:] for feature_map in feature_maps]
        image_size = image.get_legacy_shape()[-2:]
        image_sizes = [image_size]
        default_boxes = self._grid_default_boxes(grid_sizes, image_size)
        default_boxes = tt_to_torch_tensor(default_boxes).squeeze(0).squeeze(0)
        dboxes = []
        x_y_size = torch.tensor([image_size[1], image_size[0]], device=default_boxes.device)
        for _ in image_sizes:
            dboxes_in_image = default_boxes
            dboxes_in_image = torch.cat(
                [
                    (dboxes_in_image[:, :2] - 0.5 * dboxes_in_image[:, 2:]) * x_y_size,
                    (dboxes_in_image[:, :2] + 0.5 * dboxes_in_image[:, 2:]) * x_y_size,
                ],
                -1,
            )

            dboxes_in_image = torch_to_tt_tensor_rm(dboxes_in_image, self.device)
            dboxes.append(dboxes_in_image)
        return dboxes
