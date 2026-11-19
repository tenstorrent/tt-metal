# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ego-history trajectory tokens: port of the NVlabs Alpamayo 2 Edge delta tokenizer and `fuse_traj_tokens`.

Only the inference path is ported: history XYZ deltas (no yaw) fused into the prompt's history placeholders.
"""

import torch


class DeltaTrajectoryTokenizer:
    """Quantizes per-step XYZ deltas of a trajectory into `num_bins` bins per axis."""

    def __init__(
        self,
        ego_xyz_min: tuple[float, float, float] = (-4, -4, -10),
        ego_xyz_max: tuple[float, float, float] = (4, 4, 10),
        num_bins: int = 1000,
        predict_yaw: bool = False,
        pad_origin_at_beginning: bool = True,
    ):
        """`pad_origin_at_beginning=False` encodes a history that ends at the ego origin: T poses give T-1 deltas."""
        if predict_yaw:
            raise NotImplementedError("Yaw tokens are not used by the Alpamayo 2 Edge history tokenizer")
        self.ego_xyz_min = ego_xyz_min
        self.ego_xyz_max = ego_xyz_max
        self.num_bins = num_bins
        self.pad_origin_at_beginning = pad_origin_at_beginning

    def encode(self, xyz: torch.Tensor) -> torch.Tensor:
        """[B, T, 3] positions -> [B, 3 * deltas] token indices in [0, num_bins)."""
        if self.pad_origin_at_beginning:
            xyz = torch.nn.functional.pad(xyz, [0, 0, 1, 0, 0, 0])
        delta = xyz[:, 1:] - xyz[:, :-1]
        lo = torch.tensor(self.ego_xyz_min, dtype=delta.dtype, device=delta.device)
        hi = torch.tensor(self.ego_xyz_max, dtype=delta.dtype, device=delta.device)
        delta = (delta - lo) / (hi - lo)
        bins = (delta * (self.num_bins - 1)).round().long().clamp(0, self.num_bins - 1)
        return bins.reshape(bins.shape[0], -1)


def replace_pad_token(input_ids: torch.Tensor, new_ids: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Replace each row's `pad_idx` placeholders, in order, with that row's `new_ids`."""
    if input_ids.ndim != 2 or new_ids.ndim != 2 or input_ids.shape[0] != new_ids.shape[0]:
        raise ValueError(
            f"expected [B, L] input_ids and [B, K] new_ids, got {tuple(input_ids.shape)} and {tuple(new_ids.shape)}"
        )
    mask = input_ids == pad_idx
    counts = mask.sum(dim=1)
    if bool((counts != new_ids.shape[1]).any()):
        raise ValueError(
            f"placeholder counts {counts.tolist()} do not match {new_ids.shape[1]} encoded ids (pad_idx={pad_idx})"
        )
    return input_ids.masked_scatter(mask, new_ids)


def fuse_traj_tokens(
    tokenizer: DeltaTrajectoryTokenizer, input_ids: torch.Tensor, ego_history_xyz: torch.Tensor, tokens
) -> torch.Tensor:
    """Write the encoded ego history ([B, n_traj, T, 3]) into the `<|traj_history|>` placeholders of `input_ids`."""
    if ego_history_xyz.ndim != 4:
        raise ValueError("ego_history_xyz must be [B, n_traj, T, 3]")
    batch = ego_history_xyz.shape[0]
    hist_idx = tokenizer.encode(ego_history_xyz.flatten(0, 1)) + tokens.history_id0
    return replace_pad_token(input_ids, hist_idx.reshape(batch, -1), tokens.history_pad)
