# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Resolution helpers for gen-image token layout (aspect-ratio → pixel size).
#
# Mapping to HF upstream (hunyuan_image_3/tokenization_hunyuan_image_3.py)
# -------------------------------------------------------------------------
#  Step | HF                                           | This port (host)
#  -----+----------------------------------------------+----------------------------------
#   1   | Resolution (H×W parsing)                     | Resolution
#   2   | ResolutionGroup preset aspect ratios         | ResolutionGroup
#   3   | get_target_size(w, h)                        | ResolutionGroup.get_target_size
#   4   | get_base_size_and_ratio_index(w, h)          | ResolutionGroup.get_base_size_and_ratio_index
#
# Used by build_gen_image_info to pick <img_ratio_i> and latent token grid size.
#
# References
# ----------
#   ref/tokenizer/image_info.py   — build_gen_image_info consumer
#   ref/tokenizer/assets/config.json — image_base_size, vae_downsample_factor

from __future__ import annotations

import numpy as np


class Resolution:
    def __init__(self, size, *args):
        if isinstance(size, str):
            if "x" in size:
                size = size.split("x")
                size = (int(size[0]), int(size[1]))
            else:
                size = int(size)
        if len(args) > 0:
            size = (size, args[0])
        if isinstance(size, int):
            size = (size, size)

        self.h = self.height = size[0]
        self.w = self.width = size[1]
        self.r = self.ratio = self.height / self.width

    def __getitem__(self, idx):
        if idx == 0:
            return self.h
        if idx == 1:
            return self.w
        raise IndexError(f"Index {idx} out of range")


class ResolutionGroup:
    def __init__(self, base_size=None, step=None, align=1, extra_resolutions=None):
        self.align = align
        self.base_size = base_size
        if base_size is not None and base_size % align != 0:
            raise ValueError(f"base_size {base_size} is not divisible by align {align}")
        if step is None:
            step = base_size // 16
        self.step = step
        self.data = self._calc_by_step()

        if extra_resolutions is not None:
            for extra_resolution in extra_resolutions:
                height, width = extra_resolution.height, extra_resolution.width
                ratio = height / width
                if not any(resolution.ratio == ratio for resolution in self.data):
                    self.data.append(extra_resolution)

        self.ratio = np.array([x.ratio for x in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _calc_by_step(self):
        min_height = self.base_size // 2
        min_width = self.base_size // 2
        max_height = self.base_size * 2
        max_width = self.base_size * 2

        resolutions = [Resolution(self.base_size, self.base_size)]
        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height >= max_height and cur_width <= min_width:
                break
            cur_height = min(cur_height + self.step, max_height)
            cur_width = max(cur_width - self.step, min_width)
            resolutions.append(Resolution(cur_height // self.align * self.align, cur_width // self.align * self.align))

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height <= min_height and cur_width >= max_width:
                break
            cur_height = max(cur_height - self.step, min_height)
            cur_width = min(cur_width + self.step, max_width)
            resolutions.append(Resolution(cur_height // self.align * self.align, cur_width // self.align * self.align))

        return sorted(resolutions, key=lambda x: x.ratio)

    def get_target_size(self, width, height):
        ratio = height / width
        idx = int(np.argmin(np.abs(self.ratio - ratio)))
        reso = self.data[idx]
        return reso.w, reso.h

    def get_base_size_and_ratio_index(self, width, height):
        ratio = height / width
        idx = int(np.argmin(np.abs(self.ratio - ratio)))
        return self.base_size, idx
