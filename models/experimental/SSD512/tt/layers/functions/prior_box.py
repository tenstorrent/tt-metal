# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import ttnn
from itertools import product
from math import sqrt
import torch


class TtPriorBox:
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg, device=None):
        self.image_size = cfg["min_dim"]
        self.num_priors = len(cfg["aspect_ratios"])
        self.variance = cfg["variance"] or [0.1]
        self.feature_maps = cfg["feature_maps"]
        self.min_sizes = cfg["min_sizes"]
        self.max_sizes = cfg["max_sizes"]
        self.steps = cfg["steps"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.clip = cfg["clip"]
        self.device = device

        for v in self.variance:
            if v <= 0:
                raise ValueError("Variances must be greater than 0")

    def __call__(self):
        """Generate SSD prior boxes."""
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1, size: min_size
                s_k = self.min_sizes[k] / self.image_size
                mean.extend([cx, cy, s_k, s_k])

                # aspect_ratio: 1, size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean.extend([cx, cy, s_k_prime, s_k_prime])

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    ar_sqrt = sqrt(ar)
                    mean.extend([cx, cy, s_k * ar_sqrt, s_k / ar_sqrt])
                    mean.extend([cx, cy, s_k / ar_sqrt, s_k * ar_sqrt])

        # Convert to tensor and reshape to [N,4] like reference
        output = ttnn.Tensor(torch.tensor(mean), device=self.device)
        output = ttnn.reshape(output, (-1, 4))
        if self.clip:
            output = ttnn.clamp(output, min=0.0, max=1.0)
        return output
