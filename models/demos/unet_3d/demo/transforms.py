# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import numpy as np
import torch
from skimage.segmentation import find_boundaries


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms: list[Callable[[np.ndarray], np.ndarray]]):
        self.transforms = transforms

    def __call__(self, m):
        for t in self.transforms:
            m = t(m)
        return m


class StandardLabelToBoundary:
    """Converts a given volumetric label array to binary mask corresponding to borders between labels.

    Args:
        ignore_index: Label to ignore in the output.
        append_label: If True, stack the borders and original labels across channel dim. Default: False.
        mode: Boundary detection mode. Default: 'thick'.
        foreground: If True, include foreground mask (i.e everything greater than 0) in the first channel of the result.
            Default: False.
    """

    def __init__(
        self,
        ignore_index: int = None,
        append_label: bool = False,
        mode: str = "thick",
        foreground: bool = False,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.append_label = append_label
        self.mode = mode
        self.foreground = foreground

    def __call__(self, m: np.ndarray) -> np.ndarray:
        assert m.ndim == 3

        boundaries = find_boundaries(m, connectivity=2, mode=self.mode)
        boundaries = boundaries.astype("int32")

        results = []
        if self.foreground:
            foreground = (m > 0).astype("uint8")
            results.append(_recover_ignore_index(foreground, m, self.ignore_index))

        results.append(_recover_ignore_index(boundaries, m, self.ignore_index))

        if self.append_label:
            results.append(m)

        return np.stack(results, axis=0)


class BlobsToMask:
    """Returns binary mask from labeled image of blob like objects.

    Every label greater than 0 is treated as foreground.

    Args:
        append_label: If True, append original labels. Default: False.
        boundary: If True, compute boundaries. Default: False.
        cross_entropy: If True, use cross entropy format. Default: False.
    """

    def __init__(
        self,
        append_label: bool = False,
        boundary: bool = False,
        cross_entropy: bool = False,
        **kwargs,
    ):
        self.cross_entropy = cross_entropy
        self.boundary = boundary
        self.append_label = append_label

    def __call__(self, m):
        assert m.ndim == 3

        # get the segmentation mask
        mask = (m > 0).astype("uint8")
        results = [mask]

        if self.boundary:
            outer = find_boundaries(m, connectivity=2, mode="outer")
            if self.cross_entropy:
                # boundary is class 2
                mask[outer > 0] = 2
                results = [mask]
            else:
                results.append(outer)

        if self.append_label:
            results.append(m)

        return np.stack(results, axis=0)


class Standardize:
    """Apply Z-score normalization to a given input tensor.

    Re-scales the values to be 0-mean and 1-std.

    Args:
        eps: Small value to prevent division by zero. Default: 1e-10.
        mean: Pre-computed mean value.
        std: Pre-computed standard deviation value.
        channelwise: If True, normalize per-channel. Default: False.
    """

    def __init__(
        self,
        eps: float = 1e-10,
        mean: float = None,
        std: float = None,
        channelwise: bool = False,
        **kwargs,
    ):
        if mean is not None or std is not None:
            assert mean is not None and std is not None
        self.mean = mean
        self.std = std
        self.eps = eps
        self.channelwise = channelwise

    def __call__(self, m: np.ndarray) -> np.ndarray:
        if self.mean is not None:
            mean, std = self.mean, self.std
        else:
            if self.channelwise:
                # normalize per-channel
                axes = list(range(m.ndim))
                # average across channels
                axes = tuple(axes[1:])
                mean = np.mean(m, axis=axes, keepdims=True)
                std = np.std(m, axis=axes, keepdims=True)
            else:
                mean = np.mean(m)
                std = np.std(m)

        return (m - mean) / np.clip(std, a_min=self.eps, a_max=None)


class Normalize:
    """Apply simple min-max scaling to a given input tensor.

    Shrinks the range of the data to a fixed range of [-1, 1] or in case of norm01==True to [0, 1].

    Args:
        min_value: Minimum value for clipping. Default: None (use min of the input array).
        max_value: Maximum value for clipping. Default: None (use max of the input array).
        norm01: If True, normalize to [0, 1] instead of [-1, 1]. Default: False.
        eps: Small value to prevent division by zero. Default: 1e-10.
    """

    def __init__(
        self,
        min_value: float = None,
        max_value: float = None,
        norm01: bool = False,
        eps: float = 1e-10,
        **kwargs,
    ):
        if min_value is not None and max_value is not None:
            assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value
        self.norm01 = norm01
        self.eps = eps

    def __call__(self, m: np.ndarray) -> np.ndarray:
        if self.min_value is None:
            min_value = np.min(m)
        else:
            min_value = self.min_value

        if self.max_value is None:
            max_value = np.max(m)
        else:
            max_value = self.max_value

        # calculate norm_0_1 with min_value / max_value with the same dimension
        # in case of channelwise application
        norm_0_1 = (m - min_value) / (max_value - min_value + self.eps)

        if self.norm01:
            return np.clip(norm_0_1, 0, 1)
        else:
            return np.clip(2 * norm_0_1 - 1, -1, 1)


class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor.

    Args:
        expand_dims (bool): if True, adds a channel dimension to the input data
        dtype (np.dtype): the desired output data type
        normalize (bool): zero-one normalization of the input data
    """

    def __init__(
        self,
        expand_dims: bool,
        dtype: np.dtype = np.float32,
        normalize: bool = False,
        **kwargs,
    ):
        self.expand_dims = expand_dims
        self.dtype = dtype
        self.normalize = normalize

    def __call__(self, m: np.ndarray) -> torch.Tensor:
        assert m.ndim in [3, 4], "Supports only 3D (DxHxW) or 4D (CxDxHxW) images"
        # add channel dimension
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)

        if self.normalize:
            # avoid division by zero
            m = (m - np.min(m)) / (np.max(m) - np.min(m) + 1e-10)

        return torch.from_numpy(m.astype(dtype=self.dtype))


class Identity:
    def __init__(self, **kwargs):
        pass

    def __call__(self, m: np.ndarray) -> np.ndarray:
        return m


class LabelToTensor:
    """Convert a given input numpy.ndarray label array into torch.Tensor of dtype int64."""

    def __call__(self, m: np.ndarray) -> torch.Tensor:
        m = np.array(m)
        return torch.from_numpy(m.astype(dtype="int64"))


class Transformer:
    """Factory class for creating data augmentation pipelines."""

    def __init__(self, phase_config: dict, base_config: dict):
        self.phase_config = phase_config
        self.config_base = base_config

    def raw_transform(self):
        return self._create_transform("raw")

    def label_transform(self):
        return self._create_transform("label")

    @staticmethod
    def _transformer_class(class_name):
        if class_name == "StandardLabelToBoundary":
            return StandardLabelToBoundary
        if class_name == "BlobsToMask":
            return BlobsToMask
        if class_name == "Standardize":
            return Standardize
        if class_name == "Normalize":
            return Normalize
        if class_name == "ToTensor":
            return ToTensor
        if class_name == "Identity":
            return Identity
        if class_name == "LabelToTensor":
            return LabelToTensor

        raise ValueError(f"Unknown transformer class name: {class_name}")

    def _create_transform(self, name):
        assert name in self.phase_config, f"Could not find {name} transform"
        return Compose([self._create_augmentation(c) for c in self.phase_config[name]])

    def _create_augmentation(self, c):
        config = dict(self.config_base)
        config.update(c)
        aug_class = self._transformer_class(config["name"])
        return aug_class(**config)


def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input
