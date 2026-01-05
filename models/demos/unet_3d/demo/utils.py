# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import collections
import logging
from typing import Any

import h5py
import numpy as np
import torch


def configure_logging() -> logging.Logger:
    """Configure root logging once and return the module logger.

    Uses INFO level and a concise formatter; avoids duplicate handlers when imported repeatedly.
    """
    logger = logging.getLogger("unet3d_demo")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = configure_logging()


class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label ndarray based on the patch and stride shape.

    Args:
        raw_dataset: raw data
        label_dataset: ground truth labels
        patch_shape: the shape of the patch DxHxW
        stride_shape: the shape of the stride DxHxW
        kwargs: additional metadata
    """

    def __init__(
        self,
        raw_dataset: h5py.Dataset,
        label_dataset: h5py.Dataset,
        patch_shape: tuple[int, int, int],
        stride_shape: tuple[int, int, int],
        **kwargs,
    ):
        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)
        skip_shape_check = kwargs.get("skip_shape_check", False)
        if not skip_shape_check:
            self._check_patch_shape(patch_shape)

        self._raw_slices = self._build_slices(raw_dataset, patch_shape, stride_shape)
        if label_dataset is None:
            self._label_slices = None
        else:
            if raw_dataset.ndim != label_dataset.ndim:
                self._label_slices = self._build_slices(label_dataset, patch_shape, stride_shape)
                assert len(self._raw_slices) == len(self._label_slices)
            else:
                # if raw and label have the same dim, they have the same shape and thus the same slices
                self._label_slices = self._raw_slices

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @staticmethod
    def _build_slices(
        dataset: h5py.Dataset, patch_shape: tuple[int, int, int], stride_shape: tuple[int, int, int]
    ) -> list[tuple[slice, ...]]:
        """Iterates over a given n-dim dataset patch-by-patch with a given stride and builds an array of slice positions.

        Args:
            dataset: The dataset to build slices for.
            patch_shape: Shape of the patch.
            stride_shape: Shape of the stride.

        Returns:
            List of slices, i.e. [(slice, slice, slice, slice), ...] if len(shape) == 4
            or [(slice, slice, slice), ...] if len(shape) == 3.
        """
        slices = []
        logger.info(f"Building slices for dataset with shape: {dataset.shape}")
        logger.info(f"Stride shape: {stride_shape}, patch shape: {patch_shape}")
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x),
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, "Sample size has to be bigger than the patch size"
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, "patch_shape must be a 3D tuple"


def get_slice_builder(raw: h5py.Dataset, label: h5py.Dataset, config: dict) -> SliceBuilder:
    assert "name" in config
    logger.info(f"Slice builder config name: {config['name']}")
    if config["name"] == "SliceBuilder":
        return SliceBuilder(raw, label, **config)
    raise ValueError(f"Unsupported slice builder: {config['name']}")


def default_prediction_collate(batch: list) -> Any:
    """Default collate_fn to form a mini-batch of Tensor(s) for HDF5 based datasets.

    Args:
        batch: List of samples from the dataset.

    Returns:
        Collated batch.
    """
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch, strict=True)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError(error_msg.format(type(batch[0])))


def calculate_stats(img: np.array, skip: bool = False) -> dict[str, Any]:
    """
    Calculates the minimum percentile, maximum percentile, mean, and standard deviation of the image.

    Args:
        img: The input image array.
        skip: if True, skip the calculation and return None for all values.

    Returns:
        tuple[float, float, float, float]: The minimum percentile, maximum percentile, mean, and std dev
    """
    if not skip:
        pmin, pmax, mean, std = np.percentile(img, 1), np.percentile(img, 99.6), np.mean(img), np.std(img)
    else:
        pmin, pmax, mean, std = None, None, None, None

    return {"pmin": pmin, "pmax": pmax, "mean": mean, "std": std}


def mirror_pad(image: np.ndarray, padding_shape: tuple[int, int, int]) -> np.ndarray:
    """
    Pad the image with a mirror reflection of itself.

    This function is used on data in its original shape before it is split into patches.

    Args:
        image (np.ndarray): The input image array to be padded.
        padding_shape (tuple of int): Specifies the amount of padding for each dimension, should be YX or ZYX.

    Returns:
        np.ndarray: The mirror-padded image.

    Raises:
        ValueError: If any element of padding_shape is negative.
    """
    assert len(padding_shape) == 3, "Padding shape must be specified for each dimension: ZYX"

    if any(p < 0 for p in padding_shape):
        raise ValueError("padding_shape must be non-negative")

    if all(p == 0 for p in padding_shape):
        return image

    pad_width = [(p, p) for p in padding_shape]

    if image.ndim == 4:
        pad_width = [(0, 0)] + pad_width
    return np.pad(image, pad_width, mode="reflect")
