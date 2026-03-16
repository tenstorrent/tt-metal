# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from itertools import chain
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import DataLoader

import models.demos.unet_3d.demo.transforms as transforms
from models.demos.unet_3d.demo.utils import (
    calculate_stats,
    configure_logging,
    default_prediction_collate,
    get_slice_builder,
    mirror_pad,
)

logger = configure_logging()


def get_test_loaders(loaders_config: dict, num_devices):
    """Returns test DataLoader.

    Args:
        config: A top level configuration object containing the 'loaders' key.

    Returns:
        Generator of DataLoader objects.
    """

    # get dataset class
    test_datasets = create_datasets(
        dataset_config=loaders_config,
    )

    num_workers = loaders_config.get("num_workers", 1)
    logger.info(f"Number of workers for the dataloader: {num_workers}")

    batch_size = loaders_config.get("batch_size_per_device", 1) * num_devices
    # use generator in order to create data loaders lazily one by one
    for test_dataset in test_datasets:
        collate_fn = default_prediction_collate
        yield DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


def _create_padded_indexes(indexes: tuple, halo_shape: tuple):
    """Create padded indexes by extending each slice in `indexes` by the corresponding `halo_shape`."""
    if sum(halo_shape) == 0:
        return indexes
    return tuple(slice(index.start, index.stop + 2 * halo) for index, halo in zip(indexes, halo_shape, strict=True))


def traverse_h5_paths(file_paths: list[str]) -> list[str]:
    """Traverse the given list of file paths and directories to find all H5 files."""
    assert isinstance(file_paths, list)
    results = []
    for file_path in file_paths:
        file_path = Path(file_path)
        if file_path.is_dir():
            iters = [file_path.glob(ext) for ext in ["*.h5", "*.hdf", "*.hdf5", "*.hd5"]]
            for fp in chain(*iters):
                results.append(str(fp))
        else:
            results.append(str(file_path))
    return results


class HDF5Dataset:
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files.
    """

    def __init__(
        self,
        file_path: str,
        slice_builder_config: dict,
        transformer_config: dict,
        raw_internal_path: str,
        label_internal_path: str,
        global_normalization: bool = False,
    ):
        logger.info(f"Creating {self.__class__.__name__} from {file_path} ( global_norm={global_normalization})")
        self.file_path = file_path
        self.raw_internal_path = raw_internal_path
        self.label_internal_path = label_internal_path

        self.halo_shape = tuple(slice_builder_config.get("halo_shape", [0, 0, 0]))

        if global_normalization:
            with h5py.File(file_path, "r") as f:
                raw = f[raw_internal_path][:]
                stats = calculate_stats(raw)
        else:
            stats = calculate_stats(None, True)

        transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = transformer.raw_transform()
        if self.halo_shape == (0, 0, 0):
            logger.warning(
                "Found halo shape to be (0, 0, 0). This might lead to checkerboard artifacts in the "
                "prediction. Consider using a non-zero halo shape, e.g. 'halo_shape: [8, 8, 8]' in "
                "the slice_builder configuration."
            )
        self.label_transform = transformer.label_transform()

        with h5py.File(file_path, "r") as f:
            raw = f[raw_internal_path]
            self.volume_shape = raw.shape[-3:]
            label = f.get(label_internal_path, None)
            if label is not None:
                assert label.shape[-3:] == self.volume_shape, "Raw and label shapes do not match"

            slice_builder = get_slice_builder(raw, label, slice_builder_config)
            self.raw_slices = slice_builder.raw_slices
            self.label_slices = slice_builder.label_slices
        logger.info(
            f"Built {len(self.raw_slices)} patches for {self.file_path} with patch_shape="
            f"{slice_builder_config.get('patch_shape')} stride_shape={slice_builder_config.get('stride_shape')} "
            f"halo_shape={self.halo_shape}"
        )
        self._raw = None
        self._raw_padded = None
        self._label = None

    def get_label_array(self) -> np.ndarray:
        assert self.label_internal_path is not None

        with h5py.File(self.file_path, "r") as f:
            array = f[self.label_internal_path][:]
            array = self.label_transform(array)
            if len(array.shape) == 3:
                array = np.expand_dims(array, axis=0)
            return array

    def get_raw_patch(self, idx: int) -> np.ndarray:
        if self._raw is None:
            with h5py.File(self.file_path, "r") as f:
                assert self.raw_internal_path in f, f"Dataset {self.raw_internal_path} not found in {self.file_path}"
                self._raw = f[self.raw_internal_path][:]
        return self._raw[idx]

    def get_label_patch(self, idx: int) -> np.ndarray:
        if self._label is None:
            with h5py.File(self.file_path, "r") as f:
                assert (
                    self.label_internal_path in f
                ), f"Dataset {self.label_internal_path} not found in {self.file_path}"
                self._label = f[self.label_internal_path][:]
        return self._label[idx]

    def get_raw_padded_patch(self, idx: int) -> np.ndarray:
        if self._raw_padded is None:
            with h5py.File(self.file_path, "r") as f:
                assert self.raw_internal_path in f, f"Dataset {self.raw_internal_path} not found in {self.file_path}"
                self._raw_padded = mirror_pad(f[self.raw_internal_path][:], self.halo_shape)
        return self._raw_padded[idx]

    def __getitem__(self, idx: int) -> tuple:
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slices[idx]

        if len(raw_idx) == 4:
            raw_idx = raw_idx[1:]
            raw_idx_padded = (slice(None),) + _create_padded_indexes(raw_idx, self.halo_shape)
        else:
            raw_idx_padded = _create_padded_indexes(raw_idx, self.halo_shape)

        padded_patch = self.get_raw_padded_patch(raw_idx_padded)
        raw_patch_transformed = self.raw_transform(padded_patch)
        return raw_patch_transformed, raw_idx

    def __len__(self) -> int:
        return len(self.raw_slices)


def create_datasets(dataset_config: dict) -> Iterable["HDF5Dataset"]:
    dataset_type = dataset_config["type"]
    if dataset_type == "StandardHDF5Dataset":
        return create_datasets_hdf5(dataset_config)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def create_datasets_hdf5(dataset_config: dict) -> Iterable["HDF5Dataset"]:
    transformer_config = dataset_config["transformer"]
    slice_builder_config = dataset_config["slice_builder"]
    file_paths = traverse_h5_paths(dataset_config["file_paths"])

    for file_path in file_paths:
        yield HDF5Dataset(
            file_path=file_path,
            slice_builder_config=slice_builder_config,
            transformer_config=transformer_config,
            raw_internal_path=dataset_config.get("raw_internal_path"),
            label_internal_path=dataset_config.get("label_internal_path"),
            global_normalization=dataset_config.get("global_normalization", False),
        )
