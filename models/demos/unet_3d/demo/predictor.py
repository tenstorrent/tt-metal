# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import ttnn
from models.demos.unet_3d.demo.utils import configure_logging
from models.demos.unet_3d.runner.performant_runner import UNet3DRunner


def remove_padding(tensor: torch.Tensor, halo_shape: tuple[int, int, int]) -> torch.Tensor:
    if sum(halo_shape) == 0:
        return tensor
    slices = [slice(None)] * tensor.ndim
    for axis, halo in enumerate(halo_shape):
        if halo:
            slices[-3 + axis] = slice(halo, -halo)
    return tensor[tuple(slices)]


logger = configure_logging()


def prepare_input_torch(input_tensor, device):
    return input_tensor.to(device)


def prepare_input_ttnn(input_tensor, device):
    input_tensor = input_tensor.permute(0, 2, 3, 4, 1)
    return ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, device=device)


class Predictor:
    """Predictor that applies the model on the given dataset and saves the result as H5 file."""

    def __init__(
        self,
        runner: UNet3DRunner,
        output_dir: str,
        output_dataset: str,
        out_channels: int,
        performance_metric: list[str],
        label_internal_path: str | None = None,
    ):
        self.runner = runner
        assert out_channels > 0, f"Invalid number of output channels: {out_channels}"
        self.out_channels = out_channels
        self.output_dir = output_dir
        self.output_dataset = output_dataset
        self.performance_metric = performance_metric
        self.label_internal_path = label_internal_path
        logger.info(f"Predictor initialized (out_channels={out_channels}, ")

    def __call__(self, test_loader: DataLoader) -> Any:
        """Run the model prediction on the test_loader and save the results in the output_dir.

        If the performance_metric is provided, the predictions will be evaluated against the ground truth
        and returned as a tensor.
        """
        logger.info(f"Processing '{test_loader.dataset.file_path}'...")
        start = time.perf_counter()

        volume_shape = test_loader.dataset.volume_shape

        prediction_shape = (self.out_channels,) + volume_shape
        output_file = _get_output_file(dataset=test_loader.dataset, output_dir=self.output_dir)
        patch_halo = test_loader.dataset.halo_shape
        channel_slice = slice(0, self.out_channels)
        with h5py.File(output_file, "w") as h5_output_file:
            prediction_array = self._allocate_prediction_array(prediction_shape, h5_output_file)
            with torch.no_grad():
                for input, indices in tqdm(test_loader):
                    prediction = self.runner.run(input)

                    if sum(patch_halo) > 0:
                        prediction = remove_padding(prediction, patch_halo)
                    for pred, index in zip(prediction, indices, strict=True):
                        index = (channel_slice,) + tuple(index)
                        prediction_array[index] = (pred > 0.5).astype("uint16")

            self._create_prediction_dataset(h5_output_file, prediction_array)
            logger.info(f"Finished inference in {time.perf_counter() - start:.2f} seconds")

            result = {}

            for metric_name in self.performance_metric:
                assert (
                    self.label_internal_path is not None
                ), "Ground truth internal path must be provided to compute performance metric"
                gt = test_loader.dataset.get_label_array()
                if metric_name == "dice":
                    metric_value = dice_score(prediction_array, gt, n_classes=self.out_channels)
                elif metric_name == "mean_iou":
                    metric_value = mean_iou(prediction_array, gt, n_classes=self.out_channels)
                else:
                    raise ValueError(
                        f"Unsupported performance metric: {metric_name}, only dice and mean_iou are supported"
                    )
                result[metric_name] = metric_value

            return result

    def _create_prediction_dataset(self, h5_output_file, prediction_array):
        h5_output_file.create_dataset(self.output_dataset, data=prediction_array, compression="gzip")

    def _allocate_prediction_array(self, output_shape, output_file):
        return np.zeros(output_shape, dtype="float32")


def _get_output_file(dataset: object, output_dir: str | Path | None = None) -> Path:
    """
    Get the output file path for the predictions. If `output_dir` is not None the output file will be saved in
    the original dataset directory.
    """
    file_path = Path(dataset.file_path)

    if output_dir is None:
        output_dir = file_path.parent
    else:
        output_dir = Path(output_dir)

    output_filename = file_path.stem + "_predictions" + ".h5"
    return output_dir / output_filename


def mean_iou(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> list[float] | float:
    """
    Compute the mean Intersection over Union (IoU) for the given predictions and ground truth.
    """
    if n_classes == 1:
        return mean_iou_binary(pred, gt)
    pred = pred.astype("uint16")
    gt = gt.astype("uint16")
    assert pred.shape == gt.shape, f"Predictions and ground truth have different shapes: {pred.shape} != {gt.shape}"
    per_class_iou = []
    for c in range(0, n_classes):
        intersection = np.logical_and(gt == c, pred == c).sum()
        union = np.logical_or(gt == c, pred == c).sum()
        iou = intersection / union
        per_class_iou.append(iou)

    return np.array(per_class_iou)


def mean_iou_binary(pred: np.ndarray, gt: np.ndarray) -> list[float] | float:
    """
    Compute the mean Intersection over Union (IoU) for binary predictions and ground truth.
    """
    pred = pred.astype("uint16")
    gt = gt.astype("uint16")
    assert pred.shape == gt.shape, f"Predictions and ground truth have different shapes: {pred.shape} != {gt.shape}"
    per_class_iou = []
    for c in [0, 1]:
        intersection = np.logical_and(gt == c, pred == c).sum()
        union = np.logical_or(gt == c, pred == c).sum()
        iou = intersection / union
        per_class_iou.append(iou)

    return np.array(per_class_iou)


def dice_score(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> list[float] | float:
    """
    Compute the Dice score for the given predictions and ground truth.
    """
    if n_classes == 1:
        return dice_score_binary(pred, gt)

    pred = pred.astype("uint16")
    gt = gt.astype("uint16")
    assert pred.shape == gt.shape, f"Predictions and ground truth have different shapes: {pred.shape} != {gt.shape}"
    per_class_dice = []
    for c in range(n_classes):
        intersection = np.logical_and(gt == c, pred == c).sum()
        sums = (gt == c).sum() + (pred == c).sum()
        dice = 2 * intersection / sums
        per_class_dice.append(dice)

    return np.array(per_class_dice)


def dice_score_binary(pred: np.ndarray, gt: np.ndarray) -> list[float] | float:
    """
    Compute the Dice score for binary predictions and ground truth.
    """
    pred = pred.astype("uint16")
    gt = gt.astype("uint16")
    assert pred.shape == gt.shape, f"Predictions and ground truth have different shapes: {pred.shape} != {gt.shape}"
    per_class_dice = []
    for c in [0, 1]:
        intersection = np.logical_and(gt == c, pred == c).sum()
        sums = (gt == c).sum() + (pred == c).sum()
        dice = 2 * intersection / sums
        per_class_dice.append(dice)

    return np.array(per_class_dice)
