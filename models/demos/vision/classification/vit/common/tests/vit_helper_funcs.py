# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# A labelled image folder: one subfolder per class, named by the classifier's class index, holding that
# class's images. Without it, inputs are synthetic and unlabelled.
IMAGE_DIR_ENV = "VIT_IMAGE_DIR"
SYNTHETIC_IMAGE_SIZE = 224


class InputExample(object):
    def __init__(self, image, label=None):
        self.image = image
        self.label = label


def image_dir():
    """The labelled image folder from VIT_IMAGE_DIR, or None."""
    value = os.environ.get(IMAGE_DIR_ENV)
    return Path(value) if value else None


def labelled_images(root):
    """(path, class index) pairs from a labelled image folder. The order takes one image of every class
    in turn, so any prefix spreads over the classes instead of covering the first few."""
    per_class = {
        int(folder.name): sorted(path for path in folder.iterdir() if path.is_file())
        for folder in root.iterdir()
        if folder.is_dir() and folder.name.isdigit()
    }
    if not per_class:
        raise ValueError(f"{root}: expected one subfolder per class index")
    depth = max(len(paths) for paths in per_class.values())
    return [
        (paths[index], label)
        for index in range(depth)
        for label, paths in sorted(per_class.items())
        if index < len(paths)
    ]


def num_images():
    """The number of labelled images available, or None for synthetic inputs."""
    root = image_dir()
    return None if root is None else len(labelled_images(root))


def card_type():
    """The card type the kernel driver reports, e.g. "p100a", or None without one.

    Read from sysfs because asking ttnn opens the cluster in this process and keeps it, and a profiled run
    started from the same process then waits on the device forever.
    """
    paths = sorted(Path("/sys/class/tenstorrent").glob("*/tt_card_type"))
    return paths[0].read_text().strip() if paths else None


def card_target(targets, grid_x):
    """The value of targets, keyed by (card type prefix, device grid width), for this machine, or None."""
    card = card_type()
    return next(
        (value for (prefix, width), value in targets.items() if card and card.startswith(prefix) and width == grid_x),
        None,
    )


def device_grid_width():
    """The worker grid width of device 0.

    Read in a child process because opening the device here keeps the cluster open in this process, and a
    profiled run started from the same process then waits on the device forever.
    """
    script = (
        "import ttnn\n"
        "device = ttnn.open_device(device_id=0)\n"
        "print(f'grid_x={device.compute_with_storage_grid_size().x}')\n"
        "ttnn.close_device(device)\n"
    )
    output = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True).stdout
    return int(re.search(r"^grid_x=(\d+)$", output, re.MULTILINE).group(1))


def available_iterations(batch_size, iterations=None):
    """The batches a run can take: iterations, capped by the full batches of the labelled folder, or all of
    them when iterations is None. Synthetic inputs are unlimited."""
    count = num_images()
    if count is None:
        assert iterations is not None, f"set {IMAGE_DIR_ENV} to run over a whole labelled folder"
        return iterations
    full_batches = count // batch_size
    if full_batches == 0:
        raise ValueError(f"{IMAGE_DIR_ENV} holds {count} images, fewer than one batch of {batch_size}")
    return full_batches if iterations is None else min(iterations, full_batches)


def get_data_loader(batch_size, iterations, seed=0):
    """iterations batches of InputExamples: the labelled images from VIT_IMAGE_DIR when it is set, else
    distinct random images without labels."""
    root = image_dir()

    def labelled():
        images = labelled_images(root)[: batch_size * iterations]
        for start in range(0, len(images) - batch_size + 1, batch_size):
            yield [
                InputExample(image=Image.open(path), label=label) for path, label in images[start : start + batch_size]
            ]

    def synthetic():
        generator = np.random.default_rng(seed)
        for _ in range(iterations):
            yield [
                InputExample(
                    image=Image.fromarray(
                        generator.integers(0, 256, (SYNTHETIC_IMAGE_SIZE, SYNTHETIC_IMAGE_SIZE, 3), dtype=np.uint8)
                    )
                )
                for _ in range(batch_size)
            ]

    return labelled() if root is not None else synthetic()


def get_batch(data_loader, image_processor):
    loaded_images = next(data_loader)
    images = None
    labels = []
    for image in loaded_images:
        img = image.image
        labels.append(image.label)
        if img.mode != "RGB":
            img = img.convert(mode="RGB")
        img = image_processor(img, return_tensors="pt")
        img = img["pixel_values"]

        if images is None:
            images = img
        else:
            images = torch.cat((images, img), dim=0)
    return images, labels
