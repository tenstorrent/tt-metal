from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
# from common import ImageNet
import time
from datetime import datetime, timezone
from pathlib import Path
from operator import mul
from functools import reduce

import numpy as np
import torch
from torchvision.datasets.folder import *
import torchvision.transforms as transforms


class CustomDatasetFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(
            root,
            loader,
            extensions=extensions,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()
        self.dummy_sample = torch.zeros(3, 224, 224)
        self.dummy_target = 0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        start = time.time()
        sample = self.loader(path)
        load_image_time = time.time() - start
        if self.transform is not None:
            start = time.time()
            sample = self.resize(sample)
            resize_time = time.time() - start
            start = time.time()
            sample = self.to_tensor(sample)
            to_tensor_time = time.time() - start
            # sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, load_image_time, resize_time, to_tensor_time


class CustomImageFolder(CustomDatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

        self.samples = np.asarray(self.samples, dtype=object)
        self.imgs = self.samples


class ImageNet:
    def __init__(self, imagenet_dataset_path):
        self.imagenet_dataset_path = Path(imagenet_dataset_path)

        self.base_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.tensor_shape = (3, 224, 224)
        self.element_size = 4  # float32
        self.tensor_size = reduce(mul, self.tensor_shape) * self.element_size

    def get_dataset_loader(
        self, batch_size=10, num_workers=1, split="train", drop_last=False
    ):
        dataset_path = self.imagenet_dataset_path / split
        assert Path.is_dir(dataset_path)

        dataset = CustomImageFolder(dataset_path, transform=self.base_transform)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last
        )

        return dataloader


def prep_ImageNet(batch_size=1):
    root = "/mnt/MLPerf/pytorch_weka_data/imagenet/dataset/ILSVRC/Data/CLS-LOC"
    imagenet = ImageNet(root)

    dataloader = imagenet.get_dataset_loader(
        batch_size=batch_size, drop_last=True)

    return dataloader
