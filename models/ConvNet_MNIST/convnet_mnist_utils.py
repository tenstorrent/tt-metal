# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torchvision
import torchvision.transforms as transforms


def get_test_data(batch_size=64):
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.05,), std=(0.05,)),
        ]
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
    )

    batch = []
    images = []

    for i in range(batch_size):
        img, _ = test_dataset[i]
        tensor = transform(img).unsqueeze(0)
        batch.append(tensor)
        images.append(img)

    batch = torch.cat(batch)
    return batch, images
