# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from torch.optim import SGD
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import transforms


def create_mnist_dataset(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    train_dataset = datasets.MNIST(root="/tmp/data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="/tmp/data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    return train_loader, test_loader


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


def evaluate(test_loader, model):
    # Evaluate the model
    model.eval()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.bfloat16()
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = correct / total
    model.train()
    return acc


if __name__ == "__main__":
    model = MLP().bfloat16()
    criterion = nn.CrossEntropyLoss().bfloat16()
    optimizer = SGD(model.parameters(), lr=0.1)

    num_epochs = 10
    batch_size = 128
    train_loader, test_loader = create_mnist_dataset(batch_size)
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.bfloat16()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        acc = evaluate(test_loader, model)
        print(f"Epoch {epoch + 1} Accuracy: {acc}")
