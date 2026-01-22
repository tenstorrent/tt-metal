# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Consolidated datasets module combining builder.py and samplers
# Also imports dataset classes to register them

import platform
import random
import math
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import DistributedSampler as _DistributedSampler

from models.experimental.MapTR.reference.dependency import (
    collate,
    get_dist_info,
    Registry,
    build_from_cfg,
    DATASETS,
    GroupSampler,
)

if platform.system() != "Windows":
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry("Object sampler")
SAMPLER = Registry("sampler")


# ========== Samplers ==========
def build_sampler(cfg, default_args):
    return build_from_cfg(cfg, SAMPLER, default_args)


@SAMPLER.register_module()
class DistributedSampler(_DistributedSampler):
    def __init__(self, dataset=None, num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        if self.shuffle:
            assert False
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices = (indices * math.ceil(self.total_size / len(indices)))[: self.total_size]
        assert len(indices) == self.total_size

        per_replicas = self.total_size // self.num_replicas
        indices = indices[self.rank * per_replicas : (self.rank + 1) * per_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


@SAMPLER.register_module()
class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset."""

    def __init__(self, dataset, samples_per_gpu=1, num_replicas=None, rank=None, seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        assert hasattr(self.dataset, "flag")
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += (
                int(math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu / self.num_replicas))
                * self.samples_per_gpu
            )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size), generator=g).numpy())].tolist()
                extra = int(
                    math.ceil(size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[: extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j]
            for i in list(torch.randperm(len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) * self.samples_per_gpu)
        ]

        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# ========== DataLoader Builder ==========
def build_dataloader(
    dataset,
    samples_per_gpu,
    workers_per_gpu,
    num_gpus=1,
    dist=True,
    shuffle=True,
    seed=None,
    shuffler_sampler=None,
    nonshuffler_sampler=None,
    **kwargs,
):
    """Build PyTorch DataLoader."""
    rank, world_size = get_dist_info()
    if dist:
        if shuffle:
            sampler = build_sampler(
                shuffler_sampler if shuffler_sampler is not None else dict(type="DistributedGroupSampler"),
                dict(dataset=dataset, samples_per_gpu=samples_per_gpu, num_replicas=world_size, rank=rank, seed=seed),
            )
        else:
            sampler = build_sampler(
                nonshuffler_sampler if nonshuffler_sampler is not None else dict(type="DistributedSampler"),
                dict(dataset=dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed),
            )

        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        print("WARNING!!!!, Only can be used for obtain inference speed!!!!")
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs,
    )

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ========== Dataset Builder ==========
def custom_build_dataset(cfg, default_args=None):
    # Import dataset wrappers from dependency
    from models.experimental.MapTR.reference.dependency import ConcatDataset

    # Stub classes for dataset wrappers not in dependency (not needed for demo/test)
    class CBGSDataset:
        def __init__(self, dataset):
            self.dataset = dataset

    class ClassBalancedDataset:
        def __init__(self, dataset, oversample_thr):
            self.dataset = dataset
            self.oversample_thr = oversample_thr

    class RepeatDataset:
        def __init__(self, dataset, times):
            self.dataset = dataset
            self.times = times

    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([custom_build_dataset(c, default_args) for c in cfg])
    elif cfg["type"] == "ConcatDataset":
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg["datasets"]], cfg.get("separate_eval", True)
        )
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(custom_build_dataset(cfg["dataset"], default_args), cfg["times"])
    elif cfg["type"] == "ClassBalancedDataset":
        dataset = ClassBalancedDataset(custom_build_dataset(cfg["dataset"], default_args), cfg["oversample_thr"])
    elif cfg["type"] == "CBGSDataset":
        dataset = CBGSDataset(custom_build_dataset(cfg["dataset"], default_args))
    elif isinstance(cfg.get("ann_file"), (list, tuple)):
        from models.experimental.MapTR.reference.dependency import _concat_dataset

        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


# Import dataset classes to register them
