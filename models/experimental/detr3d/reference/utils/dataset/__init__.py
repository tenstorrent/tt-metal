# Copyright (c) Facebook, Inc. and its affiliates.
from .sunrgbd import SunrgbdDetectionDataset, SunrgbdDatasetConfig


DATASET_FUNCTIONS = {
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
}


def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()

    dataset_dict = {
        "test": dataset_builder(
            dataset_config, split_set="val", root_dir=args.dataset_root_dir, use_color=args.use_color, augment=False
        ),
    }
    return dataset_dict, dataset_config
