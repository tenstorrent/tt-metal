# from .nuscenes_dataset import CustomNuScenesDataset
from .nuscenes_dataset_v2 import CustomNuScenesDatasetV2

# from .builder import custom_build_dataset

# Import pipelines to ensure they are registered with the PIPELINES registry
from . import pipelines  # noqa: F401

__all__ = [
    # "CustomNuScenesDataset",
    "CustomNuScenesDatasetV2",
]
