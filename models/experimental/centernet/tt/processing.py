from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import copy
from pathlib import Path
from mmengine.fileio.io import list_dir_or_file
from mmengine.fileio import get_file_backend, isdir, join_path, list_dir_or_file
from mmengine.dataset import Compose
from mmcv.transforms import LoadImageFromFile
from mmengine.dataset.utils import pseudo_collate

InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]

preprocess_kwargs: set = set()
forward_kwargs: set = set()

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def _dispatch_kwargs(kwargs) -> Tuple[Dict, Dict, Dict, Dict]:
    visualize_kwargs = {"draw_pred", "no_save_vis", "show", "img_out_dir", "wait_time", "pred_score_thr", "return_vis"}
    postprocess_kwargs = {"print_result", "return_datasamples", "no_save_pred", "pred_out_dir"}
    method_kwargs = visualize_kwargs | postprocess_kwargs

    union_kwargs = method_kwargs | set(kwargs.keys())
    if union_kwargs != method_kwargs:
        unknown_kwargs = union_kwargs - method_kwargs
        raise ValueError(
            f"unknown argument {unknown_kwargs} for `preprocess`, " "`forward`, `visualize` and `postprocess`"
        )

    preprocess_kwargs = {}
    forward_kwargs = {}
    visualize_kwargs = {}
    postprocess_kwargs = {}

    for key, value in kwargs.items():
        if key in preprocess_kwargs:
            preprocess_kwargs[key] = value
        elif key in forward_kwargs:
            forward_kwargs[key] = value
        elif key in visualize_kwargs:
            visualize_kwargs[key] = value
        else:
            postprocess_kwargs[key] = value

    return (
        preprocess_kwargs,
        forward_kwargs,
        visualize_kwargs,
        postprocess_kwargs,
    )


def isdir(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bool:
    backend = get_file_backend(filepath, backend_args=backend_args, enable_singleton=True)
    return backend.isdir(filepath)


def _inputs_to_list(inputs: InputsType) -> list:
    if isinstance(inputs, str):
        backend = get_file_backend(inputs)
        if hasattr(backend, "isdir") and isdir(inputs):
            filename_list = list_dir_or_file(inputs, list_dir=False, suffix=IMG_EXTENSIONS)
            inputs = [join_path(inputs, filename) for filename in filename_list]

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    return list(inputs)


def _get_transform_idx(pipeline_cfg, name: Union[str, Tuple[str, type]]) -> int:
    """Returns the index of the transform in a pipeline.

    If the transform is not found, returns -1.
    """
    for i, transform in enumerate(pipeline_cfg):
        if transform["type"] in name:
            return i
    return -1


def _init_pipeline(cfg) -> Compose:
    """Initialize the test pipeline."""
    pipeline_cfg = cfg.test_dataloader.dataset.pipeline

    # For inference, the key of ``img_id`` is not used.
    if "meta_keys" in pipeline_cfg[-1]:
        pipeline_cfg[-1]["meta_keys"] = tuple(
            meta_key for meta_key in pipeline_cfg[-1]["meta_keys"] if meta_key != "img_id"
        )

    load_img_idx = _get_transform_idx(pipeline_cfg, ("LoadImageFromFile", LoadImageFromFile))
    if load_img_idx == -1:
        raise ValueError("LoadImageFromFile is not found in the test pipeline")
    pipeline_cfg[load_img_idx]["type"] = "mmdet.InferencerLoader"
    return Compose(pipeline_cfg)


def _init_collate(cfg) -> Callable:
    collate_fn = pseudo_collate
    return collate_fn  # type: ignore


def _get_chunk_data(inputs: Iterable, chunk_size: int, cfg):
    inputs_iter = iter(inputs)
    pipeline = _init_pipeline(cfg)
    while True:
        try:
            chunk_data = []
            for _ in range(chunk_size):
                inputs_ = next(inputs_iter)
                if isinstance(inputs_, dict):
                    if "img" in inputs_:
                        ori_inputs_ = inputs_["img"]
                    else:
                        ori_inputs_ = inputs_["img_path"]
                    chunk_data.append((ori_inputs_, pipeline(copy.deepcopy(inputs_))))
                else:
                    chunk_data.append((inputs_, pipeline(inputs_)))
            yield chunk_data
        except StopIteration:
            if chunk_data:
                yield chunk_data
            break


def preprocess(inputs: InputsType, batch_size, cfg, **kwargs):
    chunked_data = _get_chunk_data(inputs, batch_size, cfg)
    collate_fn = _init_collate(cfg)
    yield from map(collate_fn, chunked_data)
