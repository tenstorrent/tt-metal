# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import pkgutil
import warnings
from collections import namedtuple

import torch

if torch.__version__ != "parrots":
    # Name mapping from old mmcv._ext names to new mmcv.ops names
    _NAME_MAP = {
        "softnms": "soft_nms",
    }

    class ExtWrapper:
        """Wrapper to map old function names to new mmcv.ops names."""
        def __init__(self, ext):
            self._ext = ext

        def __getattr__(self, name):
            mapped_name = _NAME_MAP.get(name, name)
            return getattr(self._ext, mapped_name)

    def load_ext(name, funcs):
        # For mmcv 2.x, use mmcv.ops instead of mmcv._ext
        if name == "_ext":
            ext = importlib.import_module("mmcv.ops")
        else:
        ext = importlib.import_module("mmcv." + name)
        for fun in funcs:
            mapped_name = _NAME_MAP.get(fun, fun)
            assert hasattr(ext, mapped_name), f"{fun} (mapped to {mapped_name}) miss in module {name}"
        return ExtWrapper(ext)

else:
    from parrots import extension
    from parrots.base import ParrotsException

    has_return_value_ops = [
        "nms",
        "softnms",
        "nms_match",
        "nms_rotated",
        "top_pool_forward",
        "top_pool_backward",
        "bottom_pool_forward",
        "bottom_pool_backward",
        "left_pool_forward",
        "left_pool_backward",
        "right_pool_forward",
        "right_pool_backward",
        "fused_bias_leakyrelu",
        "upfirdn2d",
        "ms_deform_attn_forward",
        "pixel_group",
        "contour_expand",
        "diff_iou_rotated_sort_vertices_forward",
    ]

    def get_fake_func(name, e):
        def fake_func(*args, **kwargs):
            warnings.warn(f"{name} is not supported in parrots now")
            raise e

        return fake_func

    def load_ext(name, funcs):
        ExtModule = namedtuple("ExtModule", funcs)
        ext_list = []
        lib_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        for fun in funcs:
            try:
                ext_fun = extension.load(fun, name, lib_dir=lib_root)
            except ParrotsException as e:
                if "No element registered" not in e.message:
                    warnings.warn(e.message)
                ext_fun = get_fake_func(fun, e)
                ext_list.append(ext_fun)
            else:
                if fun in has_return_value_ops:
                    ext_list.append(ext_fun.op)
                else:
                    ext_list.append(ext_fun.op_)
        return ExtModule(*ext_list)


def check_ops_exist() -> bool:
    # For mmcv 2.x, check for mmcv.ops instead of mmcv._ext
    ext_loader = pkgutil.find_loader("mmcv.ops")
    return ext_loader is not None
