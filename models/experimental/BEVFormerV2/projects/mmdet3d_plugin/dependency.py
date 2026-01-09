from __future__ import division

# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
# This file contains all CPU-only dependencies from mmcv/mmdet/mmdet3d/mmseg
# Extracted from the following versions:
# - https://github.com/open-mmlab/mmcv/tree/v1.4.0/mmcv
# - https://github.com/open-mmlab/mmdetection/tree/v2.14.0/mmdet
# - https://github.com/open-mmlab/mmdetection3d/tree/v0.17.1/mmdet3d
# - https://github.com/open-mmlab/mmsegmentation/tree/v0.14.1/mmseg
# ---------------------------------------------
# All code is brought directly from the GitHub URLs without modifications
# to avoid licensing issues and ensure CPU-only compatibility.
# ---------------------------------------------

import copy
import functools
import logging
import math
import warnings
from abc import ABCMeta
from collections import abc as collections_abc
from collections.abc import Mapping, Sequence
from functools import partial
from logging import FileHandler

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from torch.utils.data.dataloader import default_collate
import mmcv
import tempfile
from os import path as osp

# ============================================================================
# MMCV FILEIO - https://github.com/open-mmlab/mmcv/tree/v1.4.0/mmcv/fileio
# ============================================================================

# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/fileio/io.py
import pickle
import json
import yaml


def load(file_path, file_format=None):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    Args:
        file_path (str): Path to the file to be loaded.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and "pickle/pkl".

    Returns:
        The content from the file.
    """
    if file_format is None:
        file_format = file_path.split(".")[-1]
    if file_format not in ["json", "yaml", "yml", "pickle", "pkl"]:
        raise TypeError("Unsupported format: {}".format(file_format))

    if file_format in ["yaml", "yml"]:
        with open(file_path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    elif file_format in ["pickle", "pkl"]:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        with open(file_path, "r") as f:
            return json.load(f)


mmcv.load = load

# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/datasets/samplers/group_sampler.py
# ============================================================================
# MMCV UTILITIES - https://github.com/open-mmlab/mmcv/tree/v1.4.0/mmcv/utils
# ============================================================================

# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/utils/registry.py
import inspect
from collections import abc as abc_collections


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc_collections.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
    if "type" not in cfg:
        if default_args is None or "type" not in default_args:
            raise KeyError('`cfg` or `default_args` must contain the key "type", ' f"but got {cfg}\n{default_args}")
    if not isinstance(registry, Registry):
        raise TypeError("registry must be an mmcv.Registry object, " f"but got {type(registry)}")
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError("default_args must be a dict or None, " f"but got {type(default_args)}")

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop("type")
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry.name} registry")
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f"{obj_cls.__name__}: {e}")


class Registry:
    """A registry to map strings to classes.

    Registered object could be built from registry.
    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass
        >>> resnet = MODELS.build(dict(type='ResNet'))

    Please refer to
    https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html for
    advanced usage.

    Args:
        name (str): Registry name.
        build_func(func, optional): Build function to construct instance from
            Registry, func:`build_from_cfg` is used if neither ``parent`` or
            ``build_func`` is specified. If ``parent`` is specified and
            ``build_func`` is not given,  ``build_func`` will be inherited
            from ``parent``. Default: None.
        parent (Registry, optional): Parent registry. The class registered in
            children registry could be built from parent. Default: None.
        scope (str, optional): The scope of registry. It is the key to search
            for children registry. If not specified, scope will be the name of
            the package where class is defined, e.g. mmdet, mmcls, mmseg.
            Default: None.
    """

    def __init__(self, name, build_func=None, parent=None, scope=None):
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        self._scope = self.infer_scope() if scope is None else scope

        # self.build_func will be set with the following priority:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func
        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + f"(name={self._name}, " f"items={self._module_dict})"
        return format_str

    @staticmethod
    def infer_scope():
        """Infer the scope of registry.

        The name of the package where registry is defined will be returned.

        Example:
            # in mmdet/models/backbone/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            The scope of ``ResNet`` will be ``mmdet``.


        Returns:
            scope (str): The inferred scope name.
        """
        # inspect.stack() trace where this function is called, the index-2
        # indicates the frame where `infer_scope()` is called
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        split_filename = filename.split(".")
        return split_filename[0]

    @staticmethod
    def split_scope_key(key):
        """Split scope and key.

        The first scope will be split from key.

        Examples:
            >>> Registry.split_scope_key('mmdet.ResNet')
            'mmdet', 'ResNet'
            >>> Registry.split_scope_key('ResNet')
            None, 'ResNet'

        Return:
            scope (str, None): The first scope.
            key (str): The remaining key.
        """
        split_index = key.find(".")
        if split_index != -1:
            return key[:split_index], key[split_index + 1 :]
        else:
            return None, key

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def children(self):
        return self._children

    def get(self, key):
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                return self._module_dict[real_key]
        else:
            # get from self._children
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # goto root
                parent = self.parent
                while parent.parent is not None:
                    parent = parent.parent
                return parent.get(key)

    def build(self, *args, **kwargs):
        return self.build_func(*args, **kwargs, registry=self)

    def _add_children(self, registry):
        """Add children for a registry.

        The ``registry`` will be added as children based on its scope.
        The parent registry could build objects from children registry.

        Example:
            >>> models = Registry('models')
            >>> mmdet_models = Registry('models', parent=models)
            >>> @mmdet_models.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet = models.build(dict(type='mmdet.ResNet'))
        """

        assert isinstance(registry, Registry)
        assert registry.scope is not None
        # assert registry.scope not in self.children, f"scope {registry.scope} exists in {self.name} registry"
        self.children[registry.scope] = registry

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError("module must be a class, " f"but got {type(module_class)}")

        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                raise KeyError(f"{name} is already registered " f"in {self.name}")
            self._module_dict[name] = module_class

    def deprecated_register_module(self, cls=None, force=False):
        warnings.warn(
            "The old API of register_module(module, force=False) "
            "is deprecated and will be removed, please use the new API "
            "register_module(name=None, force=False, module=None) instead."
        )
        if cls is None:
            return partial(self.deprecated_register_module, force=force)
        self._register_module(cls, force=force)
        return cls

    def register_module(self, name=None, force=False, module=None):
        """Register a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)

        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")
        # NOTE: This is a walkaround to be compatible with the old api,
        # while it may introduce unexpected bugs.
        if isinstance(name, type):
            return self.deprecated_register_module(name, force=force)

        # raise the error ahead of time
        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                "name must be either of None, an instance of str or a sequence" f"  of str, but got {type(name)}"
            )

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/utils/misc.py
from itertools import repeat
from inspect import getfullargspec


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections_abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def deprecated_api_warning(name_dict, cls_name=None):
    """A decorator to check if some arguments are deprecate and try to replace
    deprecate src_arg_name to dst_arg_name.

    Args:
        name_dict(dict):
            key (str): Deprecate argument names.
            val (str): Expected argument names.

    Returns:
        func: New function.
    """

    def api_warning_wrapper(old_func):
        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get name of the function
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f"{cls_name}.{func_name}"
            if args:
                arg_names = args_info.args[: len(args)]
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in arg_names:
                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            "instead"
                        )
                        arg_names[arg_names.index(src_arg_name)] = dst_arg_name
            if kwargs:
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in kwargs:
                        assert dst_arg_name not in kwargs, (
                            f"The expected behavior is to replace "
                            f"the deprecated key `{src_arg_name}` to "
                            f"new key `{dst_arg_name}`, but got them "
                            f"in the arguments at the same time, which "
                            f"is confusing. `{src_arg_name} will be "
                            f"deprecated in the future, please "
                            f"use `{dst_arg_name}` instead."
                        )

                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            "instead"
                        )
                        kwargs[dst_arg_name] = kwargs.pop(src_arg_name)

            # apply converted arguments to the decorated method
            output = old_func(*args, **kwargs)
            return output

        return new_func

    return api_warning_wrapper


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/parallel/data_container.py
def assert_tensor_type(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f"{args[0].__class__.__name__} has no attribute " f"{func.__name__} for type {args[0].datatype}"
            )
        return func(*args, **kwargs)

    return wrapper


class DataContainer:
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self, data, stack=False, padding_value=0, cpu_only=False, pad_dims=2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.data)})"

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @property
    def pad_dims(self):
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/parallel/collate.py
def collate(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if len(batch) == 0:
        return batch

    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append([sample.data for sample in batch[i : i + samples_per_gpu]])
            return DataContainer(stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i : i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1], sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i : i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim - 1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(F.pad(sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(default_collate([sample.data for sample in batch[i : i + samples_per_gpu]]))
                else:
                    raise ValueError("pad_dims should be either None or integers (1-3)")

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append([sample.data for sample in batch[i : i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence) and not isinstance(batch[0], (str, bytes)):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {key: collate([d[key] for d in batch], samples_per_gpu) for key in batch[0]}
    else:
        # Handle non-collatable types like type objects, None, functions, etc.
        # Check if batch[0] is a type object or None
        first_item = batch[0]
        if isinstance(first_item, type) or first_item is None:
            # If all items are the same, return the first one
            if all(item is first_item or item == first_item for item in batch):
                return first_item
            # Otherwise return as a list
            return list(batch)

        # Check if it's a function or other callable that's not a class
        if callable(first_item) and not isinstance(first_item, type):
            # If all items are the same function, return the first one
            if all(item is first_item for item in batch):
                return first_item
            # Otherwise return as a list
            return list(batch)

        try:
            return default_collate(batch)
        except (TypeError, ValueError) as e:
            # If default_collate fails (e.g., for type objects, functions, etc.)
            # Check if all items are the same
            if all(item is first_item or item == first_item for item in batch):
                return first_item
            # Otherwise return as a list
            return list(batch)


# ============================================================================
# MMCV UTILITIES - Additional Dependencies
# ============================================================================

# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/utils/parrots_wrapper.py
from functools import partial

TORCH_VERSION = torch.__version__


def is_rocm_pytorch() -> bool:
    is_rocm = False
    if TORCH_VERSION != "parrots":
        try:
            from torch.utils.cpp_extension import ROCM_HOME

            is_rocm = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
        except ImportError:
            pass
    return is_rocm


def _get_cuda_home():
    """Get CUDA home directory. For CPU-only version, returns None.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/utils/parrots_wrapper.py
    """
    if TORCH_VERSION == "parrots":
        try:
            from parrots.utils.build_extension import CUDA_HOME
        except ImportError:
            return None
    else:
        if is_rocm_pytorch():
            try:
                from torch.utils.cpp_extension import ROCM_HOME

                CUDA_HOME = ROCM_HOME
            except ImportError:
                return None
        else:
            try:
                from torch.utils.cpp_extension import CUDA_HOME
            except ImportError:
                return None
    return CUDA_HOME


def get_build_config():
    """Get build configuration.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/utils/parrots_wrapper.py
    """
    if TORCH_VERSION == "parrots":
        try:
            from parrots.config import get_build_info

            return get_build_info()
        except ImportError:
            return "parrots not available"
    else:
        return torch.__config__.show()


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/utils/logging.py
logger_initialized = {}


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode="w"):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            "logger should be either a logging.Logger object, str, " f'"silent" or None, but got {type(logger)}'
        )


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/utils/version_utils.py
from packaging.version import parse


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Default: 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert "parrots" not in version_str
    version = parse(version_str)
    assert version.release, f"failed to parse version {version_str}"
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {"a": -3, "b": -2, "rc": -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f"unknown prerelease version {version.pre[0]}, " "version checking may go wrong")
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])
    else:
        release.extend([0, 0])
    return tuple(release)


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/utils/ext_loader.py
import importlib
import pkgutil
import warnings
from types import ModuleType


def load_ext(name, funcs):
    """Load extension module.

    For CPU-only version, this will try to import from mmcv._ext if available,
    otherwise return a mock module with stub functions.
    """
    if torch.__version__ != "parrots":
        try:
            # Try to import the actual extension module
            ext = importlib.import_module("mmcv._ext")
            for fun in funcs:
                if not hasattr(ext, fun):
                    warnings.warn(f"{fun} miss in module mmcv._ext")
            return ext
        except (ImportError, ModuleNotFoundError):
            # For CPU-only, create a mock module with stub functions
            # These will raise NotImplementedError if actually called
            mock_ext = ModuleType("_ext")
            for fun in funcs:

                def make_stub_func(func_name):
                    def stub_func(*args, **kwargs):
                        raise NotImplementedError(
                            f"{func_name} is not available. CUDA extensions are required for this operation. "
                            f"For CPU-only inference, use PyTorch fallback implementations."
                        )

                    return stub_func

                # Fix closure by binding func_name properly
                setattr(mock_ext, fun, make_stub_func(fun))
            return mock_ext
    else:
        # Parrots version - not used in CPU-only
        raise NotImplementedError("Parrots not supported in CPU-only version")


def check_ops_exist():
    ext_loader = pkgutil.find_loader("mmcv._ext")
    return ext_loader is not None


# Create ext_loader object with load_ext method
class ExtLoader:
    """Extension loader object that mimics mmcv.utils.ext_loader."""

    def load_ext(self, name, funcs):
        return load_ext(name, funcs)


ext_loader = ExtLoader()


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/ops/multi_scale_deform_attn.py
def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """CPU version of multi-scale deformable attention.

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """
    import torch.nn.functional as F

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/utils/config.py
# Note: This file has dependencies on addict.Dict and yapf
# For CPU-only compatibility, we include a minimal Dict implementation

try:
    from addict import Dict
except ImportError:
    # Minimal Dict implementation if addict is not available
    class Dict(dict):
        """A dictionary that allows attribute-style access."""

        def __init__(self, *args, **kwargs):
            super(Dict, self).__init__(*args, **kwargs)
            for arg in args:
                if isinstance(arg, dict):
                    for k, v in arg.items():
                        self[k] = self._convert(v)
            if kwargs:
                for k, v in kwargs.items():
                    self[k] = self._convert(v)

        def _convert(self, value):
            if isinstance(value, dict) and not isinstance(value, Dict):
                return Dict(value)
            elif isinstance(value, list):
                return [self._convert(item) for item in value]
            return value

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        def __setattr__(self, name, value):
            self[name] = value

        def __delattr__(self, name):
            del self[name]


BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
DEPRECATION_KEY = "_deprecation_"
RESERVED_KEYS = ["filename", "text", "pretty_text"]


class ConfigDict(Dict):
    """A dictionary for config that raises KeyError on missing keys."""

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no " f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


# Note: The full Config class from mmcv.utils.config is 688 lines
# It depends on yapf, mmcv.utils.misc, mmcv.utils.path
# For now, we include ConfigDict which is the most critical part
# The full Config class can be added if needed, but it requires many more dependencies


# Placeholder for Config class - to be fully implemented if needed
# Full implementation at: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/utils/config.py
class Config(ConfigDict):
    """Placeholder for Config class.

    Full implementation available at:
    https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/utils/config.py

    This is a minimal implementation. The full Config class supports:
    - Loading from .py, .json, .yaml files
    - Base config inheritance
    - Config merging
    - Predefined variable substitution
    """


# ============================================================================
# MMCV RUNNER - https://github.com/open-mmlab/mmcv/tree/v1.4.0/mmcv/runner
# ============================================================================


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/runner/dist_utils.py
def master_only(func):
    """Decorator to ensure function only runs on master process.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/runner/dist_utils.py
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/runner/base_module.py
from collections import defaultdict
from logging import FileHandler


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab.

    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameter initialization. Compared with
    ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.

        - ``init_cfg``: the config to control the initialization.
        - ``init_weights``: The function of parameter
            initialization and recording initialization
            information.
        - ``_params_init_info``: Used to track the parameter
            initialization information. This attribute only
            exists during executing the ``init_weights``.

    Args:
        init_cfg (dict, optional): Initialization config dict.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/runner/base_module.py
    """

    def __init__(self, init_cfg=None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super(BaseModule, self).__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, "_params_init_info"):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param]["init_info"] = (
                    f"The value is the same before and "
                    f"after calling `init_weights` "
                    f"of {self.__class__.__name__} "
                )
                self._params_init_info[param]["tmp_mean_value"] = param.data.mean()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        # Get the initialized logger, if not exist,
        # create a logger named `mmcv`
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else "mmcv"

        # Note: BaseModule.init_weights depends on mmcv.cnn.initialize
        # and mmcv.cnn.utils.weight_init.update_init_info
        # These will need to be added as dependencies
        # For now, we provide a minimal implementation
        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(f"initialize {module_name} with init_cfg {self.init_cfg}", logger=logger_name)
                # TODO: Call initialize(self, self.init_cfg) when mmcv.cnn is added
                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of
                    # the pre-trained model
                    # from being overwritten by
                    # the `init_weights`
                    if self.init_cfg.get("type") == "Pretrained":
                        return

            for m in self.children():
                if hasattr(m, "init_weights"):
                    m.init_weights()
                    # TODO: Call update_init_info when mmcv.cnn.utils.weight_init is added

            self._is_init = True
        else:
            warnings.warn(f"init_weights of {self.__class__.__name__} has " f"been called more than once.")

        if is_top_level_module:
            self._dump_init_info(logger_name)

            for sub_module in self.modules():
                if hasattr(sub_module, "_params_init_info"):
                    del sub_module._params_init_info

    @master_only
    def _dump_init_info(self, logger_name):
        """Dump the initialization information to a file named
        `initialization.log.json` in workdir.

        Args:
            logger_name (str): The name of logger.
        """

        logger = get_logger(logger_name)

        with_file_handler = False
        # dump the information to the logger file if there is a `FileHandler`
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write("Name of parameter - Initialization information\n")
                for name, param in self.named_parameters():
                    handler.stream.write(
                        f"\n{name} - {param.shape}: " f"\n{self._params_init_info[param]['init_info']} \n"
                    )
                handler.stream.flush()
                with_file_handler = True
        if not with_file_handler:
            for name, param in self.named_parameters():
                print_log(
                    f"\n{name} - {param.shape}: " f"\n{self._params_init_info[param]['init_info']} \n ",
                    logger=logger_name,
                )

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f"\ninit_cfg={self.init_cfg}"
        return s


class Sequential(BaseModule, nn.Sequential):
    """Sequential module in openmmlab.

    Args:
        init_cfg (dict, optional): Initialization config dict.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/runner/base_module.py
    """

    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, nn.ModuleList):
    """ModuleList in openmmlab.

    Args:
        modules (iterable, optional): an iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/runner/base_module.py
    """

    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/cnn/bricks/transformer/transformer_layer_sequence.py
class TransformerLayerSequence(BaseModule):
    """Base class for TransformerEncoder and TransformerDecoder in vision transformer.

    Args:
        transformerlayer (list[dict] | dict): Configs of `TransformerLayer` in
            encoder/decoder. If it is a dict, it would be repeated `num_layers` times
            to a list.
        num_layers (int): The number of `TransformerLayer`. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, transformerlayer=None, num_layers=None, init_cfg=None):
        super(TransformerLayerSequence, self).__init__(init_cfg)
        self.num_layers = num_layers
        self.layers = ModuleList()
        if transformerlayer is None:
            transformerlayer = []
        if isinstance(transformerlayer, dict):
            transformerlayer = [copy.deepcopy(transformerlayer) for _ in range(num_layers)]
        else:
            assert num_layers is None or len(transformerlayer) == num_layers, (
                f"The length of "
                f"transformerlayer {len(transformerlayer)} is "
                f"not consistent with num_layers {num_layers}."
            )
        for i, layer_cfg in enumerate(transformerlayer):
            self.layers.append(build_transformer_layer(layer_cfg))


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/runner/dist_utils.py
def get_dist_info():
    """Get distributed information.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/runner/dist_utils.py
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/runner/fp16_utils.py
from inspect import getfullargspec


def cast_tensor_type(inputs, src_type, dst_type):
    """Cast type of tensor.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/runner/fp16_utils.py
    """
    if isinstance(inputs, torch.Tensor):
        return inputs.type(dst_type) if inputs.type() == src_type else inputs
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, Mapping):
        return type(inputs)({k: cast_tensor_type(v, src_type, dst_type) for k, v in inputs.items()})
    elif isinstance(inputs, (list, tuple)):
        return type(inputs)(cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs


def auto_fp16(apply_to=None, out_fp32=False):
    """Decorator to enable fp16 training automatically.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If inputs arguments are fp32 tensors, they will
    be converted to fp16 automatically. Arguments other than fp32 tensors are
    ignored. If you are using PyTorch >= 1.6, torch.cuda.amp is used as the
    backend, otherwise, original mmcv implementation will be adopted.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp32 (bool): Whether to convert the output back to fp32.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/runner/fp16_utils.py
    """

    def auto_fp16_wrapper(old_func):
        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError("@auto_fp16 can only be used to decorate the " "method of nn.Module")
            if not (hasattr(args[0], "fp16_enabled") and args[0].fp16_enabled):
                return old_func(*args, **kwargs)

            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            # NOTE: default args are not taken into consideration
            if args:
                arg_names = args_info.args[: len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(cast_tensor_type(args[i], torch.float, torch.half))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(arg_value, torch.float, torch.half)
                    else:
                        new_kwargs[arg_name] = arg_value

            # apply converted arguments to the decorated method
            output = old_func(*new_args, **new_kwargs)

            # cast the results back to fp32 if necessary
            if out_fp32 and isinstance(output, torch.Tensor):
                output = output.float()
            elif out_fp32:
                output = cast_tensor_type(output, torch.half, torch.float)
            return output

        return new_func

    return auto_fp16_wrapper


def force_fp32(apply_to=None, out_fp16=False):
    """Decorator to convert input arguments to fp32 in force.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If there are some inputs that must be processed
    in fp32 mode, then this decorator can handle it. If inputs arguments are
    fp16 tensors, they will be converted to fp32 automatically. Arguments other
    than fp16 tensors are ignored. If you are using PyTorch >= 1.6,
    torch.cuda.amp is used as the backend, otherwise, original mmcv
    implementation will be adopted.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp16 (bool): Whether to convert the output back to fp16.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/runner/fp16_utils.py
    """

    def force_fp32_wrapper(old_func):
        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError("@force_fp32 can only be used to decorate the " "method of nn.Module")
            if not (hasattr(args[0], "fp16_enabled") and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            if args:
                arg_names = args_info.args[: len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(cast_tensor_type(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = dict()
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(arg_value, torch.half, torch.float)
                    else:
                        new_kwargs[arg_name] = arg_value

            # apply converted arguments to the decorated method
            output = old_func(*new_args, **new_kwargs)

            # cast the results back to fp16 if necessary
            if out_fp16 and isinstance(output, torch.Tensor):
                output = output.half()
            elif out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            return output

        return new_func

    return force_fp32_wrapper


# ============================================================================
# MMCV CNN - https://github.com/open-mmlab/mmcv/tree/v1.4.0/mmcv/cnn
# ============================================================================

# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/cnn/utils/weight_init.py

INITIALIZERS = Registry("initializer")


def update_init_info(module, init_info):
    """Update the `_params_init_info` in the module if the value of parameters
    are changed.

    Args:
        module (obj:`nn.Module`): The module of PyTorch with a user-defined
            attribute `_params_init_info` which records the initialization
            information.
        init_info (str): The string that describes the initialization.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/cnn/utils/weight_init.py
    """
    assert hasattr(module, "_params_init_info"), f"Can not find `_params_init_info` in {module}"
    for name, param in module.named_parameters():
        assert param in module._params_init_info, (
            f"Find a new :obj:`Parameter` "
            f"named `{name}` during executing the "
            f"`init_weights` of "
            f"`{module.__class__.__name__}`. "
            f"Please do not add or "
            f"replace parameters during executing "
            f"the `init_weights`. "
        )

        # The parameter has been changed during executing the
        # `init_weights` of module
        mean_value = param.data.mean()
        if module._params_init_info[param]["tmp_mean_value"] != mean_value:
            module._params_init_info[param]["init_info"] = init_info
            module._params_init_info[param]["tmp_mean_value"] = mean_value


def constant_init(module, val, bias=0):
    """Initialize module parameters with constant values.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/cnn/utils/weight_init.py
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    """Initialize module parameters with Xavier initialization.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/cnn/utils/weight_init.py
    """
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """Initialize conv/fc bias value according to a given probability value.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/cnn/utils/weight_init.py
    """
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/cnn/builder.py
def build_model_from_cfg(cfg, registry, default_args=None):
    """Build a PyTorch model from config dict(s). Different from
    ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a config
            dict or a list of config dicts. If cfg is a list, a
            the built modules will be wrapped with ``nn.Sequential``.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/cnn/builder.py
    """
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


MODELS = Registry("model", build_func=build_model_from_cfg)

# Note: mmcv.cnn.builder also defines CONV_LAYERS, NORM_LAYERS, ACTIVATION_LAYERS, etc.
# These are used via build_conv_layer, build_norm_layer, build_activation_layer
# For now, we include the MODELS registry. Other builders can be added if needed.


# ============================================================================
# MMCV CNN BRICKS - https://github.com/open-mmlab/mmcv/tree/v1.4.0/mmcv/cnn/bricks
# ============================================================================

# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/cnn/bricks/registry.py
CONV_LAYERS = Registry("conv layer")
NORM_LAYERS = Registry("norm layer")
ACTIVATION_LAYERS = Registry("activation layer")
PADDING_LAYERS = Registry("padding layer")
UPSAMPLE_LAYERS = Registry("upsample layer")
PLUGIN_LAYERS = Registry("plugin layer")

DROPOUT_LAYERS = Registry("drop out layers")
POSITIONAL_ENCODING = Registry("position encoding")
ATTENTION = Registry("attention")
FEEDFORWARD_NETWORK = Registry("feed-forward Network")
TRANSFORMER_LAYER = Registry("transformerLayer")
TRANSFORMER_LAYER_SEQUENCE = Registry("transformer-layers sequence")
TRANSFORMER = Registry("Transformer")
TRANSFORMER = Registry("Transformer")


def build_positional_encoding(cfg, default_args=None):
    """Build positional encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/utils/positional_encoding.py
@POSITIONAL_ENCODING.register_module()
class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for BEV.

    Source: https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/utils/positional_encoding.py
    """

    def __init__(self, num_feats, row_num_embed=50, col_num_embed=50, init_cfg=None):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = (
            torch.cat((x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)), dim=-1)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(mask.shape[0], 1, 1, 1)
        )
        return pos


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/cnn/bricks/transformer/builder.py
def build_attention(cfg, default_args=None):
    """Build attention module."""
    return build_from_cfg(cfg, ATTENTION, default_args)


def build_feedforward_network(cfg, default_args=None):
    """Build feedforward network."""
    return build_from_cfg(cfg, FEEDFORWARD_NETWORK, default_args)


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/cnn/bricks/transformer/ffn.py
@FEEDFORWARD_NETWORK.register_module()
class FFN(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Defaults to 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims,
        feedforward_channels,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(FFN, self).__init__(init_cfg)
        assert num_fcs >= 2, f"num_fcs should be at least 2. got {num_fcs}."
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        if act_cfg.get("type") == "GELU":
            self.activate = nn.GELU()
        elif act_cfg.get("type") == "ReLU":
            self.activate = nn.ReLU(inplace=act_cfg.get("inplace", False))
        else:
            self.activate = nn.ReLU(inplace=False)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Linear(in_channels, feedforward_channels))
            layers.append(self.activate)
            layers.append(nn.Dropout(ffn_drop))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/cnn/bricks/transformer/base_transformer_layer.py
@TRANSFORMER_LAYER.register_module()
class BaseTransformerLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None ):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None ):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.

    Source: https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/cnn/bricks/transformer/base_transformer_layer.py
    """

    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=None,
        norm_cfg=dict(type="LN"),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        deprecated_args = dict(
            feedforward_channels="feedforward_channels", ffn_dropout="ffn_drop", ffn_num_fcs="num_fcs"
        )
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f"The arguments `{ori_name}` in BaseTransformerLayer "
                    f"has been deprecated, now you should set `{new_name}` "
                    f"and other FFN related arguments "
                    f"to a dict named `ffn_cfgs`. "
                )
                ffn_cfgs[new_name] = kwargs[ori_name]

        super(BaseTransformerLayer, self).__init__(init_cfg)

        self.batch_first = batch_first

        assert set(operation_order) & set(["self_attn", "norm", "ffn", "cross_attn"]) == set(operation_order), (
            f"The operation_order of"
            f" {self.__class__.__name__} should "
            f"contains all four operation type "
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"
        )

        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length "
                f"of attn_cfg {num_attn} is "
                f"not consistent with the number of attention"
                f"in operation_order {operation_order}."
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                if "batch_first" in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]["batch_first"]
                else:
                    attn_cfgs[index]["batch_first"] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count("ffn")
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if "embed_dims" not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]["embed_dims"] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]["embed_dims"] == self.embed_dims

            self.ffns.append(build_feedforward_network(ffn_cfgs[ffn_index]))

        self.norms = ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `BaseTransformerLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/utils/transformer.py


@TRANSFORMER_LAYER.register_module()
class DetrTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(
        self,
        attn_cfgs,
        feedforward_channels,
        ffn_dropout=0.0,
        operation_order=None,
        act_cfg=dict(type="ReLU", inplace=True),
        norm_cfg=dict(type="LN"),
        ffn_num_fcs=2,
        **kwargs,
    ):
        super(DetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs,
        )
        assert len(operation_order) == 6
        assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/ops/multi_scale_deform_attn.py
# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/cnn/bricks/transformer.py
@ATTENTION.register_module()
class MultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        dropout_layer=dict(type="Dropout", drop_prob=0.0),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        super().__init__(init_cfg)
        if "dropout" in kwargs:
            import warnings

            warnings.warn(
                "The arguments `dropout` in MultiheadAttention "
                "has been deprecated, now you can separately "
                "set `attn_drop`(float), proj_drop(float), "
                "and `dropout_layer`(dict) ",
                DeprecationWarning,
            )
            attn_drop = kwargs["dropout"]
            dropout_layer["drop_prob"] = kwargs.pop("dropout")

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        from models.experimental.BEVFormerV2.projects.mmdet3d_plugin.dependency import build_dropout

        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `MultiheadAttention`."""
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos

        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(query=query, key=key, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


@ATTENTION.register_module()
class MultiScaleDeformableAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        dropout=0.1,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        self._is_init = True

    @deprecated_api_warning({"residual": "identity"}, cls_name="MultiScaleDeformableAttention")
    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be" f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )
        # For CPU-only inference, use PyTorch fallback
        output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


def build_transformer_layer(cfg, default_args=None):
    """Build transformer layer."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER, default_args)


def build_transformer_layer_sequence(cfg, default_args=None):
    """Build transformer layer sequence."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/cnn/utils.py
def build_norm_layer(cfg, num_features, postfix=""):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
        num_features (int): Number of input channels.
        postfix (int, str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    if "type" not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")

    # Register common norm layers if not already registered
    if layer_type == "BN" and "BN" not in NORM_LAYERS:
        NORM_LAYERS.register_module(name="BN", module=nn.BatchNorm2d)
    elif layer_type == "SyncBN" and "SyncBN" not in NORM_LAYERS:
        # For CPU-only, SyncBN falls back to BatchNorm2d
        NORM_LAYERS.register_module(name="SyncBN", module=nn.BatchNorm2d)
    elif layer_type == "LN" and "LN" not in NORM_LAYERS:
        NORM_LAYERS.register_module(name="LN", module=nn.LayerNorm)

    if layer_type not in NORM_LAYERS:
        raise KeyError(f"Unrecognized norm type {layer_type}")

    norm_layer = NORM_LAYERS.get(layer_type)
    abbr_map = {"BN": "bn", "SyncBN": "bn", "LN": "ln"}
    abbr = abbr_map.get(layer_type, layer_type.lower())

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    layer = norm_layer(num_features, **cfg_)

    if not requires_grad:
        for param in layer.parameters():
            param.requires_grad = False
    else:
        for param in layer.parameters():
            param.requires_grad = True

    return name, layer


def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a conv layer.
            Default: None.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type="Conv2d")
    else:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict or None")
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")

    # Register common conv layers if not already registered
    if layer_type == "Conv2d" and "Conv2d" not in CONV_LAYERS:
        CONV_LAYERS.register_module(name="Conv2d", module=nn.Conv2d)

    if layer_type not in CONV_LAYERS:
        raise KeyError(f"Unrecognized conv type {layer_type}")

    conv_layer = CONV_LAYERS.get(layer_type)
    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/backbones/resnet.py
class BasicBlock(BaseModule):
    """Basic block for ResNet."""

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        dcn=None,
        plugins=None,
        init_cfg=None,
    ):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, "Not implemented yet."
        assert plugins is None, "Not implemented yet."

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg, inplanes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False
        )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

        if self.with_cp and x.requires_grad:
            import torch.utils.checkpoint as checkpoint

            out = checkpoint.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/backbones/resnet.py
class Bottleneck(BaseModule):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        dcn=None,
        plugins=None,
        init_cfg=None,
    ):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ["pytorch", "caffe"]
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ["after_conv1", "after_conv2", "after_conv3"]
            assert all(p["position"] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [plugin["cfg"] for plugin in plugins if plugin["position"] == "after_conv1"]
            self.after_conv2_plugins = [plugin["cfg"] for plugin in plugins if plugin["position"] == "after_conv2"]
            self.after_conv3_plugins = [plugin["cfg"] for plugin in plugins if plugin["position"] == "after_conv3"]

        if self.style == "pytorch":
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, kernel_size=1, stride=self.conv1_stride, bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop("fallback_on_stride", False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
        else:
            assert self.conv_cfg is None, "conv_cfg must be None for DCN"
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(conv_cfg, planes, planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(plugin, in_channels=in_channels, postfix=plugin.pop("postfix", ""))
            assert not hasattr(self, name), f"duplicate plugin {name}"
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            import torch.utils.checkpoint as checkpoint

            out = checkpoint.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/utils/res_layer.py
class ResLayer(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(
        self,
        block,
        inplanes,
        planes,
        num_blocks,
        stride=1,
        avg_down=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        downsample_first=True,
        **kwargs,
    ):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False)
                )
            downsample.extend(
                [
                    build_conv_layer(
                        conv_cfg, inplanes, planes * block.expansion, kernel_size=1, stride=conv_stride, bias=False
                    ),
                    build_norm_layer(norm_cfg, planes * block.expansion)[1],
                ]
            )
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs,
                )
            )
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(inplanes=inplanes, planes=planes, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs)
                )

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(inplanes=inplanes, planes=inplanes, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs)
                )
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs,
                )
            )
        super(ResLayer, self).__init__(*layers)


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/backbones/resnet.py
from torch.nn.modules.batchnorm import _BatchNorm


# Note: BACKBONES is defined later in the file (line 2073), but we need it here
# The decorator will be applied when the module is loaded
def build_plugin_layer(cfg, postfix="", in_channels=None, **kwargs):
    """Build plugin layer."""
    if cfg is None:
        return None, None
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()
    plugin_type = cfg_.pop("type")
    if plugin_type in PLUGIN_LAYERS:
        plugin_layer = PLUGIN_LAYERS.get(plugin_type)
        if in_channels is not None:
            cfg_.setdefault("in_channels", in_channels)
        layer = plugin_layer(**cfg_, **kwargs)
        name = f"{plugin_type}{postfix}"
        return name, layer
    else:
        # For CPU-only inference, return None if plugin is not available
        return None, None


# ResNet class - will be registered with BACKBONES after BACKBONES is defined
class ResNet(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style="pytorch",
        deep_stem=False,
        avg_down=False,
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        dcn=None,
        stage_with_dcn=(False, False, False, False),
        plugins=None,
        with_cp=False,
        zero_init_residual=True,
        pretrained=None,
        init_cfg=None,
    ):
        super(ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet")

        block_init_cfg = None
        assert not (init_cfg and pretrained), "init_cfg and pretrained cannot be setting at the same time"
        if isinstance(pretrained, str):
            warnings.warn("DeprecationWarning: pretrained is deprecated, " 'please use "init_cfg" instead')
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type="Kaiming", layer="Conv2d"),
                    dict(type="Constant", val=1, layer=["_BatchNorm", "GroupNorm"]),
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(type="Constant", val=0, override=dict(name="norm2"))
                    elif block is Bottleneck:
                        block_init_cfg = dict(type="Constant", val=0, override=dict(name="norm3"))
        else:
            raise TypeError("pretrained must be a str or None")

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage."""
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop("stages", None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg, in_channels, stem_channels // 2, kernel_size=3, stride=2, padding=1, bias=False
                ),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg, stem_channels // 2, stem_channels, kernel_size=3, stride=1, padding=1, bias=False
                ),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg, in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/cnn/bricks/drop.py
def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/cnn/bricks/drop.py
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


@DROPOUT_LAYERS.register_module()
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/cnn/bricks/drop.py
    """

    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


@DROPOUT_LAYERS.register_module()
class Dropout(nn.Dropout):
    """A wrapper for ``torch.nn.Dropout``, We rename the ``p`` of
    ``torch.nn.Dropout`` to ``drop_prob`` so as to be consistent with
    ``DropPath``

    Args:
        drop_prob (float): Probability of the elements to be
            zeroed. Default: 0.5.
        inplace (bool):  Do the operation inplace or not. Default: False.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/cnn/bricks/drop.py
    """

    def __init__(self, drop_prob=0.5, inplace=False):
        super().__init__(p=drop_prob, inplace=inplace)


def build_dropout(cfg, default_args=None):
    """Builder for drop out layers.

    Source: https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/cnn/bricks/drop.py
    """
    return build_from_cfg(cfg, DROPOUT_LAYERS, default_args)


# ============================================================================
# MMDET MODELS - https://github.com/open-mmlab/mmdetection/tree/v2.14.0/mmdet/models
# ============================================================================

# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/builder.py
MMCV_MODELS = Registry("models")
_MODELS_DET = Registry("models", parent=MMCV_MODELS)

BACKBONES = _MODELS_DET
NECKS = _MODELS_DET
ROI_EXTRACTORS = _MODELS_DET
SHARED_HEADS = _MODELS_DET
HEADS = _MODELS_DET
LOSSES = _MODELS_DET
DETECTORS = _MODELS_DET

# Register ResNet with BACKBONES now that BACKBONES is defined
# ResNet is defined earlier in the file (around line 2334)
# We'll register it at the end of the file after all classes are defined


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/losses/utils.py
def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/losses/focal_loss.py
def py_sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25, reduction="mean", avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(self, use_sigmoid=True, gamma=2.0, alpha=0.25, reduction="mean", loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, "Only sigmoid focal loss supported now."
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            # For CPU-only inference, use PyTorch fallback
            num_classes = pred.size(1)
            target = F.one_hot(target, num_classes=num_classes + 1)
            target = target[:, :num_classes]
            calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred, target, weight, gamma=self.gamma, alpha=self.alpha, reduction=reduction, avg_factor=avg_factor
            )

        else:
            raise NotImplementedError
        return loss_cls


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/losses/utils.py
def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction="mean", avg_factor=None, **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/losses/smooth_l1_loss.py
@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    return loss


@weighted_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction="mean", loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred, target, weight, beta=self.beta, reduction=reduction, avg_factor=avg_factor, **kwargs
        )
        return loss_bbox


@LOSSES.register_module()
class L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction="mean", loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * l1_loss(pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/core/bbox/iou_calculators/iou2d_calculator.py
def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()
    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
    """
    assert mode in ["iou", "iof", "giou"], f"Unsupported mode {mode}"
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    eps_tensor = union.new_tensor([eps])
    union = torch.max(union, eps_tensor)
    ious = overlap / union
    if mode in ["iou", "iof"]:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps_tensor)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/losses/iou_loss.py
@weighted_loss
def giou_loss(pred, target, eps=1e-7):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode="giou", is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss


@LOSSES.register_module()
class GIoULoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        if (weight is not None) and (not torch.any(weight > 0)) and (reduction_override != "none"):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(
            pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor, **kwargs
        )
        return loss


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/detectors/base.py
from abc import ABCMeta, abstractmethod


class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False
        if init_cfg is not None:
            self.init_cfg = init_cfg

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, "roi_head") and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return (hasattr(self, "roi_head") and self.roi_head.with_bbox) or (
            hasattr(self, "bbox_head") and self.bbox_head is not None
        )

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return (hasattr(self, "roi_head") and self.roi_head.with_mask) or (
            hasattr(self, "mask_head") and self.mask_head is not None
        )

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        batch_input_shape = tuple(imgs[0].size()[-2:])
        for img_meta in img_metas:
            img_meta["batch_input_shape"] = batch_input_shape

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, "imgs"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got {type(var)}")

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f"num of augmentations ({len(imgs)}) " f"!= num of image meta ({len(img_metas)})")

        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]["batch_input_shape"] = tuple(img.size()[-2:])

        if num_augs == 1:
            if "proposals" in kwargs:
                kwargs["proposals"] = kwargs["proposals"][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, "aug test does not support " "inference with batch size " f"{imgs[0].size(0)}"
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=("img",))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/dense_heads/base_dense_head.py
class BaseDenseHead(nn.Module):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__()
        self.init_cfg = init_cfg

    def init_weights(self):
        """Initialize weights of the head."""


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/dense_heads/dense_test_mixins.py
class BBoxTestMixin:
    """Mixin class for test time bbox augmentation."""

    def simple_test_bboxes(self, *args, **kwargs):
        """Test det bboxes without test-time augmentation."""


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/dense_heads/anchor_free_head.py
@HEADS.register_module()
class AnchorFreeHead(BaseDenseHead, BBoxTestMixin):
    """Anchor-free head (FCOS, Fovea, RepPoints, etc.)."""

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        stacked_convs=4,
        strides=(4, 8, 16, 32, 64),
        dcn_on_last_conv=False,
        conv_bias="auto",
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="IoULoss", loss_weight=1.0),
        conv_cfg=None,
        norm_cfg=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=dict(
            type="Normal",
            layer="Conv2d",
            std=0.01,
            override=dict(type="Normal", name="conv_cls", std=0.01, bias_prob=0.01),
        ),
    ):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == "auto" or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = build_loss(loss_cls) if isinstance(loss_cls, dict) else loss_cls
        self.loss_bbox = build_loss(loss_bbox) if isinstance(loss_bbox, dict) else loss_bbox
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

    def forward(self, feats):
        """Forward features from the upstream network."""
        return multi_apply(self.forward_single, feats)[:2]

    def forward_single(self, x):
        """Forward features of a single scale level."""


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/dense_heads/detr_head.py
@HEADS.register_module()
class DETRHead(AnchorFreeHead):
    """Implements the DETR transformer head."""

    _version = 2

    def __init__(
        self,
        num_classes,
        in_channels,
        num_query=100,
        num_reg_fcs=2,
        transformer=None,
        sync_cls_avg_factor=False,
        positional_encoding=dict(type="SinePositionalEncoding", num_feats=128, normalize=True),
        loss_cls=dict(type="CrossEntropyLoss", bg_cls_weight=0.1, use_sigmoid=False, loss_weight=1.0, class_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(
                type="HungarianAssigner",
                cls_cost=dict(type="ClassificationCost", weight=1.0),
                reg_cost=dict(type="BBoxL1Cost", weight=5.0),
                iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
            )
        ),
        test_cfg=dict(max_per_img=100),
        init_cfg=None,
        **kwargs,
    ):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get("class_weight", None)
        if class_weight is not None and (self.__class__ is DETRHead):
            assert isinstance(class_weight, float), (
                "Expected " "class_weight to have type float. Found " f"{type(class_weight)}."
            )
            bg_cls_weight = loss_cls.get("bg_cls_weight", class_weight)
            assert isinstance(bg_cls_weight, float), (
                "Expected " "bg_cls_weight to have type float. Found " f"{type(bg_cls_weight)}."
            )
            class_weight = torch.ones(num_classes + 1) * class_weight
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({"class_weight": class_weight})
            if "bg_cls_weight" in loss_cls:
                loss_cls.pop("bg_cls_weight")
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert "assigner" in train_cfg, "assigner should be provided " "when train_cfg is set."
            assigner = train_cfg["assigner"]
            assert loss_cls["loss_weight"] == assigner["cls_cost"]["weight"], (
                "The classification weight for loss and matcher should be" "exactly the same."
            )
            assert loss_bbox["loss_weight"] == assigner["reg_cost"]["weight"], (
                "The regression L1 weight for loss and matcher " "should be exactly the same."
            )
            assert loss_iou["loss_weight"] == assigner["iou_cost"]["weight"], (
                "The regression iou weight for loss and matcher should be" "exactly the same."
            )
            self.assigner = build_assigner(assigner)
            sampler_cfg = dict(type="PseudoSampler")
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls) if isinstance(loss_cls, dict) else loss_cls
        self.loss_bbox = build_loss(loss_bbox) if isinstance(loss_bbox, dict) else loss_bbox
        self.loss_iou = build_loss(loss_iou) if isinstance(loss_iou, dict) else loss_iou

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

    def forward(self, feats, img_metas):
        """Forward function."""
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single, feats, img_metas_list)

    def forward_single(self, x, img_metas):
        """Forward function for a single feature level."""

    @force_fp32(apply_to=("all_cls_scores_list", "all_bbox_preds_list"))
    def loss(
        self, all_cls_scores_list, all_bbox_preds_list, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore=None
    ):
        """Loss function."""


# ============================================================================
# MMDET DATASETS - https://github.com/open-mmlab/mmdetection/tree/v2.14.0/mmdet/datasets
# ============================================================================

# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/datasets/builder.py
DATASETS = Registry("dataset")
PIPELINES = Registry("pipeline")


def _concat_dataset(cfg, default_args=None):
    """Concat multiple datasets.

    Source: https://raw.githubusercontent.com/open-mmlab/mmdetection/v2.14.0/mmdet/datasets/builder.py

    Note: This function requires ConcatDataset from mmdet.datasets.dataset_wrappers
    which should be added as a dependency if needed.
    """
    # Note: This function depends on build_dataset and ConcatDataset
    # For now, we provide a simplified version
    # Full implementation would require adding ConcatDataset and build_dataset
    ann_files = cfg["ann_file"]
    img_prefixes = cfg.get("img_prefix", None)
    seg_prefixes = cfg.get("seg_prefix", None)
    proposal_files = cfg.get("proposal_file", None)
    separate_eval = cfg.get("separate_eval", True)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        # pop 'separate_eval' since it is not a valid key for common datasets.
        if "separate_eval" in data_cfg:
            data_cfg.pop("separate_eval")
        data_cfg["ann_file"] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg["img_prefix"] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg["seg_prefix"] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg["proposal_file"] = proposal_files[i]
        datasets.append(build_from_cfg(data_cfg, DATASETS, default_args))

    # Note: Full implementation requires ConcatDataset from mmdet.datasets.dataset_wrappers
    # For now, return a list - the actual usage should handle this
    # TODO: Add ConcatDataset class if needed
    if len(datasets) > 1:
        # Simplified: return first dataset if multiple, full implementation needs ConcatDataset
        return datasets[0] if len(datasets) == 1 else datasets
    return datasets[0] if datasets else None


# ============================================================================
# MMDET3D CORE - https://github.com/open-mmlab/mmdetection3d/tree/v0.17.1/mmdet3d/core
# ============================================================================


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/core/bbox/transforms.py
def bbox3d2result(bboxes, scores, labels, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
        labels (torch.Tensor): Labels with shape of (n, ).
        scores (torch.Tensor): Scores with shape of (n, ).
        attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    result_dict = dict(boxes_3d=bboxes.to("cpu"), scores_3d=scores.cpu(), labels_3d=labels.cpu())

    if attrs is not None:
        result_dict["attrs_3d"] = attrs.cpu()

    return result_dict


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/models/detectors/base.py
# from mmcv.parallel import DataContainer as DC
DC = DataContainer


class Base3DDetector(BaseDetector):
    """Base class for detectors."""

    def forward_test(self, points, img_metas, img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(points, "points"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError("num of augmentations ({}) != num of image meta ({})".format(len(points), len(img_metas)))

        if num_augs == 1:
            img = [img] if img is None else img
            return self.simple_test(points[0], img_metas[0], img[0], **kwargs)
        else:
            return self.aug_test(points, img_metas, img, **kwargs)

    @auto_fp16(apply_to=("img", "points"))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (list[dict]): Input points and the information of the sample.
            result (list[dict]): Prediction results.
            out_dir (str): Output directory of visualization result.
        """
        for batch_id in range(len(result)):
            if isinstance(data["points"][0], DC):
                points = data["points"][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data["points"][0], torch.Tensor):
                points = data["points"][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} " f"for visualization!")
            if isinstance(data["img_metas"][0], DC):
                pts_filename = data["img_metas"][0]._data[0][batch_id]["pts_filename"]
                box_mode_3d = data["img_metas"][0]._data[0][batch_id]["box_mode_3d"]
            elif mmcv.is_list_of(data["img_metas"][0], dict):
                pts_filename = data["img_metas"][0][batch_id]["pts_filename"]
                box_mode_3d = data["img_metas"][0][batch_id]["box_mode_3d"]
            else:
                ValueError(f"Unsupported data type {type(data['img_metas'][0])} " f"for visualization!")
            file_name = osp.split(pts_filename)[-1].split(".")[0]

            assert out_dir is not None, "Expect out_dir, got none."

            pred_bboxes = result[batch_id]["boxes_3d"]

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR, Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d, Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(f"Unsupported box_mode_3d {box_mode_3d} for convertion!")
            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name)


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/models/detectors/mvx_two_stage.py
import warnings

# from mmcv.runner import force_fp32


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments."""
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def merge_aug_bboxes_3d(aug_bboxes, img_metas, test_cfg):
    """Merge augmented detection bboxes and scores."""
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        recovered_bboxes.append(bboxes)
    return recovered_bboxes[0] if len(recovered_bboxes) == 1 else recovered_bboxes


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/core/bbox/prior_generators.py
def build_prior_generator(cfg, default_args=None):
    """Builder for prior generator."""
    # For now, return a placeholder - full implementation would require AnchorGenerator classes
    return None


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/dense_heads/anchor_head.py
class AnchorTrainMixin:
    """Mixin class for anchor head training."""

    def add_sin_difference(self, boxes1, boxes2):
        """Convert the rotation difference to difference in sin."""
        return boxes1, boxes2


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/models/dense_heads/anchor3d_head.py
@HEADS.register_module()
class Anchor3DHead(BaseModule, AnchorTrainMixin):
    """Anchor head for SECOND/PointPillars/MVXNet/PartA2."""

    def __init__(
        self,
        num_classes,
        in_channels,
        train_cfg,
        test_cfg,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type="Anchor3DRangeGenerator",
            range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
            strides=[2],
            sizes=[[1.6, 3.9, 1.56]],
            rotations=[0, 1.57],
            custom_values=[],
            reshape_out=False,
        ),
        assigner_per_size=False,
        assign_per_class=False,
        diff_rad_by_sin=True,
        dir_offset=0,
        dir_limit_offset=1,
        bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder"),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type="CrossEntropyLoss", loss_weight=0.2),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.diff_rad_by_sin = diff_rad_by_sin
        self.use_direction_classifier = use_direction_classifier
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner_per_size = assigner_per_size
        self.assign_per_class = assign_per_class
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.fp16_enabled = False

        self.anchor_generator = build_prior_generator(anchor_generator)
        self.num_anchors = self.anchor_generator.num_base_anchors if self.anchor_generator else 1
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size if hasattr(self.bbox_coder, "code_size") else 7

        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        self.sampling = loss_cls["type"] not in ["FocalLoss", "GHMC"]
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls) if isinstance(loss_cls, dict) else loss_cls
        self.loss_bbox = build_loss(loss_bbox) if isinstance(loss_bbox, dict) else loss_bbox
        self.loss_dir = build_loss(loss_dir) if isinstance(loss_dir, dict) else loss_dir
        self.fp16_enabled = False

    def forward_single(self, x):
        """Forward function on a single-scale feature map."""

    def forward(self, feats):
        """Forward pass."""
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, input_metas, device="cuda"):
        """Get anchors according to feature map sizes."""
        return []

    def loss(self, *args, **kwargs):
        """Calculate loss."""

    def get_bboxes(self, *args, **kwargs):
        """Get bboxes."""


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/models/dense_heads/free_anchor3d_head.py
@HEADS.register_module()
class FreeAnchor3DHead(Anchor3DHead):
    """FreeAnchor head for 3D detection."""

    def __init__(self, pre_anchor_topk=50, bbox_thr=0.6, gamma=2.0, alpha=0.5, init_cfg=None, **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.pre_anchor_topk = pre_anchor_topk
        self.bbox_thr = bbox_thr
        self.gamma = gamma
        self.alpha = alpha

    @force_fp32(apply_to=("cls_scores", "bbox_preds", "dir_cls_preds"))
    def loss(self, cls_scores, bbox_preds, dir_cls_preds, gt_bboxes, gt_labels, input_metas, gt_bboxes_ignore=None):
        """Calculate loss of FreeAnchor head."""

    def get_bboxes(self, *args, **kwargs):
        """Get bboxes."""


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/models/detectors/mvx_two_stage.py
@DETECTORS.register_module()
class MVXTwoStageDetector(Base3DDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(
        self,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        super(MVXTwoStageDetector, self).__init__(init_cfg=init_cfg)

        if pts_voxel_layer:
            self.pts_voxel_layer = None
        if pts_voxel_encoder:
            self.pts_voxel_encoder = None
        if pts_middle_encoder:
            self.pts_middle_encoder = None
        if pts_backbone:
            self.pts_backbone = build_backbone(pts_backbone) if isinstance(pts_backbone, dict) else pts_backbone
        if pts_fusion_layer:
            self.pts_fusion_layer = None
        if pts_neck is not None:
            self.pts_neck = None
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = build_head(pts_bbox_head) if isinstance(pts_bbox_head, dict) else pts_bbox_head

        if img_backbone:
            self.img_backbone = build_backbone(img_backbone) if isinstance(img_backbone, dict) else img_backbone
        if img_neck is not None:
            self.img_neck = None
        if img_rpn_head is not None:
            self.img_rpn_head = None
        if img_roi_head is not None:
            self.img_roi_head = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get("img", None)
            pts_pretrained = pretrained.get("pts", None)
        else:
            raise ValueError(f"pretrained should be a dict, got {type(pretrained)}")

        if self.with_img_backbone:
            if img_pretrained is not None:
                warnings.warn(
                    "DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg"
                )
                if hasattr(self.img_backbone, "init_cfg"):
                    self.img_backbone.init_cfg = dict(type="Pretrained", checkpoint=img_pretrained)
        if self.with_img_roi_head:
            if img_pretrained is not None:
                warnings.warn(
                    "DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg"
                )
                if hasattr(self.img_roi_head, "init_cfg"):
                    self.img_roi_head.init_cfg = dict(type="Pretrained", checkpoint=img_pretrained)

        if self.with_pts_backbone:
            if pts_pretrained is not None:
                warnings.warn(
                    "DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg"
                )
                if hasattr(self.pts_backbone, "init_cfg"):
                    self.pts_backbone.init_cfg = dict(type="Pretrained", checkpoint=pts_pretrained)

    @property
    def with_img_shared_head(self):
        """bool: Whether the detector has a shared head in image branch."""
        return hasattr(self, "img_shared_head") and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self, "pts_bbox_head") and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return hasattr(self, "img_bbox_head") and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, "img_backbone") and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, "pts_backbone") and self.pts_backbone is not None

    @property
    def with_fusion(self):
        """bool: Whether the detector has a fusion layer."""
        return hasattr(self, "pts_fusion_layer") and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, "img_neck") and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, "pts_neck") and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, "img_rpn_head") and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, "img_roi_head") and self.img_roi_head is not None

    @property
    def with_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(self, "voxel_encoder") and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        """bool: Whether the detector has a middle encoder."""
        return hasattr(self, "middle_encoder") and self.middle_encoder is not None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        return None

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
    ):
        """Forward training function."""
        img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals,
            )
            losses.update(losses_img)
        return losses

    def forward_pts_train(self, pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore=None):
        """Forward function for point cloud branch."""
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_img_train(self, x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, proposals=None, **kwargs):
        """Forward function for image branch."""
        losses = dict()
        return losses

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict["pts_bbox"] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = []
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict["img_bbox"] = img_bbox
        return bbox_list

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats = multi_apply(self.extract_feat, points, imgs, img_metas)
        return img_feats, pts_feats

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton."""
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            bbox_list = self.pts_bbox_head.get_bboxes(*outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels) for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas, self.pts_bbox_head.test_cfg)
        return merged_bboxes


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/core/bbox/structures/box_3d_mode.py
class Box3DMode:
    LIDAR = 0
    DEPTH = 1
    CAMERA = 2
    CAM = 2  # Alias for CAMERA
    LIDAR_BOX = 0
    DEPTH_BOX = 1
    CAMERA_BOX = 2

    @staticmethod
    def convert(boxes, src, dst, rt_mat=None):
        if src == dst:
            return boxes
        if isinstance(boxes, torch.Tensor):
            return Box3DMode._convert_tensor(boxes, src, dst, rt_mat)
        elif isinstance(boxes, np.ndarray):
            return Box3DMode._convert_numpy(boxes, src, dst, rt_mat)
        else:
            raise NotImplementedError

    @staticmethod
    def _convert_tensor(boxes, src, dst, rt_mat=None):
        if src == Box3DMode.LIDAR and dst == Box3DMode.DEPTH:
            return boxes.clone()
        elif src == Box3DMode.DEPTH and dst == Box3DMode.LIDAR:
            return boxes.clone()
        else:
            return boxes.clone()

    @staticmethod
    def _convert_numpy(boxes, src, dst, rt_mat=None):
        if src == Box3DMode.LIDAR and dst == Box3DMode.DEPTH:
            return boxes.copy()
        elif src == Box3DMode.DEPTH and dst == Box3DMode.LIDAR:
            return boxes.copy()
        else:
            return boxes.copy()


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/core/bbox/structures/coord_3d_mode.py
class Coord3DMode:
    LIDAR = 0
    DEPTH = 1
    CAMERA = 2

    @staticmethod
    def convert_point(points, src, dst, rt_mat=None):
        if src == dst:
            return points
        if isinstance(points, torch.Tensor):
            return Coord3DMode._convert_tensor(points, src, dst, rt_mat)
        elif isinstance(points, np.ndarray):
            return Coord3DMode._convert_numpy(points, src, dst, rt_mat)
        else:
            raise NotImplementedError

    @staticmethod
    def _convert_tensor(points, src, dst, rt_mat=None):
        if src == Coord3DMode.LIDAR and dst == Coord3DMode.DEPTH:
            return points.clone()
        elif src == Coord3DMode.DEPTH and dst == Coord3DMode.LIDAR:
            return points.clone()
        else:
            return points.clone()

    @staticmethod
    def _convert_numpy(points, src, dst, rt_mat=None):
        if src == Coord3DMode.LIDAR and dst == Coord3DMode.DEPTH:
            return points.copy()
        elif src == Coord3DMode.DEPTH and dst == Coord3DMode.LIDAR:
            return points.copy()
        else:
            return points.copy()


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/core/bbox/structures/utils.py
def get_box_type(box_type):
    """Get the type and mode of box structure.

    Args:
        box_type (str): The type of box structure.
            The valid value are "LiDAR", "Camera", or "Depth".

    Returns:
        tuple: Box type and box mode.
    """
    box_type_lower = box_type.lower()
    if box_type_lower == "lidar":
        box_type_3d = LiDARInstance3DBoxes
        box_mode_3d = Box3DMode.LIDAR
    elif box_type_lower == "camera":
        # CameraInstance3DBoxes is not implemented, use LiDARInstance3DBoxes as fallback
        box_type_3d = LiDARInstance3DBoxes
        box_mode_3d = Box3DMode.CAM
    elif box_type_lower == "depth":
        # DepthInstance3DBoxes is not implemented, use LiDARInstance3DBoxes as fallback
        box_type_3d = LiDARInstance3DBoxes
        box_mode_3d = Box3DMode.DEPTH
    else:
        raise ValueError('Only "box_type" of "camera", "lidar", "depth"' f" are supported, got {box_type}")

    return box_type_3d, box_mode_3d


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/core/bbox/structures/lidar_box3d.py
class LiDARInstance3DBoxes:
    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((0, box_dim)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()
        self.tensor = tensor.clone()
        self.box_dim = box_dim
        self.with_yaw = with_yaw

    @property
    def gravity_center(self):
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def bottom_center(self):
        return self.tensor[:, :3]

    @property
    def dims(self):
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        return self.tensor[:, 6]

    def convert_to(self, dst, rt_mat=None):
        if dst == Box3DMode.LIDAR:
            return self
        else:
            return self

    def __getitem__(self, item):
        return type(self)(self.tensor[item], box_dim=self.box_dim, with_yaw=self.with_yaw)


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/core/visualizer/show_result.py
def show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name, show=False):
    pass


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/datasets/pipelines/compose.py
class Compose:
    def __init__(self, transforms):
        assert isinstance(transforms, collections_abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError("transform must be callable or a dict")

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data


# ============================================================================
# MMDET PIPELINES - https://github.com/open-mmlab/mmdetection/tree/v2.14.0/mmdet/datasets/pipelines
# ============================================================================


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/datasets/pipelines/formating.py
def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


# ============================================================================
# MMDET3D PIPELINES - https://github.com/open-mmlab/mmdetection3d/tree/v0.17.1/mmdet3d/datasets/pipelines
# ============================================================================


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/datasets/pipelines/formating.py
@PIPELINES.register_module()
class DefaultFormatBundle3D(object):
    # DataContainer is defined earlier in this file
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names=None, with_label=True):
        self.class_names = class_names
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if "img" in results:
            if isinstance(results["img"], list):
                imgs = [img.transpose(2, 0, 1) for img in results["img"]]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results["img"] = DataContainer(to_tensor(imgs), stack=True)
            else:
                img = results["img"].transpose(2, 0, 1)
                results["img"] = DataContainer(to_tensor(img), stack=True)

        for key in ["proposals", "gt_bboxes", "gt_bboxes_ignore", "gt_labels"]:
            if key not in results:
                continue
            if isinstance(results[key], list) and len(results[key]) > 0:
                if isinstance(results[key][0], torch.Tensor):
                    results[key] = DataContainer([to_tensor(bbox) for bbox in results[key]])
                else:
                    results[key] = DataContainer(to_tensor(results[key]))
            else:
                results[key] = DataContainer(to_tensor(results[key]))

        if "gt_bboxes_3d" in results:
            if isinstance(results["gt_bboxes_3d"], list):
                results["gt_bboxes_3d"] = DataContainer([bbox for bbox in results["gt_bboxes_3d"]], cpu_only=True)
            else:
                results["gt_bboxes_3d"] = DataContainer(results["gt_bboxes_3d"], cpu_only=True)

        if "gt_labels_3d" in results and self.with_label:
            if isinstance(results["gt_labels_3d"], list):
                results["gt_labels_3d"] = DataContainer([to_tensor(label) for label in results["gt_labels_3d"]])
            else:
                results["gt_labels_3d"] = DataContainer(to_tensor(results["gt_labels_3d"]))

        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(class_names={self.class_names}, with_label={self.with_label})"


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/datasets/pipelines/test_time_aug.py
@PIPELINES.register_module()
class MultiScaleFlipAug3D(object):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        img_scale (tuple | list[tuple]): Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            scaling.
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip direction. Options
            are 'horizontal' and 'vertical'. Default: 'horizontal'.
        transforms (list[dict]): Transforms to apply in each augmentation.
    """

    def __init__(self, img_scale=None, pts_scale_ratio=1, flip=False, flip_direction="horizontal", transforms=None):
        self.img_scale = img_scale if isinstance(img_scale, list) else [img_scale]
        self.pts_scale_ratio = pts_scale_ratio if isinstance(pts_scale_ratio, list) else [pts_scale_ratio]
        assert isinstance(self.pts_scale_ratio, list)
        assert isinstance(self.img_scale, list)
        self.flip = flip
        self.flip_direction = flip_direction if isinstance(flip_direction, list) else [flip_direction]
        self.transforms = Compose(transforms) if transforms else None

    def __call__(self, results):
        """Call function to apply test time augmentations on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: Results after applying test time augmentations,
                'img_scale', 'flip', and 'flip_direction' keys are added
                into result dict.
        """
        aug_data = []
        if self.img_scale is None or self.img_scale[0] is None:
            img_scale = [None]
        else:
            img_scale = self.img_scale
        for img_scale_item, pts_scale_ratio_item in zip(img_scale, self.pts_scale_ratio):
            for flip_item in [False, True] if self.flip else [False]:
                for flip_direction_item in self.flip_direction:
                    _results = results.copy()
                    _results["img_scale"] = img_scale_item
                    _results["flip"] = flip_item
                    _results["flip_direction"] = flip_direction_item
                    _results["pts_scale_ratio"] = pts_scale_ratio_item
                    data = self.transforms(_results) if self.transforms else _results
                    aug_data.append(data)
        results["aug_data"] = aug_data
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(img_scale={self.img_scale}, "
        repr_str += f"pts_scale_ratio={self.pts_scale_ratio}, "
        repr_str += f"flip={self.flip}, "
        repr_str += f"flip_direction={self.flip_direction}, "
        repr_str += f"transforms={self.transforms})"
        return repr_str


# ============================================================================
# MMDET3D DATASETS - https://github.com/open-mmlab/mmdetection3d/tree/v0.17.1/mmdet3d/datasets
# ============================================================================


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/datasets/custom_3d.py
class Custom3DDataset:
    CLASSES = None

    def __init__(
        self,
        data_root=None,
        ann_file=None,
        pipeline=None,
        classes=None,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
    ):
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.CLASSES = self.get_classes(classes)
        self.data_infos = self.load_annotations(self.ann_file) if self.ann_file else []
        self.pipeline = Compose(pipeline) if pipeline else None

    def get_classes(self, classes=None):
        if classes is not None:
            return classes
        return self.CLASSES

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file) if ann_file else []

    def get_data_info(self, index):
        raise NotImplementedError

    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d

    def prepare_train_data(self, index):
        raise NotImplementedError

    def prepare_test_data(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        if self.test_mode:
            return self.prepare_test_data(index)
        else:
            return self.prepare_train_data(index)

    def __len__(self):
        return len(self.data_infos)


# https://github.com/open-mmlab/mmdetection3d/blob/v0.17.1/mmdet3d/datasets/nuscenes_dataset.py
@DATASETS.register_module()
class NuScenesDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    NameMapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    AttrMapping = {
        "cycle.with_rider": 0,
        "cycle.without_rider": 1,
        "pedestrian.moving": 2,
        "pedestrian.standing": 3,
        "pedestrian.sitting_lying_down": 4,
        "vehicle.moving": 5,
        "vehicle.parked": 6,
        "vehicle.stopped": 7,
    }
    AttrMapping_rev = [
        "cycle.with_rider",
        "cycle.without_rider",
        "pedestrian.moving",
        "pedestrian.standing",
        "pedestrian.sitting_lying_down",
        "vehicle.moving",
        "vehicle.parked",
        "vehicle.stopped",
    ]
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
    ):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        try:
            from nuscenes.eval.detection.config import config_factory

            self.eval_detection_configs = config_factory(self.eval_version)
        except ImportError:
            self.eval_detection_configs = None
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def get_cat_ids(self, idx):
        """Get category distribution of single scene."""
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        if hasattr(self, "cat2id"):
            for name in gt_names:
                if name in self.CLASSES:
                    cat_ids.append(self.cat2id[name])
        else:
            for name in gt_names:
                if name in self.CLASSES:
                    cat_ids.append(self.CLASSES.index(name))
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file."""
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index."""
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
        )

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])
                lidar2img_rts.append(cam_info["lidar2img"])

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                )
            )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index."""
        info = self.data_infos[index]
        if "gt_boxes" in info:
            gt_boxes = info["gt_boxes"]
            gt_names = info["gt_names"]
            gt_labels = []
            for cat in gt_names:
                if cat in self.CLASSES:
                    gt_labels.append(self.CLASSES.index(cat))
                else:
                    gt_labels.append(-1)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            annos = dict(
                gt_boxes=gt_boxes,
                gt_names=gt_names,
                gt_labels=gt_labels,
            )
        else:
            annos = dict()
        return annos

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend="disk"),
            ),
            dict(type="LoadPointsFromMultiSweeps", sweeps_num=10, file_client_args=dict(backend="disk")),
            dict(type="DefaultFormatBundle3D", class_names=self.CLASSES, with_label=False),
            dict(type="Collect3D", keys=["points"]),
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization."""
        assert out_dir is not None, "Expect out_dir, got none."
        pipeline = self._get_pipeline(pipeline) if pipeline else self._build_default_pipeline()
        for i, result in enumerate(results):
            if "pts_bbox" in result.keys():
                result = result["pts_bbox"]
            data_info = self.data_infos[i]
            pts_path = data_info["lidar_path"]
            file_name = osp.split(pts_path)[-1].split(".")[0]
            show_result(None, None, None, out_dir, file_name, show)

    def _get_pipeline(self, pipeline):
        return Compose(pipeline) if pipeline else self.pipeline

    def _extract_data(self, index, pipeline, key):
        data_info = self.get_data_info(index)
        data = {key: None}
        return torch.tensor([])

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation)."""
        result_files, tmp_dir = self._format_bbox(results, jsonfile_prefix)
        return result_files, tmp_dir

    def _format_bbox(self, results, jsonfile_prefix):
        """Convert the results to the standard format."""
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        result_files = dict()
        for name in results[0]:
            results_ = [out[name] for out in results]
            tmp_file_ = osp.join(jsonfile_prefix, name)
            result_files.update({name: tmp_file_})
        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """Evaluation in nuScenes protocol."""
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                ret_dict = self._evaluate_single(result_files[name], logger=logger, metric=metric, result_name=name)
                results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(
                result_files, logger=logger, metric=metric, result_name=result_names[0] if result_names else "pts_bbox"
            )

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

    def _evaluate_single(self, result_path, logger=None, metric="bbox", result_name="pts_bbox"):
        """Evaluation for a single model in nuScenes protocol."""
        from nuscenes import NuScenes

        self.nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=True)
        output_dir = osp.join(*osp.split(result_path)[:-1])
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        try:
            from models.experimental.BEVFormerV2.projects.mmdet3d_plugin.datasets.nuscnes_eval import (
                NuScenesEval_custom,
            )
        except ImportError:
            from nuscenes.eval.detection.evaluate import NuScenesEval as NuScenesEval_custom
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
        detail = dict()
        metric_prefix = f"{result_name}_NuScenes"
        for name in self.CLASSES:
            for k, v in metrics["label_aps"][name].items():
                val = float("{:.4f}".format(v))
                detail["{}/{}_AP_dist_{}".format(metric_prefix, name, k)] = val
            for k, v in metrics["label_tp_errors"][name].items():
                val = float("{:.4f}".format(v))
                detail["{}/{}_{}".format(metric_prefix, name, k)] = val
        for k, v in metrics["tp_errors"].items():
            val = float("{:.4f}".format(v))
            detail["{}/{}".format(metric_prefix, self.ErrNameMapping[k])] = val
        detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
        detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]
        return detail


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/datasets/samplers/group_sampler.py
# from __future__ import division


class GroupSampler(Sampler):
    """Sampler that groups samples together.

    Source: https://raw.githubusercontent.com/open-mmlab/mmdetection/v2.14.0/mmdet/datasets/samplers/group_sampler.py
    """

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, "flag")
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(math.ceil(size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(math.ceil(size / self.samples_per_gpu)) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, np.random.choice(indice, num_extra)])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu : (i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


# ============================================================================
# MMDET CORE - https://github.com/open-mmlab/mmdetection/tree/v2.14.0/mmdet/core
# ============================================================================


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/core/utils/misc.py
def reduce_mean(tensor):
    """Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/cnn/bricks/linear.py
# Linear is just an alias for torch.nn.Linear in mmcv v1.4.0
Linear = nn.Linear


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/utils/builder.py
def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/core/bbox/builder.py
BBOX_ASSIGNERS = Registry("bbox_assigner")
BBOX_SAMPLERS = Registry("bbox_sampler")
BBOX_CODERS = Registry("bbox_coder")


def build_assigner(cfg, **default_args):
    """Builder of box assigner.

    Source: https://raw.githubusercontent.com/open-mmlab/mmdetection/v2.14.0/mmdet/core/bbox/builder.py
    """
    return build_from_cfg(cfg, BBOX_ASSIGNERS, default_args)


def build_sampler(cfg, **default_args):
    """Builder of box sampler.

    Source: https://raw.githubusercontent.com/open-mmlab/mmdetection/v2.14.0/mmdet/core/bbox/builder.py
    """
    return build_from_cfg(cfg, BBOX_SAMPLERS, default_args)


def build_bbox_coder(cfg, **default_args):
    """Builder of box coder.

    Source: https://raw.githubusercontent.com/open-mmlab/mmdetection/v2.14.0/mmdet/core/bbox/builder.py
    """
    return build_from_cfg(cfg, BBOX_CODERS, default_args)


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/utils/transformer.py
def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse function of sigmoid, has same
            shape with input.

    Source: https://raw.githubusercontent.com/open-mmlab/mmdetection/v2.14.0/mmdet/models/utils/transformer.py
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/core/bbox/match_costs/builder.py
MATCH_COST = Registry("Match Cost")


def build_match_cost(cfg, default_args=None):
    """Builder of IoU calculator.

    Source: https://raw.githubusercontent.com/open-mmlab/mmdetection/v2.14.0/mmdet/core/bbox/match_costs/builder.py
    """
    return build_from_cfg(cfg, MATCH_COST, default_args)


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/core/bbox/assigners/base_assigner.py
from abc import abstractmethod


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes.

    Source: https://raw.githubusercontent.com/open-mmlab/mmdetection/v2.14.0/mmdet/core/bbox/assigners/base_assigner.py
    """

    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign boxes to either a ground truth boxes or a negative boxes."""


# Minimal NiceRepr mixin for AssignResult
# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/utils/util_mixins.py
class NiceRepr:
    """Mixin class for creating nice representation of objects using __nice__ method.

    Source: Simplified version based on mmdet.utils.util_mixins
    """

    def __repr__(self):
        nice = self.__nice__()
        classname = self.__class__.__name__
        return f"<{classname}({nice})>"

    def __nice__(self):
        return ""


# https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/core/bbox/assigners/assign_result.py
class AssignResult(NiceRepr):
    """Stores assignments between predicted and truth boxes.

    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment
        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.
        max_overlaps (FloatTensor): the iou between the predicted box and its
            assigned truth box.
        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.

    Source: https://raw.githubusercontent.com/open-mmlab/mmdetection/v2.14.0/mmdet/core/bbox/assigners/assign_result.py
    """

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            "num_gts": self.num_gts,
            "num_preds": self.num_preds,
            "gt_inds": self.gt_inds,
            "max_overlaps": self.max_overlaps,
            "labels": self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    def __nice__(self):
        """str: a "nice" summary string describing this assign result"""
        parts = []
        parts.append(f"num_gts={self.num_gts!r}")
        if self.gt_inds is not None:
            parts.append(f"gt_inds.shape={tuple(self.gt_inds.shape)!r}")
        if self.max_overlaps is not None:
            parts.append(f"max_overlaps.shape={tuple(self.max_overlaps.shape)!r}")
        if self.labels is not None:
            parts.append(f"labels.shape={tuple(self.labels.shape)!r}")
        return ", ".join(parts)


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/parallel/_functions.py
import torch
from torch.nn.parallel._functions import _get_stream


def scatter(input, devices, streams=None):
    """Scatters tensor across multiple GPUs."""
    if streams is None:
        streams = [None] * len(devices)

    if isinstance(input, list):
        chunk_size = (len(input) - 1) // len(devices) + 1
        outputs = [scatter(input[i], [devices[i // chunk_size]], [streams[i // chunk_size]]) for i in range(len(input))]
        return outputs
    elif isinstance(input, torch.Tensor):
        output = input.contiguous()
        # TODO: copy to a pinned buffer first (if copying from CPU)
        stream = streams[0] if output.numel() > 0 else None
        if devices != [-1]:
            with torch.cuda.device(devices[0]), torch.cuda.stream(stream):
                output = output.cuda(devices[0], non_blocking=True)
        else:
            # unsqueeze the first dimension thus the tensor's shape is the
            # same as those scattered with GPU.
            output = output.unsqueeze(0)
        return output
    else:
        raise Exception(f"Unknown type {type(input)}.")


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/parallel/data_container.py
# Copyright (c) OpenMMLab. All rights reserved.
import functools

import torch


def assert_tensor_type(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f"{args[0].__class__.__name__} has no attribute " f"{func.__name__} for type {args[0].datatype}"
            )
        return func(*args, **kwargs)

    return wrapper


class DataContainer:
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self, data, stack=False, padding_value=0, cpu_only=False, pad_dims=2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.data)})"

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @property
    def pad_dims(self):
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/parallel/_functions.py

# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn.parallel._functions import _get_stream


def scatter(input, devices, streams=None):
    """Scatters tensor across multiple GPUs."""
    if streams is None:
        streams = [None] * len(devices)

    if isinstance(input, list):
        chunk_size = (len(input) - 1) // len(devices) + 1
        outputs = [scatter(input[i], [devices[i // chunk_size]], [streams[i // chunk_size]]) for i in range(len(input))]
        return outputs
    elif isinstance(input, torch.Tensor):
        output = input.contiguous()
        # TODO: copy to a pinned buffer first (if copying from CPU)
        stream = streams[0] if output.numel() > 0 else None
        if devices != [-1]:
            with torch.cuda.device(devices[0]), torch.cuda.stream(stream):
                output = output.cuda(devices[0], non_blocking=True)
        else:
            # unsqueeze the first dimension thus the tensor's shape is the
            # same as those scattered with GPU.
            output = output.unsqueeze(0)
        return output
    else:
        raise Exception(f"Unknown type {type(input)}.")


def synchronize_stream(output, devices, streams):
    if isinstance(output, list):
        chunk_size = len(output) // len(devices)
        for i in range(len(devices)):
            for j in range(chunk_size):
                synchronize_stream(output[i * chunk_size + j], [devices[i]], [streams[i]])
    elif isinstance(output, torch.Tensor):
        if output.numel() != 0:
            with torch.cuda.device(devices[0]):
                main_stream = torch.cuda.current_stream()
                main_stream.wait_stream(streams[0])
                output.record_stream(main_stream)
    else:
        raise Exception(f"Unknown type {type(output)}.")


def get_input_device(input):
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, torch.Tensor):
        return input.get_device() if input.is_cuda else -1
    else:
        raise Exception(f"Unknown type {type(input)}.")


class Scatter:
    @staticmethod
    def forward(target_gpus, input):
        input_device = get_input_device(input)
        streams = None
        if input_device == -1 and target_gpus != [-1]:
            # Perform CPU to GPU copies in a background stream
            streams = [_get_stream(device) for device in target_gpus]

        outputs = scatter(input, target_gpus)
        # Synchronize with the copy stream
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs)


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/parallel/scatter_gather.py
# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn.parallel._functions import Scatter as OrigScatter

# from ._functions import Scatter
# from .data_container import DataContainer


def scatter(inputs, target_gpus, dim=0):
    """Scatter inputs to target gpus.

    The only difference from original :func:`scatter` is to add support for
    :type:`~mmcv.parallel.DataContainer`.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            if target_gpus != [-1]:
                return OrigScatter.apply(target_gpus, None, dim, obj)
            else:
                # for CPU inference, just return the tensor wrapped in a list
                return [obj]
        if isinstance(obj, DataContainer):
            if obj.cpu_only:
                return obj.data
            else:
                return Scatter.forward(target_gpus, obj.data)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """Scatter with support for kwargs dictionary."""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


# Register ResNet with BACKBONES at the end of the file
# This ensures BACKBONES is defined and ResNet class is fully defined
BACKBONES.register_module(name="ResNet", module=ResNet)


# https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/wrappers.py
class Conv2d(nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    Supports norm and activation parameters like detectron2's Conv2d.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                if self.norm is not None:
                    assert not isinstance(self.norm, nn.SyncBatchNorm), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list.
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


# https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/boxes.py
from enum import IntEnum, unique
from typing import Union, List, Tuple

_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]


@unique
class BoxMode(IntEnum):
    """
    Enum of different ways to represent a box.
    """

    XYXY_ABS = 0
    """
    (x0, y0, x1, y1) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    """
    XYWH_ABS = 1
    """
    (x0, y0, w, h) in absolute floating points coordinates.
    """
    XYXY_REL = 2
    """
    Not yet supported!
    (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
    """
    XYWH_REL = 3
    """
    Not yet supported!
    (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
    """
    XYWHA_ABS = 4
    """
    (xc, yc, w, h, a) in absolute floating points coordinates.
    (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
    """

    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) == 4 or len(box) == 5, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor," " where k == 4 or 5"
            )
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            else:
                arr = box.clone()

        assert to_mode not in [BoxMode.XYXY_REL, BoxMode.XYWH_REL] and from_mode not in [
            BoxMode.XYXY_REL,
            BoxMode.XYWH_REL,
        ], "Relative mode not yet supported!"

        if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
            assert arr.shape[-1] == 5, "The last dimension of input shape must be 5 for XYWHA format"
            original_dtype = arr.dtype
            arr = arr.double()

            w = arr[:, 2]
            h = arr[:, 3]
            a = arr[:, 4]
            c = torch.abs(torch.cos(a * math.pi / 180.0))
            s = torch.abs(torch.sin(a * math.pi / 180.0))
            # This basically computes the horizontal bounding rectangle of the rotated box
            new_w = c * w + s * h
            new_h = c * h + s * w

            # convert center to top-left corner
            arr[:, 0] -= new_w / 2.0
            arr[:, 1] -= new_h / 2.0
            # bottom-right corner
            arr[:, 2] = arr[:, 0] + new_w
            arr[:, 3] = arr[:, 1] + new_h

            arr = arr[:, :4].to(dtype=original_dtype)
        elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
            original_dtype = arr.dtype
            arr = arr.double()
            arr[:, 0] += arr[:, 2] / 2.0
            arr[:, 1] += arr[:, 3] / 2.0
            angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
            arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
        else:
            if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
                arr[:, 2] += arr[:, 0]
                arr[:, 3] += arr[:, 1]
            elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
                arr[:, 2] -= arr[:, 0]
                arr[:, 3] -= arr[:, 1]
            else:
                raise NotImplementedError(
                    "Conversion from BoxMode {} to {} is not supported yet".format(from_mode, to_mode)
                )

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr


# https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/env.py
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
"""
PyTorch version as a tuple of 2 ints. Useful for comparison.
"""


# https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


# https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/shape_spec.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class ShapeSpec:
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    """

    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None


# https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/instances.py
import itertools
from typing import Any, Dict, List, Tuple, Union


class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        with warnings.catch_warnings(record=True):
            data_len = len(value)
        if len(self._fields):
            assert len(self) == data_len, "Adding a field of length {} to a Instances of length {}".format(
                data_len, len(self)
            )
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) is int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        if not isinstance(image_size, torch.Tensor):  # could be a tensor in tracing
            for i in instance_lists[1:]:
                assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__


# https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/masks.py
class BitMasks:
    """
    This class stores the segmentation masks for all objects in one image, in
    the form of bitmaps.

    Attributes:
        tensor: bool Tensor of N,H,W, representing N instances in the image.
    """

    def __init__(self, tensor):
        """
        Args:
            tensor: bool Tensor of N,H,W, representing N instances in the image.
        """
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to(torch.bool)
        else:
            tensor = torch.as_tensor(tensor, dtype=torch.bool, device=torch.device("cpu"))
        assert tensor.dim() == 3, tensor.size()
        self.image_size = tensor.shape[1:]
        self.tensor = tensor

    def to(self, *args, **kwargs):
        return BitMasks(self.tensor.to(*args, **kwargs))

    @property
    def device(self):
        return self.tensor.device


class ROIMasks:
    """
    Represent masks by N smaller masks defined in some ROIs. Once ROI boxes are given,
    full-image bitmask can be obtained by "pasting" the mask on the region defined
    by the corresponding ROI box.
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor: (N, M, M) mask tensor that defines the mask within each ROI.
        """
        if tensor.dim() != 3:
            raise ValueError("ROIMasks must take a masks of 3 dimension.")
        self.tensor = tensor

    def to(self, device: torch.device) -> "ROIMasks":
        return ROIMasks(self.tensor.to(device))

    @property
    def device(self):
        return self.tensor.device

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, item) -> "ROIMasks":
        """
        Returns:
            ROIMasks: Create a new :class:`ROIMasks` by indexing.
        """
        t = self.tensor[item]
        if t.dim() != 3:
            raise ValueError(f"Indexing on ROIMasks with {item} returns a tensor with shape {t.shape}!")
        return ROIMasks(t)

    def to_bitmasks(self, boxes, height, width, threshold=0.5):
        """
        Convert ROI masks to bitmasks. This is a simplified version for CPU-only.
        For full functionality, detectron2's mask_ops would be needed.
        """
        # Simplified implementation for CPU-only inference
        # Full implementation would use paste_masks_in_image from detectron2.layers.mask_ops
        # This creates a minimal bitmask by upsampling the ROI masks
        # For inference, this may not be called if masks are not used
        if isinstance(boxes, torch.Tensor):
            num_boxes = boxes.shape[0]
        else:
            num_boxes = len(boxes)

        # Create a minimal bitmask tensor (N, H, W) by upsampling ROI masks
        # This is a simplified version - full implementation would paste masks at box locations
        bitmasks = (
            F.interpolate(
                self.tensor.unsqueeze(1).float(), size=(height, width), mode="bilinear", align_corners=False
            ).squeeze(1)
            > threshold
        )
        return BitMasks(bitmasks)


# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/postprocessing.py
def detector_postprocess(results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        if isinstance(results.pred_masks, ROIMasks):
            roi_masks = results.pred_masks
        else:
            # pred_masks is a tensor of shape (N, 1, M, M)
            roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        results.pred_masks = roi_masks.to_bitmasks(
            results.pred_boxes, output_height, output_width, mask_threshold
        ).tensor  # TODO return ROIMasks/BitMask object in the future

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results


# https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/nms.py
try:
    from torchvision.ops import boxes as box_ops
except ImportError:
    box_ops = None


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float):
    """
    Same as torchvision.ops.boxes.batched_nms, but with float().
    """
    assert boxes.shape[-1] == 4
    if box_ops is None:
        raise ImportError("torchvision is required for batched_nms")
    # Fp16 does not have enough range for batched NMS, so adding float().
    return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)


# https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/boxes.py
class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32, device=torch.device("cpu"))
        else:
            tensor = tensor.to(torch.float32)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self):
        return self.tensor.device

    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor


# https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


_LOCAL_PROCESS_GROUP = None
_MISSING_LOCAL_PG_ERROR = (
    "Local process group is not yet created! Please use detectron2's `launch()` "
    "to start processes and initialize pytorch process group. If you need to start "
    "processes in other ways, please call comm.create_local_process_group("
    "num_workers_per_machine) after calling torch.distributed.init_process_group()."
)


@functools.lru_cache()
def create_local_process_group(num_workers_per_machine: int) -> None:
    """
    Create a process group that contains ranks within the same machine.

    Detectron2's launch() in engine/launch.py will call this function. If you start
    workers without launch(), you'll have to also call this. Otherwise utilities
    like `get_local_rank()` will not work.

    This function contains a barrier. All processes must call it together.

    Args:
        num_workers_per_machine: the number of worker processes per machine. Typically
          the number of GPUs.
    """
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None
    assert get_world_size() % num_workers_per_machine == 0
    num_machines = get_world_size() // num_workers_per_machine
    machine_rank = get_rank() // num_workers_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_workers_per_machine, (i + 1) * num_workers_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            _LOCAL_PROCESS_GROUP = pg


def get_local_process_group():
    """
    Returns:
        A torch process group which only includes processes that are on the same
        machine as the current process. This group can be useful for communication
        within a machine, e.g. a per-machine SyncBN.
    """
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return _LOCAL_PROCESS_GROUP


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.

    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


# https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
# Simplified get_norm - only supports basic normalization types for CPU-only inference
def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm_map = {
            "BN": nn.BatchNorm2d,
            "SyncBN": nn.SyncBatchNorm,  # Simplified - use standard SyncBatchNorm
            "FrozenBN": nn.BatchNorm2d,  # Simplified - use standard BatchNorm2d
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,
        }
        if norm not in norm_map:
            raise ValueError(f"Unknown norm type: {norm}")
        norm = norm_map[norm]
    return norm(out_channels)


# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/utils/timer.py
# Copyright (c) OpenMMLab. All rights reserved.
from time import time


class TimerError(Exception):
    def __init__(self, message):
        self.message = message
        super(TimerError, self).__init__(message)


class Timer:
    """A flexible Timer class.

    :Example:

    >>> import time
    >>> import mmcv
    >>> with mmcv.Timer():
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    1.000
    >>> with mmcv.Timer(print_tmpl='it takes {:.1f} seconds'):
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    it takes 1.0 seconds
    >>> timer = mmcv.Timer()
    >>> time.sleep(0.5)
    >>> print(timer.since_start())
    0.500
    >>> time.sleep(0.5)
    >>> print(timer.since_last_check())
    0.500
    >>> print(timer.since_start())
    1.000
    """

    def __init__(self, start=True, print_tmpl=None):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else "{:.3f}"
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        """Start the timer."""
        if not self._is_running:
            self._t_start = time()
            self._is_running = True
        self._t_last = time()

    def since_start(self):
        """Total time since the timer is started.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError("timer is not running")
        self._t_last = time()
        return self._t_last - self._t_start

    def since_last_check(self):
        """Time since the last checking.

        Either :func:`since_start` or :func:`since_last_check` is a checking
        operation.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError("timer is not running")
        dur = time() - self._t_last
        self._t_last = time()
        return dur


_g_timers = {}  # global timers


def check_time(timer_id):
    """Add check points in a single line.

    This method is suitable for running a task on a list of items. A timer will
    be registered when the method is called for the first time.

    :Example:

    >>> import time
    >>> import mmcv
    >>> for i in range(1, 6):
    >>>     # simulate a code block
    >>>     time.sleep(i)
    >>>     mmcv.check_time('task1')
    2.000
    3.000
    4.000
    5.000

    Args:
        timer_id (str): Timer identifier.
    """
    if timer_id not in _g_timers:
        _g_timers[timer_id] = Timer()
        return 0
    else:
        return _g_timers[timer_id].since_last_check()


# ============================================================================
# MMCV PROGRESS BAR - https://github.com/open-mmlab/mmcv/tree/v1.4.0/mmcv/utils/progressbar.py
# ============================================================================

# https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/utils/progressbar.py
# Copyright (c) OpenMMLab. All rights reserved.
import sys
from collections.abc import Iterable
from multiprocessing import Pool
from shutil import get_terminal_size

# from .timer import Timer


class ProgressBar:
    """A progress bar which can print the progress."""

    def __init__(self, task_num=0, bar_width=50, start=True, file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.file.write(f'[{" " * self.bar_width}] 0/{self.task_num}, ' "elapsed: 0s, ETA:")
        else:
            self.file.write("completed: 0, elapsed: 0s")
        self.file.flush()
        self.timer = Timer()

    def update(self, num_tasks=1):
        assert num_tasks > 0
        self.completed += num_tasks
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float("inf")
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = (
                f"\r[{{}}] {self.completed}/{self.task_num}, "
                f"{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, "
                f"ETA: {eta:5}s"
            )

            bar_width = min(self.bar_width, int(self.terminal_width - len(msg)) + 2, int(self.terminal_width * 0.6))
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = ">" * mark_width + " " * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(f"completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s," f" {fps:.1f} tasks/s")
        self.file.flush()


def track_progress(func, tasks, bar_width=50, file=sys.stdout, **kwargs):
    """Track the progress of tasks execution with a progress bar.

    Tasks are done with a simple for-loop.

    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        bar_width (int): Width of progress bar.

    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError('"tasks" must be an iterable object or a (iterator, int) tuple')
    prog_bar = ProgressBar(task_num, bar_width, file=file)
    results = []
    for task in tasks:
        results.append(func(task, **kwargs))
        prog_bar.update()
    prog_bar.file.write("\n")
    return results


def init_pool(process_num, initializer=None, initargs=None):
    if initializer is None:
        return Pool(process_num)
    elif initargs is None:
        return Pool(process_num, initializer)
    else:
        if not isinstance(initargs, tuple):
            raise TypeError('"initargs" must be a tuple')
        return Pool(process_num, initializer, initargs)


def track_parallel_progress(
    func,
    tasks,
    nproc,
    initializer=None,
    initargs=None,
    bar_width=50,
    chunksize=1,
    skip_first=False,
    keep_order=True,
    file=sys.stdout,
):
    """Track the progress of parallel task execution with a progress bar.

    The built-in :mod:`multiprocessing` module is used for process pools and
    tasks are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        nproc (int): Process (worker) number.
        initializer (None or callable): Refer to :class:`multiprocessing.Pool`
            for details.
        initargs (None or tuple): Refer to :class:`multiprocessing.Pool` for
            details.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
        bar_width (int): Width of progress bar.
        skip_first (bool): Whether to skip the first sample for each worker
            when estimating fps, since the initialization step may takes
            longer.
        keep_order (bool): If True, :func:`Pool.imap` is used, otherwise
            :func:`Pool.imap_unordered` is used.

    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError('"tasks" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    task_num -= nproc * chunksize * int(skip_first)
    prog_bar = ProgressBar(task_num, bar_width, start, file=file)
    results = []
    if keep_order:
        gen = pool.imap(func, tasks, chunksize)
    else:
        gen = pool.imap_unordered(func, tasks, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                prog_bar.start()
                continue
        prog_bar.update()
    prog_bar.file.write("\n")
    pool.close()
    pool.join()
    return results


def track_iter_progress(tasks, bar_width=50, file=sys.stdout):
    """Track the progress of tasks iteration or enumeration with a progress
    bar.

    Tasks are yielded with a simple for-loop.

    Args:
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        bar_width (int): Width of progress bar.

    Yields:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError('"tasks" must be an iterable object or a (iterator, int) tuple')
    prog_bar = ProgressBar(task_num, bar_width, file=file)
    for task in tasks:
        yield task
        prog_bar.update()
    prog_bar.file.write("\n")
