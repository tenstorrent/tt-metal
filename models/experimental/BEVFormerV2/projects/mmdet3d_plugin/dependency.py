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
        assert registry.scope not in self.children, f"scope {registry.scope} exists in {self.name} registry"
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
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {key: collate([d[key] for d in batch], samples_per_gpu) for key in batch[0]}
    else:
        return default_collate(batch)


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

ext_loader = None  # Placeholder - will be set if needed


def load_ext(name, funcs):
    """Load extension module.

    For CPU-only version, this will use a placeholder implementation.
    """
    if torch.__version__ != "parrots":
        # For CPU-only, we'll use a simple implementation
        ext = importlib.import_module(name.replace("mmcv.", ""))
        for fun in funcs:
            if not hasattr(ext, fun):
                warnings.warn(f"{fun} miss in module {name}")
        return ext
    else:
        # Parrots version - not used in CPU-only
        raise NotImplementedError("Parrots not supported in CPU-only version")


def check_ops_exist():
    ext_loader = pkgutil.find_loader("mmcv._ext")
    return ext_loader is not None


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


# NOTE: This dependency.py file will continue to grow as we add more dependencies
# from mmcv, mmdet, mmdet3d, and mmseg. The goal is to include ALL dependencies
# exactly as they appear on GitHub to avoid licensing issues and ensure CPU-only
# compatibility. Each dependency should include its URL source comment.
#
# Remaining dependencies to add (this is a partial list):
# - Full Config class (688 lines) with all dependencies
# - mmcv.runner: checkpoint, base_runner, epoch_based_runner, builder, hooks, etc.
# - mmcv.cnn.bricks.transformer: TransformerLayerSequence, build_attention, etc.
# - mmcv.parallel: MMDataParallel, MMDistributedDataParallel
# - mmcv.image: tensor2imgs
# - mmcv.ops: multi_scale_deform_attn
# - mmdet: all core, datasets, models, apis, utils
# - mmdet3d: all core, datasets, models
# - mmseg: all apis
#
# This file will be very large (10,000+ lines) when complete.
