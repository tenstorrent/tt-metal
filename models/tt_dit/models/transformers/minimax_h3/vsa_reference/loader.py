# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Load the vendored FastVideo VSA modules without a fastvideo install.

``video_sparse_attn.py`` and ``video_sparse_attn_h3.py`` in this directory are
vendored untouched from github.com/hao-ai-lab/FastVideo (main, 2026-08-31) so
tests can cross-check our ports against genuine upstream code. Upstream
imports (``fastvideo.*``, ``fastvideo_kernel``) are satisfied here with inert
stubs -- enough for the geometry/mask/pooling functions the tests use; the
CUDA kernel entry points stay ``None`` exactly as they would without a
``fastvideo_kernel`` install.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path

_VENDOR_DIR = Path(__file__).parent


def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _install_stubs() -> None:
    if "fastvideo" in sys.modules:
        return

    for name in ("fastvideo", "fastvideo.attention", "fastvideo.attention.backends"):
        _stub_module(name)

    logger_mod = _stub_module("fastvideo.logger")

    class _Logger(logging.getLoggerClass()):
        def info_once(self, *args, **kwargs):
            self.info(*args, **kwargs)

        def warning_once(self, *args, **kwargs):
            self.warning(*args, **kwargs)

    def init_logger(name: str):
        logger = _Logger(name)
        return logger

    logger_mod.init_logger = init_logger

    distributed_mod = _stub_module("fastvideo.distributed")

    def get_sp_group():
        return types.SimpleNamespace(world_size=1)

    distributed_mod.get_sp_group = get_sp_group

    abstract_mod = _stub_module("fastvideo.attention.backends.abstract")

    @dataclass
    class AttentionMetadata:
        current_timestep: int
        VSA_sparsity: float

    class AttentionBackend:
        pass

    class AttentionImpl:
        pass

    class AttentionMetadataBuilder:
        pass

    def layer_idx_from_prefix(prefix: str, default: int = -1) -> int:
        return default

    abstract_mod.AttentionMetadata = AttentionMetadata
    abstract_mod.AttentionBackend = AttentionBackend
    abstract_mod.AttentionImpl = AttentionImpl
    abstract_mod.AttentionMetadataBuilder = AttentionMetadataBuilder
    abstract_mod.layer_idx_from_prefix = layer_idx_from_prefix

    probe_mod = _stub_module("fastvideo.attention.backends.video_sparse_attn_h3_probe")
    probe_mod.probe_enabled = lambda: None
    probe_mod.record_probe = lambda *args, **kwargs: None


def _load_vendored(module_name: str, file_name: str) -> types.ModuleType:
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _VENDOR_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_upstream() -> tuple[types.ModuleType, types.ModuleType]:
    """Return the vendored (video_sparse_attn, video_sparse_attn_h3) modules."""
    _install_stubs()
    base = _load_vendored("fastvideo.attention.backends.video_sparse_attn", "video_sparse_attn.py")
    h3 = _load_vendored("fastvideo.attention.backends.video_sparse_attn_h3", "video_sparse_attn_h3.py")
    return base, h3
