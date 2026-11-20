# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in reuse of one open mesh device and its built prefill models across tests in a process.

Enabled with ``DS_PREFILL_REUSE_MODEL=1``. The ``mesh_device`` fixture in this directory's conftest
then hands every test the same open device while its parametrized ``device_params`` stay the same,
and the chunked-transformer run functions fetch an already built model through
``acquire_transformer`` instead of loading the weights again; ``acquire_module`` does the same for any
other device module. Everything is released at session end, when ``device_params`` change, or after
any failed test, so a test that leaves the device in an unknown state is followed by a fresh open and
a fresh build. Host weights from ``cached_host`` hold no device state and stay for the whole session.
"""

import json
import os

from loguru import logger

REUSE_ENV = "DS_PREFILL_REUSE_MODEL"


def reuse_enabled() -> bool:
    return os.environ.get(REUSE_ENV) == "1"


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    payload = getattr(value, "max_packet_payload_size_bytes", None)
    if payload is not None:
        return ("fabric_router_config", payload)
    return str(value)


def _config_fingerprint(config) -> str:
    to_dict = getattr(config, "to_dict", None)
    data = to_dict() if callable(to_dict) else vars(config)
    return json.dumps(data, sort_keys=True, default=str)


class _Registry:
    def __init__(self):
        self.device = None
        self.device_gen = None
        self.device_key = None
        self.models = {}
        self.modules = {}
        self.host = {}
        self.dirty = False

    def get_device(self, request, silicon_arch_name, device_params):
        from conftest import mesh_device as root_mesh_device

        key = (_freeze(getattr(request, "param", None)), _freeze(device_params))
        if self.dirty:
            self.close_all()
        if self.device is not None and key == self.device_key:
            logger.info("reusing open mesh device")
            return self.device
        self.close_all()
        gen = root_mesh_device.__wrapped__(request, silicon_arch_name, device_params)
        self.device = next(gen)
        self.device_gen = gen
        self.device_key = key
        return self.device

    def acquire_transformer(self, transformer_cls, mesh_device, variant, config, **ctor_kwargs):
        # Switched on the built model, so it is not part of the cache key.
        tp_shard_kv = ctor_kwargs.pop("tp_shard_kv", None)
        key = (
            id(mesh_device),
            transformer_cls.__qualname__,
            getattr(variant, "name", str(variant)),
            _config_fingerprint(config),
            _freeze(ctor_kwargs),
        )
        transformer = self.models.get(key)
        if transformer is not None:
            logger.info(
                f"reusing {transformer_cls.__name__} ({ctor_kwargs.get('num_layers')} layers, "
                f"max_seq_len={ctor_kwargs.get('max_seq_len')})"
            )
        else:
            transformer = transformer_cls(mesh_device=mesh_device, config=config, **ctor_kwargs)
            self.models[key] = transformer
        transformer.set_tp_shard_kv(tp_shard_kv)
        return transformer

    def acquire_module(self, mesh_device, key, build, release):
        key = (id(mesh_device), _freeze(key))
        entry = self.modules.get(key)
        if entry is not None:
            logger.info(f"reusing {type(entry[0]).__name__}")
            return entry[0]
        module = build()
        self.modules[key] = (module, release)
        return module

    def cached_host(self, key, build):
        key = _freeze(key)
        if key not in self.host:
            self.host[key] = build()
        else:
            logger.info(f"reusing host weights {key}")
        return self.host[key]

    def release_models(self):
        for transformer in self.models.values():
            try:
                transformer.set_trace_controller(None)
                transformer.release_sub_device_managers()
            except Exception as e:
                logger.warning(f"releasing a cached transformer failed: {e}")
        self.models.clear()
        for module, release in self.modules.values():
            try:
                if release is not None:
                    release(module)
            except Exception as e:
                logger.warning(f"releasing a cached {type(module).__name__} failed: {e}")
        self.modules.clear()

    def close_all(self):
        self.dirty = False
        if self.device is None:
            self.models.clear()
            self.modules.clear()
            return
        from models.demos.deepseek_v3_d_p.tt.tt_ccl import clear_tt_ccl_cache

        self.release_models()
        clear_tt_ccl_cache()
        gen, self.device, self.device_gen, self.device_key = self.device_gen, None, None, None
        next(gen, None)


_REGISTRY = _Registry()


def get_device(request, silicon_arch_name, device_params):
    return _REGISTRY.get_device(request, silicon_arch_name, device_params)


def acquire_transformer(transformer_cls, mesh_device, variant, config, **ctor_kwargs):
    """Build the model, or return the one already built with the same arguments on this device."""
    if not reuse_enabled():
        return transformer_cls(mesh_device=mesh_device, config=config, **ctor_kwargs)
    return _REGISTRY.acquire_transformer(transformer_cls, mesh_device, variant, config, **ctor_kwargs)


def acquire_module(mesh_device, key, build, release=None):
    """``build()``, or the module an earlier test built under the same ``key`` on this device.

    ``key`` must name every argument ``build`` depends on. ``release(module)`` runs before the device
    closes. A reused module must hold no per-call state: the next test sees it as the last one left it.
    """
    if not reuse_enabled():
        return build()
    return _REGISTRY.acquire_module(mesh_device, key, build, release)


def cached_host(key, build):
    """``build()``, or the host-only result an earlier test built under the same ``key``.

    The result must not be mutated by its users, since every later test receives the same object.
    """
    if not reuse_enabled():
        return build()
    return _REGISTRY.cached_host(key, build)


def finish_transformer(transformer):
    """End-of-run cleanup. A cached model keeps its sub-device managers for the next test."""
    if reuse_enabled():
        transformer.mesh_device.clear_loaded_sub_device_manager()
    else:
        transformer.release_sub_device_managers()


def close_all():
    _REGISTRY.close_all()


def mark_dirty():
    """Drop everything at the next close_if_dirty() or get_device(), not now: fixture finalizers may still use it."""
    _REGISTRY.dirty = True


def close_if_dirty():
    if _REGISTRY.dirty:
        _REGISTRY.close_all()
