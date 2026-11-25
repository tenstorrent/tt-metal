# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in on-disk cache for converted device weights (QWEN_WEIGHT_CACHE=<dir>).

Weight conversion (transposes, packing, BFP8 tilization) dominates the ~12-20 minute model load. During model
construction every ttnn.from_torch(..., device=mesh) call is keyed by (scope, call ordinal, shape, dtype, layout,
mapper, memory config, a content sample of the source tensor) and stored as a host tensor file; later loads read
the file and move it to the device. Any change in code order, config or checkpoint changes the key (miss -> rebuild).
"""

import contextlib
import hashlib
import os

import torch

import ttnn


def _sample_digest(t):
    flat = t.detach().reshape(-1)
    n = flat.numel()
    if n == 0:
        return "empty"
    k = min(n, 4096)
    idx = (torch.arange(k, dtype=torch.int64) * (n - 1)) // max(k - 1, 1)
    sample = flat[idx].to(torch.float32).contiguous()
    h = hashlib.sha1(sample.numpy().tobytes())
    h.update(str(float(flat.to(torch.float64).sum())).encode() if n <= (1 << 22) else b"")
    return h.hexdigest()[:16]


@contextlib.contextmanager
def weight_cache(scope):
    root = os.environ.get("QWEN_WEIGHT_CACHE")
    if not root:
        yield
        return
    os.makedirs(root, exist_ok=True)
    original = ttnn.from_torch
    counter = [0]

    def cached(tensor, dtype=None, *args, **kwargs):
        device = kwargs.get("device")
        memory_config = kwargs.get("memory_config")
        if (
            device is None
            or args
            or not isinstance(tensor, torch.Tensor)
            or tensor.numel() < (1 << 16)
            or (memory_config is not None and memory_config != ttnn.DRAM_MEMORY_CONFIG)
        ):
            return original(tensor, dtype, *args, **kwargs)
        counter[0] += 1
        mapper = kwargs.get("mesh_mapper")
        layout = kwargs.get("layout")
        key = "|".join(
            str(x)
            for x in (
                scope,
                counter[0],
                tuple(tensor.shape),
                tensor.dtype,
                dtype,
                layout,
                type(mapper).__name__ if mapper is not None else None,
                getattr(mapper, "dim", None) if mapper is not None else None,
                memory_config,
                _sample_digest(tensor),
            )
        )
        path = os.path.join(root, hashlib.sha1(key.encode()).hexdigest() + ".tensorbin")
        host_kwargs = {k: v for k, v in kwargs.items() if k not in ("device", "memory_config")}
        if mapper is None and hasattr(device, "get_num_devices"):
            host_kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
        if os.path.exists(path):
            try:
                host = ttnn.load_tensor(path)
                return ttnn.to_device(host, device, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)
            except Exception:
                os.remove(path)
        host = original(tensor, dtype, **host_kwargs)
        tmp = path[: -len(".tensorbin")] + f".tmp{os.getpid()}.tensorbin"
        ttnn.dump_tensor(tmp, host)
        os.replace(tmp, path)
        return ttnn.to_device(host, device, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)

    ttnn.from_torch = cached
    try:
        yield
    finally:
        ttnn.from_torch = original
