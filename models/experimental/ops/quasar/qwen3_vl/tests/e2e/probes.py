# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Debug probes: run one prefill-only op on dummy tensors before a resumed decode, to find device state it leaves behind.

Used with --qwen-resume-prefill and --qwen-probe-before-decode NAME[,NAME...]; outputs are discarded.
"""
import torch

import ttnn


def _up(dev, *shape, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(torch.randn(*shape).bfloat16(), layout=layout, device=dev)


def _ckc(fidelity=ttnn.MathFidelity.HiFi4):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )


def _fill(dev, memory_config, bytes_per_core_or_total):
    """As many random bf16 [32, 1024] tensors as fit, up to the byte budget, in `memory_config`."""
    out, tile_bytes = [], 32 * 1024 * 2
    for _ in range(bytes_per_core_or_total // tile_bytes):
        try:
            out.append(
                ttnn.from_torch(
                    (torch.randn(1, 1, 32, 1024) * 100).bfloat16(),
                    layout=ttnn.TILE_LAYOUT,
                    device=dev,
                    memory_config=memory_config,
                )
            )
        except RuntimeError:  # out of memory: as full as it gets
            break
    return out


def _sdpa_prog(dev):
    g = dev.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), exp_approx_mode=False, q_chunk_size=64, k_chunk_size=64
    )


PROBES = {
    "gelu": lambda dev: ttnn.gelu(_up(dev, 1, 1, 256, 4096)),
    "linear_gelu": lambda dev: ttnn.linear(
        _up(dev, 1, 1, 256, 1024),
        _up(dev, 1024, 4096),
        bias=_up(dev, 1, 4096),
        activation="gelu",
        compute_kernel_config=_ckc(ttnn.MathFidelity.HiFi2),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ),
    "layer_norm": lambda dev: ttnn.layer_norm(
        _up(dev, 1, 1, 256, 1024),
        epsilon=1e-6,
        weight=_up(dev, 1, 32, 1024),
        bias=_up(dev, 1, 32, 1024),
        compute_kernel_config=_ckc(),
    ),
    "sdpa_prefill": lambda dev: ttnn.experimental.quasar.transformer.scaled_dot_product_attention(
        _up(dev, 1, 16, 256, 64),
        _up(dev, 1, 16, 256, 64),
        _up(dev, 1, 16, 256, 64),
        is_causal=False,
        scale=64**-0.5,
        compute_kernel_config=_ckc(),
        program_config=_sdpa_prog(dev),
    ),
    "rope_prefill": lambda dev: ttnn.experimental.rotary_embedding_llama(
        _up(dev, 1, 16, 256, 64),
        _up(dev, 1, 1, 256, 64),
        _up(dev, 1, 1, 256, 64),
        _up(dev, 1, 1, 32, 32),
        is_decode_mode=False,
    ),
    "add_bcast": lambda dev: ttnn.experimental.quasar.add(_up(dev, 1, 1, 256, 3072), _up(dev, 1, 3072)),
    # Fill free memory with random data (allocated, then freed by run()), so reads of never-written memory see garbage.
    "dirty_l1": lambda dev: _fill(dev, ttnn.L1_MEMORY_CONFIG, 3 * 1024 * 1024),
    "dirty_dram": lambda dev: _fill(dev, ttnn.DRAM_MEMORY_CONFIG, 256 * 1024 * 1024),
}


def run(names, dev):
    for name in names:
        out = PROBES[name](dev)
        ttnn.synchronize_device(dev)
        for t in out if isinstance(out, (list, tuple)) else [out]:
            ttnn.deallocate(t)


def device_tensors(root, skip=()):
    """name -> ttnn.Tensor for every device tensor reachable from `root` through attributes, lists and dicts."""
    found, seen, skip_ids = {}, set(), {id(t) for t in skip}
    stack = [("model", root)]
    while stack:
        name, obj = stack.pop()
        if id(obj) in seen or id(obj) in skip_ids:
            continue
        seen.add(id(obj))
        if isinstance(obj, ttnn.Tensor):
            if obj.storage_type() == ttnn.StorageType.DEVICE and obj.is_allocated():
                found[name] = obj
        elif isinstance(obj, dict):
            stack += [(f"{name}[{k!r}]", v) for k, v in obj.items()]
        elif isinstance(obj, (list, tuple)):
            stack += [(f"{name}[{i}]", v) for i, v in enumerate(obj)]
        elif hasattr(obj, "__dict__") and type(obj).__module__.startswith("models."):
            stack += [(f"{name}.{k}", v) for k, v in vars(obj).items()]
    return found


def checksums(tensors):
    from models.experimental.ops.quasar.qwen3_vl.tests.e2e.recorder import to_host

    return {name: (t.buffer_address(), float(to_host(t).double().abs().sum())) for name, t in tensors.items()}
