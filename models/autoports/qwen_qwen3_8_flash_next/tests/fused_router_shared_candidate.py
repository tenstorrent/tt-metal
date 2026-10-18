# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Common-LHS router/shared-expert projection control for the fused MoE.

The original BF16 router [2560,512] and shared gate/up/scale [2560,1312]
weights become one BF16 [2560,1824] setup tensor. Both 32-row expert paths
perform one projection before top-k and directly slice the router, shared gate,
shared up, and scale regions. There is no intermediate shared-region copy.
Routing, expert execution, weighted reduction, NaN masking, shared
activation/down projection, batching, and logical-tail handling stay unchanged.

Packing changes the matmul output geometry and can change numerical rounding,
especially at routing ties. Device PCC and warmed trace latency are required.
Importing this module alone performs no TTNN import or hardware operation.
"""

import hashlib
import inspect
import json
import textwrap
from pathlib import Path

_ORIGINAL_FACTORY = None
_ORIGINAL_METHODS = None
_GENERATED_METHODS = {}
_SELECTED_NAME = None


def rewritten_source(source):
    """Replace exactly the two common-LHS projection sites, on the host."""
    source = textwrap.dedent(source)
    replacements = (
        (
            "    logits = ttnn.linear(x, self.router, compute_kernel_config=self.compute)",
            "    router_shared = ttnn.linear(x, self.router_shared_packed, compute_kernel_config=self.compute)\n"
            "    logits = router_shared[..., :512]",
        ),
        (
            "    shared_packed = ttnn.linear(x, self.shared_packed, compute_kernel_config=self.compute)\n"
            "    sg, su = shared_packed[..., :640], shared_packed[..., 640:1280]\n"
            "    scale = shared_packed[..., 1280:1281]",
            "    sg, su = router_shared[..., 512:1152], router_shared[..., 1152:1792]\n"
            "    scale = router_shared[..., 1792:1793]",
        ),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"Expected exactly one router/shared projection source match: {old!r}")
        source = source.replace(old, new, 1)
    compile(source, __file__, "exec")
    return source


def _packed_factory(cls, state, **kwargs):
    import torch

    import ttnn

    obj = _ORIGINAL_FACTORY(cls, state, **kwargs)
    if (obj.hidden, obj.experts, obj.intermediate) != (2560, 512, 640):
        raise ValueError("Router/shared packing is qualified only for H2560/E512/I640")
    # Rebuild from the same checkpoint tensors as the selected factories. No
    # arithmetic is folded into the weights; only column concatenation changes.
    weight = torch.cat(
        (
            state["mlp.gate.weight"].T,
            state["mlp.shared_expert.gate_proj.weight"].T,
            state["mlp.shared_expert.up_proj.weight"].T,
            torch.nn.functional.pad(state["mlp.shared_expert_gate.weight"].T, (0, 31)),
        ),
        dim=-1,
    ).contiguous()
    if tuple(weight.shape) != (2560, 1824):
        raise ValueError(f"Unexpected packed router/shared weight shape: {tuple(weight.shape)}")
    obj.router_shared_packed = ttnn.from_torch(
        weight,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=kwargs["mesh_device"],
    )
    del obj.router, obj.shared_packed
    return obj


def install(name="packed_router_shared"):
    """Patch the existing fused class before construction; preserve type checks."""
    global _ORIGINAL_FACTORY, _ORIGINAL_METHODS, _SELECTED_NAME
    if name != "packed_router_shared":
        raise ValueError(f"Unknown router/shared projection candidate {name!r}")
    from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused

    if _ORIGINAL_FACTORY is None:
        _ORIGINAL_FACTORY = fused.FusedMoE.from_state_dict.__func__
        _ORIGINAL_METHODS = {key: getattr(fused.FusedMoE, key) for key in ("_group", "_group_unified")}
    methods = {}
    for key, original in _ORIGINAL_METHODS.items():
        source = rewritten_source(inspect.getsource(original))
        namespace = dict(original.__globals__)
        exec(compile(source, f"{__file__}:{key}", "exec"), namespace)
        method = namespace[key]
        method.__module__ = __name__
        method._rewritten_source = source
        methods[key] = method
        _GENERATED_METHODS[key] = source
    fused.FusedMoE.from_state_dict = classmethod(_packed_factory)
    for key, method in methods.items():
        setattr(fused.FusedMoE, key, method)
    _SELECTED_NAME = name
    return fused.FusedMoE


def snapshot_provenance(directory):
    if _SELECTED_NAME is None:
        raise RuntimeError("Install the candidate before recording provenance")
    root = Path(__file__).resolve().parents[1]
    output = Path(directory) / "router_shared_sources"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "candidate": _SELECTED_NAME,
        "packed_weight_shape": [2560, 1824],
        "column_regions": {
            "router": [0, 512],
            "shared_gate": [512, 1152],
            "shared_up": [1152, 1792],
            "shared_scale_padded": [1792, 1824],
        },
        "runtime_slices": {
            "router": [0, 512],
            "shared_gate": [512, 1152],
            "shared_up": [1152, 1792],
            "shared_scale": [1792, 1793],
        },
        "intermediate_shared_region_slice": False,
        "sources": {},
        "generated_methods": {},
    }
    for path in (Path(__file__).resolve(), root / "tt/fused_decoder.py", root / "tt/moe.py"):
        data = path.read_bytes()
        snapshot = output / (path.name + ".txt")
        snapshot.write_bytes(data)
        manifest["sources"][str(path.relative_to(root))] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "snapshot": snapshot.name,
        }
    for name, source in _GENERATED_METHODS.items():
        snapshot = output / ("generated" + name + ".py.txt")
        snapshot.write_text(source)
        original_source = textwrap.dedent(inspect.getsource(_ORIGINAL_METHODS[name]))
        manifest["generated_methods"][name] = {
            "original_sha256": hashlib.sha256(original_source.encode()).hexdigest(),
            "generated_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "snapshot": snapshot.name,
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
