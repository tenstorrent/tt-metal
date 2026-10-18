# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated native masked-softmax alternatives for MoE routing.

The selected group paths both produce BF16 TILE [1,1,32,32] top-k values.
The existing top_mask is BF16 TILE logical [1,1,1,32], with zero for retained
experts and -inf elsewhere. The native noncausal mask broadcasts its first row.
Alternatively, moe_topk10 exposes only the selected logical columns, allowing
softmax's padding mask to replace the explicit top_mask addition; scatter reads
only those logical indices. Target self.topk is 10. Setup top_mask is untouched.
Import is host-only; installation imports TTNN transitively but opens no device.
"""

import hashlib
import inspect
import json
import textwrap
from pathlib import Path

CANDIDATE_NAMES = ("moe_mask_softmax", "moe_topk10")
_ORIGINAL = None
_SELECTED = None
_OLD = """    selected = ttnn.softmax(
        ttnn.add(values, self.top_mask), dim=-1, numeric_stable=True, compute_kernel_config=self.compute
    )"""
_NEW = """    selected = ttnn.scale_mask_softmax(
        values, scale=None, mask=self.top_mask, is_causal_mask=False,
        compute_kernel_config=self.compute, numeric_stable=True,
    )"""


def generated_source(source, name="moe_mask_softmax"):
    if name not in CANDIDATE_NAMES:
        raise ValueError(f"Unknown MoE routing candidate: {name}")
    source = textwrap.dedent(source)
    if source.count(_OLD) != 1:
        raise RuntimeError("Expected exactly one add/top_mask/softmax sequence in MoE group")
    if name == "moe_mask_softmax":
        source = source.replace(_OLD, _NEW, 1)
    else:
        topk = "values, indices = ttnn.topk(logits, k=32, dim=-1, sorted=True)"
        if source.count(topk) != 1:
            raise RuntimeError("Expected exactly one padded MoE top-k call")
        # The target config has topk=10. Rank4/last-axis topk returns logical10
        # over physical32 without a device copy. Softmax masks padded columns;
        # scatter's factory and reader use the logical index stick length.
        source = source.replace(topk, "values, indices = ttnn.topk(logits, k=self.topk, dim=-1, sorted=True)", 1)
        source = source.replace(
            _OLD,
            "    selected = ttnn.softmax(values, dim=-1, numeric_stable=True, compute_kernel_config=self.compute)",
            1,
        )
    compile(source, f"{__file__}:generated_group", "exec")
    return source


def install(name="moe_mask_softmax"):
    """Patch both selected methods and the coherent packed baseline method."""
    global _ORIGINAL, _SELECTED
    if name not in CANDIDATE_NAMES:
        raise ValueError(f"Unknown MoE mask-softmax candidate: {name}")
    from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused

    if _ORIGINAL is None:
        _ORIGINAL = [
            (fused.FusedMoE, "_group", fused.FusedMoE._group),
            (fused.FusedMoE, "_group_unified", fused.FusedMoE._group_unified),
            (fused._PackedMoE, "_group", fused._PackedMoE._group),
        ]
    generated = {}
    for cls, method_name, original in _ORIGINAL:
        source = generated_source(inspect.getsource(original), name)
        namespace = dict(original.__globals__)
        exec(compile(source, f"{__file__}:{cls.__name__}.{method_name}", "exec"), namespace)
        method = namespace[method_name]
        method._rewritten_source = source
        setattr(cls, method_name, method)
        generated[cls.__name__ + "." + method_name] = source
    _SELECTED = {"candidate": name, "runtime_path": Path(fused.__file__).resolve(), "generated": generated}
    return fused.FusedMoE


def snapshot_provenance(directory):
    """Preserve binding, validation, factory and kernel contracts with this run."""
    if _SELECTED is None:
        raise RuntimeError("Install candidate before snapshotting")
    output = Path(directory) / "moe_mask_softmax_sources"
    output.mkdir(parents=True, exist_ok=True)
    checkout = Path(__file__).resolve().parents[4]
    native = checkout / "ttnn/cpp/ttnn/operations/normalization/softmax"
    paths = [
        Path(__file__).resolve(),
        _SELECTED["runtime_path"],
        native / "softmax_nanobind.cpp",
        native / "softmax.cpp",
        native / "device/softmax_device_operation.cpp",
        native / "device/softmax_program_factory_attention_optimized.cpp",
        native / "device/kernels/attention/compute/softmax.cpp",
        native / "device/kernels/attention/dataflow/reader_unary_interleaved_sm.cpp",
        checkout / "ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp",
        checkout / "ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp",
        checkout / "ttnn/cpp/ttnn/operations/data_movement/scatter/scatter.cpp",
        checkout / "ttnn/cpp/ttnn/operations/data_movement/scatter/device/scatter_program_factory.cpp",
        checkout / "ttnn/cpp/ttnn/operations/data_movement/scatter/device/kernels/dataflow/reader_scatter.cpp",
        checkout / "ttnn/cpp/ttnn/operations/data_movement/scatter/device/kernels/common.hpp",
    ]
    manifest = {"candidate": _SELECTED["candidate"], "sources": {}, "generated": {}}
    for index, path in enumerate(paths):
        data = path.read_bytes()
        filename = f"source_{index:02d}.txt"
        (output / filename).write_bytes(data)
        manifest["sources"][str(path.relative_to(checkout))] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "snapshot": filename,
        }
    for name, source in _SELECTED["generated"].items():
        filename = "generated_" + name + ".py.txt"
        (output / filename).write_text(source)
        manifest["generated"][name] = {"sha256": hashlib.sha256(source.encode()).hexdigest(), "snapshot": filename}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
