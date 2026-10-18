# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PLE gate-chain and normalization output-cast fusion controls.

The scalar maximum overload ignores its activation arguments in this checkout,
so maximum(..., input_tensor_a_activations=[ABS], activations=[SQRT]) is invalid
as a fusion. Instead multiply's first operand executes ABS -> RELU_MIN(eps) ->
SQRT, and its second operand executes SIGN. RELU_MIN's float SFPU body computes
max(x, threshold), matching the positive epsilon clamp after ABS.

Binary preprocessing packs a completed operand chain before multiplication,
but there is no BF16 pack between the clamp and SQRT inside the chain. The
epsilon remains the original 1e-6; any resulting accuracy delta must be tested.
The separate norm-output control requests BF16 output from FP32-input multiply,
whose input dtype retains FP32 arithmetic and whose output cast runs in the same
kernel. The combined mode applies these two independent rewrites together.
Importing this module alone does not import TTNN or access a device.
"""

import hashlib
import inspect
import json
import textwrap
from pathlib import Path

CANDIDATE_NAMES = ("ple_gate_chain", "ple_norm_output", "ple_adjacent_both", "ple_dot_scale")
_ORIGINALS = None
_GENERATED = {}
_SELECTED_NAME = None
_OLD = "    gate = ttnn.multiply(ttnn.sqrt(ttnn.maximum(ttnn.abs(gate), 1e-6)), ttnn.sign(gate))"
_NEW = """    gate = ttnn.multiply(
        gate,
        gate,
        input_tensor_a_activations=[
            ttnn.UnaryOpType.ABS,
            ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MIN, 1e-6),
            ttnn.UnaryOpType.SQRT,
        ],
        input_tensor_b_activations=[ttnn.UnaryOpType.SIGN],
    )"""


def rewritten_source(source):
    source = textwrap.dedent(source)
    if source.count(_OLD) != 1:
        raise RuntimeError("PLE signed-root source changed; reassess before applying the gate chain")
    result = source.replace(_OLD, _NEW, 1)
    compile(result, __file__, "exec")
    return result


def rewritten_norm_source(source):
    source = textwrap.dedent(source)
    old = "    return ttnn.typecast(ttnn.multiply(normed, weight), ttnn.bfloat16)"
    new = "    return ttnn.multiply(normed, weight, dtype=ttnn.bfloat16)"
    if source.count(old) != 1:
        raise RuntimeError("PLE weighted norm source changed; reassess the output cast fold")
    result = source.replace(old, new, 1)
    compile(result, __file__, "exec")
    return result


def rewritten_dot_source(source):
    source = textwrap.dedent(source)
    old = (
        "    dot = ttnn.sum(ttnn.multiply(key, query), dim=-1, keepdim=True)\n"
        "    gate = ttnn.multiply(dot, 2560**-0.5)"
    )
    new = "    gate = ttnn.sum(ttnn.multiply(key, query), dim=-1, keepdim=True, scalar=2560**-0.5)"
    if source.count(old) != 1:
        raise RuntimeError("PLE dot/scale source changed; reassess the reduction scalar fold")
    result = source.replace(old, new, 1)
    compile(result, __file__, "exec")
    return result


def install(name="ple_gate_chain"):
    global _ORIGINALS, _SELECTED_NAME
    if name not in CANDIDATE_NAMES:
        raise ValueError(f"Unknown PLE gate candidate {name!r}")
    from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused

    if _ORIGINALS is None:
        _ORIGINALS = {key: getattr(fused.FusedPLE, key) for key in ("forward", "_norm")}
    _GENERATED.clear()
    for key, original in _ORIGINALS.items():
        setattr(fused.FusedPLE, key, original)
    changes = {}
    if name in ("ple_gate_chain", "ple_adjacent_both"):
        changes["forward"] = rewritten_source
    if name in ("ple_norm_output", "ple_adjacent_both"):
        changes["_norm"] = rewritten_norm_source
    if name == "ple_dot_scale":
        changes["forward"] = rewritten_dot_source
    for key, rewrite in changes.items():
        original = _ORIGINALS[key]
        source = rewrite(inspect.getsource(original))
        namespace = dict(original.__globals__)
        exec(compile(source, __file__, "exec"), namespace)
        method = namespace[key]
        method.__module__ = __name__
        method._rewritten_source = source
        setattr(fused.FusedPLE, key, method)
        _GENERATED[key] = source
    _SELECTED_NAME = name
    return fused.FusedPLE


def snapshot_provenance(directory):
    if _SELECTED_NAME is None:
        raise RuntimeError("Install the candidate before recording provenance")
    root = Path(__file__).resolve().parents[4]
    model = Path(__file__).resolve().parents[1]
    relative_sources = [
        Path(__file__).resolve().relative_to(root),
        (model / "tt/fused_decoder.py").relative_to(root),
        Path("ttnn/cpp/ttnn-nanobind/activation.cpp"),
        Path("ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp"),
        Path("ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp"),
        Path("ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp"),
        Path("ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp"),
        Path("ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp"),
        Path("ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp"),
        Path("ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp"),
        Path("ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions_nanobind.hpp"),
        Path("ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp"),
        Path("ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp"),
        Path("ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp"),
        Path("ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce.cpp"),
        Path(
            "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp"
        ),
    ]
    compute = Path("ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute")
    relative_sources.extend(
        compute / name
        for name in (
            "eltwise_binary_sfpu_no_bcast.cpp",
            "eltwise_binary_no_bcast.cpp",
            "eltwise_utils_common.hpp",
            "eltwise_utils_sfpu.hpp",
        )
    )
    sfpu = Path("tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu")
    relative_sources.extend(sfpu / f"ckernel_sfpu_{name}.h" for name in ("abs", "relu", "sqrt", "sign"))
    output = Path(directory) / "ple_gate_sources"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "candidate": _SELECTED_NAME,
        "gate_chain_enabled": _SELECTED_NAME in ("ple_gate_chain", "ple_adjacent_both"),
        "norm_output_fold_enabled": "_norm" in _GENERATED,
        "dot_scale_fold_enabled": _SELECTED_NAME == "ple_dot_scale",
        "input_a_chain": ["ABS", ["RELU_MIN", 1e-6], "SQRT"],
        "input_b_chain": ["SIGN"],
        "scalar_maximum_activation_arguments_ignored": True,
        "removed_rounding_boundary": (
            "BF16 dot-sum output before scale multiply"
            if _SELECTED_NAME == "ple_dot_scale"
            else (
                "BF16 clamp output before SQRT" if _SELECTED_NAME in ("ple_gate_chain", "ple_adjacent_both") else None
            )
        ),
        "norm_output_fold": "multiply's FP32 inputs retain FP32 DEST; BF16 output uses TYPECAST postactivation",
        "sources": {},
        "generated_methods": {},
    }
    for index, relative in enumerate(relative_sources):
        data = (root / relative).read_bytes()
        destination = output / f"source_{index:02d}.txt"
        destination.write_bytes(data)
        manifest["sources"][str(relative)] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "snapshot": destination.name,
        }
    for name, source in _GENERATED.items():
        generated = output / ("generated" + name + ".py.txt")
        generated.write_text(source)
        manifest["generated_methods"][name] = {
            "sha256": hashlib.sha256(generated.read_bytes()).hexdigest(),
            "snapshot": generated.name,
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
