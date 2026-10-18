# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fold QSA boolean output casts into comparisons, preserving FP32 inputs.

The comparisons still consume the original FP32 operands and retain FP32 DEST;
only their 0/1 output is packed directly to BF16. Both values are exact in BF16.
The selected-index FLOAT32 cast remains before comparison, preserving IDs at
large contexts. BF16 boolean concat makes scalar where produce BF16 0/-inf
directly, so its separate output typecast is redundant. Neither 0 nor -inf loses
representation. Device boundary, context, and timing evidence is still required.

Importing this module does not import TTNN or access hardware.
"""

import hashlib
import inspect
import json
import textwrap
from pathlib import Path

_ORIGINAL = None
_GENERATED = None
_REPLACEMENTS = (
    (
        "ttnn.typecast(ttnn.lt(self.block_ids, complete), ttnn.bfloat16)",
        "ttnn.lt(self.block_ids, complete, dtype=ttnn.bfloat16)",
    ),
    (
        "valid = ttnn.lt(ttnn.typecast(selected, ttnn.float32), complete)",
        "valid = ttnn.lt(ttnn.typecast(selected, ttnn.float32), complete, dtype=ttnn.bfloat16)",
    ),
    (
        "tail = ttnn.lt(self.tail_offsets, tail_count)",
        "tail = ttnn.lt(self.tail_offsets, tail_count, dtype=ttnn.bfloat16)",
    ),
    (
        'mask = ttnn.typecast(ttnn.where(ttnn.concat((selected_tokens, tail), dim=3), 0.0, float("-inf")), ttnn.bfloat16)',
        'mask = ttnn.where(ttnn.concat((selected_tokens, tail), dim=3), 0.0, float("-inf"))',
    ),
)


def rewritten_source(source):
    source = textwrap.dedent(source)
    for old, new in _REPLACEMENTS:
        if source.count(old) != 1:
            raise RuntimeError(f"Expected exactly one QSA mask source match: {old!r}")
        source = source.replace(old, new, 1)
    compile(source, __file__, "exec")
    return source


def install(name="qsa_mask_output"):
    global _ORIGINAL, _GENERATED
    if name != "qsa_mask_output":
        raise ValueError(f"Unknown QSA mask candidate {name!r}")
    from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused

    if _ORIGINAL is None:
        _ORIGINAL = fused.FusedQSA._chunk_forward
    source = rewritten_source(inspect.getsource(_ORIGINAL))
    namespace = dict(_ORIGINAL.__globals__)
    exec(compile(source, __file__, "exec"), namespace)
    method = namespace["_chunk_forward"]
    method.__module__ = __name__
    method._rewritten_source = source
    fused.FusedQSA._chunk_forward = method
    _GENERATED = source
    return method


def snapshot_provenance(directory):
    if _GENERATED is None:
        raise RuntimeError("Install the candidate before recording provenance")
    root = Path(__file__).resolve().parents[4]
    model = Path(__file__).resolve().parents[1]
    relative_sources = [
        Path(__file__).resolve().relative_to(root),
        (model / "tt/fused_decoder.py").relative_to(root),
        (model / "tt/qsa.py").relative_to(root),
        Path("ttnn/cpp/ttnn/operations/eltwise/binary/binary_nanobind.cpp"),
        Path("ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp"),
        Path("ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp"),
        Path("ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_no_bcast.cpp"),
        Path("ttnn/cpp/ttnn/operations/eltwise/ternary/ternary.cpp"),
        Path("ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp"),
        Path("ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp"),
    ]
    output = Path(directory) / "qsa_mask_sources"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "candidate": "qsa_mask_output",
        "comparison_input_dtypes_unchanged": True,
        "selected_indices_cast_to_fp32_preserved": True,
        "comparison_outputs": "BF16 exact 0/1",
        "scalar_where_outputs": "BF16 exact 0/-inf",
        "sources": {},
    }
    for index, relative in enumerate(relative_sources):
        data = (root / relative).read_bytes()
        snapshot = output / f"source_{index:02d}.txt"
        snapshot.write_bytes(data)
        manifest["sources"][str(relative)] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "snapshot": snapshot.name,
        }
    generated = output / "generated_chunk_forward.py.txt"
    generated.write_text(_GENERATED)
    manifest["generated_sha256"] = hashlib.sha256(generated.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
