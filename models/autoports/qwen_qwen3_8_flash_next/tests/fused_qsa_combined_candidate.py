# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compose shared RoPE lookup and direct BF16 mask output from original source.

Both transformations consume strings from the untouched runtime methods. No
inspect call is made on an exec-generated function. Import is host-only;
install is reserved for the serialized hardware owner.
"""

import hashlib
import inspect
import json
from pathlib import Path

_ORIGINAL = None
_SELECTED = None


def generated_sources(*, chunk, rope):
    from models.autoports.qwen_qwen3_8_flash_next.tests.fused_qsa_mask_candidate import rewritten_source
    from models.autoports.qwen_qwen3_8_flash_next.tests.fused_qsa_shared_rope_candidate import (
        generated_sources as share,
    )

    sources = share(chunk=chunk, rope=rope)
    sources["chunk"] = rewritten_source(sources["chunk"])
    for key, source in sources.items():
        compile(source, f"{__file__}:{key}", "exec")
    return sources


def install(name="qsa_adjacent_both"):
    global _ORIGINAL, _SELECTED
    if name != "qsa_adjacent_both":
        raise ValueError(f"Unknown combined QSA candidate {name!r}")
    from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused

    if _ORIGINAL is None:
        _ORIGINAL = {
            "class": fused.FusedQSA,
            "chunk": fused.FusedQSA._chunk_forward,
            "rope": fused.FusedQSA._chunk_rope,
        }
    sources = generated_sources(**{key: inspect.getsource(_ORIGINAL[key]) for key in ("chunk", "rope")})
    namespace = dict(_ORIGINAL["chunk"].__globals__)
    for key, source in sources.items():
        exec(compile(source, f"{__file__}:{key}", "exec"), namespace)
    _ORIGINAL["class"]._chunk_forward = namespace["_chunk_forward"]
    _SELECTED = {"candidate": name, "generated": sources, "runtime_path": Path(fused.__file__).resolve()}
    return _ORIGINAL["class"]


def snapshot_provenance(directory):
    if _SELECTED is None:
        raise RuntimeError("Install the combined QSA candidate before snapshotting")
    checkout = Path(__file__).resolve().parents[4]
    tests = Path(__file__).resolve().parent
    native = checkout / "ttnn/cpp/ttnn/operations"
    paths = [
        Path(__file__).resolve(),
        tests / "fused_qsa_mask_candidate.py",
        tests / "fused_qsa_shared_rope_candidate.py",
        _SELECTED["runtime_path"],
        native / "experimental/transformer/rotary_embedding/rotary_embedding_nanobind.cpp",
        native / "experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation.cpp",
        native / "experimental/transformer/rotary_embedding/device/rotary_embedding_program_factory.cpp",
        native / "eltwise/binary_ng/device/binary_ng_device_operation.cpp",
        native / "eltwise/binary_ng/device/binary_ng_program_factory.cpp",
        native / "eltwise/unary/unary.cpp",
        native / "eltwise/ternary/ternary.cpp",
    ]
    output = Path(directory) / "qsa_combined_sources"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"candidate": _SELECTED["candidate"], "sources": {}, "generated": {}}
    for index, path in enumerate(paths):
        data = path.read_bytes()
        name = f"source_{index:02d}.txt"
        (output / name).write_bytes(data)
        manifest["sources"][str(path.relative_to(checkout))] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "snapshot": name,
        }
    for key, source in _SELECTED["generated"].items():
        name = f"generated_{key}.py.txt"
        (output / name).write_text(source)
        manifest["generated"][key] = {"sha256": hashlib.sha256(source.encode()).hexdigest(), "snapshot": name}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
