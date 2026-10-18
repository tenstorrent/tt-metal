# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Share same-position RoPE table lookups across Q, K and index-Q only.

Tables are invocation-local device tensors. Pooled-key block-start positions
retain their independent original lookup. No TTNN imports occur until install.
"""

import hashlib
import inspect
import json
import textwrap
from pathlib import Path

CANDIDATE_NAMES = ("qsa_shared_rope",)
_ORIGINAL = None
_SELECTED = None
_LOOKUP = """    indices = ttnn.reshape(
        ttnn.typecast(ttnn.to_layout(positions, ttnn.ROW_MAJOR_LAYOUT), ttnn.uint32), (batch, seq)
    )
    cos = ttnn.reshape(ttnn.embedding(indices, self.cos, layout=ttnn.TILE_LAYOUT), (batch, 1, seq, 64))
    sin = ttnn.reshape(ttnn.embedding(indices, self.sin, layout=ttnn.TILE_LAYOUT), (batch, 1, seq, 64))
"""


def _replace(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one shared-RoPE source target: {old!r}")
    return source.replace(old, new, 1)


def generated_sources(*, chunk, rope):
    chunk, rope = map(textwrap.dedent, (chunk, rope))
    tables = "def _shared_rope_tables(self, positions, batch, seq):\n" + _LOOKUP + "    return cos, sin\n"
    apply = _replace(rope, "def _chunk_rope(self, x, positions):", "def _shared_apply_rope(self, x, cos, sin):")
    apply = _replace(apply, _LOOKUP, "")
    chunk = _replace(
        chunk,
        "    q = self._chunk_rope(self._norm(q, self.nq), positions)",
        "    # These three consumers use exactly the same absolute positions.\n"
        "    rope_cos, rope_sin = _shared_rope_tables(self, positions, batch, seq)\n"
        "    q = _shared_apply_rope(self, self._norm(q, self.nq), rope_cos, rope_sin)",
    )
    chunk = _replace(
        chunk,
        "k = self._chunk_rope(self._norm(k, self.nk), positions)",
        "k = _shared_apply_rope(self, self._norm(k, self.nk), rope_cos, rope_sin)",
    )
    chunk = _replace(
        chunk,
        "iq = self._chunk_rope(self._norm(heads(index[..., :512], 4, 128), self.niq), positions)",
        "iq = _shared_apply_rope(self, self._norm(heads(index[..., :512], 4, 128), self.niq), rope_cos, rope_sin)",
    )
    pooled = "pooled = self._chunk_rope(self._norm(pooled, self.nik), block_start)"
    if chunk.count(pooled) != 1 or chunk.count("self._chunk_rope(") != 1:
        raise RuntimeError("Shared tables must leave exactly the pooled-key RoPE lookup independent")
    sources = {"tables": tables, "apply": apply, "chunk": chunk}
    for key, source in sources.items():
        compile(source, f"{__file__}:{key}", "exec")
    return sources


def install(name="qsa_shared_rope"):
    global _ORIGINAL, _SELECTED
    if name not in CANDIDATE_NAMES:
        raise ValueError(f"Unknown shared-RoPE candidate: {name}")
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
        raise RuntimeError("Install shared-RoPE candidate before snapshotting")
    output = Path(directory) / "qsa_shared_rope_sources"
    output.mkdir(parents=True, exist_ok=True)
    checkout = Path(__file__).resolve().parents[4]
    native = checkout / "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding"
    paths = [
        Path(__file__).resolve(),
        _SELECTED["runtime_path"],
        native / "rotary_embedding_nanobind.cpp",
        native / "device/rotary_embedding_device_operation.cpp",
        native / "device/rotary_embedding_program_factory.cpp",
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
    for key, source in _SELECTED["generated"].items():
        filename = "generated_" + key + ".py.txt"
        (output / filename).write_text(source)
        manifest["generated"][key] = {"sha256": hashlib.sha256(source.encode()).hexdigest(), "snapshot": filename}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
