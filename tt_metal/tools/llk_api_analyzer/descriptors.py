"""Parser for the generated ``chlkc_descriptors.h`` kernel-config header.

This header sits beside the compiled ELFs (``<kernel>/<hash>/chlkc_descriptors.h``)
and contains the per-circular-buffer data formats, tile sizes and face geometry,
plus the scalar configuration (``DST_ACCUM_MODE``, ``DST_SYNC_MODE``,
``MATH_FIDELITY``) that the compute kernel was built with.

The LLK APIs read most of these as runtime values from the arrays below, so they
do not appear as constants in the DWARF; parsing the header recovers them
directly. This is intentionally a light-weight regex parser (not a C++ parser):
the file is machine-generated with a stable shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .data_formats import UNUSED_FORMAT_CODE, data_format_name

_ARRAY_RE = re.compile(
    r"constexpr\s+(?:unsigned\s+char|uint8_t|uint16_t|uint32_t|int)\s+" r"(\w+)\s*\[\d+\]\s*=\s*\{([^}]*)\}",
    re.DOTALL,
)
_SCALAR_BOOL_RE = re.compile(r"constexpr\s+bool\s+(\w+)\s*=\s*(true|false)\s*;")
_DST_SYNC_RE = re.compile(r"#define\s+DST_SYNC_MODE\s+DstSync::(\w+)")
_MATH_FIDELITY_RE = re.compile(r"MATH_FIDELITY\s*=\s*static_cast<ckernel::MathFidelity>\((\d+)\)")

_MATH_FIDELITY_NAMES = {0: "LoFi", 2: "HiFi2", 3: "HiFi3", 4: "HiFi4"}


@dataclass
class CircularBufferConfig:
    """Resolved per-CB configuration for the slots that are actually in use."""

    index: int
    data_format: str
    tile_size_bytes: int | None = None
    tile_r_dim: int | None = None
    tile_c_dim: int | None = None
    num_faces: int | None = None


@dataclass
class KernelDescriptors:
    """Compile-time kernel configuration recovered from ``chlkc_descriptors.h``."""

    dst_accum_mode: bool | None = None
    dst_sync_mode: str | None = None
    math_fidelity: str | None = None
    approx_mode: bool | None = None
    unpack_inputs: list[CircularBufferConfig] = field(default_factory=list)
    pack_outputs: list[CircularBufferConfig] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dst_accum_mode": self.dst_accum_mode,
            "dst_sync_mode": self.dst_sync_mode,
            "math_fidelity": self.math_fidelity,
            "approx_mode": self.approx_mode,
            "unpack_inputs": [vars(cb) for cb in self.unpack_inputs],
            "pack_outputs": [vars(cb) for cb in self.pack_outputs],
        }


def _parse_int_array(text: str) -> list[int]:
    return [int(tok) for tok in text.replace("\n", "").split(",") if tok.strip()]


def parse_descriptors(path: str | Path) -> KernelDescriptors:
    """Parse a ``chlkc_descriptors.h`` file into a :class:`KernelDescriptors`."""
    source = Path(path).read_text()
    arrays = {name: _parse_int_array(body) for name, body in _ARRAY_RE.findall(source)}

    descriptors = KernelDescriptors()

    bools = dict(_SCALAR_BOOL_RE.findall(source))
    if "DST_ACCUM_MODE" in bools:
        descriptors.dst_accum_mode = bools["DST_ACCUM_MODE"] == "true"
    if "APPROX" in bools:
        descriptors.approx_mode = bools["APPROX"] == "true"

    sync_match = _DST_SYNC_RE.search(source)
    if sync_match:
        descriptors.dst_sync_mode = sync_match.group(1)

    fidelity_match = _MATH_FIDELITY_RE.search(source)
    if fidelity_match:
        code = int(fidelity_match.group(1))
        descriptors.math_fidelity = _MATH_FIDELITY_NAMES.get(code, f"Fidelity({code})")

    descriptors.unpack_inputs = _collect_cbs(arrays, "unpack")
    descriptors.pack_outputs = _collect_cbs(arrays, "pack")
    return descriptors


def _collect_cbs(arrays: dict[str, list[int]], prefix: str) -> list[CircularBufferConfig]:
    """Build CB configs for every slot whose format is not the 'unused' sentinel."""
    formats = arrays.get(f"{prefix}_src_format")
    if not formats:
        return []
    tile_sizes = arrays.get(f"{prefix}_tile_size", [])
    tile_r = arrays.get(f"{prefix}_tile_r_dim", [])
    tile_c = arrays.get(f"{prefix}_tile_c_dim", [])
    num_faces = arrays.get(f"{prefix}_tile_num_faces", [])

    result: list[CircularBufferConfig] = []
    for index, code in enumerate(formats):
        if code == UNUSED_FORMAT_CODE:
            continue
        result.append(
            CircularBufferConfig(
                index=index,
                data_format=data_format_name(code),
                tile_size_bytes=tile_sizes[index] if index < len(tile_sizes) else None,
                tile_r_dim=tile_r[index] if index < len(tile_r) else None,
                tile_c_dim=tile_c[index] if index < len(tile_c) else None,
                num_faces=num_faces[index] if index < len(num_faces) else None,
            )
        )
    return result
