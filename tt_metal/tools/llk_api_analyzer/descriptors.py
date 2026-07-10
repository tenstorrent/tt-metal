# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

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

# Code marking an unconfigured / unused circular-buffer slot (tensix ``Invalid``).
UNUSED_FORMAT_CODE = 255

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


def parse_descriptors(path: str | Path, format_names: dict[int, str] | None = None) -> KernelDescriptors:
    """Parse a ``chlkc_descriptors.h`` file into a :class:`KernelDescriptors`.

    ``format_names`` maps a data-format code to its name, taken from the ELF's
    own tensix ``DataFormat`` enum, so the codes decode correctly per
    architecture. Codes not present in that enum are reported as
    ``Unknown(<code>)``.

    Raises ``FileNotFoundError`` if ``path`` does not exist and ``ValueError``
    if the file exists but is not a recognizable ``chlkc_descriptors.h`` (i.e.
    it is missing the circular-buffer format arrays), so a moved/renamed header
    or an unexpected file shape fails loudly instead of yielding empty config.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"descriptor header not found: {path}")
    source = path.read_text()
    arrays = {name: _parse_int_array(body) for name, body in _ARRAY_RE.findall(source)}

    # The generated header always defines these L1-format arrays. Their absence
    # means the file is not the expected shape (wrong file, truncated, or a
    # format change), so fail explicitly rather than silently reporting no CBs.
    if "unpack_src_format" not in arrays and "pack_dst_format" not in arrays:
        raise ValueError(
            f"{path} is missing the expected unpack_src_format/pack_dst_format arrays; "
            "not a recognizable chlkc_descriptors.h"
        )

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

    # Both circular-buffer directions must use the L1 (memory) format, not the
    # in-register format. Per genfiles.cpp, ``unpack_src_format`` is the input
    # CB's L1 format and ``pack_dst_format`` is the output CB's L1 format
    # ("pack dst == L1"). ``pack_src_format`` is the packer's DST-register read
    # format instead, which diverges from L1 under fp32_dest_acc_en (and for
    # Fp8/MX formats), so it must not be used for the output CB format.
    descriptors.unpack_inputs = _collect_cbs(arrays, "unpack", "unpack_src_format", format_names)
    descriptors.pack_outputs = _collect_cbs(arrays, "pack", "pack_dst_format", format_names)
    return descriptors


def _format_name(code: int, format_names: dict[int, str] | None) -> str:
    """Resolve a data-format code via the ELF's tensix ``DataFormat`` enum."""
    if format_names and code in format_names:
        return format_names[code]
    return f"Unknown({code})"


def _collect_cbs(
    arrays: dict[str, list[int]],
    prefix: str,
    format_array: str,
    format_names: dict[int, str] | None = None,
) -> list[CircularBufferConfig]:
    """Build CB configs for every slot whose format is not the 'unused' sentinel.

    ``format_array`` is the L1 data-format array to read (``unpack_src_format``
    for inputs, ``pack_dst_format`` for outputs); tile-geometry arrays are keyed
    by ``prefix`` (``unpack``/``pack``). ``format_names`` (from the ELF's tensix
    ``DataFormat`` enum) is preferred for decoding format codes.
    """
    formats = arrays.get(format_array)
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
                data_format=_format_name(code, format_names),
                tile_size_bytes=tile_sizes[index] if index < len(tile_sizes) else None,
                tile_r_dim=tile_r[index] if index < len(tile_r) else None,
                tile_c_dim=tile_c[index] if index < len(tile_c) else None,
                num_faces=num_faces[index] if index < len(num_faces) else None,
            )
        )
    return result
