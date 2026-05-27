"""Device registry, reference-defaults extractor, and tt-smi reset helper.

Three distinct responsibilities, kept separate within this module:

1. DEVICE_REGISTRY — static data table mapping user-facing device names
   (n150/n300/p150/t3k/tg) to arch + mesh_shape + num_devices.
2. extract_defaults(reference_path) — greps a reference TTNN model folder
   for known device-init constants and returns them as a dict.
3. tt_smi_reset() — wraps `tt-smi -r` and returns the exit code without
   raising.

Pure Python — no torch, no ttnn imports.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from typing import Optional, Union

# ---------------------------------------------------------------------------
# 1. Device registry
# ---------------------------------------------------------------------------

DEVICE_REGISTRY: dict[str, dict] = {
    "n150": {"arch": "wormhole_b0", "mesh_shape": (1, 1), "num_devices": 1},
    "n300": {"arch": "wormhole_b0", "mesh_shape": (1, 2), "num_devices": 2},
    "p150": {"arch": "blackhole", "mesh_shape": (1, 1), "num_devices": 1},
    "t3k": {"arch": "wormhole_b0", "mesh_shape": (1, 8), "num_devices": 8},
    "tg": {"arch": "wormhole_b0", "mesh_shape": (8, 4), "num_devices": 32},
}


class UnknownDeviceError(KeyError):
    """Raised when a device name is not in DEVICE_REGISTRY."""


def device_info(name: str) -> dict:
    """Look up a device in the registry.

    Returns a *copy* so callers can't mutate the registry. Raises
    UnknownDeviceError if `name` is not in DEVICE_REGISTRY.
    """
    if name not in DEVICE_REGISTRY:
        raise UnknownDeviceError(name)
    # Shallow copy is sufficient — registry values are tuples/ints/strs.
    return dict(DEVICE_REGISTRY[name])


# ---------------------------------------------------------------------------
# 2. Reference-defaults extractor
# ---------------------------------------------------------------------------

# Pattern for grep -E: matches any of the three keys as a word.
_GREP_PATTERN = r"l1_small_size|trace_region_size|mesh_shape"

# Line-level regexes (applied to each grep stdout line).
#
# A grep line looks like:  path/to/foo.py:42:    l1_small_size = 16384,
#
# We anchor on the key, then `=` or `:` (allowing surrounding spaces), then
# capture the RHS up to the first comma / closing-paren-at-end / EOL.
_INT_LINE_RE = re.compile(
    r"\b(?P<key>l1_small_size|trace_region_size)[\"']?\s*[=:]\s*" r"(?P<val>0x[0-9A-Fa-f_]+|[0-9][0-9_]*)"
)

_MESH_LINE_RE = re.compile(r"\bmesh_shape[\"']?\s*[=:]\s*(?P<val>[\(\[][^\)\]]*[\)\]])")


def _parse_int_literal(token: str) -> Optional[int]:
    """Parse an int literal supporting decimal, hex, and underscores."""
    try:
        # int(token, 0) handles 0x prefix; underscores allowed in 3.6+.
        return int(token, 0)
    except (ValueError, TypeError):
        return None


def _parse_mesh_literal(token: str) -> Optional[tuple]:
    """Parse a (a, b) or [a, b] tuple/list literal via ast.literal_eval."""
    try:
        val = ast.literal_eval(token)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(val, (tuple, list)):
        return None
    try:
        coerced = tuple(int(x) for x in val)
    except (TypeError, ValueError):
        return None
    if len(coerced) != 2:
        return None
    return coerced


def extract_defaults(reference_path: Union[str, Path]) -> dict:
    """Extract device-init defaults from a reference TTNN model folder.

    Greps `reference_path` recursively for `l1_small_size`, `trace_region_size`,
    and `mesh_shape` literal assignments, parses the first hit for each, and
    returns a dict:

        {
            "l1_small_size": <int|None>,
            "trace_region_size": <int|None>,
            "mesh_shape": <tuple(int,int)|None>,
        }

    Missing or unparseable values are None (not raised).
    """
    result: dict = {
        "l1_small_size": None,
        "trace_region_size": None,
        "mesh_shape": None,
    }

    proc = subprocess.run(
        ["grep", "-rEn", _GREP_PATTERN, str(reference_path)],
        capture_output=True,
        check=False,
        text=True,
    )

    if proc.returncode != 0 or not proc.stdout:
        return result

    for line in proc.stdout.splitlines():
        # First-hit-wins for each key independently.
        if result["l1_small_size"] is None or result["trace_region_size"] is None:
            m = _INT_LINE_RE.search(line)
            if m:
                key = m.group("key")
                if result[key] is None:
                    parsed = _parse_int_literal(m.group("val"))
                    if parsed is not None:
                        result[key] = parsed

        if result["mesh_shape"] is None:
            m2 = _MESH_LINE_RE.search(line)
            if m2:
                parsed = _parse_mesh_literal(m2.group("val"))
                if parsed is not None:
                    result["mesh_shape"] = parsed

        # Early exit when all three are populated.
        if all(v is not None for v in result.values()):
            break

    return result


# ---------------------------------------------------------------------------
# 3. tt-smi reset helper
# ---------------------------------------------------------------------------


def tt_smi_reset() -> int:
    """Run `tt-smi -r` to reset the device. Returns the exit code.

    Does NOT raise on non-zero exit — the orchestrator handles failure
    surfacing.
    """
    result = subprocess.run(["tt-smi", "-r"], check=False, capture_output=True)
    return result.returncode
