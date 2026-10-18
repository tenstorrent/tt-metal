# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate short-vector HC normalization and summation rounding.

``install(name)`` patches the existing class method; decoder type checks remain
valid. Importing this file does not import TTNN or access a device. Candidates
are experimental until the real HF rows and traced latency have been measured.
"""

import argparse
import hashlib
import inspect
import json
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

CANDIDATE_NAMES = ("fp32_norm", "sequential_sum", "fp32_norm_sequential_sum", "precise_norm", "stable_norm")
_ORIGINAL = None
_SELECTED_SOURCE = None
_SELECTED_NAME = None

_NORM_OLD = """    streams = ttnn.multiply(
        ttnn.rms_norm(streams, epsilon=self.eps, compute_kernel_config=self.compute), self.norm_packed
    )"""
_NORM_NEW = """    # Preserve normalization until the per-stream affine multiply. The
    # selected BF16 path packs once before gamma and again after gamma.
    # Explicit FP32 operands keep this candidate's gamma product in FP32;
    # the final BF16 cast matches the split weighted-RMS output boundary.
    streams = ttnn.typecast(streams, ttnn.float32)
    streams = ttnn.rms_norm(streams, epsilon=self.eps, compute_kernel_config=self.compute)
    streams = ttnn.typecast(
        ttnn.multiply(streams, ttnn.typecast(self.norm_packed, ttnn.float32)), ttnn.bfloat16
    )"""
_SUM_OLD = """    mixed = ttnn.mean(mixed, dim=2, keepdim=True)"""
_SUM_NEW = """    # Keep the vector normalization and products, but reproduce the
    # split path's three BF16 additions and final exact power-of-two scale.
    products = mixed
    mixed = products[:, :, :1, :]
    for i in range(1, self.count):
        mixed = ttnn.add(mixed, products[:, :, i : i + 1, :])
    mixed = ttnn.multiply(mixed, 1 / self.count)"""


def rewritten_source(source, name):
    """Pure host transformation, also usable for source-only verification."""
    if name not in CANDIDATE_NAMES:
        raise ValueError(f"Unknown HC precision candidate: {name}")
    source = textwrap.dedent(source)
    replacements = []
    if name in ("fp32_norm", "fp32_norm_sequential_sum"):
        replacements.append((_NORM_OLD, _NORM_NEW))
    if name in ("precise_norm", "stable_norm"):
        precise = """    streams = ttnn.typecast(streams, ttnn.float32)
    variance = ttnn.multiply(ttnn.sum(ttnn.multiply(streams, streams), dim=-1, keepdim=True), 1 / self.width)
    inv = ttnn.rsqrt(ttnn.add(variance, self.eps))
    streams = ttnn.typecast(ttnn.multiply(ttnn.multiply(streams, inv), ttnn.typecast(self.norm_packed, ttnn.float32)), ttnn.bfloat16)"""
        if name == "stable_norm":
            native = "\n".join(line for line in _NORM_NEW.splitlines() if not line.lstrip().startswith("#"))
            precise = (
                "    if seq == 1:\n"
                + textwrap.indent(precise, "    ")
                + "\n    else:\n"
                + textwrap.indent(native, "    ")
            )
        replacements.append((_NORM_OLD, precise))
    if name in ("sequential_sum", "fp32_norm_sequential_sum"):
        replacements.append((_SUM_OLD, _SUM_NEW))
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"Expected exactly one HC replacement: {old!r}")
        source = source.replace(old, new, 1)
    compile(source, f"{__file__}:{name}", "exec")
    return source


def install(name):
    """Install one independent method patch before constructing the decoder."""
    global _ORIGINAL, _SELECTED_SOURCE, _SELECTED_NAME
    from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused

    if _ORIGINAL is None:
        _ORIGINAL = fused.FusedHyperConnection.read
    source = rewritten_source(inspect.getsource(_ORIGINAL), name)
    namespace = dict(_ORIGINAL.__globals__)
    exec(compile(source, f"{__file__}:{name}", "exec"), namespace)
    method = namespace["read"]
    method.__module__ = __name__
    method._rewritten_source = source
    fused.FusedHyperConnection.read = method
    _SELECTED_SOURCE, _SELECTED_NAME = source, name
    return method


def snapshot_provenance(directory):
    """Record exact candidate, selected runtime, and generated method sources."""
    if _SELECTED_SOURCE is None:
        raise RuntimeError("Install a candidate before recording provenance")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    files = (Path(__file__).resolve(), root / "tt/fused_decoder.py", root / "tt/hyperconnection.py")
    manifest = {"candidate": _SELECTED_NAME, "sources": {}}
    for path in files:
        data = path.read_bytes()
        destination = directory / (path.name + ".txt")
        destination.write_bytes(data)
        manifest["sources"][str(path.relative_to(root))] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "snapshot": destination.name,
        }
    generated = directory / "generated_read.py.txt"
    generated.write_text(_SELECTED_SOURCE)
    manifest["generated_read_sha256"] = hashlib.sha256(generated.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=CANDIDATE_NAMES)
    parser.add_argument("--layers", default="layer0 or layer1")
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    import pytest

    from models.autoports.qwen_qwen3_8_flash_next.tests import test_fused_decoder as suite

    root = Path(__file__).resolve().parents[1]
    selected = root / "doc/fused_decoder"
    output = (
        args.evidence_dir
        or selected
        / "hc_precision_controls"
        / (args.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    ).resolve()
    if output == selected.resolve():
        parser.error("Choose a separate control evidence directory")
    install(args.name)
    snapshot_provenance(output / "sources")
    (output / "control.json").write_text(
        json.dumps(
            {
                "candidate": args.name,
                "layers": args.layers,
                "settings": {
                    key: os.environ.get(key)
                    for key in (
                        "QWEN38_WEIGHT_SOURCE",
                        "QWEN38_DECODER_BATCHES",
                        "QWEN38_DECODER_LENGTHS",
                        "QWEN38_DECODER_CHUNK_SIZE",
                        "TT_METAL_TRACE_ALLOC_TRACKING",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"HC_PRECISION_CONTROL candidate={args.name} evidence_dir={output}", flush=True)
    with patch.object(suite, "EVIDENCE", output):
        return pytest.main(["-s", "-q", "--timeout=0", str(root / "tests/test_fused_decoder.py"), "-k", args.layers])


if __name__ == "__main__":
    raise SystemExit(main())
