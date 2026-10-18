# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-class HC controls that separate normalization, packing, and rounding.

Importing this module does not import TTNN. ``install`` must be called by the
serialized hardware owner before model construction. The selected fused class
and its write method remain in use. The functional-arithmetic control compiles
the original read method's exact source into that class; it does not delegate
runtime execution to a functional decoder or a substitute HC object.
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

CANDIDATE_NAMES = (
    "current",
    "split_all",
    "functional_arithmetic",
    "split_unfold_sigmoid",
    "split_weighted_vector_rest",
    "packed_unscaled_functional",
)
_ORIGINAL_FACTORY = None
_ORIGINAL_READ = None
_ORIGINAL_SPLIT = None
_SELECTED_SOURCE = None
_SELECTED_NAME = None


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one HC source match: {old!r}")
    return source.replace(old, new, 1)


def rewritten_source(functional_source, split_source, vector_source, name):
    """Pure host transformation with explicit source-match checks."""
    if name not in CANDIDATE_NAMES:
        raise ValueError(f"Unknown HC split control: {name}")
    functional = textwrap.dedent(functional_source)
    split = textwrap.dedent(split_source)
    vector = textwrap.dedent(vector_source)
    if name == "current":
        source = vector
    elif name == "split_all":
        source = _replace_once(split, "def _read_prefill(self, x):", "def read(self, x):")
    elif name in ("functional_arithmetic", "packed_unscaled_functional"):
        source = functional
        if name == "packed_unscaled_functional":
            source = _replace_once(
                source,
                "    mix = ttnn.linear(normalized, self.down, compute_kernel_config=self.compute)",
                "    packed = ttnn.linear(normalized, self.read_packed_unscaled, compute_kernel_config=self.compute)\n"
                "    mix = packed[..., :320]",
            )
            source = _replace_once(
                source,
                "    inject = ttnn.linear(normalized, self.inject, compute_kernel_config=self.compute)",
                "    inject = packed[..., 320:324]",
            )
    elif name == "split_unfold_sigmoid":
        source = _replace_once(split, "def _read_prefill(self, x):", "def read(self, x):")
        source = _replace_once(
            source,
            '    mix = ttnn.linear(mix, self.up, activation="sigmoid", compute_kernel_config=self.compute)',
            "    mix = ttnn.sigmoid(ttnn.linear(mix, self.up, compute_kernel_config=self.compute))",
        )
        source = _replace_once(
            source,
            "    inject = ttnn.multiply(packed[..., 320:324], 2, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])",
            "    inject = ttnn.multiply(ttnn.sigmoid(packed[..., 320:324]), 2)",
        )
    else:
        source = _replace_once(
            vector,
            "    if x.shape[2] > 4:\n        return self._read_prefill(x)\n",
            "",
        )
        source = _replace_once(
            source,
            "    streams = ttnn.reshape(x, (1, b * seq, self.count, self.width))\n"
            "    streams = ttnn.multiply(\n"
            "        ttnn.rms_norm(streams, epsilon=self.eps, compute_kernel_config=self.compute), self.norm_packed\n"
            "    )\n"
            "    normalized = ttnn.reshape(streams, (1, b, seq, self.count * self.width))",
            "    split_streams = [\n"
            "        ttnn.rms_norm(self._part(x, i), weight=self.norm[i], epsilon=self.eps, compute_kernel_config=self.compute)\n"
            "        for i in range(self.count)\n"
            "    ]\n"
            "    normalized = ttnn.concat(split_streams, dim=-1)\n"
            "    streams = ttnn.reshape(normalized, (1, b * seq, self.count, self.width))",
        )
    compile(source, f"{__file__}:{name}", "exec")
    return source


def _factory_with_weights(cls, state, *, prefix, hf_config, mesh_device):
    """Setup-only additions; upload byte-identical baseline BF16 weights."""
    import torch

    import ttnn

    obj = _ORIGINAL_FACTORY(cls, state, prefix=prefix, hf_config=hf_config, mesh_device=mesh_device)

    def upload(value):
        return ttnn.from_torch(
            value.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh_device,
        )

    if _SELECTED_NAME == "functional_arithmetic":
        obj.down = upload(state[prefix + ".input_mix_weight_down.weight"].T)
        obj.inject = upload(state[prefix + ".block_inject_weight.weight"].T)
    elif _SELECTED_NAME == "packed_unscaled_functional":
        obj.read_packed_unscaled = upload(
            torch.cat(
                (
                    state[prefix + ".input_mix_weight_down.weight"].T,
                    torch.nn.functional.pad(state[prefix + ".block_inject_weight.weight"].T, (0, 28)),
                ),
                dim=-1,
            )
        )
    else:
        raise RuntimeError("This setup wrapper is only for controls that need additional weights")
    return obj


def install(name):
    """Patch methods on the existing class, preserving the suite's type checks."""
    global _ORIGINAL_FACTORY, _ORIGINAL_READ, _ORIGINAL_SPLIT, _SELECTED_SOURCE, _SELECTED_NAME
    from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused
    from models.autoports.qwen_qwen3_8_flash_next.tt.hyperconnection import HyperConnection

    if _ORIGINAL_FACTORY is None:
        _ORIGINAL_FACTORY = fused.FusedHyperConnection.from_state_dict.__func__
        _ORIGINAL_READ = fused.FusedHyperConnection.read
        _ORIGINAL_SPLIT = fused.FusedHyperConnection._read_prefill
    source = rewritten_source(
        inspect.getsource(HyperConnection.read),
        inspect.getsource(_ORIGINAL_SPLIT),
        inspect.getsource(_ORIGINAL_READ),
        name,
    )
    namespace = dict(HyperConnection.read.__globals__)
    exec(compile(source, f"{__file__}:{name}", "exec"), namespace)
    method = namespace["read"]
    method.__module__ = __name__
    method._rewritten_source = source
    _SELECTED_SOURCE, _SELECTED_NAME = source, name
    factory = (
        _factory_with_weights if name in ("functional_arithmetic", "packed_unscaled_functional") else _ORIGINAL_FACTORY
    )
    fused.FusedHyperConnection.from_state_dict = classmethod(factory)
    fused.FusedHyperConnection.read = method
    return method


def snapshot_provenance(directory):
    if _SELECTED_SOURCE is None:
        raise RuntimeError("Install a candidate before recording provenance")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    manifest = {"candidate": _SELECTED_NAME, "sources": {}}
    for path in (Path(__file__).resolve(), root / "tt/fused_decoder.py", root / "tt/hyperconnection.py"):
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


def install_conv_bf16():
    """Keep the native op's documented default arithmetic; construct config in setup."""
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused

    original_factory = fused.FusedGatedDeltaNet.from_state_dict.__func__

    def factory(cls, *args, **kwargs):
        obj = original_factory(cls, *args, **kwargs)
        obj.conv_compute = ttnn.init_device_compute_kernel_config(
            obj.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False
        )
        return obj

    method = fused.FusedGatedDeltaNet._chunk
    source = textwrap.dedent(inspect.getsource(method))
    old = "program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),\n                    compute_kernel_config=self.compute,"
    new = "program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),\n                    compute_kernel_config=self.conv_compute,"
    assert source.count(old) == 1
    source = source.replace(old, new)
    namespace = dict(method.__globals__)
    exec(compile(source, __file__, "exec"), namespace)
    fused.FusedGatedDeltaNet.from_state_dict = classmethod(factory)
    fused.FusedGatedDeltaNet._chunk = namespace["_chunk"]
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=CANDIDATE_NAMES)
    parser.add_argument("--layers", default="layer0 or layer1")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--gdn-control", choices=["none", "conv", "norm", "both", "conv_bf16"], default="none")
    args = parser.parse_args()
    import pytest

    from models.autoports.qwen_qwen3_8_flash_next.tests import test_fused_decoder as suite

    root = Path(__file__).resolve().parents[1]
    selected = root / "doc/fused_decoder"
    output = (
        args.evidence_dir
        or selected / "hc_split_controls" / (args.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    ).resolve()
    if output == selected.resolve():
        parser.error("Choose a separate control evidence directory")
    install(args.name)
    from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused

    if args.gdn_control in ("conv", "both"):
        fused.FusedGatedDeltaNet.native_conv = False
    if args.gdn_control in ("norm", "both"):
        fused.FusedGatedDeltaNet.native_norm = False
    if args.gdn_control == "conv_bf16":
        source = textwrap.dedent(inspect.getsource(fused.FusedGatedDeltaNet._chunk))
        old = "program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),\n                    compute_kernel_config=self.compute,"
        new = "program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),\n                    compute_kernel_config=ttnn.init_device_compute_kernel_config(self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False),"
        assert source.count(old) == 1
        source = source.replace(old, new)
        namespace = dict(fused.FusedGatedDeltaNet._chunk.__globals__)
        exec(compile(source, __file__, "exec"), namespace)
        fused.FusedGatedDeltaNet._chunk = namespace["_chunk"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "generated_gdn_chunk.py.txt").write_text(source)
    snapshot_provenance(output / "sources")
    (output / "control.json").write_text(
        json.dumps(
            {
                "candidate": args.name,
                "gdn_control": args.gdn_control,
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
    print(f"HC_SPLIT_CONTROL candidate={args.name} evidence_dir={output}", flush=True)
    with patch.object(suite, "EVIDENCE", output):
        return pytest.main(["-s", "-q", "--timeout=0", str(root / "tests/test_fused_decoder.py"), "-k", args.layers])


if __name__ == "__main__":
    raise SystemExit(main())
