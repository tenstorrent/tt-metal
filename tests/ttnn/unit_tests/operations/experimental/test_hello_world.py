# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess
import sys

import pytest
import torch

import ttnn


def _make(device, shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=None):
    # The torch reference must carry the same dtype as the device tensor: the
    # identity output is compared against it, and allclose requires matching dtypes.
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]
    torch_tensor = torch.randn(*shape, dtype=torch_dtype)
    kwargs = {"device": device, "dtype": dtype, "layout": layout}
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    return torch_tensor, ttnn.from_torch(torch_tensor, **kwargs)


def test_hello_world_identity_bf16(device):
    """hello_world is an identity op: output data must equal input data."""
    torch_tensor, tt_tensor = _make(device, (1, 1, 128, 128))  # 16 tiles (tile = 32x32 = 1024 elements)
    result = ttnn.experimental.hello_world(tt_tensor)
    assert torch.allclose(ttnn.to_torch(result), torch_tensor)


def test_hello_world_identity_fp32(device):
    """fp32 input exercises a different CB data format.

    The copy runs through the FPU (matrix engine), whose DST register is TF32
    (11-bit significand), not full fp32 -- see
    docs/source/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.rst.
    So an fp32 tensor is quantized to TF32 on its way through the FPU: the error is
    at most one TF32 ulp (relative 2**-10), far looser than the default allclose
    tolerance (rtol=1e-5, which this would fail). The bf16 test above stays exact
    because bf16 fits inside TF32. (ttnn.identity is exact for fp32 because it uses
    the SFPU / vector engine, which has full 32-bit precision -- a different path.)
    atol covers denormals, where the relative bound is vacuous.
    """
    torch_tensor, tt_tensor = _make(device, (1, 1, 128, 64), dtype=ttnn.float32)  # 8 tiles
    result = ttnn.experimental.hello_world(tt_tensor)
    assert torch.allclose(ttnn.to_torch(result), torch_tensor, rtol=2**-10, atol=2**-24)


def test_hello_world_single_tile(device):
    """One tile -> one core: the minimal case."""
    torch_tensor, tt_tensor = _make(device, (1, 1, 32, 32))  # 1 tile
    result = ttnn.experimental.hello_world(tt_tensor)
    assert torch.allclose(ttnn.to_torch(result), torch_tensor)


def test_hello_world_identity_many_tiles(device):
    """More tiles than cores forces an uneven split across the whole grid:
    pins the per-core tile count / offset math in the factory."""
    torch_tensor, tt_tensor = _make(device, (1, 1, 32, 6400))  # 200 tiles
    result = ttnn.experimental.hello_world(tt_tensor)
    assert torch.allclose(ttnn.to_torch(result), torch_tensor)


def test_hello_world_program_cache_hit(device):
    """Two calls with the same spec but different data must both be correct:
    the second call is a program cache hit and must re-point the output buffer."""
    for _ in range(2):
        torch_tensor, tt_tensor = _make(device, (1, 1, 128, 128))
        result = ttnn.experimental.hello_world(tt_tensor)
        assert torch.allclose(ttnn.to_torch(result), torch_tensor)


def test_hello_world_rejects_row_major(device):
    torch_tensor, tt_tensor = _make(device, (1, 1, 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
    with pytest.raises(Exception, match="TILE layout"):
        ttnn.experimental.hello_world(tt_tensor)


def test_hello_world_rejects_sharded(device):
    torch_tensor, _ = _make(device, (32, 32))
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [32, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    tt_tensor = ttnn.from_torch(
        torch_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=mem_cfg
    )
    with pytest.raises(Exception, match="DRAM interleaved"):
        ttnn.experimental.hello_world(tt_tensor)


def _cpp_tt_logger_debug_active(out: str) -> bool:
    """True if the C++ tt-logger emitted debug-level lines in ``out``.

    The C++ ``log_debug`` calls print a lowercase level tag, e.g.
    ``... | debug    |              Op | [hello_world] ...`` — and the framework
    does the same (``| debug    |           Metal | ...``). The Python-side
    ``ttnn`` logger uses an uppercase tag (``| DEBUG    |``), which is present
    regardless of the build. A stock ``./build_metal.sh`` Release build defaults
    ``TT_METAL_ENABLE_LOGGING=OFF`` and compiles ``log_debug`` out, so it emits
    no lowercase ``debug`` lines. Keying on the lowercase tag therefore
    distinguishes a logging-enabled build (assert the ``[hello_world]`` trace
    lines) from a stock build (skip that assertion).
    """
    return bool(re.search(r"\|\s*debug\s+\|", out))


def test_hello_world_trace_and_dprint():
    """Verify BOTH debug channels end-to-end in one run, in a subprocess because
    the logger and dprint settings are read once, before any device is opened:

    - device-side: the compute kernel DPRINTs ``Hello, world!`` from every core.
      Works in any build; enabled here via ``TT_METAL_DPRINT_CORES=all``.
    - host-side: the factory traces every step it goes through via ``log_debug``
      (``[hello_world]`` lines). Requires a build compiled with
      ``TT_METAL_ENABLE_LOGGING=ON``; enabled here via ``TT_LOGGER_LEVEL=debug``.
      On a stock Release build (``TT_METAL_ENABLE_LOGGING=OFF``) the host-trace
      assertion is skipped, but the device-side DPRINT assertion still runs.
    """
    script = (
        "import torch, ttnn\n"
        "device = ttnn.open_device(device_id=0)\n"
        "x = ttnn.from_torch(torch.rand(1, 1, 128, 64, dtype=torch.bfloat16), device=device,\n"
        "                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)\n"
        "ttnn.experimental.hello_world(x)\n"
        "ttnn.close_device(device)\n"
    )
    env = os.environ.copy()
    env["TT_LOGGER_LEVEL"] = "debug"
    env["TT_METAL_DPRINT_CORES"] = "all"
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600, env=env)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"subprocess failed:\n{out[-4000:]}"
    # Device-side DPRINT: fires in any build, one 'Hello, world!' line per placed core.
    assert "Hello, world!" in out, "device-side 'Hello, world!' DPRINT lines are missing"
    # Host-side trace: only present in a build compiled with TT_METAL_ENABLE_LOGGING=ON.
    if not _cpp_tt_logger_debug_active(out):
        pytest.skip(
            "no C++ tt-logger 'debug' lines in output; the host-side [hello_world] trace "
            "requires a build compiled with -D TT_METAL_ENABLE_LOGGING=ON (a stock "
            "./build_metal.sh Release build compiles log_debug out). The device-side "
            "DPRINT assertion above still passed. Rebuild with TT_METAL_ENABLE_LOGGING=ON "
            "and re-run to exercise the host-side trace."
        )
    assert "[hello_world]" in out, (
        "host-side [hello_world] trace lines are missing even though the build is "
        "logging-enabled (C++ tt-logger 'debug' lines are present); the factory's "
        "log_debug calls may have been removed"
    )
