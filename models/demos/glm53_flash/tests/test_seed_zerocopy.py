# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only check of the seed server's zero-copy body (tt-blaze G53F-0189): ``st_parts`` concatenated is a safetensors
file that ``safetensors.torch.load`` (the ring's ``seed_transport.tensors_from_bytes``) reads back to tensors equal to
the input, for the dtypes and shapes a seed carries (bf16 latent / pool rows, fp32 rings and KDA carries), edge shapes
included; and ``_send_parts`` frames it exactly as ``_send`` frames ``save``'s bytes.

    pytest models/demos/glm53_flash/tests/test_seed_zerocopy.py
"""

from __future__ import annotations

import importlib.util
import socket
import struct
import sys
import threading
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load, save


def _server_module():
    """The seed server's helpers without importing ttnn (this test runs on any host)."""
    stub = types.ModuleType("ttnn")
    stub.bfloat8_b = stub.bfloat4_b = object()
    saved = {k: sys.modules.get(k) for k in ("ttnn",)}
    sys.modules["ttnn"] = stub
    for name in (
        "models.demos.glm53_flash.reference.checkpoint",
        "models.demos.glm53_flash.tests.test_prefill_vs_blaze_golden",
        "models.demos.glm53_flash.tt.config",
        "models.demos.glm53_flash.tt.mesh",
        "models.demos.glm53_flash.tt.mesh_model",
    ):
        m = types.ModuleType(name)
        for attr in (
            "default_snapshot_dir",
            "_dsa_state_host",
            "_kda_state_host",
            "GLM53FlashConfig",
            "GLM53MeshModel",
        ):
            setattr(m, attr, None)
        m.mesh_device_params = lambda *a, **k: {}
        saved.setdefault(name, sys.modules.get(name))
        sys.modules[name] = m
    try:
        path = Path(__file__).with_name("test_prefill_seed_server.py")
        spec = importlib.util.spec_from_file_location("_seed_server_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


SRV = _server_module()


def _cases():
    g = torch.Generator().manual_seed(0)
    yield "dsa", {
        "latent": torch.randn(4093, 512, generator=g).to(torch.bfloat16),
        "index_pool": torch.randn(1024, 128, generator=g).to(torch.bfloat16),
        "index_ring_k": torch.randn(4, 128, generator=g),
        "index_ring_g": torch.randn(4, 32, generator=g),
    }
    yield "kda", {
        "recurrent_state": torch.randn(1, 64, 128, 128, generator=g),
        "conv_history": torch.randn(1, 3, 24576, generator=g).to(torch.bfloat16),
    }
    yield "edge", {
        "scalar": torch.tensor(3.5),
        "empty": torch.zeros(0, 512, dtype=torch.bfloat16),
        "ints": torch.arange(7, dtype=torch.int32),
        "noncontig": torch.randn(8, 6, generator=g).t(),
        "b": torch.tensor([True, False, True]),
    }


@pytest.mark.parametrize("name,tensors", list(_cases()))
def test_st_parts_roundtrip(name, tensors):
    parts = SRV.st_parts(tensors)
    blob = b"".join(bytes(p) for p in parts)
    assert len(blob) == SRV.parts_len(parts)
    got = load(blob)
    ref = load(save({k: v.contiguous() for k, v in tensors.items()}))
    assert set(got) == set(tensors) == set(ref)
    for k, v in tensors.items():
        assert got[k].dtype == v.dtype and tuple(got[k].shape) == tuple(v.shape), k
        assert torch.equal(got[k], v) and torch.equal(got[k], ref[k]), k


def test_send_parts_frame_equals_send():
    tensors = dict(_cases())["dsa"]
    hdr = {"op": "seed", "n": 4093, "slot": 0, "request": 1}

    def capture(fn, body):
        a, b = socket.socketpair()
        out = bytearray()

        def rd():
            while True:
                c = b.recv(1 << 20)
                if not c:
                    break
                out.extend(c)

        t = threading.Thread(target=rd)
        t.start()
        fn(a, hdr, body)
        a.close()
        t.join()
        b.close()
        return bytes(out)

    raw = capture(SRV._send, save({k: v.contiguous() for k, v in tensors.items()}))
    zc = capture(SRV._send_parts, SRV.st_parts(tensors))
    # same frame structure: header json, u64 body length, a body that loads to the same tensors
    (hl,) = struct.unpack("<I", raw[:4])
    (hl2,) = struct.unpack("<I", zc[:4])
    assert raw[: 4 + hl] == zc[: 4 + hl2]
    (bl,) = struct.unpack("<Q", raw[4 + hl : 12 + hl])
    (bl2,) = struct.unpack("<Q", zc[4 + hl2 : 12 + hl2])
    assert bl == len(raw) - 12 - hl and bl2 == len(zc) - 12 - hl2
    a, b = load(raw[12 + hl :]), load(zc[12 + hl2 :])
    assert all(torch.equal(a[k], b[k]) for k in tensors)
