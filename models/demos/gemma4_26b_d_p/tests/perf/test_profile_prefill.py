# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-time profile of Gemma-4 chunked SP prefill with RANDOM weights (real shapes, no weight load).

Profiles one sliding layer + one full layer (every sliding layer costs the same) for a 4096-token chunk at three
cache depths (kv_actual = 0, 4096, 28672 in a 32k cache), after an untimed warm-up chunk. Each profiled chunk is
bracketed by tracy signposts; ops are read back per chunk with ttnn.ReadDeviceProfiler.

Run:  ./scripts/run_safe_pytest.sh --profile models/demos/gemma4_26b_d_p/tests/perf/test_profile_prefill.py -k 4x1
"""

import json
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.demos.gemma4_26b_d_p.reference.config import DEFAULT_CKPT_DIR, Gemma4TextConfig
from models.demos.gemma4_26b_d_p.reference.weights import TEXT_PREFIX
from models.demos.gemma4_26b_d_p.tests.mesh import MESH_PARAMS
from models.demos.gemma4_26b_d_p.tt.model import TtGemma4Model

try:
    from tracy import signpost
except ImportError:  # profiler not available: keep the test runnable
    def signpost(*_a, **_k):
        pass


class RandomReader:
    """CheckpointReader look-alike: real tensor shapes from the safetensors headers, random values."""

    def __init__(self, ckpt_dir=DEFAULT_CKPT_DIR, seed=0):
        d = Path(ckpt_dir)
        idx = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
        self.shapes = {}
        for fname in sorted(set(idx.values())):
            with safe_open(str(d / fname), framework="pt") as f:
                for k in f.keys():
                    if k.startswith(TEXT_PREFIX):
                        self.shapes[k[len(TEXT_PREFIX) :]] = f.get_slice(k).get_shape()
        self.g = torch.Generator().manual_seed(seed)

    def get(self, key):
        shape = self.shapes[key]
        if "layernorm" in key or key.endswith("norm.weight") or key.endswith("scale") or key.endswith("layer_scalar"):
            return torch.ones(shape)
        return torch.randn(shape, generator=self.g) * 0.02

    def keys(self, prefix=""):
        return [k for k in self.shapes if k.startswith(prefix)]

    def substate(self, prefix):
        p = prefix if prefix.endswith(".") else prefix + "."
        return {k[len(p) :]: self.get(k) for k in self.keys(p)}

    def layer_state(self, i):
        return self.substate(f"layers.{i}")


CHUNK = 4096
MAX_SEQ = 32768
DEPTHS = [0, 4096, MAX_SEQ - CHUNK]


def _ring1d_params():
    from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_y_device_params

    p = torus_y_device_params()
    p["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING
    return p


# 4x1 on a 1D ring fabric: enables the MoE dispatch sparse multicast (one send per destination chip, not per expert).
EXTRA = pytest.mark.parametrize(
    "mesh_device, device_params", [pytest.param((4, 1), _ring1d_params(), id="4x1-1dring")], indirect=["mesh_device", "device_params"]
)


@EXTRA
def test_profile_prefill_1dring(mesh_device, device_params):
    _profile(mesh_device, device_params)


@MESH_PARAMS
def test_profile_prefill(mesh_device, device_params):
    _profile(mesh_device, device_params)


def _profile(mesh_device, device_params):
    """GEMMA4_NUM_LINKS / GEMMA4_PROFILE_FABRIC=1d_ring select the MoE link count / a 1D fabric (sparse-mcast dispatch)."""
    cfg = Gemma4TextConfig.from_json()
    sp, tp = tuple(mesh_device.shape)
    layers = [4, 5]  # one sliding, one full (layer 5 = first full layer)
    model = TtGemma4Model(mesh_device, cfg, RandomReader(), fabric_config=device_params["fabric_config"], max_seq_len=MAX_SEQ,
                          chunk_size=CHUNK, layers=layers, build_lm_head=False)
    ids = torch.randint(0, cfg.vocab_size, (CHUNK,))
    C = CHUNK // sp

    def run(kv_actual):
        tok = model.tokens_to_device(ids)
        x = model.forward_device(model.embed_device(tok), kv_actual)
        x.deallocate(True)
        ttnn.synchronize_device(mesh_device)

    run(0)  # warm-up / compile, untimed
    ttnn.ReadDeviceProfiler(mesh_device)
    for d in DEPTHS:
        signpost(f"chunk_kv{d}_start")
        run(d)
        signpost(f"chunk_kv{d}_end")
        ttnn.ReadDeviceProfiler(mesh_device)
    logger.info(f"profiled mesh={sp}x{tp} layers={layers} chunk={CHUNK} depths={DEPTHS}")
