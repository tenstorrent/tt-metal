# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-profiler run of the production V4.1 blocks (G2 per-op table, bead 8y7.9.3).

Run only under the Tracy device profiler (``scripts/run_safe_pytest.sh --profile <id>``); without it the test is a
plain warm run. Same setup as ``test_v41_perf`` (``text_stack``): one 5120-token chunk of real text at start 0 through
the sharing schedule 0 -> 2 -> 3 -> 20 -> 21 -> 24, real bfp8 experts, LoudBox 2x4, BF16 KV.

Phases, each delimited by Tracy signposts so the ops CSV can be attributed offline:

* ``warm``: compile pass (discarded);
* ``untraced``: one steady-state forward per layer. Every sublayer / MoE / attention stage and every mHC call
  (split, collapse, hc_post) is wrapped in ``B <stage>`` / ``E <stage>`` signposts (host-side only, no device
  synchronize), so each device op falls into its innermost stage; the per-op device kernel durations come from here;
* ``traced``: one trace replay per layer (captured with ``test_v41_trace.capture``), for the traced layer total and
  the per-op traced kernel durations.

The device profiler buffer is read after every forward / replay (``ttnn.ReadDeviceProfiler``); set
``TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT`` above one forward's program count.
"""

import time

import pytest
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_perf import CHUNK, LAYERS, text_stack
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_trace import MESH, capture

EXPERT_DTYPE = "bfp8"


class _Signposted:
    """Wraps a callable in ``B <name>`` / ``E <name>`` signposts while ``flag['on']``."""

    def __init__(self, fn, name, flag):
        self._fn, self._name, self._flag = fn, name, flag

    def __call__(self, *args, **kwargs):
        if not self._flag["on"]:
            return self._fn(*args, **kwargs)
        signpost(f"B {self._name}")
        out = self._fn(*args, **kwargs)
        signpost(f"E {self._name}")
        return out

    def __getattr__(self, name):
        return getattr(self._fn, name)


def _instrument(block, flag):
    def wrap(owner, attr, name):
        if getattr(owner, attr, None) is not None:
            setattr(owner, attr, _Signposted(getattr(owner, attr), name, flag))

    attn, moe, res = block.attn, block.ffn.moe, block.residual
    for attr in ("compressor", "index_keys", "indexer"):
        wrap(attn, attr, f"attention.{attr}")
    for attr in ("gate", "dispatch_module", "routed_expert", "combine_module", "reduce_module", "shared_expert"):
        wrap(moe, attr, f"moe.{attr}")
    for site in ("attn_site", "ffn_site"):
        for attr in ("split", "collapse", "hc_post"):
            wrap(getattr(res, site), attr, f"mhc.{site}.{attr}")
    wrap(block, "attn_norm", "attn_norm")
    wrap(block, "attn", "attention")
    wrap(block, "ffn_norm", "ffn_norm")
    wrap(block, "ffn", "moe")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_block_profile(mesh_device, device_params):
    layers = LAYERS
    blocks, x0, pre0, fresh = text_stack(mesh_device, EXPERT_DTYPE)
    flag = {"on": False}
    for layer in layers:
        _instrument(blocks[layer], flag)

    def stack(phase, inputs=None):
        state = fresh()
        x, pre = x0, pre0
        for layer in layers:
            if inputs is not None:
                inputs[layer] = (x, pre)
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            signpost(f"B {phase} L{layer}")
            x, pre = blocks[layer](x, pre, state, CHUNK)
            signpost(f"E {phase} L{layer}")
            ttnn.synchronize_device(mesh_device)
            ttnn.ReadDeviceProfiler(mesh_device)
            logger.info(f"{phase} L{layer} {(time.perf_counter() - t0) * 1e3:.1f} ms (profiled)")
        return state

    t0 = time.perf_counter()
    stack("warm")
    logger.info(f"warm stack {time.perf_counter() - t0:.1f}s")
    flag["on"] = True
    stack("untraced")
    flag["on"] = False

    inputs = {}
    state = stack("staging", inputs=inputs)
    for layer in layers:
        x, pre = inputs[layer]
        trace, _ = capture(mesh_device, lambda: blocks[layer](x, pre, state, CHUNK), [blocks[layer].ffn])
        trace.replay()  # warm replay
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        t0 = time.perf_counter()
        signpost(f"B traced L{layer}")
        trace.replay()
        signpost(f"E traced L{layer}")
        ttnn.synchronize_device(mesh_device)
        logger.info(f"traced L{layer} {(time.perf_counter() - t0) * 1e3:.1f} ms (profiled, {trace.num_segments} seg)")
        ttnn.ReadDeviceProfiler(mesh_device)
        trace.release()
