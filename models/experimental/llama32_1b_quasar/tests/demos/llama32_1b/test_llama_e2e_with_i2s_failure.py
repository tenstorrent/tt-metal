# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the Quasar interleaved_to_sharded (i2s) fault IN THE MODEL CONTEXT.

The standalone i2s repro (tests/ttnn/unit_tests/operations/test_quasar_i2s_rope_fault.py) PASSES for the
same [1,1,1,64] shape -- the fault is context/allocator-dependent, and the model's allocation is
deterministic (the watcher fault address 0x04a46b43 is byte-identical across model runs). So the only
reliable repro is the model itself, run WITHOUT the i2s route-around that test_llama_e2e.py installs.

This test is that repro. It is identical to test_llama_e2e.py's Quasar setup (weight-upload tilize routed
to the Gen2 quasar op so it reaches the forward) EXCEPT:
  * it does NOT install _install_quasar_i2s (so the faulting mainline ttnn.interleaved_to_sharded runs), and
  * it installs _install_quasar_i2s_logger, which logs every i2s call's input spec + buffer address and
    output memcfg. The LAST ``[i2s-log]`` line before the device fault is the culprit call; its
    ``in_buf_addr`` is the placement a standalone repro must match to trigger the fault deterministically.

Expected outcome on an unfixed device: the RoPE cos/sin decode i2s ([1,1,1,64] -> single-core
HEIGHT_SHARDED) trips a watcher assert (DM1 UNALIGNED_LOAD, neighbour core (1,0)) and the process aborts --
so this "test" crashes rather than fails cleanly; its value is the [i2s-log] trail on stdout. If the i2s
is fixed, the forward completes and the test passes.

Run (Quasar sim, watcher on so the fault is caught):
    TT_METAL_WATCHER=12 MESH_DEVICE=<qsr> LLAMA32_1B_DEMO_NUM_LAYERS=1 \
        pytest models/experimental/llama32_1b_quasar/tests/demos/llama32_1b/test_llama_e2e_with_i2s_failure.py
"""

import os

import pytest
from loguru import logger


from models.experimental.llama32_1b_quasar.tests.demos.llama32_1b import demo
from models.experimental.llama32_1b_quasar.tests.demos.llama32_1b.demo import (
    EXPECTED_METRICS,
    _run_token_accuracy,
    create_model,
    get_device_name,
    lazy_weight_cache_dir_for_demo,
    mesh_device,  # noqa: F401 — pytest fixture, used by injection
)

# Reuse the exact Quasar setup helpers from the main e2e test so this repro only differs in the i2s hook.
from models.experimental.llama32_1b_quasar.tests.demos.llama32_1b.test_llama_e2e import (
    _install_quasar_tilize_from_torch,
    _install_quasar_i2s_logger,
)
from models.experimental.llama32_1b_quasar.utility_functions import is_quasar

pytestmark = demo.pytestmark


@pytest.mark.parametrize("optimizations", ["performance"])
def test_llama_e2e_with_i2s_failure(mesh_device, optimizations, monkeypatch):  # noqa: F811 — imported fixture
    """Run the model WITHOUT the i2s route-around to reproduce the mainline i2s fault, with per-call i2s
    logging. Faults (process abort) on an unfixed device; passes if i2s is fixed."""
    quasar = is_quasar()
    if not quasar:
        pytest.skip("i2s fault repro is Quasar-only")

    if "LLAMA_PCC_LOG" not in os.environ:
        monkeypatch.setenv("LLAMA_PCC_LOG", "1")
    monkeypatch.setenv("DISABLE_MINIMAL_MATMUL", "1")
    monkeypatch.setenv("LLAMA32_1B_DEMO_NUM_LAYERS", os.environ.get("LLAMA32_1B_DEMO_NUM_LAYERS", "1"))
    # Reach the forward: weight-upload tilize must still be routed to the Gen2 op (mainline hangs).
    _install_quasar_tilize_from_torch(monkeypatch)
    # Log every i2s call but DO NOT route it -- let the faulting mainline op run so the fault reproduces.
    _install_quasar_i2s_logger(monkeypatch)

    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
    cache_dir = lazy_weight_cache_dir_for_demo(mesh_device, hf_model)
    device_name = get_device_name(mesh_device)
    expected = EXPECTED_METRICS.get(optimizations, {}).get(device_name, {})

    model = create_model(mesh_device, optimizations, cache_dir, max_batch_size=1, max_seq_len=4096)

    # Run the forward. On an unfixed device this aborts at the RoPE cos/sin i2s (watch the last [i2s-log]
    # line). If it returns, the i2s no longer faults -- log and pass (token-accuracy is not the point here).
    try:
        _run_token_accuracy(model, mesh_device, expected)
    except AssertionError as e:
        logger.warning(f"[i2s-repro] forward completed WITHOUT an i2s fault (token-accuracy aside): {e}")
    logger.info("[i2s-repro] forward returned without a device fault — the mainline i2s did not fault this run")
