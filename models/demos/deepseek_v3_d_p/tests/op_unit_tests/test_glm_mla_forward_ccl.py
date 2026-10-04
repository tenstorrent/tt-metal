# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One GLM-5.2 sparse MLA forward, to size what the TP collectives cost.

MLA only -- _run_chunked_prefill builds ttMLA directly, no block or transformer around it. GLM-5.2's
config carries the indexer, so this exercises the sparse (DSA) path including the full-mesh KVPE gather.

Under the SP-batch scheme (one request per TP column) mla.py drops six TP-axis collectives, all already
guarded by `if self.tp_factor > 1`: the q_a_proj reduce-scatter and its re-gather, the KV all-gather and
its fast_reduce_nc_split, the output-gate all-gather, and the o_proj reduce-scatter. Run under tracy to
attribute per-op time and size that saving:

    python -m tracy -r -p -v -m pytest \
        models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_glm_mla_forward_ccl.py
"""

import pytest
from loguru import logger

from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.test_mla import _run_chunked_prefill
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=1152),
            id="torus-xy-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["glm_5_2"], indirect=True, ids=["glm52"])
@pytest.mark.parametrize("chunk", [5120, 2048], ids=["chunk5120", "chunk2048"])
@pytest.mark.timeout(0)
def test_glm_mla_forward(request, mesh_device, device_params, variant, chunk):
    """50k cached prefix plus one fresh chunk, profiled. reference=None: accuracy is other tests' job."""
    topology = per_axis_topology(device_params["fabric_config"])
    total_ns = _run_chunked_prefill(
        request,
        mesh_device,
        reference=None,
        topology=topology,
        profile=True,
        iters_isl=[chunk],
        prefill_len=50 * 1024,
        # GLM's sparse KV rows are bf16 (576 x 2 = 1152 B), which is ttMLA's default and what
        # the kvpe-gather sweep measured; the helper's own default is Kimi's BFP8_TILE.
        cache_format=MlaKvCacheFormat.BF16_RM,
    )
    logger.info(f"GLM-5.2 sparse MLA forward, 50k + {chunk}: {total_ns:,.0f} ns ({total_ns / 1e6:.3f} ms)")
