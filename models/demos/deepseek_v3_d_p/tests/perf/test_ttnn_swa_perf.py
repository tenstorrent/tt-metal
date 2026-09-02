# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-perf for the V4 sliding-only attention layer over a chunked prefill at 5120 tokens a chunk,
on 8x4.

Chunked and not one shot, because the per-chunk work is what a long prefill repeats, and the only thing
a gate on this layer can defend. The state is allocated outside the profiled region.

Timed with the realtime (lightweight) profiler, like test_ttnn_hca_perf.py, and not with Tracy: the
Blaze pipeline builds without the Tracy tools. Every program contributes its MAX duration across chips,
so these numbers are NOT comparable to a Tracy merge, which averages collectives.

Set HCA_PERF_BREAKDOWN=1 for a per-op split of one chunk. The gate only needs the total, but the
profiler already returns per-program records and there is no Tracy here.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_swa import _MESH_CONFIGS, _SEED, _build_reference, _config
from models.demos.deepseek_v3_d_p.tt.mla.sliding_window_attention import TtSWA
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_high_power
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged

_CHUNK = PREFILL_CHUNK_TOKENS
_CHUNKS = 2
_MAX_SEQ = 56_320  # the demo context, 11 chunks of 5120
_MARGIN = 0.05

# Measured, not targets. Roughly half of HCA's 10.9 / 18.4 ms on the same mesh: no compressor, no
# cache write, and Sk is 768 rather than 5792. Taken on a host is_high_power() calls false, where the
# HCA gate nonetheless lands inside its own 14kW-derived band -- so these are comparable, and the 5%
# margin absorbs the rest.
_BASELINES = [
    pytest.param("flash", DeepSeekV4FlashConfig, 5_124_000, id="flash"),  # both chunks -> 2.56 ms a chunk
    pytest.param("pro", DeepSeekV4ProConfig, 8_192_000, id="pro"),  # both chunks -> 4.10 ms a chunk
]

# The team gates perf on the 14kW hosts. Set this to run anywhere for bring-up, where the baselines
# describe nothing and only "does it run" is being checked.
_IGNORE_POWER = os.environ.get("HCA_PERF_IGNORE_POWER") == "1"
_BREAKDOWN = os.environ.get("HCA_PERF_BREAKDOWN") == "1"


def _log_breakdown(per_program, chunk_ns, top=100):
    """Per-op device time inside one chunk, grouped by each program's kernel set."""
    by_kernel: dict[tuple, list] = {}
    for entry in per_program.values():
        key = tuple(sorted({source.rsplit("/", 1)[-1] for source in entry["kernel_sources"]}))
        by_kernel.setdefault(key, []).append(entry["duration_ns"])
    rows = sorted(((sum(v), len(v), k) for k, v in by_kernel.items()), reverse=True)
    for ns, count, key in rows[:top]:
        logger.info(f"    {ns / 1e3:9.1f} us {100 * ns / chunk_ns:5.1f}%  x{count:<3} {', '.join(key)}")


@pytest.mark.skipif(not is_blackhole(), reason="V4 SWA requires Blackhole")
@pytest.mark.skipif(
    not (is_high_power() or _IGNORE_POWER),
    reason="perf job requires a high-power (>=130W TDP) galaxy; guards the exabox.tenstorrent.com/power=14kw "
    "label. HCA_PERF_IGNORE_POWER=1 runs it anyway, for bring-up only",
)
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant, model_config, expected_ns", _BASELINES)
def test_swa_layer_perf_galaxy(mesh_device, device_params, topology, variant, model_config, expected_ns):
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("realtime profiler inactive (needs Blackhole, WORKER dispatch, fabric-tensix DM off)")

    torch.manual_seed(_SEED)
    config = _config(model_config)
    ref = _build_reference(config)
    tt_model = TtSWA.from_reference(mesh_device, ref, config, sp_axis=0, tp_axis=1, topology=topology)
    state = tt_model.alloc_state(_MAX_SEQ, chunk_tokens=_CHUNK)
    ms = tuple(mesh_device.shape)

    total_ns = 0.0
    for it in range(_CHUNKS):
        tt_in = ttnn.from_torch(
            torch.randn(1, _CHUNK, config.hidden_size).unsqueeze(1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=ms, dims=(2, 3)),
        )
        _, per_program = profile_realtime_program_merged(
            mesh_device, lambda: tt_model(tt_in, seq_len_actual=_CHUNK, state=state)
        )
        chunk_ns = sum(entry["duration_ns"] for entry in per_program.values())
        total_ns += chunk_ns
        logger.info(f"  chunk {it}: {chunk_ns / 1e6:.3f} ms over {len(per_program)} programs")
        if _BREAKDOWN:
            _log_breakdown(per_program, chunk_ns)

    logger.info(f"swa {variant} {_CHUNKS}x{_CHUNK} realtime perf: {total_ns:,.0f} ns ({total_ns / 1e6:.3f} ms)")
    if expected_ns is None:
        pytest.skip(f"no baseline yet for {variant}; measured {total_ns:,.0f} ns ({total_ns / 1e6:.3f} ms)")

    lower, upper = expected_ns * (1 - _MARGIN), expected_ns * (1 + _MARGIN)
    assert lower <= total_ns <= upper, (
        f"device time {total_ns:,.0f} ns outside band [{lower:,.0f}, {upper:,.0f}] "
        f"(expected {expected_ns:,} ns, margin +/- {_MARGIN * 100:.1f}%)"
    )
