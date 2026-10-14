# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device time of one MoE forward on the 8x1 LoudBox, per model and per dispatch/combine transport.

Report only, no gate. The number is the real-time profiler's, as in test_kimi_moe_perf.py: over the programs
the warm forward dispatched, the sum of each program's critical path (its max across chips), split by the op
directory its kernels come from. The shared expert does not overlap dispatch here, as it does in the model:
the records of programs running side by side on sub-devices are not reliable (a direct dispatch read 0.7 us),
so every program runs alone on its cores and the sum is an upper bound on the forward's span.

Transports:
  direct    direct dispatch, then the routed expert, then direct combine
  overlap   direct dispatch, then the routed expert and combine_fabric2d as one program
  fabric2d  dispatch_fabric2d, then the routed expert and combine_fabric2d as one program

The overlap runs untraced here, which TtMoe only does for this kind of measurement: it is host-bound eager,
but the profiler times device work. The mesh opens with the payload a model does, half a token plus the
forwarding metadata, so both fabric2d ops send every token as two packets.

DSv4-Pro runs its own dims (384 experts / top-6 / 3072) on a host gate, since the device gate's grouped path
needs 256 experts; the gate is then not device time.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_y_device_params
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_moe import run_model
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import ROUTED_EXPERT_ACTIVATION_BY_NAME
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

_RECORD_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class _Model:
    label: str
    config: type
    gate: GateComputeMode
    capacity_factor: int
    extra: dict = field(default_factory=dict)


# GLM runs under the deepseek_v3 gate, as test_glm_moe does: its hand-built config carries no grouping. Its
# routed experts still take GLM's own hybrid threshold.
_GLM = _Model(
    "glm_5_3", GLM53Config, GateComputeMode.DEVICE_FP32, 8, dict(hybrid_token_threshold_model_config=GLM53Config)
)
_K2_7 = _Model("kimi_k2_7", KimiK27Config, GateComputeMode.DEVICE_FP32, 5)
_K3 = _Model(
    "kimi_k3",
    KimiK3Config,
    GateComputeMode.DEVICE_FP32,
    5,
    dict(
        routed_emb_dim=KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
        shared_hidden_dim=KimiK3Config.SHARED_EXPERT_INTERMEDIATE_SIZE,
        latent_use_norm=KimiK3Config.LATENT_MOE_USE_NORM,
        rms_norm_eps=KimiK3Config.RMS_NORM_EPS,
        routed_activation=ROUTED_EXPERT_ACTIVATION_BY_NAME[KimiK3Config.ROUTED_EXPERT_ACTIVATION],
        shared_activation=KimiK3Config.SHARED_EXPERT_ACTIVATION,
    ),
)
_DSV4_PRO = _Model(
    "deepseek_v4_pro",
    DeepSeekV4ProConfig,
    GateComputeMode.HOST_ALL,
    8,
    dict(
        hybrid_token_threshold_model_config=DeepSeekV4ProConfig,
        routed_activation=ROUTED_EXPERT_ACTIVATION_BY_NAME[DeepSeekV4ProConfig.ROUTED_EXPERT_ACTIVATION],
        shared_activation=DeepSeekV4ProConfig.SHARED_EXPERT_ACTIVATION,
        clamped_silu_glu_limit=DeepSeekV4ProConfig.SWIGLU_LIMIT,
    ),
)

_TRANSPORTS = {
    "direct": dict(dispatch_impl="direct", combine_impl="direct", overlap_routed_expert_with_combine=False),
    "overlap": dict(
        dispatch_impl="direct",
        combine_impl="direct",
        overlap_routed_expert_with_combine=True,
        overlap_routed_expert_with_combine_eager=True,
    ),
    "fabric2d": dict(
        dispatch_impl="fabric2d",
        combine_impl="fabric2d",
        overlap_routed_expert_with_combine=True,
        overlap_routed_expert_with_combine_eager=True,
    ),
}


def _cases():
    for variant, model in (
        ("deepseek_v3_d_p", _GLM),
        ("kimi_k2_7", _K2_7),
        ("kimi_k3", _K3),
        ("deepseek_v3_d_p", _DSV4_PRO),
    ):
        device_params = torus_y_device_params(fabric_payload_size=moe_fabric_payload(model.config))
        for transport in _TRANSPORTS:
            yield pytest.param(
                variant,
                model,
                transport,
                (8, 1),
                device_params,
                2,
                marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="ring"),
                id=f"{model.label}-{transport}-torus-y-8x1",
            )


def _op_of(kernel_sources) -> str:
    """The op directory a program's kernels come from, e.g. dispatch_fabric2d or hybrid_routed_expert_ffn."""
    for source in kernel_sources:
        for marker in ("deepseek_prefill/", "operations/"):
            if marker in source:
                return source.split(marker, 1)[1].split("/", 1)[0]
    return "other"


@pytest.mark.skipif(not is_blackhole(), reason="the routed expert / combine overlap is Blackhole-only")
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "variant, model, transport, mesh_device, device_params, num_links",
    list(_cases()),
    indirect=["variant", "mesh_device", "device_params"],
)
def test_moe_transport_perf_8x1(variant, model, transport, config_only, mesh_device, device_params, num_links, request):
    require_realtime_profiler(f"the {model.label} {transport} MoE perf")
    per_program = {}

    def measure(forward):
        # Warm-up pass, discarded: a cold forward's JIT compile lands inside the real-time records.
        warm = forward()
        ttnn.synchronize_device(mesh_device)
        del warm
        result, records = profile_realtime_program_merged(
            mesh_device, forward, record_timeout_seconds=_RECORD_TIMEOUT_S
        )
        per_program.update(records)
        return result

    run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        PREFILL_CHUNK_TOKENS_PER_CHIP,
        model.config.EMB_SIZE,
        model.config.MOE_INTERMEDIATE_SIZE,
        model.config.NUM_ROUTED_EXPERTS,
        model.config.NUM_EXPERTS_PER_TOKEN,
        model.capacity_factor,
        False,  # run_pcc_check
        num_links,
        per_axis_topology(device_params["fabric_config"]),
        model.gate,
        request,
        measure=measure,
        overlap_shared_expert_with_dispatch=False,
        **model.extra,
        **_TRANSPORTS[transport],
    )
    assert per_program, f"real-time profiler produced no program records for the {model.label} MoE forward"

    by_op = defaultdict(lambda: [0.0, 0])
    for entry in per_program.values():
        op = _op_of(entry["kernel_sources"])
        by_op[op][0] += entry["duration_ns"]
        by_op[op][1] += 1
    total_ns = sum(ns for ns, _ in by_op.values())
    breakdown = ", ".join(
        f"{op} {ns / 1e3:,.1f} us ({n})" for op, (ns, n) in sorted(by_op.items(), key=lambda kv: -kv[1][0])
    )
    logger.info(
        f"MOE_TRANSPORT_PERF {model.label} {transport}: {total_ns / 1e3:,.1f} us over {len(per_program)} programs "
        f"| {breakdown}"
    )
