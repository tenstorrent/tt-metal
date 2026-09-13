# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""SCRATCH: gpt-oss fused-vs-composite crossover, WITH the expert biases these FFNs carry.

The shared op runners cannot do this: run_single_routed_expert takes no bias arguments, so a
bias-free comparison is the only thing they can produce -- and the model config records that as the
wrong basis, because biasing costs the composite ~30% at 512 against the fused op's ~8%. So both
sides are driven through TtRoutedExpert instead, which carries torch_biases.

One module per op rather than one hybrid module: threshold >= max_tokens is fused-only, threshold
None is composite-only, so each dispatch contains exactly one of the two ops and the profiler can
attribute the time by kernel directory.
"""

import os
import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.reference.gpt_oss_120b_config import GptOss120BConfig
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_p150
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

_RE_DIR = "/unified_routed_expert_ffn/"
_SWIGLU_DIR = "/moe_fused_swiglu/"
_ITERS = 3
_MAX_TOKENS = 5120
_ACTIVES = [int(a) for a in os.environ.get("GX_ACTIVES", "128,256,512,768,1024,2048,4096,5120").split(",")]
EMB, HIDDEN = GptOss120BConfig.EMB_SIZE, GptOss120BConfig.MOE_INTERMEDIATE_SIZE


def _build(mesh_device, threshold):
    torch.manual_seed(42)
    weights = [
        {
            "gate_proj": torch.randn(HIDDEN, EMB, dtype=torch.float32) * 0.02,
            "up_proj": torch.randn(HIDDEN, EMB, dtype=torch.float32) * 0.02,
            "down_proj": torch.randn(EMB, HIDDEN, dtype=torch.float32) * 0.02,
        }
    ]
    biases = [
        {
            "gate_proj_bias": torch.randn(HIDDEN, dtype=torch.float32) * 0.05,
            "up_proj_bias": torch.randn(HIDDEN, dtype=torch.float32) * 0.05,
            "down_proj_bias": torch.randn(EMB, dtype=torch.float32) * 0.05,
        }
    ]

    def idx(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.uint32
        )

    expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=1,
        global_expert_idx_table=idx([0]),
        emb_dim=EMB,
        hidden_dim=HIDDEN,
        max_tokens=_MAX_TOKENS,
        torch_weights=weights,
        torch_biases=biases,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=ttnn.RoutedExpertActivation.SwiGluOai,
        hybrid_token_threshold=threshold,
    )
    x = ttnn.from_torch(
        torch.randn(_MAX_TOKENS, EMB, dtype=torch.float32),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    return expert, x, idx


def _measure_ns(mesh_device, fn, kernel_dir):
    def run_all():
        for _ in range(_ITERS):
            fn()

    _, per_program = profile_realtime_program_merged(mesh_device, run_all)
    matched = [
        e["duration_ns"]
        for e in per_program.values()
        if any(kernel_dir in s.replace("\\", "/") for s in e["kernel_sources"])
    ]
    assert len(matched) == _ITERS, f"expected {_ITERS} programs matching {kernel_dir}, got {len(matched)}"
    return statistics.median(matched)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")],
    indirect=True,
)
@pytest.mark.parametrize(
    "op, threshold", [pytest.param("fused", _MAX_TOKENS, id="fused"), pytest.param("composite", None, id="composite")]
)
@pytest.mark.skipif(
    not os.environ.get("RE_GPTOSS_XOVER"),
    reason="measurement harness, not a gate; opt in with RE_GPTOSS_XOVER=1",
)
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="both routed-expert fused paths are Blackhole-only")
@pytest.mark.skipif(not is_p150(), reason="timings are P150-specific")
@pytest.mark.timeout(0)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing.")
def test_gptoss_xover(mesh_device, op, threshold):
    require_realtime_profiler("gpt-oss crossover")
    expert, x, idx = _build(mesh_device, threshold)
    kernel_dir = _SWIGLU_DIR if op == "fused" else _RE_DIR
    for active in _ACTIVES:
        counts, offsets = idx([active]), idx([0])
        try:
            ns = _measure_ns(mesh_device, lambda: expert(x, counts, offsets), kernel_dir)
            logger.info(f"GX {op} gptoss {EMB}x{HIDDEN} isl-{active}: {ns / 1000:.1f} us")
        except Exception as exc:
            logger.error(f"GX {op} gptoss isl-{active}: FAILED {type(exc).__name__}: {str(exc)[:120]}")
