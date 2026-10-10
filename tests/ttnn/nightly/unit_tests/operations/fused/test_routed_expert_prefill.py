# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Prefill-shape probes for unified_routed_expert_moe.

These tests are intentionally non-gating: they log correctness and latency for
agents/engineers optimizing the op, but they do not assert thresholds.

Each case is the per-chip call one prefill MoE layer makes for a production
model, run on a single chip: the routed-expert op owns no CCL, so one chip
sees exactly the production local shapes once the mesh-level constants
(experts_per_chip, dispatch buffer rows, max tokens per expert) are derived the
way compute_constants does for the production mesh.
"""

import time
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.gpt_oss_120b_config import GptOss120BConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.reference.minimax_m3_config import MiniMaxM3Config
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import (
    ACTIVATION_CLAMPED_SILU_GLU,
    ACTIVATION_SILU,
    ACTIVATION_SITU,
    ACTIVATION_SWIGLUOAI,
    apply_glu_activation,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import (
    COMPUTE_KERNEL_CONFIG_LOFI,
    routed_expert_weight_memory_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

_WARMUP_ITERS = 3
_MEASURE_ITERS = 10
_WEIGHT_SCALE = 0.02
_WEIGHTS_DTYPE = ttnn.bfloat4_b
_BFP4_BYTES_PER_ELEM = 576 / 1024  # 512 B mantissas + 64 B shared exponents per 32x32 tile

_TORCH_ACTIVATION = {
    ttnn.RoutedExpertActivation.Silu: ACTIVATION_SILU,
    ttnn.RoutedExpertActivation.SituGlu: ACTIVATION_SITU,
    ttnn.RoutedExpertActivation.SwiGluOai: ACTIVATION_SWIGLUOAI,
    ttnn.RoutedExpertActivation.ClampedSiluGlu: ACTIVATION_CLAMPED_SILU_GLU,
}


@dataclass(frozen=True)
class RoutedExpertCase:
    model_cfg: type
    emb_dim: int
    activation: "ttnn.RoutedExpertActivation"
    has_bias: bool
    mesh_shape: tuple
    chunk_size: int
    capacity_factor: int

    @property
    def num_devices(self) -> int:
        return self.mesh_shape[0] * self.mesh_shape[1]

    @property
    def dispatch_group_size(self) -> int:
        return self.mesh_shape[0]

    @property
    def hidden_dim(self) -> int:
        return self.model_cfg.MOE_INTERMEDIATE_SIZE

    @property
    def experts_per_chip(self) -> int:
        return self.model_cfg.NUM_ROUTED_EXPERTS // self.num_devices

    @property
    def max_tokens_per_expert(self) -> int:
        return self.dispatch_group_size * (self.chunk_size // self.dispatch_group_size)

    @property
    def dispatch_buffer_rows(self) -> int:
        raw = self.max_tokens_per_expert * self.capacity_factor
        return raw + ttnn.TILE_SIZE * (min(raw, self.experts_per_chip) - 1)


# The DeepSeek-family prefill runner (models/demos/common/prefill/runners/prefill_runner.py) defaults to an
# (8, 4) SP x TP mesh, 5120-token chunks and capacity factor 8. MiniMax-M3 and GPT-OSS have their own runtimes.
_DS_MESH, _DS_CHUNK, _DS_CF = (8, 4), 5120, 8
_SILU = ttnn.RoutedExpertActivation.Silu

_CASES = [
    pytest.param(
        RoutedExpertCase(DeepSeekV3Config, DeepSeekV3Config.EMB_SIZE, _SILU, False, _DS_MESH, _DS_CHUNK, _DS_CF),
        id="deepseek-v3-h7168-i2048",
    ),
    pytest.param(
        RoutedExpertCase(KimiK27Config, KimiK27Config.EMB_SIZE, _SILU, False, _DS_MESH, _DS_CHUNK, _DS_CF),
        id="kimi-k2-7-h7168-i2048",
    ),
    pytest.param(
        # LatentMoE: the routed experts run on the 7168 -> 3584 projected hidden.
        RoutedExpertCase(
            KimiK3Config,
            KimiK3Config.ROUTED_EXPERT_HIDDEN_SIZE,
            ttnn.RoutedExpertActivation.SituGlu,
            False,
            _DS_MESH,
            _DS_CHUNK,
            _DS_CF,
        ),
        id="kimi-k3-latent-h3584-i3072",
    ),
    pytest.param(
        RoutedExpertCase(GLM53Config, GLM53Config.EMB_SIZE, _SILU, False, _DS_MESH, _DS_CHUNK, _DS_CF),
        id="glm-5-3-h6144-i2048",
    ),
    pytest.param(
        RoutedExpertCase(MistralSmall4Config, MistralSmall4Config.EMB_SIZE, _SILU, False, _DS_MESH, _DS_CHUNK, _DS_CF),
        id="mistral-small-4-h4096-i2048",
    ),
    pytest.param(
        RoutedExpertCase(
            DeepSeekV4FlashConfig,
            DeepSeekV4FlashConfig.EMB_SIZE,
            ttnn.RoutedExpertActivation.ClampedSiluGlu,
            False,
            _DS_MESH,
            _DS_CHUNK,
            _DS_CF,
        ),
        id="deepseek-v4-flash-h4096-i2048",
    ),
    pytest.param(
        RoutedExpertCase(
            DeepSeekV4ProConfig,
            DeepSeekV4ProConfig.EMB_SIZE,
            ttnn.RoutedExpertActivation.ClampedSiluGlu,
            False,
            _DS_MESH,
            _DS_CHUNK,
            _DS_CF,
        ),
        id="deepseek-v4-pro-h7168-i3072",
    ),
    pytest.param(
        # models/demos/minimax_m3: (8, 4) mesh, 5120-token chunks, capacity factor = top-k (drop-free buffer).
        RoutedExpertCase(
            MiniMaxM3Config,
            MiniMaxM3Config.EMB_SIZE,
            ttnn.RoutedExpertActivation.SwiGluOai,
            False,
            (8, 4),
            5120,
            MiniMaxM3Config.NUM_EXPERTS_PER_TOKEN,
        ),
        id="minimax-m3-h6144-i3072",
    ),
    pytest.param(
        # models/demos/gpt_oss_d_p: (4, 8) mesh, 8192-token chunks, capacity factor 2, per-expert biases.
        RoutedExpertCase(
            GptOss120BConfig,
            GptOss120BConfig.EMB_SIZE,
            ttnn.RoutedExpertActivation.SwiGluOai,
            True,
            (4, 8),
            8192,
            2,
        ),
        id="gpt-oss-120b-h2880-i2880",
    ),
]


def _route_tokens(case: RoutedExpertCase, num_chips: int) -> list[list[int]]:
    """Per-chip, per-local-expert token counts from uniform top-k routing of one dispatch group's chunk.

    Chip d holds global experts [d * experts_per_chip, (d + 1) * experts_per_chip).
    """
    num_experts = case.model_cfg.NUM_ROUTED_EXPERTS
    epc = case.experts_per_chip
    scores = torch.rand(case.max_tokens_per_expert, num_experts)
    topk_ids = torch.topk(scores, case.model_cfg.NUM_EXPERTS_PER_TOKEN, dim=-1).indices
    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts).tolist()
    return [counts[d * epc : (d + 1) * epc] for d in range(num_chips)]


def _region_offsets(counts: list[int]) -> list[int]:
    """Tile-aligned region starts, as offset_cumsum lays them out."""
    offsets, cursor = [], 0
    for count in counts:
        offsets.append(cursor)
        cursor += -(-count // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    return offsets


def _tile_ceil(n: int) -> int:
    return -(-n // ttnn.TILE_SIZE) * ttnn.TILE_SIZE


def _shard_per_device(
    mesh_device, per_device: torch.Tensor, layout, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
) -> ttnn.Tensor:
    """Place per_device[d] on chip d with the leading device dim squeezed, so local shapes match production."""
    host = ttnn.from_torch(
        per_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        layout=layout,
        dtype=dtype,
    )
    return ttnn.to_device(ttnn.squeeze(host, dim=0), mesh_device, memory_config=memory_config)


def _make_uint32_vectors(mesh_device, values: torch.Tensor) -> ttnn.Tensor:
    return _shard_per_device(mesh_device, values.to(torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)


def _make_weights(mesh_device, torch_weights: torch.Tensor) -> ttnn.Tensor:
    memory_config = routed_expert_weight_memory_config(mesh_device, torch_weights.shape[-1], dram_nd_sharded=True)
    return _shard_per_device(mesh_device, torch_weights, ttnn.TILE_LAYOUT, _WEIGHTS_DTYPE, memory_config)


def _make_biases(mesh_device, torch_biases: torch.Tensor) -> ttnn.Tensor:
    return _shard_per_device(mesh_device, torch_biases.unsqueeze(1), ttnn.TILE_LAYOUT, ttnn.bfloat16)


def _torch_expert(x, w_gate, w_up, w_down, biases, activation) -> torch.Tensor:
    gate, up = x @ w_gate, x @ w_up
    if biases is not None:
        gate, up = gate + biases[0], up + biases[1]
    activated = apply_glu_activation(
        gate,
        up,
        activation=_TORCH_ACTIVATION[activation],
        situ_beta=KimiK3Config.ACTIVATION_SITU_BETA,
        situ_linear_beta=KimiK3Config.ACTIVATION_SITU_LINEAR_BETA,
    )
    out = activated @ w_down
    return out + biases[2] if biases is not None else out


def _run_once(x, offsets, counts, idx_table, weights, biases, case: RoutedExpertCase) -> ttnn.Tensor:
    gate_projs, up_projs, down_projs = weights
    gate_biases, up_biases, down_biases = biases if biases is not None else (None, None, None)
    return ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        x,
        offsets,
        counts,
        idx_table,
        gate_projs,
        up_projs,
        down_projs,
        max_dispatched_tokens_per_expert=case.max_tokens_per_expert,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
        activation=case.activation,
        gate_biases=gate_biases,
        up_biases=up_biases,
        down_biases=down_biases,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("seed", [0], ids=["seed0"])
def test_routed_expert_prefill(mesh_device, case: RoutedExpertCase, seed):
    """Run correctness + perf logging for one chip's routed-expert call in a production prefill MoE layer.

    The op has no CCL, so a single chip runs the production-shaped call: its local experts, routing
    counts and dispatch buffer match one chip of the production mesh in ``case``. A larger mesh_device
    also works: each chip then gets its own experts, counts and buffer. The dispatch buffer is ROW_MAJOR bf16 (the dispatch output), weights are bf4 DRAM
    ND-sharded (the Blackhole default) and the compute config is the routed-expert LoFi default, so the
    op takes the same fused tilize path it takes in production. Token counts come from uniform top-k
    routing of the dispatch group's chunk; every expert runs on unified_routed_expert_moe (no hybrid split).
    """
    torch.manual_seed(seed)
    num_chips = mesh_device.get_num_devices()
    emb_dim, hidden_dim, epc = case.emb_dim, case.hidden_dim, case.experts_per_chip
    num_experts = case.model_cfg.NUM_ROUTED_EXPERTS
    assert num_chips * epc <= num_experts

    chip_counts = _route_tokens(case, num_chips)
    chip_offsets = [_region_offsets(counts) for counts in chip_counts]
    used_rows = max(offs[-1] + _tile_ceil(counts[-1]) for offs, counts in zip(chip_offsets, chip_counts))
    assert used_rows <= case.dispatch_buffer_rows

    torch_input = torch.zeros(num_chips, case.dispatch_buffer_rows, emb_dim, dtype=torch.bfloat16)
    for d in range(num_chips):
        for offset, count in zip(chip_offsets[d], chip_counts[d]):
            torch_input[d, offset : offset + count] = torch.randn(count, emb_dim, dtype=torch.bfloat16)

    gate_projs, up_projs, down_projs = [], [], []
    gate_biases, up_biases, down_biases = [], [], []
    torch_reference = torch.zeros(num_chips, used_rows, emb_dim, dtype=torch.float32)
    for e in range(epc):
        w_gate = torch.randn(num_chips, emb_dim, hidden_dim) * _WEIGHT_SCALE
        w_up = torch.randn(num_chips, emb_dim, hidden_dim) * _WEIGHT_SCALE
        w_down = torch.randn(num_chips, hidden_dim, emb_dim) * _WEIGHT_SCALE
        b_gate = b_up = b_down = None
        if case.has_bias:
            b_gate = torch.randn(num_chips, hidden_dim) * _WEIGHT_SCALE
            b_up = torch.randn(num_chips, hidden_dim) * _WEIGHT_SCALE
            b_down = torch.randn(num_chips, emb_dim) * _WEIGHT_SCALE
            gate_biases.append(_make_biases(mesh_device, b_gate))
            up_biases.append(_make_biases(mesh_device, b_up))
            down_biases.append(_make_biases(mesh_device, b_down))
        for d in range(num_chips):
            offset, count = chip_offsets[d][e], chip_counts[d][e]
            expert_biases = (b_gate[d], b_up[d], b_down[d]) if case.has_bias else None
            torch_reference[d, offset : offset + count] = _torch_expert(
                torch_input[d, offset : offset + count].float(),
                w_gate[d],
                w_up[d],
                w_down[d],
                expert_biases,
                case.activation,
            )
        gate_projs.append(_make_weights(mesh_device, w_gate))
        up_projs.append(_make_weights(mesh_device, w_up))
        down_projs.append(_make_weights(mesh_device, w_down))
    weights = (gate_projs, up_projs, down_projs)
    biases = (gate_biases, up_biases, down_biases) if case.has_bias else None

    x = _shard_per_device(mesh_device, torch_input, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)
    full_counts = torch.zeros(num_chips, 1, num_experts, dtype=torch.int32)
    full_offsets = torch.zeros(num_chips, 1, num_experts, dtype=torch.int32)
    idx_table = torch.zeros(num_chips, epc, dtype=torch.int32)
    for d in range(num_chips):
        global_ids = slice(d * epc, (d + 1) * epc)
        full_counts[d, 0, global_ids] = torch.tensor(chip_counts[d])
        full_offsets[d, 0, global_ids] = torch.tensor(chip_offsets[d])
        idx_table[d] = torch.arange(d * epc, (d + 1) * epc)
    counts = _make_uint32_vectors(mesh_device, full_counts)
    offsets = _make_uint32_vectors(mesh_device, full_offsets)
    idx_table = _make_uint32_vectors(mesh_device, idx_table)
    ttnn.synchronize_device(mesh_device)

    output = None
    for _ in range(_WARMUP_ITERS):
        output = _run_once(x, offsets, counts, idx_table, weights, biases, case)
    ttnn.synchronize_device(mesh_device)

    start = time.perf_counter()
    for _ in range(_MEASURE_ITERS):
        output = _run_once(x, offsets, counts, idx_table, weights, biases, case)
    ttnn.synchronize_device(mesh_device)
    elapsed_s = time.perf_counter() - start

    tt_output = ttnn.to_torch(
        ttnn.slice(output, [0, 0], [used_rows, emb_dim]),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    ).float()
    tt_output = tt_output.reshape(num_chips, used_rows, emb_dim)
    ref_rows, tt_rows = [], []
    for d in range(num_chips):
        active = torch.cat([torch.arange(o, o + c) for o, c in zip(chip_offsets[d], chip_counts[d])])
        ref_rows.append(torch_reference[d, active])
        tt_rows.append(tt_output[d, active])
    ref_active, tt_active = torch.cat(ref_rows), torch.cat(tt_rows)
    _, pcc_msg = comp_pcc(ref_active, tt_active)
    max_abs = torch.max(torch.abs(ref_active - tt_active)).item()
    mean_abs = torch.mean(torch.abs(ref_active - tt_active)).item()
    avg_ms = elapsed_s * 1000.0 / _MEASURE_ITERS

    tokens_per_chip = [sum(counts) for counts in chip_counts]
    all_counts = [c for counts in chip_counts for c in counts]
    active_experts_per_chip = sum(1 for c in all_counts if c > 0) / num_chips
    tflops_per_chip = 2 * 3 * sum(tokens_per_chip) / num_chips * emb_dim * hidden_dim / (avg_ms * 1e-3) / 1e12
    weight_gbps_per_chip = (
        active_experts_per_chip * 3 * emb_dim * hidden_dim * _BFP4_BYTES_PER_ELEM / (avg_ms * 1e-3) / 1e9
    )

    logger.info(
        "unified_routed_expert_moe prefill probe "
        f"model={case.model_cfg.__name__} activation={case.activation} bias={case.has_bias} "
        f"prod_mesh={case.mesh_shape} chunk={case.chunk_size} capacity_factor={case.capacity_factor} "
        f"test_mesh={tuple(mesh_device.shape)} experts_per_chip={epc} "
        f"local_x_shape={(case.dispatch_buffer_rows, emb_dim)} "
        f"local_gate_up_shape={(emb_dim, hidden_dim)} local_down_shape={(hidden_dim, emb_dim)} "
        f"max_tokens_per_expert={case.max_tokens_per_expert} tokens_per_chip={tokens_per_chip} "
        f"counts_min={min(all_counts)} counts_max={max(all_counts)} "
        f"avg_ms={avg_ms:.3f} tflops_per_chip={tflops_per_chip:.2f} weight_gbps_per_chip={weight_gbps_per_chip:.1f} "
        f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} {pcc_msg}"
    )
