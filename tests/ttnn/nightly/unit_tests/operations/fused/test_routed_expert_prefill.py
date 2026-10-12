# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Prefill-shape probes for unified_routed_expert_moe.

These tests are intentionally non-gating: they log correctness and latency for
agents/engineers optimizing the op, but they do not assert thresholds.

One case is ONE routed expert of a production model, on ONE chip, fed ``num_tokens`` tokens.
Those two axes are the only things the op's caller does not choose: the model fixes the expert
(its K and N, its activation, whether it carries biases, the precision it is served at) and the
sequence being prefilled fixes how many tokens reach it. Everything else is the
implementation's own business and is absent here -- there is no expert-count, no dispatch
buffer larger than the expert's own tokens, and so no region offset, token-count vector or
local->global expert map, none of which carries information when one expert owns the buffer.
Weight placement comes from routed_expert_prefill_tensor_setup, which the op implementor owns.
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
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import apply_glu_activation
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import (
    COMPUTE_KERNEL_CONFIG_LOFI,
    ROUTED_EXPERT_ACTIVATION_BY_NAME,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.nightly.unit_tests.operations.fused.routed_expert_prefill_tensor_setup import (
    TensorSetup,
    routed_expert_prefill_tensor_setup,
)

_WARMUP_ITERS = 3
_MEASURE_ITERS = 10
_WEIGHT_SCALE = 0.02

# The model configs are torch-only, so they name their precisions as strings; this is where the
# names become ttnn dtypes, with the bytes/element each one costs on device for the bandwidth
# figure in the log line.
_DTYPE_BY_NAME = {
    "bfloat4_b": ttnn.bfloat4_b,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat16": ttnn.bfloat16,
    "float32": ttnn.float32,
}
_BYTES_PER_ELEM = {
    ttnn.bfloat4_b: 576 / 1024,  # 512 B mantissas + 64 B shared exponents per 32x32 tile
    ttnn.bfloat8_b: 1088 / 1024,  # 1024 B mantissas + 64 B shared exponents per 32x32 tile
    ttnn.bfloat16: 2.0,
    ttnn.float32: 4.0,
}


@dataclass(frozen=True)
class RoutedExpertCase:
    """One production model's routed expert. Every field of it is read off the model config."""

    model_cfg: type

    @property
    def emb_dim(self) -> int:
        # The expert's K/N: LatentMoE (Kimi-K3) projects 7168 -> 3584 before the experts and names
        # the projected width; every other model feeds them the embedding width. Same rule as
        # modeling_kimi_moe's moe_hidden_size and tt_prefill_block's routed_emb_dim.
        return getattr(self.model_cfg, "ROUTED_EXPERT_HIDDEN_SIZE", None) or self.model_cfg.EMB_SIZE

    @property
    def hidden_dim(self) -> int:
        return self.model_cfg.MOE_INTERMEDIATE_SIZE

    @property
    def torch_activation(self) -> str:
        return self.model_cfg.ROUTED_EXPERT_ACTIVATION

    @property
    def activation(self) -> "ttnn.RoutedExpertActivation":
        # The config names the activation once; the kernel enum and the torch reference are both
        # keyed by that one name, so the two cannot drift apart.
        return ROUTED_EXPERT_ACTIVATION_BY_NAME[self.torch_activation]

    @property
    def has_bias(self) -> bool:
        return self.model_cfg.ROUTED_EXPERT_BIAS

    @property
    def weights_dtype(self) -> ttnn.DataType:
        return _DTYPE_BY_NAME[self.model_cfg.ROUTED_EXPERT_WEIGHTS_DTYPE]

    @property
    def bias_dtype(self) -> ttnn.DataType:
        return _DTYPE_BY_NAME[self.model_cfg.ROUTED_EXPERT_BIAS_DTYPE]


def _case(model_cfg: type, name: str):
    case = RoutedExpertCase(model_cfg)
    return pytest.param(case, id=f"{name}-h{case.emb_dim}-i{case.hidden_dim}")


_CASES = [
    _case(DeepSeekV3Config, "deepseek-v3"),
    _case(KimiK27Config, "kimi-k2-7"),
    _case(KimiK3Config, "kimi-k3-latent"),
    _case(GLM53Config, "glm-5-3"),
    _case(MistralSmall4Config, "mistral-small-4"),
    _case(DeepSeekV4FlashConfig, "deepseek-v4-flash"),
    _case(DeepSeekV4ProConfig, "deepseek-v4-pro"),
    _case(MiniMaxM3Config, "minimax-m3"),
    _case(GptOss120BConfig, "gpt-oss-120b"),
]

# Tokens landing on this one expert. Spans the range a prefill chunk puts on a single expert,
# from a nearly idle one to a heavily loaded one.
_NUM_TOKENS = [64, 128, 256, 384, 512, 768, 1024, 1280, 2048]


def _place(mesh_device, torch_tensor: torch.Tensor, dtype: ttnn.DataType, setup: TensorSetup) -> ttnn.Tensor:
    """One weight on the device: the model's precision, the op's placement."""
    host = ttnn.from_torch(torch_tensor, layout=setup.layout, dtype=dtype)
    return ttnn.to_device(host, mesh_device, memory_config=setup.memory_config)


def _torch_expert(x, w_gate, w_up, w_down, biases, case: RoutedExpertCase) -> torch.Tensor:
    gate, up = x @ w_gate, x @ w_up
    if biases is not None:
        gate, up = gate + biases[0], up + biases[1]
    activated = apply_glu_activation(
        gate,
        up,
        activation=case.torch_activation,
        situ_beta=getattr(case.model_cfg, "ACTIVATION_SITU_BETA", 1.0),
        situ_linear_beta=getattr(case.model_cfg, "ACTIVATION_SITU_LINEAR_BETA", None),
    )
    out = activated @ w_down
    return out + biases[2] if biases is not None else out


def _run_once(x, weights, biases, num_tokens: int, case: RoutedExpertCase) -> ttnn.Tensor:
    gate_proj, up_proj, down_proj = weights
    gate_bias, up_bias, down_bias = biases if biases is not None else (None, None, None)
    return ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        x,
        [gate_proj],
        [up_proj],
        [down_proj],
        max_dispatched_tokens_per_expert=num_tokens,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
        activation=case.activation,
        gate_biases=None if gate_bias is None else [gate_bias],
        up_biases=None if up_bias is None else [up_bias],
        down_biases=None if down_bias is None else [down_bias],
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("num_tokens", _NUM_TOKENS, ids=[f"{n}tok" for n in _NUM_TOKENS])
@pytest.mark.parametrize("seed", [0], ids=["seed0"])
def test_routed_expert_prefill(mesh_device, case: RoutedExpertCase, num_tokens: int, seed):
    """Correctness + perf logging for one routed expert of ``case.model_cfg`` fed ``num_tokens`` tokens.

    The buffer is the expert's tokens and nothing else. The token buffer is ROW_MAJOR bf16 -- the
    format the upstream dispatch produces, which is why it is the test's to prescribe; the weights
    are placed per routed_expert_prefill_tensor_setup at the precision the model config names.
    """
    torch.manual_seed(seed)
    emb_dim, hidden_dim = case.emb_dim, case.hidden_dim
    assert num_tokens % ttnn.TILE_SIZE == 0, "the op's buffer M is tile-aligned"
    setup = routed_expert_prefill_tensor_setup(mesh_device, emb_dim, hidden_dim)

    torch_input = torch.randn(num_tokens, emb_dim, dtype=torch.bfloat16)
    w_gate = torch.randn(emb_dim, hidden_dim) * _WEIGHT_SCALE
    w_up = torch.randn(emb_dim, hidden_dim) * _WEIGHT_SCALE
    w_down = torch.randn(hidden_dim, emb_dim) * _WEIGHT_SCALE
    torch_biases = None
    if case.has_bias:
        torch_biases = (
            torch.randn(1, hidden_dim) * _WEIGHT_SCALE,
            torch.randn(1, hidden_dim) * _WEIGHT_SCALE,
            torch.randn(1, emb_dim) * _WEIGHT_SCALE,
        )
    torch_reference = _torch_expert(torch_input.float(), w_gate, w_up, w_down, torch_biases, case)

    x = ttnn.to_device(
        ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16),
        mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    weights = (
        _place(mesh_device, w_gate, case.weights_dtype, setup.gate_proj),
        _place(mesh_device, w_up, case.weights_dtype, setup.up_proj),
        _place(mesh_device, w_down, case.weights_dtype, setup.down_proj),
    )
    biases = (
        None
        if torch_biases is None
        else (
            _place(mesh_device, torch_biases[0], case.bias_dtype, setup.gate_bias),
            _place(mesh_device, torch_biases[1], case.bias_dtype, setup.up_bias),
            _place(mesh_device, torch_biases[2], case.bias_dtype, setup.down_bias),
        )
    )
    ttnn.synchronize_device(mesh_device)

    output = None
    for _ in range(_WARMUP_ITERS):
        output = _run_once(x, weights, biases, num_tokens, case)
    ttnn.synchronize_device(mesh_device)

    start = time.perf_counter()
    for _ in range(_MEASURE_ITERS):
        output = _run_once(x, weights, biases, num_tokens, case)
    ttnn.synchronize_device(mesh_device)
    elapsed_s = time.perf_counter() - start

    tt_output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
    _, pcc_msg = comp_pcc(torch_reference, tt_output)
    max_abs = torch.max(torch.abs(torch_reference - tt_output)).item()
    mean_abs = torch.mean(torch.abs(torch_reference - tt_output)).item()
    avg_ms = elapsed_s * 1000.0 / _MEASURE_ITERS

    tflops = 2 * 3 * num_tokens * emb_dim * hidden_dim / (avg_ms * 1e-3) / 1e12
    weight_bytes = 3 * emb_dim * hidden_dim * _BYTES_PER_ELEM[case.weights_dtype]
    weight_gbps = weight_bytes / (avg_ms * 1e-3) / 1e9

    logger.info(
        "unified_routed_expert_moe prefill probe "
        f"model={case.model_cfg.__name__} activation={case.activation} bias={case.has_bias} "
        f"num_tokens={num_tokens} x_shape={(num_tokens, emb_dim)} "
        f"gate_up_shape={(emb_dim, hidden_dim)} down_shape={(hidden_dim, emb_dim)} "
        f"weights_dtype={case.weights_dtype} "
        f"avg_ms={avg_ms:.3f} tflops={tflops:.2f} weight_gbps={weight_gbps:.1f} "
        f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} {pcc_msg}"
    )
