# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Integration test for `models.common.modules.moe.tt_moe_decode.TTMoEDecode`.

Setup mirrors `test_optimized_moe_decode_block.py`: build torch weights / inputs,
push them through the TTMoEDecode module, and verify the final output against a
torch reference. Combine output verification is intentionally skipped — that
intermediate buffer is exercised by the optimized-block test directly.

Parametrized over every YAML model config in `models/common/modules/moe/configs/`.
"""

from __future__ import annotations

import faulthandler
import os
import random
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from loguru import logger
from ttnn.operations.ccl import MoEActivationFunction

import ttnn
from models.common.modules.moe.tt_moe_decode import TTMoEDecode
from models.common.modules.moe.tt_moe_decode_config import TTMoEDecodeConfig
from models.common.utility_functions import is_blackhole
from models.perf.benchmarking_utils import BenchmarkProfiler

try:
    from tracy import signpost
except ImportError:  # tracy is only available in profiling builds; perf signposts are best-effort

    def signpost(*_args, **_kwargs):
        pass


from models.demos.deepseek_v3.tests.fused_op_unit_tests.moe.test_optimized_moe_decode_block import (
    create_torch_dispatch_input_expert_scores_tensor,
    create_torch_dispatch_input_tensor,
    verify_output,
)
from tests.nightly.tg.ccl.moe.test_moe_compute_6U import _swiglu_reference

faulthandler.enable()


# occasionally running this test hangs due to conflicts with mesh device teardown. Enable this fixture if encountered
@pytest.fixture(autouse=False)
def _hang_watchdog():
    faulthandler.dump_traceback_later(300, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def _print_exception_and_fail(reason: str) -> None:
    """Print the active exception to the original (uncaptured) stderr and pytest.fail.

    pytest's stderr capture buffers `logger.exception()` output and only flushes it
    after the test exits. When the watchdog `_exit()`s the process (e.g. because
    a ttnn tensor `__repr__` was waiting on a wedged device), that buffer is lost.
    Writing to `sys.__stderr__` and flushing bypasses capture so the trace survives.
    Then pytest.fail(pytrace=False) skips pytest's own saferepr-driven traceback
    rendering, which is itself prone to deadlocking on hung device tensors.
    """
    traceback.print_exc(file=sys.__stderr__)
    sys.__stderr__.flush()
    pytest.fail(reason, pytrace=False)


MESH_GRAPH_DESC_16x1 = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_16x1_torus_graph_descriptor.textproto"
)
MESH_GRAPH_DESC_BH_LB_8x1 = "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_lb_8x1_line_graph_descriptor.textproto"


def is_mesh_graph_descriptor_set(expected_path):
    """Check if TT_MESH_GRAPH_DESC_PATH is set to the expected path."""
    return os.environ.get("TT_MESH_GRAPH_DESC_PATH") == expected_path


# ---------------------------------------------------------------------------
# torch reference helpers
#
# `_swiglu_reference`, `create_torch_dispatch_input_tensor`,
# `create_torch_dispatch_input_expert_scores_tensor`, and `verify_output` are
# imported above from the existing MoE tests — same logic, no need to duplicate.
# Helpers that diverge (per-expert weight/bias init, activation/bias-aware
# matmul, output-golden assembly) are defined locally.
# ---------------------------------------------------------------------------


torch.set_num_threads(max(1, os.cpu_count() or 1))


@torch.no_grad()
def _matmul_golden(
    token: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation_type: MoEActivationFunction = MoEActivationFunction.SILU,
    b0: torch.Tensor | None = None,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> torch.Tensor:
    """MoE expert reference (num_layers=1 throughout).

    SILU:   `silu(x @ w0 + b0) * (x @ w1 + b1) @ w2 + b2`
    SWIGLU: `(up + 1) * gate * sigmoid(alpha * gate) @ w2 + b2` with clamping (GPT-OSS),
            where `gate = x @ w0 + b0` and `up = x @ w1 + b1`.
    GELU:   `gelu(x @ w0 + b0, tanh) * (x @ w1 + b1) @ w2 + b2` (tanh approximation
            matches the on-device kernel).

    Per-expert bias shapes: `b0`/`b1` are `[num_layers, 1, N]`, `b2` is
    `[num_layers, 1, hidden_size]`. `unsqueeze(-2)` broadcasts over the token dim.
    """

    _orig_dtype = token.dtype
    token = token.float()
    w0 = w0.float()
    w1 = w1.float()
    w2 = w2.float()

    gate = token @ w0
    if b0 is not None:
        gate = gate + b0.float().unsqueeze(-2)
    up = token @ w1
    if b1 is not None:
        up = up + b1.float().unsqueeze(-2)

    if activation_type == MoEActivationFunction.SILU:
        intermediate = torch.nn.functional.silu(gate) * up
    elif activation_type == MoEActivationFunction.SWIGLU:
        intermediate = _swiglu_reference(gate, up)
    elif activation_type == MoEActivationFunction.GELU:
        intermediate = torch.nn.functional.gelu(gate, approximate="tanh") * up
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")

    output = intermediate @ w2
    if b2 is not None:
        output = output + b2.float().unsqueeze(-2)
    return output.to(_orig_dtype)


def _create_per_expert_weights(num_layers: int, num_experts: int, h: int, n: int) -> torch.Tensor:
    """Returns a [num_layers, num_experts, h, n] tensor of expert weights with calibrated scale.

    TLDR: stabilize output statistics with random weights. Aiming for 0.987 PCC and ATOL < 20

    Weights are drawn from `U[-c, c]` with `c = sqrt(81/h)`. Reasoning:
    - Tokens are `U[-0.5, 0.5]` (Var = 1/12). `Var(matmul_out) = h * Var(token) * Var(w)`,
      so for output std ≈ 1.5 we want `Var(w) = 27/h`, i.e. `c = sqrt(81/h)` for uniform.
    - Why std ≈ 1.5 specifically: empirically the PCC sweet spot. Going smaller (std ≈ 1)
      pushes the bulk of gate/up values into the bf16 rounding floor → PCC drops. Going
      larger (std ≈ 2) compounds bf4 quantization noise through the three matmul cascade
      faster than it benefits from any silu-asymptotic stability → also drops. ~1.5
      threads the needle.
    - Uniform (not normal) is critical for bf4_b: bf4 quantization uses a shared exponent
      per 16-element block set by the block's max-abs. Uniform draws produce nearly
      identical block max-abs across blocks → consistent quantization step everywhere.
      Normal draws give some blocks fat-tailed maxes that crush the precision of their
      smaller siblings, injecting position-dependent noise that tanks PCC.

      To take advantage of this calibration set c = (81.0 / h) ** 0.5

      Note (AM): I have disabled this - c = 0.5 - until I am really confident that any PCC variance is benign

    `h` is the matmul input dim for all three of w0, w1, w2 (w2 is called with
    `h=intermediate_size`).
    """
    c = 0.5
    return ((torch.rand((num_layers, num_experts, h, n), dtype=torch.float32) - 0.5) * (2.0 * c)).to(torch.bfloat16)


def _create_per_expert_biases(num_layers: int, num_experts: int, dim: int) -> torch.Tensor:
    """Returns a [num_layers, num_experts, dim] tensor of expert biases.

    Variance matches the bias init in `test_moe_compute_6U.py` (std=0.12), but draws are
    uniform `U[-c, c]` (`c = sqrt(3) * std`) rather than normal. Reason: the bias row is
    packed into the same bf4_b tile as the weights, and bf4's per-block shared exponent
    is set by the block's max-abs. Normal draws produce fat-tail blocks that crush the
    quantization of their smaller siblings — same issue we fixed for weights. Uniform
    draws keep block max-abs consistent and the per-element quantization error tight.

    Bias still adds ~8% of the matmul output at the current scale, so any extra
    position-dependent noise on the bias channel shows up directly in PCC.
    """
    _bias_std = 0.12
    c = (3.0**0.5) * _bias_std
    return ((torch.rand(num_layers, num_experts, dim, dtype=torch.float32) - 0.5) * (2.0 * c)).to(torch.bfloat16)


def _create_expert_indices(batch: int, num_experts: int, select_k: int) -> torch.Tensor:
    """[batch, 1, 1, select_k] — random unique experts per token."""
    out = torch.full((batch, 1, 1, select_k), -1, dtype=torch.int32)
    for b in range(batch):
        for k, e in enumerate(random.sample(range(num_experts), select_k)):
            out[b, 0, 0, k] = e
    return out


def _create_balanced_expert_indices(batch: int, num_experts: int, select_k: int) -> torch.Tensor:
    """[batch, 1, 1, select_k] — each expert assigned the same token count (random, distinct per token).

    Perf-mode routing: the `batch * select_k` total token→expert assignments are spread as evenly as
    possible over `num_experts` (each expert gets `batch*select_k // num_experts`, and the remainder
    experts get one extra — so counts differ by at most 1). This gives a realistic, non-hotspotted
    load for the fabric/compute perf measurement rather than the arbitrary per-token sampling used for
    correctness. Assignments are random but each token still gets `select_k` *distinct* experts.
    """
    assert 1 <= select_k <= num_experts, f"select_k={select_k} out of range for num_experts={num_experts}"
    total = batch * select_k
    base, rem = divmod(total, num_experts)
    if rem:
        logger.info(
            f"Balanced routing not exact: {total} assignments over {num_experts} experts → counts {base}/{base+1}"
        )
    # Pool of expert ids with the target per-expert multiplicity, then dealt into tokens.
    pool = [e for e in range(num_experts) for _ in range(base + (1 if e < rem else 0))]
    random.shuffle(pool)
    grid = [pool[i * select_k : (i + 1) * select_k] for i in range(batch)]

    # Repair any token that drew a duplicate expert via count-preserving swaps with another token.
    for i in range(batch):
        guard = 0
        while len(set(grid[i])) != select_k:
            guard += 1
            assert guard < 1_000_000, "failed to construct a balanced distinct assignment"
            dup_pos = next(p for p in range(select_k) if grid[i].count(grid[i][p]) > 1)
            v = grid[i][dup_pos]
            j, q = random.randrange(batch), random.randrange(select_k)
            other = grid[j][q]
            # Swap is safe iff it removes the dup in row i without creating one in row j.
            if j == i or other == v or other in grid[i] or v in grid[j]:
                continue
            grid[i][dup_pos], grid[j][q] = other, v

    out = torch.full((batch, 1, 1, select_k), -1, dtype=torch.int32)
    for b in range(batch):
        for k, e in enumerate(grid[b]):
            out[b, 0, 0, k] = e
    return out


@torch.no_grad()
def _gen_output_golden(
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_scores: torch.Tensor,
    w0_per_expert: list[torch.Tensor],
    w1_per_expert: list[torch.Tensor],
    w2_per_expert: list[torch.Tensor],
    batch: int,
    hidden_size: int,
    select_k: int,
    activation_type: MoEActivationFunction = MoEActivationFunction.SILU,
    b0_per_expert: list[torch.Tensor] | None = None,
    b1_per_expert: list[torch.Tensor] | None = None,
    b2_per_expert: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """[batch, 1, 1, hidden_size] — sum_k(score_k * matmul(token, expert_k))."""
    out = torch.zeros((batch, 1, 1, hidden_size), dtype=torch.bfloat16)
    for t in range(batch):
        for k in range(select_k):
            e = expert_indices[t, 0, 0, k].item()
            contrib = _matmul_golden(
                tokens[t],
                w0_per_expert[e],
                w1_per_expert[e],
                w2_per_expert[e],
                activation_type,
                b0=b0_per_expert[e] if b0_per_expert is not None else None,
                b1=b1_per_expert[e] if b1_per_expert is not None else None,
                b2=b2_per_expert[e] if b2_per_expert is not None else None,
            )
            out[t] = out[t] + expert_scores[t, 0, 0, k] * contrib
    return out


def _create_shared_expert_weights(
    shared_expert_ids: list[int], num_layers: int, h: int, n: int, h2: int
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """`shared_id -> [num_layers, 1, ...]` tensors for w0/w1/w2.

    Matches the format `_TTMoEDecodeExpertState` / `add_shared_expert_weights` expect:
    each shared expert is stored individually keyed by its global id.
    """
    # Same calibrated uniform scaling as `_create_per_expert_weights`: bf4_b quantizes
    # uniform draws far more cleanly than normal draws (no per-block fat-tail outliers
    # → consistent quantization step). w0/w1 input dim is `h`, w2 input dim is `n`.
    # c_h = (81.0 / h) ** 0.5
    # c_n = (81.0 / n) ** 0.5

    # Note: I have disabled this, c = 0.5, until I am really confident that any PCC/ATOL variance is benign

    c_h = 0.5  # (81.0 / h) ** 0.5
    c_n = 0.5  # (81.0 / n) ** 0.5
    shared_w0 = {
        sid: ((torch.rand((num_layers, 1, h, n), dtype=torch.float32) - 0.5) * (2.0 * c_h)).to(torch.bfloat16)
        for sid in shared_expert_ids
    }
    shared_w1 = {
        sid: ((torch.rand((num_layers, 1, h, n), dtype=torch.float32) - 0.5) * (2.0 * c_h)).to(torch.bfloat16)
        for sid in shared_expert_ids
    }
    shared_w2 = {
        sid: ((torch.rand((num_layers, 1, n, h2), dtype=torch.float32) - 0.5) * (2.0 * c_n)).to(torch.bfloat16)
        for sid in shared_expert_ids
    }
    return shared_w0, shared_w1, shared_w2


@torch.no_grad()
def _add_shared_experts_to_golden(
    out: torch.Tensor,
    tokens: torch.Tensor,
    batch: int,
    shared_w0: dict[int, torch.Tensor],
    shared_w1: dict[int, torch.Tensor],
    shared_w2: dict[int, torch.Tensor],
    shared_expert_scale: float,
    activation_type: MoEActivationFunction,
) -> torch.Tensor:
    """Every token sees every shared expert; contributions add with a fixed scalar scale.

    Mirrors `deepseek_moe_fast_reduce_nc_fused`'s shared-expert behavior: no per-token
    score, just `shared_expert_scale` applied uniformly.
    """
    for sid in shared_w0:
        w0, w1, w2 = shared_w0[sid], shared_w1[sid], shared_w2[sid]
        for t in range(batch):
            contrib = _matmul_golden(tokens[t], w0, w1, w2, activation_type)
            out[t] = out[t] + shared_expert_scale * contrib
    return out


def _add_shared_experts_to_golden_tp(
    out: torch.Tensor,
    tokens: torch.Tensor,
    batch: int,
    shared_w0: dict[int, torch.Tensor],
    shared_w1: dict[int, torch.Tensor],
    shared_w2: dict[int, torch.Tensor],
    shared_expert_scale: float,
    activation_type: MoEActivationFunction,
    num_tp: int,
) -> torch.Tensor:
    """Shared-expert golden that mimics the tensor-parallel device path step-for-step.

    Each shared expert's intermediate dim `N` is partitioned into `num_tp` contiguous
    chunks — one per device along the TP axis (`1 - cluster_axis`). Each device's chunk is
    zero-padded back to full `N` (front block `[0:N/num_tp]` real, rest zero — exactly the
    layout `add_shared_expert_weights` produces: W0/W1 padded on the intermediate dim, W2
    on its row dim), the full-`N` FFN is run on the padded weights to get that device's
    partial, and the partials are summed (the reduce-scatter), then scaled by
    `shared_expert_scale`.

    Because SiLU/SwiGLU/GELU are column-separable and each device's W2 rows are zero outside
    its chunk, the summed partials equal the full FFN — so this matches
    `_add_shared_experts_to_golden` exactly (modulo bf16 accumulation order). It's written
    this way on purpose: it tracks what each device actually computes, so if the device
    output diverges from this golden the fault is in the kernel's handling of the
    zero-padded TP layout, not the decomposition. No `num_replicated` factor — the sum of
    disjoint partials is one full copy, not `num_tp` copies.
    """
    for sid in shared_w0:
        w0, w1, w2 = shared_w0[sid], shared_w1[sid], shared_w2[sid]
        n = w0.shape[-1]
        assert n % num_tp == 0, f"shared expert intermediate dim {n} not divisible by num_tp {num_tp}"
        chunk = n // num_tp
        for t in range(batch):
            partial_sum = torch.zeros_like(out[t])
            for d in range(num_tp):
                lo, hi = d * chunk, (d + 1) * chunk
                # Device d: its chunk, front-block zero-padded to full N. W0/W1 partition the
                # intermediate (last) dim; W2 partitions its row (second-to-last) dim.
                w0_d = torch.zeros_like(w0)
                w1_d = torch.zeros_like(w1)
                w2_d = torch.zeros_like(w2)
                w0_d[..., :chunk] = w0[..., lo:hi]
                w1_d[..., :chunk] = w1[..., lo:hi]
                w2_d[..., :chunk, :] = w2[..., lo:hi, :]
                partial_sum = partial_sum + _matmul_golden(tokens[t], w0_d, w1_d, w2_d, activation_type)
            out[t] = out[t] + shared_expert_scale * partial_sum
    return out


CONFIGS_DIR = Path(__file__).resolve().parents[3] / "modules" / "moe" / "configs"
CONFIG_PATHS = sorted(CONFIGS_DIR.glob("*.yaml"))

assert CONFIG_PATHS, f"no YAML configs found in {CONFIGS_DIR}"


def _config_id(path: Path) -> str:
    return path.stem


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

# known failures
# Note: it would be better to test all of these and let them fail but some cause hard crashes and derail the test
SKIP_LIST = [
    "ling_1t.yaml",
    "mistral_large_3.yaml",
    "deepseek_v4_pro.yaml",
]


# ---------------------------------------------------------------------------
# shared setup helpers (used by both the correctness and perf tests)
# ---------------------------------------------------------------------------


def _setup_decode(
    mesh_device: ttnn.MeshDevice,
    device_params: dict,
    config_path: Path,
    divide_k_for_slice: bool = False,
) -> SimpleNamespace:
    """Resolve config, build torch weights/biases, and construct the `TTMoEDecode` module.

    Returns a `SimpleNamespace` bundling the module, the resolved config, derived sizes, and
    everything `_make_dispatch_inputs` / `_make_golden` need. Applies the same skip logic as the
    correctness test (fabric/mesh compatibility, SKIP_LIST, sliceable mesh, shared-expert+bias).
    Both `test_tt_moe_decode` and `test_tt_moe_decode_perf` funnel through here so the setup lives
    in exactly one place.

    `divide_k_for_slice` (perf only): when the config's authored mesh is sliced down to the device
    mesh along the replicated axis, `select_experts_k` is divided by that slice degree — a token that
    would route to K experts across the full mesh only reaches K/degree of them on our slice (e.g.
    (8,4)→(8,1) with K=8 gives K=2). This is applied to the *raw* config data before the config object
    is built, so every K-derived field (memory configs, effective_experts_k, dispatch layout) derives
    consistently from the reduced K.
    """
    torch.manual_seed(2005)
    random.seed(2005)

    mesh_shape = tuple(mesh_device.shape)
    fabric_config = device_params["fabric_config"]

    if str(config_path.name) in SKIP_LIST:
        pytest.skip(f"{config_path} is a known failure")

    topology = ttnn.Topology.Ring if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING else ttnn.Topology.Linear
    raw_text = config_path.read_text()
    config = TTMoEDecodeConfig.from_yaml(raw_text, topology=topology)

    if divide_k_for_slice:
        other = 1 - config.cluster_axis
        slice_divisor = config.mesh_shape[other] // mesh_shape[other] if mesh_shape[other] else 0
        if slice_divisor > 1:
            if config.select_experts_k % slice_divisor != 0:
                pytest.skip(
                    f"select_experts_k={config.select_experts_k} not divisible by slice divisor {slice_divisor}"
                )
            new_k = config.select_experts_k // slice_divisor
            logger.info(
                f"Perf: reducing select_experts_k {config.select_experts_k} -> {new_k} "
                f"(slice divisor {slice_divisor} along axis {other})"
            )
            raw = yaml.safe_load(raw_text)
            raw["select_experts_k"] = new_k
            config = TTMoEDecodeConfig.from_yaml(yaml.safe_dump(raw), topology=topology)

    if config.mesh_shape != mesh_shape:
        try:
            config = config.with_mesh_shape(mesh_shape)
        except ValueError as e:
            pytest.skip(f"config mesh_shape {config.mesh_shape} can't slice to device mesh_shape {mesh_shape}: {e}")
        logger.info(f"Sliced config mesh_shape to {mesh_shape}; num_routed_experts={config.num_routed_experts}")

    # --- derived sizes (all from config) ---
    cluster_axis = config.cluster_axis
    routed_experts = config.num_routed_experts
    hidden_size = config.hidden_size
    intermediate_size = config.compute.intermediate_size
    select_experts_k = config.select_experts_k
    batches_per_device = config.batch_per_device

    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis]
    batch = batches_per_device * num_dispatch_devices

    shard_dim = 0
    shard_dims = (shard_dim, None) if cluster_axis == 0 else (None, shard_dim)

    logger.info(
        f"Setup [{config_path.stem}]: mesh_shape={mesh_shape} cluster_axis={cluster_axis} "
        f"num_devices={num_devices} batch={batch} hidden={hidden_size} N={intermediate_size} "
        f"routed_experts={routed_experts} select_experts_k={select_experts_k} "
        f"has_bias={config.has_bias} activation={config.compute.activation_type.name}"
    )

    # --- weights: [num_layers=1, routed_experts, H/N, N/H] ---
    num_layers = 1
    torch_w0 = _create_per_expert_weights(num_layers, routed_experts, hidden_size, intermediate_size)
    torch_w1 = _create_per_expert_weights(num_layers, routed_experts, hidden_size, intermediate_size)
    torch_w2 = _create_per_expert_weights(num_layers, routed_experts, intermediate_size, hidden_size)
    w0_per_expert = [torch_w0[:, e : e + 1, ...] for e in range(routed_experts)]
    w1_per_expert = [torch_w1[:, e : e + 1, ...] for e in range(routed_experts)]
    w2_per_expert = [torch_w2[:, e : e + 1, ...] for e in range(routed_experts)]

    # --- biases (optional): [num_layers=1, routed_experts, N or hidden_size] ---
    torch_b0 = torch_b1 = torch_b2 = None
    b0_per_expert = b1_per_expert = b2_per_expert = None
    if config.has_bias:
        torch_b0 = _create_per_expert_biases(num_layers, routed_experts, intermediate_size)
        torch_b1 = _create_per_expert_biases(num_layers, routed_experts, intermediate_size)
        torch_b2 = _create_per_expert_biases(num_layers, routed_experts, hidden_size)
        b0_per_expert = [torch_b0[:, e : e + 1, :] for e in range(routed_experts)]
        b1_per_expert = [torch_b1[:, e : e + 1, :] for e in range(routed_experts)]
        b2_per_expert = [torch_b2[:, e : e + 1, :] for e in range(routed_experts)]

    # --- shared experts (optional): id -> [num_layers, 1, ...] weight dicts ---
    shared_id_to_torch_w0 = shared_id_to_torch_w1 = shared_id_to_torch_w2 = None
    if config.num_shared_experts > 0:
        if config.has_bias:
            pytest.skip("TTMoEDecode does not yet support has_bias=True with shared experts")
        shared_expert_ids = sorted(config.experts.shared_expert_ids_to_devices.keys())
        shared_id_to_torch_w0, shared_id_to_torch_w1, shared_id_to_torch_w2 = _create_shared_expert_weights(
            shared_expert_ids, num_layers, hidden_size, intermediate_size, hidden_size
        )
        logger.info(
            f"Shared experts: {len(shared_expert_ids)} ids={shared_expert_ids} "
            f"scale={config.reduce.shared_expert_scale}"
        )

    # --- build module ---
    # Wrap in try/except + pytest.fail(pytrace=False) — pytest's pretty-traceback
    # saferepr'ing the deeply nested config/mesh args takes long enough that the
    # 300s faulthandler watchdog _exit()s the process before any output is shown.
    try:
        decode = TTMoEDecode(
            mesh_device=mesh_device,
            config=config,
            torch_w0=torch_w0,
            torch_w1=torch_w1,
            torch_w2=torch_w2,
            torch_b0=torch_b0,
            torch_b1=torch_b1,
            torch_b2=torch_b2,
            shared_id_to_torch_w0=shared_id_to_torch_w0,
            shared_id_to_torch_w1=shared_id_to_torch_w1,
            shared_id_to_torch_w2=shared_id_to_torch_w2,
        )
    except Exception as e:
        _print_exception_and_fail(f"TTMoEDecode init failed: {type(e).__name__}")

    logger.info("Module Setup complete")

    return SimpleNamespace(
        decode=decode,
        config=config,
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        cluster_axis=cluster_axis,
        shard_dims=shard_dims,
        batch=batch,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        routed_experts=routed_experts,
        select_experts_k=select_experts_k,
        w0_per_expert=w0_per_expert,
        w1_per_expert=w1_per_expert,
        w2_per_expert=w2_per_expert,
        b0_per_expert=b0_per_expert,
        b1_per_expert=b1_per_expert,
        b2_per_expert=b2_per_expert,
        shared_id_to_torch_w0=shared_id_to_torch_w0,
        shared_id_to_torch_w1=shared_id_to_torch_w1,
        shared_id_to_torch_w2=shared_id_to_torch_w2,
    )


def _make_dispatch_inputs(ctx: SimpleNamespace, balanced: bool = False):
    """One iteration of on-device dispatch inputs. Returns `(tokens, indices, scores, tt_x, tt_indices, tt_scores)`.

    `tokens`/`indices`/`scores` are the torch sources (kept for golden computation); the `tt_*`
    values are the DRAM-resident device tensors fed to `decode.forward`. `balanced=True` (perf) spreads
    tokens evenly across experts via `_create_balanced_expert_indices`; the default samples per-token.
    """
    tokens = create_torch_dispatch_input_tensor(ctx.batch, 1, ctx.hidden_size, ttnn.bfloat16)
    if balanced:
        indices = _create_balanced_expert_indices(ctx.batch, ctx.routed_experts, ctx.select_experts_k)
    else:
        indices = _create_expert_indices(ctx.batch, ctx.routed_experts, ctx.select_experts_k)
    scores = create_torch_dispatch_input_expert_scores_tensor(ctx.batch, 1, ctx.select_experts_k, ttnn.bfloat16)

    mesh_mapper = ttnn.ShardTensor2dMesh(ctx.mesh_device, dims=ctx.shard_dims, mesh_shape=ctx.mesh_shape)
    tt_x = ttnn.from_torch(
        tokens,
        device=ctx.mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    tt_indices = ttnn.from_torch(
        indices,
        device=ctx.mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    tt_scores = ttnn.from_torch(
        scores,
        device=ctx.mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    return tokens, indices, scores, tt_x, tt_indices, tt_scores


@torch.no_grad()
def _make_golden(ctx: SimpleNamespace, tokens: torch.Tensor, indices: torch.Tensor, scores: torch.Tensor):
    """torch reference output for one dispatch iteration, including shared experts when configured."""
    golden = _gen_output_golden(
        tokens,
        indices,
        scores,
        ctx.w0_per_expert,
        ctx.w1_per_expert,
        ctx.w2_per_expert,
        ctx.batch,
        ctx.hidden_size,
        ctx.select_experts_k,
        activation_type=ctx.config.compute.activation_type,
        b0_per_expert=ctx.b0_per_expert,
        b1_per_expert=ctx.b1_per_expert,
        b2_per_expert=ctx.b2_per_expert,
    )
    if ctx.shared_id_to_torch_w0 is not None:
        num_tp = ctx.mesh_shape[1 - ctx.cluster_axis]
        golden = _add_shared_experts_to_golden_tp(
            golden,
            tokens,
            ctx.batch,
            ctx.shared_id_to_torch_w0,
            ctx.shared_id_to_torch_w1,
            ctx.shared_id_to_torch_w2,
            shared_expert_scale=ctx.config.reduce.shared_expert_scale,
            activation_type=ctx.config.compute.activation_type,
            num_tp=num_tp,
        )
    return golden


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((16, 4), id="16x4"),
        pytest.param(
            (16, 1),
            id="16x1",
            marks=pytest.mark.skipif(
                not is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_16x1),
                reason=f"16x1 mesh requires TT_MESH_GRAPH_DESC_PATH={MESH_GRAPH_DESC_16x1}",
            ),
        ),
        pytest.param(
            (8, 4),
            id="8x4",
            marks=pytest.mark.skipif(is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_16x1), reason=f"16x1 MGD is set"),
        ),
        pytest.param(
            (8, 1),
            id="8x1",
            marks=pytest.mark.skipif(not is_blackhole(), reason=f"8x1 grid is only for BH testing"),
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 500_000,
            },
            id="fabric_1D_ring",
        ),
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 500_000,
            },
            id="fabric_1D",
            marks=pytest.mark.skipif(
                not is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_BH_LB_8x1),
                reason="FABRIC_1D only for BH LB 8x1 line topology",
            ),
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [3])
@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=_config_id)
@pytest.mark.timeout(900)
@torch.no_grad()
def test_tt_moe_decode(
    mesh_device: ttnn.MeshDevice,
    device_params: dict,
    config_path: Path,
    num_iterations: int,
):
    ctx = _setup_decode(mesh_device, device_params, config_path)
    decode = ctx.decode
    mesh_shape = ctx.mesh_shape

    # --- per-iteration inputs + goldens ---
    tt_dispatch_inputs = []
    tt_dispatch_indices = []
    tt_dispatch_scores = []
    output_goldens = []
    for _ in range(num_iterations):
        tokens, indices, scores, tt_x, tt_indices, tt_scores = _make_dispatch_inputs(ctx)
        tt_dispatch_inputs.append(tt_x)
        tt_dispatch_indices.append(tt_indices)
        tt_dispatch_scores.append(tt_scores)
        output_goldens.append(_make_golden(ctx, tokens, indices, scores))

    logger.info("Goldens computed")

    # --- run + collect outputs ---
    logger.info("Running forward iterations")
    tt_outputs = []
    for it in range(num_iterations):
        try:
            output = decode.forward(
                tt_x=tt_dispatch_inputs[it],
                tt_scores=tt_dispatch_scores[it],
                tt_indices=tt_dispatch_indices[it],
                layer_id=0,
            )
            if output.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                final_output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(output)
            else:
                final_output = output
            tt_outputs.append(final_output)
            ttnn.synchronize_device(mesh_device, sub_device_ids=[ttnn.SubDeviceId(0)])
        except Exception as e:
            _print_exception_and_fail(f"forward iteration {it} failed: {type(e).__name__}")
        logger.info(f"Op iteration {it} complete")

    # --- verify ---
    logger.info("Verifying outputs")
    all_passed = True
    for it in range(num_iterations):
        # ATOL is a tad high, only observed this large for big models (deepseek) might improve with col reduction
        # might just need to use calibrated values (see above) but I am still fairly sure it is benign.
        if not verify_output(it, mesh_device, mesh_shape, tt_outputs[it], output_goldens[it], atol_threshold=800):
            all_passed = False

    assert all_passed, f"TTMoEDecode output verification failed for {config_path.stem}"


# ---------------------------------------------------------------------------
# perf test
# ---------------------------------------------------------------------------


def _run_forward_with_trace(num_iters: int, op_func, mesh_device: ttnn.MeshDevice, profiler: BenchmarkProfiler):
    """Trace-capture a SINGLE `forward` and time `num_iters` back-to-back executions of it.

    Capturing one forward and replaying it `num_iters` times (rather than capturing `num_iters`
    forwards into one trace) is deliberate. `forward` reuses the module's *single* persistent
    dispatch/combine buffers and global semaphores every call. Executing the one-forward trace
    repeatedly reproduces steady-state decode exactly — each replay is a complete forward over those
    persistent buffers, serialized on the command queue. Capturing many forwards into one trace
    instead pipelines them with no sync between iterations, so a later iteration's dispatch can
    overwrite a buffer an earlier iteration's combine still reads; the combine then emits a malformed
    packet and the fabric mux forwards a header with a garbage payload size, tripping the
    `size_bytes <= buffer_size_bytes` assert in `edm_fabric_worker_adapters.hpp`. The eager
    correctness test avoids this by `synchronize_device`-ing between iterations.
    """
    logger.info("Compiling model")
    op_func(1)
    ttnn.synchronize_device(mesh_device)

    logger.info("Capturing single-forward trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    op_func(1)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    logger.info("Warming up (executing captured trace, untimed)")
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    logger.info(f"Starting trace perf test ({num_iters} iterations)...")
    signpost("start")
    profiler.start("tt-moe-decode-trace")
    for _ in range(num_iters):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt-moe-decode-trace")
    signpost("stop")

    ttnn.release_trace(mesh_device, trace_id)

    return profiler.get_duration("tt-moe-decode-trace") / num_iters * 1e6


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((16, 4), id="16x4"),
        pytest.param(
            (16, 1),
            id="16x1",
            marks=pytest.mark.skipif(
                not is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_16x1),
                reason=f"16x1 mesh requires TT_MESH_GRAPH_DESC_PATH={MESH_GRAPH_DESC_16x1}",
            ),
        ),
        pytest.param(
            (8, 4),
            id="8x4",
            marks=pytest.mark.skipif(is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_16x1), reason=f"16x1 MGD is set"),
        ),
        pytest.param(
            (8, 1),
            id="8x1",
            marks=pytest.mark.skipif(not is_blackhole(), reason=f"8x1 grid is only for BH testing"),
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 500_000,
            },
            id="fabric_1D_ring",
        ),
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 500_000,
            },
            id="fabric_1D",
            marks=pytest.mark.skipif(
                not is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_BH_LB_8x1),
                reason="FABRIC_1D only for BH LB 8x1 line topology",
            ),
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [10])
@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=_config_id)
@pytest.mark.timeout(900)
@torch.no_grad()
def test_tt_moe_decode_perf(
    mesh_device: ttnn.MeshDevice,
    device_params: dict,
    config_path: Path,
    num_iterations: int,
):
    """Trace-mode perf measurement for `TTMoEDecode.forward`.

    Shares all setup with `test_tt_moe_decode` via `_setup_decode` / `_make_dispatch_inputs`, then times
    `num_iterations` executions of a single-forward trace and reports the average per-iteration wall time.
    Differs from the correctness test in two perf-oriented ways: `select_experts_k` is reduced to reflect
    the mesh slice (`divide_k_for_slice`), and tokens are balanced evenly across experts (`balanced`).
    Correctness is not re-checked here (see `test_tt_moe_decode`); a single set of dispatch inputs is
    reused across all iterations and outputs are not retained.
    """
    ctx = _setup_decode(mesh_device, device_params, config_path, divide_k_for_slice=True)
    decode = ctx.decode

    # A single persistent set of dispatch inputs, reused every iteration. `forward` only ever
    # deallocates its own internally-formatted copies (see `_format_dispatch_inputs`), never the
    # caller-owned tensors, so reuse across the trace is safe. Tokens are spread evenly across experts
    # (balanced routing) so the measurement reflects a realistic, non-hotspotted load.
    _, _, _, tt_x, tt_indices, tt_scores = _make_dispatch_inputs(ctx, balanced=True)

    def _run_op(n: int):
        tt_out = None
        for _ in range(n):
            tt_out = decode.forward(tt_x=tt_x, tt_scores=tt_scores, tt_indices=tt_indices, layer_id=0)
        return tt_out

    profiler = BenchmarkProfiler()
    try:
        avg_us = _run_forward_with_trace(num_iterations, _run_op, mesh_device, profiler)
    except Exception as e:
        _print_exception_and_fail(f"forward trace perf run failed: {type(e).__name__}")

    logger.info(f"[{config_path.stem}] TTMoEDecode.forward avg over {num_iterations} iters: {avg_us:.2f} us")
