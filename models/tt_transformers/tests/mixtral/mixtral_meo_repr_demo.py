# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import time

import pytest
import torch

import ttnn
from ttnn import ReplicateTensorToMesh, ShardTensorToMesh

# Change if needed, this works fine for triggering the issue
_TRACE_REGION_SIZE = 20_000_000

# Current defaults. Could change later...
_DIM = 4096
_REPLAY_ITERS = 1
_REPLAY_BLOCKING = False
_SYNC_BEFORE_CAPTURE = False
_NUM_GATING_BLOCKS_PER_FORWARD = 32

# Deterministic pattern observed to trigger the hang.
_MINIMAL_HANG_PATTERN = [128, 1024, 128, 1024]
_TRANSITION_WARMUP_REPLAYS = 32

# Optional allocator pressure while traces are active.
_PRESSURE_ALLOCATIONS = 2
_PRESSURE_SEQ_LEN = 1024
_PRESSURE_COPY = False
_MAX_PRESSURE_BUFFERS = 16
# Keep control path clean; allocator pressure is used to amplify legacy-only failure odds.
_PRESSURE_ON_KEEPDIM_TRUE = False


def _resolve_keepdim_variants():
    env_value = os.getenv("MIXTRAL_KEEPDIM_FIX")
    if env_value is None or env_value == "":
        return [
            pytest.param(True, id="keepdim_true"),
            pytest.param(False, id="keepdim_false"),
        ]

    normalized = env_value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return [pytest.param(True, id="keepdim_true")]
    if normalized in {"0", "false", "f", "no", "n"}:
        return [pytest.param(False, id="keepdim_false")]

    raise RuntimeError(
        f"Invalid MIXTRAL_KEEPDIM_FIX={env_value!r}. "
        "Expected one of: 0/1, false/true, no/yes."
    )


_KEEPDIM_FIX_VARIANTS = _resolve_keepdim_variants()


def _gate_mm_kernel_config():
    if not hasattr(ttnn, "WormholeComputeKernelConfig") or not hasattr(ttnn, "MathFidelity"):
        return None
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


_GATE_MM_OUTPUT_MEMCFG = ttnn.L1_MEMORY_CONFIG
_GATE_MM_KERNEL_CONFIG = _gate_mm_kernel_config()


def _debug(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[mixtral-freeze-debug {ts}] {msg}", flush=True)


class _MinimalMixtralLikeGating:
    """
    Minimal Mixtral MoE gating path with legacy rank-change toggle:
    - keepdim=True  => fixed path
    - keepdim=False => legacy reduce(rank-3) + reshape(rank-4) path
    """

    def __init__(self, mesh_device, dim: int, keepdim_fix: bool):
        self.mesh_device = mesh_device
        self.dim = dim
        self.keepdim_fix = keepdim_fix
        self.num_devices = mesh_device.get_num_devices()
        assert self.num_devices == 8, "This repro expects an 8-device mesh (Mixtral/T3K-style)."

        gate_weight = torch.randn(8, dim, dtype=torch.float32)
        gates_tensor = torch.nn.functional.pad(gate_weight.permute(1, 0), (0, 56), "constant", 0).unsqueeze(0).unsqueeze(0)

        gates_tensor_list = []
        for dev in range(self.num_devices):
            i, j = 0, dev
            gates_tensor_dev = gates_tensor.clone()
            gates_tensor_dev[:, :, :, [i, j]] = gates_tensor_dev[:, :, :, [j, i]]
            gates_tensor_list.append(gates_tensor_dev)

        self.gates_H8 = ttnn.as_tensor(
            torch.cat(gates_tensor_list, dim=1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=None,
            device=self.mesh_device,
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=1),
        )

        top8_mask = torch.full((1, 1, 1, 64), fill_value=torch.finfo(torch.float).min)
        top8_mask[:, :, :, :8] = 0.0
        self.top8_mask_11B_64 = ttnn.from_torch(
            top8_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
        )
        self.top8_mask_11B_64 = ttnn.sum(self.top8_mask_11B_64, dim=2, keepdim=self.keepdim_fix)

        top2_mask = torch.full((1, 1, 1, 32), fill_value=torch.finfo(torch.float).min)
        top2_mask[:, :, :, :2] = 0.0
        self.top2_mask_11BB = ttnn.from_torch(
            top2_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
        )
        self.top2_mask_11BB = ttnn.sum(self.top2_mask_11BB, dim=2, keepdim=self.keepdim_fix)

    def _single_gate_block(self, x, debug_label: str):
        _debug(f"[{debug_label}] gate matmul")
        gate_logits_1SB8 = ttnn.matmul(
            x,
            self.gates_H8,
            memory_config=_GATE_MM_OUTPUT_MEMCFG,
            compute_kernel_config=_GATE_MM_KERNEL_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
        )

        _debug(f"[{debug_label}] add top8 mask")
        gate_logits_1SB8 = ttnn.add(gate_logits_1SB8, self.top8_mask_11B_64)

        _debug(f"[{debug_label}] topk")
        topk_values, topk_indices = ttnn.topk(gate_logits_1SB8, 32)

        _debug(f"[{debug_label}] add top2 mask")
        topk_values = ttnn.add(topk_values, self.top2_mask_11BB)

        _debug(f"[{debug_label}] eqz+typecast")
        mask_B2 = ttnn.eqz(topk_indices)
        mask_B2 = ttnn.typecast(mask_B2, dtype=ttnn.bfloat16)

        _debug(f"[{debug_label}] reduce sum on gating weights")
        weights_reduce = ttnn.sum(ttnn.softmax(topk_values, dim=-1) * mask_B2, dim=3, keepdim=self.keepdim_fix)
        _debug(f"[{debug_label}] reduce sum returned")

        if self.keepdim_fix:
            weights_1SB1 = weights_reduce
        else:
            _debug(f"[{debug_label}] reshape (legacy rank-change point)")
            weights_1SB1 = ttnn.reshape(weights_reduce, [1, 1, x.shape[-2], 1])
            _debug(f"[{debug_label}] reshape returned")

        _debug(f"[{debug_label}] final mul")
        out = ttnn.mul(x, weights_1SB1)
        _debug(f"[{debug_label}] final mul returned")

        topk_values.deallocate(True)
        topk_indices.deallocate(True)
        mask_B2.deallocate(True)
        gate_logits_1SB8.deallocate()
        weights_1SB1.deallocate(True)
        return out

    def forward(self, x, debug_label: str):
        if _NUM_GATING_BLOCKS_PER_FORWARD <= 1:
            return self._single_gate_block(x, debug_label)

        _debug(f"[{debug_label}] entering {_NUM_GATING_BLOCKS_PER_FORWARD} stacked gating blocks")
        current = x
        for block_idx in range(_NUM_GATING_BLOCKS_PER_FORWARD):
            block_label = f"{debug_label}-blk{block_idx + 1}"
            out = self._single_gate_block(current, block_label)
            if block_idx > 0:
                ttnn.deallocate(current)
            current = out
        _debug(f"[{debug_label}] stacked gating blocks done")
        return current


def _build_input(mesh_device, seq_len, dim):
    shape = [1, 1, seq_len, dim]
    x_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device)
    x_host = ttnn.from_torch(
        torch.randn(*shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    ttnn.copy_host_to_device_tensor(x_host, x_dev)
    return x_dev


def _refresh_input(x_dev, mesh_device, seq_len, dim, label=""):
    prefix = f"[{label}] " if label else ""
    _debug(f"{prefix}refresh input host build start (seq_len={seq_len})")
    x_host = ttnn.from_torch(
        torch.randn(1, 1, seq_len, dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    _debug(f"{prefix}refresh input H2D copy start")
    ttnn.copy_host_to_device_tensor(x_host, x_dev)
    _debug(f"{prefix}refresh input H2D copy done")


def _capture_trace_for_seq_len(mesh_device, gating, seq_len, dim, keepdim_fix):
    label_prefix = f"trace-build-s{seq_len}-keepdim-{keepdim_fix}"

    compile_x = _build_input(mesh_device, seq_len, dim)
    _debug(f"[{label_prefix}] compile run start")
    compile_out = gating.forward(compile_x, debug_label=f"{label_prefix}-compile")
    _debug(f"[{label_prefix}] compile run returned")
    ttnn.deallocate(compile_out)
    ttnn.deallocate(compile_x)

    if _SYNC_BEFORE_CAPTURE:
        _debug(f"[{label_prefix}] sync before capture start")
        ttnn.synchronize_device(mesh_device)
        _debug(f"[{label_prefix}] sync before capture done")

    trace_x = _build_input(mesh_device, seq_len, dim)
    _debug(f"[{label_prefix}] begin_trace_capture")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_out = gating.forward(trace_x, debug_label=f"{label_prefix}-capture")
    _debug(f"[{label_prefix}] end_trace_capture")
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    _debug(f"[{label_prefix}] trace captured")
    return trace_id, trace_out, trace_x


def _capture_or_get_trace(mesh_device, gating, trace_cache, trace_build_order, seq_len, dim, keepdim_fix):
    if seq_len in trace_cache:
        return trace_cache[seq_len]

    trace_cache[seq_len] = _capture_trace_for_seq_len(
        mesh_device=mesh_device,
        gating=gating,
        seq_len=seq_len,
        dim=dim,
        keepdim_fix=keepdim_fix,
    )
    trace_build_order.append(seq_len)
    return trace_cache[seq_len]


def _replay_trace(mesh_device, trace_id, replay_iters):
    for i in range(replay_iters):
        _debug(f"execute_trace replay {i + 1}/{replay_iters}, blocking={_REPLAY_BLOCKING}")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=_REPLAY_BLOCKING)


def _release_trace_entry(mesh_device, trace_entry):
    trace_id, trace_out, x_dev = trace_entry
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(trace_out)
    ttnn.deallocate(x_dev)


def _allocate_pressure_buffers(mesh_device, dim, pressure_buffers, label):
    if _PRESSURE_ALLOCATIONS <= 0:
        return
    for idx in range(_PRESSURE_ALLOCATIONS):
        _debug(f"[{label}] pressure alloc {idx + 1}/{_PRESSURE_ALLOCATIONS}, seq_len={_PRESSURE_SEQ_LEN}, copy={_PRESSURE_COPY}")
        p_dev = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, _PRESSURE_SEQ_LEN, dim]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            mesh_device,
        )
        if _PRESSURE_COPY:
            p_host = ttnn.from_torch(
                torch.randn(1, 1, _PRESSURE_SEQ_LEN, dim, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ReplicateTensorToMesh(mesh_device),
            )
            ttnn.copy_host_to_device_tensor(p_host, p_dev)
        pressure_buffers.append(p_dev)
        if _MAX_PRESSURE_BUFFERS > 0 and len(pressure_buffers) > _MAX_PRESSURE_BUFFERS:
            ttnn.deallocate(pressure_buffers.pop(0))


def _run_transition_stress(mesh_device, gating, trace_cache, trace_build_order, dim, keepdim_fix):
    _debug(
        "TRANSITION_STRESS("
        f"keepdim_fix={keepdim_fix}, small=128, large=1024, warmup_replays={_TRANSITION_WARMUP_REPLAYS})"
    )
    trace_small = _capture_or_get_trace(
        mesh_device=mesh_device,
        gating=gating,
        trace_cache=trace_cache,
        trace_build_order=trace_build_order,
        seq_len=128,
        dim=dim,
        keepdim_fix=keepdim_fix,
    )
    trace_small_id, _, trace_small_x = trace_small
    _refresh_input(trace_small_x, mesh_device, 128, dim, label="transition-small")
    _replay_trace(mesh_device, trace_small_id, _TRANSITION_WARMUP_REPLAYS)

    _capture_or_get_trace(
        mesh_device=mesh_device,
        gating=gating,
        trace_cache=trace_cache,
        trace_build_order=trace_build_order,
        seq_len=1024,
        dim=dim,
        keepdim_fix=keepdim_fix,
    )


def _replay_and_pressure(mesh_device, trace_entry, seq_len, dim, pressure_buffers, label, apply_pressure: bool):
    trace_id, _, trace_x = trace_entry
    _debug(f"[{label}] trace lookup done for padded len {seq_len}")
    _refresh_input(trace_x, mesh_device, seq_len, dim, label=label)
    _debug(f"[{label}] input refresh done; replaying trace")
    _replay_trace(mesh_device, trace_id, _REPLAY_ITERS)
    if apply_pressure:
        _allocate_pressure_buffers(mesh_device, dim, pressure_buffers, label)
    else:
        _debug(f"[{label}] pressure skipped for keepdim control path")


@pytest.mark.timeout(900)
@pytest.mark.parametrize("keepdim_fix", _KEEPDIM_FIX_VARIANTS)
@pytest.mark.parametrize("device_params", [{"fabric_config": True, "trace_region_size": _TRACE_REGION_SIZE}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_mixtral_like_gating_trace_h2d_refresh_hang_minimal(mesh_device, reset_seeds, device_params, keepdim_fix):
    """
    Minimal deterministic reproducer:
    hangs in legacy path during host->device refresh copy for seq_len=1024.
    """
    _debug(f"H2D_REFRESH_HANG_MINIMAL(keepdim_fix={keepdim_fix}, pattern={_MINIMAL_HANG_PATTERN})")
    apply_pressure = (not keepdim_fix) or _PRESSURE_ON_KEEPDIM_TRUE
    _debug(f"PRESSURE_MODE(apply_pressure={apply_pressure}, keepdim_fix={keepdim_fix})")
    gating = _MinimalMixtralLikeGating(mesh_device, dim=_DIM, keepdim_fix=keepdim_fix)
    trace_cache = {}
    trace_build_order = []
    pressure_buffers = []

    try:
        _run_transition_stress(
            mesh_device=mesh_device,
            gating=gating,
            trace_cache=trace_cache,
            trace_build_order=trace_build_order,
            dim=_DIM,
            keepdim_fix=keepdim_fix,
        )

        for step_idx, seq_len in enumerate(_MINIMAL_HANG_PATTERN):
            label = f"mini-step{step_idx + 1}-pad{seq_len}"
            _debug(f"[{label}] start")
            if seq_len not in trace_cache:
                _capture_or_get_trace(
                    mesh_device=mesh_device,
                    gating=gating,
                    trace_cache=trace_cache,
                    trace_build_order=trace_build_order,
                    seq_len=seq_len,
                    dim=_DIM,
                    keepdim_fix=keepdim_fix,
                )
            _replay_and_pressure(
                mesh_device=mesh_device,
                trace_entry=trace_cache[seq_len],
                seq_len=seq_len,
                dim=_DIM,
                pressure_buffers=pressure_buffers,
                label=label,
                apply_pressure=apply_pressure,
            )
            _debug(f"[{label}] done")
    finally:
        for seq_len in reversed(trace_build_order):
            if seq_len in trace_cache:
                _release_trace_entry(mesh_device, trace_cache.pop(seq_len))
        for p_dev in reversed(pressure_buffers):
            ttnn.deallocate(p_dev)
