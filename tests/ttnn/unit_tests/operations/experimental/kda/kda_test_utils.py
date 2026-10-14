# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light numerical assertions shared by KDA tests."""

from __future__ import annotations
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
import ttnn


def _finiteness(name: str, tensor: torch.Tensor) -> tuple[list[str], str]:
    element_count = tensor.numel()
    nan_count = int(torch.isnan(tensor).sum().item()) if tensor.dtype.is_floating_point else 0
    positive_inf_count = int(torch.isposinf(tensor).sum().item()) if tensor.dtype.is_floating_point else 0
    negative_inf_count = int(torch.isneginf(tensor).sum().item()) if tensor.dtype.is_floating_point else 0
    non_finite_count = nan_count + positive_inf_count + negative_inf_count
    summary = (
        f"{name} finiteness: non_finite={non_finite_count}/{element_count}, "
        f"nan={nan_count}/{element_count}, +inf={positive_inf_count}/{element_count}, "
        f"-inf={negative_inf_count}/{element_count}"
    )
    failures = [] if non_finite_count == 0 else [summary]
    return failures, summary


def _pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    expected_flat = expected.detach().float().reshape(-1)
    actual_flat = actual.detach().float().reshape(-1)
    if torch.equal(expected_flat, actual_flat):
        return 1.0
    expected_centered = expected_flat.double() - expected_flat.double().mean()
    actual_centered = actual_flat.double() - actual_flat.double().mean()
    denominator = torch.linalg.vector_norm(expected_centered) * torch.linalg.vector_norm(actual_centered)
    if denominator == 0:
        return float(torch.allclose(expected_flat, actual_flat, rtol=1e-5, atol=1e-4))
    return float(torch.dot(expected_centered, actual_centered) / denominator)


def _relative_rmse(expected: torch.Tensor, actual: torch.Tensor) -> float:
    """RMS error as a fraction of the signal's own RMS.

    PCC is a correlation, so it is dominated by the bulk of a tensor and can stay
    above 0.999 while a small region is badly wrong -- eleven wrong rows in 1280
    passed a PCC gate here. Relative RMSE grows with the error's total energy, so
    it registers a localised fault that PCC averages away, and being normalised it
    needs no per-tensor threshold.
    """
    if not expected.numel():
        return 0.0
    difference = (expected.float() - actual.float()).pow(2).mean().sqrt()
    scale = expected.float().pow(2).mean().sqrt()
    return float(difference / scale) if float(scale) > 0 else float(difference)


def assert_accurate(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    name: str = "accuracy",
    pcc_threshold: float = 0.999,
    rmse_threshold: float | None = None,
    linf_threshold: float | None = None,
) -> float:
    """Require matching metadata, finite tensors, PCC, relative RMSE and relative L-inf.

    Three metrics because they fail differently. PCC is a correlation and RMSE is a
    mean, so both average over the whole tensor: eleven wrong rows in 1280 held PCC
    at 0.9996 and lifted relative RMSE only from 1.3e-2 to 1.6e-2, inside the BF16
    output's own noise. Relative L-inf does not average, and on the same data it
    separated cleanly -- 1.6e-2 when correct against 1.3e-1 to 2.7e-1 when a few
    rows were fed the wrong carry. That is the failure mode this layer keeps
    producing, so the peak matters more than the mean.

    Relative L-inf is always reported but gated only when ``linf_threshold`` is
    given, because a peak divided by a whole-tensor RMS
    depends on how concentrated the signal is: correct runs here span 1.4e-2 on a
    convolution carry to 1.6e+0 on a recurrent state, so no single bound is both
    safe and useful. Pass the clean value you measured for the tensor at hand.
    """
    failures = []
    if expected.shape != actual.shape:
        failures.append(f"{name} shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    if expected.dtype != actual.dtype:
        failures.append(f"{name} dtype {actual.dtype} != {expected.dtype}")
    expected_failures, expected_summary = _finiteness(f"{name} expected", expected)
    actual_failures, actual_summary = _finiteness(f"{name} actual", actual)
    failures.extend(expected_failures)
    failures.extend(actual_failures)
    if failures:
        raise AssertionError("\n".join(failures))

    pcc = _pcc(expected, actual)
    rmse = _relative_rmse(expected, actual)
    max_abs = float((expected.float() - actual.float()).abs().max()) if expected.numel() else 0.0
    scale = float(expected.float().pow(2).mean().sqrt()) if expected.numel() else 0.0
    linf = max_abs / scale if scale > 0 else max_abs
    print(expected_summary)
    print(actual_summary)
    print(f"{name}: PCC={pcc:.6f}, rel_RMSE={rmse:.3e}, rel_Linf={linf:.3e}, max_abs={max_abs:.6e}")
    if pcc < pcc_threshold:
        raise AssertionError(f"{name} PCC {pcc:.6f} < {pcc_threshold}")
    if rmse_threshold is not None and rmse > rmse_threshold:
        raise AssertionError(f"{name} relative RMSE {rmse:.3e} > {rmse_threshold:.3e}")
    if linf_threshold is not None and linf > linf_threshold:
        raise AssertionError(
            f"{name} relative L-inf {linf:.3e} > {linf_threshold:.3e} (max_abs {max_abs:.3e}) -- "
            "a localised fault, typically a few rows given the wrong carry"
        )
    return pcc


def assert_equal(expected: torch.Tensor, actual: torch.Tensor, *, name: str = "equality") -> None:
    """Require finite tensors with identical metadata and values."""
    failures = []
    if expected.shape != actual.shape:
        failures.append(f"{name} shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    if expected.dtype != actual.dtype:
        failures.append(f"{name} dtype {actual.dtype} != {expected.dtype}")
    failures.extend(_finiteness(f"{name} expected", expected)[0])
    failures.extend(_finiteness(f"{name} actual", actual)[0])
    if expected.shape == actual.shape and expected.dtype == actual.dtype and not torch.equal(expected, actual):
        failures.append(f"{name} values differ")
    if failures:
        raise AssertionError("\n".join(failures))


def assert_bit_identical(expected: torch.Tensor, actual: torch.Tensor, *, name: str = "determinism") -> None:
    """Require finite tensors with identical metadata and bit patterns."""
    assert_equal(expected, actual, name=name)
    expected_bytes = expected.detach().contiguous().reshape(-1).view(torch.uint8)
    actual_bytes = actual.detach().contiguous().reshape(-1).view(torch.uint8)
    if not torch.equal(expected_bytes, actual_bytes):
        raise AssertionError(f"{name} bit patterns differ")


def collect_accuracy_and_determinism_results(
    device: ttnn.Device,
    run: Callable[[], Sequence[ttnn.Tensor]],
    *,
    count: int = 3,
) -> tuple[tuple[ttnn.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor]:
    """Run repeatedly, retaining only first outputs and one device-side mismatch marker."""
    if count <= 1:
        raise ValueError("count must be greater than one")

    reference_outputs = tuple(run())
    if not reference_outputs:
        raise ValueError("run must return at least one output")
    mismatch_scratch = tuple(
        ttnn.empty(
            output.shape,
            dtype=ttnn.bfloat16,
            layout=output.layout,
            device=device,
            memory_config=output.memory_config(),
        )
        for output in reference_outputs
    )
    mismatch_marker = None
    for _ in range(1, count):
        outputs = tuple(run())
        if len(outputs) != len(reference_outputs):
            for output in outputs:
                ttnn.deallocate(output)
            raise ValueError("run returned a different number of outputs")
        for reference, output, scratch in zip(reference_outputs, outputs, mismatch_scratch, strict=True):
            if (
                output.shape != reference.shape
                or output.dtype != reference.dtype
                or output.layout != reference.layout
                or output.memory_config() != reference.memory_config()
            ):
                for repeat_output in outputs:
                    ttnn.deallocate(repeat_output)
                raise ValueError("run returned output with different metadata")
            ttnn.ne(reference, output, dtype=ttnn.bfloat16, output_tensor=scratch)
            current_mismatch = ttnn.max(scratch)
            ttnn.deallocate(output)
            if mismatch_marker is None:
                mismatch_marker = current_mismatch
            else:
                updated_marker = ttnn.maximum(mismatch_marker, current_mismatch)
                ttnn.deallocate(mismatch_marker)
                ttnn.deallocate(current_mismatch)
                mismatch_marker = updated_marker

    assert mismatch_marker is not None
    reference_outputs_host = tuple(ttnn.to_torch(output).clone() for output in reference_outputs)
    mismatch_marker_host = ttnn.to_torch(mismatch_marker).clone()
    for scratch in mismatch_scratch:
        ttnn.deallocate(scratch)
    ttnn.deallocate(mismatch_marker)
    return reference_outputs, reference_outputs_host, mismatch_marker_host


def height_sharded_memory_config(
    device: ttnn.Device, leading: int, matrix_height: int, matrix_width: int
) -> ttnn.MemoryConfig:
    cores = ttnn.num_cores_to_corerangeset(leading, device.compute_with_storage_grid_size(), row_wise=True)
    return ttnn.create_sharded_memory_config(
        (leading, matrix_height, matrix_width),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def qkv_host_inputs(
    *,
    sequence: int = 64,
    widths: tuple[int, int, int] = (512, 512, 512),
    batch: int = 1,
    history_rows: int = 3,
    seed: int = 223,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    generator = torch.Generator().manual_seed(seed)
    channels = sum(widths)
    inputs = torch.randn(batch, sequence, channels, generator=generator, dtype=torch.bfloat16)
    history = torch.randn(batch, history_rows, channels, generator=generator, dtype=torch.bfloat16)
    taps = tuple(torch.randn(1, 1, channels, generator=generator, dtype=torch.bfloat16) for _ in range(4))
    return inputs, history, taps


def qkv_to_device(
    tensor: torch.Tensor,
    device: ttnn.Device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def qkv_device_inputs(
    device: ttnn.Device,
    *,
    sequence: int = 64,
    widths: tuple[int, int, int] = (512, 512, 512),
    batch: int = 1,
    history_rows: int = 3,
    seed: int = 223,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]],
    tuple[ttnn.Tensor, ttnn.Tensor, tuple[ttnn.Tensor, ...]],
]:
    host = qkv_host_inputs(
        sequence=sequence,
        widths=widths,
        batch=batch,
        history_rows=history_rows,
        seed=seed,
    )
    inputs, history, taps = host
    return host, (
        qkv_to_device(inputs, device, layout=ttnn.ROW_MAJOR_LAYOUT),
        qkv_to_device(history, device, layout=ttnn.ROW_MAJOR_LAYOUT),
        tuple(qkv_to_device(tap, device, layout=ttnn.TILE_LAYOUT) for tap in taps),
    )


def qkv_reference(
    inputs: torch.Tensor,
    history: torch.Tensor,
    taps: tuple[torch.Tensor, ...],
    widths: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    window = torch.cat((history, inputs), dim=1)
    convolved = sum(window[:, tap : tap + inputs.shape[1]] * taps[tap] for tap in range(4))
    return F.silu(convolved).split(widths, dim=-1)


def make_actual_start(device: ttnn.MeshDevice, actual_start: int = 0) -> ttnn.Tensor:
    """Allocate caller-owned start metadata before any trace capture."""
    return ttnn.from_torch(
        torch.tensor([actual_start], dtype=torch.int64),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
