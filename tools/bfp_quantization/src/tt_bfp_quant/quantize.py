# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Model-independent weight transforms. Returned tensors are CPU float32."""
from dataclasses import dataclass
import time

import numpy as np
import torch

from .native import resolve_backend, search_native, gptq_native_block
from .rounding import search_numpy, validate_options


def _matrix(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or min(value.shape) <= 0 or not value.is_floating_point():
        raise ValueError(f"{name} must be a nonempty floating-point 2D Torch tensor")
    value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values after conversion to float32")
    return value


def _threads(threads):
    threads = torch.get_num_threads() if threads is None else threads
    if type(threads) is not int or threads <= 0:
        raise ValueError("threads must be a positive integer")
    return threads


def _splits(size, output_splits):
    if output_splits is None:
        return (size,)
    result = tuple(output_splits)
    if any(type(n) is not int or n <= 0 for n in result) or sum(result) != size:
        raise ValueError("output_splits must be positive output widths summing to weight.shape[0]")
    return result


def _search(x, bits, deltas, backend, threads):
    return search_native(x, bits, deltas, threads) if backend == "native" else search_numpy(x, bits, deltas)


def _add_counts(total, part):
    for key, count in part.items():
        total[key] += count


@torch.inference_mode()
def search_packed(weight, bits=4, exponent_deltas=(0, -1), *, backend="auto", threads=None, row_chunk=128):
    """Search one physical 2D shard in final TT layout (typically [K,N]).

    Groups run along the last axis, with independent zero padding per row.
    Call separately for each device shard after any concat/transpose/reorder.
    The same layout and shard boundaries must be used at inference time.
    """
    deltas = validate_options(bits, exponent_deltas)
    threads, backend = _threads(threads), resolve_backend(backend)
    if type(row_chunk) is not int or row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    started = time.monotonic()
    weight = _matrix(weight, "weight")
    rows, columns = weight.shape
    padded_columns = (columns + 15) // 16 * 16
    out = torch.empty_like(weight)
    counts = {str(d): 0 for d in deltas}
    for start in range(0, rows, row_chunk):
        stop = min(start + row_chunk, rows)
        x = np.zeros((stop - start, padded_columns), dtype=np.float32)
        x[:, :columns] = weight[start:stop].numpy()
        q, selected = _search(x, bits, deltas, backend, threads)
        out[start:stop] = torch.from_numpy(q[:, :columns])
        _add_counts(counts, selected)
    return out, {
        "method": "exponent_search",
        "bits": bits,
        "layout": "packed",
        "shape": list(weight.shape),
        "exponent_deltas": list(deltas),
        "group_choices": counts,
        "backend": backend,
        "threads": threads,
        "seconds": time.monotonic() - started,
        "group_count_includes_last_partial_group": True,
    }


@torch.inference_mode()
def search_linear(
    weight, bits=4, exponent_deltas=(0, -1), *, output_splits=None, backend="auto", threads=None, row_chunk=128
):
    """Exponent search for Linear weights [out_features,in_features].

    output_splits resets groups at each output-shard boundary. Input sharding
    does not change the 16-output grouping. Defaults reproduce max/max−1.
    Set exponent_deltas=(0,) for the ordinary-rounding baseline.
    """
    started = time.monotonic()
    weight = _matrix(weight, "weight")
    splits = _splits(weight.shape[0], output_splits)
    pieces, records = [], []
    for shard in weight.split(splits, dim=0):
        q, info = search_packed(shard.T, bits, exponent_deltas, backend=backend, threads=threads, row_chunk=row_chunk)
        pieces.append(q.T.contiguous())
        records.append(info)
    counts = {str(d): 0 for d in exponent_deltas}
    for record in records:
        _add_counts(counts, record["group_choices"])
    return torch.cat(pieces, dim=0), {
        **records[0],
        "layout": "linear",
        "shape": list(weight.shape),
        "output_splits": list(splits),
        "group_choices": counts,
        "seconds": time.monotonic() - started,
    }


@dataclass(frozen=True)
class HessianFactor:
    order: torch.Tensor
    inverse_order: torch.Tensor
    dead: torch.Tensor
    upper: torch.Tensor
    damping: float
    act_order: bool
    seconds: float


@torch.inference_mode()
def factor_hessian(hessian, damping=0.01, act_order=True):
    """Factor H = mean(X.T @ X), once per distinct layer input.

    gate/up (and sometimes Q/K/V) may reuse this factor only if their exact
    input activations and channel ordering are the same.
    """
    started = time.monotonic()
    h = _matrix(hessian, "hessian").clone()
    if h.shape[0] != h.shape[1] or not torch.allclose(h, h.T, rtol=1e-5, atol=1e-7):
        raise ValueError("hessian must be a symmetric square second-moment matrix")
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be positive and finite")
    if (h.diag() < 0).any() or not (h.diag() > 0).any():
        raise ValueError("hessian has negative diagonal values or no activation energy")
    dead = h.diag() == 0
    h[dead, dead] = 1
    order = torch.argsort(h.diag(), descending=True) if act_order else torch.arange(len(h))
    h = h[order][:, order].contiguous()
    h.diagonal().add_(damping * h.diag().mean())
    try:
        inverse = torch.cholesky_inverse(torch.linalg.cholesky(h))
        upper = torch.linalg.cholesky(inverse, upper=True)
    except torch.linalg.LinAlgError as error:
        raise ValueError("Hessian factorization failed; check calibration or explicitly increase damping") from error
    if not torch.isfinite(upper).all():
        raise ValueError("Hessian factorization produced non-finite values")
    return HessianFactor(order, torch.argsort(order), dead, upper, damping, bool(act_order), time.monotonic() - started)


def _gptq_numpy_block(block, upper, deltas):
    q, errors = torch.empty_like(block), torch.empty_like(block)
    counts = {str(d): 0 for d in deltas}
    for column in range(block.shape[1]):
        values = block[:, column].contiguous()
        array, selected = search_numpy(values.numpy(), 4, deltas)
        rounded = torch.from_numpy(array)
        q[:, column] = rounded
        error = (values - rounded) / upper[column, column]
        block[:, column:] -= error[:, None] * upper[column, column:][None, :]
        errors[:, column] = error
        _add_counts(counts, selected)
    return q, errors, counts


@torch.inference_mode()
def gptq_search(
    weight,
    hessian=None,
    *,
    factor=None,
    damping=0.01,
    act_order=True,
    exponent_deltas=(0, -1, -2),
    block_size=128,
    output_splits=None,
    backend="auto",
    threads=None,
):
    """BFP4 GPTQ for [out,in] weights; undo activation ordering before export.

    Each sequential column step chooses Emax/Emax−1/Emax−2 by weight SSE,
    then propagates its error using the damped inverse-Hessian factor. This
    matches the experimental GPTQ + search, not a separate activation-scored
    clipping-threshold sweep. No scales or permutation remain at runtime.
    """
    started = time.monotonic()
    deltas = validate_options(4, exponent_deltas)
    threads, backend = _threads(threads), resolve_backend(backend)
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if (hessian is None) == (factor is None):
        raise ValueError("Provide exactly one of hessian or factor")
    weight = _matrix(weight, "weight")
    splits = _splits(weight.shape[0], output_splits)
    factor_reused = factor is not None
    factor = factor_hessian(hessian, damping, act_order) if factor is None else factor
    if not isinstance(factor, HessianFactor) or factor.upper.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("factor input dimension does not match weight.shape[1]")
    width = weight.shape[1]
    for name, dtype, shape in (
        ("upper", torch.float32, (width, width)),
        ("order", torch.int64, (width,)),
        ("inverse_order", torch.int64, (width,)),
        ("dead", torch.bool, (width,)),
    ):
        value = getattr(factor, name)
        if value.device.type != "cpu" or value.dtype != dtype or value.shape != shape:
            raise ValueError(f"factor.{name} must be a CPU {dtype} tensor of shape {shape}")
    indices = torch.arange(width)
    if (
        not torch.equal(factor.order.sort().values, indices)
        or not torch.equal(factor.inverse_order.sort().values, indices)
        or not torch.equal(factor.order[factor.inverse_order], indices)
    ):
        raise ValueError("factor contains invalid column permutations")
    counts = {str(d): 0 for d in deltas}
    result = torch.empty_like(weight)
    first = 0
    for shard in weight.split(splits, dim=0):
        rows, columns = shard.shape
        padded_rows = (rows + 15) // 16 * 16
        w = torch.zeros((padded_rows, columns), dtype=torch.float32)
        w[:rows] = shard
        w[:, factor.dead] = 0
        w = w[:, factor.order].contiguous()
        q = torch.empty_like(w)
        for begin in range(0, columns, block_size):
            end = min(begin + block_size, columns)
            block = w[:, begin:end].clone()
            upper = factor.upper[begin:end, begin:end]
            if backend == "native":
                rounded, errors, selected = gptq_native_block(block, upper, deltas, threads)
            else:
                rounded, errors, selected = _gptq_numpy_block(block, upper, deltas)
            if not torch.isfinite(errors).all():
                raise ValueError("GPTQ compensation overflowed; check calibration and weight magnitudes")
            q[:, begin:end] = rounded
            _add_counts(counts, selected)
            w[:, end:] -= errors @ factor.upper[begin:end, end:]
        result[first : first + rows] = q[:rows, factor.inverse_order]
        first += rows
    if not torch.isfinite(result).all():
        raise ValueError("GPTQ produced non-finite weights; check calibration and damping")
    return result, {
        "method": "gptq_search",
        "bits": 4,
        "layout": "linear",
        "shape": list(weight.shape),
        "output_splits": list(splits),
        "exponent_deltas": list(deltas),
        "group_choices": counts,
        "damping": factor.damping,
        "act_order": factor.act_order,
        "block_size": block_size,
        "backend": backend,
        "threads": threads,
        "factor_reused": factor_reused,
        "factor_seconds": factor.seconds,
        "seconds": time.monotonic() - started,
    }


def to_bf16_exact(weight):
    """Compact, lossless carrier for prepared values; not a packed BFP file."""
    weight = _matrix(weight, "weight")
    stored = weight.bfloat16()
    if not torch.equal(stored.float(), weight):
        raise ValueError("Prepared weights cannot be represented exactly in BF16")
    return stored
