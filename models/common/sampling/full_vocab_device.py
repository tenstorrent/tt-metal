"""Staged public-op composition for unrestricted top-p=1 device sampling.

This module is wired only through an explicitly enabled staged route in
:class:`SamplingGenerator` and does not advertise a runtime capability.  It
keeps logits, probability math,
random variates, and the B-row token result on device while the slower
per-row scalar-seed uniform composition is qualified.  Full-vocabulary
nucleus sampling (top_p < 1), traces, and distributed/sharded logits fail
closed rather than silently narrowing the distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


_INACTIVE_DEVICE_SEED = (1 << 32) - 1
@dataclass(frozen=True)
class DeviceCategoricalPrototypeResult:
    """Device result plus owned scratch that the caller must release."""

    token_ids: object
    valid_distribution: object
    owned_tensors: tuple[object, ...]


def merge_unrestricted_rows(
    categorical: DeviceCategoricalPrototypeResult,
    native_tokens,
    *,
    unrestricted_selector,
    invalid_token_ids,
    ops,
) -> DeviceCategoricalPrototypeResult:
    """Merge categorical rows into the native ``[1,1,1,B]`` token ABI.

    Invalid categorical rows are replaced on device with ``vocab_size`` (the
    caller-provided sentinel).  The existing serving output bounds check then
    fails the request without copying logits/probabilities to the host.
    Bounded and greedy rows remain exactly the native sampler's output.
    """

    native_shape = _shape(native_tokens)
    if len(native_shape) != 4 or native_shape[:3] != (1, 1, 1):
        raise ValueError("native_tokens must have shape [1,1,1,B]")
    batch = native_shape[3]
    expected = (1, 1, 1, batch)
    if _shape(unrestricted_selector) != expected or _shape(invalid_token_ids) != expected:
        raise ValueError("selector and invalid-token tensors must match [1,1,1,B]")
    if _shape(categorical.token_ids) != (1, 1, batch, 1):
        raise ValueError("categorical tokens must have shape [1,1,B,1]")

    owned = list(categorical.owned_tensors)
    merge_owned = []

    def own(tensor):
        owned.append(tensor)
        merge_owned.append(tensor)
        return tensor

    # reshape is a view of categorical-owned storage, not a new allocation.
    # Keep the original allocation in the ownership list and never deallocate
    # the view as a second owner.
    try:
        full_tokens = ops.reshape(categorical.token_ids, expected)
        valid = ops.reshape(categorical.valid_distribution, expected)
        if full_tokens.dtype != native_tokens.dtype:
            full_tokens = own(ops.typecast(full_tokens, dtype=native_tokens.dtype))
        if invalid_token_ids.dtype != native_tokens.dtype:
            invalid_token_ids = own(ops.typecast(invalid_token_ids, dtype=native_tokens.dtype))
        # Compose selection arithmetically from an explicit 0/1 mask.  This
        # keeps semantics stable on runtimes where the public ternary kernel's
        # tensor/tensor branches do not match its documented Python ordering.
        valid_mask = own(ops.typecast(valid, dtype=native_tokens.dtype))
        invalid_mask = own(ops.subtract(1, valid_mask))
        safe_full_tokens = own(
            ops.add(
                own(ops.multiply(valid_mask, full_tokens)),
                own(ops.multiply(invalid_mask, invalid_token_ids)),
            )
        )
        selector = own(ops.gt(unrestricted_selector, 0))
        selector_mask = own(ops.typecast(selector, dtype=native_tokens.dtype))
        native_mask = own(ops.subtract(1, selector_mask))
        merged = own(
            ops.add(
                own(ops.multiply(selector_mask, safe_full_tokens)),
                own(ops.multiply(native_mask, native_tokens)),
            )
        )
        unique_owned = tuple({id(tensor): tensor for tensor in owned}.values())
        return DeviceCategoricalPrototypeResult(merged, valid, unique_owned)
    except Exception:
        _release_owned(merge_owned, ops=ops)
        raise


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _release_owned(tensors: Sequence[object], *, ops) -> None:
    """Release allocations created by this call exactly once.

    Call sites deliberately exclude borrowed inputs and view aliases from the
    owned list.  Release errors are not suppressed: an ownership error must be
    visible during qualification instead of being mistaken for a clean
    allocator plateau.
    """

    seen = set()
    for tensor in reversed(tuple(tensors)):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        if tensor.is_allocated():
            ops.deallocate(tensor)


def _prefix_sum_fp32(tensor, *, dim: int, ops, own: Callable[[object], object]):
    """Inclusive Hillis-Steele scan using public FP32 add operations.

    The stock cumsum path is intentionally not used: the currently qualified
    runtime can quantize FP32 prefixes internally.  This composition retains
    the explicit operation sequence for numerical/silicon qualification.
    """

    rank = len(_shape(tensor))
    dim %= rank
    transpose_back = dim == rank - 1
    if transpose_back:
        tensor = own(ops.transpose(tensor, dim - 1, dim))
        dim -= 1
    width = _shape(tensor)[dim]
    result = tensor
    offset = 1
    while offset < width:
        end = list(_shape(result))
        end[dim] = offset
        zeros = own(ops.multiply(own(ops.slice(result, [0] * rank, end)), 0.0))
        begin = [0] * rank
        end[dim] = width - offset
        previous = own(ops.slice(result, begin, end))
        shifted = own(ops.concat([zeros, previous], dim=dim))
        result = own(ops.add(result, shifted, dtype=ops.float32))
        offset *= 2
    return own(ops.transpose(result, dim, dim + 1)) if transpose_back else result


def _per_slot_uniform_rows(
    *,
    row_scratch: Sequence[object],
    seed_values: Sequence[int],
    active_rows: Sequence[bool],
    ops,
    own: Callable[[object], object],
):
    """Compose a [1,1,B,1] device uniform without host random variates."""

    batch = len(row_scratch)
    if len(seed_values) != batch or len(active_rows) != batch:
        raise ValueError("scratch, seeds, and active_rows must have equal B length")
    rows = []
    for slot, (scratch, seed, active) in enumerate(zip(row_scratch, seed_values, active_rows)):
        if _shape(scratch) != (1, 1, 1, 1):
            raise ValueError(f"row_scratch[{slot}] must have logical shape [1,1,1,1]")
        if not active:
            if int(seed) != _INACTIVE_DEVICE_SEED:
                raise ValueError("inactive rows must carry the TTNN all-ones seed sentinel")
            # Do not derive an inactive value from scratch: NaN * 0 remains
            # NaN.  The scratch tensor is a shape carrier, not initialized
            # probability state.
            rows.append(own(ops.zeros_like(scratch)))
            continue
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 < seed < _INACTIVE_DEVICE_SEED:
            raise ValueError("active uniform seeds must be uint32 values excluding zero and all-ones")
        # TTNN's public uniform wrapper converts the requested half-open
        # interval [0, 1) to inclusive *representable* endpoints, so 1.0 is
        # never emitted.  Calling that contract directly avoids a second
        # rounding step through an integer-valued FP32 range.
        # ttnn.uniform is explicitly in-place and returns ``scratch``.  That
        # tensor is borrowed caller state reused across tokens, so it must not
        # enter this call's owned scratch list.
        rows.append(ops.uniform(scratch, 0.0, 1.0, seed=seed))
    return own(ops.concat(rows, dim=2))


def sample_unrestricted_top_p_one(
    logits,
    *,
    inverse_temperature,
    row_scratch: Sequence[object],
    seed_values: Sequence[int],
    active_rows: Sequence[bool],
    vocab_size: int,
    ops,
    trace_enabled: bool = False,
) -> DeviceCategoricalPrototypeResult:
    """Sample unrestricted categorical rows and return device token IDs.

    ``logits`` must be a single-device or already-replicated FP32-compatible
    tensor shaped ``[1,1,B,W]``.  ``W`` may include a tile tail, but every tail
    logit at ``vocab_size:W`` must already be ``-inf``.  The prototype cannot
    inspect device values to enforce that semantic condition; its eventual
    admission contract must bind a producer-side tail-mask receipt.

    No token or vocabulary tensor is copied to host.  Scalar request seeds are
    host control inputs, matching the existing manual-seed ABI.  Dynamic public
    ops allocate scratch, so trace capture is rejected until fixed buffers and
    lifetime ownership are implemented and qualified.
    """

    if trace_enabled:
        raise RuntimeError("unrestricted categorical prototype is not trace-safe")
    shape = _shape(logits)
    if len(shape) != 4 or shape[:2] != (1, 1):
        raise ValueError("logits must have shape [1,1,B,W]")
    batch, width = shape[2], shape[3]
    if batch <= 0 or batch > 32 or width < 32 or width % 32:
        raise ValueError("prototype requires 1<=B<=32 and tile-aligned W>=32")
    if not 0 < vocab_size <= width:
        raise ValueError("vocab_size must be in (0, W]")
    if _shape(inverse_temperature) != (1, 1, batch, 1):
        raise ValueError("inverse_temperature must have device shape [1,1,B,1]")
    if len(active_rows) != batch or not any(active_rows):
        raise ValueError("active_rows must identify at least one of the B rows")

    owned: list[object] = []

    def own(tensor):
        owned.append(tensor)
        return tensor

    try:
        # The categorical accumulation is an FP32 contract.  BF16 logits are a
        # valid producer format, but must be widened before max/sub/exp/scan.
        logits_fp32 = logits if logits.dtype == ops.float32 else own(ops.typecast(logits, dtype=ops.float32))
        inverse_temperature_fp32 = (
            inverse_temperature
            if inverse_temperature.dtype == ops.float32
            else own(ops.typecast(inverse_temperature, dtype=ops.float32))
        )
        maximum = own(ops.max(logits_fp32, dim=-1, keepdim=True))
        centered = own(ops.subtract(logits_fp32, maximum))
        scaled = own(ops.multiply(centered, inverse_temperature_fp32))
        weights = own(ops.exp(scaled))
        cdf = _prefix_sum_fp32(weights, dim=-1, ops=ops, own=own)
        total = own(ops.slice(cdf, [0, 0, 0, width - 1], [1, 1, batch, width]))
        draws = _per_slot_uniform_rows(
            row_scratch=row_scratch,
            seed_values=seed_values,
            active_rows=active_rows,
            ops=ops,
            own=own,
        )
        threshold = own(ops.multiply(total, draws))
        above = own(ops.gt(cdf, threshold))
        token_ids = own(ops.argmax(above, dim=-1, keepdim=True))
        any_selected = own(ops.max(above, dim=-1, keepdim=True))
        finite_total = own(ops.isfinite(total))
        positive_total = own(ops.gt(total, 0.0))
        valid_distribution = own(
            ops.logical_and(own(ops.logical_and(finite_total, positive_total)), any_selected)
        )
        unique_owned = tuple({id(tensor): tensor for tensor in owned}.values())
        return DeviceCategoricalPrototypeResult(token_ids, valid_distribution, unique_owned)
    except Exception:
        if hasattr(ops, "deallocate"):
            _release_owned(owned, ops=ops)
        raise
