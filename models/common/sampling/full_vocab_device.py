"""Staged public-op composition for unrestricted device sampling.

This module is wired only through an explicitly enabled staged route in
:class:`SamplingGenerator` and does not advertise a runtime capability.  It
keeps logits, probability math,
random variates, and the B-row token result on device while the slower
per-row scalar-seed uniform composition is qualified.  The nucleus helper
uses a mathematically sufficient top-k envelope: for a vocabulary of width
``V``, the largest ``K`` probabilities always contain at least ``K / V`` of
the total mass.  Requests outside the documented public top-k envelope,
traces, and distributed/sharded logits fail closed rather than silently
narrowing the distribution.
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


@dataclass(frozen=True)
class ExactNucleusCandidatePlan:
    """Host control plan for an exact device nucleus candidate envelope."""

    candidate_count: int
    maximum_supported_top_p: float


@dataclass(frozen=True)
class StableTopKResult:
    """Stable values/global vocabulary IDs plus exact owned allocations."""

    values: object
    global_indices: object
    owned_tensors: tuple[object, ...]


@dataclass(frozen=True)
class DeviceSelectedLogprobResult:
    """Raw full-vocabulary selected-token logprobs and device validity."""

    logprobs: object
    valid_distribution: object
    owned_tensors: tuple[object, ...]


def plan_exact_nucleus_candidates(
    top_p: float,
    *,
    vocab_size: int,
    public_topk_max_candidates: int,
    tile_width: int = 32,
) -> ExactNucleusCandidatePlan:
    """Return the smallest tile-aligned top-k envelope guaranteed exact.

    If probabilities are sorted descending, the sum of their first ``K``
    entries is at least ``K / vocab_size``.  Therefore rounding
    ``ceil(top_p * vocab_size)`` up to a tile boundary is sufficient for every
    possible finite distribution, not merely for typical language-model
    logits.  The public operation's qualified K limit is an explicit input;
    exceeding it is an unsupported request domain, never a reason to clamp
    the user's ``top_p``.
    """

    import math

    if not math.isfinite(top_p) or not 0.0 < top_p < 1.0:
        raise ValueError("exact nucleus planning requires finite 0 < top_p < 1")
    if vocab_size <= 0 or public_topk_max_candidates <= 0 or tile_width <= 0:
        raise ValueError("vocab_size, public top-k limit, and tile width must be positive")
    required = math.ceil(top_p * vocab_size)
    candidate_count = min(vocab_size, tile_width * math.ceil(required / tile_width))
    if candidate_count > public_topk_max_candidates:
        supported = min(vocab_size, public_topk_max_candidates) / vocab_size
        raise ValueError(
            "requested top_p exceeds the exact public top-k nucleus envelope "
            f"(requested={top_p}, maximum={supported}, vocab_size={vocab_size}, "
            f"max_candidates={public_topk_max_candidates})"
        )
    return ExactNucleusCandidatePlan(candidate_count, candidate_count / vocab_size)


def hierarchical_stable_topk(
    logits,
    *,
    k: int,
    max_local_width: int,
    ops,
    tile_width: int = 32,
) -> StableTopKResult:
    """Return exact stable top-k using a qualified narrow-index local route.

    The caller supplies the runtime-qualified maximum local width.  Every
    contiguous chunk uses native stable top-k while its local indices fit the
    runtime's narrow index representation.  Local indices are widened only
    after selection and offset to global vocabulary IDs.  Pairwise merges
    concatenate the lower-ID group first and stable-top-k the ``<=2*k``
    winners, preserving global lowest-ID tie order inductively.

    This is intentionally fail-closed when the requested ``k`` cannot fit in
    every balanced tile-aligned chunk.  It does not infer a resource envelope
    from a model name or silently select the unstable wide-index kernel.
    """

    import math

    shape = _shape(logits)
    if len(shape) != 4 or shape[:2] != (1, 1):
        raise ValueError("logits must have shape [1,1,B,W]")
    width = shape[-1]
    if min(k, max_local_width, tile_width) <= 0:
        raise ValueError("k, max_local_width, and tile_width must be positive")
    if width % tile_width or max_local_width % tile_width:
        raise ValueError("width and max_local_width must be tile aligned")
    if k > width:
        raise ValueError("k must not exceed logits width")

    # Balance whole tiles so the final chunk is never a too-small remainder.
    chunk_count = math.ceil(width / max_local_width)
    total_tiles = width // tile_width
    if chunk_count > total_tiles:
        raise ValueError("qualified local width cannot cover one tile per chunk")
    base_tiles, extra = divmod(total_tiles, chunk_count)
    chunk_widths = [tile_width * (base_tiles + (1 if i < extra else 0)) for i in range(chunk_count)]
    if min(chunk_widths) < k or max(chunk_widths) > max_local_width:
        raise ValueError("k does not fit the qualified balanced local top-k chunks")
    merge_width = tile_width * math.ceil((2 * k) / tile_width)
    if chunk_count > 1 and merge_width > max_local_width:
        raise ValueError("pairwise top-k merge exceeds the qualified local width")

    owned: list[object] = []

    def own(tensor):
        owned.append(tensor)
        return tensor

    def release_now(tensors):
        if not hasattr(ops, "deallocate"):
            return
        unique = {id(tensor): tensor for tensor in tensors}
        _release_owned(tuple(unique.values()), ops=ops)
        released_ids = set(unique)
        owned[:] = [tensor for tensor in owned if id(tensor) not in released_ids]

    try:
        groups = []
        start = 0
        for chunk_width in chunk_widths:
            chunk = own(
                ops.slice(
                    logits,
                    [0, 0, 0, start],
                    [shape[0], shape[1], shape[2], start + chunk_width],
                )
            )
            values, local_indices = ops.topk(
                chunk,
                k=k,
                dim=-1,
                largest=True,
                sorted=True,
                stable=True,
            )
            own(values)
            own(local_indices)
            if local_indices.dtype != ops.uint16:
                raise RuntimeError(
                    "qualified hierarchical stable top-k requires uint16 local indices"
                )
            local_global_indices = own(ops.typecast(local_indices, dtype=ops.uint32))
            if local_global_indices is local_indices:
                raise RuntimeError("uint16-to-uint32 local index widening must allocate distinct storage")
            if start:
                global_indices = own(ops.add(local_global_indices, start, dtype=ops.uint32))
                release_now((local_global_indices,))
            else:
                global_indices = local_global_indices
            groups.append((values, global_indices))
            release_now((chunk, local_indices))
            start += chunk_width

        while len(groups) > 1:
            next_groups = []
            for index in range(0, len(groups), 2):
                if index + 1 == len(groups):
                    next_groups.append(groups[index])
                    continue
                left_values, left_ids = groups[index]
                right_values, right_ids = groups[index + 1]
                merged_values = own(ops.concat((left_values, right_values), dim=-1))
                merged_ids = own(ops.concat((left_ids, right_ids), dim=-1))
                winners, positions = ops.topk(
                    merged_values,
                    k=k,
                    dim=-1,
                    largest=True,
                    sorted=True,
                    stable=True,
                )
                own(winners)
                own(positions)
                winner_ids = own(ops.gather(merged_ids, dim=-1, index=positions))
                release_now(
                    (
                        left_values,
                        left_ids,
                        right_values,
                        right_ids,
                        merged_values,
                        merged_ids,
                        positions,
                    )
                )
                next_groups.append((winners, winner_ids))
            groups = next_groups

        values, global_indices = groups[0]
        return StableTopKResult(
            values,
            global_indices,
            tuple({id(tensor): tensor for tensor in owned}.values()),
        )
    except Exception:
        if hasattr(ops, "deallocate"):
            _release_owned(owned, ops=ops)
        raise


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


def _normalize_gather_index_layout(index, values, *, ops):
    """Match the public gather index layout to its values tensor."""

    if index.layout == values.layout:
        return index
    return ops.to_layout(index, values.layout)


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


def calculate_selected_raw_logprobs(
    raw_logits,
    selected_token_ids,
    *,
    vocab_size: int,
    ops,
) -> DeviceSelectedLogprobResult:
    """Compute selected-token raw log-softmax on device over the full vocab.

    ``raw_logits`` is the model-head output before penalties, temperature,
    top-k, or nucleus masking.  Its padded tail must already be ``-inf``.
    This recomputes a full FP32 normalizer and never reuses the
    temperature-scaled categorical CDF or truncated nucleus mass.
    """

    shape = _shape(raw_logits)
    if len(shape) != 4 or shape[:2] != (1, 1):
        raise ValueError("raw_logits must have shape [1,1,B,W]")
    batch, width = shape[2:]
    if batch <= 0 or batch > 32 or width < 32 or width % 32:
        raise ValueError("raw logprob path requires 1<=B<=32 and tile-aligned W>=32")
    if not 0 < vocab_size <= width:
        raise ValueError("vocab_size must be in (0, W]")
    if _shape(selected_token_ids) != (1, 1, 1, batch):
        raise ValueError("selected_token_ids must have shape [1,1,1,B]")

    owned: list[object] = []

    def own(tensor):
        owned.append(tensor)
        return tensor

    try:
        logits_fp32 = (
            raw_logits
            if raw_logits.dtype == ops.float32
            else own(ops.typecast(raw_logits, dtype=ops.float32))
        )
        maximum = own(ops.max(logits_fp32, dim=-1, keepdim=True))
        centered = own(ops.subtract(logits_fp32, maximum))
        weights = own(ops.exp(centered))
        cdf = _prefix_sum_fp32(weights, dim=-1, ops=ops, own=own)
        total = own(ops.slice(cdf, [0, 0, 0, width - 1], [1, 1, batch, width]))

        selected_by_row = ops.reshape(selected_token_ids, (1, 1, batch, 1))
        selected_for_gather = _normalize_gather_index_layout(selected_by_row, logits_fp32, ops=ops)
        if selected_for_gather is not selected_by_row:
            own(selected_for_gather)
        if selected_for_gather.dtype not in (ops.uint16, ops.uint32):
            selected_for_gather = own(ops.typecast(selected_for_gather, dtype=ops.uint32))
        token_in_vocab = own(ops.lt(selected_for_gather, vocab_size))
        valid_index_mask = own(ops.typecast(token_in_vocab, dtype=selected_for_gather.dtype))
        safe_selected_for_gather = own(ops.multiply(selected_for_gather, valid_index_mask))
        selected_logits = own(ops.gather(logits_fp32, dim=-1, index=safe_selected_for_gather))
        log_total = own(ops.log(total))
        logprobs = own(ops.subtract(own(ops.subtract(selected_logits, maximum)), log_total))

        finite_total = own(ops.isfinite(total))
        positive_total = own(ops.gt(total, 0.0))
        if token_in_vocab.dtype != finite_total.dtype:
            token_in_vocab = own(ops.typecast(token_in_vocab, dtype=finite_total.dtype))
        finite_logprob = own(ops.isfinite(logprobs))
        if finite_logprob.dtype != finite_total.dtype:
            finite_logprob = own(ops.typecast(finite_logprob, dtype=finite_total.dtype))
        valid = own(
            ops.logical_and(
                own(ops.logical_and(finite_total, positive_total)),
                own(ops.logical_and(token_in_vocab, finite_logprob)),
            )
        )
        # The plain sampled-logprob serving ABI has no side-band validity
        # tensor.  Keep validity for device tests, and poison an invalid row's
        # returned value so malformed IDs/distributions fail closed when the
        # serving adapter materializes the result.
        zero = own(ops.multiply(total, 0.0))
        negative_infinity = own(ops.log(zero))
        invalid_value = own(ops.subtract(negative_infinity, negative_infinity))
        logprobs = own(ops.where(valid, logprobs, invalid_value))
        return DeviceSelectedLogprobResult(
            logprobs,
            valid,
            tuple({id(tensor): tensor for tensor in owned}.values()),
        )
    except Exception:
        if hasattr(ops, "deallocate"):
            _release_owned(owned, ops=ops)
        raise


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


def sample_unrestricted_nucleus(
    logits,
    *,
    inverse_temperature,
    top_p,
    row_scratch: Sequence[object],
    seed_values: Sequence[int],
    active_rows: Sequence[bool],
    vocab_size: int,
    candidate_count: int,
    stable_topk_max_local_width: int = 0,
    ops,
    trace_enabled: bool = False,
) -> DeviceCategoricalPrototypeResult:
    """Sample exact unrestricted nucleus rows from a sufficient top-k set.

    ``candidate_count`` must come from :func:`plan_exact_nucleus_candidates`
    for the largest active ``top_p``.  A redundant device coverage predicate
    verifies that the selected candidates actually contain the requested
    probability mass before any token is accepted.  Both full-vocabulary and
    candidate CDFs use the explicitly qualified FP32 Hillis-Steele scan rather
    than the stock cumulative-sum operation.

    This helper returns global vocabulary IDs because ``ttnn.topk`` indices
    index the already-gathered full row.  It does not calculate logprobs.
    """

    if trace_enabled:
        raise RuntimeError("unrestricted nucleus prototype is not trace-safe")
    shape = _shape(logits)
    if len(shape) != 4 or shape[:2] != (1, 1):
        raise ValueError("logits must have shape [1,1,B,W]")
    batch, width = shape[2], shape[3]
    if batch <= 0 or batch > 32 or width < 32 or width % 32:
        raise ValueError("prototype requires 1<=B<=32 and tile-aligned W>=32")
    if not 0 < vocab_size <= width:
        raise ValueError("vocab_size must be in (0, W]")
    if not 0 < candidate_count <= vocab_size:
        raise ValueError("candidate_count must be positive and no larger than vocab_size")
    if hasattr(ops, "bfloat16") and logits.dtype != ops.bfloat16:
        raise ValueError("qualified nucleus top-k requires bfloat16 producer logits")
    expected_rows = (1, 1, batch, 1)
    if _shape(inverse_temperature) != expected_rows or _shape(top_p) != expected_rows:
        raise ValueError("temperature and top_p must have device shape [1,1,B,1]")
    if len(active_rows) != batch or not any(active_rows):
        raise ValueError("active_rows must identify at least one of the B rows")

    owned: list[object] = []

    def own(tensor):
        owned.append(tensor)
        return tensor

    try:
        logits_fp32 = logits if logits.dtype == ops.float32 else own(ops.typecast(logits, dtype=ops.float32))
        inverse_temperature_fp32 = (
            inverse_temperature
            if inverse_temperature.dtype == ops.float32
            else own(ops.typecast(inverse_temperature, dtype=ops.float32))
        )
        top_p_fp32 = top_p if top_p.dtype == ops.float32 else own(ops.typecast(top_p, dtype=ops.float32))

        # Temperature is positive for every row admitted to this route, so it
        # preserves descending order.  Run stable top-k on the producer BF16
        # logits so a tie at the nucleus boundary follows lowest-vocabulary-ID
        # order.  Exact runtime selection may use the slower stable engine for
        # wide K; performance is a separate qualification gate.  Candidate
        # probability math is widened and scaled below, identically to the
        # full-distribution denominator.
        scaled_logits = own(ops.multiply(logits_fp32, inverse_temperature_fp32))
        maximum = own(ops.max(scaled_logits, dim=-1, keepdim=True))
        centered = own(ops.subtract(scaled_logits, maximum))
        full_weights = own(ops.exp(centered))
        full_cdf = _prefix_sum_fp32(full_weights, dim=-1, ops=ops, own=own)
        full_total = own(ops.slice(full_cdf, [0, 0, 0, width - 1], [1, 1, batch, width]))

        if stable_topk_max_local_width and width > stable_topk_max_local_width:
            stable_topk = hierarchical_stable_topk(
                logits,
                k=candidate_count,
                max_local_width=stable_topk_max_local_width,
                ops=ops,
            )
            for tensor in stable_topk.owned_tensors:
                own(tensor)
            candidate_logits = stable_topk.values
            candidate_indices = stable_topk.global_indices
        else:
            candidate_logits, candidate_indices = ops.topk(
                logits,
                k=candidate_count,
                dim=-1,
                largest=True,
                sorted=True,
                stable=True,
            )
            own(candidate_logits)
            own(candidate_indices)
        candidate_logits_fp32 = own(ops.typecast(candidate_logits, dtype=ops.float32))
        candidate_scaled = own(ops.multiply(candidate_logits_fp32, inverse_temperature_fp32))
        candidate_weights = own(ops.exp(own(ops.subtract(candidate_scaled, maximum))))
        candidate_cdf = _prefix_sum_fp32(candidate_weights, dim=-1, ops=ops, own=own)
        candidate_total = own(
            ops.slice(
                candidate_cdf,
                [0, 0, 0, candidate_count - 1],
                [1, 1, batch, candidate_count],
            )
        )
        requested_mass = own(ops.multiply(full_total, top_p_fp32))

        # Standard nucleus semantics retain the first token that crosses p:
        # keep[i] iff cdf[i] - weight[i] < p * total.  This formulation avoids
        # a host-visible boundary and handles ties according to stable top-k.
        previous_mass = own(ops.subtract(candidate_cdf, candidate_weights))
        keep = own(ops.gt(requested_mass, previous_mass))
        keep_fp32 = own(ops.typecast(keep, dtype=ops.float32))
        nucleus_weights = own(ops.multiply(candidate_weights, keep_fp32))
        nucleus_cdf = _prefix_sum_fp32(nucleus_weights, dim=-1, ops=ops, own=own)
        nucleus_total = own(
            ops.slice(
                nucleus_cdf,
                [0, 0, 0, candidate_count - 1],
                [1, 1, batch, candidate_count],
            )
        )
        draws = _per_slot_uniform_rows(
            row_scratch=row_scratch,
            seed_values=seed_values,
            active_rows=active_rows,
            ops=ops,
            own=own,
        )
        draw_threshold = own(ops.multiply(nucleus_total, draws))
        above = own(ops.gt(nucleus_cdf, draw_threshold))
        candidate_position = own(ops.argmax(above, dim=-1, keepdim=True))
        # Exact740 argmax emits ROW_MAJOR while stable top-k indices are TILE;
        # public gather requires identical layouts for input and index tensors.
        candidate_position_for_gather = own(
            _normalize_gather_index_layout(candidate_position, candidate_indices, ops=ops)
        )
        token_ids = own(ops.gather(candidate_indices, dim=-1, index=candidate_position_for_gather))

        any_selected = own(ops.max(above, dim=-1, keepdim=True))
        token_in_vocab = own(ops.lt(token_ids, vocab_size))
        if token_in_vocab.dtype != any_selected.dtype:
            token_in_vocab = own(ops.typecast(token_in_vocab, dtype=any_selected.dtype))
        finite_total = own(ops.isfinite(full_total))
        positive_total = own(ops.gt(full_total, 0.0))
        # candidate_total >= requested_mass, expressed with the already
        # qualified strict comparison primitive so equality remains valid.
        candidate_coverage = own(ops.logical_not(own(ops.gt(requested_mass, candidate_total))))
        valid_distribution = own(
            ops.logical_and(
                own(ops.logical_and(finite_total, positive_total)),
                own(
                    ops.logical_and(
                        candidate_coverage,
                        own(ops.logical_and(any_selected, token_in_vocab)),
                    )
                ),
            )
        )
        unique_owned = tuple({id(tensor): tensor for tensor in owned}.values())
        return DeviceCategoricalPrototypeResult(token_ids, valid_distribution, unique_owned)
    except Exception:
        if hasattr(ops, "deallocate"):
            _release_owned(owned, ops=ops)
        raise
