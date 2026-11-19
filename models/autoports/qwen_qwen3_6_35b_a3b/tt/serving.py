# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-model explicit decode commands; no plugin adapter or capability claim.

Source implementation awaiting hardware validation. Only prepared model and
sampling traces execute here. Sampling-parameter presence selects device token
output; absence selects full host logits, whose caller owns host sampling.
"""

from math import isfinite
from numbers import Integral

import numpy as np
import torch

import ttnn

from .serving_state import ReloadCommands, normalize_remap


def _integer_rows(value, name):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in ("i", "u"):
            raise ValueError(f"{name} NumPy arrays must have integer dtype")
        value = value.reshape(-1).tolist()
    elif isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be CPU metadata")
        value = value.reshape(-1).tolist()
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a list or CPU tensor")
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in value):
        raise ValueError(f"{name} must contain integer values")
    return tuple(int(v) for v in value)


def _field(params, name, default):
    return params.get(name, default) if isinstance(params, dict) else getattr(params, name, default)


def _parameter_rows(params, name, default, count):
    value = _field(params, name, default)
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Sampling parameters must be CPU metadata")
        value = value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            raise ValueError(f"Sampling {name} row count must match submitted slots")
        return list(value)
    return [value] * count


def _validate_sampling(values, *, device_sampling, device_top_k_limit):
    for temperature, top_k, top_p, seed in zip(*(values[key] for key in ("temperature", "top_k", "top_p", "seed"))):
        if not isinstance(temperature, (float, int)) or not isfinite(temperature) or temperature < 0:
            raise ValueError("Temperature must be finite and nonnegative")
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < -1:
            raise ValueError("top_k must be an integer >= -1")
        if not isinstance(top_p, (float, int)) or not isfinite(top_p) or not 0 <= top_p <= 1:
            raise ValueError("top_p must be finite and within [0,1]")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise ValueError("Seed must be an integer or None")
        if device_sampling and temperature > 0 and (top_k <= 0 or top_k > device_top_k_limit):
            raise ValueError(
                f"Device sampling supports top_k <= {device_top_k_limit}; use explicit host-logit mode for this request"
            )


def _neutral_device_features(params, count):
    for name, default in (
        ("presence_penalty", 0.0),
        ("frequency_penalty", 0.0),
        ("repetition_penalty", 1.0),
        ("enable_log_probs", False),
        ("num_logprobs", 0),
    ):
        values = _parameter_rows(params, name, default, count)
        if any(value not in (None, default) for value in values):
            raise ValueError(f"Device {name} is not supported by this prepared sampler; request host logits")
    if _field(params, "grammar_bitmask", None) is not None:
        raise ValueError("Grammar sampling requires host logits")


def _history_after_reset(old, incoming, rows, order, vocab_size, reset_slots):
    """Build the host shadow before any mutation, preserving off-batch owners."""
    if incoming is None:
        new = torch.empty((len(rows), 0), dtype=torch.int64)
    else:
        new = torch.as_tensor(incoming)
        if new.device.type != "cpu" or new.ndim != 2 or new.shape[0] != len(rows):
            raise ValueError("Token history must be a CPU integer [submitted rows, history] tensor")
        if new.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint32,
            torch.uint64,
        ):
            raise ValueError("Token history must be integer")
        new = new.to(torch.int64)
        if ((new < -1) | (new >= vocab_size)).any():
            raise ValueError("Token history contains a token outside the real vocabulary")
    if old is not None:
        if old.shape[0] != 32:
            raise ValueError("Remembered sampler history must have 32 physical rows")
        old = old[list(order) + list(range(len(order), 32))]
    width = max(new.shape[1], 0 if old is None else old.shape[1])
    result = torch.full((32, width), -1, dtype=torch.int64)
    if old is not None:
        result[:, : old.shape[1]] = old
    for row, slot in enumerate(rows):
        if slot not in reset_slots:
            continue
        result[slot] = -1
        result[slot, : new.shape[1]] = new[row]
    return result


def _validated_pages(generator, page_table, page_tables_per_layer, rows, active, positions):
    groups = [states for states in generator.cache if "page_table" in states[0]]
    if positions is not None:
        for group in groups:
            for row, slot in enumerate(rows):
                state = group[slot]
                limit = state.get("context_length", state["page_table"].shape[-1] * state["k"].shape[-2])
                if slot in active and positions[row] >= limit:
                    raise ValueError("The requested position exceeds this cache's allocated context")
    if page_table is None and page_tables_per_layer is None:
        return None
    tables = [page_table] * len(groups) if page_tables_per_layer is None else list(page_tables_per_layer)
    if len(tables) != len(groups):
        raise ValueError("Provide one page table per full-attention layer")
    normalized = []
    for group, table in zip(groups, tables):
        value = torch.as_tensor(table)
        if value.device.type != "cpu" or value.ndim != 2 or value.shape[0] < len(rows):
            raise ValueError("Page tables must be CPU [submitted rows, logical pages] tensors")
        # The pinned runner can supply group zero plus authoritative per-layer
        # tables, padded to capacity. The submitted row-order prefix is used.
        value = value[: len(rows)]
        if value.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint32,
            torch.uint64,
        ):
            raise ValueError("Page table entries must be integer")
        width = group[0]["page_table"].shape[-1]
        if value.shape[1] > width:
            raise ValueError("Page table exceeds the prepared logical context capacity")
        pages = torch.full((len(rows), width), -1, dtype=torch.int32)
        values = value.to(torch.int64)
        physical_pages = group[0]["k"].shape[0]
        if ((values < -1) | (values >= physical_pages)).any():
            raise ValueError("Page table references a page outside this cache's physical pool")
        pages[:, : value.shape[1]] = values.to(torch.int32)
        for row, slot in enumerate(rows):
            # Attention may prefetch causal padding. Keep every stored entry a
            # physical page, matching the decoder's last-owned-page padding.
            count = int((pages[row] >= 0).sum())
            if count and (pages[row, :count] < 0).any():
                raise ValueError("Owned logical pages must form a contiguous prefix")
            if slot in active and not count:
                raise ValueError("An active row requires at least one physical page")
        if positions is not None:
            block_size = group[0]["k"].shape[-2]
            for row, slot in enumerate(rows):
                if slot in active and (pages[row, : positions[row] // block_size + 1] < 0).any():
                    raise ValueError("An active decode row lacks a required physical page")
        for row in range(len(rows)):
            owned = pages[row][pages[row] >= 0]
            pages[row][pages[row] < 0] = owned[-1] if len(owned) else 0
        normalized.append(pages)
    return normalized


def decode_with_commands(
    generator,
    tokens,
    start_pos,
    *,
    page_table=None,
    page_tables_per_layer=None,
    kv_cache=None,
    sampling_params=None,
    prompt_tokens=None,
    output_tokens=None,
    slot_remap=None,
    slots=None,
    reload_inputs=True,
    reload_page_table=False,
    reload_sampling_params=False,
    reset_sampling_state=False,
    read_from_device=True,
    enable_trace=True,
):
    """Submit ordinary width-one decode and optionally consume its owned result.

    The public v1 wrapper must reject legacy/unknown keywords rather than swallow
    them. Nondefault device penalties/logprobs are rejected before mutation;
    the host-logit caller owns those operations. No host sampling occurs here.
    """
    commands = ReloadCommands(reload_inputs, reload_page_table, reload_sampling_params, reset_sampling_state)
    if type(read_from_device) is not bool or type(enable_trace) is not bool:
        raise TypeError("Decode read/trace selectors must be booleans")
    if not enable_trace:
        raise ValueError("This prepared decode contract requires tracing")
    if generator.closed or not generator.prepared:
        raise RuntimeError("Prepare an open generator before command-based decode")
    if kv_cache is not None and kv_cache is not generator.cache:
        raise ValueError("Decode cache must be the stable cache prepared with this generator")
    if generator.reload_executor.poisoned or generator.slot_state.poisoned:
        raise RuntimeError("Reconstruct authoritative state after a poisoned decode")
    capacity = generator.max_batch_size
    order = normalize_remap(slot_remap, capacity)
    device_sampling = sampling_params is not None
    previous_mode = getattr(generator, "serving_device_sampling", None)
    if previous_mode is not None and previous_mode != device_sampling:
        if not commands.reload_inputs:
            raise ValueError("Sampling-mode transitions require authoritative input reload")
        if device_sampling and not (commands.reload_sampling_params and commands.reset_sampling_state):
            raise ValueError("Host-to-device sampling requires parameter reload and state reset")

    # Never inspect tokens/start_pos in the resident branch. Even their shape
    # can belong to an earlier submission under the pinned async contract.
    positions = token_values = None
    if commands.reload_inputs:
        positions = _integer_rows(start_pos, "start_pos")
        rows = tuple(range(len(positions))) if slots is None else _integer_rows(slots, "slots")
        if len(rows) != len(positions):
            raise ValueError("Position rows and physical slots must match")
        if any(pos < -1 or pos >= generator.model.max_seq_len for pos in positions):
            raise ValueError("Decode position is outside the supported context")
        source = torch.as_tensor(tokens)
        if source.device.type != "cpu" or tuple(source.shape) != (len(rows), 1):
            raise ValueError("Ordinary decode tokens must be a CPU integer [rows,1] tensor")
        token_values = _integer_rows(source, "tokens")
        active = tuple(slot for slot, position in zip(rows, positions) if position >= 0)
        if any(
            not 0 <= token_values[row] < generator.model.vocab_size
            for row, position in enumerate(positions)
            if position >= 0
        ):
            raise ValueError("Active input token is outside the real vocabulary")
    else:
        remembered = getattr(generator, "serving_rows", tuple(range(capacity)))
        rows = remembered if slots is None else _integer_rows(slots, "slots")
        if tuple(rows) != tuple(remembered):
            raise ValueError("Changing the submitted slot layout requires authoritative inputs")
        old_active = set(generator.active_slots)
        active = tuple(i for i in range(capacity) if order[i] in old_active)
    if not rows or len(set(rows)) != len(rows) or any(slot < 0 or slot >= capacity for slot in rows):
        raise ValueError("Submitted slots must be nonempty, unique and within prepared capacity")
    if any(slot not in rows for slot in active):
        raise ValueError("The submitted layout omits a resident active slot")
    rows = tuple(rows)
    if active and generator.max_batch_size > 1 and "model_fixed" not in generator.traces:
        raise ValueError("This active layout has no prepared model trace")
    trace_key = generator.model_trace_key(active)
    if active and trace_key not in generator.traces:
        raise ValueError("This active layout has no prepared model trace")
    if device_sampling and not {"sampling", "sampling_64"}.issubset(generator.traces):
        raise ValueError("Device sampling has no prepared trace")
    if generator.sampler._penalties_active or generator.sampler.tt_sampling.log_probs_calculator.enable_log_probs:
        raise ValueError("This entry point requires the prepared neutral device penalty/logprob configuration")

    pages = None
    if commands.reload_inputs or commands.reload_page_table:
        pages = _validated_pages(generator, page_table, page_tables_per_layer, rows, active, positions)
        if commands.reload_page_table and pages is None:
            raise ValueError("Page-only reload requires explicit page tables")

    if commands.reload_inputs and any("page_table" in group[0] for group in generator.cache):
        from .cache_contract import validate_page_ownership

        ownership_pages = pages
        if ownership_pages is None:
            ownership_pages = []
            for group in generator.cache:
                if "page_table" not in group[0]:
                    continue
                shadows = [group[order[slot]].get("_host_pages") for slot in rows]
                if any(shadow is None for shadow in shadows):
                    raise ValueError("Provide explicit page mappings when no CPU page-table shadow exists")
                ownership_pages.append(torch.cat(shadows, dim=0))
        # Authoritative positions define the complete visible logical prefix.
        # Inactive rows own no submitted pages. Resident/page-only commands do
        # not read back device positions: their extents and off-batch ownership
        # remain the scheduler's responsibility.
        validate_page_ownership(
            generator.cache,
            ownership_pages,
            slots=rows,
            prompt_lens=tuple(max(position + 1, 0) for position in positions),
        )

    # Project remembered parameters through the same full ownership permutation
    # before validating. Parameter uploads never implicitly reset live streams.
    values = {
        key: [generator.sampling_values[key][order[slot]] for slot in rows]
        for key in ("temperature", "top_k", "top_p", "seed")
    }
    if commands.reload_sampling_params and sampling_params is not None:
        _neutral_device_features(sampling_params, len(rows))
        values = {
            key: _parameter_rows(sampling_params, key, default, len(rows))
            for key, default in (("temperature", 0.0), ("top_k", 1), ("top_p", 1.0), ("seed", None))
        }
    _validate_sampling(
        values, device_sampling=device_sampling, device_top_k_limit=generator.sampler.tt_sampling.max_top_k
    )
    # configure_sampling's policy also rejects unbounded device parameters when
    # the owning generator is configured for device mode, even for a host call.
    if commands.reload_sampling_params or commands.reset_sampling_state:
        _validate_sampling(
            values,
            device_sampling=not generator.host_sampling,
            device_top_k_limit=generator.sampler.tt_sampling.max_top_k,
        )
    prompt_history = output_history = None
    if commands.reset_sampling_state:
        prompt_history = _history_after_reset(
            generator.sampler.tt_penalties._prompt_tokens_host,
            prompt_tokens,
            rows,
            order,
            generator.model.vocab_size,
            active,
        )
        output_history = _history_after_reset(
            getattr(generator, "serving_output_history", None),
            output_tokens,
            rows,
            order,
            generator.model.vocab_size,
            active,
        )
    elif prompt_tokens is not None or output_tokens is not None:
        # Histories are ignored when no state reset is commanded, just as
        # parameter payloads are ignored without parameter reload.
        pass

    changing = (
        order != tuple(range(capacity))
        or commands.reload_inputs
        or commands.reload_sampling_params
        or commands.reset_sampling_state
    )
    if changing:
        generator.output_pool.require_drained()
    if all(lease is not None for lease in generator.output_pool.leases):
        raise RuntimeError("Consume a pending decode output before submitting another")

    def reload_pages():
        if pages is not None:
            generator._bind_pages(None, generator.cache, rows, page_tables_per_layer=pages)

    def reload_forward():
        generator._set_active_slots(active)
        if active:
            selected = [row for row, pos in enumerate(positions) if pos >= 0]
            generator._refresh([token_values[row] for row in selected], [positions[row] for row in selected], active)
        reload_pages()
        generator.serving_rows = rows

    def configure(reset):
        selected = [row for row, slot in enumerate(rows) if slot in active]
        if selected:
            parameters = {key: [entries[row] for row in selected] for key, entries in values.items()}
            generator.configure_sampling(**parameters, slots=[rows[row] for row in selected], reset_seed_state=reset)

    def reset_sampling():
        configure(True)
        # Active device penalties are deliberately unsupported. Their neutral
        # buffers are never accumulated by this trace. Preserve/reset requested
        # history shadows for ownership, without allocating device masks late.
        generator.sampler.tt_penalties._prompt_tokens_host = prompt_history
        generator.serving_output_history = output_history

    def submit():
        handle = generator.output_pool.reserve(rows, is_tokens=device_sampling)
        try:
            if active:
                ttnn.execute_trace(generator.mesh, generator.traces[trace_key], cq_id=0, blocking=False)
                generator.trace_replays[trace_key] = generator.trace_replays.get(trace_key, 0) + 1
                generator.counters["model_replays"] += 1
                if device_sampling:
                    # Explicitly bypass _sample(), whose compatibility mode may
                    # read logits and sample on the host.
                    generator.replay_sampling()
            generator.output_pool.publish(handle, generator.tokens if device_sampling else generator.sampler_logits)
        except BaseException:
            generator.output_pool.discard(handle)
            raise
        generator.serving_device_sampling = device_sampling
        return handle

    result = generator.reload_executor.execute(
        commands,
        slot_remap=order,
        reload_forward=reload_forward,
        reload_pages=reload_pages,
        reload_parameters=lambda: configure(False),
        reset_sampling=reset_sampling,
        forward=submit,
    )
    if not read_from_device:
        return result
    try:
        host = generator.output_pool.process(result, is_tokens=device_sampling)
        generator.counters["readbacks"] += 1
        return host
    except BaseException:
        # The forward was accepted; read failure is not a second forward and
        # must not roll back token feedback. Drain ownership before recovery.
        generator.output_pool.discard(result)
        raise


def prefill_with_sampling(
    generator,
    tokens,
    *,
    low_level,
    prompt_lens,
    start_pos=None,
    page_table=None,
    page_tables_per_layer=None,
    kv_cache=None,
    slots=None,
    empty_slots=None,
    sampling_params=None,
    read_from_device=True,
    enable_trace=False,
):
    """Stage final prefill logits and return sampled tokens or full host logits.

    ``low_level`` is the existing unsampled prefill implementation, explicitly
    supplied to avoid recursion through a public serving wrapper. It receives
    ``device_output=True`` and returns one singleton logit tensor per row. The
    existing bounded executor owns prompt padding/chunks, masks, positions, KV
    fill and recurrence boundaries. Only the existing sampler trace samples.
    """
    if type(read_from_device) is not bool or type(enable_trace) is not bool:
        raise TypeError("Prefill read/trace selectors must be booleans")
    if enable_trace:
        raise ValueError("Prefill graph tracing is not prepared; use decode-only tracing")
    if not callable(low_level):
        raise TypeError("An explicit unsampled prefill implementation is required")
    if generator.closed or not generator.prepared or not generator.prefill_logit_rows.prepared:
        raise RuntimeError("Prepare generator traces and prefill-logit staging before serving prefill")
    if generator.reload_executor.poisoned or generator.slot_state.poisoned:
        raise RuntimeError("Reconstruct authoritative state after a poisoned submission")
    if kv_cache is not None and kv_cache is not generator.cache:
        raise ValueError("Prefill must use its prepared cache owner")
    if slots is not None and empty_slots is not None:
        raise ValueError("Specify slots or empty_slots, not both")
    ends = _integer_rows(prompt_lens, "prompt_lens")
    selected = empty_slots if slots is None else slots
    rows = tuple(range(len(ends))) if selected is None else _integer_rows(selected, "slots")
    starts = (0,) * len(rows) if start_pos is None else _integer_rows(start_pos, "start_pos")
    if not rows or len(rows) != len(ends) or len(starts) != len(rows):
        raise ValueError("Prompt lengths, positions and physical slots must have equal nonempty row counts")
    if len(set(rows)) != len(rows) or any(slot < 0 or slot >= generator.max_batch_size for slot in rows):
        raise ValueError("Prefill slots must be unique and within prepared capacity")
    source = torch.as_tensor(tokens)
    if source.device.type != "cpu" or source.ndim != 2 or source.shape[0] != len(rows):
        raise ValueError("Prefill tokens must be a CPU integer [rows,padded prompt length] tensor")
    for row, (start, end) in enumerate(zip(starts, ends)):
        if not 0 <= start < end <= generator.model.max_seq_len or end > source.shape[1]:
            raise ValueError("Invalid prefill logical interval or input padding")
        values = _integer_rows(source[row, start:end], "prompt tokens")
        if any(token < 0 or token >= generator.model.vocab_size for token in values):
            raise ValueError("Prompt token lies outside the real vocabulary")
    pages = _validated_pages(generator, page_table, page_tables_per_layer, rows, rows, tuple(end - 1 for end in ends))
    if len(rows) > 1 and pages is None:
        raise ValueError("Mixed prefill requires explicit disjoint page mappings")
    if pages is not None:
        # Every used logical page must be owned uniquely across submitted rows.
        # Off-batch cache ownership remains the scheduler/caller's obligation.
        full_groups = [group for group in generator.cache if "page_table" in group[0]]
        for table, group in zip(pages, full_groups):
            owned = set()
            for row, end in enumerate(ends):
                count = (end + group[0]["k"].shape[-2] - 1) // group[0]["k"].shape[-2]
                used = table[row, :count].tolist()
                if len(set(used)) != len(used) or owned.intersection(used):
                    raise ValueError("Live prefill page ownership overlaps within this submission")
                owned.update(used)
    device_sampling = sampling_params is not None
    values = None
    if device_sampling:
        _neutral_device_features(sampling_params, len(rows))
        values = {
            key: _parameter_rows(sampling_params, key, default, len(rows))
            for key, default in (("temperature", 0.0), ("top_k", 1), ("top_p", 1.0), ("seed", None))
        }
        _validate_sampling(values, device_sampling=True, device_top_k_limit=generator.sampler.tt_sampling.max_top_k)
        if not {"sampling", "sampling_64"}.issubset(generator.traces):
            raise RuntimeError("Prefill device sampling requires the prepared sampling trace")
    if generator.sampler._penalties_active or generator.sampler.tt_sampling.log_probs_calculator.enable_log_probs:
        raise ValueError("Use host-logit sampling for nondefault penalties/logprobs")
    # Build requested prompt-history shadow entirely on CPU before state changes.
    history = source.to(torch.int64).clone()
    for row, end in enumerate(ends):
        history[row, end:] = -1
    prompt_history = _history_after_reset(
        generator.sampler.tt_penalties._prompt_tokens_host,
        history,
        rows,
        tuple(range(generator.max_batch_size)),
        generator.model.vocab_size,
        rows,
    )
    output_history = _history_after_reset(
        getattr(generator, "serving_output_history", None),
        None,
        rows,
        tuple(range(generator.max_batch_size)),
        generator.model.vocab_size,
        rows,
    )
    generator.output_pool.require_drained()
    handle = None
    logits = []
    try:
        logits = low_level(
            source,
            prompt_lens=ends,
            start_pos=starts,
            slots=rows,
            kv_cache=generator.cache,
            page_tables_per_layer=pages,
            device_output=True,
            return_all_logits=False,
        )
        if not isinstance(logits, (list, tuple)) or len(logits) != len(rows):
            raise RuntimeError("Unsampled prefill must return one final device logit tensor per submitted row")
        # The low-level path checks that earlier outputs are drained itself.
        # Reserve only after it returns, so it cannot mistake this submission's
        # lease for an outstanding earlier decode.
        handle = generator.output_pool.reserve(rows, is_tokens=device_sampling)
        for tensor, slot in zip(logits, rows):
            generator.prefill_logit_rows.copy(tensor, generator.sampler_logits, slot)
        # Sampling traces must not observe live late allocations. The queue has
        # consumed these inputs before their storage can be reused by replay.
        for tensor in logits:
            ttnn.deallocate(tensor)
        logits = []
        generator._set_active_slots(rows)
        if device_sampling:
            generator.configure_sampling(**values, slots=rows, reset_seed_state=True)
            generator.replay_sampling()
        generator.sampler.tt_penalties._prompt_tokens_host = prompt_history
        generator.serving_output_history = output_history
        generator.output_pool.publish(handle, generator.tokens if device_sampling else generator.sampler_logits)
        # A later decode establishes its own row layout; mode is retained so a
        # host-to-device transition cannot silently skip the mandated reset.
        generator.serving_rows = rows
        generator.serving_device_sampling = device_sampling
    except BaseException:
        for tensor in logits:
            ttnn.deallocate(tensor)
        generator.reload_executor.poisoned = generator.slot_state.poisoned = True
        if handle is not None:
            generator.output_pool.discard(handle)
        raise
    if not read_from_device:
        return handle
    try:
        result = generator.output_pool.process(handle, is_tokens=device_sampling)
        generator.counters["readbacks"] += 1
        return result
    except BaseException:
        generator.output_pool.discard(handle)
        raise
