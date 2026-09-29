"""Behavioral controls for traced prefill and device-owned token history."""

import argparse
import json
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path

import torch

import ttnn

from ..tt.generator import K2Generator


def install_retained_qkv_inputs(gen, *, inspect_first_token=False, norm_inputs=False, finish_inputs=False):
    """Keep exact intermediate owners without inserting any device operation."""
    active = False
    retained = {}
    metadata = {}
    address_metadata = {}
    captured = {}
    captured_metadata = {}
    captured_address_metadata = {}
    captured_trace = None
    scope = None
    invocation = None
    scope_id = 0
    invocation_id = 0
    capture_generation = 0
    gen.retained_snapshot_metadata = metadata
    gen.diagnostic_address_ledger = address_metadata
    gen._diagnostic_prefill_active = False

    def begin_scope(execution):
        nonlocal scope, scope_id
        scope_id += 1
        scope = {"request_id": scope_id, "execution": execution}
        retained.clear()
        metadata.clear()
        address_metadata.clear()

    def record_address(name, tensor, **details):
        """Copy host metadata only: no Tensor owner or allocation exemption."""
        if not active or invocation is None:
            raise RuntimeError("Prefill address records require an active prefill invocation")
        if name in address_metadata and address_metadata[name]["invocation_id"] != invocation["invocation_id"]:
            name = f"{name}_inv{invocation['invocation_id']}"
        padded_shape = list(tensor.padded_shape)
        address_metadata[name] = {
            **invocation,
            **details,
            "buffer_unique_id": tensor.buffer_unique_id(),
            "address": tensor.buffer_address(),
            "shape": list(tensor.shape),
            "padded_shape": padded_shape,
            "padded_elements": math.prod(padded_shape),
            "dtype": str(tensor.dtype),
            "layout": str(tensor.layout),
        }

    gen._diagnostic_record_tensor = record_address

    def keep(name, tensor):
        if not active or invocation is None:
            raise RuntimeError("Retained prefill tensors require an active prefill invocation")
        if inspect_first_token:
            from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

            # Diagnostic references are valid only immediately after prefill.
            # Decode deliberately overwrites these trace workspace buffers.
            TraceAllocationTracker.acknowledge_corruptible(tensor)
        # Preserve separate chunks/slots even when their physical shapes match.
        # Repeated taps of one tensor in the same invocation keep their usual key.
        if name in metadata and metadata[name]["invocation_id"] != invocation["invocation_id"]:
            name = f"{name}_inv{invocation['invocation_id']}"
        retained[name] = tensor
        metadata[name] = {
            **invocation,
            "buffer_unique_id": tensor.buffer_unique_id(),
            "address": tensor.buffer_address(),
            "shape": list(tensor.shape),
        }

    gen._diagnostic_keep_tensor = keep

    original_public_prefill = gen.prefill_forward

    def public_prefill(*args, **kwargs):
        nonlocal scope
        previous_scope = scope
        # One public request can contain multiple logical chunks (e.g. 4353).
        # Clear once at this boundary, not once per prefill_from_device call.
        begin_scope("eager")
        try:
            return original_public_prefill(*args, **kwargs)
        finally:
            scope = previous_scope

    gen.prefill_forward = public_prefill

    original_prefill = gen.model.prefill_from_device

    def prefill(*args, **kwargs):
        nonlocal active, scope, invocation, invocation_id
        previous_active, previous_scope, previous_invocation = active, scope, invocation
        if scope is None:
            begin_scope("eager")  # Standalone preparation warmup.
        invocation_id += 1
        plan = kwargs["plan"]
        invocation = {
            **scope,
            "invocation_id": invocation_id,
            "capture_generation": capture_generation if scope["execution"] == "capture" else None,
            "trace_id": None,
            "logical_length": plan.seq_len,
            "start_pos": plan.start_pos,
        }
        active = True
        gen._diagnostic_prefill_active = True
        try:
            return original_prefill(*args, **kwargs)
        finally:
            active, scope, invocation = previous_active, previous_scope, previous_invocation
            gen._diagnostic_prefill_active = active

    gen.model.prefill_from_device = prefill

    original_capture_prefill = gen._capture_prefill

    def capture_prefill(*args, **kwargs):
        nonlocal scope, capture_generation, captured_trace
        previous_scope = scope
        capture_generation += 1
        begin_scope("capture")
        try:
            result = original_capture_prefill(*args, **kwargs)
            captured_trace = gen.prefill_state["trace"]
            for item in metadata.values():
                item["trace_id"] = str(captured_trace)
            for item in address_metadata.values():
                item["trace_id"] = str(captured_trace)
            captured.clear()
            captured.update(retained)
            captured_metadata.clear()
            captured_metadata.update({name: dict(item) for name, item in metadata.items()})
            captured_address_metadata.clear()
            captured_address_metadata.update({name: dict(item) for name, item in address_metadata.items()})
            return result
        finally:
            scope = previous_scope

    gen._capture_prefill = capture_prefill

    original_replay_prefill = gen._replay_prefill

    def replay_prefill(*args, **kwargs):
        trace = gen.prefill_state["trace"]
        if captured_trace is None or trace != captured_trace:
            raise RuntimeError("No retained owner map for the current prefill trace")
        # A same-signature eager call may have replaced the visible owner map.
        # Replay executes no Python layer hooks, so restore the frozen map here.
        retained.clear()
        retained.update(captured)
        metadata.clear()
        metadata.update({name: dict(item) for name, item in captured_metadata.items()})
        address_metadata.clear()
        address_metadata.update({name: dict(item) for name, item in captured_address_metadata.items()})
        return original_replay_prefill(*args, **kwargs)

    gen._replay_prefill = replay_prefill

    original_release_traces = gen._release_traces

    def release_traces(*args, **kwargs):
        nonlocal captured_trace
        result = original_release_traces(*args, **kwargs)
        captured_trace = None
        captured.clear()
        captured_metadata.clear()
        captured_address_metadata.clear()
        return result

    gen._release_traces = release_traces

    def wrap_linear(index, layer, linear):
        def inspect(x, weight):
            record = active and weight is layer.wqkv and x.shape[2] >= 4095
            if record:
                keep(f"qkv_norm_full{index:02d}_s{x.shape[2]}", x)
            out = linear(x, weight)
            if record:
                keep(f"qkv_full{index:02d}_s{x.shape[2]}", out)
            return out

        return inspect

    def wrap_fused(index, layer, fused):
        def inspect(x, weight, **kwargs):
            retain_qkv = active and weight is layer.wqkv and x.shape[2] >= 4095
            record_agmm = active and x.shape[2] >= 256 and (weight is layer.wqkv or weight is layer.wswiglu)
            if not retain_qkv and not record_agmm:
                return fused(x, weight, **kwargs)
            role = "qkv" if weight is layer.wqkv else "mlp"
            original_cast = ttnn.typecast

            def cast(tensor, *args, **cast_kwargs):
                out = original_cast(tensor, *args, **cast_kwargs)
                if tensor is x:
                    if retain_qkv:
                        keep(f"qkv_cast_full{index:02d}_s{x.shape[2]}", out)
                    if record_agmm:
                        record_address(
                            f"agmm_cast_full{index:02d}_{role}_s{x.shape[2]}",
                            out,
                            layer=index,
                            role=role,
                            operation="agmm",
                            buffer_role="cast_input",
                        )
                return out

            ttnn.typecast = cast
            try:
                out = fused(x, weight, **kwargs)
                if record_agmm:
                    record_address(
                        f"agmm_gather_full{index:02d}_{role}_s{x.shape[2]}",
                        layer._prefill_gather_buffer,
                        layer=index,
                        role=role,
                        operation="agmm",
                        buffer_role="gather",
                    )
                return out
            finally:
                ttnn.typecast = original_cast

        return inspect

    for index, layer in enumerate(gen.model.layers):
        layer._linear = wrap_linear(index, layer, layer._linear)
        layer._fused_prefill_projection = wrap_fused(index, layer, layer._fused_prefill_projection)
    if norm_inputs:

        def wrap_norms(index, layer):
            original_layer = layer.prefill_forward
            original_norm = layer._norm
            norm_index = 0

            def forward(*args, **kwargs):
                nonlocal norm_index
                norm_index = 0
                return original_layer(*args, **kwargs)

            def norm(x):
                nonlocal norm_index
                record = active and x.shape[2] >= 4095
                role = "qkv" if norm_index == 0 else "mlp"
                if record:
                    keep(f"{role}_prenorm_full{index:02d}_s{x.shape[2]}", x)
                    norm_index += 1
                out = original_norm(x)
                if record:
                    keep(f"{role}_norm_full{index:02d}_s{x.shape[2]}", out)
                return out

            layer.prefill_forward = forward
            layer._norm = norm

        for index, layer in enumerate(gen.model.layers):
            wrap_norms(index, layer)
    if finish_inputs:

        def wrap_finish(index, layer, original):
            def finish(x, attention):
                if not active or x.shape[2] < 256:
                    return original(x, attention)
                length = x.shape[2]

                def tap(name, tensor):
                    keep(f"{name}_full{index:02d}_s{length}", tensor)
                    return tensor

                tap("attention", attention)
                projected = tap("o_partial", layer._linear(attention, layer.wo))
                projected = tap("o_reduced", layer._reduce(projected))
                residual_input = tap("attention_residual_input", ttnn.to_memory_config(x, projected.memory_config()))
                residual = tap("attention_residual", ttnn.add(residual_input, projected))
                mlp = tap("mlp", layer._mlp(layer._norm(residual), length, False))
                projected = tap("down_partial", layer._linear(mlp, layer.wdown))
                projected = tap("down_reduced", layer._reduce(projected))
                residual_input = tap("mlp_residual_input", ttnn.to_memory_config(residual, projected.memory_config()))
                return tap("mlp_residual", ttnn.add(residual_input, projected))

            return finish

        for index, layer in enumerate(gen.model.layers):
            layer._finish = wrap_finish(index, layer, layer._finish)
    return retained


def install_retained_rs_stages(gen, *, retain="both"):
    """Use one raw RS binding while varying only staging-buffer ownership."""
    if retain not in ("both", "main", "penult", "drop"):
        raise ValueError(f"Unknown RS staging-retention mode: {retain}")
    retained_stages = {
        "both": ("main", "penult"),
        "main": ("main",),
        "penult": ("penult",),
        "drop": (),
    }[retain]
    gen.diagnostic_rs_stages = {
        "mode": retain,
        "retained_stages": list(retained_stages),
        "binding": "_debug_reduce_scatter_buffers",
        "rs_configuration_overrides": False,
    }
    debug_rs = ttnn._ttnn.operations.experimental.ccl_experimental._debug_reduce_scatter_buffers

    def wrap(index, layer):
        original_forward, original_reduce = layer.prefill_forward, layer._reduce
        reduce_index = 0

        def forward(*args, **kwargs):
            nonlocal reduce_index
            reduce_index = 0
            return original_forward(*args, **kwargs)

        def reduce(x):
            nonlocal reduce_index
            if not getattr(gen, "_diagnostic_prefill_active", False) or x.shape[2] < 256:
                return original_reduce(x)
            role = "o" if reduce_index == 0 else "down"
            reduce_index += 1
            original_rs = ttnn.experimental.reduce_scatter_minimal_async

            def inspect(tensor, **kwargs):
                assert kwargs["persistent_output_buffers"] is None
                assert kwargs["dim"] == 3 and kwargs["topology"] == ttnn.Topology.Ring
                buffers = debug_rs(
                    tensor,
                    kwargs["multi_device_global_semaphore"],
                    **{
                        key: kwargs[key]
                        for key in (
                            "barrier_semaphore",
                            "num_links",
                            "chunks_per_sync",
                            "num_workers_per_link",
                            "num_buffers_per_channel",
                        )
                    },
                )
                for buffer_index, name in enumerate(("main", "output", "penult")):
                    gen._diagnostic_record_tensor(
                        f"rs_buffer_{name}_full{index:02d}_{role}_s{x.shape[2]}",
                        buffers[buffer_index],
                        layer=index,
                        role=role,
                        operation="reduce_scatter",
                        buffer_role=name,
                    )
                output = buffers[1]
                if "main" in retained_stages:
                    gen._diagnostic_keep_tensor(f"rs_stage_main_full{index:02d}_{role}_s{x.shape[2]}", buffers[0])
                if "penult" in retained_stages:
                    gen._diagnostic_keep_tensor(f"rs_stage_penult_full{index:02d}_{role}_s{x.shape[2]}", buffers[2])
                # Drop every unselected staging owner before any further
                # diagnostic work. In "drop" mode this mirrors the public
                # wrapper's output-only return while retaining the raw binding.
                del buffers
                gen._diagnostic_keep_tensor(f"rs_stage_output_full{index:02d}_{role}_s{x.shape[2]}", output)
                return output

            ttnn.experimental.reduce_scatter_minimal_async = inspect
            try:
                return original_reduce(x)
            finally:
                ttnn.experimental.reduce_scatter_minimal_async = original_rs

        layer.prefill_forward, layer._reduce = forward, reduce

    for index, layer in enumerate(gen.model.layers):
        wrap(index, layer)


def install_terminal_snapshots(
    gen, *, layer_snapshots=False, inner_snapshots=False, qkv_snapshots=False, qkv_input_snapshots=False
):
    """Diagnostic device copies only; production timing never uses this hook."""
    model = gen.model
    original_prefill = model.prefill_from_device
    original_final = model.final_logits
    active = False
    snapshots = {}

    def save(name, tensor):
        source = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
        if name not in snapshots:
            snapshots[name] = ttnn.empty_like(source)
        ttnn.copy(source, snapshots[name])

    def prefill(*args, **kwargs):
        nonlocal active
        active = True
        try:
            return original_prefill(*args, **kwargs)
        finally:
            active = False

    def terminal(x, *, decode):
        if not active or not decode:
            return original_final(x, decode=decode)
        first = model.layers[0]
        save("residual", x)
        x = ttnn.to_memory_config(x, first.decode_residual_memory)
        save("norm_input", x)
        x = ttnn.rms_norm(
            x,
            epsilon=first.eps,
            weight=model.norm_weight,
            program_config=first.decode_norm_program,
            compute_kernel_config=first.compute,
        )
        save("normalized", x)
        x = first._gather(x)
        save("gathered", x)
        x = ttnn.to_memory_config(x, model.head_input_memory)
        save("head_input", x)
        return model.head_decode(x)

    model.prefill_from_device = prefill
    model.final_logits = terminal
    if layer_snapshots:

        def wrap_layer(index, forward):
            def inspect(x, **kwargs):
                length = x.shape[2]
                if length >= 4095:
                    start = ((length - 1) // 32) * 32
                    if index == 0:
                        save(f"embedding_s{length}", x[:, :, start:length, :])
                out = forward(x, **kwargs)
                if length >= 4095:
                    save(f"layer{index:02d}_s{length}", out[:, :, start:length, :])
                return out

            return inspect

        for index, layer in enumerate(model.layers):
            layer.prefill_forward = wrap_layer(index, layer.prefill_forward)
    if inner_snapshots:

        def wrap_finish(index, layer, finish):
            def inspect(x, attention):
                if not active or x.shape[2] < 4095:
                    return finish(x, attention)
                length = x.shape[2]
                start = ((length - 1) // 32) * 32

                def tap(name, tensor):
                    save(f"inner{index:02d}_{name}_s{length}", tensor[:, :, start:length, :])

                tap("attention", attention)
                projected = layer._linear(attention, layer.wo)
                tap("o_partial", projected)
                projected = layer._reduce(projected)
                tap("o_reduced", projected)
                residual = ttnn.add(ttnn.to_memory_config(x, projected.memory_config()), projected)
                tap("attention_residual", residual)
                normed = layer._norm(residual)
                tap("mlp_norm", normed)
                mlp = layer._mlp(normed, x.shape[2], False)
                tap("mlp", mlp)
                projected = layer._linear(mlp, layer.wdown)
                tap("down_partial", projected)
                projected = layer._reduce(projected)
                tap("down_reduced", projected)
                return ttnn.add(ttnn.to_memory_config(residual, projected.memory_config()), projected)

            return inspect

        for index, layer in enumerate(model.layers):
            layer._finish = wrap_finish(index, layer, layer._finish)
    if qkv_snapshots:

        def wrap_linear(index, layer, linear):
            def inspect(x, weight):
                if qkv_input_snapshots and active and weight is layer.wqkv and x.shape[2] >= 4095:
                    save(f"qkv_norm_full{index:02d}_s{x.shape[2]}", x)
                out = linear(x, weight)
                if active and weight is layer.wqkv and x.shape[2] >= 4095:
                    save(f"qkv_full{index:02d}_s{x.shape[2]}", out)
                return out

            return inspect

        for index, layer in enumerate(model.layers):
            layer._linear = wrap_linear(index, layer, layer._linear)
    if qkv_input_snapshots:

        def wrap_fused(index, layer, fused):
            def inspect(x, weight, **kwargs):
                if not active or weight is not layer.wqkv or x.shape[2] < 4095:
                    return fused(x, weight, **kwargs)
                original_cast = ttnn.typecast

                def cast(tensor, *args, **cast_kwargs):
                    out = original_cast(tensor, *args, **cast_kwargs)
                    if tensor is x:
                        save(f"qkv_cast_full{index:02d}_s{x.shape[2]}", out)
                    return out

                ttnn.typecast = cast
                try:
                    return fused(x, weight, **kwargs)
                finally:
                    ttnn.typecast = original_cast

            return inspect

        for index, layer in enumerate(model.layers):
            layer._fused_prefill_projection = wrap_fused(index, layer, layer._fused_prefill_projection)
    return snapshots


def run(gen, output, *, boundaries=True, lengths=None, long_repeats=10):
    result = {"layers": gen.model.num_layers, "records": []}
    if hasattr(gen, "diagnostic_rs_stages"):
        result["diagnostic_rs_stages"] = dict(gen.diagnostic_rs_stages)
    ids = gen.tokenizer.encode("The sky appears blue because sunlight scatters in the atmosphere. " * 600)
    gen._ensure_owned_cache(1, 512)
    result["initial_cache_capacity"] = gen.capacity
    result["initial_page_table_shape"] = list(gen.page_table.shape)

    def failure_state():
        # Only inspect after the unmodified request has already failed.
        original_sampler_input = getattr(gen, "diagnostic_prefill_logits", None)
        ttnn.synchronize_device(gen.mesh)
        evidence = {"perf": gen.last_perf.copy(), "tensors": {}}
        for name, tensor in {
            "tokens": gen.state["tokens"],
            "history_index": gen.history_index,
            "token_history": gen.token_history,
        }.items():
            host = tensor.cpu(blocking=True)
            shards = [ttnn.to_torch(shard).long() for shard in ttnn.get_device_tensors(host)]
            evidence["tensors"][name] = {
                "address": tensor.buffer_address(),
                "shape": list(tensor.shape),
                "rank_values": [
                    shard.reshape(-1, 32)[:16, 0].tolist() if name == "token_history" else shard.flatten()[:32].tolist()
                    for shard in shards
                ],
            }
        evidence["trace_ids"] = {key: str(value) for key, value in gen.state.items() if "trace" in key}
        if getattr(gen, "terminal_snapshots", None):
            captured = {}
            evidence["terminal_snapshots"] = {}
            for name, tensor in gen.terminal_snapshots.items():
                if name.startswith("rs_stage_"):
                    # Staging has intentionally unwritten slices. Inspect only
                    # the pages consumed by the first corrupt output tile below.
                    continue
                host = tensor.cpu(blocking=True)
                shards = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(host)]
                full_qkv = "_full" in name
                captured[name] = [shard[..., -32:, :].clone() for shard in shards] if full_qkv else shards
                evidence["terminal_snapshots"][name] = [
                    {
                        "finite": bool(torch.isfinite(shard).all()),
                        "min": float(shard.min()),
                        "max": float(shard.max()),
                        "abs_top10": torch.topk(shard.abs().flatten(), 10).values.tolist(),
                    }
                    for shard in shards
                ]
                if full_qkv:
                    for rank, shard in enumerate(shards):
                        bad = (shard.abs() > 1e6) | ~torch.isfinite(shard)
                        rows = torch.nonzero(bad.any(dim=(0, 1, 3))).flatten()
                        evidence["terminal_snapshots"][name][rank]["bad_rows"] = rows.tolist()
                        if rows.numel():
                            captured[f"{name}_bad_rank{rank}"] = shard.index_select(2, rows[:32])
            for name, summaries in evidence["terminal_snapshots"].items():
                if (
                    name.startswith("inner")
                    and "_attention_s" in name
                    and any(max(abs(item["min"]), abs(item["max"])) > 1e6 for item in summaries)
                ):
                    layer_index = int(name[5:7])
                    evidence["bad_attention_cache_layer"] = layer_index
                    evidence["bad_attention_cache"] = {}
                    for role, cache in zip(("key", "value"), gen.kv_cache[layer_index]):
                        cache_host = cache.cpu(blocking=True)
                        cache_shards = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(cache_host)]
                        captured[f"cache_{role}_layer{layer_index}"] = cache_shards
                        evidence["bad_attention_cache"][role] = [
                            {
                                "finite": bool(torch.isfinite(shard).all()),
                                "min": float(shard.min()),
                                "max": float(shard.max()),
                            }
                            for shard in cache_shards
                        ]
                    break
            for name, summaries in evidence["terminal_snapshots"].items():
                match = re.fullmatch(r"(o|down)_reduced_full(\d+)_s(\d+)", name)
                if match is None:
                    continue
                bad_rank = next((rank for rank, item in enumerate(summaries) if item["bad_rows"]), None)
                if bad_rank is None:
                    continue
                role, layer, length = match.groups()
                owner = gen.terminal_snapshots[name]
                shards = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(owner.cpu(blocking=True))]
                bad = (shards[bad_rank].abs() > 1e6) | ~torch.isfinite(shards[bad_rank])
                row, col = torch.nonzero(bad.reshape(-1, 1024))[0].tolist()
                tile_row, tile_col = row // 32, col // 32
                tile_id = tile_row * 32 + tile_col
                output_tile = shards[bad_rank][
                    ..., tile_row * 32 : (tile_row + 1) * 32, tile_col * 32 : (tile_col + 1) * 32
                ]
                captured["rs_first_bad_output_tile"] = output_tile
                partial = gen.terminal_snapshots[f"{role}_partial_full{layer}_s{length}"]
                partials = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(partial.cpu(blocking=True))]
                captured["rs_first_bad_partial_tiles"] = [
                    t[
                        ...,
                        tile_row * 32 : (tile_row + 1) * 32,
                        bad_rank * 1024 + tile_col * 32 : bad_rank * 1024 + (tile_col + 1) * 32,
                    ].clone()
                    for t in partials
                ]
                evidence["rs_first_bad_tile"] = {
                    "layer": int(layer),
                    "role": role,
                    "length": int(length),
                    "rank": bad_rank,
                    "tile_id": tile_id,
                    "tile_row": tile_row,
                    "tile_col": tile_col,
                    "partial_maxabs": [float(t.abs().max()) for t in captured["rs_first_bad_partial_tiles"]],
                }
                chunks_per_slice = (int(length) + 31) // 32 * 32 // 8
                chunk, chunk_tile = divmod(tile_id, 8)
                for kind, page in (("main", bad_rank * chunks_per_slice + chunk), ("penult", chunk)):
                    stage_name = f"rs_stage_{kind}_full{layer}_{role}_s{length}"
                    if stage_name not in gen.terminal_snapshots:
                        continue
                    stage = gen.terminal_snapshots[stage_name]
                    raw = ttnn.to_torch(ttnn.get_device_tensors(stage.cpu(blocking=True))[bad_rank])
                    raw_tile_bytes = raw.reshape(-1, 16384)[page, chunk_tile * 2048 : (chunk_tile + 1) * 2048].clone()
                    raw_tile = raw_tile_bytes.view(torch.bfloat16).float()
                    decoded = raw_tile.reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)
                    captured[f"rs_first_bad_{kind}_raw_tile"] = raw_tile
                    captured[f"rs_first_bad_{kind}_raw_bytes"] = raw_tile_bytes
                    captured[f"rs_first_bad_{kind}_decoded_tile"] = decoded
                    evidence["rs_first_bad_tile"][kind] = {
                        "shape": list(stage.shape),
                        "page": page,
                        "tile_offset": chunk_tile,
                        "finite": bool(torch.isfinite(decoded).all()),
                        "maxabs": float(decoded.abs().max()),
                    }
                break
            if hasattr(gen, "retained_snapshot_metadata"):
                evidence["retained_snapshot_metadata"] = gen.retained_snapshot_metadata
            snapshot_path = str(Path(output).with_suffix(".terminal.pt"))
            torch.save(captured, snapshot_path)
            evidence["terminal_snapshot_file"] = snapshot_path
        if hasattr(gen, "diagnostic_address_ledger"):
            evidence["address_ledger"] = {name: dict(item) for name, item in gen.diagnostic_address_ledger.items()}
        if gen.prefill_state is not None and gen.prefill_state.get("trace") is not None:
            logits = gen.prefill_state["logits"]
            values = gen._read_logits(logits).float().reshape(-1, gen.model.vocab_size)[0]
            evidence["retained_prefill_logits"] = {
                "address": logits.buffer_address(),
                "shape": list(logits.shape),
                "finite": bool(torch.isfinite(values).all()),
                "top20_ids": torch.topk(values, 20).indices.tolist(),
                "top20_values": torch.topk(values, 20).values.tolist(),
            }
            gen._sample(gen.model.sampler_logits(logits), strategy="split")
            evidence["retained_prefill_resampled_token"] = int(gen._read_tokens(batch=1)[0])
        if original_sampler_input is not None:
            logits = original_sampler_input
            values = gen._read_logits(logits).float().reshape(-1, gen.model.vocab_size)[0]
            evidence["retained_sampler_input"] = {
                "address": logits.buffer_address(),
                "shape": list(logits.shape),
                "finite": bool(torch.isfinite(values).all()),
                "top20_ids": torch.topk(values, 20).indices.tolist(),
                "top20_values": torch.topk(values, 20).values.tolist(),
            }
            gen._sample(logits, strategy="split")
            evidence["retained_sampler_resampled_token"] = int(gen._read_tokens(batch=1)[0])
        return evidence

    def check(prompt, steps, label, **sampling):
        original_read_tokens = gen._read_tokens
        inspecting_failure = False
        expected_first = {"boundary4095": 14040, "boundary4353": 317}.get(label) if gen.model.num_layers == 36 else None
        if getattr(gen, "inspect_first_token", False):

            def inspect_first(**kwargs):
                nonlocal inspecting_failure
                tokens = original_read_tokens(**kwargs)
                if getattr(gen, "inspect_first_samples", False) and label == "boundary4353" and not inspecting_failure:
                    logits = gen.diagnostic_prefill_logits
                    values = gen._read_logits(logits).float().reshape(-1, gen.model.vocab_size)[0]
                    result.setdefault("first_sample_observations", []).append(
                        {
                            "case": label,
                            "phase": phase,
                            "sampled_token_before_logit_read": int(tokens[0]),
                            "cpu_argmax": int(values.argmax()),
                            "finite": bool(torch.isfinite(values).all()),
                            "top10_ids": torch.topk(values, 10).indices.tolist(),
                            "top10_values": torch.topk(values, 10).values.tolist(),
                        }
                    )
                    Path(output).write_text(json.dumps(result, indent=2) + "\n")
                if not inspecting_failure and expected_first is not None and int(tokens[0]) != expected_first:
                    inspecting_failure = True
                    evidence = failure_state()
                    evidence["last_completed_request_perf"] = evidence.pop("perf")
                    result["failure"] = {
                        "case": label,
                        "kind": "first_token_before_decode",
                        "current_prompt_length": len(prompt),
                        "phase": phase,
                        "expected_first_token": expected_first,
                        "actual_first_token": int(tokens[0]),
                        "device_state_after_failure": evidence,
                    }
                    Path(output).write_text(json.dumps(result, indent=2) + "\n")
                    raise AssertionError(label + " first token (captured before decode)")
                return tokens

            gen._read_tokens = inspect_first
        original_sample, original_capture = gen._sample, gen._capture
        capture_active = False
        sample_observations = []
        phase = "reference"
        if label == "boundary4095" and os.environ.get("K2_DIAGNOSTIC4095_POST_SAMPLE"):

            @contextmanager
            def mark_capture():
                nonlocal capture_active
                with original_capture() as trace:
                    capture_active = True
                    try:
                        yield trace
                    finally:
                        capture_active = False

            def inspect_sample(logits, **kwargs):
                result_sample = original_sample(logits, **kwargs)
                if not capture_active:
                    # Read only AFTER the original sampler has been enqueued;
                    # no diagnostic sync/copy is inserted before sampling.
                    values = gen._read_logits(logits).float().reshape(32, -1)[0]
                    token = int(gen._read_tokens(batch=1)[0])
                    sample_observations.append(
                        {
                            "phase": phase,
                            "strategy": kwargs["strategy"],
                            "sampled_token": token,
                            "cpu_argmax": int(values.argmax()),
                            "finite": bool(torch.isfinite(values).all()),
                            "top20_ids": torch.topk(values, 20).indices.tolist(),
                            "top20_values": torch.topk(values, 20).values.tolist(),
                        }
                    )
                    result["diagnostic_post_sample"] = sample_observations
                    Path(output).write_text(json.dumps(result, indent=2) + "\n")
                return result_sample

            gen._sample, gen._capture = inspect_sample, mark_capture
        original_prefill = gen.model.prefill_chunk
        snapshots = []
        if label == "boundary4095" and os.environ.get("K2_DIAGNOSTIC4095_LOGITS"):

            def inspect_prefill(*args, **kwargs):
                logits = original_prefill(*args, **kwargs)
                values = gen._read_logits(logits).float().flatten()
                snapshots.append(
                    {
                        "top20_ids": torch.topk(values, 20).indices.tolist(),
                        "top20_values": torch.topk(values, 20).values.tolist(),
                        "finite": bool(torch.isfinite(values).all()),
                    }
                )
                result["diagnostic_eager4095_snapshots"] = snapshots
                Path(output).write_text(json.dumps(result, indent=2) + "\n")
                return logits

            gen.model.prefill_chunk = inspect_prefill
        try:
            reference = gen.generate(prompt, steps, token_output="per_token", trace_prefill=False, **sampling)
        finally:
            gen.model.prefill_chunk = original_prefill
        baseline_perf = gen.last_perf.copy()
        expected_first = reference[0]
        if label == "boundary4095" and os.environ.get("K2_DIAGNOSTIC4095_LOGITS"):
            result["diagnostic_eager_repeat_before_traced"] = gen.generate(
                prompt, steps, token_output="per_token", trace_prefill=False, **sampling
            )
            Path(output).write_text(json.dumps(result, indent=2) + "\n")
        phase = "actual"
        actual = gen.generate(prompt, steps, **sampling)
        if actual != reference:
            result["failure"] = {
                "case": label,
                "reference_tokens": reference,
                "actual_tokens": actual,
                "baseline_perf": baseline_perf,
                "optimized_perf": gen.last_perf.copy(),
                "cache_capacity": gen.capacity,
                "page_table_shape": list(gen.page_table.shape),
                "repeated_modes": [],
                "device_state_after_failure": failure_state(),
            }
            Path(output).write_text(json.dumps(result, indent=2) + "\n")
            for traced, mode in [(False, "per_token"), (True, "buffered"), (False, "buffered"), (True, "per_token")]:
                phase = f"repeat_{traced}_{mode}"
                tokens = gen.generate(prompt, steps, trace_prefill=traced, token_output=mode, **sampling)
                result["failure"]["repeated_modes"].append(
                    {"traced_prefill": traced, "output_mode": mode, "tokens": tokens, "perf": gen.last_perf.copy()}
                )
                Path(output).write_text(json.dumps(result, indent=2) + "\n")
            if (
                gen.prefill_state is not None
                and gen.prefill_state["trace"] is not None
                and gen.prefill_state["key"][0] == len(prompt)
            ):
                gen.reset()
                traced_logits = gen._read_logits(gen._replay_prefill(prompt)).float().flatten()
                gen.reset()
                eager_logits = (
                    gen.prefill_forward(
                        torch.tensor([prompt]),
                        page_table=gen.page_table,
                        kv_cache=gen.kv_cache,
                        prompt_lens=[len(prompt)],
                        sampling_mode="host",
                    )
                    .float()
                    .flatten()
                )
                result["failure"]["repeated_logits"] = {
                    "exact": torch.equal(traced_logits, eager_logits),
                    "pcc": float(torch.corrcoef(torch.stack([traced_logits, eager_logits]))[0, 1]),
                    "max_abs": float((traced_logits - eager_logits).abs().max()),
                    "trace_top20": torch.topk(traced_logits, 20).indices.tolist(),
                    "eager_top20": torch.topk(eager_logits, 20).indices.tolist(),
                }
            else:
                result["failure"]["repeated_logits"] = {"skipped": "No matching prepared prefill trace"}
            Path(output).write_text(json.dumps(result, indent=2) + "\n")
        gen._sample, gen._capture = original_sample, original_capture
        assert actual == reference, label
        perf = gen.last_perf.copy()
        counts = perf["steady_state_counters"]
        assert counts.get("token_readbacks", 0) == counts.get("token_refreshes", 0) == 0
        assert counts.get("history_writes", 0) == steps - 1 and counts["history_readbacks"] == 1
        repeat_count = long_repeats if len(prompt) >= 4095 else 1
        for repeat_index in range(repeat_count):
            repeated = gen.generate(prompt, steps, **sampling)
            if repeated != actual:
                result["failure"] = {
                    "case": label,
                    "kind": "repeat",
                    "repeat_index": repeat_index,
                    "reference_tokens": actual,
                    "actual_tokens": repeated,
                    "device_state_after_failure": failure_state(),
                }
                Path(output).write_text(json.dumps(result, indent=2) + "\n")
            assert repeated == actual, label + " repeated"
        result["records"].append(
            {
                "case": label,
                "prompt_tokens": prompt,
                "steps": steps,
                "sampling": sampling,
                "tokens": actual,
                "matches_eager_per_token": True,
                "repeat_exact": True,
                "repeat_count": repeat_count,
                "baseline_perf": baseline_perf,
                "optimized_perf": perf,
                "repeat_perf": gen.last_perf.copy(),
                "history_capacity": gen.token_history.shape[2],
                **(
                    {
                        "address_ledger": {name: dict(item) for name, item in gen.diagnostic_address_ledger.items()},
                    }
                    if len(prompt) >= 4095 and hasattr(gen, "diagnostic_address_ledger")
                    else {}
                ),
            }
        )
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
        print("PASS", label, flush=True)
        gen._read_tokens = original_read_tokens

    check(ids[:128], 32, "greedy128")
    check(ids[1:129], 32, "changed_prompt_same_shape")
    check(ids[:33], 32, "top_k_top_p33", top_k=32, top_p=0.9, temperature=2.0, seed=991)
    old_capacity = gen.token_history.shape[2]
    old_history = gen.token_history.buffer_address()
    check(ids[:33], 129, "history_growth129")
    assert gen.token_history.shape[2] > old_capacity
    result["growth"] = {
        "old_capacity": old_capacity,
        "new_capacity": gen.token_history.shape[2],
        "old_address": old_history,
        "new_address": gen.token_history.buffer_address(),
    }
    check(ids[:33], 8, "short_request_after_growth")
    result["external_batch_return"] = []
    for length in [31, 32, 33, 257]:
        prompt = ids[:length]
        expected = gen.generate(prompt, 4)
        owned_ids = tuple(id(tensor) for pair in gen.kv_cache for tensor in pair)
        external_cache, external_table = gen.model.allocate_cache(batch_size=2, capacity=128)
        gen.decode_forward(
            torch.tensor([[ids[0]], [ids[1]]]),
            torch.tensor([0, 0]),
            page_table=external_table,
            kv_cache=external_cache,
        )
        assert gen.prefill_state["trace"] is None
        assert {key[1][-2] for key in gen.model.pool.tensors} == {2}
        actual = gen.generate(prompt, 4)
        assert actual == expected
        assert owned_ids == tuple(id(tensor) for pair in gen.kv_cache for tensor in pair)
        result["external_batch_return"].append({"length": length, "exact_tokens": True, "owned_cache_preserved": True})
        del external_cache
        print("EXTERNAL_BATCH_RETURN_PASS", length, flush=True)
    # Invalid embedding IDs must be rejected before any device state changes.
    before = gen.counters.copy()
    for token in [-1, gen.model.vocab_size]:
        try:
            gen.generate([token], 4)
        except ValueError as error:
            assert "vocabulary" in str(error)
        else:
            raise AssertionError("Invalid prompt token accepted")
    assert gen.counters == before
    result["invalid_prompt_ids_rejected_before_device_work"] = True
    if boundaries:
        for length in (
            lengths
            if lengths is not None
            else [1, 31, 32, 127, 129, 223, 224, 225, 255, 256, 257, 511, 512, 513, 4095, 4096, 4097, 4353]
        ):
            check(ids[:length], 4, f"boundary{length}")
    result["pass"] = True
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--output", required=True)
    p.add_argument("--short", action="store_true")
    p.add_argument("--lengths", nargs="+", type=int)
    p.add_argument("--cache-capacity", type=int)
    p.add_argument("--long-repeats", type=int, default=10)
    p.add_argument("--terminal-snapshots", action="store_true")
    p.add_argument("--layer-snapshots", action="store_true")
    p.add_argument("--inner-snapshots", action="store_true")
    p.add_argument("--qkv-snapshots", action="store_true")
    p.add_argument("--qkv-input-snapshots", action="store_true")
    p.add_argument("--retain-prefill-logits", action="store_true")
    p.add_argument("--retain-qkv-inputs", action="store_true")
    p.add_argument("--inspect-first-token", action="store_true")
    p.add_argument("--retain-norm-inputs", action="store_true")
    p.add_argument("--retain-finish-inputs", action="store_true")
    p.add_argument("--rs-workers", type=int)
    p.add_argument("--rs-chunks-per-sync", type=int)
    rs_stages = p.add_mutually_exclusive_group()
    rs_stages.add_argument(
        "--retain-rs-stages", action="store_true", help="Raw RS binding; retain both staging buffers"
    )
    rs_stages.add_argument("--retain-rs-main", action="store_true", help="Raw RS binding; retain only main staging")
    rs_stages.add_argument("--retain-rs-penult", action="store_true", help="Raw RS binding; retain only penult staging")
    rs_stages.add_argument(
        "--rs-debug-drop-stages", action="store_true", help="Raw RS binding; drop both staging buffers"
    )
    p.add_argument("--inspect-first-samples", action="store_true")
    args = p.parse_args()
    rs_stage_mode = (
        "both"
        if args.retain_rs_stages
        else "main"
        if args.retain_rs_main
        else "penult"
        if args.retain_rs_penult
        else "drop"
        if args.rs_debug_drop_stages
        else None
    )
    if rs_stage_mode is not None and (args.rs_workers is not None or args.rs_chunks_per_sync is not None):
        p.error("All native RS diagnostic modes require the unmodified RS configuration")
    if args.inspect_first_samples and not (args.inspect_first_token and args.retain_prefill_logits):
        p.error("First-sample observations require --inspect-first-token --retain-prefill-logits")
    if args.rs_workers is not None or args.rs_chunks_per_sync is not None:
        original_rs = ttnn.experimental.reduce_scatter_minimal_async

        def configured_rs(tensor, *op_args, **kwargs):
            if tensor.shape[2] >= 256:
                if args.rs_workers is not None:
                    kwargs["num_workers_per_link"] = args.rs_workers
                if args.rs_chunks_per_sync is not None:
                    kwargs["chunks_per_sync"] = args.rs_chunks_per_sync
            return original_rs(tensor, *op_args, **kwargs)

        ttnn.experimental.reduce_scatter_minimal_async = configured_rs
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    try:
        gen = K2Generator(mesh, override_num_layers=args.layers)
        gen.inspect_first_token = args.inspect_first_token
        gen.inspect_first_samples = args.inspect_first_samples
        if args.retain_prefill_logits:
            original_sample = gen._sample

            def retain_sample_input(logits, **kwargs):
                result = original_sample(logits, **kwargs)
                if args.inspect_first_token:
                    from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

                    TraceAllocationTracker.acknowledge_corruptible(logits)
                gen.diagnostic_prefill_logits = logits
                return result

            gen._sample = retain_sample_input
        if args.qkv_input_snapshots:
            args.qkv_snapshots = True
        if args.terminal_snapshots or args.layer_snapshots or args.inner_snapshots or args.qkv_snapshots:
            gen.terminal_snapshots = install_terminal_snapshots(
                gen,
                layer_snapshots=args.layer_snapshots or args.inner_snapshots or args.qkv_snapshots,
                inner_snapshots=args.inner_snapshots or args.qkv_snapshots,
                qkv_snapshots=args.qkv_snapshots,
                qkv_input_snapshots=args.qkv_input_snapshots,
            )
        if rs_stage_mode is not None:
            args.retain_norm_inputs = args.retain_finish_inputs = True
        if args.retain_qkv_inputs or args.retain_norm_inputs or args.retain_finish_inputs:
            gen.terminal_snapshots = install_retained_qkv_inputs(
                gen,
                inspect_first_token=args.inspect_first_token,
                norm_inputs=args.retain_norm_inputs,
                finish_inputs=args.retain_finish_inputs,
            )
        if rs_stage_mode is not None:
            install_retained_rs_stages(gen, retain=rs_stage_mode)
        if args.cache_capacity is not None:
            gen._ensure_owned_cache(1, args.cache_capacity)
        run(gen, args.output, boundaries=not args.short, lengths=args.lengths, long_repeats=args.long_repeats)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
