# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""All-output, real-weight QSA prefill and prefix-continuation probe.

The device receives ordinary hidden states and starts with empty cache. Every
logical input row is projected, every K/V row is filled on device, and every
output row is compared. No host-seeded prefix or CPU-selected IDs enter TT.
--seq-len 262144 --decode-steps 0 and --seq-len 262143 --decode-steps 1 exercise
the advertised context and its awkward tail without changing the HF config.
Adding --full-boundary-branch to the latter also validates all262144 prefill
outputs in one pass, checkpoints TT state before its final32 rows, and reuses
that prefix for the actual31-row tail plus traced final decode. The checkpoint
is a device-to-device test copy, never an HF cache or a forward host fallback.

The CPU reference uses real installed-HF projections/norms/RoPE and the exact
QSA selection rule, retaining compressed keys once per group. It applies HF
eager attention to the selected keys in increasing token order. This avoids
an N-by-N matrix and recomputing every old group's compression for each query.
Before its long run, the reference is checked against the unmodified HF QSA
module, including exact selected sets, on a 65-token case split at 17/32/33
and a separate CPU-only seeded-prefix control crossing actual pruning at2052.
This is an attention-layer context harness, not a full MoE decoder test.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
import time
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import fixtures, load_reference, pcc

FIXTURE_SEED = 1979


class StreamingHFQSA:
    """Bounded-memory independently controlled reference; CPU test code only."""

    def __init__(self, model, rope, capacity):
        self.model, self.rope = model, rope
        self.position = 0
        self.k = torch.zeros(1, 2, capacity, 256, dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)
        self.compressed = torch.zeros(1, capacity // 4, 128, dtype=torch.bfloat16)
        self.history = torch.zeros(1, 3, 128, dtype=torch.bfloat16)

    @torch.inference_mode()
    def forward_chunk(self, hidden):
        from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb, repeat_kv

        model = self.model
        count = hidden.shape[1]
        start, end = self.position, self.position + count
        positions = torch.arange(start, end).reshape(1, -1)
        cos, sin = self.rope(hidden, positions)
        group_cos, group_sin = self.rope(hidden, positions.div(4, rounding_mode="floor") * 4)
        q_gate = model.q_proj(hidden).reshape(1, count, 24, 512)
        query, gate = q_gate.chunk(2, dim=-1)
        query = model.q_norm(query).transpose(1, 2)
        query = apply_rotary_pos_emb(query, cos=cos, sin=sin)
        gate = gate.transpose(1, 2)
        key = model.k_norm(model.k_proj(hidden).reshape(1, count, 2, 256)).transpose(1, 2)
        key = apply_rotary_pos_emb(key, cos=cos, sin=sin)
        value = model.v_proj(hidden).reshape(1, count, 2, 256).transpose(1, 2)
        self.k[:, :, start:end] = key
        self.v[:, :, start:end] = value

        qk = model.indexer.index_qk_proj(hidden)
        index_query = model.indexer.q_layernorm(qk[..., :512].reshape(1, count, 4, 128)).transpose(1, 2)
        index_query = apply_rotary_pos_emb(index_query, cos=cos, sin=sin)
        history = torch.cat([self.history, qk[..., 512:]], dim=1)
        pooled = history.float().unfold(1, 4, 1).mean(-1).bfloat16()
        pooled = model.indexer.k_layernorm(pooled)
        pooled = apply_rotary_pos_emb(pooled.unsqueeze(1), cos=group_cos, sin=group_sin).squeeze(1)
        for row, position in enumerate(range(start, end)):
            if position % 4 == 3:
                self.compressed[:, position // 4] = pooled[:, row]
        self.history = history[:, -3:].clone()
        total_complete = end // 4
        if total_complete:
            scores = torch.matmul(
                index_query.float(), self.compressed[:, :total_complete].float().transpose(-1, -2).unsqueeze(1)
            )
            scores = scores.relu().sum(dim=1)[0] / math.sqrt(128)
        outputs, selections = [], []
        for row, position in enumerate(range(start, end)):
            complete = (position + 1) // 4
            if complete:
                selected_blocks = scores[row, :complete].topk(min(512, complete)).indices
                selected = (selected_blocks[:, None] * 4 + torch.arange(4)).flatten()
            else:
                selected = torch.empty(0, dtype=torch.int64)
            selected = torch.cat([selected, torch.arange(complete * 4, position + 1)]).sort().values
            keys = repeat_kv(self.k[:, :, selected], 12)
            values = repeat_kv(self.v[:, :, selected], 12)
            logits = torch.matmul(query[:, :, row : row + 1], keys.transpose(-1, -2)) * (256**-0.5)
            probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
            attention = torch.matmul(probabilities, values)
            attention *= torch.sigmoid(gate[:, :, row : row + 1])
            outputs.append(model.o_proj(attention.reshape(1, 1, 6144)))
            selections.append(selected)
        self.position = end
        return torch.cat(outputs, dim=1), selections


@torch.inference_mode()
def control_reference(model, rope):
    """Compare bounded reference with actual installed HF forward, not itself."""
    generator = torch.Generator().manual_seed(1978)
    hidden = (torch.randn(1, 65, 2560, generator=generator) * 0.1).bfloat16()
    cos, sin = rope(hidden, torch.arange(65).reshape(1, -1))
    causal = torch.zeros(1, 1, 65, 65, dtype=torch.bfloat16)
    causal.masked_fill_(torch.ones(65, 65, dtype=torch.bool).triu(1), torch.finfo(torch.bfloat16).min)
    expected, _ = model(hidden, (cos, sin), causal)
    expected_mask = model.indexer(hidden, (cos, sin), causal, None) == 0
    reference = StreamingHFQSA(model, rope, 128)
    outputs, selections = [], []
    for start, end in ((0, 17), (17, 32), (32, 33), (33, 65)):
        output, selected = reference.forward_chunk(hidden[:, start:end])
        outputs.append(output)
        selections.extend(selected)
    actual = torch.cat(outputs, dim=1)
    row_pcc = [pcc(expected[:, row], actual[:, row]) for row in range(65)]
    for row, selected in enumerate(selections):
        actual_mask = torch.zeros(65, dtype=torch.bool)
        actual_mask[selected] = True
        assert torch.equal(actual_mask, expected_mask[0, 0, row])
    assert min(row_pcc) >= 0.995
    report = {
        "rows": 65,
        "whole_pcc": pcc(expected, actual),
        "minimum_row_pcc": min(row_pcc),
        "selection_exact": True,
        "chunk_ends": [17, 32, 33, 65],
    }
    # A seeded prefix is used only to control this CPU reference against literal
    # HF cached forward at the pruning boundary. Device tests still start empty.
    fixture = fixtures(model, rope, prefix=2048, steps=5)
    reference = StreamingHFQSA(model, rope, 2176)
    reference.position = 2048
    reference.k[:, :, :2048] = fixture["keys"]
    reference.v[:, :, :2048] = fixture["values"]
    reference.compressed[:, :512] = fixture["compressed"]
    reference.history = fixture["raw_history"].squeeze(1).clone()
    outputs, selections = [], []
    for start, end in ((0, 3), (3, 4), (4, 5)):
        output, selected = reference.forward_chunk(fixture["hidden"][:, start:end])
        outputs.append(output)
        selections.extend(selected)
    actual = torch.cat(outputs, dim=1)
    expected = torch.cat(fixture["outputs"], dim=1)
    row_pcc = [pcc(expected[:, row], actual[:, row]) for row in range(5)]
    for row, selected in enumerate(selections):
        actual_mask = torch.zeros(2049 + row, dtype=torch.bool)
        actual_mask[selected] = True
        assert torch.equal(actual_mask, fixture["masks"][row])
    assert min(row_pcc) >= 0.995
    report["pruning_control"] = {
        "prefix": 2048,
        "rows": 5,
        "whole_pcc": pcc(expected, actual),
        "minimum_row_pcc": min(row_pcc),
        "selection_exact": True,
        "maximum_dropped_tokens": max(row["dropped_tokens"] for row in fixture["rows"]),
    }
    return report


@torch.inference_mode()
def make_fixture(model, rope, *, length, decode_steps, capacity):
    generator = torch.Generator().manual_seed(FIXTURE_SEED)
    hidden = (torch.randn(1, length + decode_steps, 2560, generator=generator) * 0.1).bfloat16()
    reference = StreamingHFQSA(model, rope, capacity)
    expected = torch.empty_like(hidden)
    digest = hashlib.sha256()
    start_time = time.perf_counter()
    for start in range(0, length, 32):
        count = min(32, length - start)
        result, selections = reference.forward_chunk(hidden[:, start : start + count])
        expected[:, start : start + count] = result
        for selected in selections:
            digest.update(selected.numpy().tobytes())
        completed = start + count
        if completed % 4096 == 0 or completed == length:
            print(
                json.dumps(
                    {
                        "cpu_reference_rows": completed,
                        "target_prefill_rows": length,
                        "elapsed_seconds": time.perf_counter() - start_time,
                    }
                ),
                flush=True,
            )
    for step in range(decode_steps):
        position = length + step
        result, selections = reference.forward_chunk(hidden[:, position : position + 1])
        expected[:, position : position + 1] = result
        digest.update(selections[0].numpy().tobytes())
    return (
        hidden,
        expected,
        {
            "reference_wall_seconds": time.perf_counter() - start_time,
            "all_rows_computed": length + decode_steps,
            "selected_ids_sha256": digest.hexdigest(),
            "final_position": reference.position,
        },
    )


def cached_fixture(args, model, rope, provenance, capacity):
    """Only read local torch tensor/primitive payloads, with an exact contract key.

    Snapshot commit path, source/config hashes, shape, capacity and seed prevent
    reuse across a changed reference. weights_only loading excludes arbitrary
    pickle globals. A temporary sibling plus os.replace prevents partial cache
    files being mistaken for completed fixtures after interruption.
    """
    key = {
        "version": 1,
        "snapshot": str(args.snapshot.resolve()),
        "config_sha256": provenance["config_sha256"],
        "reference_sha256": provenance["reference_sha256"],
        "fixture_source_sha256": hashlib.sha256(
            "\n".join(
                inspect.getsource(function)
                for function in (StreamingHFQSA, control_reference, make_fixture, fixtures, load_reference, pcc)
            ).encode()
        ).hexdigest(),
        "torch_version": str(torch.__version__),
        "transformers": provenance["transformers"],
        "length": args.seq_len,
        "decode_steps": args.decode_steps,
        "capacity": capacity,
        "seed": FIXTURE_SEED,
    }
    cache = args.fixture_cache
    if cache is not None and cache.exists():
        data = torch.load(cache, map_location="cpu", weights_only=True)
        source_key = data.get("key")
        # The sole permitted split change uses the identical full-context token
        # stream and causal all-output reference: final token262143 is consumed
        # by device decode instead of prefill. Literal HF controls above already
        # establish reference chunk independence. Keep every other identity
        # field exact, and retain both keys rather than relabeling the fixture.
        boundary_split = (
            key["length"] == 262143
            and key["decode_steps"] == 1
            and key["capacity"] == 262144
            and source_key == {**key, "length": 262144, "decode_steps": 0}
        )
        if source_key == key or boundary_split:
            expected_shape = (1, args.seq_len + args.decode_steps, 2560)
            for name in ("hidden", "expected"):
                value = data[name]
                if tuple(value.shape) != expected_shape or value.dtype != torch.bfloat16:
                    raise ValueError(f"Fixture cache {name} shape/dtype does not match its key")
            print(
                json.dumps({"fixture_cache": str(cache), "status": "HIT_BOUNDARY_SPLIT" if boundary_split else "HIT"}),
                flush=True,
            )
            return (
                data["control"],
                data["hidden"],
                data["expected"],
                data["reference_report"],
                {
                    "path": str(cache),
                    "hit": True,
                    "key": key,
                    "source_key": source_key,
                    "requested_key": key,
                    "boundary_split_reuse": boundary_split,
                    "reference_semantics": (
                        "full-prefill causal outputs, final row used as decode oracle"
                        if boundary_split
                        else "exact requested fixture"
                    ),
                },
            )
        print(json.dumps({"fixture_cache": str(cache), "status": "KEY_MISMATCH_REBUILD"}), flush=True)
        del data
    control = control_reference(model, rope)
    hidden, expected, reference_report = make_fixture(
        model, rope, length=args.seq_len, decode_steps=args.decode_steps, capacity=capacity
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(cache.name + f".tmp-{os.getpid()}")
        try:
            torch.save(
                {
                    "key": key,
                    "control": control,
                    "hidden": hidden,
                    "expected": expected,
                    "reference_report": reference_report,
                },
                temporary,
            )
            os.replace(temporary, cache)
        finally:
            temporary.unlink(missing_ok=True)
        print(json.dumps({"fixture_cache": str(cache), "status": "SAVED"}), flush=True)
    return (
        control,
        hidden,
        expected,
        reference_report,
        {"path": str(cache) if cache else None, "hit": False, "key": key},
    )


def output_metrics(expected, actual):
    actual = actual.reshape(expected.shape)
    minimum = 1.0
    worst = 0
    failures = []
    lowest = []
    for row in range(expected.shape[1]):
        score = pcc(expected[:, row], actual[:, row])
        if not math.isfinite(score):
            return {"finite": False, "minimum_row_pcc": None, "worst_row": row, "passed": False}
        if score < minimum:
            minimum, worst = score, row
        if score < 0.995:
            failures.append(row)
        lowest.append((score, row))
    # FP32 corrcoef over hundreds of millions of values can round/clamp to1
    # while individual rows disagree. Accumulate the global statistic in FP64
    # with bounded scratch memory; row-level gates remain mandatory.
    sx = sy = sxx = syy = sxy = 0.0
    for start in range(0, expected.shape[1], 4096):
        left = expected[:, start : start + 4096].double().flatten()
        right = actual[:, start : start + 4096].double().flatten()
        sx += left.sum().item()
        sy += right.sum().item()
        sxx += left.square().sum().item()
        syy += right.square().sum().item()
        sxy += (left * right).sum().item()
    elements = expected.numel()
    denominator = math.sqrt(max(0.0, sxx - sx * sx / elements) * max(0.0, syy - sy * sy / elements))
    whole = (sxy - sx * sy / elements) / denominator if denominator else float("nan")
    finite = bool(torch.isfinite(actual).all())
    return {
        "whole_pcc": whole,
        "whole_pcc_accumulation": "FP64 streaming sums",
        "minimum_row_pcc": minimum,
        "worst_row": worst,
        "finite": finite,
        "rows_compared": expected.shape[1],
        "rows_below_threshold": len(failures),
        "first_failing_rows": failures[:32],
        "lowest_rows": [{"row": row, "pcc": score} for score, row in sorted(lowest)[:16]],
        "worst_reference_std": expected[:, worst].float().std().item(),
        "worst_actual_std": actual[:, worst].float().std().item(),
        "passed": finite and math.isfinite(whole) and whole >= 0.995 and minimum >= 0.995,
    }


def run_device(
    config,
    weights,
    hidden,
    expected,
    *,
    length,
    decode_steps,
    capacity,
    continuation_cut,
    report,
    device_output_cache=None,
    prefix_state_cache=None,
    full_boundary_branch=False,
):
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa import QSA
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa_chunked import ChunkedQSA

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        model = ChunkedQSA(QSA(WeightLoader(weights, mesh), config))
        permutation = torch.randperm(capacity // 32, generator=torch.Generator().manual_seed(1980)).reshape(1, -1).int()
        if permutation[0, 0] == 0:
            permutation = permutation.roll(1, dims=1)
        state = model.attention.make_state(capacity=capacity, page_table=permutation)
        state_names = ("k_cache", "v_cache", "raw_history", "compressed_keys", "position")

        def host(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout)

        def snapshot():
            return {
                name: host(ttnn.to_torch(getattr(state, name)), getattr(state, name).dtype, getattr(state, name).layout)
                for name in state_names
            }

        def restore(saved):
            for name in state_names:
                ttnn.copy_host_to_device_tensor(saved[name], getattr(state, name))

        if full_boundary_branch and (length % 32 != 31 or decode_steps != 1 or continuation_cut is not None):
            raise ValueError("Full boundary branch requires length32n-1, one decode token, and no continuation cut")
        prefill_length = length + 1 if full_boundary_branch else length
        inputs = ttnn.to_device(host(hidden[:, :prefill_length].unsqueeze(0)), mesh)
        decode_input = ttnn.to_device(host(hidden[:, length : length + 1].unsqueeze(0)), mesh) if decode_steps else None
        initial = snapshot() if continuation_cut is not None else None
        checkpoint = None
        original_chunk = model.forward_chunk
        chunk_offset = 0
        checkpoint_count = 0
        if full_boundary_branch:
            checkpoint_position = prefill_length - 32
            # Allocate before the measured pass. The full prefill only copies
            # existing device buffers; no host tensor access enters its audit.
            checkpoint = {name: ttnn.to_device(value, mesh) for name, value in snapshot().items()}

            def checkpointed_chunk(value, *, state):
                nonlocal chunk_offset, checkpoint_count
                if chunk_offset == checkpoint_position:
                    for name in state_names:
                        ttnn.copy(getattr(state, name), checkpoint[name])
                    checkpoint_count += 1
                chunk_offset += value.shape[-2]
                result = original_chunk(value, state=state)
                if chunk_offset % 4096 == 0 or chunk_offset == prefill_length:
                    # Host dispatch progress only: no device readback or extra
                    # synchronization enters this audited model pass.
                    print(json.dumps({"device_prefill_rows_enqueued": chunk_offset}), flush=True)
                return result

            model.forward_chunk = checkpointed_chunk
        start_time = time.perf_counter()
        try:
            with runtime_audit():
                output = model.prefill_forward(inputs, state=state)
        finally:
            model.forward_chunk = original_chunk
        ttnn.synchronize_device(mesh)
        report["prefill_host_wall_seconds"] = time.perf_counter() - start_time
        actual = ttnn.to_torch(output).reshape(expected[:, :prefill_length].shape)
        if device_output_cache is not None:
            device_output_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "actual": actual,
                    "page_table": permutation,
                    "length": prefill_length,
                    "reference_fixture": report["fixture_cache"],
                },
                device_output_cache,
            )
            report["device_output_cache"] = str(device_output_cache)
        report["prefill"] = output_metrics(expected[:, :prefill_length], actual)
        report["position_after_prefill"] = int(ttnn.to_torch(state.position).reshape(-1)[0])
        assert report["position_after_prefill"] == prefill_length
        ttnn.deallocate(output)
        if full_boundary_branch:
            assert checkpoint_count == 1 and chunk_offset == prefill_length
            report["full_boundary_branch"] = {
                "full_prefill_length": prefill_length,
                "full_prefill": report["prefill"],
                "full_prefill_host_wall_seconds": report["prefill_host_wall_seconds"],
                "position_after_full_prefill": report["position_after_prefill"],
                "checkpoint_position": checkpoint_position,
                "checkpoint_count": checkpoint_count,
                "checkpoint_origin": "Device-to-device copies during full prefill from empty TT state",
                "checkpoint_buffers": list(state_names),
                "page_table": "Same immutable permutation in both branches",
                "reused_output_rows": checkpoint_position,
                "tail_prefill_rows": 31,
                "decode_position": length,
            }
            assert report["prefill"]["passed"], "Full prefill all-row gate failed before boundary continuation"
            with runtime_audit():
                for name in state_names:
                    ttnn.copy(checkpoint[name], getattr(state, name))
            assert int(ttnn.to_torch(state.position).reshape(-1)[0]) == checkpoint_position
            tail_start = time.perf_counter()
            with runtime_audit():
                tail = model.prefill_forward(inputs[:, :, checkpoint_position:length], state=state)
            ttnn.synchronize_device(mesh)
            report["full_boundary_branch"]["tail_prefill_host_wall_seconds"] = time.perf_counter() - tail_start
            tail_actual = ttnn.to_torch(tail).reshape(1, 31, 2560)
            report["full_boundary_branch"]["tail_prefill"] = output_metrics(
                expected[:, checkpoint_position:length], tail_actual
            )
            ttnn.deallocate(tail)
            # Earlier outputs belong to the identical TT prefix checkpoint.
            # Only the newly computed31-row tail contributes to this branch.
            actual = actual[:, :length]
            actual[:, checkpoint_position:length] = tail_actual
            report["prefill"] = output_metrics(expected[:, :length], actual)
            report["position_after_prefill"] = int(ttnn.to_torch(state.position).reshape(-1)[0])
            assert report["position_after_prefill"] == length
            report[
                "prefill_measurement_scope"
            ] = "Full length+1 pass including device checkpoint; tail measured separately"
            for tensor in checkpoint.values():
                ttnn.deallocate(tensor)
        if prefix_state_cache is not None:
            from models.autoports.qwen_qwen3_8_flash_next.tests.qsa_state_fixture import export_prefix

            report["prefix_state_export"] = export_prefix(
                state,
                prefix_state_cache,
                prefix=length,
                fixture_cache=report["fixture_cache"],
                provenance=report["provenance"],
                prefix_metric=report["prefill"],
            )

        report["decode"] = []
        if decode_steps:
            prefix = snapshot()
            if continuation_cut is not None:
                # Compile every continuation physical shape before recording
                # the decode trace that must survive their later execution.
                restore(initial)
                for start, end in ((0, continuation_cut), (continuation_cut, length)):
                    warm_part = model.prefill_forward(inputs[:, :, start:end, :], state=state)
                    ttnn.synchronize_device(mesh)
                    ttnn.deallocate(warm_part)
                restore(prefix)
            warmed = model.forward_chunk(decode_input, state=state)
            model.advance(state, 1)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(warmed)
            restore(prefix)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            with runtime_audit():
                decoded = model.forward_chunk(decode_input, state=state)
                model.advance(state, 1)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            # Exact output only: sole trace overwrites it completely, and each
            # synchronous readback finishes before the next replay. No state or
            # input allocation is excluded from tracking.
            trace_allocation_tracker.acknowledge_corruptible(decoded)
            restore(prefix)
            first_decode_outputs = []
            for step in range(decode_steps):
                position = length + step
                ttnn.copy_host_to_device_tensor(host(hidden[:, position : position + 1].unsqueeze(0)), decode_input)
                with runtime_audit():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                actual = ttnn.to_torch(decoded).reshape(1, 1, 2560)
                first_decode_outputs.append(actual.clone())
                report["decode"].append(
                    {"position": position, **output_metrics(expected[:, position : position + 1], actual)}
                )
            report["decode_trace_id"] = int(trace)
            report["capture_count"] = 1
            restore(prefix)
            report["decode_repeat_equal"] = []
            for step in range(decode_steps):
                position = length + step
                ttnn.copy_host_to_device_tensor(host(hidden[:, position : position + 1].unsqueeze(0)), decode_input)
                with runtime_audit():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                actual = ttnn.to_torch(decoded).reshape(1, 1, 2560)
                report["decode_repeat_equal"].append(torch.equal(actual, first_decode_outputs[step]))

        if continuation_cut is not None:
            restore(initial)
            first = model.prefill_forward(inputs[:, :, :continuation_cut, :], state=state)
            first_actual = ttnn.to_torch(first).reshape(1, continuation_cut, 2560)
            ttnn.deallocate(first)
            resumed = model.prefill_forward(inputs[:, :, continuation_cut:, :], state=state)
            resumed_actual = ttnn.to_torch(resumed).reshape(1, length - continuation_cut, 2560)
            ttnn.deallocate(resumed)
            report["continuation"] = {
                "cut": continuation_cut,
                "prefix": output_metrics(expected[:, :continuation_cut], first_actual),
                "resumed_rows": output_metrics(expected[:, continuation_cut:length], resumed_actual),
                "final_position": int(ttnn.to_torch(state.position).reshape(-1)[0]),
            }
            # Deliberately retain the absolute position but discard every
            # prefix cache/history buffer. This must fail the SAME row gate.
            restore(initial)
            position_host = host(torch.tensor([continuation_cut], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(position_host, state.position)
            dropped = model.prefill_forward(inputs[:, :, continuation_cut:, :], state=state)
            dropped_actual = ttnn.to_torch(dropped).reshape(1, length - continuation_cut, 2560)
            ttnn.deallocate(dropped)
            dropped_metrics = output_metrics(expected[:, continuation_cut:length], dropped_actual)
            report["continuation"]["dropped_history_control"] = dropped_metrics
            report["continuation"]["negative_control_detected"] = (
                dropped_metrics["finite"] and not dropped_metrics["passed"]
            )
            if decode_steps:
                restore(prefix)
                ttnn.copy_host_to_device_tensor(host(hidden[:, length : length + 1].unsqueeze(0)), decode_input)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                actual = ttnn.to_torch(decoded).reshape(1, 1, 2560)
                report["interleaved_prefill_same_trace_equal"] = torch.equal(actual, first_decode_outputs[0])
        if trace is not None:
            ttnn.release_trace(mesh, trace)
            trace = None
        report["page_table"] = permutation.tolist()
        report["runtime_audit"] = {
            "full_prefill": True,
            "boundary_device_checkpoint_and_tail": full_boundary_branch,
            "decode_capture_and_replays": bool(decode_steps),
            "torch_function_mode": True,
            "forbid_from_torch_to_torch": True,
            "throw_exception_on_fallback": True,
        }
        report["passed"] = report["prefill"]["passed"] and all(row["passed"] for row in report["decode"])
        if decode_steps:
            report["passed"] &= all(report["decode_repeat_equal"])
        if full_boundary_branch:
            report["passed"] &= (
                report["full_boundary_branch"]["full_prefill"]["passed"]
                and report["full_boundary_branch"]["tail_prefill"]["passed"]
            )
        if continuation_cut is not None:
            report["passed"] &= (
                report["continuation"]["prefix"]["passed"]
                and report["continuation"]["resumed_rows"]["passed"]
                and report["continuation"]["final_position"] == length
                and report["continuation"]["negative_control_detected"]
                and (not decode_steps or report["interleaved_prefill_same_trace_equal"])
            )
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=33)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--continuation-cut", type=int)
    parser.add_argument(
        "--full-boundary-branch",
        action="store_true",
        help="Also gate length+1 full prefill, restoring its TT checkpoint for31-row tail plus traced decode",
    )
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--fixture-cache", type=Path, help="Persistent local CPU hidden/reference tensor cache")
    parser.add_argument("--device-output-cache", type=Path, help="Save complete prefill outputs after the device pass")
    parser.add_argument(
        "--prefix-state-cache", type=Path, help="Export passing TT prefix state before decode, at a test boundary"
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    config, reference, rope, weights, provenance = load_reference(args.snapshot)
    total = args.seq_len + args.decode_steps
    if args.seq_len < 1 or args.decode_steps < 0 or total > config.max_position_embeddings:
        parser.error("Logical tokens must fit the unchanged advertised context")
    capacity = args.capacity or max(128, math.ceil(total / 128) * 128)
    if capacity < total or capacity > config.max_position_embeddings or capacity % 128:
        parser.error("Physical capacity must cover tokens and be a valid multiple of128")
    if args.continuation_cut is not None and not 0 < args.continuation_cut < args.seq_len:
        parser.error("Continuation cut must be inside the logical prompt")
    if args.full_boundary_branch and (
        args.seq_len % 32 != 31 or args.decode_steps != 1 or args.continuation_cut is not None
    ):
        parser.error("Full boundary branch requires seq-len32n-1, decode-steps1, and no continuation-cut")
    control, hidden, expected, reference_report, cache_report = cached_fixture(
        args, reference, rope, provenance, capacity
    )
    report = {
        "scope": "QSA attention-layer all-output prefill with initially empty paged/indexer state",
        "reference": "bounded selected-key CPU reference controlled against installed HF forward",
        "provenance": provenance,
        "reference_control": control,
        "reference_run": reference_report,
        "fixture_cache": cache_report,
        "seq_len": args.seq_len,
        "decode_steps": args.decode_steps,
        "full_boundary_branch_requested": args.full_boundary_branch,
        "physical_capacity": capacity,
        "threshold": 0.995,
        "device_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not args.cpu_only:
            report["device_executed"] = True
            run_device(
                config,
                weights,
                hidden,
                expected,
                length=args.seq_len,
                decode_steps=args.decode_steps,
                capacity=capacity,
                continuation_cut=args.continuation_cut,
                report=report,
                device_output_cache=args.device_output_cache,
                prefix_state_cache=args.prefix_state_cache,
                full_boundary_branch=args.full_boundary_branch,
            )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {str(exc).split('backtrace:')[0]}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "HF_REFERENCE_READY" if args.cpu_only else ("PASS" if report["passed"] else "FAIL"),
                "seq_len": args.seq_len,
                "output": str(args.output),
            }
        )
    )
    if not args.cpu_only and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
