# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded TP4 adapter correctness probe; run only after the server has stopped.

Uses real device pages and a deterministic host allocator, not the vLLM scheduler.
Both paths keep identical precision and allocate only currently needed pages.
The control processes raw device outputs like the plugin's nonoverlap finalizer
and resets from fresh host inputs. The candidate
uses stale host inputs, overlaps one pending read, and resets only at initial
binding and a drained slot permutation. Neither path reports performance.

Example, after sourcing ../run-env.sh from the tt-metal root:
    python_env/bin/python -m models.autoports.qwen_qwen3_8_27b.tests.check_vllm_adapter \
        --output models/autoports/qwen_qwen3_8_27b/doc/vllm_integration/adapter_device.json
"""

import argparse
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tests.check_deferred_generation import assert_tokens
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric
from models.autoports.qwen_qwen3_8_27b.tt.generator_vllm import Qwen38ForCausalLM

BATCH = 4
CONTEXT = 262144
POOL_PAGES = 8192
LENGTHS = (31, 45, 27, 53)
ACTIVE_REQUESTS = (0, 1)
REMAP_STEP = 36
REORDERED = (1, 0, 3, 2)


class PagePool:
    """Reserve physical page zero for padding and allocate distinct scattered IDs."""

    def __init__(self, page_size):
        self.page_size = page_size
        self.pages = [[] for _ in range(BATCH)]
        self.cursor = 0

    def ensure(self, request, tokens):
        required = (tokens + self.page_size - 1) // self.page_size
        new = []
        while len(self.pages[request]) < required:
            assert self.cursor < POOL_PAGES - 1, "Physical page pool exhausted"
            page = 1 + self.cursor * 37 % (POOL_PAGES - 1)
            self.cursor += 1
            self.pages[request].append(page)
            new.append(page)
        return new

    def table(self, order):
        table = torch.zeros(BATCH, CONTEXT // self.page_size, dtype=torch.int32)
        for row, request in enumerate(order):
            table[row, : len(self.pages[request])] = torch.tensor(self.pages[request], dtype=torch.int32)
        return table


def sampling_params():
    return SimpleNamespace(temperature=[0.0] * BATCH, top_k=[1] * BATCH, top_p=[1.0] * BATCH, seed=[37] * BATCH)


def make_prompts(tokenizer):
    texts = (
        "Count the bright stars above the quiet mountain. ",
        "Describe the green leaves around a shallow stream. ",
        "A brass compass rests beside a map of islands. ",
        "The bakery opens early and serves warm sesame bread. ",
    )
    prompts = []
    for text, length in zip(texts, LENGTHS):
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert ids
        prompts.append((ids * ((length + len(ids) - 1) // len(ids)))[:length])
    return prompts


def inactive_state(gen, order):
    """Read every TP shard of nonzero recurrent/conv sentinel requests."""
    result = {}
    for layer_idx, state in zip(gen.model.layer_indices, gen.cache.layers):
        for name in ("conv", "recurrent"):
            tensor = getattr(state, name)
            if tensor is None:
                continue
            for shard, local in enumerate(ttnn.get_device_tensors(tensor)):
                host = ttnn.to_torch(local)
                assert host.shape[0] == BATCH, (name, host.shape)
                for row, request in enumerate(order):
                    if request not in ACTIVE_REQUESTS:
                        result[(layer_idx, name, shard, request)] = host[row].clone()
    assert result, "The reduced stack must include recurrent state"
    assert all(torch.count_nonzero(value).item() > 0 for value in result.values()), "Inactive sentinels are vacuous"
    return result


def assert_state_equal(actual, expected, label):
    assert actual.keys() == expected.keys(), label
    for key in expected:
        assert torch.equal(actual[key], expected[key]), f"{label}: inactive state changed at {key}"


def fresh_path(adapter, prompts, page_size):
    """Restore the same physical cache, trace inputs, seeds, and adapter state."""
    gen = adapter.generator
    ttnn.synchronize_device(gen.mesh)
    gen._release_traces()
    gen.reset()
    gen._refresh_table(torch.zeros(BATCH, CONTEXT // page_size, dtype=torch.int32))
    adapter._sampling_key = None
    adapter._decode_bound = False
    pool = PagePool(page_size)
    for request, prompt in enumerate(prompts):
        pool.ensure(request, len(prompt))
    padded = torch.zeros(BATCH, max(LENGTHS), dtype=torch.int32)
    for row, prompt in enumerate(prompts):
        padded[row, : len(prompt)] = torch.tensor(prompt, dtype=torch.int32)
    first, rope_deltas = adapter.prefill_forward(
        padded,
        page_table=pool.table(range(BATCH)),
        kv_cache=adapter.cache,
        prompt_lens=list(LENGTHS),
        sampling_params=sampling_params(),
    )
    assert torch.equal(torch.as_tensor(rope_deltas).reshape(-1), torch.zeros(BATCH, dtype=torch.int64))
    assert tuple(first.shape) == (BATCH, 1)
    return pool, first[:, 0].tolist(), inactive_state(gen, tuple(range(BATCH)))


def run_path(adapter, prompts, page_size, steps, *, steady, report, save):
    gen = adapter.generator
    pool, latest, sentinels = fresh_path(adapter, prompts, page_size)
    order = tuple(range(BATCH))
    positions = list(LENGTHS)
    outputs = [[] for _ in range(BATCH)]
    first = latest.copy()
    pending = deque()
    retained = []
    rows = report["steps"]
    baseline = gen.counters.copy()
    report.update(
        status="running",
        first_tokens=first,
        outputs=outputs,
        output_boundary="async_read_then_process_host" if steady else "raw_device_to_process_host",
    )

    def finish(item):
        for event in item["events"]:
            ttnn.event_synchronize(event)
        value = adapter.process_decode_output_host(item["host"], is_tokens=True).clone()
        assert tuple(value.shape) == (BATCH, 1)
        item["expected_snapshot"] = value
        if item.get("retain", True):
            retained.append(item)
        for row, request in enumerate(item["order"]):
            latest[request] = int(value[row, 0])
            if request in ACTIVE_REQUESTS:
                outputs[request].append(latest[request])

    def drain():
        count = len(pending)
        while pending:
            finish(pending.popleft())
        return count

    for step in range(steps):
        reset = not steady or step in (0, REMAP_STEP)
        drained = drain() if reset else 0
        remap = None
        if step == REMAP_STEP:
            assert_state_equal(inactive_state(gen, order), sentinels, "Before slot remap")
            remap = [order.index(request) for request in REORDERED]
            order = REORDERED
        growth = {}
        for request in ACTIVE_REQUESTS:
            new = pool.ensure(request, positions[request] + 1)
            if new:
                assert positions[request] % page_size == 0, (step, request, positions[request])
                growth[str(request)] = dict(
                    position=positions[request], logical_page=positions[request] // page_size, ids=new
                )
        table = pool.table(order)
        changed = not torch.equal(table, gen.page_host)
        before = gen.counters.copy()
        traces_before = gen.trace, gen.sample_trace
        identity = gen.tokens, gen.positions, gen.page_table
        if reset:
            host_tokens = [latest[request] for request in order]
            host_positions = [positions[request] if request in ACTIVE_REQUESTS else -1 for request in order]
        else:
            # Last prompt input and its old position stay stale across every
            # steady submit, including allocations while a prior read is pending.
            host_tokens = [prompts[request][-1] for request in order]
            host_positions = [LENGTHS[request] - 1 if request in ACTIVE_REQUESTS else -1 for request in order]
        had_pending = bool(pending)
        device_output = adapter.decode_forward(
            torch.tensor(host_tokens, dtype=torch.int32).reshape(BATCH, 1),
            start_pos=torch.tensor(host_positions, dtype=torch.int32),
            page_table=table,
            kv_cache=adapter.cache,
            sampling_params=sampling_params(),
            reset_batch=reset,
            slot_remap=remap,
            read_from_device=False,
        )
        if steady:
            host, events = adapter.read_decode_output(device_output, async_read=True)
            assert events, "Async output must carry a completion event"
            pending.append(dict(host=host, events=events, order=order, step=step))
        else:
            # Mirror submit_decode(read_from_device=False, async_read=False)
            # followed by finalize_decode. No caller-side read may hide a
            # processor that cannot accept the raw multi-device token tensor.
            finish(dict(host=device_output, events=[], order=order, step=step, retain=False))
        delta = dict(gen.counters - before)
        assert delta.get("page_table_refreshes", 0) == int(changed), (step, growth, delta)
        assert tuple(gen.page_table.shape) == (BATCH, CONTEXT // page_size)
        assert all(a is b for a, b in zip(identity, (gen.tokens, gen.positions, gen.page_table)))
        assert torch.equal(gen.page_host, table), (step, "Host snapshot lost page growth")
        for name in ("model_replays", "sampling_replays", "token_readbacks"):
            assert delta.get(name, 0) == 1, (step, name, delta)
        assert delta.get("full_logits_readbacks", 0) == 0, delta
        if not reset:
            for name in ("token_refreshes", "position_refreshes", "rope_refreshes", "seed_refreshes", "trace_captures"):
                assert delta.get(name, 0) == 0, (step, name, delta)
            assert (gen.trace, gen.sample_trace) == traces_before, (step, "Steady trace was replaced")
        rows.append(
            dict(
                step=step,
                order=list(order),
                positions=positions.copy(),
                growth=growth,
                reset=reset,
                remap=remap,
                pending_before_submit=had_pending,
                drained_before_reset=drained,
                page_table_changed=changed,
                counters=delta,
            )
        )
        for request in ACTIVE_REQUESTS:
            positions[request] += 1
        if steady and len(pending) > 1:
            # Step N has already enqueued decode and read before finalizing N-1.
            finish(pending.popleft())
        save()
        print("ADAPTER_STEP", "steady" if steady else "control", step, "growth", sorted(growth), flush=True)

    drain()
    ttnn.synchronize_device(gen.mesh)
    # All later replays have finished: old host readback storage must still own
    # each earlier token snapshot, even though the feedback device buffer changed.
    for item in retained:
        reread = adapter.process_decode_output_host(item["host"], is_tokens=True)
        assert torch.equal(reread, item["expected_snapshot"]), (item["step"], "Readback snapshot was overwritten")
    distinct_snapshots = {tuple(item["expected_snapshot"].reshape(-1).tolist()) for item in retained}
    if steady:
        assert len(distinct_snapshots) > 1, "Snapshot ownership needs distinguishable decode outputs"
    assert_state_equal(inactive_state(gen, order), sentinels, "After decode and slot remap")
    device_positions = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1).tolist()
    expected_positions = [positions[request] if request in ACTIVE_REQUESTS else -1 for request in order]
    assert device_positions == expected_positions, (device_positions, expected_positions)
    device_table = ttnn.to_torch(ttnn.get_device_tensors(gen.page_table)[0])
    assert torch.equal(device_table, pool.table(order)), "Final device page table differs from allocation ledger"
    assert all(len(outputs[r]) == steps for r in ACTIVE_REQUESTS)
    assert all(sum(str(r) in row["growth"] for row in rows) >= 2 for r in ACTIVE_REQUESTS)
    if steady:
        assert any(row["growth"] and row["pending_before_submit"] and not row["reset"] for row in rows)
        assert rows[REMAP_STEP]["drained_before_reset"] > 0, "Remap failed to exercise a pending-output drain"
        assert any(not row["reset"] and not row["page_table_changed"] for row in rows)
    report.update(
        status="passed",
        first_tokens=first,
        outputs=outputs,
        counters=dict(gen.counters - baseline),
        allocated_pages=pool.pages,
        inactive_state_preserved=True,
        retained_snapshots=len(retained),
        distinct_snapshots=len(distinct_snapshots),
        snapshot_ownership_verified=True if steady else None,
        final_device_positions=device_positions,
    )
    save()
    return first, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=68)
    args = parser.parse_args()
    if not 68 <= args.steps <= 128:
        parser.error("--steps must be 68..128 to cover growth before and after remap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="running",
        mesh=[1, 4],
        batch=BATCH,
        context=CONTEXT,
        pool_pages=POOL_PAGES,
        prompt_lengths=list(LENGTHS),
        active_requests=list(ACTIVE_REQUESTS),
        remap_step=REMAP_STEP,
        allocator="deterministic disjoint physical-page pool; not the vLLM scheduler",
        control=dict(status="pending", steps=[]),
        steady=dict(status="pending", steps=[]),
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    try:
        gen = build_generator(Path(__file__).resolve().parents[1], mesh, layer_indices=[0, 3])
        adapter = Qwen38ForCausalLM(gen, BATCH, CONTEXT)
        assert not adapter.host_compatibility, "This probe requires device token-out sampling"
        page_sizes = {layer.PAGE_SIZE for layer in gen.model.layers}
        assert page_sizes == {32}, page_sizes
        page_size = page_sizes.pop()
        adapter.allocate_kv_cache(
            (POOL_PAGES, 1, page_size, 256), getattr(ttnn, gen.model.precision["kv_cache_dtype"]), 2
        )
        assert adapter.cache is gen.cache
        assert {layer.kind for layer in gen.model.layers} == {"linear_attention", "full_attention"}
        report.update(
            layers=gen.model.layer_indices,
            precision=gen.model.precision,
            page_size=page_size,
            page_table_shape=list(gen.page_table.shape),
            sdpa_k=[layer.policy.get("sdpa_k") for layer in gen.model.layers if layer.kind == "full_attention"],
        )
        prompts = make_prompts(gen.tokenizer)
        report["prompt_tokens"] = prompts
        save()
        expected_first, expected = run_path(
            adapter, prompts, page_size, args.steps, steady=False, report=report["control"], save=save
        )
        actual_first, actual = run_path(
            adapter, prompts, page_size, args.steps, steady=True, report=report["steady"], save=save
        )
        assert actual_first == expected_first, (actual_first, expected_first)
        for request in ACTIVE_REQUESTS:
            assert_tokens(actual[request], expected[request], args.steps, f"Async steady request {request}")
        report.update(status="passed", exact_active_token_streams_equal=True)
        save()
        print("VLLM_ADAPTER_DEVICE_CHECK_PASSED", flush=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
