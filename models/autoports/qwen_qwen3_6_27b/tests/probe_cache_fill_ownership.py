# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP4 full-layer cache-fill lease, mutation, resume and prefill-trace checks.

Requires the reviewed integration. No model hooks: plain PagedState supplies the
ordered reference for the same layer/weights. Initial HF-derived single-chip
oracle gates remain 0.995, including contribution, users and complete KV state.
"""

import argparse
import copy
import hashlib
import json
import os
import traceback
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--length", type=int, default=33)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--policy-file", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    assert 2 <= a.batch <= 32 and a.length >= 33
    assert not a.output.exists(), "Use a fresh output path"
    import torch
    import ttnn
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import load_layer
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc
    from models.autoports.qwen_qwen3_6_27b.tests.run_multichip_decoder import runtime_device_only
    from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder
    from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import PagedState
    from models.autoports.qwen_qwen3_6_27b.tt.page_table_ownership import page_table_for_attention, page_table_certified

    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(8)
    torch.manual_seed(17)
    x = torch.randn(a.batch, a.length, 5120, dtype=torch.bfloat16) * 0.1
    d = torch.randn(a.batch, 1, 5120, dtype=torch.bfloat16) * 0.1
    pages = (a.length + 33 + 31) // 32
    table = torch.randperm(a.batch * pages).reshape(a.batch, pages)
    ref = torch.load(a.reference, weights_only=True)
    digest = lambda t: hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    expected_metadata = {
        "layer": a.layer,
        "batch": a.batch,
        "length": a.length,
        "input_sha256": digest(x),
        "decode_input_sha256": digest(d),
        "page_table_sha256": digest(table),
        "optimized_source_sha256": hashlib.sha256((root / "tt/optimized_decoder.py").read_bytes()).hexdigest(),
        "hf_revision": "8f33b98937c6031bb49e713dee3e8e185aafd26f",
    }
    assert (
        ref["metadata"] == expected_metadata
    ), "Generate matching synthetic run_multichip_decoder --baseline reference"
    policy = json.loads(a.policy_file.read_text()) if a.policy_file else {}
    policy["batched_prefill_cache"] = True
    cfg = Qwen3_5TextConfig(**json.loads((root / "doc/hf_config.json").read_text())["text_config"])
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    report = {
        "status": "running",
        "pass": False,
        "mesh": [1, 4],
        "checks": [],
        "reference_metadata": expected_metadata,
        "environment": {
            k: os.environ.get(k)
            for k in ("TT_METAL_WATCHER", "TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE")
        },
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                root / "tt/multichip_decoder.py",
                root / "tt/flat_prefill.py",
                root / "tt/page_table_ownership.py",
                root / "tt/batched_cache_fill_candidate.py",
            ]
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def record(name, **values):
        report["checks"].append({"case": name, **values})
        save()
        print(name, values, flush=True)

    save()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
    tids = []
    try:
        model = MultichipDecoder.from_state_dict(
            load_layer(a.layer), hf_config=cfg, layer_idx=a.layer, mesh_device=mesh, policy=policy
        )
        assert model.kind == "full_attention"
        report["selected_policy"] = dict(model.policy)

        def upload_table(t):
            return model._upload(t.to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        def make(host=table, ordered=False):
            # Keep physical allocation identical even when an aliased mapping
            # stops referring to the highest numbered physical page.
            state = model.create_paged_state(table)
            if host is not table:
                assert ordered, "Mapped controls must use ordered execution"
                ttnn.copy(upload_table(host), state.page_table)
            return PagedState(state.key, state.value, state.page_table) if ordered else state

        def prepare(host, start=0):
            positions = torch.arange(start, start + host.shape[1]).expand(a.batch, -1)
            return model.upload_residual(host), tuple(model._upload(t.unsqueeze(1)) for t in rotary(host, positions))

        def forward(state, inputs, start=0):
            with runtime_device_only():
                return model.prefill_forward(inputs[0], state=state, rope=inputs[1], start_pos=start)

        def read_output(t):
            return torch.cat([ttnn.to_torch(v) for v in ttnn.get_device_tensors(t)], -1).reshape(a.batch, -1, 5120)

        def read_cache(state):
            return [
                torch.cat([ttnn.to_torch(v) for v in ttnn.get_device_tensors(t)], 1) for t in (state.key, state.value)
            ]

        def snapshot(out, state):
            return [read_output(out), *read_cache(state)]

        def exact(name, actual, expected, **extra):
            counts = [int((x != y).sum()) for x, y in zip(actual, expected)]
            finite = all(bool(torch.isfinite(t).all()) for t in actual)
            record(name, mismatches=counts, finite=finite, **extra)
            assert len(counts) == 3 and not any(counts) and finite, name

        def counts():
            return model._prefill_cache_bulk_calls, model._prefill_cache_ordered_calls

        inputs = prepare(x)
        state = make()
        before = counts()
        out = forward(state, inputs)
        initial = snapshot(out, state)
        initial_dispatch = [v - u for u, v in zip(before, counts())]
        assert initial_dispatch[0] > 0
        td = model.upload_decode_residual(d)
        pos = upload_table(torch.full((a.batch,), a.length, dtype=torch.int32))
        dr = tuple(model._upload(t.unsqueeze(1)) for t in rotary(d, torch.full((a.batch, 1), a.length)))
        with runtime_device_only():
            dec = model.decode_forward(td, state=state, current_pos=pos, rope=dr)
        decoded = snapshot(dec, state)
        gates = {}
        for phase, actual, inp in [("prefill", initial, x), ("decode", decoded, d)]:
            gates[phase] = pcc(ref[phase], actual[0])
            gates[phase + "_contribution"] = pcc(ref[phase] - inp, actual[0] - inp)
            gates[phase + "_users"] = [pcc(ref[phase][i], actual[0][i]) for i in range(a.batch)]
            gates[phase + "_state"] = [pcc(e, v) for e, v in zip(ref[phase + "_state"], actual[1:])]
        flat = [v for value in gates.values() for v in (value if isinstance(value, list) else [value])]
        record("initial_oracle", gates=gates, dispatch=initial_dispatch)
        assert min(flat) >= 0.995 and all(bool(torch.isfinite(t).all()) for t in initial + decoded)
        ordered = make(ordered=True)
        exact("initial_bulk_vs_ordered", initial, snapshot(forward(ordered, inputs), ordered))
        physical = ttnn.to_torch(ttnn.get_device_tensors(page_table_for_attention(state))[0]).to(torch.int32)
        remap = torch.flip(physical, [0, 1]).contiguous()
        aliased = remap.clone()
        aliased[1] = aliased[0]
        for mode, host in [
            ("exposed_disjoint", remap),
            ("shallow_alias", aliased),
            ("deep_alias", aliased),
            ("replacement", aliased),
            ("fresh_disjoint", remap),
            ("fresh_alias", aliased),
        ]:
            target = make()
            if mode.startswith("fresh_"):
                model.replace_paged_table_from_host(target, host)
            elif mode == "replacement":
                target.page_table = upload_table(host)
            else:
                alias = (
                    copy.copy(target)
                    if mode == "shallow_alias"
                    else (
                        copy.deepcopy(target, {id(target.key): target.key, id(target.value): target.value})
                        if mode == "deep_alias"
                        else target
                    )
                )
                ttnn.copy(upload_table(host), alias.page_table)
            should_bulk = mode == "fresh_disjoint"
            assert page_table_certified(target) == should_bulk
            control = make(host, ordered=True)
            before = counts()
            actual = snapshot(forward(target, inputs), target)
            delta = [v - u for u, v in zip(before, counts())]
            expected = snapshot(forward(control, inputs), control)
            exact(mode, actual, expected, dispatch=delta)
            assert (delta[0] > 0) if should_bulk else (delta[0] == 0 and delta[1] > 0)
            del target, control
        # Cumulative public resume includes partial pages; compare both complete
        # cache and each suffix against the unchanged ordered execution.
        stream = torch.randn(a.batch, 65, 5120, dtype=torch.bfloat16) * 0.1
        for start in (1, 31, 32, 33):
            target, control = make(), make(ordered=True)
            prefix, suffix = prepare(stream[:, :start]), prepare(stream[:, start : start + 8], start)
            exact(
                f"resume_{start}_prefix",
                snapshot(forward(target, prefix), target),
                snapshot(forward(control, prefix), control),
            )
            exact(
                f"resume_{start}_suffix",
                snapshot(forward(target, suffix, start), target),
                snapshot(forward(control, suffix, start), control),
            )
            del target, control
        # All buffers and eager reference executions precede capture. Capture
        # retains ordered fill even while the state's lease is still private.
        target = make()
        raw = page_table_for_attention(target)
        zero = [ttnn.clone(t) for t in (target.key, target.value)]
        banks = [prepare(x), prepare(-x)]
        device_tables = [upload_table(t) for t in (physical, remap, aliased)]
        active = prepare(x)

        def reset():
            for src, dst in zip(zero, (target.key, target.value)):
                ttnn.copy(src, dst)

        def set_case(index):
            source = banks[index % 2]
            ttnn.copy(source[0], active[0])
            for src, dst in zip(source[1], active[1]):
                ttnn.copy(src, dst)
            ttnn.copy(device_tables[index], raw)

        expected = []
        for index, host in enumerate((physical, remap, aliased)):
            control = make(host, ordered=True)
            expected.append(snapshot(forward(control, banks[index % 2]), control))
            del control
        # Compile ordered and bulk calls plus every input/reset copy beforehand.
        for _ in range(2):
            for index in range(3):
                set_case(index)
                reset()
                out = forward(PagedState(target.key, target.value, raw), active)
                del out
            set_case(0)
            reset()
            out = forward(target, active)
            del out
        assert page_table_certified(target)
        set_case(0)
        reset()
        before = counts()
        with runtime_device_only():
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            captured = model.prefill_forward(active[0], state=target, rope=active[1])
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        tids.append(tid)
        delta = [v - u for u, v in zip(before, counts())]
        assert delta[0] == 0 and delta[1] > 0 and page_table_certified(target)
        assert target.page_table is raw and not page_table_certified(target)
        for repeat in range(2):
            for index in range(3):
                set_case(index)
                reset()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                exact(
                    f"prefill_trace_{repeat}_{index}",
                    snapshot(captured, target),
                    expected[index],
                    capture_dispatch=delta,
                )
        assert not torch.equal(expected[0][0], expected[1][0]), "Changed input negative control"
        report.update(status="pass")
        report["pass"] = True
        save()
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        save()
        raise
    finally:
        for tid in tids:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
