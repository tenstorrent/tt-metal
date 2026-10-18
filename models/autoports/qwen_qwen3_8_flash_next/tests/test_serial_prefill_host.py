# SPDX-License-Identifier: Apache-2.0
"""Host ownership/row-order checks; no TTNN import or hardware claims."""

import ast
import copy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]


def test_qsa_view_discards_inherited_gather_partition_before_own_prepare():
    source = ROOT / "tt/prefill_view.py"
    node = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_decoder"
    )
    prepared = []

    def clone(owner, ops, cls=None):
        result = copy.copy(owner)
        if hasattr(result, "prepare"):
            result.prepare = lambda lengths: prepared.append(hasattr(result.attention, "_pooled_gather_partition"))
        return result

    ops = SimpleNamespace(upload=lambda *args: object())
    tensor = SimpleNamespace(shape=(1, 1, 1, 1))
    owner = SimpleNamespace(
        ops=ops,
        mesh=object(),
        kind="indexed_attention",
        ple=None,
        attn_hyper=SimpleNamespace(),
        mlp_hyper=SimpleNamespace(),
        mlp=SimpleNamespace(capacity=tensor),
        attention=SimpleNamespace(
            paged_pooled=True,
            output_projection=SimpleNamespace(),
            pooled_keys=tensor,
            _pooled_gather_partition=dict(kind="cyclic", work=8192, workers=110),
        ),
        prepare=lambda lengths: None,
    )
    namespace = dict(
        torch=torch,
        ttnn=SimpleNamespace(float32="float32", uint32="uint32", ROW_MAJOR_LAYOUT="row_major"),
        _ops=lambda value: value,
        _clone=clone,
        _shared_zero=lambda *args: object(),
        _PrefillDecoder=object,
        _RowOutputProjection=object,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    result = namespace["_decoder"](owner, {}, [1, 256])
    assert prepared == [False]
    assert not hasattr(result.attention, "_pooled_gather_partition")
    assert result.mlp.prefill_group_size == 512
    assert not hasattr(owner.mlp, "prefill_group_size")
    assert owner.attention._pooled_gather_partition == dict(kind="cyclic", work=8192, workers=110)


def method(filename, class_name, name, **globals_):
    tree = ast.parse((ROOT / "tt" / filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch, **globals_}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / "tt" / filename), "exec"), namespace)
    return namespace[name]


def test_ragged_view_keeps_logical_rows_distinct_from_physical_slots():
    operations, events = [], []
    obj = SimpleNamespace(
        buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        slots="slots",
        positions="positions",
        pages="pages",
        inputs={n: f"tokens{n}" for n in (1, 2, 4, 8, 16, 32, 64, 128, 256)},
        counters=Counter(),
        last_physical_chunks=[{"stale": True}],
        _copy=lambda value, target: operations.append((target, value.clone())),
        _begin_logits=lambda: operations.append(("begin",)),
        _store_row=lambda row: operations.append(("store", row)),
        _finish_logits=lambda: operations.append(("finish",)),
    )
    tokens = torch.arange(3 * 257, dtype=torch.int32).reshape(3, 257)
    pages = torch.arange(3 * 16, dtype=torch.int32).reshape(3, 16)
    method("prefill_view.py", "SerialPrefillView", "forward")(
        obj,
        tokens,
        [257, 0, 10],
        torch.tensor([31, 0, 127]),
        [9, 4, 2],
        pages,
        [True, False, True],
        replay=lambda name: operations.append(("replay", name)),
        event=lambda name, **fields: events.append((name, fields)),
    )
    assert [op[1] for op in operations if op[0] == "store"] == [0, 2]
    assert [op for op in operations if op[0] == "finish"] == [("finish",)]
    assert [op[1] for op in operations if op[0] == "replay"] == [
        "prefill_view_256",
        "prefill_view_1",
        "prefill_view_8",
        "prefill_view_2",
    ]
    assert [chunk["start_positions"] for chunk in obj.last_physical_chunks] == [[31], [287], [127], [135]]
    assert [chunk["physical_slot"] for chunk in obj.last_physical_chunks] == [9, 9, 2, 2]
    assert [chunk["logical_row"] for chunk in obj.last_physical_chunks] == [0, 0, 2, 2]
    assert all(chunk["batch"] == 1 and chunk["active"] == [True] for chunk in obj.last_physical_chunks)
    assert torch.equal(next(op[1] for op in operations if op[0] == "tokens256"), tokens[:1, :256])
    assert torch.equal(next(op[1] for op in operations if op[0] == "tokens8"), tokens[2:3, :8])
    assert [int(op[1][0, 0]) for op in operations if op[0] == "slots"] == [9, 2]
    assert len(events) == 4


def test_ragged_prefill_preserves_current_owner_logits_after_intervening_decode():
    def slice_write(source, destination, begin, end, step):
        destination[tuple(slice(a, b, s) for a, b, s in zip(begin, end, step))].copy_(source)

    fake_tt = SimpleNamespace(
        ROW_MAJOR_LAYOUT="row",
        TILE_LAYOUT="tile",
        to_layout=lambda value, layout: value.clone(),
        copy=lambda source, destination: destination.copy_(source),
        experimental=SimpleNamespace(slice_write=slice_write),
    )
    obj = SimpleNamespace(
        output=torch.tensor([[[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]]]),
        rows=torch.full((1, 1, 3, 2), -100.0),
        logits=torch.zeros(1, 1, 1, 2),
        buckets=[1],
        slots=object(),
        positions=object(),
        pages=object(),
        inputs={1: object()},
        counters=Counter(),
        _copy=lambda *args: None,
    )
    for name in ("_begin_logits", "_store_row", "_finish_logits"):
        fn = method("prefill_view.py", "SerialPrefillView", name, ttnn=fake_tt)
        setattr(obj, name, lambda *args, fn=fn: fn(obj, *args))
    forward = method("prefill_view.py", "SerialPrefillView", "forward")

    def run(active, value):
        forward(
            obj,
            torch.zeros(3, 1, dtype=torch.int32),
            [int(flag) for flag in active],
            torch.zeros(3, dtype=torch.int32),
            [2, 0, 1],
            torch.zeros(3, 16, dtype=torch.int32),
            active,
            replay=lambda name: obj.logits.fill_(value),
            event=lambda *args, **kwargs: None,
        )

    run([True, False, False], 40)
    assert obj.output.flatten().tolist() == [40, 40, 20, 21, 30, 31]
    # A real decode changes the owner while the view's row-major matrix retains
    # old values. The next ragged prefill must preserve this newer owner state.
    obj.output.copy_(torch.tensor([[[[50.0, 51.0], [60.0, 61.0], [70.0, 71.0]]]]))
    run([False, False, True], 80)
    assert obj.output.flatten().tolist() == [50, 51, 60, 61, 80, 80]


def test_original_generator_samples_once_after_view_and_restores_full_metadata():
    calls = []
    model = SimpleNamespace(context=262144, max_requests=3, scratch_page_base=8192)
    pages = torch.arange(3 * 16, dtype=torch.int32).reshape(3, 16)
    view = SimpleNamespace(forward=lambda *args, **kwargs: calls.append(("view", args[1].copy(), args[3].copy())))
    obj = SimpleNamespace(
        batch_size=3,
        model=model,
        prefill_view=view,
        sampling_mode="device",
        _bind_cache=lambda cache: None,
        prepare=lambda: None,
        _request_slots=lambda slots: list(slots),
        _request_pages=lambda supplied: supplied.clone(),
        _active_rows=lambda active: calls.append(("active", active.copy())),
        _slots=lambda slots: calls.append(("slots", slots.copy())),
        _pages=lambda supplied: calls.append(("pages", supplied.clone())),
        _positions=lambda positions: calls.append(("positions", positions.clone())),
        _replay=lambda mode: calls.append(("sample", mode)),
        _read_tokens=lambda: [71, -1, 93],
        event=lambda *args, **kwargs: None,
    )
    result = method("generator.py", "QwenFlashNextGenerator", "prefill_forward")(
        obj,
        torch.zeros(3, 129, dtype=torch.int32),
        page_table=pages,
        kv_cache=object(),
        prompt_lens=[129, 0, 10],
        start_pos=torch.tensor([31, 0, 127]),
        slots=[2, 0, 1],
        sampling="sample",
    )
    assert result.tolist() == [71, -1, 93]
    assert calls[0] == ("view", [129, 0, 10], [2, 0, 1])
    assert calls[-1] == ("sample", "sample")
    assert len([call for call in calls if call[0] == "sample"]) == 1
    assert next(call[1] for call in calls if call[0] == "slots") == [2, 4, 1]
    assert next(call[1] for call in calls if call[0] == "positions").tolist() == [160, 0, 137]
    final_pages = next(call[1] for call in calls if call[0] == "pages")
    assert torch.equal(final_pages[0], pages[0]) and torch.equal(final_pages[2], pages[2])
    assert torch.equal(final_pages[1], 8192 + 16 + torch.arange(16))


def test_full_logits_diagnostic_uses_fixed_owner_without_sampling_view():
    calls = []
    obj = SimpleNamespace(
        batch_size=1,
        model=SimpleNamespace(
            context=262144, max_requests=1, scratch_page_base=8192, cfg=SimpleNamespace(vocab_size=2)
        ),
        prefill_view=SimpleNamespace(forward=lambda *args, **kwargs: calls.append("unexpected_view")),
        buckets=[1, 2, 4, 8],
        prefill_inputs={n: n for n in (1, 2, 4, 8)},
        sampling_mode="device",
        _bind_cache=lambda cache: None,
        prepare=lambda: None,
        _request_slots=lambda slots: [0],
        _request_pages=lambda pages: pages.clone(),
        _active_rows=lambda active: None,
        _slots=lambda slots: None,
        _pages=lambda pages: None,
        _positions=lambda positions: None,
        _copy=lambda value, target: None,
        _prefill_chunk=lambda n: torch.full((1, n, 2), float(n)),
        _store_last_logits=lambda logits, n: calls.append(n),
        _read_logits=lambda logits: logits,
        _replay=lambda mode: calls.append("unexpected_sample"),
        event=lambda *args, **kwargs: None,
    )
    output = method("generator.py", "QwenFlashNextGenerator", "prefill_forward")(
        obj,
        torch.zeros(1, 9, dtype=torch.int32),
        page_table=torch.zeros(1, 16, dtype=torch.int32),
        kv_cache=object(),
        prompt_lens=[9],
        return_all_logits=True,
    )
    assert calls == [8, 1]
    assert output.shape == (1, 9, 2)
    assert torch.equal(output[0, :8], torch.full((8, 2), 8.0).bfloat16())
    assert torch.equal(output[0, 8:], torch.ones(1, 2).bfloat16())
