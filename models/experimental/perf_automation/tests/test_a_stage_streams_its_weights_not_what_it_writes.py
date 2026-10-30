# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A decode step was charged for every buffer it was HANDED, and handed is not read.

Devstral 2026-10-15, per chip: trace_replay's hook reported 9.03 GB for one decode step. The step
streams about 4.6 GB of weights. The rest was the whole 4096-row KV cache allocation (2.85 GB, of
which the step reads the rows filled so far and writes one), the replicated bf16 embedding table
(1.34 GB, of which a token reads one row) and the activations the step itself produced. The memory
floor came out at 17.6 ms, a 17.98 ms step read as 98% of peak bandwidth -- "in band" -- and the
stop gate cleared the stack with half its ceiling still above it.

The hook now reports what the step STREAMS:

    * a buffer an op produced or updated during the step is state the step writes -- not streamed;
    * a buffer an op returns when it was handed only that buffer is a view -- still streamed;
    * a table the pipeline DECLARED it reads by index (stage_seams.GATHERED_TABLES_ATTR), handed to an
      op beside an integer index, is charged the rows that came out -- whatever the op is called.

What the step writes is priced on top by the byte model's per-item term, which summary now adds to
the measured figure exactly as it always added it to the estimate.
"""
import importlib
import sys
import types
from pathlib import Path


_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))

_BPE = {"bfloat16": 2, "bfloat8_b": 1.0625, "float32": 4, "uint32": 4, "int32": 4}


class _T:
    """A device tensor as the hook sees it: a buffer address, an element count, a dtype name."""

    def __init__(self, numel, addr, dt="bfloat16"):
        self._numel, self._addr, self._dt = numel, addr, dt

    def buffer_address(self):
        return self._addr

    def view(self):
        """The same buffer seen through another Tensor object -- what reshape returns."""
        return _T(self._numel, self._addr, self._dt)

    @property
    def nbytes(self):
        return int(self._numel * _BPE[self._dt])


def _load(monkeypatch):
    """trace_replay imports ttnn at module scope and the tool tree has none, so a stub goes into
    sys.modules first. The op class carries python_fully_qualified_name the way FastOperation does;
    each test sets it per op and decides what the call returns."""
    dec = types.ModuleType("ttnn.decorators")

    class FastOperation:
        python_fully_qualified_name = "ttnn.anything"

        def __init__(self, name="ttnn.anything", returns=None):
            self.python_fully_qualified_name = name
            self._returns = returns

        def __call__(self, *a, **k):
            return self._returns(*a, **k) if callable(self._returns) else self._returns

    dec.FastOperation = FastOperation
    pkg = types.ModuleType("ttnn")
    pkg.decorators = dec
    monkeypatch.setitem(sys.modules, "ttnn", pkg)
    monkeypatch.setitem(sys.modules, "ttnn.decorators", dec)
    monkeypatch.delitem(sys.modules, "agent.trace_replay", raising=False)
    tr = importlib.import_module("agent.trace_replay")
    from agent import weight_census as wc

    monkeypatch.setattr(wc, "_tensor_entry", lambda t: (t._numel, t._dt) if isinstance(t, _T) else None)
    monkeypatch.setattr(wc, "bytes_per_elem", lambda name: _BPE[str(name)])
    return tr, FastOperation


def _streamed(tr, fn, tables=None):
    detail = {}
    n, b = tr._count_op_dispatches(fn, detail, tables)
    return n, b, detail


# ---- what the step writes is not streamed --------------------------------------------------------


def test_what_an_op_produces_is_not_streamed(monkeypatch):
    """An activation is made by one op and read by the next. It is the step's own output, not a
    weight it streams; counting it was how the step's working set grew with its depth."""
    tr, Op = _load(monkeypatch)
    w, act = _T(1_000_000, 0x10), _T(50_000, 0x20)
    matmul = Op(returns=act)  # x @ w -> act
    consume = Op(returns=_T(50_000, 0x30))
    x = _T(10_000, 0x40)
    n, b, d = _streamed(tr, lambda: consume(matmul(x, w)))
    assert n == 2
    assert b == w.nbytes + x.nbytes, b  # the weight, and the step's input; not act
    assert d["in_step_bytes"] == act.nbytes


def test_a_cache_updated_in_place_is_not_streamed(monkeypatch):
    """THE KV CACHE. update_cache is handed the whole allocation and the new row, writes the row,
    and returns the allocation. Attention then reads the filled rows -- how many is a property of
    the context, priced by the byte model from seq_len, not of the allocation's size."""
    tr, Op = _load(monkeypatch)
    cache = _T(32 * 2 * 4096 * 128, 0x100, "bfloat8_b")  # one layer's K cache, 4096-row capacity
    w, q = _T(2_000_000, 0x200), _T(4096, 0x300)
    new_row = _T(4096, 0x400)
    update = Op("ttnn.experimental.paged_update_cache", returns=lambda c, row, **k: c)  # in place
    attend = Op(returns=_T(4096, 0x500))
    projected = _T(4096, 0x700)
    proj = Op(returns=projected)
    pos = _T(32, 0x600, "int32")

    def step():
        update(cache, new_row, update_idxs_tensor=pos)
        attend(proj(q, w), cache)

    n, b, d = _streamed(tr, step)
    assert n == 3
    assert b == w.nbytes + q.nbytes + new_row.nbytes + pos.nbytes, b
    assert d["in_step_bytes"] == cache.nbytes + projected.nbytes, "the allocation was charged as streamed"


def test_a_preallocated_output_handed_in_a_list_is_not_streamed(monkeypatch):
    """generic_op is handed [x, w, out] and returns out: the output rides in the io list, as an
    argument, and must not be mistaken for something the program read."""
    tr, Op = _load(monkeypatch)
    x, w, out = _T(5_000, 0x1), _T(3_000_000, 0x2), _T(5_000, 0x3)
    generic = Op("ttnn.generic_op", returns=lambda io, desc: io[-1])
    n, b, d = _streamed(tr, lambda: generic([x, w, out], object()))
    assert b == x.nbytes + w.nbytes, b
    assert d["in_step_bytes"] == out.nbytes


# ---- a view is not a write ----------------------------------------------------------------------


def test_a_weight_seen_through_a_view_stays_streamed(monkeypatch):
    """reshape returns the same buffer it was handed and nothing else was handed: nothing was
    written. A weight reshaped before its matmul must still count once, in full."""
    tr, Op = _load(monkeypatch)
    w = _T(1_000_000, 0x10)
    reshape = Op("ttnn.reshape", returns=lambda t, shape: t.view())
    matmul = Op(returns=_T(100, 0x99))
    x = _T(100, 0x50)
    _n, b, d = _streamed(tr, lambda: matmul(x, reshape(w, (1, 1000, 1000))))
    assert b == w.nbytes + x.nbytes, b
    assert d["in_step_bytes"] == 0


def test_an_op_that_returns_one_of_several_inputs_wrote_into_it(monkeypatch):
    """output_tensor=: the op is handed a destination among other tensors and returns it. With
    others to write from, returning an input is an update, not a view."""
    tr, Op = _load(monkeypatch)
    a, c, dst = _T(1000, 0x1), _T(1000, 0x2), _T(1000, 0x3)
    add = Op("ttnn.add", returns=lambda x, y, output_tensor=None: output_tensor)
    _n, b, d = _streamed(tr, lambda: add(a, c, output_tensor=dst))
    assert b == a.nbytes + c.nbytes, b
    assert d["in_step_bytes"] == dst.nbytes


# ---- a declared table is charged what came out of it ----------------------------------------------


def test_a_declared_table_is_charged_the_rows_that_came_out(monkeypatch):
    """A 131072 x 5120 bf16 embedding table is 1.34 GB; a step with 32 users reads 32 rows of it.
    The index tensor is read whole, and is tiny."""
    tr, Op = _load(monkeypatch)
    table = _T(131072 * 5120, 0x10)
    ids = _T(32, 0x20, "uint32")
    rows = _T(32 * 5120, 0x30)
    embed = Op("ttnn.embedding", returns=rows)
    _n, b, d = _streamed(tr, lambda: embed(ids, table), tables=[table])
    assert b == rows.nbytes + ids.nbytes, b
    assert d["gathered_saved_bytes"] == table.nbytes - rows.nbytes


def test_the_rule_does_not_care_what_the_op_is_called(monkeypatch):
    """Devstral run 4 (2026-10-18), round 1: the agent replaced ttnn.embedding with its own generic_op
    program -- io list [ids, table, out], out preallocated -- and a rule keyed on the op's API name
    charged the whole 1.34 GB table again. The declaration names the TENSOR; the op's name is nothing."""
    tr, Op = _load(monkeypatch)
    table = _T(131072 * 5120, 0x10)
    ids = _T(32, 0x20, "uint32")
    out = _T(32 * 5120, 0x30, "float32")
    gather = Op("ttnn.generic_op", returns=lambda io, *a, **k: io[-1])
    _n, b, d = _streamed(tr, lambda: gather([ids, table, out]), tables=[table])
    assert b == out.nbytes + ids.nbytes, b  # the table at what came out of it; the index whole
    assert d["in_step_bytes"] == out.nbytes, "the preallocated output is what the step wrote"
    assert d["gathered_saved_bytes"] == table.nbytes - out.nbytes


def test_a_declared_table_without_an_index_beside_it_is_streamed(monkeypatch):
    """Tied embeddings: the same table is gathered for the input and multiplied for the logits. The
    matmul is handed no index, so it streams the table, and streaming it once is the real cost."""
    tr, Op = _load(monkeypatch)
    table = _T(131072 * 5120, 0x10)
    ids = _T(32, 0x20, "uint32")
    h = _T(32 * 5120, 0x50)
    embed = Op("ttnn.embedding", returns=_T(32 * 5120, 0x30))
    head = Op("ttnn.matmul", returns=_T(32 * 131072, 0x40))
    _n, b, d = _streamed(tr, lambda: (embed(ids, table), head(h, table)), tables=[table])
    assert b == table.nbytes + ids.nbytes + h.nbytes, b
    assert d["gathered_saved_bytes"] == 0


def test_an_undeclared_table_is_charged_in_full_whatever_the_op_is_called(monkeypatch):
    """The trade-off, stated: a pipeline that declares nothing has its tables counted whole. The floor
    then reads too high and the stack too short, which keeps the run working rather than stopping
    it; the marker carries gathered_saved=0, so the omission is visible, not silent."""
    tr, Op = _load(monkeypatch)
    table = _T(131072 * 5120, 0x10)
    ids = _T(32, 0x20, "uint32")
    embed = Op("ttnn.embedding", returns=_T(32 * 5120, 0x30))
    _n, b, d = _streamed(tr, lambda: embed(ids, table))
    assert b == table.nbytes + ids.nbytes, b
    assert d["gathered_saved_bytes"] == 0


def test_a_declared_table_handed_to_an_op_that_returned_nothing_is_streamed(monkeypatch):
    """Nothing came out to price the read by, so nothing is assumed: the table is read whole."""
    tr, Op = _load(monkeypatch)
    table = _T(131072 * 5120, 0x10)
    ids = _T(32, 0x20, "uint32")
    op = Op("ttnn.generic_op", returns=None)
    _n, b, d = _streamed(tr, lambda: op([ids, table]), tables=[table])
    assert b == table.nbytes + ids.nbytes, b
    assert d["gathered_saved_bytes"] == 0


def test_an_integer_the_op_wrote_is_not_its_index(monkeypatch):
    """A sampler handed a declared table and a preallocated uint32 output selects nothing from the
    table by that output: the integer tensor an op WRITES is not the index it reads by, so the table
    is streamed in full."""
    tr, Op = _load(monkeypatch)
    table = _T(131072 * 5120, 0x10)
    drawn = _T(32, 0x20, "uint32")
    op = Op("ttnn.generic_op", returns=lambda io, *a, **k: io[-1])
    _n, b, d = _streamed(tr, lambda: op([table, drawn]), tables=[table])
    assert b == table.nbytes, b
    assert d["in_step_bytes"] == drawn.nbytes and d["gathered_saved_bytes"] == 0


def test_an_integer_beside_an_undeclared_weight_does_not_make_a_gather(monkeypatch):
    """MoE routing hands expert indices beside expert weights and reads the experts, not rows of
    them. An index alone proves nothing; the declaration is what names a table."""
    tr, Op = _load(monkeypatch)
    experts = _T(8 * 4096 * 14336, 0x10, "bfloat8_b")
    routes = _T(32 * 2, 0x20, "uint32")
    moe = Op("ttnn.experimental.moe", returns=_T(32 * 4096, 0x30))
    _n, b, _d = _streamed(tr, lambda: moe(_T(32 * 4096, 0x40), experts, routes))
    assert b == experts.nbytes + routes.nbytes + _T(32 * 4096, 0x40).nbytes, b


def test_the_declaration_is_matched_by_buffer_not_by_object_and_may_nest(monkeypatch):
    """The pipeline declares its tables however it holds them -- a dict of lists is fine -- and an op
    handed the same buffer through another Tensor object (a view) still reads rows of it."""
    tr, Op = _load(monkeypatch)
    table = _T(131072 * 5120, 0x10)
    pos = _T(4096 * 128, 0x60)
    ids = _T(32, 0x20, "uint32")
    rows = _T(32 * 5120, 0x30)
    prow = _T(32 * 128, 0x70)
    embed = Op("ttnn.embedding", returns=rows)
    rope = Op("ttnn.generic_op", returns=prow)
    _n, b, d = _streamed(
        tr, lambda: (embed(ids, table.view()), rope([ids, pos])), tables={"embed": [table], "rope": (pos,)}
    )
    assert b == rows.nbytes + prow.nbytes + ids.nbytes, b
    assert d["gathered_saved_bytes"] == (table.nbytes - rows.nbytes) + (pos.nbytes - prow.nbytes)


# ---- the figure still means what its consumers expect -------------------------------------------


def test_a_weight_read_by_two_ops_still_counts_once(monkeypatch):
    tr, Op = _load(monkeypatch)
    w = _T(1_000_000, 0x10)
    op = Op(returns=None)
    _n, b, _d = _streamed(tr, lambda: (op(w), op(w)))
    assert b == w.nbytes


def test_the_two_tuple_contract_holds_without_a_detail_dict(monkeypatch):
    """Every existing caller unpacks (dispatches, bytes); the breakdown is opt-in."""
    tr, Op = _load(monkeypatch)
    w = _T(10, 0x10)
    got = tr._count_op_dispatches(lambda: Op(returns=None)(w))
    assert got == (1, w.nbytes)


def test_the_marker_carries_the_breakdown_behind_the_integer_the_parser_reads(monkeypatch, capsys):
    tr, _ = _load(monkeypatch)
    tr._report_read_set(
        "decode", 309, 4_600_000_000, {"in_step_bytes": 2_900_000_000, "gathered_saved_bytes": 1_340_000_000}
    )
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("TRACE_STAGE_BYTES[decode]="))
    assert line == "TRACE_STAGE_BYTES[decode]=4600000000 ops=309 in_step=2900000000 gathered_saved=1340000000"
    # the same read perf_mcp performs on the line
    assert int(float(line.split("]=", 1)[1].split()[0])) == 4_600_000_000
    assert line.split("TRACE_STAGE_BYTES[", 1)[1].split("]", 1)[0] == "decode"


# ---- summary adds the per-item term to whichever source answered -------------------------------


def _srb(monkeypatch, pinned=None):
    import importlib.util

    spec = importlib.util.spec_from_file_location("sm_streams", _PA / "cc_optimize" / "summary.py")
    sm = importlib.util.module_from_spec(spec)
    sys.modules["sm_streams"] = sm
    spec.loader.exec_module(sm)
    monkeypatch.setattr(sm, "_pinned_stage_bytes", lambda stage, model="", task="": pinned)
    return sm.stage_read_bytes


def test_the_per_item_term_rides_on_the_measured_streamed_set(monkeypatch):
    srb = _srb(monkeypatch)
    got = srb(
        "decode",
        measured={"decode": 4_600_000_000},
        estimate=lambda: 999,
        per_item=lambda: 90_000_000,
        with_source=True,
    )
    assert got == (4_690_000_000, "measured"), got


def test_the_per_item_term_rides_on_the_pinned_and_the_estimated_set_alike(monkeypatch):
    assert _srb(monkeypatch, pinned=1000)("decode", measured={"decode": 500}, per_item=lambda: 7) == 1007
    assert _srb(monkeypatch)("decode", estimate=lambda: 300, per_item=lambda: 7) == 307


def test_no_streamed_set_means_no_floor_even_with_a_per_item_term(monkeypatch):
    """A KV term on top of no weights would be a confident floor resting on nothing streamed."""
    assert _srb(monkeypatch)("decode", estimate=lambda: 0, per_item=lambda: 7) == 0


def test_a_per_item_model_that_raises_leaves_the_streamed_set_standing(monkeypatch):
    def _boom():
        raise RuntimeError("no geometry")

    assert _srb(monkeypatch)("decode", measured={"decode": 500}, per_item=_boom) == 500


def test_callers_that_pass_no_per_item_term_see_the_old_answer(monkeypatch):
    srb = _srb(monkeypatch)
    assert srb("decode", measured={"decode": 500}, estimate=lambda: 999) == 500
    assert srb("decode", estimate=lambda: 999) == 999


def test_the_roof_adds_the_same_per_item_term_to_the_measured_and_the_estimated_set(monkeypatch):
    """_stage_roofs: with a measurement, bytes = measured + per_item; without, the estimate's own
    two addends -- and bytes_now is in the same currency as bytes."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("sm_roofs", _PA / "cc_optimize" / "summary.py")
    sm = importlib.util.module_from_spec(spec)
    sys.modules["sm_roofs"] = sm
    spec.loader.exec_module(sm)
    mf = {"total_params": 1_000_000, "layers": 2, "hidden_size": 64}
    monkeypatch.setattr(sm, "_model_facts", lambda: mf)
    monkeypatch.setattr(sm, "_pinned_stage_bytes", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_pinned_ceiling_input", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_pinned_peak_flops", lambda *a, **k: None)
    monkeypatch.setattr(sm, "_prompt_tokens", lambda: 128)
    monkeypatch.setattr(sm, "_request_batch", lambda: 1)
    monkeypatch.setattr(sm, "_fidelity_breakdown", lambda p: ([], None))
    monkeypatch.setattr(sm, "_per_request_stages", lambda: set())
    monkeypatch.setattr(sm, "_share_and_basis", lambda mf_, stage: (1.0, "single tower"))
    monkeypatch.setattr(sm, "_stage_units", lambda n, pt, prof: 1)
    monkeypatch.setattr(sm, "_peak_for_stage", lambda *a, **k: (0.0, ""))
    monkeypatch.setattr(sm, "_observed_rung", lambda *a, **k: "")
    monkeypatch.setattr(
        sm, "_ledger", lambda: types.SimpleNamespace(anchor_value=lambda *a, **k: None, KIND_PEAK_FLOPS="peak")
    )
    import agent.perf_target as pt

    monkeypatch.setattr(
        pt, "active_bytes", lambda mf_, **kw: 1000 + 10 * int(kw.get("seq_len") or 0) * int(kw.get("batch") or 1)
    )
    monkeypatch.setattr(pt, "ceiling_params", lambda mf_: 1_000_000)
    monkeypatch.setattr(pt, "chip_peak_flops", lambda *a, **k: 1.0e12)
    per_item = 10 * 128  # (1000 + 1280) - 1000, tp=1

    monkeypatch.setattr(sm, "_measured_stage_bytes", lambda model, task: {"decode": 4_600_000})
    r = sm._stage_roofs(10_000_000, 512.0, 1, "tok/s/u", {"buckets": []}, {"decode": 17.98})
    assert r["decode"]["bytes_source"] == "measured"
    assert r["decode"]["bytes"] == 4_600_000 + per_item, r["decode"]
    assert r["decode"]["bytes_now"] == r["decode"]["bytes"]

    monkeypatch.setattr(sm, "_measured_stage_bytes", lambda model, task: {})
    r = sm._stage_roofs(10_000_000, 512.0, 1, "tok/s/u", {"buckets": []}, {"decode": 17.98})
    assert r["decode"]["bytes_source"] == "estimate"
    assert r["decode"]["bytes"] == 10_000_000 + per_item, r["decode"]
    assert r["decode"]["bytes_now"] is None
