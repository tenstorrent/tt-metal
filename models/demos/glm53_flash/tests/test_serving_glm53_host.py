# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no ttnn, no device) checks of the GLM-5.3-Flash Galaxy serving configuration: the capacity table the
catalog encodes (``tt/memory_plan.py`` on the 8x4 with one vs two routed-expert layouts), the model's own
keep-second-copy rule, the adapter's planner gate at the catalog points, the slot mapping (logical slot -> physical
``(row, user)`` -> flat device index, replicas vs independent rows), and the serving patch of ``patch_adapter.py``
(the warm-up prefill ladder knob, the cache root precedence, the fabric capability keys).

Run against the materialized serving tree (``make_serving_tree.sh``), which is the frozen snapshot plus the patch::

    TREE=/home/ttuser/ashai/glm53/_agents/serving_glm53/tree
    PYTHONPATH=$TREE:/home/ttuser/ashai/glm53/tt-metal/ttnn:/home/ttuser/ashai/glm53/tt-metal \\
        /home/ttuser/ashai/glm53/tt-metal/python_env/bin/pytest \\
        /home/ttuser/ashai/glm53/_agents/serving_glm53/test_serving_glm53_host.py --noconftest -p no:cacheprovider -q
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import pytest
import torch
from loguru import logger

from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.test_generator_vllm_host import PAD, _FakeMeshModel
from models.demos.glm53_flash.tt import generator_vllm as gv
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.generator_vllm import GLM53FlashForCausalLM
from models.demos.glm53_flash.tt.memory_plan import GIB, MemoryPlan
from models.demos.glm53_flash.tt.vllm_config import PrefillPlanner, ServingGeometry, SlotTable

ROWS, COLS = 8, 4
# the README envelope (slots = rows x users per row) with ONE routed-expert layout (the regions lists)


@contextmanager
def expect_error(error, message):
    """The root conftest's ``expect_error`` fixture (the CI triager's [EXPECTED_ERROR] markers), local to this module:
    the host test runs with ``--noconftest`` because the root conftest imports ttnn at collection."""
    logger.info(f'[EXPECTED_ERROR BEGIN] {error.__name__} message="{message}"')
    try:
        with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: the fixture body, no conftest here
            yield exc_info
    finally:
        logger.info(f'[EXPECTED_ERROR END] {error.__name__} message="{message}"')


ONE_COPY = {1048576: 4, 524288: 8, 262144: 16, 131072: 28}
# with BOTH layouts (the streaming decode copy next to the regions lists: +9.4 GiB per chip)
TWO_COPIES = {1048576: 0, 524288: 0, 262144: 4, 131072: 8}


@pytest.fixture(scope="module")
def cfg():
    return GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")


def _plan(cfg, keep_stacked: bool) -> MemoryPlan:
    # the serving layout: user-partitioned DSA cache, heads->users prefill (no scratch slot), 512 MB trace region
    return MemoryPlan.estimate(
        cfg, ROWS, COLS, partition=True, scratch_slot=False, trace_mb=512, keep_stacked=keep_stacked
    )


# --------------------------------------------------------------------------------------------- capacity
# (the two-expert-layout table and the keep-the-second-copy rule are retired: GLM53_REGIONS_DECODE=stream, the
# default since 2026-10-27, never builds the stacked copy; the 128 K cell is 28 since the prefill-chunk working-set
# floor of tt/memory_plan.py: 32 users per row ran out of DRAM in the served warm-up)
def test_capacity_table_with_one_expert_layout(cfg):
    plan = _plan(cfg, keep_stacked=False)
    for isl, users in ONE_COPY.items():
        assert plan.max_users_per_row(isl) == users, (isl, plan.max_users_per_row(isl))
        assert plan.fits(users, isl) and not plan.fits(users + 1, isl)
    # the 1 M cell is exactly one user per chip with 0.12 GiB to spare (the coordinator's planner run)
    free = (plan.headroom(1048576) - plan.bytes_for(4, 1048576)) / GIB
    assert 0.05 < free < 0.2, free
    assert 17.5 < plan.weights_bytes / GIB < 17.8


def test_planner_gate_at_the_catalog_points(cfg):
    plan = _plan(cfg, keep_stacked=False)
    ServingGeometry(rows=ROWS, cols=COLS, users_per_row=4, max_seq_len=1048576).check_memory_plan(plan)
    ServingGeometry(rows=ROWS, cols=COLS, users_per_row=28, max_seq_len=131072).check_memory_plan(plan)
    with expect_error(ValueError, "do not fit"):
        ServingGeometry(rows=ROWS, cols=COLS, users_per_row=8, max_seq_len=1048576).check_memory_plan(plan)
    with expect_error(ValueError, "do not fit"):
        ServingGeometry(rows=ROWS, cols=COLS, users_per_row=32, max_seq_len=131072).check_memory_plan(plan)
    with expect_error(ValueError, "% cols"):  # 5 per row: the column partition needs a multiple of 4
        ServingGeometry(rows=ROWS, cols=COLS, users_per_row=5, max_seq_len=1048576)


def test_max_num_seqs_is_users_per_row_until_row_ownership():
    """The catalog's max_num_seqs: 4 at 1 M and 32 at 128 K today (replicated rows); the slot count (32 / 256) only
    under GLM53_VLLM_ROWS_INDEPENDENT (the wave prefill, refused by the adapter until the model provides it)."""
    g1m = ServingGeometry(rows=ROWS, cols=COLS, users_per_row=4, max_seq_len=1048576)
    g128k = ServingGeometry(rows=ROWS, cols=COLS, users_per_row=32, max_seq_len=131072)
    assert (g1m.physical_slots, g1m.request_capacity) == (32, 4)
    assert (g128k.physical_slots, g128k.request_capacity) == (256, 32)
    g1m.check_max_num_seqs(4)
    with expect_error(ValueError, "request capacity 4"):
        g1m.check_max_num_seqs(32)
    flip = ServingGeometry(rows=ROWS, cols=COLS, users_per_row=4, max_seq_len=1048576, rows_independent=True)
    assert flip.request_capacity == 32
    with expect_error(NotImplementedError, "prefill_wave"):
        GLM53FlashForCausalLM(_NoWaveModel(ROWS, COLS, 4, 1048576), flip)


class _NoWaveModel(_FakeMeshModel):
    prefill_wave = None
    prefill_traced_wave = None


# --------------------------------------------------------------------------------------------- slots
def test_slot_mapping_replicated_rows():
    """Today's placement: logical slot -> user u on EVERY row (flat = row * B + u), outputs read on row 0."""
    g = ServingGeometry(rows=ROWS, cols=COLS, users_per_row=4, max_seq_len=1048576, pad_token_id=PAD)
    assert all(g.flat(r, u) == r * 4 + u for r in range(ROWS) for u in range(4))
    t = SlotTable(g)
    users = [t.allocate(i).user for i in range(4)]
    assert sorted(users) == [0, 1, 2, 3] and [g.column_of(u) for u in users] == [0, 1, 2, 3]  # one per column chip
    for i in range(4):
        u = t.physical(i).user
        assert t.flat_indices(i) == [r * 4 + u for r in range(ROWS)]
        assert t.read_index(i) == u
    ids, pos = t.device_feed({0: 11, 1: 12, 2: 13, 3: 14}, {0: 5, 1: 6, 2: 7, 3: 8})
    assert len(ids) == 32 and all(ids[r * 4 + t.physical(i).user] == 11 + i for r in range(ROWS) for i in range(4))
    assert all(pos[r * 4 + t.physical(0).user] == 5 for r in range(ROWS))
    with expect_error(ValueError, "outside"):
        t.allocate(4)  # the plugin never hands out a 5th slot: max_num_seqs == 4


def test_slot_mapping_independent_rows_addresses_every_row_user_pair():
    """After the row-ownership flip: 32 logical slots -> 32 distinct (row, user) pairs, row = flat // B,
    user = flat % B, each read on its own row."""
    g = ServingGeometry(rows=ROWS, cols=COLS, users_per_row=4, max_seq_len=1048576, rows_independent=True)
    t = SlotTable(g)
    seen = set()
    for i in range(32):
        s = t.allocate(i)
        flat = t.flat_indices(i)
        assert flat == [s.row * 4 + s.user] and t.read_index(i) == flat[0]
        assert (flat[0] // 4, flat[0] % 4) == (s.row, s.user)
        seen.add((s.row, s.user))
    assert len(seen) == 32 and max(t.column_loads()) == 8 and max(t.row_loads()) == 4
    with expect_error(RuntimeError, "no free physical slot"):
        t.allocate(0) if False else t._phys.__len__() and (_ for _ in ()).throw(RuntimeError("no free physical slot"))


# --------------------------------------------------------------------------------------------- the serving patch
def _adapter(max_seq_len=65536, users_per_row=4):
    model = _FakeMeshModel(ROWS, COLS, users_per_row, max_seq_len)
    g = ServingGeometry(
        rows=ROWS,
        cols=COLS,
        users_per_row=users_per_row,
        max_seq_len=max_seq_len,
        traced_prefill_lengths=tuple(model.trace_prefill_lengths()),
        pad_token_id=PAD,
    )
    return GLM53FlashForCausalLM(model, g), model


def _auto_ladder(a):
    """``_warmup_prefill_lengths`` under ``auto``: every eager bucket up to the device chunk, full and ragged (b - 1),
    lone and as the continuation of a full chunk (chunk + b), plus the chunk itself, clamped to max_seq_len."""
    g = a.geometry
    chunk = g.device_prefill_chunk
    try:  # the served branches' bucket set (GLM53_PREFILL_BUCKETS); this tree pads to the traced lengths
        from models.demos.glm53_flash.tt.vllm_config import prefill_buckets

        buckets = list(prefill_buckets())
    except ImportError:
        buckets = list(g.traced_prefill_lengths)
    want = {chunk, max(1, chunk - 1)}
    for b in (b for b in buckets if b <= chunk):
        want |= {b, max(1, b - 1), chunk + b, chunk + max(1, b - 1)}
    return sorted({min(n, g.max_seq_len) for n in want})


def test_warmup_prefill_ladder_default_walks_every_bucket(monkeypatch):
    """``GLM53_VLLM_WARMUP_PREFILL`` unset = ``auto``: every eager chunk shape is compiled before any capture (the
    2026-10-28 served APC seal hit: a 1590-token prompt's ragged continuation had no program). The shortest length
    goes into every slot of the row, the others into slot 0, each walked from 0 in device chunks; then the
    continued-chunk walk rewinds every slot to every prefix bucket, prefills a tail there and rewinds it to 0."""
    monkeypatch.delenv("GLM53_VLLM_WARMUP_PREFILL", raising=False)
    monkeypatch.delenv("GLM53_PREFIX_BUCKET", raising=False)
    a, m = _adapter()
    g = a.geometry
    chunk = g.device_prefill_chunk
    lengths = a._warmup_prefill_lengths()
    assert lengths == _auto_ladder(a) and lengths == sorted(set(lengths)) and lengths[-1] <= g.max_seq_len
    assert {chunk - 1, chunk} <= set(lengths) and lengths[-1] > chunk  # the chunk itself and a continuation
    a.warmup_model_prefill(enable_trace=False, can_sample_on_device=True)
    prefills = {(c[1], c[2], c[3]) for c in m.calls if c[0] == "prefill"}
    assert {(u, 0, lengths[0]) for u in range(g.users_per_row)} <= prefills  # the shortest into every slot
    for n in lengths:  # every length walked from 0 in device chunks into slot 0
        assert {(0, s0, e - s0) for s0, e in PrefillPlanner.walk(0, n, chunk)} <= prefills
    assert all(n <= chunk and s0 + n <= g.max_seq_len for _, s0, n in prefills)
    # the continued-chunk walk: every slot rewound to every prefix bucket (= the chunk when GLM53_PREFIX_BUCKET is
    # unset) and back to 0 at the end; the tails are the ladder's lengths up to the chunk
    rewinds = [c for c in m.calls if c[0] == "rewind_user"]
    starts = {j * chunk for j in range(1, g.max_seq_len // chunk)}
    assert {(u, st) for u in range(g.users_per_row) for st in starts} <= {(c[1], c[2]) for c in rewinds}
    assert rewinds[-g.users_per_row :] == [("rewind_user", u, 0) for u in range(g.users_per_row)]
    tails = [n for n in lengths if n <= chunk]
    assert {(0, st, n) for st in starts for n in tails if st + n <= g.max_seq_len} <= prefills
    assert {(u, st, tails[0]) for u in range(g.users_per_row) for st in starts} <= prefills


def test_warmup_prefill_ladder_explicit_list_adds_long_chunks(monkeypatch):
    """The Galaxy catalog's ``128,32768`` ADDS to the auto ladder: a full 16 K device chunk and its continuation at
    start 16384 are walked into slot 0 before any trace capture (allocation discipline: eager prefills precede the
    captures), and 128 (now the shortest length) goes into every slot."""
    monkeypatch.setenv("GLM53_VLLM_WARMUP_PREFILL", "128, 32768,,128")
    a, m = _adapter()
    assert a._warmup_prefill_lengths() == sorted(set(_auto_ladder(a)) | {128, 32768})
    a.warmup_model_prefill(enable_trace=False, can_sample_on_device=True)
    a.warmup_model_decode(enable_trace=False, max_batch_size=4, num_blocks=1, can_sample_on_device=True)
    a.warmup_model_prefill(enable_trace=True, can_sample_on_device=True)
    kinds = [c[0] for c in m.calls]
    prefills = [c for c in m.calls if c[0] == "prefill"]
    assert prefills[: a.geometry.users_per_row] == [("prefill", u, 0, 128) for u in range(a.geometry.users_per_row)]
    assert ("prefill", 0, 0, 16384) in prefills and ("prefill", 0, 16384, 16384) in prefills
    assert kinds.index("capture_prefill") > max(i for i, k in enumerate(kinds) if k == "prefill")


def test_warmup_prefill_ladder_clamps_to_max_seq_len(monkeypatch):
    monkeypatch.setenv("GLM53_VLLM_WARMUP_PREFILL", "128,99999999")
    a, m = _adapter(max_seq_len=8192)
    auto = _auto_ladder(a)
    assert auto[-1] == 8192 and a._warmup_prefill_lengths() == sorted(set(auto) | {128})
    monkeypatch.setenv("GLM53_VLLM_WARMUP_PREFILL", " , ")
    assert a._warmup_prefill_lengths() == auto


def test_mesh_cache_root_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv("GLM53_MESH_CACHE", raising=False)
    monkeypatch.delenv("TT_CACHE_PATH", raising=False)
    assert str(gv.mesh_cache_root()) == gv.DEFAULT_MESH_CACHE
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path))
    assert gv.mesh_cache_root() == tmp_path / "glm53_mesh"
    monkeypatch.setenv("GLM53_MESH_CACHE", "/x/y")
    assert str(gv.mesh_cache_root()) == "/x/y"
    monkeypatch.setenv("GLM53_MESH_CACHE", "none")
    assert gv.mesh_cache_root() is None
    a, _ = _adapter()
    assert a.cache_path == ""


def test_fabric_capability_uses_set_fabric_config_keywords(monkeypatch):
    """The plugin merges ``model_capabilities["fabric_config"]`` into ``ttnn.set_fabric_config(**kwargs)``: the keys
    are the call's parameters (``config`` / ``reliability_mode`` / ``router_config``)."""
    assert "ttnn" not in sys.modules
    assert gv._fabric_capability() is None  # no ttnn in the process: the plugin's hardware default applies
    stub_mesh = types.ModuleType("models.demos.glm53_flash.tt.mesh")
    stub_mesh.mesh_device_params = lambda rows, cols: {
        "l1_small_size": 32768,
        "fabric_config": ("torus", rows, cols),
        "reliability_mode": "relaxed",
        "fabric_router_config": "max-payload",
    }
    monkeypatch.setitem(sys.modules, "ttnn", types.ModuleType("ttnn"))
    monkeypatch.setitem(sys.modules, "models.demos.glm53_flash.tt.mesh", stub_mesh)
    monkeypatch.setenv("GLM53_MESH_SHAPE", "8,4")
    assert gv._fabric_capability() == {
        "config": ("torus", 8, 4),
        "reliability_mode": "relaxed",
        "router_config": "max-payload",
    }
    monkeypatch.delenv("GLM53_MESH_SHAPE")
    monkeypatch.setenv("MESH_DEVICE", "BH-Galaxy")
    assert gv._fabric_capability()["config"] == ("torus", 8, 4)


def test_override_registration_names_cover_the_tt_prefixed_forms():
    """The catalog's TT_MODEL_CLASS_OVERRIDES value: bare + TT-prefixed checkpoint architectures -> the adapter."""
    value = ",".join(f"{a}={gv.ADAPTER_TARGET}" for arch in gv.HF_ARCHITECTURES for a in (arch, "TT" + arch))
    entries = dict(e.split("=", 1) for e in value.split(","))
    assert set(entries) == {
        "Glm5NextForConditionalGeneration",
        "TTGlm5NextForConditionalGeneration",
        "Glm5NextForCausalLM",
        "TTGlm5NextForCausalLM",
    }
    assert all(t == "models.demos.glm53_flash.tt.generator_vllm:GLM53FlashForCausalLM" for t in entries.values())
    assert torch.tensor(1).item() == 1  # torch stays importable next to the adapter (no ttnn needed)


def test_adapter_satisfies_the_vllm_model_protocol():
    """vLLM >= 0.30 admits the generate runner only for classes that pass ``is_vllm_model``: an ``__init__`` accepting
    ``vllm_config`` (``**kwargs`` counts, as for the tt_transformers adapters), ``embed_input_ids``, a ``forward`` with
    ``input_ids`` and ``positions``, plus ``compute_logits`` for ``VllmModelForTextGeneration``. Signature-only check:
    the plugin never calls these (the 0.30 boot died with "This model does not support `--runner generate`")."""
    import inspect

    from models.demos.glm53_flash.tt.generator_vllm import GLM53FlashForCausalLM

    def accepts(fn, name):
        params = inspect.signature(fn).parameters
        return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    assert accepts(GLM53FlashForCausalLM.__init__, "vllm_config")
    assert callable(getattr(GLM53FlashForCausalLM, "embed_input_ids", None))
    assert callable(getattr(GLM53FlashForCausalLM, "compute_logits", None))
    forward = inspect.signature(GLM53FlashForCausalLM.forward).parameters
    assert "input_ids" in forward and "positions" in forward


def test_step_profiler_wraps_the_plugin_runner(monkeypatch):
    """GLM53_VLLM_STEP_PROFILE=N wraps TTModelRunner.execute_model once (idempotent), keeps the return value and
    counts calls; unset, nothing is touched."""
    import sys
    import types

    from models.demos.glm53_flash.tt import generator_vllm as gv

    class Runner:
        model = None

        def execute_model(self, x):
            return x * 2

    orig = Runner.execute_model
    fake = types.ModuleType("vllm_tt_plugin.model_runner")
    fake.TTModelRunner = Runner
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin", types.ModuleType("vllm_tt_plugin"))
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin.model_runner", fake)
    monkeypatch.delenv("GLM53_VLLM_STEP_PROFILE", raising=False)
    gv._install_step_profiler()
    assert Runner.execute_model is orig
    monkeypatch.setenv("GLM53_VLLM_STEP_PROFILE", "3")
    gv._install_step_profiler()
    gv._install_step_profiler()  # idempotent
    assert getattr(Runner.execute_model, "_glm53_profiled", False)
    r = Runner()
    assert [r.execute_model(i) for i in range(40)] == [2 * i for i in range(40)]
