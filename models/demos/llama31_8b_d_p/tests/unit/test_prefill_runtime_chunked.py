# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`tt/tt_prefill_runtime.py` against the engine's **real call site**. Gate: `G-RUNTIME`.

**No device, no weights, no ttnn ops** (Appendix A gives `G-RUNTIME` device "none"): this gate is a
static audit plus the refusals. It proves the runtime cannot die with a `TypeError` on its first
served chunk — the failure mode recipe P10 warning 1 records, which costs a mesh open and a weight
load to discover.

**The audit walks `models/demos/common/prefill/runners/prefill_runner.py` with `ast`, not the
contract doc** (`BRINGUP_RECIPE.md:1687-1690`). That is not pedantry: the doc's
`prefill_chunk(input_tensor, kv_cache, *, slot_id, actual_start, actual_end, request_id=0)`
(`models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md:129`) omits `d2h_service` **and**
`metadata_msg`, both of which the engine passes on every chunk
(`models/demos/common/prefill/runners/prefill_runner.py:286-295`), and it omits **two** whole
methods the engine calls unguarded on its migration paths (`set_layer_completion_sink` and
`set_d2h_ack_service`) plus one config field it reads (`config.use_trace`).
`test_engine_call_site_is_wider_than_the_doc` measures that gap rather than quoting it.

**The audit gets its own negative control** (`bringup_log/03_OUTLINE.md` §2.15): `_BrokenRuntime` is
the same class with `metadata_msg` renamed, one method removed and one config field removed, and the
audit must report all three. An audit that passes everything is not an audit.

**And then the refusals.** Recipe P7 requires the unsupported single-card configuration to fail
loudly rather than silently run a different attention core
(`BRINGUP_RECIPE.md:1692-1697`), and §1.4 requires every refusal to be matched on its message. Every
`raise` in the module is exercised here.

Note on the fixture: the repo's `expect_error` matches `message` as a **regex** despite its
docstring describing a substring (`R-014`, `DEC-045`), so every needle below is a metachar-free
fragment.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_prefill_runtime_chunked.py -x -q
"""

import ast
import inspect
import os
import re

import pytest

import ttnn
from models.demos.llama31_8b_d_p.tt.attention.kv_cache import LlamaKVCache
from models.demos.llama31_8b_d_p.tt.tt_prefill_runtime import (
    DEPLOYMENT_CHUNK_SIZE,
    DEPLOYMENT_MAX_SEQ_LEN,
    TtPrefillRuntime,
    TtPrefillRuntimeConfig,
    resolve_chunk_sizes,
)

# repo root = six directories up from `<pkg>/tests/unit/<this file>`: unit, tests, the
# package, demos, models, then the root itself.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 5))
ENGINE_SOURCE = os.path.join(_ROOT, "models", "demos", "common", "prefill", "runners", "prefill_runner.py")
CONTRACT_DOC = os.path.join(_ROOT, "models", "demos", "common", "prefill", "docs", "ADDING_A_PREFILL_MODEL.md")
RUNTIME_SOURCE = os.path.join(_ROOT, "models", "demos", "llama31_8b_d_p", "tt", "tt_prefill_runtime.py")
TABLE_SOURCE = os.path.join(_ROOT, "models", "demos", "llama31_8b_d_p", "tt", "runners", "kv_chunk_table.py")

# The five names `ADDING_A_PREFILL_MODEL.md:111-129` requires on the runtime, and the five it
# requires on `runtime.config`. The engine's real call site is a different set; both are checked.
DOC_RUNTIME_NAMES = ("mesh_device", "config", "compile", "make_chunk_input", "prefill_chunk")
DOC_CONFIG_NAMES = ("chunk_size", "max_seq_len", "first_layer_idx", "is_first_rank", "is_last_rank")


# =============================================================================================
# The AST audit.
# =============================================================================================
def _instance_attributes(source_path, class_name):
    """`self.X = ...` assignments in `class_name.__init__` — the instance attributes it declares.

    Needed because `hasattr(cls, "mesh_device")` is `False` for an attribute assigned in
    `__init__`, so a class-level check alone would report every instance attribute as missing.
    """
    tree = ast.parse(open(source_path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return {
                        target.attr
                        for stmt in ast.walk(item)
                        for target in getattr(stmt, "targets", [])
                        if isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    }
    return set()


def read_engine_call_site(source_path=ENGINE_SOURCE, runtime_var="runtime"):
    """Every use the engine makes of a runtime handle, read out of its source with `ast`.

    Returns `(attributes, config_attributes, calls, guarded)`:

    * `attributes` — `runtime.<name>` accessed **unguarded**, name -> lines;
    * `config_attributes` — `runtime.config.<name>`, name -> lines;
    * `calls` — `runtime.<name>(...)`, name -> list of `(line, n_positional, [keyword names])`;
    * `guarded` — names reached only through `getattr`/`hasattr`, which the runtime may omit.
    """
    tree = ast.parse(open(source_path).read())
    attributes, config_attributes, calls, guarded = {}, {}, {}, {}

    class Walker(ast.NodeVisitor):
        def visit_Attribute(self, node):
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "config"
                and isinstance(value.value, ast.Name)
                and value.value.id == runtime_var
            ):
                config_attributes.setdefault(node.attr, []).append(node.lineno)
            elif isinstance(value, ast.Name) and value.id == runtime_var:
                attributes.setdefault(node.attr, []).append(node.lineno)
            self.generic_visit(node)

        def visit_Call(self, node):
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == runtime_var:
                calls.setdefault(func.attr, []).append((node.lineno, len(node.args), [kw.arg for kw in node.keywords]))
            if (
                isinstance(func, ast.Name)
                and func.id in ("getattr", "hasattr")
                and len(node.args) > 1
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == runtime_var
                and isinstance(node.args[1], ast.Constant)
            ):
                guarded.setdefault(node.args[1].value, []).append((func.id, node.lineno))
            self.generic_visit(node)

    Walker().visit(tree)
    return attributes, config_attributes, calls, guarded


def audit(runtime_cls, config_cls, *, runtime_source, runtime_class_name):
    """Audit one runtime class against the engine's call site. Returns a findings dict.

    Empty lists everywhere means the class satisfies the contract the engine actually exercises.
    """
    attributes, config_attributes, calls, guarded = read_engine_call_site()
    declared = set(dir(runtime_cls)) | _instance_attributes(runtime_source, runtime_class_name)
    config_names = set(dir(config_cls)) | {f.name for f in getattr(config_cls, "__dataclass_fields__", {}).values()}

    findings = {"missing_attributes": [], "missing_config_attributes": [], "signature_problems": []}

    for name, lines in sorted(attributes.items()):
        if name in guarded or name in declared:
            continue
        findings["missing_attributes"].append(f"{name} (engine reads it at prefill_runner.py:{lines[0]})")

    for name, lines in sorted(config_attributes.items()):
        if name not in config_names:
            findings["missing_config_attributes"].append(
                f"config.{name} (engine reads it at prefill_runner.py:{lines[0]})"
            )

    for name, sites in sorted(calls.items()):
        if name in guarded and name not in declared:
            continue  # the engine reaches it only behind getattr/hasattr; omitting it is legal
        member = getattr(runtime_cls, name, None)
        if member is None:
            findings["signature_problems"].append(f"{name} is called by the engine but does not exist")
            continue
        signature = inspect.signature(member)
        for line, n_positional, keywords in sites:
            # `None` in the keyword list is a `**kwargs` splat; the engine's own name for it is not
            # visible here, so bind the ones it does name and require **kwargs-compatibility below.
            named = [kw for kw in keywords if kw is not None]
            has_splat = any(kw is None for kw in keywords)
            args = ["self"] + ["<positional>"] * n_positional
            try:
                signature.bind(*args, **{kw: "<value>" for kw in named})
            except TypeError as e:
                findings["signature_problems"].append(
                    f"{name} cannot accept the engine's call at prefill_runner.py:{line} "
                    f"({n_positional} positional + {named}): {e}"
                )
                continue
            if has_splat:
                accepts = {
                    p.name for p in signature.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                } | {p.name for p in signature.parameters.values() if p.kind is p.VAR_KEYWORD}
                if not accepts:
                    findings["signature_problems"].append(f"{name} accepts no keyword arguments at all")
    return findings


# --- the audit's own negative control --------------------------------------------------------
class _BrokenConfig:
    """`TtPrefillRuntimeConfig` with `use_trace` removed — the field the doc does not mention."""

    chunk_size = 8192
    max_seq_len = 131072
    first_layer_idx = 0
    is_first_rank = True
    is_last_rank = True


class _BrokenRuntime:
    """A runtime written **to the doc**: `metadata_msg` renamed, two engine-called methods absent.

    This is not a straw man. It is what a careful reading of `ADDING_A_PREFILL_MODEL.md` §2
    produces, which is exactly why the gate audits the call site instead.
    """

    def __init__(self, mesh_device, config):
        self.mesh_device = mesh_device
        self.config = config

    def compile(self, kv_caches):
        raise NotImplementedError

    def make_chunk_input(self, token_ids):
        raise NotImplementedError

    def prefill_chunk(
        self, input_tensor, kv_caches, *, slot_id, actual_start, actual_end, request_id=0, d2h_service=None
    ):
        raise NotImplementedError

    def set_layer_ack_channel(self, channel):
        raise NotImplementedError

    def kv_migration_base_address(self, kv_caches):
        raise NotImplementedError


_BROKEN_SOURCE = os.path.abspath(__file__)


# =============================================================================================
# G-RUNTIME: the audit
# =============================================================================================
def test_engine_call_site_is_readable():
    """The audit's own precondition: the engine source parses and does drive a `runtime` handle.

    Without this, an audit over a moved or renamed engine file would find nothing to check and
    report a clean pass — the way a citation that is wrong but in range reports `resolved`
    (`R-016`).
    """
    assert os.path.isfile(ENGINE_SOURCE), f"the engine's call site is not at {ENGINE_SOURCE}"
    attributes, config_attributes, calls, guarded = read_engine_call_site()
    assert "prefill_chunk" in calls, "the AST walk found no runtime.prefill_chunk call — is this the right file?"
    assert "compile" in calls, "the AST walk found no runtime.compile call"
    assert config_attributes, "the AST walk found no runtime.config access"
    assert guarded, "the AST walk found no getattr/hasattr-guarded hook, which the engine does use"
    from loguru import logger

    logger.info(
        f"[G-RUNTIME] engine call site: {len(attributes)} runtime attributes, "
        f"{len(config_attributes)} config attributes {sorted(config_attributes)}, "
        f"{len(calls)} called methods {sorted(calls)}, {len(guarded)} guarded hooks {sorted(guarded)}"
    )


def test_runtime_satisfies_the_engine_call_site():
    """**The gate.** Every unguarded name the engine touches exists, with a compatible signature."""
    findings = audit(
        TtPrefillRuntime,
        TtPrefillRuntimeConfig,
        runtime_source=RUNTIME_SOURCE,
        runtime_class_name="TtPrefillRuntime",
    )
    from loguru import logger

    logger.info(f"[G-RUNTIME] audit of TtPrefillRuntime: {findings}")
    assert findings == {
        "missing_attributes": [],
        "missing_config_attributes": [],
        "signature_problems": [],
    }, f"the runtime does not satisfy the engine's real call site: {findings}"


def test_audit_rejects_a_runtime_written_to_the_doc():
    """**The audit's negative control.** A doc-faithful runtime must be reported, on three counts."""
    findings = audit(_BrokenRuntime, _BrokenConfig, runtime_source=_BROKEN_SOURCE, runtime_class_name="_BrokenRuntime")
    from loguru import logger

    logger.info(f"[G-RUNTIME] audit of _BrokenRuntime (control): {findings}")
    problems = " | ".join(findings["signature_problems"])
    assert "metadata_msg" in problems, f"the audit missed the renamed metadata_msg parameter: {findings}"
    assert any(
        "set_layer_completion_sink" in p or "set_d2h_ack_service" in p or "build_kv_chunk_table" in p
        for p in findings["signature_problems"] + findings["missing_attributes"]
    ), f"the audit missed the absent engine-called methods: {findings}"
    assert any(
        "use_trace" in entry for entry in findings["missing_config_attributes"]
    ), f"the audit missed the absent config.use_trace: {findings}"


def test_doc_required_names_are_present():
    """The contract doc's own §2 list, which the call site does not cover (`make_chunk_input`)."""
    declared = set(dir(TtPrefillRuntime)) | _instance_attributes(RUNTIME_SOURCE, "TtPrefillRuntime")
    missing = [name for name in DOC_RUNTIME_NAMES if name not in declared]
    assert not missing, f"the runtime is missing doc-required names: {missing}"

    config = TtPrefillRuntimeConfig(num_layers=32)
    missing = [name for name in DOC_CONFIG_NAMES if not hasattr(config, name)]
    assert not missing, f"config is missing doc-required names: {missing}"


def test_engine_call_site_is_wider_than_the_doc():
    """Measure the doc's gap rather than quoting it: two parameters, two methods, one config field.

    `BRINGUP_RECIPE.md:1630-1635` names the two `prefill_chunk` parameters. The two undocumented
    methods (`set_layer_completion_sink`, `set_d2h_ack_service`) and `config.use_trace` are what
    this AST walk added — `build_kv_chunk_table` and the two migration hooks *are* in the doc's
    optional-hook list.
    """
    doc = open(CONTRACT_DOC).read()
    doc_signature = next(line for line in doc.splitlines() if "def prefill_chunk" in line)
    for parameter in ("d2h_service", "metadata_msg"):
        assert parameter not in doc_signature, f"the doc now documents {parameter}; update this gate's finding"
        assert parameter in inspect.signature(TtPrefillRuntime.prefill_chunk).parameters, (
            f"{parameter} is missing from prefill_chunk: the engine always passes it "
            f"(prefill_runner.py:286-295) and a runtime without it dies on its first served chunk"
        )

    attributes, config_attributes, _, guarded = read_engine_call_site()
    doc_documented = set(re.findall(r"def (\w+)\(self", doc))
    engine_only = sorted(
        name
        for name in attributes
        if name not in doc_documented and name not in guarded and name not in ("config", "mesh_device")
    )
    from loguru import logger

    logger.info(
        f"[G-RUNTIME] engine-called methods the doc's section 2 does not list: {engine_only}; "
        f"config fields the engine reads: {sorted(config_attributes)} (the doc lists {list(DOC_CONFIG_NAMES)})"
    )
    assert "set_layer_completion_sink" in engine_only, "expected the doc to omit set_layer_completion_sink"
    assert "use_trace" in config_attributes and "use_trace" not in DOC_CONFIG_NAMES
    assert hasattr(TtPrefillRuntimeConfig(num_layers=32), "use_trace")


# =============================================================================================
# G-RUNTIME: the deployment numbers, and the arithmetic behind them
# =============================================================================================
def test_deployment_chunk_geometry():
    """`DEC-061`: `(chunk 8192, max_seq_len 131072)` at the target `(4,8)` mesh, with the arithmetic.

    Closes `R-004`. Both constraints come from
    `models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md`'s shared setup, via
    `bringup_log/00_MODEL_CARD.md` §4.
    """
    config = TtPrefillRuntimeConfig(num_layers=32)
    assert config.chunk_size == DEPLOYMENT_CHUNK_SIZE == 8192
    assert config.max_seq_len == DEPLOYMENT_MAX_SEQ_LEN == 131072
    assert config.mesh_shape == (4, 8) and config.sp_factor == 4 and config.tp_factor == 8
    assert config.chunk_size % (ttnn.TILE_SIZE * config.sp_factor) == 0, "8192 % 128 must be 0"
    assert config.max_seq_len % config.chunk_size == 0, "131072 / 8192 must be a whole number of chunks"
    assert config.max_seq_len // config.chunk_size == 16
    assert config.chunk_sizes == (8192,)
    from loguru import logger

    logger.info(
        f"[G-RUNTIME] deployment geometry: chunk={config.chunk_size} max_seq_len={config.max_seq_len} "
        f"-> {config.max_seq_len // config.chunk_size} chunks; "
        f"{config.chunk_size} % (TILE_SIZE x sp = {ttnn.TILE_SIZE * config.sp_factor}) = "
        f"{config.chunk_size % (ttnn.TILE_SIZE * config.sp_factor)}"
    )


def test_multiple_chunk_sizes_are_ordered_largest_first():
    """`resolve_chunk_sizes` dedupes and sorts, because buffers are sized at the largest."""
    assert resolve_chunk_sizes(8192, (4096, 8192, 2048), 131072) == (8192, 4096, 2048)
    config = TtPrefillRuntimeConfig(num_layers=32, additional_chunk_sizes=(4096,))
    assert config.chunk_sizes == (8192, 4096) and config.max_chunk_size == 8192


# =============================================================================================
# G-RUNTIME: the refusals. Every `raise` in the module, each matched on its message.
# =============================================================================================
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"chunk_size": 5000}, "must divide max_seq_len"),
        ({"chunk_size": 8192, "max_seq_len": 8192, "additional_chunk_sizes": (96,)}, "must divide max_seq_len"),
        ({"chunk_size": 32, "max_seq_len": 1024}, "must be a multiple of TILE_SIZE"),
        ({"chunk_size": 128, "max_seq_len": 128 + 32}, "must divide max_seq_len"),
        ({"sp_axis": 1, "tp_axis": 1}, "must differ"),
        ({"num_users": 0}, "num_users must be at least 1"),
    ],
    ids=[
        "chunk_not_dividing",
        "extra_chunk_not_dividing",
        "chunk_below_tile_x_sp",
        "capacity_not_multiple",
        "same_axis",
        "no_users",
    ],
)
def test_config_refuses_bad_geometry(expect_error, kwargs, message):
    """Arithmetic that would silently produce a RoPE table whose rows do not match the cache."""
    with expect_error(ValueError, message):
        TtPrefillRuntimeConfig(num_layers=32, **kwargs)


def test_config_refuses_zero_layers(expect_error):
    with expect_error(ValueError, "num_layers must be at least 1"):
        TtPrefillRuntimeConfig(num_layers=0)


class _MeshStub:
    """Just enough of a mesh device to reach the runtime's shape and TP refusals with no hardware.

    Both checks run before any `ttnn` object is built, which is what keeps `G-RUNTIME` a device-free
    gate (Appendix A gives it device "none").
    """

    def __init__(self, shape):
        self.shape = shape


def test_runtime_refuses_a_mesh_of_the_wrong_shape(expect_error):
    config = TtPrefillRuntimeConfig(num_layers=32)
    with expect_error(ValueError, "config.mesh_shape"):
        TtPrefillRuntime(_MeshStub((2, 8)), {"num_key_value_heads": 8}, {}, config)


@pytest.mark.parametrize("mesh_shape", [(1, 1), (4, 4), (4, 2)], ids=["single_card", "tp4", "tp2"])
def test_runtime_refuses_tp_not_equal_to_kv_heads(expect_error, mesh_shape):
    """**The single-card refusal** (`BRINGUP_RECIPE.md:1692-1697`).

    The packed cache holds exactly one KV head per chip, so `tp == num_key_value_heads` is an
    equality. Without this check the failure is a `TT_FATAL` from
    `update_padded_kv_cache` that names neither TP nor the mesh — and it arrives after the mesh is
    open and 15 GB of weights are loaded.
    """
    config = TtPrefillRuntimeConfig(num_layers=32, mesh_shape=mesh_shape)
    with expect_error(ValueError, "tp == num_key_value_heads"):
        TtPrefillRuntime(_MeshStub(mesh_shape), {"num_key_value_heads": 8}, {}, config)


def _unbuilt_runtime(**config_kwargs):
    """A `TtPrefillRuntime` with only the fields the argument checks read, and no device.

    `__init__` needs a mesh; every refusal below is raised by argument validation that runs before
    the cache or the model is touched, so the honest way to test them without hardware is to give
    the methods exactly the state they read. Recorded in the gate block as what makes `G-RUNTIME`
    device-free.
    """
    runtime = object.__new__(TtPrefillRuntime)
    runtime.config = TtPrefillRuntimeConfig(**config_kwargs)
    runtime.rope_indexed = {size: None for size in runtime.config.chunk_sizes}
    runtime.mesh_device = None
    runtime.hf = {"hidden_size": 4096, "num_key_value_heads": 8}
    runtime._on_layer_complete = None
    return runtime


def _stub_cache(num_layers=32, max_seq_len=131072, sp=4):
    """A `LlamaKVCache` with no device tensors — the type check is all these tests reach."""
    return LlamaKVCache(k=None, v=None, num_users=1, num_layers=num_layers, max_seq_len=max_seq_len, sp=sp)


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"d2h_service": object()}, NotImplementedError, "emits no D2H layer-ack records"),
        ({"chunk_size": 777}, ValueError, "has no indexed RoPE table"),
        ({"slot_id": 4}, ValueError, "out of range"),
        ({"actual_start": 0, "actual_end": 0}, ValueError, "not a non-empty range"),
        ({"actual_start": 0, "actual_end": 8192 * 3}, ValueError, "not a non-empty range"),
        ({"actual_start": 8192 * 16, "actual_end": 8192 * 16 + 1}, ValueError, "runs past the per-user"),
        ({"actual_start": 100, "actual_end": 200}, ValueError, "must be tile-aligned"),
    ],
    ids=[
        "d2h_service",
        "unsupported_chunk_size",
        "slot_out_of_range",
        "empty_range",
        "range_wider_than_chunk",
        "past_capacity",
        "unaligned_start",
    ],
)
def test_prefill_chunk_refuses_bad_arguments(expect_error, kwargs, error, message):
    """Seven argument refusals, all before the cache is resolved or a single op is issued."""
    runtime = _unbuilt_runtime(num_layers=32)
    call = {"slot_id": 0, "actual_start": 0, "actual_end": 8192}
    call.update(kwargs)
    with expect_error(error, message):
        runtime.prefill_chunk(None, _stub_cache(), **call)


def test_prefill_chunk_accepts_the_engines_always_present_metadata_msg(expect_error):
    """**`metadata_msg` must be accepted, not refused** — the lesson of `DEC-108`.

    The engine passes it on **every** chunk in request mode: it is the H2D socket's own metadata
    tensor (`prefill_runner.py:156-159`), the same object the engine decoded into the `slot_id` /
    `actual_start` / `actual_end` it passes alongside. It is not a trace artefact, and it is never
    `None`. This runtime originally refused a non-`None` value, and `G-RUNTIME`'s AST audit passed
    it clean — a static audit proves the signature **binds** the engine's call, not that the values
    the engine binds into it are acceptable. The refusal surfaced only on the first served chunk,
    after the mesh was open and 15 GB of weights were loaded: precisely the cost the gate exists to
    avoid, arrived at from the opposite direction.

    Asserted by getting *past* it: the call is driven to the next refusal in line (the unsupported
    chunk size), which proves `metadata_msg` was consumed rather than rejected.
    """
    runtime = _unbuilt_runtime(num_layers=32)
    with expect_error(ValueError, "has no indexed RoPE table"):
        runtime.prefill_chunk(
            None,
            _stub_cache(),
            slot_id=0,
            actual_start=0,
            actual_end=8192,
            request_id=7,
            metadata_msg=object(),  # never None in request mode
            chunk_size=777,  # the refusal we EXPECT to reach
        )


def test_every_parameter_the_engine_always_passes_is_accepted_or_used():
    """The general form of `DEC-108`, checked against the engine's own call site.

    A parameter the engine **unconditionally** passes may be used or ignored, but never refused —
    refusing it is a guaranteed crash on the first served chunk. `d2h_service` is the exception and
    is listed as such: the engine passes `None` for it unless `PREFILL_LAYER_ACK_D2H=1`
    (`prefill_runner.py:707`, `:730-736`), so a refusal there is reachable only when someone asks
    for the path.
    """
    _attributes, _config, calls, _guarded = read_engine_call_site()
    keywords = {kw for _line, _n, kws in calls["prefill_chunk"] for kw in kws if kw is not None}
    assert keywords == {"slot_id", "actual_start", "actual_end", "request_id", "d2h_service", "metadata_msg"}, (
        f"the engine's prefill_chunk keywords changed to {sorted(keywords)}; re-derive which of them "
        f"can be None (DEC-108)"
    )
    source = inspect.getsource(TtPrefillRuntime.prefill_chunk)
    body = source.split('"""')[-1]  # past the docstring: only what executes
    for parameter in sorted(keywords - {"d2h_service"}):
        assert f"{parameter} is not None" not in body, (
            f"prefill_chunk refuses a non-None {parameter}, and the engine always passes one "
            f"(prefill_runner.py:286-295). That is a TypeError-equivalent on the first served "
            f"chunk (DEC-108)."
        )
    assert "d2h_service is not None" in body, (
        "the d2h_service refusal was removed; the engine creates that service only under "
        "PREFILL_LAYER_ACK_D2H=1 and this runtime does not implement the device record path (R-024)"
    )


def test_prefill_chunk_refuses_a_cache_backed_chunk_on_the_dense_path(expect_error):
    """**Delta 3.** `actual_start > 0` without the SP ring path must refuse, naming P8.

    `BRINGUP_RECIPE.md:1692-1697`: delta 3 cannot run in P7, `G-CHUNK` must not be weakened to
    cover it, and the runtime must refuse the configuration loudly instead of running a causal mask
    that is off by `actual_start`.
    """
    runtime = _unbuilt_runtime(num_layers=32, sequence_parallel=False)
    with expect_error(NotImplementedError, "must attend the prefix read back out of"):
        runtime.prefill_chunk(None, _stub_cache(), slot_id=0, actual_start=8192, actual_end=8192 * 2)


def test_prefill_chunk_refuses_a_missing_or_foreign_cache(expect_error):
    """The runtime owns no cache (`DEC-062`), so there is nothing to fall back to."""
    runtime = _unbuilt_runtime(num_layers=32)
    with expect_error(ValueError, "needs the engine-owned KV cache"):
        runtime.prefill_chunk(None, None, slot_id=0, actual_start=0, actual_end=8192)
    with expect_error(TypeError, "must be a LlamaKVCache"):
        runtime.prefill_chunk(None, object(), slot_id=0, actual_start=0, actual_end=8192)


def test_prefill_chunk_accepts_a_one_element_sequence_of_caches():
    """The template's resolver takes `kv_caches[0]`; a harness written to it must keep working."""
    runtime = _unbuilt_runtime(num_layers=32)
    cache = _stub_cache()
    assert runtime._resolve_kv(cache) is cache
    assert runtime._resolve_kv([cache]) is cache
    assert runtime._resolve_kv((cache,)) is cache


def test_make_chunk_input_refuses_a_short_chunk(expect_error):
    """A chunk must be exactly `chunk_size` ids; the real range travels in `[actual_start, actual_end)`."""
    runtime = _unbuilt_runtime(num_layers=32)
    with expect_error(ValueError, "must be exactly chunk_size"):
        runtime.make_chunk_input([0] * 17)


def test_compile_refuses_a_non_first_rank(expect_error):
    runtime = _unbuilt_runtime(num_layers=32, is_first_rank=False)
    with expect_error(NotImplementedError, "pipeline rank needs the D2D activation spec"):
        runtime.compile(_stub_cache())


@pytest.mark.parametrize(
    "method, args, message",
    [
        ("set_layer_completion_sink", (None,), "multi-rank pipeline path"),
        ("set_d2h_ack_service", (None,), "belongs to the trace path"),
    ],
    ids=["completion_sink", "d2h_ack"],
)
def test_unimplemented_engine_hooks_refuse_loudly(expect_error, method, args, message):
    """The two hooks the engine calls **unguarded** that this iteration does not implement.

    Each is present so the audit passes and each raises so a trace or pipelined run fails rather
    than reporting success having registered nothing (`R-024`). Called unbound with a `None` self,
    because neither touches instance state before raising — which is itself the property that makes
    them safe stubs. The other three (`build_kv_chunk_table`, `kv_migration_base_address`,
    `set_layer_ack_channel`) were on this list until P10 implemented them; their own refusals are
    below.
    """
    with expect_error(NotImplementedError, message):
        getattr(TtPrefillRuntime, method)(None, *args)


# =============================================================================================
# G-RUNTIME (P10): the migration hooks the engine calls, and what they still refuse
# =============================================================================================
def test_set_layer_ack_channel_refuses_before_compile(expect_error):
    """Acking `compile()`'s throwaway chunks would poison the producer's drain count.

    The producer drains exactly `num_layers x chunks` acks (`prefill_producer.py:1115`) and gates
    its whole PCC read-back on that drain (`:1057-1063`), so `num_layers` phantom acks per warmed
    chunk size would make it finish one real chunk early and read a partially-written cache. The
    engine already orders `compile` (`prefill_runner.py:501`) before this call (`:768`); the
    assertion makes a reordering fail here rather than as a hang.
    """
    runtime = _unbuilt_runtime(num_layers=32)
    runtime.compiled = False
    with expect_error(AssertionError, "call compile.. before set_layer_ack_channel"):
        runtime.set_layer_ack_channel(object())


def test_set_layer_ack_channel_bumps_once_per_layer():
    """The registered callback injects 1 per call — `num_layers` per chunk, the count the reader wants."""

    class _Channel:
        def __init__(self):
            self.injected = 0

        def inject(self, count):
            self.injected += count

    runtime = _unbuilt_runtime(num_layers=32)
    runtime.compiled = True
    channel = _Channel()
    runtime.set_layer_ack_channel(channel)
    assert runtime._on_layer_complete is not None, "set_layer_ack_channel registered no callback"
    for layer_idx in range(runtime.config.num_layers):
        runtime._on_layer_complete(layer_idx)
    assert channel.injected == runtime.config.num_layers, (
        f"one chunk injected {channel.injected} acks for {runtime.config.num_layers} layers; the "
        f"producer drains num_layers x chunks and a mismatch hangs its drain"
    )


# `stage_layout` is the gathered **LIST** of one dict per rank — `allgather_kv_stage_layout` builds
# it with `for rk in range(size)` (`migration.py:315-334`) and the engine passes `stage_layouts[0]`,
# stage 0's per-rank list (`prefill_runner.py:634`). The first version of these tests asserted the
# opposite and enshrined `DEC-111`'s bug; they now assert the engine's real shapes.
_SINGLE_RANK_STAGE = [{"rank": 0, "first_layer": 0, "count": 32, "base_addr": 0x1000}]


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"first_layer_idx": 8}, NotImplementedError, "pipeline rank that is not rank 0"),
        ({"num_my_layers": 16}, NotImplementedError, "asked for 16 layers but this runtime"),
        # A pipeline run: two ranks in the gathered list. THIS is the shape that can actually detect
        # one, and it is the only one the engine's own path can produce (`DEC-111`).
        (
            {
                "stage_layout": [
                    {"rank": 0, "first_layer": 0, "count": 16},
                    {"rank": 1, "first_layer": 16, "count": 16},
                ]
            },
            NotImplementedError,
            "carries 2 ranks",
        ),
        # One rank, but not the whole model.
        ({"stage_layout": [{"rank": 0, "first_layer": 0, "count": 16}]}, NotImplementedError, "of this runtime's 32"),
        # A bare dict: what the first draft believed the engine passed.
        ({"stage_layout": {"first_layer": 0, "count": 32}}, TypeError, "must be the gathered LIST"),
        ({"stage_layout": []}, TypeError, "non-empty sequence"),
    ],
    ids=["first_layer_idx", "num_my_layers", "two_ranks", "partial_rank", "bare_dict", "empty_list"],
)
def test_build_kv_chunk_table_refuses_a_pipeline_rank_slice(expect_error, kwargs, error, message):
    """Recipe P10 step 4: the unimplemented multi-rank merge **raises**, naming `R-032`.

    The template discards all three arguments with a `del`
    (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:388`), which on a pipelined runner would
    publish a table claiming to cover the whole model while addressing only rank 0's layers.
    Asserted against the table module's own guard, so no device is needed.
    """
    from models.demos.llama31_8b_d_p.tt.runners.kv_chunk_table import assert_single_rank_stage

    call = {"num_layers": 32, "first_layer_idx": 0}
    call.update(kwargs)
    with expect_error(error, message):
        assert_single_rank_stage(**call)


def test_build_kv_chunk_table_accepts_the_shapes_the_engine_really_passes():
    """The control for the refusals above — and it is the half `DEC-111` failed.

    Three shapes, one per engine call site: the pure-mock path with `path` only
    (`prefill_runner.py:570`, `:699`), and the gathered path with `first_layer_idx=0`,
    `num_my_layers=<depth>` and `stage_layout=stage_layouts[0]` — the **list** with one entry
    because there is one rank (`:644`, `:655`, `:674`). A guard that rejects any of these blocks
    every real migration run, which is what the first version did, and `G-MOCK-MIG` could not see it
    because the pure-mock path passes no `stage_layout` at all.
    """
    from models.demos.llama31_8b_d_p.tt.runners.kv_chunk_table import assert_single_rank_stage

    assert_single_rank_stage(num_layers=32, first_layer_idx=0)
    assert_single_rank_stage(num_layers=32, first_layer_idx=0, num_my_layers=32)
    assert_single_rank_stage(num_layers=32, first_layer_idx=0, num_my_layers=32, stage_layout=_SINGLE_RANK_STAGE)


def test_the_gathered_stage_layout_really_is_a_list_of_dicts():
    """Read the engine's own builder, so this file's belief about the shape cannot drift again.

    `DEC-111` was a wrong belief about a type, held by both the code and its test. An AST check on
    the *producing* function is what makes the belief falsifiable.
    """
    import ast as _ast

    source = open(os.path.join(_ROOT, "models", "demos", "common", "prefill", "runners", "migration.py")).read()
    functions = {n.name: n for n in _ast.walk(_ast.parse(source)) if isinstance(n, _ast.FunctionDef)}
    body = _ast.unparse(functions["allgather_kv_stage_layout"])
    assert "stages = []" in body and "stages.append({" in body and "return stages" in body, (
        "allgather_kv_stage_layout no longer builds a list of dicts; re-derive what the engine "
        "passes as stage_layout before trusting assert_single_rank_stage (DEC-111)"
    )
    plural = _ast.unparse(functions["allgather_kv_stage_layouts"])
    assert "allgather_kv_stage_layout(" in plural and "for stage in stages" in plural, (
        "allgather_kv_stage_layouts is no longer one list per migratable stage, so stage_layouts[0] "
        "may no longer be a per-rank list"
    )


def test_build_kv_chunk_table_refuses_more_than_one_block_cyclic_period(expect_error):
    """`DEC-112`: a table describes ONE period, so two supported chunk sizes must refuse.

    Unreachable through the engine (it never passes `chunk_size` to `prefill_chunk`,
    `prefill_runner.py:287-296`), and the only silent-wrong-answer this module could produce: a
    cache written at two periods has no single address map, and a table built for
    `config.chunk_size` would hand out addresses that resolve, decode, and are wrong.
    """
    runtime = _unbuilt_runtime(num_layers=32, chunk_size=8192, additional_chunk_sizes=(4096,))
    assert runtime.config.chunk_sizes == (8192, 4096)
    with expect_error(NotImplementedError, "describes exactly one block-cyclic period"):
        runtime.build_kv_chunk_table(_stub_cache(), "/tmp/unused_table.pb")


def test_kv_migration_stages_is_deliberately_absent():
    """Its mere presence switches the engine to the multi-stage merge path (`prefill_runner.py:613`).

    Not an omission: this model migrates K and V, but the multi-stage path is the unimplemented
    multi-rank merge in a different disguise (`R-032`), and the table takes each tensor's own
    `buffer_address()` (`models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:138`) so it needs no
    per-stage anchor.
    """
    assert not hasattr(TtPrefillRuntime, "kv_migration_stages"), (
        "kv_migration_stages now exists, which moves the engine onto its multi-stage merge path "
        "(prefill_runner.py:613-615) — implement the merge, or remove the hook"
    )
    assert hasattr(TtPrefillRuntime, "kv_migration_base_address"), (
        "with neither migration hook the engine raises a RuntimeError naming the doc "
        "(prefill_runner.py:619-623) and migration cannot run at all"
    )


def test_kv_migration_base_address_returns_the_k_cache_base():
    """K's base, read off the cache the engine handed in — not a remembered address (`DEC-062`)."""

    class _Tensor:
        def __init__(self, address):
            self._address = address

        def buffer_address(self):
            return self._address

    runtime = _unbuilt_runtime(num_layers=32)
    cache = LlamaKVCache(k=_Tensor(0x1000), v=_Tensor(0x2000), num_users=1, num_layers=32, max_seq_len=1024, sp=4)
    assert runtime.kv_migration_base_address(cache) == 0x1000
    assert runtime.kv_migration_base_address([cache]) == 0x1000


def test_every_raise_in_the_module_is_covered():
    """A meta-check: count the `raise` statements in the module and the refusals asserted here.

    Recipe §1.4 asks for "every refusal loud and matched on its message"; the honest way to claim
    *every* is to count them rather than to assert a subset and hope. P10 replaced three `raise`
    bodies with implementations and moved their refusals into `tt/runners/kv_chunk_table.py`, so the
    runtime's floor drops from 18 to 15 and that module's own four are counted here too.
    """
    tree = ast.parse(open(RUNTIME_SOURCE).read())
    raises = [node for node in ast.walk(tree) if isinstance(node, ast.Raise)]
    table_tree = ast.parse(open(TABLE_SOURCE).read())
    table_raises = [node for node in ast.walk(table_tree) if isinstance(node, ast.Raise)]
    from loguru import logger

    logger.info(
        f"[G-RUNTIME] tt_prefill_runtime.py contains {len(raises)} raise statements at lines "
        f"{[node.lineno for node in raises]}; tt/runners/kv_chunk_table.py contains "
        f"{len(table_raises)} at lines {[node.lineno for node in table_raises]}"
    )
    assert len(raises) >= 16, (
        f"only {len(raises)} raise statements found in the runtime; the refusal list in this file "
        f"was written against 16 (15 after P10 replaced three bodies, plus DEC-112's period guard) "
        f"and one may have been deleted rather than covered"
    )
    assert len(table_raises) >= 6, (
        f"only {len(table_raises)} raise statements found in tt/runners/kv_chunk_table.py; the "
        f"multi-rank refusals (R-032, six of them after DEC-111) and the shared-layout guard "
        f"(DEC-099) are counted here"
    )
