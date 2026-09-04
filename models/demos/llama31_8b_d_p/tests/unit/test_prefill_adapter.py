# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`tt/runners/adapters/llama.py` against the engine's adapter contract. Gate: `G-ADAPTER`.

**No device, no weights** (Appendix A gives `G-ADAPTER` device "—"): this gate is the checklist at
the end of `models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md:246-255`, item by item, with
evidence — plus the four things `BRINGUP_RECIPE.md:1935-1938` adds to it: zero abstract methods
left, `PREFILL_MODEL=llama31_8b_d_p` resolving through the registry, the registry-fed pytest
`variant` fixture picking it up, every `model_config` constant equal to `config.json`, and the
**measured** import cost.

**Why the import cost is a gate and not a style note.** Two importers pull every registered
adapter without wanting a device stack: the H2D producer resolves one at module scope
(`models/demos/common/prefill/runners/prefill_producer.py:89`) and the DeepSeek test conftest
builds an instance of **every** entry in `ADAPTER_PATHS` eagerly
(`models/demos/deepseek_v3_d_p/tests/conftest.py:33`), so a heavy import here is charged to another
package's whole test session. The measurement runs in a **subprocess** — an in-process check would
pass trivially, because pytest has already imported `torch` and `ttnn` — and it gets a negative
control: the same probe with this package's `tt/model_config.py` added must report heavy modules.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_prefill_adapter.py -x -q
"""

import inspect
import json
import os
import subprocess
import sys
import textwrap

import pytest

import ttnn
from models.demos.common.prefill.adapter import ADAPTER_PATHS, KvCaches, PrefillModelAdapter, PrefillRunParams
from models.demos.llama31_8b_d_p.tests.test_factory import GALAXY_MESH_SHAPE
from models.demos.llama31_8b_d_p.tt.attention.kv_cache import LlamaKVCache
from models.demos.llama31_8b_d_p.tt.config import derive_head_dim
from models.demos.llama31_8b_d_p.tt.model_config import BUNDLED_CONFIG_PATH, ModelArgs
from models.demos.llama31_8b_d_p.tt.runners.adapters.llama import Llama31_8BConfig, LlamaHfConfig, LlamaPrefillAdapter
from models.demos.llama31_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

MODEL_NAME = "llama31_8b_d_p"
ADAPTER_MODULE = "models.demos.llama31_8b_d_p.tt.runners.adapters.llama"
MANIFEST_PATH = "models/demos/llama31_8b_d_p/tt/runners/manifests/llama31_8b_d_p.json"

# The modules an adapter import must NOT pull. `numpy` is not on the list: `loguru` and the engine's
# own `adapter.py` are pure-python, and nothing in this package's import chain reaches numpy, but
# numpy is cheap and its presence would not be evidence of a device stack.
HEAVY_MODULES = ("torch", "ttnn", "transformers", "safetensors")

# Under 1 s in a cold subprocess. This is a ceiling on "did a device stack get imported", not a perf
# target: importing `ttnn` alone costs seconds and opens the cluster, so anything under a second is
# proof the heavy chain was not walked. Measured value is logged, not asserted tightly (`DEC-101`).
IMPORT_BUDGET_S = 1.0


def _executable_code(obj) -> str:
    """`obj`'s source with every docstring and comment removed, via `ast.unparse`.

    The prose in this package cites the traps it avoids by name — `AutoConfig`, `get_num_devices` —
    so a plain substring search over `inspect.getsource` matches the *warning* as well as the
    offence. Round-tripping through `ast` keeps only what executes. (First written as a raw
    substring check, which failed on its own docstrings: `DEC-102`.)
    """
    import ast

    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            if isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


@pytest.fixture
def adapter():
    return LlamaPrefillAdapter()


# =============================================================================================
# Checklist item 1 — every abstract method implemented; identity and defaults set
# =============================================================================================
def test_no_abstract_methods_are_left():
    """`ADDING_A_PREFILL_MODEL.md:248`. An unimplemented one is a `TypeError` at instantiation."""
    assert (
        LlamaPrefillAdapter.__abstractmethods__ == frozenset()
    ), f"abstract methods still unimplemented: {sorted(LlamaPrefillAdapter.__abstractmethods__)}"
    declared = {
        name for name, member in vars(PrefillModelAdapter).items() if getattr(member, "__isabstractmethod__", False)
    }
    assert declared == {"load_hf_config", "weight_cache_path", "allocate_kv_cache", "build_runtime"}, (
        f"the engine's abstract set changed to {sorted(declared)}; this gate was written against "
        f"the four in models/demos/common/prefill/adapter.py:142-188"
    )
    for name in sorted(declared):
        own = vars(LlamaPrefillAdapter).get(name)
        assert own is not None, f"{name} is inherited, not implemented, on LlamaPrefillAdapter"


def test_identity_and_default_paths_are_set(adapter):
    """`ADDING_A_PREFILL_MODEL.md:248-249` (`name`, `model_config`, the default paths)."""
    assert adapter.name == MODEL_NAME
    assert adapter.model_config is Llama31_8BConfig
    # `hf_model_default` must be a real directory holding a real config.json: the engine prints it
    # (`prefill_runner.py:375`) and `load_hf_config` reads it.
    assert os.path.isdir(adapter.hf_model_default), f"hf_model_default {adapter.hf_model_default} is not a directory"
    assert os.path.isfile(os.path.join(adapter.hf_model_default, "config.json"))
    # Both empty by design: the cache root comes from PREFILL_TTNN_CACHE / TT_CACHE_PATH and the
    # golden trace from PREFILL_TRACE_DIR, neither of which has a defensible in-repo default
    # (`DEC-066`; `models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:48-49` does the same).
    assert adapter.ttnn_cache_default == ""
    assert adapter.prefill_trace_default == ""
    assert adapter.supports_dflash is False
    assert adapter.l1_small_size == 0
    assert adapter.pipeline_activation_emb_tp_sharded is True


def test_model_config_constants_equal_config_json():
    """`BRINGUP_RECIPE.md:1938`: every `model_config` constant equals `config.json`.

    The class exists because the engine (`runner_utils.py:41`) and the producer's read-back
    (`prefill_producer.py:550`) both need dimensions with no device and no `ModelArgs` — so it is a
    second copy of the numbers by construction, and this is what stops it becoming a second
    *answer*.
    """
    dims = ModelArgs.load_bundled_config()
    for attribute, key in Llama31_8BConfig.CONFIG_JSON_KEYS.items():
        assert getattr(Llama31_8BConfig, attribute) == dims[key], (
            f"Llama31_8BConfig.{attribute} = {getattr(Llama31_8BConfig, attribute)} but "
            f"config.json:{key} = {dims[key]}"
        )
    # `HEAD_DIM` is the one derived entry: config.json has no head_dim key (`DEC-020`).
    assert "head_dim" not in dims, "config.json now has a head_dim key; read it instead of deriving"
    assert Llama31_8BConfig.HEAD_DIM == derive_head_dim(dims) == 128
    # The engine turns this into the fabric router's max packet payload (`runner_utils.py:41`).
    assert Llama31_8BConfig.FABRIC_PAYLOAD_SIZE == dims["hidden_size"]
    # The producer's packed-GQA reader indexes these two off `ADAPTER.model_config`
    # (`prefill_producer.py:550`), and defaults ROTARY_DIM to HEAD_DIM (`:552`), which is right for
    # Llama: the whole head is rotated.
    assert Llama31_8BConfig.NUM_KEY_VALUE_HEADS == dims["num_key_value_heads"]
    assert getattr(Llama31_8BConfig, "ROTARY_DIM", Llama31_8BConfig.HEAD_DIM) == Llama31_8BConfig.HEAD_DIM


def test_the_tp_equality_the_cache_forces_holds_at_the_deployment_mesh():
    """`TP == num_key_value_heads == 8` — an equality, not a bound (`bringup_log/00_MODEL_CARD.md` §4.1).

    Stated here as well as in the runtime because the *table* also depends on it: it maps head `h`
    to TP column `h` (`models/demos/gpt_oss_d_p/tt/runners/kv_chunk_table.py:96-99`).
    """
    assert GALAXY_MESH_SHAPE[1] == Llama31_8BConfig.NUM_KEY_VALUE_HEADS == 8


# =============================================================================================
# Checklist item 2 — build_runtime returns a section-2 runtime
# =============================================================================================
def test_build_runtime_constructs_the_audited_runtime():
    """`ADDING_A_PREFILL_MODEL.md:250-251`. The runtime itself is audited by `G-RUNTIME`.

    What this adds is that the adapter builds **that** class — so `G-RUNTIME`'s audit is about the
    object the engine will actually be handed — and that the cache is passed in rather than stored
    (`owns_kv_cache` has no equivalent here: `DEC-062`).
    """
    source = _executable_code(LlamaPrefillAdapter.build_runtime)
    assert "TtPrefillRuntime(" in source, "build_runtime does not construct TtPrefillRuntime"
    for name in ("mesh_device", "config", "compile", "make_chunk_input", "prefill_chunk"):
        assert hasattr(TtPrefillRuntime, name) or name in inspect.getsource(
            TtPrefillRuntime.__init__
        ), f"the runtime build_runtime returns is missing the doc-required name {name}"
    assert not hasattr(TtPrefillRuntime, "owns_kv_cache"), "the runtime must not own the cache (DEC-062)"


@pytest.mark.parametrize(
    "params_kwargs, message",
    [
        ({"use_trace": True}, "PREFILL_USE_TRACE=1 needs capture_trace"),
        ({"dflash_enabled": True}, "PREFILL_DFLASH=1 needs a DFlash drafter"),
    ],
    ids=["use_trace", "dflash"],
)
def test_build_runtime_refuses_unimplemented_engine_features(expect_error, adapter, params_kwargs, message):
    """Both refusals fire **before** the mesh or the weights are touched, which is the point.

    `PREFILL_USE_TRACE=1` would otherwise reach `set_d2h_ack_service` (`prefill_runner.py:746`)
    after the mesh is open and 15 GB of weights are loaded — the same expensive-discovery shape as
    recipe P10 warning 1. `mesh_device=None` is safe precisely because nothing is touched first.
    """
    params = _deployment_params(**params_kwargs)
    with expect_error(NotImplementedError, message):
        adapter.build_runtime(mesh_device=None, hf_config=None, params=params)


def test_build_runtime_refuses_a_missing_checkpoint(expect_error, adapter, monkeypatch):
    """There is no cache-only build path here (`DEC-098`), so an unset `HF_MODEL` is refused early."""
    monkeypatch.delenv("HF_MODEL", raising=False)
    with expect_error(ValueError, "HF_MODEL is unset"):
        adapter.build_runtime(mesh_device=None, hf_config=None, params=_deployment_params())


def _deployment_params(**overrides) -> PrefillRunParams:
    """The `PrefillRunParams` the engine builds for a single-rank deployment run (`prefill_runner.py:479`)."""
    kwargs = dict(
        mesh_shape=GALAXY_MESH_SHAPE,
        num_layers=32,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=131072,
        chunk_size=8192,
        num_users=1,
        capacity_factor=8,
        num_links=2,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=False,
        weight_cache_path=None,
    )
    kwargs.update(overrides)
    return PrefillRunParams(**kwargs)


# =============================================================================================
# Checklist item 3 — no heavy imports at module load, MEASURED
# =============================================================================================
def _probe_import(*modules) -> tuple:
    """Import `modules` in a cold subprocess; return `(seconds, [heavy modules present])`."""
    script = (
        "import sys, time, json\n"
        "t0 = time.perf_counter()\n" + "".join(f"import {module}\n" for module in modules) + "elapsed = "
        "time.perf_counter() - t0\n"
        f"heavy = [name for name in {HEAVY_MODULES!r} if name in sys.modules]\n"
        'print(json.dumps({"elapsed": elapsed, "heavy": heavy}))\n'
    )
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 4))
    env = dict(os.environ, PYTHONPATH=root)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, cwd=root, timeout=300)
    assert out.returncode == 0, f"the import probe failed:\n{out.stdout}\n{out.stderr}"
    result = json.loads(out.stdout.strip().splitlines()[-1])
    return result["elapsed"], result["heavy"]


def test_adapter_import_is_cheap_and_pulls_no_device_stack():
    """`ADDING_A_PREFILL_MODEL.md:252` + `BRINGUP_RECIPE.md:1938` ("adapter import is measured")."""
    from loguru import logger

    elapsed, heavy = _probe_import(ADAPTER_MODULE)
    logger.info(f"[G-ADAPTER] importing {ADAPTER_MODULE} took {elapsed * 1000:.1f} ms; heavy modules: {heavy}")
    assert heavy == [], (
        f"importing the adapter pulled {heavy}. The H2D producer (prefill_producer.py:89) and the "
        f"DeepSeek conftest (models/demos/deepseek_v3_d_p/tests/conftest.py:33) both import every "
        f"registered adapter; move the import inside the method that needs it."
    )
    assert elapsed < IMPORT_BUDGET_S, f"the adapter import took {elapsed:.2f}s (budget {IMPORT_BUDGET_S}s)"


def test_the_import_probe_can_actually_fail():
    """**Negative control.** The same probe, with this package's config reader added, must trip.

    Without this, "no heavy module was found" is indistinguishable from "the probe looks for the
    wrong thing" — the failure mode `R-016` describes for citations.
    """
    from loguru import logger

    elapsed, heavy = _probe_import(ADAPTER_MODULE, "models.demos.llama31_8b_d_p.tt.model_config")
    logger.info(f"[G-ADAPTER] control: adapter + tt/model_config.py took {elapsed * 1000:.1f} ms; heavy: {heavy}")
    assert (
        "torch" in heavy and "ttnn" in heavy
    ), f"the control imported tt/model_config.py and still reported {heavy}; the probe is blind"


def test_the_adapter_module_imports_nothing_heavy_at_module_scope():
    """The static half of the same claim: the module's own top-level imports, read with `ast`.

    Complements the subprocess measurement — a module could import something heavy inside a
    `try` at module scope and swallow the failure, which the subprocess probe would catch only on a
    machine where the import succeeds.
    """
    import ast

    source = inspect.getsource(sys.modules[ADAPTER_MODULE])
    tree = ast.parse(source)
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.append(node.module)
    offenders = [name for name in top_level for heavy in HEAVY_MODULES if name.split(".")[0] == heavy]
    assert not offenders, f"module-scope heavy imports: {offenders} (imports found: {top_level})"


# =============================================================================================
# Checklist item 4 — registered in ADAPTER_PATHS, and reachable the three ways
# =============================================================================================
def test_registered_in_adapter_paths():
    """`ADDING_A_PREFILL_MODEL.md:253`. One line, and the import stays lazy."""
    assert ADAPTER_PATHS[MODEL_NAME] == f"{ADAPTER_MODULE}:LlamaPrefillAdapter"


def test_prefill_model_resolves_through_the_registry():
    """`BRINGUP_RECIPE.md:1938`: `PREFILL_MODEL=llama31_8b_d_p` resolves, and memoizes."""
    from models.demos.common.prefill.adapter import get_adapter

    resolved = get_adapter(MODEL_NAME)
    assert isinstance(resolved, LlamaPrefillAdapter)
    assert get_adapter(MODEL_NAME) is resolved, "get_adapter is documented as memoizing (adapter.py:295)"


def test_the_registry_fed_variant_fixture_picks_it_up():
    """`BRINGUP_RECIPE.md:1935` (`models/demos/deepseek_v3_d_p/tests/conftest.py:365`).

    That conftest builds `TEST_VARIANTS = {name: get_adapter(name) for name in ADAPTER_PATHS}` at
    **import** time (`:33`), so registering this model puts its adapter into another package's test
    collection — which is the concrete reason the import budget above is a gate.
    """
    from models.demos.deepseek_v3_d_p.tests.conftest import TEST_VARIANTS

    assert MODEL_NAME in TEST_VARIANTS
    assert isinstance(TEST_VARIANTS[MODEL_NAME], LlamaPrefillAdapter)


def test_the_model_manifest_is_valid_and_agrees_with_the_adapter():
    """The manifest is applied with `setdefault` by both processes (`prefill_runner.py:43`,
    `prefill_producer.py:39`), so a wrong value here is a silently wrong default on every rank.

    Every pinned value is checked against the thing that owns it, and the keys a *caller* owns —
    the prompt, the chunk count, the user count — must be absent (recipe P10 step 2).
    """
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    assert set(manifest) == {"env"}, f"the runner reads only the env block (prefill_runner.py:43); got {set(manifest)}"
    env = manifest["env"]
    assert env["PREFILL_MODEL"] == MODEL_NAME
    assert int(env["PREFILL_NUM_LAYERS"]) == Llama31_8BConfig.NUM_LAYERS
    assert (int(env["PREFILL_SP"]), int(env["PREFILL_TP"])) == GALAXY_MESH_SHAPE
    assert int(env["PREFILL_TP"]) == Llama31_8BConfig.NUM_KEY_VALUE_HEADS
    # The deployment geometry, and its arithmetic (`DEC-061`, `G-RUNTIME`).
    chunk, capacity = int(env["PREFILL_CHUNK_SIZE"]), int(env["PREFILL_MAX_SEQ_LEN"])
    assert capacity % chunk == 0 and chunk % (ttnn.TILE_SIZE * int(env["PREFILL_SP"])) == 0
    assert capacity > chunk, "max_seq_len == chunk_size selects the SP bootstrap core, not the ring (R-039)"
    assert capacity == Llama31_8BConfig.MAX_POSITION_EMBEDDINGS
    # `1d` is what this galaxy has; `1d_ring` cannot be initialised at all (`R-030`, `DEC-097`).
    assert env["PREFILL_FABRIC_MODE"] == "1d"
    assert env["PREFILL_USE_TRACE"] == "0" and env["PREFILL_DFLASH"] == "0"
    for caller_owned in (
        "PREFILL_NUM_USERS",
        "PREFILL_PRODUCER_CHUNKS",
        "PREFILL_TRACE_DIR",
        "PREFILL_H2D_SERVICE_ID",
        "PREFILL_MOCK_MIGRATION",
        "PREFILL_ENABLE_MIGRATION",
    ):
        assert (
            caller_owned not in env
        ), f"{caller_owned} belongs to the caller, not the model manifest (BRINGUP_RECIPE.md:1897)"


def test_the_manifest_env_block_is_readable_by_the_engines_own_applier():
    """Applied through the engine's own parser, not a re-implementation of it.

    `_apply_manifest_env` reads the file with `yaml.safe_load` (`prefill_producer.py:26-29`), which
    accepts JSON — so a `.json` manifest is legal, and this asserts it rather than assuming it.
    """
    import yaml

    with open(MANIFEST_PATH) as f:
        assert yaml.safe_load(f)["env"]["PREFILL_MODEL"] == MODEL_NAME


# =============================================================================================
# Checklist item 5 — weight cache populated, golden trace staged
# =============================================================================================
def test_weight_cache_path_mirrors_the_packages_own_layout(adapter, monkeypatch, tmp_path):
    """`ADDING_A_PREFILL_MODEL.md:68`: "Mirror the layout the cache-populate run wrote".

    The cache-populate run is this package's own (P8's `G-WEIGHTS` extension and `G-MESH-KV`), which
    went through `ModelArgs.weight_cache_path` (`tt/model_config.py:250`). So the adapter's answer
    is asserted **equal** to `ModelArgs`', not merely plausible: a runner that invented the engine's
    `{name}_{arch}_{N}dev/{sp}x{tp}` convention instead would silently re-derive 15 GB of weights
    (`DEC-095`).
    """
    monkeypatch.setenv("PREFILL_TTNN_CACHE", str(tmp_path))

    class _MeshStub:
        shape = GALAXY_MESH_SHAPE

    args = ModelArgs(_MeshStub(), hf_config=ModelArgs.load_bundled_config(), model_path=str(tmp_path))
    expected = args.weight_cache_path(ttnn.bfloat8_b, cache_root=str(tmp_path))
    assert adapter.weight_cache_path(GALAXY_MESH_SHAPE) == expected
    assert expected.name == "tensor_cache_bfp8_4x8"


def test_weight_cache_path_falls_back_to_the_packages_variable_then_refuses(
    expect_error, adapter, monkeypatch, tmp_path
):
    """`PREFILL_TTNN_CACHE` (the engine's) first, `TT_CACHE_PATH` (this package's) second, then raise.

    Returning `None` is legal in the contract ("None only if the cache is explicitly empty",
    `models/demos/common/prefill/adapter.py:163`) and is refused here anyway: it would mean every
    runner start re-tilizes every weight, and this package already refuses to default the root to
    the checkpoint directory (`R-003`, `DEC-048`).
    """
    monkeypatch.delenv("PREFILL_TTNN_CACHE", raising=False)
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path))
    assert adapter.weight_cache_path((4, 8)).parent == tmp_path

    monkeypatch.delenv("TT_CACHE_PATH", raising=False)
    with expect_error(ValueError, "weight cache root is unset"):
        adapter.weight_cache_path((4, 8))


def test_weight_cache_path_does_not_touch_a_device(adapter, monkeypatch, tmp_path):
    """It is called before `open_mesh_device` (`prefill_runner.py:377` vs `:472`)."""
    monkeypatch.setenv("PREFILL_TTNN_CACHE", str(tmp_path))
    source = _executable_code(LlamaPrefillAdapter.weight_cache_path)
    assert "get_num_devices" not in source, (
        "get_num_devices() can abort with co-located migration workers and the mesh is not open yet "
        "(models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:78-81)"
    )
    # sp * tp, from the argument — so a (1,8) call cannot pick up the (4,8) cache.
    assert adapter.weight_cache_path((1, 8)) != adapter.weight_cache_path((4, 8))


@pytest.mark.skipif(not os.getenv("TT_CACHE_PATH"), reason="TT_CACHE_PATH unset; the populated cache is elsewhere")
def test_the_weight_cache_this_deployment_reads_is_populated(adapter):
    """`ADDING_A_PREFILL_MODEL.md:254`. A ledger fact, checked rather than claimed."""
    from loguru import logger

    path = adapter.weight_cache_path(GALAXY_MESH_SHAPE)
    cached = sorted(p for p in path.rglob("*.tensorbin"))
    logger.info(f"[G-ADAPTER] weight cache {path}: {len(cached)} tensorbin files")
    assert cached, f"no cached weights under {path}; run a populate pass before serving"


@pytest.mark.skipif(not os.getenv("PREFILL_TRACE_DIR"), reason="PREFILL_TRACE_DIR unset (the golden trace)")
def test_the_golden_trace_is_staged_in_the_layout_the_producer_reads(adapter):
    """`ADDING_A_PREFILL_MODEL.md:254`, and the shape the packed-GQA reader assumes.

    The reader opens `<trace>/kv_cache/layer_<i>.safetensors` and takes
    `key_cache_layer_<i>[0, :, :real_len, :]` (`prefill_producer.py:588-589`), so the trace's own
    `num_kv_heads` / `head_dim` must be this model's or the PCC compares differently-shaped tensors.
    """
    from models.demos.common.prefill.runners.runner_utils import resolve_trace_dir

    trace = resolve_trace_dir(os.environ["PREFILL_TRACE_DIR"])
    with open(trace / "metadata.json") as f:
        metadata = json.load(f)
    assert metadata["num_kv_heads"] == Llama31_8BConfig.NUM_KEY_VALUE_HEADS
    assert metadata["head_dim"] == Llama31_8BConfig.HEAD_DIM
    assert metadata["dtype"] == "float32", "the golden must be fp32 (`DEC-059`)"
    assert metadata["num_layers"] == Llama31_8BConfig.NUM_LAYERS
    for layer in range(metadata["num_layers"]):
        assert (trace / "kv_cache" / f"layer_{layer}.safetensors").is_file()


# =============================================================================================
# The two engine behaviours no doc states (recipe P10 warnings 1 and 2)
# =============================================================================================
def test_load_hf_config_returns_a_mutable_object(adapter):
    """Recipe P10 warning 2: the engine assigns `max_seq_len` on the next line (`prefill_runner.py:477`).

    A frozen dataclass raises `FrozenInstanceError` at runner startup, after `_print_config` has
    already printed a healthy-looking table.
    """
    config = adapter.load_hf_config()
    config.max_seq_len = 2816
    assert config.max_seq_len == 2816
    # The engine also reads `.hidden_size` off it (`prefill_runner.py:516`).
    assert config.hidden_size == Llama31_8BConfig.EMB_SIZE
    # ...and this package's modules take the DICT (recipe P1 trap 2), which is what `.dims` is.
    assert isinstance(config.dims, dict) and config.dims["num_hidden_layers"] == 32
    assert config.head_dim == derive_head_dim(config.dims)


def test_load_hf_config_does_not_route_theta_through_getattr():
    """Recipe P1 trap 1 / `R-005`, the highest-severity silent-wrongness trap in this bring-up.

    The template loads its config with `AutoConfig.from_pretrained`
    (`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:68`); on `transformers` 5.12.1 the
    resulting object has no `rope_theta`, so a `getattr(cfg, "rope_theta", DEFAULT)` anywhere
    downstream returns the default and RoPE is wrong at every position with nothing raised.
    """
    import ast

    source = _executable_code(sys.modules[ADAPTER_MODULE])
    assert "AutoConfig" not in source, "AutoConfig reintroduces the rope_theta trap (R-005)"

    # Every three-argument `getattr` in the module — the exact shape of the trap, which substitutes
    # its default silently when the attribute has moved (here, into `rope_parameters`).
    defaulted = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) == 3 and isinstance(node.args[1], ast.Constant):
                defaulted.add(node.args[1].value)
    assert defaulted <= {"max_seq_len"}, (
        f"the adapter reads {sorted(defaulted)} with a silent getattr default. `max_seq_len` is the "
        f"only legal one — the engine assigns it after load_hf_config returns (prefill_runner.py:477), "
        f"so it is genuinely absent until then. Every dimension comes from the dict (R-005)."
    )
    # And the values that trap would have replaced are the checkpoint's, read out of the dict.
    dims = LlamaPrefillAdapter().load_hf_config().dims
    assert dims["rope_theta"] == 500000.0 and dims["rope_scaling"]["rope_type"] == "llama3"


def test_load_hf_config_refuses_a_disagreeing_checkpoint_config(expect_error, adapter, monkeypatch, tmp_path):
    """**Negative control** for the config path: a foreign `config.json` must be refused, not used.

    Every dimension, every threshold and the whole weight cache were built against the bundled copy
    (`DEC-001`), so `PREFILL_HF_MODEL` pointing at a differently-shaped checkpoint is a `ValueError`
    rather than a second source of dimensions (`DEC-094`).
    """
    dims = ModelArgs.load_bundled_config()
    dims["num_hidden_layers"] = 80
    (tmp_path / "config.json").write_text(json.dumps(dims))
    monkeypatch.setenv("PREFILL_HF_MODEL", str(tmp_path))
    with expect_error(ValueError, "num_hidden_layers"):
        adapter.load_hf_config()

    # ...and the positive half: a byte-identical copy elsewhere is accepted.
    (tmp_path / "config.json").write_text(open(BUNDLED_CONFIG_PATH).read())
    assert adapter.load_hf_config().num_hidden_layers == 32


def test_load_hf_config_refuses_a_directory_with_no_config(expect_error, adapter, monkeypatch, tmp_path):
    monkeypatch.setenv("PREFILL_HF_MODEL", str(tmp_path))
    with expect_error(FileNotFoundError, "has no config.json"):
        adapter.load_hf_config()


# =============================================================================================
# allocate_kv_cache — the handle's type, without a device
# =============================================================================================
def test_the_allocated_cache_is_the_engines_opaque_handle():
    """`ADDING_A_PREFILL_MODEL.md:70-76`: `allocate_kv_cache` returns a `KvCaches` subclass.

    This package's `LlamaKVCache` **is** one (`tt/attention/kv_cache.py:56`), so it needs no
    one-element-list wrapper (`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:31-38`) —
    and the runtime's `_resolve_kv` accepts both forms anyway (`DEC-062`). Device-free: the gate's
    device is "—", and the allocation itself is exercised by `G-KV-TABLE` and `G-MOCK-MIG`.
    """
    assert issubclass(LlamaKVCache, KvCaches)
    signature = inspect.signature(LlamaPrefillAdapter.allocate_kv_cache)
    assert set(signature.parameters) == {"self", "mesh_device", "hf_config", "params"}
    for name in ("mesh_device", "hf_config", "params"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"the engine calls allocate_kv_cache with keywords only (prefill_runner.py:500); "
            f"{name} is {signature.parameters[name].kind}"
        )
    source = _executable_code(LlamaPrefillAdapter.allocate_kv_cache)
    assert (
        "params.num_layers" in source and "params.max_seq_len" in source and "params.num_users" in source
    ), "the cache must be sized from params, never from os.environ (BRINGUP_RECIPE.md:1894)"


def test_the_adapter_reads_no_knob_from_the_environment_except_the_two_the_contract_defines():
    """`BRINGUP_RECIPE.md:1894`: "Read knobs from `params`, **never** from `os.environ`".

    Three reads survive, and each is the engine's own documented override of a *class attribute*
    rather than a run knob: `PREFILL_HF_MODEL` over `hf_model_default` and `PREFILL_TTNN_CACHE` over
    `ttnn_cache_default` (`models/demos/common/prefill/adapter.py:116-117`), plus `TT_CACHE_PATH`,
    this package's own cache root (`tt/model_config.py:240`) and `HF_MODEL`, its checkpoint
    (`:123`). Anything else is a knob that should have come off `params`.
    """
    import ast

    tree = ast.parse(_executable_code(sys.modules[ADAPTER_MODULE]))
    read = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("get", "getenv"):
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                read.add(node.args[0].value)
    assert read <= {
        "PREFILL_HF_MODEL",
        "PREFILL_TTNN_CACHE",
        "TT_CACHE_PATH",
        "HF_MODEL",
    }, f"the adapter reads {sorted(read)} from the environment; knobs come from PrefillRunParams"


def test_the_hf_config_view_is_not_a_transformers_config(expect_error):
    """`ModelArgs` refuses a config *object* (`tt/model_config.py:101-107`), so `.dims` is what flows."""
    config = LlamaHfConfig({"hidden_size": 4096, "num_hidden_layers": 32, "num_key_value_heads": 8}, head_dim=128)
    assert not hasattr(config, "to_dict"), "a transformers-config-shaped object would slip past ModelArgs' check"

    class _MeshStub:
        shape = GALAXY_MESH_SHAPE

    with expect_error(TypeError, "must be the raw config.json dict"):
        ModelArgs(_MeshStub(), hf_config=config)
