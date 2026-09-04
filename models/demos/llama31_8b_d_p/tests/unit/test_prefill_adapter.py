# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`G-ADAPTER` — the disaggregated-prefill contract, item by item, with no device.

Covers the closing checklist of `models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md`:

1. the adapter implements every abstract `PrefillModelAdapter` method and sets `name`,
   `model_config` and the default paths;
2. `build_runtime` returns a runtime satisfying the §2 interface (signature-level here; `G-REQUEST`
   runs it);
3. **no reference-modeling / heavy imports at module load** — measured in a subprocess, because a
   parent that already imported ttnn cannot observe it;
4. the adapter is registered in `ADAPTER_PATHS` and `PREFILL_MODEL=llama31_8b_d_p` resolves;
5. the manifest is well-formed and its knobs are the ones this package's gates ran with.

Plus the two cross-package invariants P10 took on, neither of which any other test would catch:

* the producer's KV read-back is **not** adapter-dispatched — it branches on `ADAPTER.name`
  (`prefill_producer.py`), and if our name is missing it silently falls through to the MLA reader
  and PCCs the wrong bytes (`DEC-105`, `08_PREFILL_INTEGRATION.md`);
* that branch is only correct while our DRAM block geometry equals gpt-oss's, so the two constants
  are asserted equal here rather than assumed.
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from models.demos.common.prefill.adapter import ADAPTER_PATHS, PrefillModelAdapter, get_adapter
from models.demos.llama31_8b_d_p.tt.model_dims import Llama31_8BConfig

MODEL_NAME = "llama31_8b_d_p"
_REPO_ROOT = Path(__file__).resolve().parents[5]
_PKG_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER_MODULE = "models.demos.llama31_8b_d_p.tt.runners.adapters.llama"
_MANIFEST = _PKG_ROOT / "tt" / "runners" / "manifests" / f"{MODEL_NAME}.json"


@pytest.fixture(scope="module")
def adapter():
    return get_adapter(MODEL_NAME)


# ---------------------------------------------------------------------------------------
# Checklist item 4 — registration
# ---------------------------------------------------------------------------------------
def test_registered_in_adapter_paths_and_resolves(adapter):
    """`PREFILL_MODEL=llama31_8b_d_p` reaches our class through the registry."""
    assert MODEL_NAME in ADAPTER_PATHS, f"{MODEL_NAME} is not in ADAPTER_PATHS; the runner cannot select it"
    dotted = ADAPTER_PATHS[MODEL_NAME]
    assert dotted == f"{_ADAPTER_MODULE}:LlamaPrefillAdapter", dotted
    assert isinstance(adapter, PrefillModelAdapter)
    assert adapter.name == MODEL_NAME, "name is also the weight-cache dir prefix; it must be the registry key"
    # get_adapter memoises, so the engine and the tests share one instance.
    assert get_adapter(MODEL_NAME) is adapter


def test_registry_entry_is_lazy(adapter):
    """`ADAPTER_PATHS` holds a dotted STRING, so importing the common module imports no model."""
    assert isinstance(ADAPTER_PATHS[MODEL_NAME], str)


def test_unknown_model_name_still_raises_listing_ours(expect_error):
    with expect_error(KeyError, MODEL_NAME):
        get_adapter("llama31_8b_d_p_typo")


# ---------------------------------------------------------------------------------------
# Checklist item 1 — every abstract method + the identity attributes
# ---------------------------------------------------------------------------------------
def test_no_abstract_methods_remain(adapter):
    assert not getattr(type(adapter), "__abstractmethods__", set()), (
        f"{type(adapter).__name__} still has abstract methods "
        f"{sorted(type(adapter).__abstractmethods__)}; instantiating it would raise at runner start"
    )


@pytest.mark.parametrize("method", ["load_hf_config", "weight_cache_path", "allocate_kv_cache", "build_runtime"])
def test_implements_the_four_contract_methods_itself(adapter, method):
    """Each of the four is defined on OUR class, not inherited as the abstract stub."""
    own = type(adapter).__dict__.get(method)
    assert own is not None, f"{method} is not defined on {type(adapter).__name__}"
    assert not getattr(own, "__isabstractmethod__", False), f"{method} is still abstract"


def test_identity_and_default_path_attributes(adapter):
    assert adapter.model_config is Llama31_8BConfig
    # hf_model_default must be a real directory holding config.json -- an unresolvable default is a
    # startup failure in the runner, not a fallback.
    default_cfg = Path(adapter.hf_model_default) / "config.json"
    assert default_cfg.is_file(), f"hf_model_default={adapter.hf_model_default} has no config.json"
    # "" is the documented "must come from the env var" value for both of these (DEC-057).
    assert adapter.ttnn_cache_default == ""
    assert adapter.prefill_trace_default == ""
    assert adapter.l1_small_size == 0, "no op in this package routes semaphores to L1_SMALL"
    assert adapter.supports_dflash is False, "the DFlash drafter is a Kimi-only checkpoint"


def test_model_config_exposes_everything_the_engine_reads(adapter):
    """The three call sites that read `ADAPTER.model_config` and cannot take an instance."""
    mc = adapter.model_config
    # runner_utils.open_mesh_device -> FabricRouterConfig.max_packet_payload_size_bytes
    assert isinstance(mc.FABRIC_PAYLOAD_SIZE, int) and mc.FABRIC_PAYLOAD_SIZE > 0
    # prefill_producer._read_slot_kv_and_check_pcc_gpt_oss
    assert mc.NUM_KEY_VALUE_HEADS == 8
    assert mc.HEAD_DIM == 128
    assert getattr(mc, "ROTARY_DIM", mc.HEAD_DIM) == mc.HEAD_DIM, "Llama-3.1 is full rotary"


def test_model_dims_match_the_bundled_config(adapter):
    """`tt/model_dims.py` is a second view of the same numbers, so it must not drift from the first."""
    from models.demos.llama31_8b_d_p.tests.test_factory import llama_config_dims

    dims = llama_config_dims()
    mc = adapter.model_config
    assert mc.EMB_SIZE == dims["hidden_size"]
    assert mc.INTERMEDIATE_SIZE == dims["intermediate_size"]
    assert mc.NUM_LAYERS == dims["num_hidden_layers"]
    assert mc.NUM_ATTENTION_HEADS == dims["num_attention_heads"]
    assert mc.NUM_KEY_VALUE_HEADS == dims["num_key_value_heads"]
    assert mc.HEAD_DIM == dims["head_dim"]
    assert mc.GQA_GROUP_SIZE == dims["gqa_group_size"]
    assert mc.VOCAB_SIZE == dims["vocab_size"]
    assert mc.MAX_POSITION_EMBEDDINGS == dims["max_position_embeddings"]
    assert mc.RMS_NORM_EPS == dims["rms_norm_eps"]
    assert mc.ROPE_THETA == dims["rope_theta"]
    assert mc.ROPE_TYPE == dims["rope_scaling"]["rope_type"]
    assert mc.ROPE_SCALING_FACTOR == dims["rope_scaling"]["factor"]
    assert mc.ROPE_ORIG_CONTEXT_LEN == dims["rope_scaling"]["original_max_position_embeddings"]
    assert mc.FABRIC_PAYLOAD_SIZE == mc.EMB_SIZE, "the repo-wide convention (DEC-103)"


def test_load_hf_config_returns_the_normalised_object_and_accepts_max_seq_len(adapter):
    """`DEC-009` (normalised object, not a dict) and `DEC-100` (the engine stamps `max_seq_len`)."""
    from models.demos.llama31_8b_d_p.tt.model_config import LlamaHFConfig

    cfg = adapter.load_hf_config()
    assert isinstance(cfg, LlamaHFConfig)
    for attr in ("head_dim", "num_key_value_heads", "rope_theta", "rope_type", "hidden_size"):
        assert getattr(cfg, attr) is not None, attr
    assert cfg.rope_theta == 500000.0, "R-014: a getattr default would silently give 150000.0 here"
    # This is the exact line the runner runs next (prefill_runner.py:475).
    cfg.max_seq_len = 4096
    assert cfg.max_seq_len == 4096


def test_load_hf_config_keeps_every_declared_dimension_frozen(adapter, expect_error):
    """`DEC-100`'s subclass must not weaken `DEC-009`: fields stay immutable."""
    from dataclasses import FrozenInstanceError

    cfg = adapter.load_hf_config()
    with expect_error(FrozenInstanceError, "cannot assign to field"):
        cfg.rope_theta = 1.0
    with expect_error(FrozenInstanceError, "cannot assign to field"):
        cfg.head_dim = 64


def test_load_hf_config_refuses_a_path_without_config_json(adapter, expect_error, monkeypatch, tmp_path):
    monkeypatch.setenv("PREFILL_HF_MODEL", str(tmp_path))
    with expect_error(AssertionError, "no config.json at"):
        adapter.load_hf_config()


def test_weight_cache_path_matches_model_args(adapter, monkeypatch, tmp_path):
    """The adapter and `ModelArgs` must agree, or the runner rebuilds a cache the package has.

    `ModelArgs` needs only `mesh_device.shape`, so a shape-only stand-in exercises the real method
    without a device (which the adapter side could not use anyway -- the runner calls it from
    `_print_config`, before `open_mesh_device`).
    """
    from models.demos.llama31_8b_d_p.tt.model_config import ModelArgs

    monkeypatch.setenv("LLAMA31_8B_TTNN_CACHE", str(tmp_path))
    monkeypatch.delenv("PREFILL_TTNN_CACHE", raising=False)
    monkeypatch.delenv("TT_CACHE_PATH", raising=False)

    class _ShapeOnlyMesh:
        shape = (4, 8)

    import ttnn

    args = ModelArgs(
        mesh_device=_ShapeOnlyMesh(),
        weights_path=str(_PKG_ROOT / "configs" / "Llama-3.1-8B-Instruct"),
    )
    assert adapter.weight_cache_path((4, 8)) == args.weight_cache_path(ttnn.bfloat8_b)


def test_weight_cache_path_is_mesh_shape_and_dtype_specific(adapter, monkeypatch, tmp_path):
    """`R-017`: a `(1,1)` cache replayed at `(4,8)` is silently garbage, so the shape is in the path."""
    monkeypatch.setenv("PREFILL_TTNN_CACHE", str(tmp_path))
    p48 = adapter.weight_cache_path((4, 8))
    p11 = adapter.weight_cache_path((1, 1))
    assert p48 != p11
    assert p48.name == "tensor_cache_bfp8" and "4x8" in p48.parts and "llama31_8b_d_p_bh_32dev" in p48.parts
    assert "1x1" in p11.parts and "llama31_8b_d_p_bh_1dev" in p11.parts


def test_prefill_ttnn_cache_wins_over_the_model_args_roots(adapter, monkeypatch, tmp_path):
    """The engine's own knob has to be able to redirect a deployment's cache."""
    monkeypatch.setenv("PREFILL_TTNN_CACHE", str(tmp_path / "engine"))
    monkeypatch.setenv("LLAMA31_8B_TTNN_CACHE", str(tmp_path / "package"))
    assert str(tmp_path / "engine") in str(adapter.weight_cache_path((4, 8)))


# ---------------------------------------------------------------------------------------
# Checklist item 2 — the runtime interface `build_runtime` promises
# ---------------------------------------------------------------------------------------
def test_build_runtime_signature_is_keyword_only_as_the_engine_calls_it(adapter):
    sig = inspect.signature(type(adapter).build_runtime)
    assert list(sig.parameters) == ["self", "mesh_device", "hf_config", "params"]
    for name in ("mesh_device", "hf_config", "params"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_allocate_kv_cache_signature_and_kvcaches_subclass(adapter):
    from models.demos.common.prefill.adapter import KvCaches
    from models.demos.llama31_8b_d_p.tt.runners.adapters.llama import LlamaKvCaches

    sig = inspect.signature(type(adapter).allocate_kv_cache)
    assert list(sig.parameters) == ["self", "mesh_device", "hf_config", "params"]
    assert issubclass(LlamaKvCaches, KvCaches)
    # The engine treats it as opaque; the runtime pulls the cache back with [0].
    handle = LlamaKvCaches(["sentinel"])
    assert handle[0] == "sentinel" and len(handle) == 1


def test_runtime_exposes_the_five_config_names_and_three_methods_the_engine_drives():
    """§2: `config.{chunk_size,max_seq_len,first_layer_idx,is_first_rank,is_last_rank}` + the calls."""
    from models.demos.llama31_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    cfg = TtPrefillRuntimeConfig(num_layers=1, max_seq_len=256, default_chunk_size=128)
    for name in ("chunk_size", "max_seq_len", "first_layer_idx", "is_first_rank", "is_last_rank"):
        assert hasattr(cfg, name), name
    assert cfg.chunk_size == cfg.default_chunk_size, "DEC-054"

    for name in ("compile", "make_chunk_input", "prefill_chunk", "set_layer_ack_channel"):
        assert callable(getattr(TtPrefillRuntime, name, None)), name
    for name in ("build_kv_chunk_table", "kv_migration_base_address"):
        assert callable(getattr(TtPrefillRuntime, name, None)), f"optional migration hook {name} missing"


def test_prefill_chunk_accepts_every_kwarg_the_engine_passes():
    """`_compute_and_send` passes `request_id`, `d2h_service` AND `metadata_msg` on every chunk.

    Omitting any one is a `TypeError` on the first served request, after the mesh is open and the
    weights are loaded -- the most expensive possible place to find it (`DEC-106`).
    """
    from models.demos.llama31_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    params = inspect.signature(TtPrefillRuntime.prefill_chunk).parameters
    for name in ("slot_id", "actual_start", "actual_end", "request_id", "d2h_service", "metadata_msg"):
        assert name in params, f"prefill_chunk does not accept {name!r}, which prefill_runner always passes"


def test_the_engine_call_site_passes_nothing_prefill_chunk_lacks():
    """Read the engine's own call and compare, so a new engine kwarg fails here, not on device."""
    import models.demos.common.prefill.runners.prefill_runner as runner
    from models.demos.llama31_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    tree = ast.parse(Path(inspect.getsourcefile(runner)).read_text())
    passed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "prefill_chunk":
            passed |= {kw.arg for kw in node.keywords if kw.arg is not None}
    assert passed, "found no runtime.prefill_chunk(...) call in prefill_runner; the audit is broken"
    accepted = set(inspect.signature(TtPrefillRuntime.prefill_chunk).parameters)
    assert passed <= accepted, f"prefill_runner passes {sorted(passed - accepted)}, which prefill_chunk rejects"


# ---------------------------------------------------------------------------------------
# Checklist item 3 — import-light
# ---------------------------------------------------------------------------------------
_IMPORT_PROBE = """
import json, sys, time
t = time.perf_counter()
import {module}
dt = time.perf_counter() - t
heavy = sorted(m for m in ("ttnn", "torch", "transformers", "safetensors") if m in sys.modules)
print("RESULT " + json.dumps({{"seconds": dt, "heavy": heavy}}))
"""


def _probe_import(module: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE.format(module=module)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, f"importing {module} failed:\n{out.stderr[-4000:]}"
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


def test_adapter_module_imports_nothing_heavy():
    """The checklist's "no reference-modeling / heavy imports at module load", measured.

    A subprocess, because this pytest process has already imported ttnn and torch and so cannot
    observe what the module pulls in. `models/demos/deepseek_v3_d_p/tests/conftest.py:33`
    instantiates every registered adapter at collection time, so this cost is paid by that suite too.
    """
    res = _probe_import(_ADAPTER_MODULE)
    assert res["heavy"] == [], (
        f"importing the adapter pulled in {res['heavy']} at module scope. Move those imports inside "
        f"the methods that need them -- the H2D producer imports this module in a process that "
        f"never opens a device."
    )
    assert res["seconds"] < 1.0, f"adapter import took {res['seconds']:.2f}s; something heavy crept in"


def test_model_dims_module_imports_nothing_at_all():
    """`tt/model_dims.py` is imported at adapter module scope, so it must stay dependency-free."""
    src = (_PKG_ROOT / "tt" / "model_dims.py").read_text()
    tree = ast.parse(src)
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == [], f"tt/model_dims.py must import nothing; found {len(imports)} import statement(s)"


def test_adapter_module_scope_has_no_package_device_imports():
    """Structural guard: no top-level `import ttnn` / `from ...tt.<device module>` in the adapter."""
    src = (_PKG_ROOT / "tt" / "runners" / "adapters" / "llama.py").read_text()
    tree = ast.parse(src)
    banned = []
    for node in tree.body:  # module scope only -- imports inside functions are the point
        if isinstance(node, ast.Import):
            banned += [a.name for a in node.names if a.name.split(".")[0] in ("ttnn", "torch", "transformers")]
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in ("ttnn", "torch", "transformers"):
                banned.append(node.module)
            if node.module.startswith("models.demos.llama31_8b_d_p.tt.") and not node.module.endswith("model_dims"):
                banned.append(node.module)
    assert banned == [], f"module-scope heavy imports in the adapter: {banned}"


# ---------------------------------------------------------------------------------------
# Checklist item 5 (config side) — the manifest
# ---------------------------------------------------------------------------------------
def test_manifest_is_valid_and_selects_this_model():
    assert _MANIFEST.is_file(), f"{_MANIFEST} missing"
    manifest = json.loads(_MANIFEST.read_text())
    env = manifest["env"]
    assert env["PREFILL_MODEL"] == MODEL_NAME
    # Every value must be a string: the runner does os.environ.setdefault(key, str(val)), and a JSON
    # int would work here but read back differently from a YAML global_env.
    assert all(isinstance(v, str) for v in env.values()), env
    assert int(env["PREFILL_NUM_LAYERS"]) == Llama31_8BConfig.NUM_LAYERS
    assert int(env["PREFILL_TP"]) == Llama31_8BConfig.NUM_KEY_VALUE_HEADS, "R-027: one KV head per chip"
    assert int(env["PREFILL_SP"]) * int(env["PREFILL_TP"]) == 32, "the (4,8) Blackhole galaxy"
    # Ring collectives need the cyclic torus route; the engine would otherwise default to FABRIC_1D
    # at sp<=8 and the Ring topology would hang rather than error (DEC-020 / DEC-081).
    assert env["PREFILL_FABRIC_MODE"] == "1d_ring"
    assert env["PREFILL_TOPOLOGY"] == "ring"
    # Workload knobs must NOT be pinned here: they have to match the producer exactly.
    for knob in ("PREFILL_CHUNK_SIZE", "PREFILL_MAX_SEQ_LEN", "PREFILL_NUM_USERS"):
        assert knob not in env, f"{knob} belongs on the run, not in the model manifest"


# ---------------------------------------------------------------------------------------
# The two cross-package invariants P10 took on
# ---------------------------------------------------------------------------------------
def test_producer_dispatches_our_name_to_the_packed_gqa_reader():
    """Without this branch the gate falls through to the MLA reader and PCCs the wrong bytes."""
    import models.demos.common.prefill.runners.prefill_producer as producer

    assert MODEL_NAME in producer._PACKED_GQA_MODELS, (
        f"{MODEL_NAME} is not in prefill_producer._PACKED_GQA_MODELS, so "
        f"_read_slot_kv_and_check_pcc falls through to _read_slot_kv_and_check_pcc_mla -- which "
        f"reads a merged MLA cache and would report a meaningless PCC for our packed K/V "
        f"(08_PREFILL_INTEGRATION.md, DEC-105)."
    )
    assert callable(producer._read_slot_kv_and_check_pcc_gpt_oss)


def test_our_dram_block_geometry_still_equals_gpt_oss():
    """The branch above is only legitimate while the two layouts agree — so assert it, don't assume.

    The reader imports gpt-oss's own `NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK` even when it is reading a
    Llama cache, so a change on either side silently misaligns every read.
    """
    from models.demos.gpt_oss_d_p.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK as GPT_OSS_BLOCK
    from models.demos.llama31_8b_d_p.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK as OUR_BLOCK

    assert OUR_BLOCK == GPT_OSS_BLOCK == 32, (
        f"our DRAM bank block is {OUR_BLOCK} tokens and gpt-oss's is {GPT_OSS_BLOCK}. The producer's "
        f"packed-GQA read-back strides by GPT-OSS's constant while reading OUR cache, so they must "
        f"stay equal or the check reads misaligned bytes. If this has to change, write a fourth "
        f"reader instead of widening the branch (DEC-105)."
    )


def test_hf_to_meta_permutation_matches_the_producers_inline_copy():
    """The producer rebuilds the HF->Meta lane map inline; ours is the package's single definition."""
    import torch

    from models.demos.llama31_8b_d_p.scripts.verify_golden_kv import hf_to_meta_lane_permutation

    head_dim = Llama31_8BConfig.HEAD_DIM
    rotary_dim = Llama31_8BConfig.ROTARY_DIM
    half = rotary_dim // 2
    producer_perm = list(range(head_dim))
    for m in range(rotary_dim):
        producer_perm[m] = half * (m % 2) + (m // 2)
    assert torch.equal(hf_to_meta_lane_permutation(head_dim, rotary_dim), torch.tensor(producer_perm))
