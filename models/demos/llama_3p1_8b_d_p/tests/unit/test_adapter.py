# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Runner-integration tests for the Llama-3.1-8B prefill adapter (tt-blaze#4149).

The adapter is the engine <-> model boundary, so what is worth testing is the **translation**: the
engine hands it a ``PrefillRunParams`` and expects a KV cache and a runtime configured to match. A
field dropped or mistranslated here does not crash — it produces a rank that silently disagrees with
its neighbours about the mesh axes, the pipeline role, or the cache stride.

Most of this is device-free. ``build_runtime``'s translation is checked by intercepting the runtime
constructor, which keeps the assertions on the mapping itself rather than on a 16 GB weight load;
``test_adapter_builds_a_working_runtime`` then runs the real thing on a device.
"""

from __future__ import annotations

import inspect
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

import ttnn
from models.demos.common.prefill.adapter import ADAPTER_PATHS, PrefillRunParams, get_adapter
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tests.mesh_profiles import galaxy_torus_xy_device_params
from models.demos.llama_3p1_8b_d_p.tt.runners.adapters import llama_3p1_8b as llama_adapter
from models.demos.llama_3p1_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

MODEL_NAME = "llama_3p1_8b"


def _params(**overrides) -> PrefillRunParams:
    """A single-rank ``PrefillRunParams`` with the 4x8 production geometry, overridable per test."""
    base = dict(
        mesh_shape=(4, 8),
        num_layers=Llama31_8BConfig.NUM_LAYERS,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=4096,
        chunk_size=512,
        num_users=1,
        capacity_factor=1,
        num_links=1,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=False,
        weight_cache_path=None,
    )
    base.update(overrides)
    return PrefillRunParams(**base)


# =====================================================================================
# Registry
# =====================================================================================
def test_model_is_registered_and_resolvable():
    """``PREFILL_MODEL=llama_3p1_8b`` reaches this adapter.

    Registration is what #4149 adds; until it landed the model was unreachable by name on purpose,
    so that a half-built model could not be selected.
    """
    assert MODEL_NAME in ADAPTER_PATHS
    adapter = get_adapter(MODEL_NAME)
    assert adapter.name == MODEL_NAME
    assert adapter.model_config is Llama31_8BConfig
    # Memoized: the engine resolves by name from several places and must not rebuild.
    assert get_adapter(MODEL_NAME) is adapter


def test_registration_did_not_break_import_lightness():
    """Registering in ``ADAPTER_PATHS`` must not make the common registry import this model.

    The registry is a dict of dotted *strings* resolved lazily for exactly this reason: the H2D
    producers import it to read the model list, and pulling torch + ttnn + transformers into those
    processes is what the import-light contract exists to prevent. This is the test that catches
    someone "simplifying" the strings into real imports.
    """
    probe = (
        "import sys;"
        "import models.demos.common.prefill.adapter as a;"
        "assert 'llama_3p1_8b' in a.ADAPTER_PATHS;"
        "print(','.join(m for m in ('torch', 'ttnn', 'transformers', 'safetensors') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "", f"importing the registry pulled in {out}"


# =====================================================================================
# params -> cache / runtime translation
# =====================================================================================
@pytest.mark.parametrize(
    "tp, expected_heads_per_chip",
    [(8, 1), (4, 2), (2, 4), (1, 8)],
)
def test_kv_heads_per_chip_follows_tp(tp, expected_heads_per_chip):
    """8 KV heads over the TP columns. One per chip at the production TP=8."""
    adapter = get_adapter(MODEL_NAME)
    params = _params(mesh_shape=(4, tp))
    assert adapter._num_kv_heads_per_chip(params) == expected_heads_per_chip


def test_kv_caches_handle_is_constructible_off_device():
    """``Llama31KvCaches(caches=[...])`` must build and index without a device.

    The handle is only ever constructed at the end of ``allocate_kv_cache``, which needs a mesh, so
    a missing ``@dataclass`` on it (the annotation alone gives no ``__init__``, and the ABC base
    supplies none) failed as ``TypeError: Llama31KvCaches() takes no arguments`` on hardware and
    nowhere else. Constructing it with a stand-in cache keeps that a device-free failure.
    """
    from models.demos.llama_3p1_8b_d_p.tt.runners.adapters.llama_3p1_8b import Llama31KvCaches

    sentinel = object()
    caches = Llama31KvCaches(caches=[sentinel])
    # The engine only ever reaches the single full-causal cache through [0].
    assert caches[0] is sentinel


def test_tp_wider_than_the_kv_head_count_is_refused(expect_error):
    """TP=16 would need KV-head replication, which breaks the one-head-per-chip migration property.

    Refused at allocation rather than left to shard 8 heads over 16 chips, which would either pad
    or place half the columns' caches on data they do not own.
    """
    adapter = get_adapter(MODEL_NAME)
    with expect_error(ValueError, "does not divide"):
        adapter._num_kv_heads_per_chip(_params(mesh_shape=(2, 16)))


def test_build_runtime_translates_every_rank_field(monkeypatch):
    """Each ``PrefillRunParams`` field the model cares about reaches the runtime config.

    Pinned field-by-field because the failure mode is quiet: a dropped ``is_last_rank`` gives a
    pipeline where no rank applies the final norm (or every rank does), and a dropped ``tp_axis``
    transposes the whole mesh. Neither raises.

    The runtime constructor is intercepted rather than run, so this stays device-free and grades the
    mapping instead of a weight load.
    """
    import models.demos.llama_3p1_8b_d_p.tt.tt_prefill_runtime as runtime_mod

    captured = {}

    def fake_runtime(*, mesh_device, config, state_dict):
        captured.update(mesh_device=mesh_device, config=config, state_dict=state_dict)
        return "runtime"

    monkeypatch.setattr(runtime_mod, "TtPrefillRuntime", fake_runtime)
    monkeypatch.setattr(
        "models.demos.llama_3p1_8b_d_p.tt.model_config.load_llama_state_dict",
        lambda *a, **k: {"sentinel": True},
    )

    params = _params(
        mesh_shape=(4, 8),
        num_layers=8,
        first_layer_idx=24,
        is_first_rank=False,
        is_last_rank=True,
        max_seq_len=8192,
        chunk_size=1024,
        num_users=2,
        num_links=4,
        weight_cache_path=Path("/tmp/does-not-need-to-exist"),
    )

    class HfConfig:
        vocab_size = Llama31_8BConfig.VOCAB_SIZE

    adapter = get_adapter(MODEL_NAME)
    assert adapter.build_runtime(mesh_device="mesh", hf_config=HfConfig(), params=params) == "runtime"

    config = captured["config"]
    assert (config.max_seq_len, config.chunk_size) == (8192, 1024)
    assert config.mesh_shape == (4, 8)
    assert (config.num_layers, config.first_layer_idx) == (8, 24)
    assert (config.is_first_rank, config.is_last_rank) == (False, True)
    assert (config.num_users, config.num_links, config.tp_axis) == (2, 4, 1)
    assert config.vocab_size == Llama31_8BConfig.VOCAB_SIZE
    assert config.weight_cache_path == Path("/tmp/does-not-need-to-exist")
    # The engine allocated the cache via allocate_kv_cache and passes it into every call; a runtime
    # that owned one would keep a second alive across a rebuild.
    assert config.owns_kv_cache is False
    assert captured["state_dict"] == {"sentinel": True}


def test_build_runtime_skips_the_weight_read_when_serving_from_cache(monkeypatch, expect_error):
    """``LLAMA31_8B_WEIGHTS_FROM_CACHE=1`` hands the runtime an empty state dict...

    ...because with a populated TTNN cache the device tensors are read back by name and the torch
    weights are never touched, so the safetensors read is pure startup cost. It needs a cache path
    to read from, and asking for it without one is refused rather than silently building random
    weights — which would produce a model that runs and returns noise.
    """
    import models.demos.llama_3p1_8b_d_p.tt.tt_prefill_runtime as runtime_mod

    captured = {}
    monkeypatch.setattr(
        runtime_mod,
        "TtPrefillRuntime",
        lambda *, mesh_device, config, state_dict: captured.update(state_dict=state_dict),
    )
    monkeypatch.setattr(
        "models.demos.llama_3p1_8b_d_p.tt.model_config.load_llama_state_dict",
        lambda *a, **k: pytest.fail("should not read safetensors when serving from the TTNN cache"),
    )
    monkeypatch.setenv("LLAMA31_8B_WEIGHTS_FROM_CACHE", "1")

    class HfConfig:
        vocab_size = Llama31_8BConfig.VOCAB_SIZE

    adapter = get_adapter(MODEL_NAME)
    adapter.build_runtime(
        mesh_device="mesh", hf_config=HfConfig(), params=_params(weight_cache_path=Path("/tmp/cache"))
    )
    assert captured["state_dict"] == {}

    with expect_error(ValueError, "needs a weight cache"):
        adapter.build_runtime(mesh_device="mesh", hf_config=HfConfig(), params=_params(weight_cache_path=None))


# =====================================================================================
# Weight loading
# =====================================================================================
def test_weight_key_filter_covers_a_rank_slice():
    """A rank asks for its own layers plus the embedding and final norm, and nothing else.

    The filter is what keeps a 4-of-32-layer rank from materialising the whole 16 GB checkpoint, so
    it has to be exact in both directions: a missing key fails loudly at load, but an *extra* layer's
    keys would just quietly cost host memory on every rank.
    """
    from models.demos.llama_3p1_8b_d_p.tt.model_config import _wanted_keys

    keys = _wanted_keys(num_layers=4, first_layer_idx=8)
    assert "model.embed_tokens.weight" in keys
    assert "model.norm.weight" in keys
    for layer_idx in (8, 9, 10, 11):
        assert f"model.layers.{layer_idx}.self_attn.q_proj.weight" in keys
        assert f"model.layers.{layer_idx}.input_layernorm.weight" in keys
    for layer_idx in (7, 12, 31):
        assert not any(f"model.layers.{layer_idx}." in key for key in keys)
    # 7 projections + 2 norms per layer, + embedding + final norm.
    assert len(keys) == 4 * 9 + 2
    # No lm_head: prefill is headless, and loading a 128256x4096 tensor per rank to discard it is
    # ~1 GB of host memory for nothing.
    assert not any("lm_head" in key for key in keys)


def test_weight_loader_reports_a_missing_checkpoint(tmp_path, expect_error):
    """A wrong weights path fails with the path and the env var to fix it, not a bare KeyError."""
    from models.demos.llama_3p1_8b_d_p.tt.model_config import load_llama_state_dict

    with expect_error(FileNotFoundError, "no .safetensors"):
        load_llama_state_dict(tmp_path, num_layers=1)


# =====================================================================================
# Inter-process contract with the shared runner
# =====================================================================================
def test_h2d_metadata_stays_twelve_bytes():
    """The H2D metadata blob is exactly 12 B, and this model's chunk call matches it field for field.

    The shared runner owns the constant — no adapter sets it — but the size is a hard contract on
    both ends: ``H2DStreamService::forward_to_tensor`` TT_FATALs unless the span is exactly
    ``metadata_size_bytes``, so a fourth field added upstream breaks serving here with a message
    about a byte count rather than about the field someone added.

    Asserted against ``struct.calcsize`` for the three words the runner decodes and
    ``prefill_chunk`` consumes — ``(slot_id, actual_start, actual_end)`` — so the test states the
    layout it depends on instead of restating the literal 12.
    """
    from models.demos.common.prefill.runners.prefill_producer import METADATA_SIZE_BYTES as producer_size
    from models.demos.common.prefill.runners.prefill_runner import METADATA_SIZE_BYTES as runner_size

    three_uint32 = struct.calcsize("<III")  # slot_id, actual_start, actual_end
    assert three_uint32 == 12
    assert runner_size == three_uint32, f"runner metadata is {runner_size} B, not the 3-word blob"
    assert producer_size == runner_size, "producer and runner disagree on the metadata size"

    # The runtime's per-chunk signature is the Python-side half of that same triple.
    chunk_params = inspect.signature(TtPrefillRuntime.prefill_chunk).parameters
    assert {"slot_id", "actual_start", "actual_end"} <= set(chunk_params)


def test_manifest_env_precedence():
    """``global_env`` > manifest > code default, which is exactly what ``setdefault`` buys.

    Three cases in one pass, because the ordering only means something if all three hold:
    a key the rank binding already exported must survive the manifest, a key only the manifest names
    must be filled from it, and a key neither mentions must stay unset so the runner's own default
    applies. ``tt-run`` forwards only ``TT_/ARCH_/WH_/TTNN_/DEEPSEEK_/MESH_`` prefixes, so a
    shell-exported ``PREFILL_*`` never reaches the runner and ``global_env`` is the only way in —
    which makes this ordering the difference between a knob that works and one that is ignored.
    """
    from models.demos.common.prefill.runners import prefill_runner

    manifest_path = Path(llama_adapter.__file__).parents[1] / "manifests" / f"{MODEL_NAME}.json"
    manifest = json.loads(manifest_path.read_text())["env"]
    assert {"PREFILL_CHUNK_SIZE", "PREFILL_MAX_SEQ_LEN"} <= set(manifest), "manifest lost a key this test uses"
    assert "PREFILL_NUM_USERS" not in manifest, "manifest gained the key that stands in for a code default"

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ["PREFILL_MANIFEST"] = str(manifest_path)
        os.environ["PREFILL_CHUNK_SIZE"] = "4096"  # as a rank binding's global_env would set it
        os.environ.pop("PREFILL_MAX_SEQ_LEN", None)
        os.environ.pop("PREFILL_NUM_USERS", None)

        prefill_runner._apply_manifest_env()

        assert os.environ["PREFILL_CHUNK_SIZE"] == "4096", "manifest overwrote global_env"
        assert (
            os.environ["PREFILL_MAX_SEQ_LEN"] == manifest["PREFILL_MAX_SEQ_LEN"]
        ), "manifest did not fill an unset key"
        assert "PREFILL_NUM_USERS" not in os.environ, "a key no manifest names must be left to the code default"


# =====================================================================================
# Device
# =====================================================================================
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((4, 2), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="sp4-4x2"),
        # The production SP=4 x TP=8, where the KV-head assertion below pins one head per chip.
        pytest.param((4, 8), galaxy_torus_xy_device_params(), id="galaxy-sp4-tp8-4x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_adapter_allocates_a_cache_the_runtime_accepts(mesh_device, device_params, reset_seeds):
    """The two halves of the adapter agree: the cache it allocates is one its runtime will drive.

    This is the integration the engine actually performs — ``allocate_kv_cache`` then
    ``build_runtime``, and the runtime validates the cache it is handed on the first chunk. A
    disagreement about the layer stride or the KV head count between the two methods is exactly
    what this catches, and it needs no weights to catch it.
    """
    from models.demos.llama_3p1_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    adapter = get_adapter(MODEL_NAME)
    # From the device, not a constant: the runtime rejects a config whose mesh_shape disagrees with
    # the mesh it is handed, so this has to follow the arm.
    params = _params(mesh_shape=tuple(mesh_device.shape), num_layers=2, max_seq_len=1024, chunk_size=256)

    caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=None, params=params)
    cache = caches[0]
    assert cache.num_layers == params.num_layers  # rank-local stride
    assert cache.k.shape[1] == Llama31_8BConfig.NUM_KEY_VALUE_HEADS // params.tp_factor

    # Random weights: this grades the cache/runtime contract, not numerics (test_model_vs_ref does).
    runtime = TtPrefillRuntime(
        mesh_device=mesh_device,
        config=TtPrefillRuntimeConfig(
            max_seq_len=params.max_seq_len,
            chunk_size=params.chunk_size,
            mesh_shape=params.mesh_shape,
            num_layers=params.num_layers,
            num_users=params.num_users,
            vocab_size=2048,
        ),
    )
    # compile() must warm EVERY KV-length bucket, not just chunk 0: each cached_len is a different
    # program, so warming one bucket moves the JIT stall into the middle of a served request
    # instead of removing it. Counted through the completion sink, which fires once per layer per
    # chunk — 1024 / 256 = 4 chunk-aligned buckets, plus one deliberately non-chunk-aligned bucket
    # (tt-blaze#4148): a continuation resumes at aligned_resume_length, a multiple of 32, and takes
    # the rotated branch of the block-cyclic map, so it is a served path and belongs in warm-up.
    warmed = []
    runtime.set_layer_completion_sink(lambda layer_idx, request_id: warmed.append(layer_idx))
    runtime.compile(cache)
    expected_buckets = params.max_seq_len // params.chunk_size + 1
    assert (
        len(warmed) == expected_buckets * params.num_layers
    ), f"compile() warmed {len(warmed) / params.num_layers:g} bucket(s), expected {expected_buckets}"

    assert runtime.kv_migration_base_address(cache) == int(cache.k.buffer_address())
