# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The seam between the shared prefill runtime and Kimi-K3's own transformer.

`TtPrefillRuntime._build_model` calls `self.MODEL_CLS(...)` with a fixed keyword list. Kimi-K3's
transformer names the model-level knobs it understands and forwards the rest to `TtKimiK3Block` as
`**block_kwargs`, so a knob added upstream and NOT named here does not fail at the seam -- it
silently becomes a block argument and raises `TypeError` on the first layer, after mesh bringup and
weight load. That is a long way to travel for a signature mismatch, and it is only reachable through
`build_runtime`, which no eager test exercises.

`tp_shard_kv` did exactly this: it arrived with GLM-5.3's 2D KV sharding (#51968), after the branch's
merge-base, and broke every runner construction until it was named.

Hardware-free: this reads the call site's keyword list out of the source and binds it against the two
signatures. No mesh, no weights, no checkpoint.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.state_contract import KDA_CONVOLUTION_PAGE_CHANNELS
from models.demos.deepseek_v3_d_p.tt.kimi_k3 import attention as attention_module
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import (
    K3AttnContext,
    KdaChunkScalars,
    TtK3KdaAttention,
    build_attention,
    build_kda_scalars,
)
from models.demos.deepseek_v3_d_p.tt.kimi_k3.block import TtKimiK3Block
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer

_RUNTIME_SRC = Path(__file__).parents[2] / "tt" / "tt_prefill_runtime.py"


def _call_sites(callee: str, expected: int = 1) -> list[ast.Call]:
    """The calls to `callee` in `TtPrefillRuntime`, read off the source."""
    tree = ast.parse(_RUNTIME_SRC.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == callee]
    assert (
        len(calls) == expected
    ), f"expected {expected} {callee} call site(s), found {len(calls)}; the test needs updating, not deleting"
    for call in calls:
        assert not any(isinstance(arg, ast.Starred) for arg in call.args) and all(
            kw.arg or ast.unparse(kw.value).startswith("self._seed_kwargs(") for kw in call.keywords
        ), f"{callee} is called with */** expansion, which this test cannot bind; the test needs updating"
    return calls


def _bind(func, call: ast.Call, *leading) -> None:
    inspect.signature(func).bind(*leading, *[None] * len(call.args), **{kw.arg: None for kw in call.keywords if kw.arg})


def _model_cls_kwargs() -> list[str]:
    """The keywords `_build_model` passes to `MODEL_CLS`."""
    return [kw.arg for kw in _call_sites("self.MODEL_CLS")[0].keywords]


def test_every_runtime_kwarg_binds_to_the_transformer_or_the_block():
    passed = _model_cls_kwargs()
    named = set(inspect.signature(TtKimiK3Transformer.__init__).parameters)
    accepted = set(inspect.signature(TtKimiK3Block.__init__).parameters)

    swept = [k for k in passed if k not in named]
    orphans = [k for k in swept if k not in accepted]
    assert not orphans, (
        f"{orphans} reach TtKimiK3Transformer from TtPrefillRuntime._build_model, are not named in "
        f"its signature, and are not accepted by TtKimiK3Block -- so build_runtime raises TypeError "
        f"on the first layer. Name them in TtKimiK3Transformer.__init__ (rejecting the values "
        f"Kimi-K3 cannot honour) rather than widening TtKimiK3Block."
    )


def test_the_cache_check_call_binds_to_the_transformer():
    """`check_cache_complete` runs before the model is built, so a kwarg it rejects fails every runner."""
    _bind(TtKimiK3Transformer.check_cache_complete, _call_sites("self.MODEL_CLS.check_cache_complete")[0])


def test_every_forward_call_binds_to_the_transformer():
    """The eager and traced chunk paths and the packed `prefill_forward` call `self.model.forward`; none is reached by an eager test."""
    for call in _call_sites("self.model.forward", expected=3):
        _bind(TtKimiK3Transformer.forward, call, None)


@pytest.mark.parametrize(
    "mtp_kwargs",
    [
        {"mtp_union": object()},
        {"on_mtp_complete": Mock()},
        {"input_is_embedded": True},
        {"provided_levels": 1},
        {"mtp_geometry": object()},
    ],
)
def test_forward_rejects_mtp_arguments_before_device_work(expect_error, mtp_kwargs):
    model = object.__new__(TtKimiK3Transformer)
    with expect_error(ValueError, "no MTP predictor"):
        model.forward(object(), **mtp_kwargs)


def test_forward_checks_the_index_cache_against_the_config(expect_error):
    """K3 has no DSA indexer, so an index cache is a wiring mistake; K3.1's DSA layers cannot run without one."""
    model = object.__new__(TtKimiK3Transformer)
    model.has_indexer = False
    with expect_error(ValueError, "no DSA indexer"):
        model.forward(object(), index_kv_cache=object())
    model.has_indexer, model.schedule = True, SimpleNamespace(num_mla_layers=1)
    with expect_error(ValueError, "pass index_kv_cache"):
        model.forward(object())


def test_only_a_dsa_config_takes_a_sparse_kv_cache_format(expect_error):
    """K3's dense MLA cache cannot take a sparse format; refused before any device work."""
    from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

    with expect_error(ValueError, "dense MLA KV cache"):
        TtKimiK3Transformer(
            None,
            kimi_k3_hf_config(),
            KimiK3Config,
            {},
            num_layers=1,
            seq_len=5120,
            sparse_kv_cache_format=MlaKvCacheFormat.SCALED_FP8,
        )


def test_runtime_rejects_mtp_before_the_shared_build(monkeypatch, expect_error):
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.runtime import TtKimiK3Runtime
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    runtime = object.__new__(TtKimiK3Runtime)
    runtime.config = SimpleNamespace(mtp_levels=2)
    parent_build = create_autospec(TtPrefillRuntime._build_model)
    monkeypatch.setattr(TtPrefillRuntime, "_build_model", parent_build)
    with expect_error(ValueError, "no MTP predictor"):
        runtime._build_model({})
    parent_build.assert_not_called()


def test_the_block_kwargs_that_are_swept_through_are_genuinely_block_level():
    """A model-level knob landing in the block is a bug even when the block happens to accept it."""
    passed = _model_cls_kwargs()
    named = set(inspect.signature(TtKimiK3Transformer.__init__).parameters)
    swept = {k for k in passed if k not in named}
    # Expert dtypes, the dispatch capacity factor and the routing/semaphore placement are per-block
    # MoE construction arguments. Anything else appearing here is a model-level knob that has drifted
    # into the block path and should be named on the transformer instead.
    expected = {
        "dispatch_buffer_capacity_factor",
        "routed_expert_activations_dtype",
        "routed_expert_weights_dtype",
        "shared_expert_activations_dtype",
        "shared_expert_weights_dtype",
        "routing_use_l1_small_for_semaphores",
        "overlap_shared_expert_with_dispatch",
    }
    assert swept == expected, (
        f"the set of kwargs forwarded to TtKimiK3Block changed: +{sorted(swept - expected)} "
        f"-{sorted(expected - swept)}. If the addition is model-level, name it on the transformer."
    )


@pytest.mark.parametrize("sp_axis,tp_axis", [(0, 1), (1, 0)])
def test_kda_construction_passes_global_and_local_geometry(monkeypatch, sp_axis, tp_axis):
    constructor = create_autospec(ttKDA)
    constructor.return_value.convolution_page_channels = KDA_CONVOLUTION_PAGE_CHANNELS
    monkeypatch.setattr("models.demos.deepseek_v3_d_p.tt.kda.kda.ttKDA", constructor)
    monkeypatch.setattr("models.demos.deepseek_v3_d_p.tt.tt_ccl.get_tt_ccl", lambda device: None)
    build_attention(
        SimpleNamespace(shape=(8, 4)),
        None,
        None,
        {},
        1,
        SimpleNamespace(is_mla=lambda layer: False),
        seq_len=2560,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        topology=(ttnn.Topology.Linear, ttnn.Topology.Linear),
    )
    kwargs = constructor.call_args.kwargs
    assert kwargs["active_seq_len"] == 2560
    assert kwargs["program_config"].convolution_page_channels == KDA_CONVOLUTION_PAGE_CHANNELS
    # SP8 owns 320 rows (10 chunks); SP4 owns 640 rows (20 chunks).
    assert kwargs["program_config"].recurrence.summary_group_chunks == (10 if sp_axis == 0 else 20)


@pytest.mark.parametrize("host_start,host_end", [(None, None), (672, None), (672, 704), (0, 32)])
def test_build_kda_scalars_uploads_eager_slot_and_bounds_once(monkeypatch, host_start, host_end):
    device = object()
    monkeypatch.setattr(attention_module, "device_scalar", lambda on, value: (on, value))

    scalars, owned = build_kda_scalars(device, None, 2, host_start, host_end)

    # Slot first, then the start; an absent end uploads nothing and stays `None` (full capacity).
    start = (device, 0 if host_start is None else host_start)
    end = None if host_end is None else (device, host_end)
    assert isinstance(scalars, KdaChunkScalars)
    assert scalars == KdaChunkScalars((device, 2), start, end)
    assert owned == tuple(scalar for scalar in scalars if scalar is not None)


def test_build_kda_scalars_borrows_traced_metadata(monkeypatch):
    upload = Mock()
    monkeypatch.setattr(attention_module, "device_scalar", upload)
    slot, start, end = object(), object(), object()
    # The host ints are placeholders under trace; the metadata tensors are what replay rewrites, borrowed.
    scalars, owned = build_kda_scalars(object(), (slot, start, end, object()), 7, 999, 999)
    assert scalars == KdaChunkScalars(slot, start, end)
    assert owned == ()
    upload.assert_not_called()


@pytest.mark.parametrize("start,end", [(1, 64), (33, 64), (-32, 32), (32, 32), (64, 32), (32, 31)])
def test_build_kda_scalars_rejects_invalid_host_bounds_before_upload(monkeypatch, expect_error, start, end):
    upload = Mock()
    monkeypatch.setattr(attention_module, "device_scalar", upload)
    with expect_error(ValueError, "K3 KDA"):
        build_kda_scalars(object(), None, 0, start, end)
    upload.assert_not_called()


def test_kda_forward_runs_the_whole_layer_on_its_slot_in_the_slabs(monkeypatch):
    kdas = [create_autospec(ttKDA, instance=True) for _ in range(2)]
    for kda in kdas:
        kda.convolution_page_channels = KDA_CONVOLUTION_PAGE_CHANNELS
        kda.forward.return_value = (object(), None)
    slot, start, end = object(), object(), object()
    from_torch, deallocate, gather = Mock(), Mock(), Mock()
    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "deallocate", deallocate)
    monkeypatch.setattr(ttnn, "all_gather", gather)
    monkeypatch.setattr(ttnn, "squeeze", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(ttnn, "unsqueeze", lambda tensor, **kwargs: tensor)
    slabs = Mock()
    # The host ints are MLA's; a KDA layer reads only the scalars and the slabs, the same objects on every layer.
    ctx = K3AttnContext(
        actual_start=5120,
        actual_end=10240,
        cache_user_id=3,
        kda_scalars=KdaChunkScalars(slot, start, end),
        kda_states=slabs,
    )
    hidden = object()
    for layer_idx, kda in enumerate(kdas):
        output = kda.forward.return_value[0]
        assert TtK3KdaAttention(kda, layer_idx, 1, 1, ttnn.Topology.Linear).forward(hidden, ctx) is output
        # The whole per-chunk layer is ttKDA.forward on the slot's slab entry, selected by the device slot, so a
        # capture never pins the slot it was recorded on.
        kda.forward.assert_called_once_with(hidden, slabs.entry.return_value, start, end)
        slabs.entry.assert_called_with(slot, layer_idx)
    # The wrapper itself moves nothing: no gather, no slab copy, no free of the block's input.
    gather.assert_not_called()
    deallocate.assert_not_called()
    from_torch.assert_not_called()


def _mocked_layer(monkeypatch, tensor_parallel_size: int):
    """A ttKDA whose device work is mocks recording one call order."""
    calls = Mock()
    layer = object.__new__(ttKDA)
    layer.tensor_parallel_size = tensor_parallel_size
    layer.tensor_parallel_axis = 1
    layer.config = SimpleNamespace(hidden_size=256, q_dim=1, k_dim=1, v_dim=1, num_heads=2, norm_eps=1e-6)
    layer.weights = SimpleNamespace(convolution_taps=(), norm=object())
    layer.gated_rms_output_dtype = layer.norm_output_memory_config = None
    layer.convolution_page_channels = KDA_CONVOLUTION_PAGE_CHANNELS
    layer.kda_compute_config = object()
    for name in ("qkv_convolution_program_config", "qkv_memory_config", "sequence_parallel_axis"):
        setattr(layer, name, None)
    layer._history_exchange = SimpleNamespace(buffer=None, semaphores=None)
    layer._tp_gather_kwargs = None
    layer._validate_forward = Mock()
    layer._validate_runtime_bound = Mock()
    projected = SimpleNamespace(projected=object(), decay_rank_offset=0, output_gate_offset=0, beta_offset=0)
    layer._project_inputs = calls.project_inputs
    calls.project_inputs.return_value = projected
    layer._compute_decay = calls.compute_decay
    layer.recurrence = calls.recurrence
    calls.recurrence.return_value = SimpleNamespace(output=object(), final_state="final_recurrent")
    calls.recurrence.applies_output_norm = False
    layer._kda_rms_norm = calls.norm
    layer._project_output = calls.project_output
    calls.project_output.return_value = "output"
    monkeypatch.setattr(ttnn, "all_gather", calls.all_gather)
    calls.all_gather.return_value = "gathered"
    monkeypatch.setattr(ttnn, "deallocate", calls.deallocate)
    monkeypatch.setattr(ttnn.experimental.kda, "qkv_causal_conv1d_silu_with_carry", calls.conv)
    calls.conv.return_value = (object(), object(), object(), "new_convolution")
    return layer, calls


def test_kda_layer_functional_path_skips_the_gather_on_a_full_width_input(monkeypatch):
    layer, calls = _mocked_layer(monkeypatch, tensor_parallel_size=4)
    full = SimpleNamespace(shape=(1, 64, 256))
    output, state = layer.forward(full, Mock(), object())
    calls.all_gather.assert_not_called()
    calls.project_inputs.assert_called_once_with(full)
    assert output == "output" and (state.recurrent, state.convolution) == ("final_recurrent", "new_convolution")


def test_kda_layer_with_sequence_parallelism_leaves_the_norm_to_the_relay(monkeypatch):
    layer, calls = _mocked_layer(monkeypatch, tensor_parallel_size=4)
    calls.recurrence.applies_output_norm = True
    output, _ = layer.forward(SimpleNamespace(shape=(1, 64, 256)), Mock(), object())
    calls.norm.assert_not_called()
    output_norm = calls.recurrence.call_args.kwargs["output_norm"]
    assert output_norm.gate is calls.project_inputs.return_value.projected and output_norm.num_heads == 2
    calls.project_output.assert_called_once_with(calls.recurrence.return_value.output)
    assert output == "output"


def test_kda_attention_rejects_a_layer_without_the_contract_convolution_pages(expect_error):
    # A default-config layer keeps its convolution tail token-paged.
    kda = Mock(convolution_page_channels=KDAProgramConfig().convolution_page_channels)
    with expect_error(ValueError, f"convolution_page_channels={KDA_CONVOLUTION_PAGE_CHANNELS}"):
        TtK3KdaAttention(kda, 1, 1, 1, ttnn.Topology.Linear)


@pytest.mark.parametrize(
    "ctx,message",
    [
        (K3AttnContext(actual_start=0, actual_end=64, kda_states=Mock()), "kda_scalars"),
        (
            K3AttnContext(actual_start=0, actual_end=64, kda_scalars=KdaChunkScalars(object(), object(), None)),
            "state slabs",
        ),
    ],
    ids=["no-scalars", "no-slabs"],
)
def test_kda_forward_requires_scalars_and_slabs_before_device_work(monkeypatch, expect_error, ctx, message):
    gather = Mock()
    monkeypatch.setattr(ttnn, "all_gather", gather)
    attention = TtK3KdaAttention(
        Mock(convolution_page_channels=KDA_CONVOLUTION_PAGE_CHANNELS), 1, 1, 1, ttnn.Topology.Linear
    )
    with expect_error(ValueError, message):
        attention.forward(object(), ctx)
    gather.assert_not_called()


def test_runtime_rejects_unaligned_start_before_replay(monkeypatch, expect_error):
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.runtime import TtKimiK3Runtime
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    runtime = object.__new__(TtKimiK3Runtime)
    parent_forward = create_autospec(TtPrefillRuntime.prefill_chunk)
    monkeypatch.setattr(TtPrefillRuntime, "prefill_chunk", parent_forward)
    # Positional arguments exercise the same binding used by the runner.
    with expect_error(ValueError, "32-token aligned"):
        runtime.prefill_chunk(object(), object(), 0, 33, 64)
    parent_forward.assert_not_called()


def test_runtime_accepts_unaligned_end(monkeypatch):
    """A prompt's last chunk ends at its real length, which need not be 32-aligned."""
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.runtime import TtKimiK3Runtime
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    runtime = object.__new__(TtKimiK3Runtime)
    parent_forward = create_autospec(TtPrefillRuntime.prefill_chunk)
    monkeypatch.setattr(TtPrefillRuntime, "prefill_chunk", parent_forward)
    runtime.prefill_chunk(object(), object(), 0, 0, 33)
    parent_forward.assert_called_once()


def _runtime_serving(*, num_users: int, kda_layer_ids: tuple[int, ...], has_kda_layers: bool):
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.runtime import TtKimiK3Runtime

    runtime = object.__new__(TtKimiK3Runtime)
    runtime.config = SimpleNamespace(num_users=num_users)
    schedule = SimpleNamespace(rank_kda_layer_ids=lambda: kda_layer_ids)
    runtime.model = SimpleNamespace(has_kda_layers=has_kda_layers, kda_states=None, schedule=schedule)
    return runtime


@pytest.mark.parametrize(
    "has_slabs,has_kda_layers",
    [(False, False), (True, True), (False, True), (True, False)],
    ids=["no-kda-rank", "kda-rank", "slabs-missing", "slabs-without-kda-layers"],
)
def test_compile_hands_the_slabs_to_the_model_only_when_the_rank_has_kda_layers(
    monkeypatch, expect_error, has_slabs, has_kda_layers
):
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    slabs = SimpleNamespace(num_slots=2, layer_ids=(0, 3)) if has_slabs else None
    runtime = _runtime_serving(num_users=2, kda_layer_ids=(0, 3), has_kda_layers=has_kda_layers)
    parent_compile = Mock()
    monkeypatch.setattr(TtPrefillRuntime, "compile", parent_compile)
    kv_caches = SimpleNamespace(kda_states=slabs)
    if has_slabs == has_kda_layers:
        runtime.compile(kv_caches)
        assert runtime.model.kda_states is slabs
        parent_compile.assert_called_once_with(kv_caches)
    else:
        with expect_error(RuntimeError, "KDA state is split"):
            runtime.compile(kv_caches)
        parent_compile.assert_not_called()


@pytest.mark.parametrize(
    "num_slots,layer_ids", [(1, (0, 3)), (2, (0,)), (2, (0, 4))], ids=["fewer-slots", "fewer-layers", "other-layers"]
)
def test_compile_rejects_slabs_that_do_not_match_the_runtime(monkeypatch, expect_error, num_slots, layer_ids):
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    # An out-of-range slot is a silent no-op on device, so a slot count or layer set that disagrees
    # with the runtime must fail here, before the first forward.
    runtime = _runtime_serving(num_users=2, kda_layer_ids=(0, 3), has_kda_layers=True)
    parent_compile = Mock()
    monkeypatch.setattr(TtPrefillRuntime, "compile", parent_compile)
    with expect_error(RuntimeError, "KDA slabs"):
        runtime.compile(SimpleNamespace(kda_states=SimpleNamespace(num_slots=num_slots, layer_ids=layer_ids)))
    assert runtime.model.kda_states is None
    parent_compile.assert_not_called()
