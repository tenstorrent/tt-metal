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
    """The eager and traced chunk paths both call `self.model.forward`; neither is reached by an eager test."""
    for call in _call_sites("self.model.forward", expected=2):
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
        kda_scratch=object(),
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


def test_kda_forward_runs_the_whole_layer_on_its_slot_through_the_shared_scratch(monkeypatch):
    kdas = [create_autospec(ttKDA, instance=True) for _ in range(2)]
    for kda in kdas:
        kda.convolution_page_channels = KDA_CONVOLUTION_PAGE_CHANNELS
        kda.forward_slot.return_value = object()
    slot, start, end = object(), object(), object()
    from_torch, deallocate, gather = Mock(), Mock(), Mock()
    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "deallocate", deallocate)
    monkeypatch.setattr(ttnn, "all_gather", gather)
    monkeypatch.setattr(ttnn, "squeeze", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(ttnn, "unsqueeze", lambda tensor, **kwargs: tensor)
    slabs, scratch = Mock(), object()
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
        output = kda.forward_slot.return_value
        assert TtK3KdaAttention(kda, layer_idx, 1, 1, ttnn.Topology.Linear, scratch).forward(hidden, ctx) is output
        # The whole per-chunk layer (import, TP all-gather, KDA, export) is ttKDA.forward_slot, on the TP-sharded
        # input and the shared scratch. The carry is selected by the device slot, so a capture never pins the slot
        # it was recorded on.
        kda.forward_slot.assert_called_once_with(
            hidden, scratch, slabs, slot, layer_idx, actual_start=start, actual_end=end
        )
        kda.forward.assert_not_called()
    # The wrapper itself moves nothing: no gather, no slab copy, no free of the block's input.
    gather.assert_not_called()
    slabs.import_layer.assert_not_called()
    slabs.export_layer.assert_not_called()
    deallocate.assert_not_called()
    from_torch.assert_not_called()


def _mocked_layer(monkeypatch, tensor_parallel_size: int):
    """A ttKDA whose device work is mocks recording one call order."""
    calls = Mock()
    layer = object.__new__(ttKDA)
    layer.tensor_parallel_size = tensor_parallel_size
    layer.tensor_parallel_axis = 1
    layer.config = SimpleNamespace(hidden_size=256, q_dim=1, k_dim=1, v_dim=1)
    layer.weights = SimpleNamespace(convolution_taps=())
    layer.convolution_page_channels = KDA_CONVOLUTION_PAGE_CHANNELS
    for name in ("qkv_convolution_program_config", "qkv_memory_config", "sequence_parallel_axis"):
        setattr(layer, name, None)
    layer._history_exchange = SimpleNamespace(buffer=None, semaphores=None)
    layer._validate_forward = Mock()
    layer._validate_runtime_bound = Mock()
    projected = SimpleNamespace(projected=object(), decay_rank_offset=0, output_gate_offset=0, beta_offset=0)
    layer._project_inputs = calls.project_inputs
    calls.project_inputs.return_value = projected
    layer._compute_decay = calls.compute_decay
    layer.recurrence = calls.recurrence
    calls.recurrence.return_value = SimpleNamespace(output=object(), final_state="final_recurrent")
    layer._kda_rms_norm = calls.norm
    layer._project_output = calls.project_output
    calls.project_output.return_value = "output"
    monkeypatch.setattr(ttnn, "all_gather", calls.all_gather)
    calls.all_gather.return_value = "gathered"
    monkeypatch.setattr(ttnn, "deallocate", calls.deallocate)
    monkeypatch.setattr(ttnn.experimental.kda, "qkv_causal_conv1d_silu_with_carry", calls.conv)
    calls.conv.return_value = (object(), object(), object(), "new_convolution")
    return layer, calls


def test_kda_forward_slot_imports_gathers_runs_and_exports_in_order(monkeypatch):
    layer, calls = _mocked_layer(monkeypatch, tensor_parallel_size=4)
    slabs, slot, scratch, start, end = Mock(), object(), Mock(), object(), object()
    calls.attach_mock(slabs, "slabs")
    tp_shard = SimpleNamespace(shape=(1, 64, 64))  # 256 / TP 4: the model's TP-sharded residual
    assert layer.forward_slot(tp_shard, scratch, slabs, slot, 7, start, end) == "output"
    order = [c[0] for c in calls.mock_calls if not c[0].startswith("project_inputs().")]
    # import -> TP all-gather -> projection -> ... -> output projection -> export, then free the replacement carry.
    assert order.index("slabs.import_layer") < order.index("all_gather") < order.index("project_inputs")
    assert order.index("project_output") < order.index("slabs.export_layer")
    calls.slabs.import_layer.assert_called_once_with(scratch, slot, 7, zero_fill_at_zero=start)
    calls.all_gather.assert_called_once_with(tp_shard, dim=-1, cluster_axis=1)
    calls.project_inputs.assert_called_once_with("gathered")
    exported = calls.slabs.export_layer.call_args.args
    assert (exported[0].recurrent, exported[0].convolution, exported[1:]) == (
        "final_recurrent",
        "new_convolution",
        (slot, 7),
    )
    freed = [c.args[0] for c in calls.deallocate.call_args_list]
    assert "gathered" in freed and freed[-2:] == ["final_recurrent", "new_convolution"]


def test_kda_layer_functional_path_skips_the_gather_on_a_full_width_input(monkeypatch):
    layer, calls = _mocked_layer(monkeypatch, tensor_parallel_size=4)
    full = SimpleNamespace(shape=(1, 64, 256))
    output, state = layer.forward(full, Mock(), object())
    calls.all_gather.assert_not_called()
    calls.project_inputs.assert_called_once_with(full)
    assert output == "output" and (state.recurrent, state.convolution) == ("final_recurrent", "new_convolution")


def test_kda_attention_rejects_a_layer_without_the_contract_convolution_pages(expect_error):
    # A default-config layer keeps its convolution tail token-paged.
    kda = Mock(convolution_page_channels=KDAProgramConfig().convolution_page_channels)
    with expect_error(ValueError, f"convolution_page_channels={KDA_CONVOLUTION_PAGE_CHANNELS}"):
        TtK3KdaAttention(kda, 1, 1, 1, ttnn.Topology.Linear, object())


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
        Mock(convolution_page_channels=KDA_CONVOLUTION_PAGE_CHANNELS), 1, 1, 1, ttnn.Topology.Linear, object()
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


def _runtime_serving(*, num_users: int, kda_layer_ids: tuple[int, ...], scratch):
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.runtime import TtKimiK3Runtime

    runtime = object.__new__(TtKimiK3Runtime)
    runtime.config = SimpleNamespace(num_users=num_users)
    schedule = SimpleNamespace(rank_kda_layer_ids=lambda: kda_layer_ids)
    runtime.model = SimpleNamespace(kda_scratch=scratch, kda_states=None, schedule=schedule)
    return runtime


@pytest.mark.parametrize(
    "has_slabs,scratch",
    [(False, None), (True, object()), (False, object()), (True, None)],
    ids=["no-kda-rank", "kda-rank", "slabs-missing", "slabs-without-kda-layers"],
)
def test_compile_hands_the_slabs_to_the_model_only_when_the_rank_has_kda_layers(
    monkeypatch, expect_error, has_slabs, scratch
):
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    slabs = SimpleNamespace(num_slots=2, layer_ids=(0, 3)) if has_slabs else None
    runtime = _runtime_serving(num_users=2, kda_layer_ids=(0, 3), scratch=scratch)
    parent_compile = Mock()
    monkeypatch.setattr(TtPrefillRuntime, "compile", parent_compile)
    kv_caches = SimpleNamespace(kda_states=slabs)
    if has_slabs == (scratch is not None):
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
    runtime = _runtime_serving(num_users=2, kda_layer_ids=(0, 3), scratch=object())
    parent_compile = Mock()
    monkeypatch.setattr(TtPrefillRuntime, "compile", parent_compile)
    with expect_error(RuntimeError, "KDA slabs"):
        runtime.compile(SimpleNamespace(kda_states=SimpleNamespace(num_slots=num_slots, layer_ids=layer_ids)))
    assert runtime.model.kda_states is None
    parent_compile.assert_not_called()
