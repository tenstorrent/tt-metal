# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The seam between the shared prefill runtime and Kimi-K3's own transformer.

`TtPrefillRuntime._build_model` calls `self.MODEL_CLS(...)` with a fixed keyword list. Kimi-K3's
transformer names the model-level knobs it understands and forwards the rest to `TtKimiK3Block` as
`**block_kwargs`, so a knob added upstream and NOT named here does not fail at the seam -- it
silently becomes a block argument and raises `TypeError` on the first layer, after mesh bringup and
weight load. That is a long way to travel for a signature mismatch, and it is only reachable through
`build_runtime`, which no eager test exercises.

`tp_shard_kv` did exactly this: it arrived with GLM-5.2's 2D KV sharding (#51968), after the branch's
merge-base, and broke every runner construction until it was named.

Hardware-free: this reads the call site's keyword list out of the source and binds it against the two
signatures. No mesh, no weights, no checkpoint.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.demos.deepseek_v3_d_p.tt.kimi_k3.attention import K3AttnContext, TtK3KdaAttention, build_attention
from models.demos.deepseek_v3_d_p.tt.kimi_k3.block import TtKimiK3Block
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer

_RUNTIME_SRC = Path(__file__).parents[2] / "tt" / "tt_prefill_runtime.py"


def _model_cls_kwargs() -> list[str]:
    """The keywords `_build_model` passes to `MODEL_CLS`, read off the call site."""
    src = _RUNTIME_SRC.read_text(encoding="utf-8")
    start = src.index("self.MODEL_CLS(")
    call = src[start : src.index("\n        )\n", start)]
    names = re.findall(r"^\s{12}(\w+)=", call, re.M)
    assert names, "could not parse the MODEL_CLS call site; the test needs updating, not deleting"
    return names


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
    # SP8 owns 320 rows (10 chunks); SP4 owns 640 rows (20 chunks).
    assert kwargs["program_config"].recurrence.summary_group_chunks == (10 if sp_axis == 0 else 20)


@pytest.mark.parametrize(
    "host_start,host_end,use_metadata",
    [(None, None, False), (672, None, False), (672, 704, False), (0, 32, False), (0, 1024, True)],
)
def test_kda_forward_uses_live_metadata_or_eager_position(monkeypatch, host_start, host_end, use_metadata):
    kda = create_autospec(ttKDA, instance=True)
    kda.device = object()
    output, new_state, scalar, end_scalar = object(), object(), object(), object()
    kda.forward.return_value = output, new_state
    states = Mock()
    from_torch = Mock(side_effect=[scalar, end_scalar])
    mapper = object()
    deallocate = Mock()
    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda device: mapper)
    monkeypatch.setattr(ttnn, "deallocate", deallocate)
    monkeypatch.setattr(ttnn, "all_gather", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(ttnn, "squeeze", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(ttnn, "unsqueeze", lambda tensor, **kwargs: tensor)
    attention = TtK3KdaAttention(kda, 1, 1, 1, ttnn.Topology.Linear, states)
    metadata = (object(), scalar, end_scalar) if use_metadata else None
    hidden = object()
    assert (
        attention.forward(
            hidden, K3AttnContext(actual_start=host_start, actual_end=host_end, cache_user_id=2, metadata=metadata)
        )
        is output
    )
    states.read.assert_called_once_with(1, 2)
    kda.forward.assert_called_once_with(
        hidden,
        states.read.return_value,
        actual_start=scalar,
        actual_end=end_scalar if use_metadata or host_end is not None else None,
    )
    states.commit.assert_called_once_with(1, new_state, 2)
    if use_metadata:
        from_torch.assert_not_called()
        assert all(call.args not in ((scalar,), (end_scalar,)) for call in deallocate.call_args_list)
    else:
        assert from_torch.call_count == (1 if host_end is None else 2)
        assert torch.equal(
            from_torch.call_args_list[0].args[0],
            torch.tensor([0 if host_start is None else host_start], dtype=torch.int64),
        )
        if host_end is not None:
            assert torch.equal(from_torch.call_args_list[1].args[0], torch.tensor([host_end], dtype=torch.int64))
            assert any(call.args == (end_scalar,) for call in deallocate.call_args_list)
        assert from_torch.call_args.kwargs == {
            "dtype": ttnn.uint32,
            "layout": ttnn.ROW_MAJOR_LAYOUT,
            "device": kda.device,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "mesh_mapper": mapper,
        }
        assert any(call.args == (scalar,) for call in deallocate.call_args_list)


@pytest.mark.parametrize("start,end", [(1, 64), (0, 33), (-32, 32), (32, 32), (64, 32)])
def test_kda_rejects_invalid_host_bounds_before_device_work(monkeypatch, expect_error, start, end):
    gather = Mock()
    monkeypatch.setattr(ttnn, "all_gather", gather)
    attention = TtK3KdaAttention(Mock(), 1, 1, 1, ttnn.Topology.Linear, Mock())
    with expect_error(ValueError, "K3 KDA"):
        attention.forward(object(), K3AttnContext(actual_start=start, actual_end=end))
    gather.assert_not_called()


def test_runtime_rejects_unaligned_end_before_reset_or_replay(monkeypatch, expect_error):
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.runtime import TtKimiK3Runtime
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    runtime = object.__new__(TtKimiK3Runtime)
    runtime.model = SimpleNamespace(kda_states=Mock())
    parent_forward = create_autospec(TtPrefillRuntime.prefill_chunk)
    monkeypatch.setattr(TtPrefillRuntime, "prefill_chunk", parent_forward)
    # Positional arguments exercise the same binding used by the runner.
    with expect_error(ValueError, "32-token aligned"):
        runtime.prefill_chunk(object(), object(), 0, 0, 33)
    runtime.model.kda_states.reset.assert_not_called()
    parent_forward.assert_not_called()
