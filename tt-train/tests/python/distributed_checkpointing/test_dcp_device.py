# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Same-world round trips of ``ttml.distributed_checkpointing`` on the shared ``[1, 2]`` TP and FSDP fixtures.

Every parameter and optimizer moment must come back device for device, bit for bit, into a fresh model and
optimizer; the header and the optimizer scalars must come back; the FSDP gate must refuse a gathered module; a
strict load must name what is missing. Cross-world loads live in ``test_dcp_cross_world.py``, which needs no open
device of its own.

``tp_mesh`` and ``fsdp_mesh`` are both module-scoped and each reopens the device with its own axis names, so the two
groups must not interleave: the TP tests come first, the FSDP test last.
"""

import numpy as np
import pytest
import torch
import ttnn
from tp_block import TPBlock, fresh_tp, make_adamw, train_step

import ttml
import ttml.distributed_checkpointing as dcp
from ttml.fsdp import _ShardedState
from ttml.modules import AbstractModuleBase, LinearLayer, Parameter

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

DIM = 64  # tile-aligned per device after a 2-way split (32 rows / cols per shard)
NATIVE = ttml.autograd.PreferredPrecision.NATIVE


def _shards(tensor):
    return [ttnn.to_torch(s) for s in ttnn.get_device_tensors(tensor.get_value(NATIVE))]


def _model_shards(params):
    return {name: _shards(t) for name, t in params.items()}


def _state(optimizer):
    """``(tensor shards by key, scalars by key)`` of an optimizer's state dict."""
    shards, scalars = {}, {}

    def walk(node, path):
        if isinstance(node, ttml.NamedParameters):
            for name, tensor in node.items():
                shards["/".join(path) + ":" + name] = _shards(tensor)
        elif isinstance(node, dict):
            for key, child in node.items():
                walk(child, path + (str(key),))
        else:
            scalars["/".join(path)] = node

    walk(optimizer.get_state_dict(), ())
    return shards, scalars


def _assert_same_shards(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key in expected:
        assert len(actual[key]) == len(expected[key]), key
        for device, (a, e) in enumerate(zip(actual[key], expected[key])):
            assert torch.equal(a, e), f"{key} differs on device {device}"


def _assert_some_differ(actual: dict, expected: dict) -> None:
    assert any(
        not torch.equal(a, e) for key in expected for a, e in zip(actual[key], expected[key])
    ), "precondition: fresh state differs"


class _Stack(AbstractModuleBase):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.a = LinearLayer(dim, dim)
        self.b = LinearLayer(dim, dim)

    def forward(self, x):
        return self.b(self.a(x))


class _TPBlockPlus(TPBlock):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.extra = LinearLayer(dim, dim)


def test_tp_round_trip_is_bit_exact_on_the_same_world(tp_mesh, tmp_path):
    model, params, opt = fresh_tp(DIM)
    train_step(model, opt, DIM)
    before = _model_shards(params)
    before_state, before_scalars = _state(opt)
    assert before_scalars, "the optimizer state dict carries scalars (steps, lr, ...)"
    path = str(tmp_path / dcp.step_dir_name(1))

    result = dcp.save(path, model=model, optimizer=opt, header={"step": 1, "note": "tp"})

    assert dcp.is_complete(path) and dcp.find_latest(str(tmp_path)) == path
    assert result.keys == len(before) + len(before_state) and result.bytes_written > 0
    assert len(result.files) == 2, "one data file per device of the [1, 2] mesh"
    assert dcp.read_header(path) == {"step": 1, "note": "tp"}
    assert _state(opt)[0].keys() == before_state.keys()
    _assert_same_shards(_model_shards(params), before)  # the save left the live tensors alone

    model2, params2, opt2 = fresh_tp(DIM)
    _assert_some_differ(_model_shards(params2), before)
    loaded = dcp.load(path, model=model2, optimizer=opt2)

    assert loaded.header == {"step": 1, "note": "tp"}
    assert loaded.missing == [] and loaded.unexpected == []
    assert set(loaded.fast_path_keys) == set(f"model/{n}" for n in before) | set(
        f"optimizer/{k.replace(':', '/')}" for k in before_state
    )
    _assert_same_shards(_model_shards(params2), before)
    after_state, after_scalars = _state(opt2)
    _assert_same_shards(after_state, before_state)
    assert after_scalars == before_scalars


def test_a_step_after_the_load_matches_the_uninterrupted_run(tp_mesh, tmp_path):
    """The loaded optimizer behaves as the saved one: one more step from the checkpoint gives bit-identical
    parameters and moments to one more step of the run that never stopped. A fresh optimizer on the same
    parameters does not, so it is the restored moments and step count that make the step agree."""
    model, params, opt = fresh_tp(DIM)
    for _ in range(3):
        train_step(model, opt, DIM)
    path = str(tmp_path / dcp.step_dir_name(3))
    dcp.save(path, model=model, optimizer=opt, header={"step": 3})
    train_step(model, opt, DIM, input_scale=0.5)  # step 4 of the uninterrupted run, on a batch the resumed run repeats
    expected = _model_shards(params)
    expected_state, expected_scalars = _state(opt)

    model2, params2, opt2 = fresh_tp(DIM)
    dcp.load(path, model=model2, optimizer=opt2)
    train_step(model2, opt2, DIM, input_scale=0.5)
    _assert_same_shards(_model_shards(params2), expected)
    _assert_same_shards(_state(opt2)[0], expected_state)
    assert _state(opt2)[1] == expected_scalars

    model3, params3, opt3 = fresh_tp(DIM)
    dcp.load(path, model=model3, strict=False)  # the parameters alone: the optimizer starts from nothing
    train_step(model3, opt3, DIM, input_scale=0.5)
    _assert_some_differ(_model_shards(params3), expected)
    _assert_some_differ(_state(opt3)[0], expected_state)


def test_strict_load_names_the_mismatch(tp_mesh, tmp_path):
    model, params, opt = fresh_tp(DIM)
    path = str(tmp_path / "step_0000001")
    dcp.save(path, model=model, optimizer=opt)

    other = _TPBlockPlus(DIM)
    with pytest.raises(dcp.MissingKeyError):  # allow-pytest.raises: the fixture matches on device error text
        dcp.load(path, model=other)
    lenient = dcp.load(path, model=other, strict=False)
    assert lenient.missing and lenient.unexpected
    assert all(k.startswith("model/_TPBlockPlus/") for k in lenient.missing)
    # Only the roles passed count: with no optimizer, the checkpoint's optimizer entries are not "unexpected".
    assert all(k.startswith("model/TPBlock/") for k in lenient.unexpected)

    # The other way round: an optimizer passed against a model-only checkpoint has nothing to load.
    model_only = str(tmp_path / "step_0000002")
    dcp.save(model_only, model=model)
    model2, params2, opt2 = fresh_tp(DIM)
    with pytest.raises(dcp.MissingKeyError):  # allow-pytest.raises: the fixture matches on device error text
        dcp.load(model_only, model=model2, optimizer=opt2)
    partial = dcp.load(model_only, model=model2, optimizer=opt2, strict=False)
    assert partial.missing and all(k.startswith("optimizer/") for k in partial.missing)
    assert not partial.unexpected


def test_incomplete_directory_and_existing_directory_are_refused(tp_mesh, tmp_path):
    model, params, opt = fresh_tp(DIM)
    with pytest.raises(dcp.IncompleteCheckpoint):  # allow-pytest.raises: the fixture matches on device error text
        dcp.load(str(tmp_path / "nowhere"), model=model)
    path = str(tmp_path / "step_0000001")
    dcp.save(path, model=model)
    with pytest.raises(dcp.CheckpointError):  # allow-pytest.raises: the fixture matches on device error text
        dcp.save(path, model=model)
    dcp.save(path, model=model, overwrite=True)
    assert dcp.is_complete(path)


class _LoraQKV(AbstractModuleBase):
    def __init__(self, dim: int) -> None:
        super().__init__()
        from ttml.modules import ColumnParallelLinear
        from ttml.modules.lora import LoraColumnParallelLinear, LoraConfig

        base = ColumnParallelLinear(dim, 2 * dim, has_bias=False, fused_segments=(dim, dim // 2, dim // 2))
        self.qkv_linear = LoraColumnParallelLinear(base, LoraConfig(rank=8))

    def forward(self, x):
        return self.qkv_linear(x)


def test_lora_on_a_fused_layer_declares_the_rows_it_adds_to(tp_mesh, expect_error):
    from ttml.distributed_checkpointing._layouts import fused_segments
    from ttml.distributed_checkpointing._state_dict import collect_model
    from ttml.modules import ColumnParallelLinear

    infos = {lt.key.rsplit("/", 1)[-1]: lt.info for lt in collect_model(_LoraQKV(DIM))}
    segments = (DIM, DIM // 2, DIM // 2)
    assert infos["weight"].transform == fused_segments(2, segments, 2)
    assert infos["lora_B"].transform == fused_segments(2, segments, 2), "lora_B follows the base's per-shard rows"
    # A segment that does not split evenly over the TP shards is refused when the layer is built, not at the first save.
    with expect_error(ValueError, "must each divide"):
        ColumnParallelLinear(DIM, 96, has_bias=False, fused_segments=(63, 33))


def test_save_refuses_bad_state_before_writing_and_leaks_nothing(tp_mesh, tmp_path, monkeypatch, expect_error):
    import os

    from ttml.distributed_checkpointing import _save

    model, params, opt = fresh_tp(DIM)
    train_step(model, opt, DIM)
    with expect_error(dcp.CheckpointError, r"header at /step: a int64 has no JSON form"):
        dcp.save(str(tmp_path / "bad_header"), model=model, optimizer=opt, header={"step": np.int64(3)})
    assert os.listdir(tmp_path) == [], "refused before any staging directory existed"

    # A data file that cannot be created mid-way: the save fails, and every descriptor it opened is closed.
    open_fds = len(os.listdir("/proc/self/fd"))
    real_create = _save.create_file
    calls = []

    def failing_create(path, header, data_bytes):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("simulated EDQUOT")
        return real_create(path, header, data_bytes)

    monkeypatch.setattr(_save, "create_file", failing_create)
    with expect_error(OSError, "simulated EDQUOT"):
        dcp.save(str(tmp_path / "quota"), model=model, optimizer=opt)
    monkeypatch.setattr(_save, "create_file", real_create)
    assert len(os.listdir("/proc/self/fd")) == open_fds
    assert os.listdir(tmp_path) == [], "the staging directory is removed"


def test_a_truncated_data_file_is_refused_before_anything_is_placed(tp_mesh, tmp_path, expect_error):
    import glob
    import os

    model, params, opt = fresh_tp(DIM)
    train_step(model, opt, DIM)
    path = str(tmp_path / dcp.step_dir_name(1))
    dcp.save(path, model=model, optimizer=opt)
    victim = sorted(glob.glob(os.path.join(path, "data", "*.safetensors")))[0]
    os.truncate(victim, os.path.getsize(victim) - 1)

    model2, params2, opt2 = fresh_tp(DIM)
    before = _model_shards(params2)
    with expect_error(dcp.IncompleteCheckpoint, "truncated or corrupt"):
        dcp.load(path, model=model2, optimizer=opt2)
    _assert_same_shards(_model_shards(params2), before)


class _OddRows(AbstractModuleBase):
    """One ``[1, 1, 96, DIM]`` weight, split on dim 2 over the TP axis or replicated. Split over two devices, each
    holds 48 rows: a tile and a half, so every shard ends inside a tile."""

    def __init__(self, sharded: bool, seed: int) -> None:
        super().__init__()
        rows = np.random.default_rng(seed).standard_normal((1, 1, 96, DIM), dtype=np.float32)
        mapper = ttml.mesh().axis_mapper("tp", tdim=2) if sharded else None
        self.weight = Parameter(ttml.autograd.Tensor.from_numpy(rows, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper))

    def forward(self, x):
        return x


def test_shards_that_end_inside_a_tile_round_trip_and_reshard(tp_mesh, tmp_path):
    sharded = _OddRows(sharded=True, seed=0)
    saved = _model_shards(sharded.parameters())
    (pieces,) = saved.values()
    assert [p.shape[2] for p in pieces] == [48, 48], "precondition: a tile and a half per device"
    path = str(tmp_path / "sharded")
    dcp.save(path, model=sharded)

    same = _OddRows(sharded=True, seed=1)
    _assert_some_differ(_model_shards(same.parameters()), saved)
    assert dcp.load(path, model=same).fast_path_keys, "each device reads back its own chunk"
    _assert_same_shards(_model_shards(same.parameters()), saved)

    # Onto a replicated weight: every device gets all 96 rows, put back together from the two 48-row chunks.
    replicated = _OddRows(sharded=False, seed=2)
    assert not dcp.load(path, model=replicated).fast_path_keys
    (whole,) = _model_shards(replicated.parameters()).values()
    assert all(torch.equal(w, torch.cat(pieces, dim=2)) for w in whole)

    # And back: the replicated weight's checkpoint cut into 48-row shards.
    path = str(tmp_path / "replicated")
    dcp.save(path, model=replicated)
    resharded = _OddRows(sharded=True, seed=3)
    assert not dcp.load(path, model=resharded).fast_path_keys
    _assert_same_shards(_model_shards(resharded.parameters()), saved)


def _cpp_llama():
    """The C++ tensor-parallel Llama: its fused kv_linear cannot declare its layout from Python."""
    cfg = ttml.models.llama.CppLlamaConfig()
    cfg.num_heads, cfg.num_groups, cfg.embedding_dim, cfg.num_blocks = 2, 2, DIM, 1
    cfg.vocab_size, cfg.max_sequence_length = DIM, 32
    return ttml.models.distributed.llama.create_llama_model(cfg)


def test_a_cpp_model_round_trips_on_the_same_world_and_warns_that_only_this_tp_size_works(tp_mesh, tmp_path):
    model = _cpp_llama()
    saved = _model_shards(model.parameters())
    path = str(tmp_path / "ckpt")
    with pytest.warns(UserWarning, match="restores correctly only onto the same TP size"):
        dcp.save(path, model=model)
    fresh = _cpp_llama()
    _assert_some_differ(_model_shards(fresh.parameters()), saved)
    with pytest.warns(UserWarning, match="restores correctly only onto the same TP size"):
        dcp.load(path, model=fresh)
    _assert_same_shards(_model_shards(fresh.parameters()), saved)


def test_fsdp_round_trip_and_the_sharded_state_gate(fsdp_mesh, tmp_path):
    model = _Stack(DIM)
    ttml.fsdp.fully_shard(model)
    params = model.parameters()
    opt = make_adamw(params)
    train_step(model, opt, DIM)
    before = _model_shards(params)
    before_state, before_scalars = _state(opt)
    path = str(tmp_path / dcp.step_dir_name(1))

    dcp.save(path, model=model, optimizer=opt, header={"step": 1})

    model2 = _Stack(DIM)
    ttml.fsdp.fully_shard(model2)
    params2 = model2.parameters()
    opt2 = make_adamw(params2)
    _assert_some_differ(_model_shards(params2), before)
    loaded = dcp.load(path, model=model2, optimizer=opt2)
    assert loaded.header == {"step": 1} and loaded.fast_path_keys
    _assert_same_shards(_model_shards(params2), before)
    _assert_same_shards(_state(opt2)[0], before_state)
    assert _state(opt2)[1] == before_scalars

    # One more step from the checkpoint equals one more step of the run that never stopped: the sharded moments
    # and the step count feed the update kernel as they did before the save.
    train_step(model, opt, DIM, input_scale=0.5)
    train_step(model2, opt2, DIM, input_scale=0.5)
    _assert_same_shards(_model_shards(params2), _model_shards(params))
    _assert_same_shards(_state(opt2)[0], _state(opt)[0])
    assert _state(opt2)[1] == _state(opt)[1]

    # A module left gathered by a forward with reshard_after_forward=False is refused unless resharded first.
    model3 = _Stack(DIM)
    ttml.fsdp.fully_shard(model3, reshard_after_forward=False)
    params3 = model3.parameters()
    opt3 = make_adamw(params3)
    train_step(model3, opt3, DIM)
    x = ttml.autograd.Tensor.from_numpy(
        np.zeros((1, 1, DIM, DIM), dtype=np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )
    model3(x)  # leaves the module UNSHARDED
    assert model3._fsdp_state.sharded_state == _ShardedState.UNSHARDED, "precondition"
    with pytest.raises(dcp.FSDPStateError):  # allow-pytest.raises: the fixture matches on device error text
        dcp.save(str(tmp_path / "gathered"), model=model3, optimizer=opt3, reshard_fsdp=False)
    dcp.save(str(tmp_path / "resharded"), model=model3, optimizer=opt3)
    assert model3._fsdp_state.sharded_state == _ShardedState.SHARDED
    ttml.autograd.AutoContext.get_instance().reset_graph()
