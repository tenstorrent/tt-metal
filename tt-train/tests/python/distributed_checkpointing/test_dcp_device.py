# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Same-world round trips of ``ttml.distributed_checkpointing`` on the shared ``[1, 2]`` TP and FSDP fixtures.

Every parameter and optimizer moment must come back device for device, bit for bit, into a fresh model and
optimizer; the header and the optimizer scalars must come back; the scheduler, the dataloader and every random
generator must resume where the save left them; a gathered FSDP module is resharded first; a strict load
must name what is missing. Cross-world loads live in ``test_dcp_cross_world.py``, which needs no open
device of its own.

``tp_mesh`` and ``fsdp_mesh`` are both module-scoped and each reopens the device with its own axis names, so the two
groups must not interleave: the TP tests come first, the FSDP test last.
"""

import glob
import os
import random
import secrets
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
import ttnn
from dcp_oracle import MISREAD_RNG, WRONG_LENGTH_RNG, Loader, advance_generators, damage_stored_rng, host_draws
from dcp_shards import NATIVE, assert_same_shards, assert_some_differ, model_shards, optimizer_shards, tensor_shards
from tp_block import TPBlock, fresh_tp, make_adamw, train_step

import ttml
import ttml.distributed_checkpointing as dcp
from ttml.common.schedulers import CosineAnnealingScheduler
from ttml.distributed_checkpointing import _load, _save
from ttml.distributed_checkpointing._index import staging_dir
from ttml.distributed_checkpointing._layouts import Pad, fused_segments
from ttml.distributed_checkpointing._state_dict import collect_model
from ttml.fsdp import _ShardedState
from ttml.modules import AbstractModuleBase, Buffer, ColumnParallelLinear, LinearLayer, ModuleList, Parameter
from ttml.modules.lora import LoraColumnParallelLinear, LoraConfig, LoraLinear

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

DIM = 64  # tile-aligned per device after a 2-way split (32 rows / cols per shard)


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
    before = model_shards(params)
    before_state, before_scalars = optimizer_shards(opt)
    assert before_scalars, "the optimizer state dict carries scalars (steps, lr, ...)"
    path = str(tmp_path / dcp.step_dir_name(1))

    result = dcp.save(path, model=model, optimizer=opt, header={"step": 1, "note": "tp"})

    assert dcp.is_complete(path) and dcp.find_latest(str(tmp_path)) == path
    assert result.keys == len(before) + len(before_state) and result.bytes_written > 0
    assert len(result.files) == 2, "one data file per device of the [1, 2] mesh"
    assert dcp.read_header(path) == {"step": 1, "note": "tp"}
    assert optimizer_shards(opt)[0].keys() == before_state.keys()
    assert_same_shards(model_shards(params), before)  # the save left the live tensors alone

    model2, params2, opt2 = fresh_tp(DIM)
    assert_some_differ(model_shards(params2), before)
    loaded = dcp.load(path, model=model2, optimizer=opt2)

    assert loaded.header == {"step": 1, "note": "tp"}
    assert loaded.missing == [] and loaded.unexpected == []
    assert set(loaded.fast_path_keys) == set(f"model/{n}" for n in before) | set(
        f"optimizer/{k.replace(':', '/')}" for k in before_state
    )
    assert_same_shards(model_shards(params2), before)
    after_state, after_scalars = optimizer_shards(opt2)
    assert_same_shards(after_state, before_state)
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
    expected = model_shards(params)
    expected_state, expected_scalars = optimizer_shards(opt)

    model2, params2, opt2 = fresh_tp(DIM)
    dcp.load(path, model=model2, optimizer=opt2)
    train_step(model2, opt2, DIM, input_scale=0.5)
    assert_same_shards(model_shards(params2), expected)
    assert_same_shards(optimizer_shards(opt2)[0], expected_state)
    assert optimizer_shards(opt2)[1] == expected_scalars

    model3, params3, opt3 = fresh_tp(DIM)
    # The parameters alone, strictly: a load without an optimizer does not count the optimizer entries as unexpected.
    assert dcp.load(path, model=model3).unexpected == []
    train_step(model3, opt3, DIM, input_scale=0.5)
    assert_some_differ(model_shards(params3), expected)
    assert_some_differ(optimizer_shards(opt3)[0], expected_state)


class _Bias(AbstractModuleBase):
    def __init__(self, values: np.ndarray) -> None:
        super().__init__()
        self.bias = Buffer(ttml.autograd.Tensor.from_numpy(values, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16))


class _WithBuffers(AbstractModuleBase):
    """A layer plus state a step updates outside the optimizer, like a MoE's load-balancing bias, reachable under two
    names, and a second such buffer in a module a ModuleList holds, as a MoE's blocks are."""

    def __init__(self, dim: int, seed: int) -> None:
        super().__init__()
        self.linear = LinearLayer(dim, dim, has_bias=False)
        values = np.random.default_rng(seed).standard_normal((2, 1, 1, dim)).astype(np.float32)
        self.expert_bias = Buffer(ttml.autograd.Tensor.from_numpy(values[:1], ttnn.Layout.TILE, ttnn.DataType.BFLOAT16))
        self.alias = self.expert_bias
        self.blocks = ModuleList([_Bias(values[1:])])

    def forward(self, x):
        return self.linear(x)


def test_buffers_are_saved_and_loaded(tp_mesh, tmp_path):
    model = _WithBuffers(DIM, seed=1)
    path = str(tmp_path / "step_0000001")
    dcp.save(path, model=model)
    entries = dcp.read_index(path).entries
    entry = entries["model/_WithBuffers/expert_bias"]
    assert entry.role == "buffer" and entry.live_keys == ("model/_WithBuffers/alias", "model/_WithBuffers/expert_bias")
    assert entries["model/_WithBuffers/blocks/0/bias"].role == "buffer", "a buffer under a ModuleList is saved"

    def buffers(m):
        return {"b": tensor_shards(m.expert_bias.tensor), "listed": tensor_shards(m.blocks[0].bias.tensor)}

    fresh = _WithBuffers(DIM, seed=2)
    assert_some_differ({"b": tensor_shards(fresh.expert_bias.tensor)}, {"b": tensor_shards(model.expert_bias.tensor)})
    assert_some_differ(
        {"b": tensor_shards(fresh.blocks[0].bias.tensor)}, {"b": tensor_shards(model.blocks[0].bias.tensor)}
    )
    loaded = dcp.load(path, model=fresh)
    assert loaded.missing == [] and loaded.unexpected == []
    assert_same_shards(buffers(fresh), buffers(model))
    assert_same_shards(model_shards(fresh.parameters()), model_shards(model.parameters()))


def test_a_moment_laid_out_unlike_its_parameter_is_refused(tp_mesh, tmp_path, expect_error):
    """A moment with its TP-sharded parameter's per-device shape but labelled replicated would be saved as one device's
    piece and loaded onto every device; save refuses it by its global shape."""
    model, params, opt = fresh_tp(DIM)
    train_step(model, opt, DIM)
    key = next(k for k in params if k.endswith("up/weight"))

    def moments(node):
        if isinstance(node, ttml.NamedParameters):
            yield node
        elif isinstance(node, dict):
            for child in node.values():
                yield from moments(child)

    moment = next(m for m in moments(opt.get_state_dict()) if key in m)[key]
    value = moment.get_value()
    replicated = ttml.autograd.Tensor.from_numpy(
        np.zeros(tuple(int(n) for n in value.shape), dtype=np.float32), ttnn.Layout.TILE, value.dtype
    )
    moment.set_value(replicated.get_value())
    with expect_error(dcp.CheckpointError, r"over the mesh but its parameter .* it is not sharded as its parameter is"):
        dcp.save(str(tmp_path / "step_0000001"), model=model, optimizer=opt)
    assert not (tmp_path / "step_0000001").exists()


def test_threads_sizes_the_writers_and_the_reader_pool(tp_mesh, tmp_path, monkeypatch):
    """Save starts one writer thread per data file, or ``threads`` of them sharing the files; load reads with a pool of
    ``threads`` (32 by default). Two devices write two files."""
    model, params, opt = fresh_tp(DIM)
    train_step(model, opt, DIM)
    writers, pools = [], []
    real_writer = _save._writer

    def counting_writer(*args):
        writers.append(threading.current_thread().name)
        return real_writer(*args)

    class CountingPool(ThreadPoolExecutor):
        def __init__(self, max_workers=None, **kwargs):
            pools.append(max_workers)
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(_save, "_writer", counting_writer)
    monkeypatch.setattr(_load, "ThreadPoolExecutor", CountingPool)
    for threads, expected in ((None, 2), (1, 1), (8, 2)):
        writers.clear()
        result = dcp.save(str(tmp_path / f"threads_{threads}"), model=model, optimizer=opt, threads=threads)
        assert len(result.files) == 2 and len(writers) == expected, (threads, writers)
    for threads, expected in ((None, 32), (3, 3)):
        pools.clear()
        dcp.load(str(tmp_path / "threads_1"), model=model, optimizer=opt, threads=threads)
        assert pools == [expected], (threads, pools)


def test_strict_load_names_the_mismatch(tp_mesh, tmp_path, expect_error):
    model, params, opt = fresh_tp(DIM)
    path = str(tmp_path / "step_0000001")
    dcp.save(path, model=model, optimizer=opt)

    other = _TPBlockPlus(DIM)
    with expect_error(dcp.MissingKeyError, r"live tensor\(s\) have no entry in the checkpoint: \['model/_TPBlockPlus/"):
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
    with expect_error(dcp.MissingKeyError, r"live tensor\(s\) have no entry in the checkpoint: \['optimizer/"):
        dcp.load(model_only, model=model2, optimizer=opt2)
    partial = dcp.load(model_only, model=model2, optimizer=opt2, strict=False)
    assert partial.missing and all(k.startswith("optimizer/") for k in partial.missing)
    assert not partial.unexpected


def test_incomplete_directory_and_existing_directory_are_refused(tp_mesh, tmp_path, expect_error):
    model, params, opt = fresh_tp(DIM)
    with expect_error(dcp.IncompleteCheckpointError, "nowhere is not a complete checkpoint"):
        dcp.load(str(tmp_path / "nowhere"), model=model)
    path = str(tmp_path / "step_0000001")
    dcp.save(path, model=model)
    with expect_error(dcp.CheckpointError, "step_0000001 already exists; pass overwrite=True to replace it"):
        dcp.save(path, model=model)
    dcp.save(path, model=model, overwrite=True)
    assert dcp.is_complete(path)


def test_save_refuses_a_staging_directory_it_did_not_create(tp_mesh, tmp_path, monkeypatch, expect_error):
    """A staging directory that already exists belongs to another save of the same step that drew the same nonce, or
    to one that died: the save is refused before writing anything, and the directory is left as it was."""
    model, params, opt = fresh_tp(DIM)
    path = str(tmp_path / "step_0000002")
    monkeypatch.setattr(secrets, "randbits", lambda bits: 12345)
    other = staging_dir(path, 12345)
    os.makedirs(os.path.join(other, "data"))
    with open(os.path.join(other, "data", "other_save.safetensors"), "w") as f:
        f.write("another save's file")
    with expect_error(dcp.CheckpointError, r"step_0000002\.partial-12345 already exists: another save of this step"):
        dcp.save(path, model=model)
    assert os.listdir(os.path.join(other, "data")) == ["other_save.safetensors"], "the other save's files are kept"
    assert not os.path.exists(path)
    shutil.rmtree(other)
    dcp.save(path, model=model)
    assert dcp.is_complete(path) and not os.path.exists(other)


class _LoraQKV(AbstractModuleBase):
    def __init__(self, dim: int) -> None:
        super().__init__()
        base = ColumnParallelLinear(dim, 2 * dim, has_bias=False, fused_segments=(dim, dim // 2, dim // 2))
        self.qkv_linear = LoraColumnParallelLinear(base, LoraConfig(rank=8))

    def forward(self, x):
        return self.qkv_linear(x)


class _LoraHeads(AbstractModuleBase):
    """LoRA on two padded vocab heads, a TP one and a replicated one: 100 logical rows padded to 128."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        tp_head = ColumnParallelLinear(dim, 128, has_bias=False)
        tp_head.weight.checkpoint_layout = {"logical_extent": (2, 100)}
        self.tp_head = LoraColumnParallelLinear(tp_head, LoraConfig(rank=8))
        head = LinearLayer(dim, 128, has_bias=False)
        head.weight.checkpoint_layout = {"logical_extent": (2, 100)}
        self.head = LoraLinear(head, LoraConfig(rank=8))


def test_lora_declares_the_rows_it_adds_to(tp_mesh, tmp_path, expect_error):
    infos = {lt.key.rsplit("/", 1)[-1]: lt.info for lt in collect_model(_LoraQKV(DIM))}
    segments = (DIM, DIM // 2, DIM // 2)
    assert infos["weight"].transform == fused_segments(2, segments, 2)
    assert infos["lora_B"].transform == fused_segments(2, segments, 2), "lora_B follows the base's per-shard rows"
    # On a padded head lora_B carries the vocab padding too, so it reshards across TP sizes like the head does.
    infos = {"/".join(lt.key.rsplit("/", 2)[-2:]): lt.info for lt in collect_model(_LoraHeads(DIM))}
    for head in ("tp_head", "head"):
        assert infos[f"{head}/weight"].transform == Pad(2, 100, 128)
        assert infos[f"{head}/lora_B"].transform == Pad(2, 100, 128), head
    # The layer only records its declaration; a bad one is refused by the first save, before anything is written.
    for segments, message in (
        ((63, 33), "segment size 63 is not divisible by the 2 shards"),
        ((32, 32), r"fused_segments \(32, 32\) sum to 64 but the fused dim has 96 rows"),
    ):
        layer = ColumnParallelLinear(DIM, 96, has_bias=False, fused_segments=segments)
        with expect_error(dcp.LayoutError, message):
            dcp.save(str(tmp_path / "bad"), model=layer)
        assert not (tmp_path / "bad").exists()


def test_save_refuses_bad_state_before_writing_and_leaks_nothing(tp_mesh, tmp_path, monkeypatch, expect_error):
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
    with expect_error(dcp.CheckpointError, r"^writing the shards failed: OSError: simulated EDQUOT$") as raised:
        dcp.save(str(tmp_path / "quota"), model=model, optimizer=opt)
    assert type(raised.value.__cause__) is OSError
    monkeypatch.setattr(_save, "create_file", real_create)
    assert len(os.listdir("/proc/self/fd")) == open_fds
    assert os.listdir(tmp_path) == [], "the staging directory is removed"


def test_failed_load_says_model_is_partly_restored(tp_mesh, tmp_path, monkeypatch, expect_error):
    """A load that fails after placing tensors says the model is partly restored."""
    model, params, opt = fresh_tp(DIM)
    path = str(tmp_path / "step_0000001")
    dcp.save(path, model=model, optimizer=opt)
    real_place = _load._place
    calls = []

    def place_then_fail(live, *args):
        calls.append(live.key)
        if len(calls) == 2:
            raise RuntimeError("simulated device out of memory")
        real_place(live, *args)

    model2, _, opt2 = fresh_tp(DIM)
    monkeypatch.setattr(_load, "_place", place_then_fail)
    with expect_error(
        dcp.CheckpointError,
        r"load failed at .* \(RuntimeError: simulated device out of memory\) after 1 of \d+ tensors were placed",
    ):
        dcp.load(path, model=model2, optimizer=opt2)

    def fail(live, *args):
        raise RuntimeError("simulated device out of memory")

    monkeypatch.setattr(_load, "_place", fail)
    with expect_error(
        dcp.CheckpointError, r"^reading and placing the shards failed: RuntimeError: simulated device out of memory$"
    ) as raised:
        dcp.load(path, model=model2, optimizer=opt2)
    assert type(raised.value.__cause__) is RuntimeError, "nothing was placed: no partly-restored note"


def test_truncated_data_file_is_refused_first(tp_mesh, tmp_path, expect_error):
    """A truncated data file is refused before anything is placed."""
    model, params, opt = fresh_tp(DIM)
    train_step(model, opt, DIM)
    path = str(tmp_path / dcp.step_dir_name(1))
    dcp.save(path, model=model, optimizer=opt)
    victim = sorted(glob.glob(os.path.join(path, "data", "*.safetensors")))[0]
    os.truncate(victim, os.path.getsize(victim) - 1)

    model2, params2, opt2 = fresh_tp(DIM)
    before = model_shards(params2)
    with expect_error(dcp.IncompleteCheckpointError, "truncated or corrupt"):
        dcp.load(path, model=model2, optimizer=opt2)
    assert_same_shards(model_shards(params2), before)


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
    saved = model_shards(sharded.parameters())
    (pieces,) = saved.values()
    assert [p.shape[2] for p in pieces] == [48, 48], "precondition: a tile and a half per device"
    path = str(tmp_path / "sharded")
    dcp.save(path, model=sharded)

    same = _OddRows(sharded=True, seed=1)
    assert_some_differ(model_shards(same.parameters()), saved)
    assert dcp.load(path, model=same).fast_path_keys, "each device reads back its own chunk"
    assert_same_shards(model_shards(same.parameters()), saved)

    # Onto a replicated weight: every device gets all 96 rows, put back together from the two 48-row chunks.
    replicated = _OddRows(sharded=False, seed=2)
    assert not dcp.load(path, model=replicated).fast_path_keys
    (whole,) = model_shards(replicated.parameters()).values()
    assert all(np.array_equal(w, np.concatenate(pieces, axis=2)) for w in whole)

    # And back: the replicated weight's checkpoint cut into 48-row shards.
    path = str(tmp_path / "replicated")
    dcp.save(path, model=replicated)
    resharded = _OddRows(sharded=True, seed=3)
    assert not dcp.load(path, model=resharded).fast_path_keys
    assert_same_shards(model_shards(resharded.parameters()), saved)


def _cpp_llama():
    """The C++ tensor-parallel Llama: its fused kv_linear cannot declare its layout from Python."""
    cfg = ttml.models.llama.CppLlamaConfig()
    cfg.num_heads, cfg.num_groups, cfg.embedding_dim, cfg.num_blocks = 2, 2, DIM, 1
    cfg.vocab_size, cfg.max_sequence_length = DIM, 32
    return ttml.models.distributed.llama.create_llama_model(cfg)


def test_cpp_model_round_trips_on_the_same_world(tp_mesh, tmp_path):
    """A C++ model round trips on the same world, with a warning that only this TP size can load it."""
    model = _cpp_llama()
    saved = model_shards(model.parameters())
    path = str(tmp_path / "ckpt")
    with pytest.warns(UserWarning, match="restores correctly only onto the same TP size"):
        dcp.save(path, model=model)
    fresh = _cpp_llama()
    assert_some_differ(model_shards(fresh.parameters()), saved)
    with pytest.warns(UserWarning, match="restores correctly only onto the same TP size"):
        dcp.load(path, model=fresh)
    assert_same_shards(model_shards(fresh.parameters()), saved)
    # Its fused kv_linear is the one weight a load at another TP size would scramble, so load checks its TP split.
    with pytest.warns(UserWarning, match="restores correctly only onto the same TP size"):
        live = collect_model(fresh)
    assert [lt.key for lt in live if lt.fused_name] == ["model/llama/llama_block_0/attention/kv_linear/weight"]
    entry = dcp.read_index(path).entries["model/llama/llama_block_0/attention/kv_linear/weight"]
    assert entry.tp_split == (1, 1, 2, 1), "the index records the TP split it was saved with"


class _Qkv(AbstractModuleBase):
    """A fused-looking projection under TP2, with or without its fused_segments declaration."""

    def __init__(self, dim: int, fused_segments=None) -> None:
        super().__init__()
        self.qkv_linear = ColumnParallelLinear(dim, 2 * dim, has_bias=False, fused_segments=fused_segments)


def test_undeclared_fused_weight_needs_same_row_order(tp_mesh, tmp_path, expect_error):
    """An undeclared fused weight loads only where its row order is left alike."""
    model = _Qkv(DIM)  # no declaration: its rows are in TP2's per-shard order, which only the TP split records
    path = str(tmp_path / "undeclared")
    with pytest.warns(UserWarning, match="declares no fused_segments"):
        dcp.save(path, model=model)
    assert dcp.read_index(path).entries["model/_Qkv/qkv_linear/weight"].tp_split == (1, 1, 2, 1)
    saved = model_shards(model.parameters())

    same = _Qkv(DIM)
    with pytest.warns(UserWarning, match="declares no fused_segments"):
        dcp.load(path, model=same)  # the same TP size, undeclared on both sides: the rows copy as they are
    assert_same_shards(model_shards(same.parameters()), saved)

    # Declaring the segments now would read TP2's per-shard rows as canonical ones: refused, nothing placed.
    declared = _Qkv(DIM, fused_segments=(DIM, DIM // 2, DIM // 2))
    before = model_shards(declared.parameters())
    with expect_error(
        dcp.LayoutError,
        r"qkv_linear/weight: its name suggests a fused projection .*\[1, 1, 2, 1\] per dim undescribed and the live "
        r"model \[1, 1, 1, 1\]",
    ):
        dcp.load(path, model=declared)
    assert_same_shards(model_shards(declared.parameters()), before)


def _ttml_draw():
    """One tensor from the ttml generator (``rand`` without a seed draws its seed from it), as seen by device 0."""
    value = ttml.ops.rand((1, 1, 32, 32)).get_value(NATIVE)
    return ttnn.to_torch(ttnn.get_device_tensors(value)[0])


def _adamw_stochastic_rounding(params):
    """AdamW rounding its BF16 updates stochastically: every step draws from the ttml generator."""
    cfg = ttml.optimizers.AdamWConfig.make(
        lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, stochastic_rounding=True
    )
    return ttml.optimizers.AdamW(params, cfg)


@pytest.mark.parametrize("damage", ["ttml_position_past_624", "python_gauss_next_text"])
def test_damaged_rng_state_changes_nothing(tp_mesh, tmp_path, damage, expect_error):
    """A load asking for damaged RNG state changes nothing, and one that does not ask for RNG state goes ahead."""
    ctx = ttml.autograd.AutoContext.get_instance()
    model, params, opt = fresh_tp(DIM)
    train_step(model, opt, DIM)
    path = str(tmp_path / "ckpt")
    dcp.save(path, model=model)
    damage_stored_rng(path, MISREAD_RNG[damage])

    model2, params2, _ = fresh_tp(DIM)
    before, generator = model_shards(params2), ctx.get_generator_state()
    with expect_error(dcp.CheckpointError, "the stored RNG state is malformed .*pass rng=False to load"):
        dcp.load(path, model=model2, rng=True)
    assert_same_shards(model_shards(params2), before)
    assert ctx.get_generator_state() == generator, "refused before any generator was set"

    loaded = dcp.load(path, model=model2, rng=False)
    assert loaded.restored == [] and ctx.get_generator_state() == generator
    assert_same_shards(model_shards(params2), model_shards(params))


def test_train_state_resumes_where_the_save_left_it(tp_mesh, tmp_path, expect_error):
    """The scheduler, the dataloader and every generator continue exactly as the uninterrupted run would, and so
    does the next training step, taken with the restored moments, step count, LR and generator: the optimizer rounds
    stochastically, so the step matches only if the ttml generator was restored, not reseeded."""
    ctx = ttml.autograd.AutoContext.get_instance()
    model, params, opt = fresh_tp(DIM, _adamw_stochastic_rounding)
    sched = CosineAnnealingScheduler(opt, T_max=8)
    loader = Loader()
    ctx.set_seed(123)
    random.seed(4)
    np.random.seed(5)
    torch.manual_seed(6)
    advance_generators()  # the generators are saved mid-stream, not as seeding left them
    for _ in range(3):
        train_step(model, opt, DIM)
        sched.step()
        loader.advance()
    path = str(tmp_path / dcp.step_dir_name(3))
    dcp.save(path, model=model, optimizer=opt, scheduler=sched, dataloader=loader, header={"step": 3})
    stored = dcp.read_train_state(path)
    assert set(stored) == {"scheduler", "dataloader", "rng"} and stored["dataloader"] == {"position": 3}
    assert set(stored["rng"]) == {"format", "ranks"} and list(stored["rng"]["ranks"]) == ["0"]
    assert set(stored["rng"]["ranks"]["0"]) == {"ttml", "python", "numpy", "ttml_init", "torch"}
    lrs_at_save = sched.get_last_lrs()
    opt_lrs_at_save = opt.get_group_lrs()  # the optimizer holds float32 LRs, the scheduler doubles

    # The uninterrupted run continues for one step, a real one: this is what a resumed run must reproduce.
    train_step(model, opt, DIM, input_scale=0.5)
    after_step = model_shards(params)
    after_state, after_scalars = optimizer_shards(opt)
    sched.step()
    loader.advance()
    expected = {
        "lrs": sched.get_last_lrs(),
        "loader": loader.get_state_dict(),
        "ttml": _ttml_draw(),
        "host": host_draws(),
    }

    model2, params2, opt2 = fresh_tp(DIM, _adamw_stochastic_rounding)
    sched2 = CosineAnnealingScheduler(opt2, T_max=8)
    loader2 = Loader()
    ctx.set_seed(999)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # rng by default: restoring a scheduler and a dataloader restores the generators with them.
    loaded = dcp.load(path, model=model2, optimizer=opt2, scheduler=sched2, dataloader=loader2)
    assert loaded.restored == [
        "scheduler",
        "dataloader",
        "rng.ttml",
        "rng.python",
        "rng.numpy",
        "rng.ttml_init",
        "rng.torch",
    ]
    assert loaded.train_state == stored and loaded.missing == [] and loaded.header == {"step": 3}
    assert sched2.get_last_lrs() == lrs_at_save and opt2.get_group_lrs() == opt_lrs_at_save
    train_step(model2, opt2, DIM, input_scale=0.5)
    assert_same_shards(model_shards(params2), after_step)
    assert_same_shards(optimizer_shards(opt2)[0], after_state)
    assert optimizer_shards(opt2)[1] == after_scalars
    sched2.step()
    loader2.advance()
    assert sched2.get_last_lrs() == expected["lrs"]
    assert loader2.get_state_dict() == expected["loader"]
    assert torch.equal(_ttml_draw(), expected["ttml"])
    assert host_draws() == expected["host"]

    # The negative control: everything restored but the generators (rng=False, a deliberate reseed), and the same
    # step rounds differently.
    model3, params3, opt3 = fresh_tp(DIM, _adamw_stochastic_rounding)
    sched3 = CosineAnnealingScheduler(opt3, T_max=8)
    ctx.set_seed(999)
    dcp.load(path, model=model3, optimizer=opt3, scheduler=sched3, rng=False)
    train_step(model3, opt3, DIM, input_scale=0.5)
    assert_some_differ(model_shards(params3), after_step)

    # A dataloader alone brings the generators back too, by default.
    ctx.set_seed(999)
    alone = dcp.load(path, model=model3, dataloader=Loader())
    assert alone.restored == ["dataloader", "rng.ttml", "rng.python", "rng.numpy", "rng.ttml_init", "rng.torch"]
    assert ctx.get_generator_state() == stored["rng"]["ranks"]["0"]["ttml"]

    # A load of the model alone leaves the generators alone.
    before = ctx.get_generator_state()
    plain = dcp.load(path, model=model2)
    assert plain.restored == [] and ctx.get_generator_state() == before

    # Asked for but absent (the generators too, by default, with a scheduler): a strict load refuses before touching
    # anything, a lenient one lists them.
    bare = str(tmp_path / "bare")
    dcp.save(bare, model=model, rng=False)
    assert dcp.read_train_state(bare) == {}
    untouched = model_shards(params2), sched2.get_state_dict()
    with expect_error(
        dcp.MissingKeyError,
        r"the checkpoint holds no state for \['train_state/scheduler', 'train_state/rng'\]; rng was not passed, .* "
        r"pass rng=False to keep the current ones",
    ):
        dcp.load(bare, model=model2, scheduler=sched2)
    with expect_error(dcp.MissingKeyError, r"the checkpoint holds no state for \['train_state/rng'\]$"):
        dcp.load(bare, model=model2, rng=True)  # asked for explicitly: nothing to suggest
    assert sched2.get_state_dict() == untouched[1]
    assert_same_shards(model_shards(params2), untouched[0])
    lenient = dcp.load(bare, model=model2, scheduler=sched2, strict=False)
    assert lenient.missing == ["train_state/scheduler", "train_state/rng"] and lenient.restored == []

    # A damaged RNG block is refused before anything is placed, generators and model alike.
    damage_stored_rng(path, WRONG_LENGTH_RNG["numpy_623_words"])
    untouched = model_shards(params2), ctx.get_generator_state(), random.getstate()
    with expect_error(dcp.CheckpointError, "the stored RNG state is malformed"):
        dcp.load(path, model=model2, optimizer=opt2, rng=True)
    assert_same_shards(model_shards(params2), untouched[0])
    assert (ctx.get_generator_state(), random.getstate()) == untouched[1:]

    # And a save whose dataloader state is not JSON writes nothing.
    class _NotJson(Loader):
        def get_state_dict(self) -> dict:
            return {"position": np.int64(self.position)}

    with expect_error(dcp.CheckpointError, "dataloader state at /position: a int64 has no JSON form"):
        dcp.save(str(tmp_path / "not_json"), model=model, dataloader=_NotJson())
    assert not [name for name in os.listdir(tmp_path) if name.startswith("not_json")], "nothing on disk"


def test_fsdp_round_trip_and_a_gathered_module_resharded(fsdp_mesh, tmp_path):
    model = _Stack(DIM)
    ttml.fsdp.fully_shard(model)
    params = model.parameters()
    opt = make_adamw(params)
    train_step(model, opt, DIM)
    before = model_shards(params)
    before_state, before_scalars = optimizer_shards(opt)
    path = str(tmp_path / dcp.step_dir_name(1))

    dcp.save(path, model=model, optimizer=opt, header={"step": 1})

    model2 = _Stack(DIM)
    ttml.fsdp.fully_shard(model2)
    params2 = model2.parameters()
    opt2 = make_adamw(params2)
    assert_some_differ(model_shards(params2), before)
    loaded = dcp.load(path, model=model2, optimizer=opt2)
    assert loaded.header == {"step": 1} and loaded.fast_path_keys
    assert_same_shards(model_shards(params2), before)
    assert_same_shards(optimizer_shards(opt2)[0], before_state)
    assert optimizer_shards(opt2)[1] == before_scalars

    # One more step from the checkpoint equals one more step of the run that never stopped: the sharded moments
    # and the step count feed the update kernel as they did before the save.
    train_step(model, opt, DIM, input_scale=0.5)
    train_step(model2, opt2, DIM, input_scale=0.5)
    assert_same_shards(model_shards(params2), model_shards(params))
    assert_same_shards(optimizer_shards(opt2)[0], optimizer_shards(opt)[0])
    assert optimizer_shards(opt2)[1] == optimizer_shards(opt)[1]

    # A module left gathered by a forward with reshard_after_forward=False is resharded first.
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
    dcp.save(str(tmp_path / "resharded"), model=model3, optimizer=opt3)
    assert model3._fsdp_state.sharded_state == _ShardedState.SHARDED
    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_fsdp_load_reshards_gathered_and_refuses_early_optimizer(fsdp_mesh, tmp_path, expect_error):
    """An FSDP load reshards a gathered module first, and refuses an optimizer built before sharding."""
    model = _Stack(DIM)
    ttml.fsdp.fully_shard(model)
    opt = make_adamw(model.parameters())
    train_step(model, opt, DIM)
    path = str(tmp_path / "step_0000001")
    dcp.save(path, model=model, optimizer=opt)

    # Loading into a module left gathered: resharded first.
    gathered = _Stack(DIM)
    ttml.fsdp.fully_shard(gathered, reshard_after_forward=False)
    gathered_opt = make_adamw(gathered.parameters())
    train_step(gathered, gathered_opt, DIM)
    x = ttml.autograd.Tensor.from_numpy(
        np.zeros((1, 1, DIM, DIM), dtype=np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )
    gathered(x)
    ttml.autograd.AutoContext.get_instance().reset_graph()
    assert gathered._fsdp_state.sharded_state == _ShardedState.UNSHARDED, "precondition"
    dcp.load(path, model=gathered, optimizer=gathered_opt)
    assert gathered._fsdp_state.sharded_state == _ShardedState.SHARDED
    assert_same_shards(model_shards(gathered.parameters()), model_shards(model.parameters()))

    # An optimizer built before fully_shard holds full-size moments for sharded parameters.
    early = _Stack(DIM)
    early_opt = make_adamw(early.parameters())
    ttml.fsdp.fully_shard(early)
    with expect_error(dcp.CheckpointError, "was the optimizer created before the model was sharded"):
        dcp.save(str(tmp_path / "early"), model=early, optimizer=early_opt)
