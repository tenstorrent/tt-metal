# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: the generator re-captures a gpt_oss decode trace when programs were compiled since its capture.

Programs compiled after a decode trace capture (the first prefill of a new length) can hold long-lived device
resources, e.g. the global semaphores of the prefill's ttnn.all_reduce, that land in the trace's scratch region; every
decode replay then overwrites them and the next prefill of that length hangs or decodes garbage. For models that set
``_tt_recapture_decode_trace_on_new_programs`` the generator records the program cache size whenever it installs a
decode trace, and re-captures a trace that has become older than the program cache over its existing inputs: at the
end of a prefill call and before a decode replay, whatever the reload flags. The trace installs, the prefill call and
the decode replays below go through the generator itself, over fake devices. No device needed.

Also: gpt_oss prefill tails slice the last-token tile with runtime bounds, so a new prompt length within a padded
length compiles no new program.
"""

from types import MethodType, SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tt.model import Model
from models.ttt_compat.tt import generator as generator_module
from models.ttt_compat.tt.generator import Generator

HOST_SAMPLING, DEVICE_SAMPLING = False, True


class _Mesh:
    def __init__(self):
        self.entries = 10

    def num_program_cache_entries(self):
        return self.entries


class _Sampling:
    def __init__(self):
        self.captures = []
        self.resets = 0

    def capture_trace(self, logits, tt_out_tok=None, skip_precompile=False):
        self.captures.append((logits, tt_out_tok, skip_precompile))

    def reset_trace(self):
        self.resets += 1


class _OptedInModel:
    _tt_recapture_decode_trace_on_new_programs = True

    def __init__(self):
        self.sampling = _Sampling()
        self.captured_inputs = []

    def switch_mode(self, mode):
        pass

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        return ["host tokens", "host positions", "host rope", "host page table"]

    def ttnn_decode_forward(self, *inputs, kv_cache=None, on_device_logits=False):
        self.captured_inputs.append((inputs, kv_cache))
        return object()  # the new trace's output


class _OtherModel(_OptedInModel):
    _tt_recapture_decode_trace_on_new_programs = False


class _Device:
    """ttnn trace calls over the fake mesh."""

    def __init__(self, monkeypatch):
        self.next_trace = 0
        self.released = []
        self.replayed = []
        self.full_reloads = 0
        monkeypatch.setattr(ttnn, "begin_trace_capture", self._begin)
        monkeypatch.setattr(ttnn, "end_trace_capture", lambda mesh, trace_id, cq_id=0: None)
        monkeypatch.setattr(ttnn, "release_trace", lambda mesh, trace_id: self.released.append(trace_id))
        monkeypatch.setattr(ttnn, "execute_trace", lambda mesh, trace_id, **kw: self.replayed.append(trace_id))
        monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
        monkeypatch.setattr(generator_module, "copy_host_to_device", self._reload)

    def _begin(self, mesh, cq_id=0):
        self.next_trace += 1
        return self.next_trace

    def _reload(self, host_tensors, device_tensors):
        self.full_reloads += 1


@pytest.fixture
def fake_device(monkeypatch):
    return _Device(monkeypatch)


@pytest.fixture
def make_generator():
    generators = []

    def make(model_cls=_OptedInModel):
        mesh = _Mesh()
        model = model_cls()
        generator = Generator([model], [SimpleNamespace(mesh_device=mesh)], mesh)
        generators.append(generator)
        return generator, mesh, model

    yield make
    for generator in generators:
        generator._generator_capture_released = True  # the fake traces need no release at teardown


def _decode_inputs():
    return [["tokens", "positions", "rope", "page table"]]


def _install_pending_trace(generator, on_device_sampling, kv_cache=("kv",)):
    """Install a decode trace the way the first traced prefill call does (prepared before it, recorded after it)."""
    inputs = _decode_inputs()
    generator._pending_decode_trace = {
        "device_inputs": inputs,
        "kv_cache": kv_cache,
        "on_device_sampling": on_device_sampling,
    }
    generator._record_pending_traces()
    return inputs


def _replay(generator, on_device_sampling, reload_inputs):
    return generator._decode_forward_trace_text(
        [torch.zeros(1, 1)],
        [torch.zeros(1)],
        page_table=[torch.zeros(1, 1)],
        kv_cache=("kv",),
        on_device_sampling=on_device_sampling,
        reload_inputs=reload_inputs,
        reload_page_table=False,
    )


def test_gpt_oss_opts_in():
    assert Model._tt_recapture_decode_trace_on_new_programs is True


@pytest.mark.parametrize(
    "on_device_sampling", [HOST_SAMPLING, DEVICE_SAMPLING], ids=["host_sampling", "device_sampling"]
)
@pytest.mark.parametrize("reload_inputs", [True, False], ids=["reload", "async_step"])
def test_decode_replay_recaptures_after_new_programs(fake_device, make_generator, on_device_sampling, reload_inputs):
    generator, mesh, model = make_generator()
    inputs = _install_pending_trace(generator, on_device_sampling)
    assert generator.trace_ids_decode[on_device_sampling] == {0: 1}
    captures_at_install = len(model.sampling.captures)

    mesh.entries += 3  # e.g. a prefill of a new length compiled programs after the capture
    output = _replay(generator, on_device_sampling, reload_inputs)

    # Released and captured again before the replay, over the same persistent inputs and KV cache.
    assert fake_device.released == [1]
    assert generator.trace_ids_decode[on_device_sampling] == {0: 2}
    assert fake_device.replayed == [2]
    assert generator.trace_inputs_decode[on_device_sampling][0] is inputs[0]
    assert model.captured_inputs[-1] == (tuple(inputs[0]), "kv")
    assert output is generator.trace_output_decode[on_device_sampling]
    assert generator._decode_trace_recaptures == 1
    # The re-capture itself needs no host reload: only the caller's reload flag decides.
    assert fake_device.full_reloads == int(reload_inputs)
    if on_device_sampling:
        # The sampling traces read the released trace's output: dropped, and the current one captured again.
        assert model.sampling.resets == 1
        assert len(model.sampling.captures) == captures_at_install + 1
    else:
        # The host-sampling trace has no sampling traces; the device-sampling variant's must survive.
        assert model.sampling.resets == 0
        assert len(model.sampling.captures) == captures_at_install

    # Nothing compiled since the re-capture: replays only.
    _replay(generator, on_device_sampling, reload_inputs=False)
    assert fake_device.released == [1]
    assert fake_device.replayed == [2, 2]
    assert generator._decode_trace_recaptures == 1


def test_prefill_boundary_recaptures_precaptured_traces(fake_device, make_generator):
    """Traces installed by precapture_decode_trace_variants, then a prefill of a new length before any replay."""
    generator, mesh, model = make_generator()
    generator._prepared_decode_traces = {
        key: {"device_inputs": _decode_inputs(), "kv_cache": ("kv",), "on_device_sampling": key}
        for key in (HOST_SAMPLING, DEVICE_SAMPLING)
    }
    assert generator.precapture_decode_trace_variants(
        [None, SimpleNamespace()], torch.zeros(1, 1), torch.zeros(1), torch.zeros(1, 1), ("kv",)
    )
    installed = {key: generator.trace_ids_decode[key] for key in (HOST_SAMPLING, DEVICE_SAMPLING)}
    assert installed == {HOST_SAMPLING: {0: 1}, DEVICE_SAMPLING: {0: 2}}

    def prefill_of_a_new_length(*args, **kwargs):
        mesh.entries += 5
        return "logits"

    generator._prefill_forward_text_impl = prefill_of_a_new_length
    assert generator.prefill_forward_text(torch.zeros(1, 100, dtype=torch.long)) == "logits"

    # Both traces were re-captured at the prefill boundary, before any replay.
    assert sorted(fake_device.released) == [1, 2]
    assert fake_device.replayed == []
    assert generator._decode_trace_recaptures == 2
    assert all(generator.trace_ids_decode[key] not in (None, installed[key]) for key in installed)

    # The first replay of each finds its trace current.
    for key in (HOST_SAMPLING, DEVICE_SAMPLING):
        _replay(generator, key, reload_inputs=True)
    assert sorted(fake_device.released) == [1, 2]
    assert generator._decode_trace_recaptures == 2


def test_prefill_boundary_waits_for_pending_device_sampling(fake_device, make_generator):
    generator, mesh, _ = make_generator()
    _install_pending_trace(generator, DEVICE_SAMPLING)
    generator._pending_deferred_decode_sampling = object()  # its payload holds the current trace's logits
    generator._prefill_forward_text_impl = lambda *args, **kwargs: setattr(mesh, "entries", mesh.entries + 1)
    generator.prefill_forward_text(torch.zeros(1, 100, dtype=torch.long))
    assert fake_device.released == []

    generator._pending_deferred_decode_sampling = None  # consumed; the next replay re-captures first
    _replay(generator, DEVICE_SAMPLING, reload_inputs=False)
    assert fake_device.released == [1]
    assert fake_device.replayed == [2]


def test_models_that_do_not_opt_in_keep_their_trace(fake_device, make_generator):
    generator, mesh, _ = make_generator(_OtherModel)
    _install_pending_trace(generator, DEVICE_SAMPLING)
    mesh.entries += 3
    generator._prefill_forward_text_impl = lambda *args, **kwargs: "logits"
    generator.prefill_forward_text(torch.zeros(1, 100, dtype=torch.long))
    for reload_inputs in (True, False):
        _replay(generator, DEVICE_SAMPLING, reload_inputs)
    assert fake_device.released == []
    assert fake_device.replayed == [1, 1]
    assert generator._decode_trace_baselines == {}


class _Tensor:
    def __init__(self, shape):
        self.shape = shape

    def deallocate(self, force=True):
        pass


def test_eager_prefill_tail_programs_key_on_the_padded_length(monkeypatch):
    """Last tokens in different tiles of one padded length: same slice program, bounds written at runtime."""
    model = Model.__new__(Model)
    model.layers = []
    model.mesh_device = object()
    model._tail_slice_start, model._tail_slice_end = "start bounds", "end bounds"
    writes, slices = [], []
    monkeypatch.setattr(ttnn, "from_torch", lambda tensor, **kwargs: tensor.tolist())
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, device: writes.append((device, host)))

    def fake_slice(*args, **kwargs):
        slices.append((args[1:], {k: v for k, v in kwargs.items() if k != "input_tensor"}))
        return _Tensor((1, 1, 32, 2880))

    monkeypatch.setattr(ttnn, "slice", fake_slice)

    def tail(last_token_idx, **kwargs):
        return model._forward_layers_and_head(
            _Tensor((1, 1, 2048, 2880)),
            None,
            None,
            None,
            None,
            get_last_token=(last_token_idx // 32) * 32,
            is_decode=False,
            skip_lm_head=True,
            **kwargs,
        )

    for last_token_idx in (5, 1000, 1999):
        tail(last_token_idx)
    # One program key: the same tensor-arg slice for every offset; the offsets go to the device at runtime.
    runtime = dict(starts="start bounds", ends="end bounds", slice_dim=2, num_devices=64)
    assert slices == [((), runtime)] * 3
    assert [host for _, host in writes[::2]] == [[0, 0, 0, 0], [0, 0, 992, 0], [0, 0, 1984, 0]]

    # A forward captured into a trace (the row-sharded path) keeps literal bounds: no host write.
    slices.clear()
    writes.clear()
    tail(1000, literal_last_token_bounds=True)
    assert slices == [(((0, 0, 992, 0), (1, 1, 1024, 2880)), {})]
    assert writes == []


@pytest.mark.parametrize("skip_lm", [True, False], ids=["lm_head_skipped", "lm_head"])
def test_row_sharded_tail_offset(monkeypatch, skip_lm):
    """The row-sharded forwards slice literal bounds; when their output is not read the offset is fixed."""
    seen = []

    def forward(tokens_iter, page_table, kv_cache, fixed_glt, skip_lm_head, batch_size):
        seen.append(fixed_glt)

    model = SimpleNamespace(mesh_device=SimpleNamespace(shape=(4, 8)), vocab_size=16)
    model.prepare_row_sharded_prefill_iter = MethodType(Model.prepare_row_sharded_prefill_iter, model)
    model.run_row_sharded_prefill_forward = forward
    model.extract_prefill_logits_to_host = lambda *args, **kwargs: None
    monkeypatch.setattr(ttnn, "synchronize_device", lambda _: None)
    for length in (70, 100):  # last tokens in tiles 64 and 96 of the 128 padded length
        Model.row_sharded_batched_prefill(
            model,
            torch.ones(4, 128, dtype=torch.long),
            torch.zeros(4, 2, dtype=torch.int32),
            [[SimpleNamespace(shape=(128, 1, 64, 32))]],
            prompt_lens=[length] * 4,
            prefill_seq_lens=[128] * 4,
            enable_trace=False,
            sampling_params=object() if skip_lm else None,
            model_args=SimpleNamespace(max_local_batch_size=32, vocab_size=16),
            trace_cache={"ids": {}, "inputs": {}, "outputs": {}},
        )
    assert seen == ([0, 0] if skip_lm else [64, 96])
