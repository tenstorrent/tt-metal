# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

import ttnn
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.prefetcher import Prefetcher
from models.tt_transformers.tt.trace_io import PreparedTraceIO


@pytest.fixture
def prefetcher(monkeypatch):
    # Avoid model/device construction; exercise the real mode transitions.
    prefetcher = object.__new__(Prefetcher)
    prefetcher.mesh_device = Mock()
    prefetcher.mode = Mode.DECODE
    prefetcher.init_decode_done = True
    prefetcher.init_prefill_done = False
    prefetcher.prefetcher_sub_device = SimpleNamespace(manager_id=17, sub_devices_id=[0, 1])
    prefetcher.all_worker_cores_range_set = object()
    prefetcher.sender_receiver_mapping = object()
    prefetcher.max_tensor_block_size = 4096
    prefetcher.global_cb = None
    prefetcher._global_cb_addresses = None

    def allocate(*args, buffer_address, config_address):
        cb = Mock()
        cb.buffer_address.return_value = 8192 if buffer_address is None else buffer_address
        cb.config_address.return_value = 4096 if config_address is None else config_address
        return cb

    monkeypatch.setattr(ttnn, "create_global_circular_buffer", Mock(side_effect=allocate))
    return prefetcher


def test_prefill_before_decode_uses_default_manager(prefetcher):
    prefetcher.init_decode_done = False
    prefetcher.mode = Mode.PREFILL
    prefetcher.init(Mode.PREFILL)
    prefetcher.init(Mode.PREFILL)
    prefetcher.mesh_device.clear_loaded_sub_device_manager.assert_called_once_with()
    prefetcher.mesh_device.create_sub_device_manager.assert_not_called()
    ttnn.create_global_circular_buffer.assert_not_called()


def test_warmup_can_move_gcb_then_capture_fixes_both_addresses(prefetcher):
    prefetcher._create_global_cb()
    warmup_cb = prefetcher.global_cb
    assert prefetcher._global_cb_addresses is None
    prefetcher.init(Mode.PREFILL)
    warmup_cb.deallocate.assert_called_once_with()
    prefetcher.init(Mode.DECODE)
    assert prefetcher.global_cb is None
    prefetcher.prepare_trace()
    addresses = prefetcher._global_cb_addresses
    assert addresses == (8192, 4096)
    for _ in range(3):
        cb = prefetcher.global_cb
        prefetcher.init(Mode.PREFILL)
        cb.deallocate.assert_called_once_with()
        assert prefetcher.global_cb is None
        prefetcher.init(Mode.DECODE)
        ttnn.create_global_circular_buffer.assert_called_with(
            prefetcher.mesh_device, prefetcher.sender_receiver_mapping, 4096, buffer_address=8192, config_address=4096
        )
        prefetcher.global_cb.acknowledge_corruptible.assert_called_once_with()
        assert prefetcher._global_cb_addresses == addresses
    assert prefetcher.mesh_device.load_sub_device_manager.call_args_list == [call(17)] * 4
    prefetcher.mesh_device.create_sub_device_manager.assert_not_called()


def test_restore_collision_does_not_commit_decode_mode(prefetcher, expect_error):
    prefetcher.prepare_trace()
    prefetcher.init(Mode.PREFILL)
    prefetcher.mesh_device.reset_mock()
    allocate = ttnn.create_global_circular_buffer.side_effect
    ttnn.create_global_circular_buffer.side_effect = RuntimeError("occupied GCB address")
    with expect_error(RuntimeError, "occupied GCB address"):
        prefetcher.init(Mode.DECODE)
    assert prefetcher.mode == Mode.PREFILL
    assert prefetcher.global_cb is None
    assert prefetcher._global_cb_addresses == (8192, 4096)
    prefetcher.mesh_device.clear_loaded_sub_device_manager.assert_called_once_with()
    prefetcher.mesh_device.reset_sub_device_stall_group.assert_called_once_with()
    ttnn.create_global_circular_buffer.side_effect = allocate
    prefetcher.init(Mode.DECODE)
    assert prefetcher.mode == Mode.DECODE
    assert prefetcher.global_cb.buffer_address() == 8192
    assert prefetcher.global_cb.config_address() == 4096


def test_prefetch_stages_address_table_once_before_decode_inputs(prefetcher):
    prefetcher.callbacks = [Mock()]
    prefetcher.prefetch_done = False
    prefetcher.create_address_tensor = Mock(return_value=object())
    prefetcher.prefetch()
    assert prefetcher.prefetched_tt_addr_tensor is prefetcher.create_address_tensor.return_value
    prefetcher.init(Mode.PREFILL)
    prefetcher.prefetch()
    prefetcher.init(Mode.DECODE)
    prefetcher.prefetch()
    prefetcher.callbacks[0].assert_called_once_with()
    prefetcher.create_address_tensor.assert_called_once_with()
    ttnn.create_global_circular_buffer.assert_not_called()


@pytest.fixture
def generator(prefetcher, monkeypatch):
    prefetcher.init(Mode.PREFILL)
    mesh = prefetcher.mesh_device

    def tensor():
        return SimpleNamespace(
            spec="decode", shape=(1, 1, 32, 32), tensor_topology=lambda: "replicated", device=lambda: mesh
        )

    for name in ("copy_host_to_device_tensor", "copy", "slice", "end_trace_capture", "release_trace"):
        monkeypatch.setattr(ttnn, name, Mock())
    monkeypatch.setattr(ttnn, "begin_trace_capture", Mock(return_value=23))
    monkeypatch.setattr(ttnn, "to_device", Mock(side_effect=lambda *args, **kwargs: tensor()))
    monkeypatch.setattr(ttnn, "empty_like", Mock(side_effect=lambda *args: tensor()))
    model = Mock(prefetcher=prefetcher, sampling=None)
    model.switch_mode.side_effect = prefetcher.init
    model.prepare_decode_inputs_host.return_value = (tensor(), tensor(), tensor(), tensor())
    model.ttnn_decode_forward.return_value = (tensor(), None)
    generator = object.__new__(Generator)
    generator.model = [model]
    generator.model_args = [SimpleNamespace(mesh_device=mesh)]
    generator.data_parallel = 1
    generator.mode = Mode.PREFILL
    generator.device_grammar_enabled = False
    generator.trace_ids_decode = defaultdict(lambda: None)
    generator.trace_inputs_decode = {}
    generator.trace_output_decode = {}
    generator.trace_id_prefill = {}
    generator.trace_id_prefill_sampling = {}
    generator._prepared_decode_traces = {}
    generator._pending_decode_trace = None
    generator._trace_io_buffers = PreparedTraceIO()
    yield generator
    generator.release_persistent_capture()


@pytest.mark.parametrize("preparation", ["prefill", "eager"])
def test_prepared_prefetcher_decode_survives_prefill_capture(generator, preparation):
    page_table = torch.zeros((1, 4), dtype=torch.int32)
    if preparation == "prefill":
        generator._prepare_decode_trace_once(None, page_table, False)
        prepared = generator._pending_decode_trace
    else:
        generator.model[0].switch_mode(Mode.DECODE)
        output = generator._prepare_decode_trace_variant([0], [0], page_table=[page_table])
        prepared = generator._prepared_decode_traces[False]
        assert output == prepared["outputs"]
        generator.model[0].switch_mode(Mode.PREFILL)
    assert prepared is not None
    assert generator.model[0].prefetcher.mode == Mode.PREFILL
    allocations = ttnn.to_device.call_count
    destination = prepared["outputs"][0][0]
    grid = generator.model[0].prefetcher.all_worker_cores_range_set
    assert ttnn.slice.call_args.kwargs["sub_core_grids"] is grid
    assert ttnn.slice.call_args.kwargs["output_tensor"] is destination

    # Model the first prefill capture, which permanently closes I/O preparation.
    generator.trace_id_prefill["128_0"] = 11
    generator._trace_io().freeze()
    if preparation == "prefill":
        generator._record_pending_traces()
        assert generator.trace_ids_decode[False] == {0: 23}
        assert generator.model[0].prefetcher.mode == Mode.PREFILL
    else:
        generator.model[0].switch_mode(Mode.DECODE)
        ids, outputs, *inputs = generator._capture_decode_trace_text([0], [0], page_table=[page_table])
        assert ids == {0: 23}
        assert outputs == prepared["outputs"]
        assert inputs == prepared["device_inputs"]
    assert ttnn.to_device.call_count == allocations
    assert ttnn.slice.call_args.kwargs["sub_core_grids"] is grid
    assert ttnn.slice.call_args.kwargs["output_tensor"] is destination
    assert generator.model[0].prefetcher._global_cb_addresses == (8192, 4096)
    ttnn.copy.assert_not_called()  # Unscoped copies can span the sender sub-device.


def test_prefetcher_precapture_stages_and_records(generator):
    page_table = torch.zeros((1, 4), dtype=torch.int32)
    assert generator.precapture_decode_trace_variants([None], page_table, page_table, page_table, None)
    assert generator.trace_ids_decode[False] == {0: 23}
    assert generator._trace_io_buffers._frozen


def test_failed_warmup_restores_prefill_manager(generator, expect_error):
    generator.model[0].prepare_decode_inputs_host.side_effect = RuntimeError("compile failed")
    with expect_error(RuntimeError, "compile failed"):
        generator._prepare_decode_trace_for_warmup(None, torch.zeros((1, 4)), False)
    assert generator.mode == Mode.PREFILL
    assert generator.model[0].prefetcher.mode == Mode.PREFILL
    generator.model[0].prefetcher.mesh_device.clear_loaded_sub_device_manager.assert_called()
    ttnn.begin_trace_capture.assert_not_called()


def test_unprepared_prefetcher_decode_still_fails_before_allocating(generator, expect_error):
    generator.trace_id_prefill["128_0"] = 11
    with expect_error(RuntimeError, "preparation is closed"):
        generator._prepare_decode_trace_text([0], [0], skip_precompile=True)
    generator.model[0].prepare_decode_inputs_host.assert_not_called()
    ttnn.to_device.assert_not_called()


def test_cleanup_selects_each_trace_manager_without_reconstructing_gcb(generator):
    generator.trace_id_prefill["128_0"] = 11
    generator.trace_ids_decode[False] = {0: 23}
    prefetcher = generator.model[0].prefetcher
    prefetcher.prepare_trace = Mock(side_effect=AssertionError("cleanup must not allocate"))
    prefetcher.mode = Mode.DECODE
    events = Mock()
    events.attach_mock(prefetcher.mesh_device.clear_loaded_sub_device_manager, "prefill")
    events.attach_mock(prefetcher.mesh_device.load_sub_device_manager, "decode")
    events.attach_mock(ttnn.release_trace, "release")
    generator.release_persistent_capture()
    assert events.mock_calls == [
        call.prefill(),
        call.release(prefetcher.mesh_device, 11),
        call.decode(17),
        call.release(prefetcher.mesh_device, 23),
    ]
    ttnn.create_global_circular_buffer.assert_not_called()
