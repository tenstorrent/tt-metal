# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""P3c host-logic tests (no device): the device-direct lane import's ring rotation and the structure that makes its
reset-free admission safe, the position-only admission, the once-per-tensor mesh validation memo and the hand-off ring
selects.  The bitwise device test is ``test_vllm_fast_import_device.py``."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_batch_decode as batch_decode
from models.demos.blackhole.qwen38_flash_next.ttnn import contracts, lanes
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module


@pytest.mark.parametrize("position", (0, 1, 2, 3, 4, 7, 130, 1027))
def test_the_importer_rotation_is_the_hosts_canonical_ring_order(position) -> None:
    """Lane ring slot ``j`` takes generic slot ``(p + j) % 4`` in every GDN layer: the order of ``rotate_ring_image`` (the
    host pager's ``rotate_ring_canonical``)."""

    entries = lanes.GDN_LAYERS * lanes.gdn_module.CONV_KERNEL_SIZE
    image = torch.arange(entries).reshape(1, entries, 1, 1)
    rotated = lanes.rotate_ring_image(image, position)
    assert rotated.reshape(-1).tolist() == [lanes.canonical_ring_source(i, position) for i in range(entries)]


def test_the_importer_overwrites_every_family_the_lane_reset_zeroes() -> None:
    """The reset-free admission is exact only if the import writes whole every buffer ``reset_lane_inplace`` touches: the
    GDN recurrent state and ring slots, the QSA staging tile and raw-key ring, the PLE slots (the caches need no reset).
    """

    for name in ("recurrent", "staging", "ring", "ple"):
        assert name in lanes.FILL_FAMILIES
    importer = inspect.getsource(lanes.Qwen38TTNNLaneImporter.import_lane)
    assert "for name in FILL_FAMILIES" in importer and "ttnn.fill_cache(cache_view, filled, batch_idx=lane)" in importer
    assert "for index, slot in enumerate(target.conv)" in importer and "keeps[lane]" in importer  # the 144 ring slots
    assert "reset" not in importer
    # what the reset multiplies: GDN (recurrent + conv), QSA (staging + ring), PLE (conv slots)
    gdn = inspect.getsource(lanes.gdn_module.Qwen38TTNNGDNState.reset_lane_inplace)
    assert "self.recurrent" in gdn and "self.conv" in gdn
    qsa = inspect.getsource(qsa_module.Qwen38TTNNQSA.reset_lane_inplace)
    assert "state.kv_staging" in qsa and "state.raw_key_ring" in qsa
    assert "packed_kv_cache" not in qsa and "compressed_index_cache" not in qsa


def test_the_batch_decoder_imports_by_device_with_the_position_only_admission() -> None:
    move = inspect.getsource(batch_decode.Qwen38PlainLaneDecoder.move_into_lane)
    fast, host = move.split("slot = self.pool.slots[0]")
    assert "self.importer is not None and not self.host_import" in fast
    assert "model.admit_lane_position(ls, lane, length, rope_shift=0)" in fast and "admit_lane(" not in fast.replace(
        "admit_lane_position(", ""
    )
    assert fast.index("admit_lane_position") < fast.index("admit_lane_kv_paged") < fast.index("importer.import_lane")
    # the host path is the P1 / P2 sequence: the lane reset, then the kv scatter, then the re-admission
    assert "pager_g.evict(0, slot)" in host and "pager_g.rotate_ring_canonical(slot, length)" in host
    assert host.index("model.admit_lane(ls, lane, length, rope_shift=0)") < host.index("pager_l.readmit(slot, lane")
    # both callers go through it; the importer is built for paged lanes and warmed for every lane
    for caller in (batch_decode.Qwen38PlainLaneDecoder.admit, batch_decode.Qwen38PlainLaneDecoder.import_lane):
        assert "self.move_into_lane(lane, length)" in inspect.getsource(caller)
    prepare = inspect.getsource(batch_decode.Qwen38PlainLaneDecoder.prepare)
    assert (
        "Qwen38TTNNLaneImporter(self.pager_l, gl)" in prepare and "self.importer.import_lane(lane, lane % 4)" in prepare
    )
    assert prepare.index("self.importer.import_lane(") < prepare.index("model.forward_decode_lanes(ls)")


class _Position:
    def __init__(self) -> None:
        self.admitted = []

    def admit(self, lane, position, rope_shift=0) -> None:
        self.admitted.append((lane, position, rope_shift))


def _lane_model(contexts):
    model = model_module.Qwen38TTNNTextModel.__new__(model_module.Qwen38TTNNTextModel)
    model._validate_lane_state = lambda state: len(contexts)
    ple = SimpleNamespace(token_contexts=tuple(contexts))
    layers = [SimpleNamespace(ple=None) for _ in range(model_module.PLE_CHECKPOINT_LAYER + 1)]
    layers[model_module.PLE_CHECKPOINT_LAYER] = SimpleNamespace(ple=ple)
    return model, SimpleNamespace(position=_Position(), layers=layers, is_view=False), ple


def test_admit_lane_position_takes_the_lane_and_clears_its_context_without_a_device_write() -> None:
    model, state, ple = _lane_model([(1, 2), (3, 4), None, (5, 6)])
    model.admit_lane_position(state, 1, 130, rope_shift=0)
    assert state.position.admitted == [(1, 130, 0)]
    assert ple.token_contexts == ((1, 2), None, None, (5, 6))
    reset = inspect.getsource(model_module.Qwen38TTNNTextModel.admit_lane_position)
    assert "reset_lane_inplace" not in reset and "ttnn." not in reset


def test_admit_lane_position_refuses_before_it_clears_anything(expect_error) -> None:
    model, state, ple = _lane_model([(1, 2), (3, 4)])

    def refuse(lane, position, rope_shift=0):
        raise ValueError("residue rule")

    state.position.admit = refuse
    with expect_error(ValueError, "residue rule"):
        model.admit_lane_position(state, 0, 3)
    assert ple.token_contexts == ((1, 2), (3, 4))


class _FakeTensor:
    def __init__(self, tensor_id) -> None:
        if tensor_id is not None:
            self.tensor_id = tensor_id
        self.checks = 0


def _counting_contract(monkeypatch):
    contract = contracts.Qwen38MeshContract((0, 1, 2, 3))
    calls = []

    def validate(self, tensor, *, placement, shard_dim=None, require_device=True):
        tensor.checks += 1
        calls.append((placement, shard_dim))
        if getattr(tensor, "bad", False):
            raise RuntimeError("bad tensor")

    monkeypatch.setattr(contracts.Qwen38MeshContract, "_validate_tensor", validate)
    return contract, calls


def test_a_tensor_that_passed_twice_is_not_validated_again(monkeypatch) -> None:
    contract, calls = _counting_contract(monkeypatch)
    tensor = _FakeTensor(7)
    for _ in range(4):
        contract.validate_tensor(tensor, placement=contracts.TensorPlacement.HEAD_SHARDED, shard_dim=3)
    assert tensor.checks == 2  # the first sighting and the repeat; the rest hit the memo
    # another placement, shard dim or device requirement is another question
    for kwargs in (
        dict(placement=contracts.TensorPlacement.REPLICATED),
        dict(placement=contracts.TensorPlacement.HEAD_SHARDED, shard_dim=1),
        dict(placement=contracts.TensorPlacement.HEAD_SHARDED, shard_dim=3, require_device=False),
    ):
        contract.validate_tensor(tensor, **kwargs)
    assert tensor.checks == 5
    # a temporary validated once is not remembered (it must not evict the state tensors' entries)
    temporary = _FakeTensor(8)
    contract.validate_tensor(temporary, placement=contracts.TensorPlacement.HEAD_SHARDED, shard_dim=3)
    assert (8, contracts.TensorPlacement.HEAD_SHARDED, 3, True) not in contract.__dict__["_validated"]
    assert len(calls) == 6


def test_a_failed_validation_is_not_remembered_and_a_tensor_without_an_id_is_always_checked(
    monkeypatch, expect_error
) -> None:
    contract, _ = _counting_contract(monkeypatch)
    bad = _FakeTensor(9)
    bad.bad = True
    for _ in range(2):
        with expect_error(RuntimeError, "bad tensor"):
            contract.validate_tensor(bad, placement=contracts.TensorPlacement.REPLICATED)
    assert bad.checks == 2
    anonymous = _FakeTensor(None)
    for _ in range(3):
        contract.validate_tensor(anonymous, placement=contracts.TensorPlacement.REPLICATED)
    assert anonymous.checks == 3
    # ttnn's invalid id is shared by every unassigned / moved-from tensor: never a memo key
    unassigned = _FakeTensor(contracts.INVALID_TENSOR_ID)
    for _ in range(3):
        contract.validate_tensor(unassigned, placement=contracts.TensorPlacement.REPLICATED)
    assert unassigned.checks == 3


def test_the_validation_memo_is_bounded_and_can_be_turned_off(monkeypatch) -> None:
    contract, _ = _counting_contract(monkeypatch)
    monkeypatch.setattr(contracts, "VALIDATED_MEMO_LIMIT", 4)
    for tensor_id in range(10):
        for _ in range(2):  # every tensor repeated: all candidates for the memo
            contract.validate_tensor(_FakeTensor(tensor_id), placement=contracts.TensorPlacement.REPLICATED)
    assert len(contract.__dict__["_validated"]) <= 4 and len(contract.__dict__["_validated_once"]) <= 4
    monkeypatch.setattr(contracts, "VALIDATE_EVERY", True)
    tensor = _FakeTensor(99)
    for _ in range(3):
        contract.validate_tensor(tensor, placement=contracts.TensorPlacement.REPLICATED)
    assert tensor.checks == 3


def test_the_exact_integer_and_shape_fast_paths_keep_their_refusals(expect_error) -> None:
    assert contracts._exact_integer(5, label="n") == 5
    for bad in (True, 2.5, "3", None):
        with expect_error(RuntimeError):
            contracts._exact_integer(bad, label="n")
    assert contracts._shape_tuple([1, 2, 3]) == (1, 2, 3)
    assert contracts._shape_tuple(torch.Size([4, 5])) == (4, 5)
    for bad in ([1, True], [1, 2.5], 7):
        with expect_error(RuntimeError):
            contracts._shape_tuple(bad)


def test_the_hand_off_ring_selects_are_one_persistent_tensor_per_residue(monkeypatch) -> None:
    built = []

    def from_torch(host, **kwargs):
        built.append(host)
        return SimpleNamespace(host=host)

    fake_ttnn = SimpleNamespace(
        from_torch=from_torch, bfloat16="bf16", TILE_LAYOUT="tile", DRAM_MEMORY_CONFIG="dram", deallocate=lambda t: None
    )
    monkeypatch.setattr(model_module, "ttnn", fake_ttnn)
    monkeypatch.setattr(model_module, "replicate_tensor_2d_mesh_mapper", lambda mesh: "mapper")
    model = model_module.Qwen38TTNNTextModel.__new__(model_module.Qwen38TTNNTextModel)
    model.mesh_device = "mesh"
    model._ring_select_tensors = None
    selects = model._ring_selects()
    assert len(selects) == 32 and model._ring_selects() is selects and len(built) == 32  # built once
    for residue, tensor in enumerate(selects):
        assert torch.equal(tensor.host, qsa_module.chunk_handoff_ring_select_rows(residue))
    # the select depends on prefilled % 32 only
    for prefilled in (5, 32 + 5, 4096 + 5, 30000 + 5):
        assert torch.equal(
            qsa_module.chunk_handoff_ring_select_rows(prefilled),
            qsa_module.chunk_handoff_ring_select_rows(prefilled % 32),
        )
    model._release_ring_selects()
    assert model._ring_select_tensors is None


def test_the_hand_off_allocates_nothing_and_the_selects_exist_before_any_capture() -> None:
    finish = inspect.getsource(model_module.Qwen38TTNNTextModel.finish_prefill)
    assert "from_torch" not in finish  # the selects are not uploaded per hand-off
    allocate = inspect.getsource(model_module.Qwen38TTNNTextModel.allocate_chunk_state)
    assert allocate.index("self._ring_selects()") < allocate.index("gdn.allocate_rows_constants(rows)")
    release = inspect.getsource(model_module.Qwen38TTNNTextModel.release_chunk_state)
    assert "_ring_select_users" in release and "_release_ring_selects" in release
