# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``Qwen38PlainLaneDecoder`` on shared page pools (the chain's own K/V and index pools) over a fake model, pagers and
importer (no device): the constructor, the layouts, the import without the KV admission, the pack off, the host import
runs on the fixed-size states."""

from types import SimpleNamespace

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_batch_decode as batch
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import Qwen38PlainLaneDecoder
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging

PAGING = Qwen38LanePaging(64, 20, 128)
POOLS = {0: "pools0", 1: "pools1"}
LANES = 4


class FakeModel:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "allocate_lane_state":
                return SimpleNamespace(name="ls")
            if name == "generic_lane_layout":
                return "gl"
            if name == "lane_layout":
                return "ll"
            if name == "resolve_lane_tokens":
                return SimpleNamespace(local_indices=None, local_values=None, packed=None, shard_row=None), None
            if name == "forward_decode_lanes":
                return SimpleNamespace(release_tensors=lambda: None, logits="logits")
            return None

        return call

    def named(self, name):
        return [c for c in self.calls if c[0] == name]


class Rig:
    def __init__(self, monkeypatch, shared=True, chain_paging=PAGING, **kwargs):
        monkeypatch.delenv("QWEN38_IMPORT_HOST", raising=False)
        monkeypatch.delenv("QWEN38_PREFILL_PACK", raising=False)
        self.events = []
        events = self.events
        monkeypatch.setattr(batch.ttnn, "synchronize_device", lambda mesh: None, raising=False)

        class Pager:
            def __init__(self, mesh, contract, layout, buffers=None):
                self.layout = layout

            def warm_moves(self, slot):
                pass

            def evict(self, lane, slot):
                events.append(("evict", lane))

            def rotate_ring_canonical(self, slot, length):
                events.append(("rotate", length))

            def readmit(self, slot, lane, position_row=None):
                events.append(("readmit", lane))

        class Pool:
            def __init__(self, mesh, layout, slots=1):
                self.slots = [SimpleNamespace(position=0)]

        class Importer:
            def __init__(self, pager, layout):
                self.layout = layout

            def import_lane(self, lane, length):
                events.append(("import", lane, length))

        monkeypatch.setattr(batch, "Qwen38TTNNLanePager", Pager)
        monkeypatch.setattr(batch, "Qwen38LaneHostPool", Pool)
        monkeypatch.setattr(batch, "Qwen38TTNNLaneImporter", Importer)
        self.model = FakeModel()
        self.chain = SimpleNamespace(
            mesh="mesh",
            state="state",
            built_target=SimpleNamespace(model=self.model),
            sampling=None,
            paging=chain_paging,
            pools=POOLS,
            write_page_table=lambda ids: events.append(("table", list(ids))) or True,
        )
        self.decoder = Qwen38PlainLaneDecoder(LANES, paging=PAGING, shared_pools=shared, **kwargs)


def test_the_constructor_needs_paging_for_shared_pools(expect_error):
    with expect_error(ValueError, "need paging"):
        Qwen38PlainLaneDecoder(LANES, shared_pools=True)
    assert Qwen38PlainLaneDecoder(LANES, paging=PAGING).shared_pools is False
    assert Qwen38PlainLaneDecoder(LANES).shared_pools is False


def test_prepare_passes_the_pools_and_excludes_kv_and_compressed(monkeypatch):
    rig = Rig(monkeypatch)
    assert rig.decoder.shared_pools is True
    assert rig.decoder.pools is None
    rig.decoder.prepare(rig.chain)
    assert rig.decoder.pools is POOLS
    ((_, args, kwargs),) = rig.model.named("allocate_lane_state")
    assert kwargs["paged"] is PAGING and kwargs["pools"] is POOLS
    ((_, gargs, gkwargs),) = rig.model.named("generic_lane_layout")
    assert gkwargs == {"exclude_kv": True, "exclude_compressed": True}
    assert rig.decoder.importer.layout == "gl"
    # the compile-warm pass still runs the (no-op) admission and the import on every lane
    assert [e for e in rig.events if e[0] == "import"] == [("import", lane, lane % 4) for lane in range(LANES)]
    assert rig.model.named("admit_lane_kv_paged")


def test_unshared_prepare_keeps_the_per_state_pools(monkeypatch):
    rig = Rig(monkeypatch, shared=False)
    rig.decoder.prepare(rig.chain)
    ((_, _, kwargs),) = rig.model.named("allocate_lane_state")
    assert "pools" not in kwargs
    ((_, _, gkwargs),) = rig.model.named("generic_lane_layout")
    assert gkwargs == {"exclude_kv": True}


def test_import_skips_the_kv_admission_with_shared_pools(monkeypatch):
    rig = Rig(monkeypatch)
    decoder = rig.decoder
    decoder.prepare(rig.chain)
    rig.model.calls.clear()
    rig.events.clear()
    decoder.page_rows[1] = [3, 4]
    decoder.move_into_lane(1, 37)
    assert not rig.model.named("admit_lane_kv_paged")
    assert [c[0] for c in rig.model.calls] == ["admit_lane_position"]
    assert rig.events == [("import", 1, 37)]


def test_unshared_import_admits_the_kv(monkeypatch):
    rig = Rig(monkeypatch, shared=False)
    rig.decoder.prepare(rig.chain)
    rig.model.calls.clear()
    rig.decoder.move_into_lane(1, 37)
    assert len(rig.model.named("admit_lane_kv_paged")) == 1


def test_the_host_import_runs_with_shared_pools(monkeypatch):
    monkeypatch.setenv("QWEN38_IMPORT_HOST", "1")
    assert Qwen38PlainLaneDecoder(LANES, paging=PAGING, shared_pools=True).host_import
    rig = Rig(monkeypatch)  # the rig clears the env
    rig.decoder.prepare(rig.chain)
    rig.decoder.host_import = True
    rig.model.calls.clear()
    rig.events.clear()
    rig.decoder.move_into_lane(0, 5)
    assert [e[0] for e in rig.events] == ["evict", "rotate", "readmit"]
    assert len(rig.model.named("admit_lane")) == 1
    assert not rig.model.named("admit_lane_kv_paged")


def test_the_pack_switch_ignores_shared_pools(expect_error):
    environ = {"QWEN38_PREFILL_PACK": "1"}
    assert batch.prefill_pack_enabled(environ, lanes=32, shared_pools=True) is True
    assert batch.prefill_pack_enabled({}, lanes=32, shared_pools=True) is True
    assert batch.prefill_pack_enabled({"QWEN38_PREFILL_PACK": "0"}, lanes=32, shared_pools=True) is False
    with expect_error(ValueError, "must be 0 or 1"):
        batch.prefill_pack_enabled({"QWEN38_PREFILL_PACK": "x"}, lanes=32, shared_pools=True)


def test_shared_pool_pack_buckets_keep_rows_of_at_least_one_block(monkeypatch):
    monkeypatch.delenv("QWEN38_PACK_BUCKETS", raising=False)
    monkeypatch.setenv("QWEN38_PREFILL_PACK", "1")
    assert batch.pack_buckets_for_block((128, 256, 512, 1024), 256) == (256, 512, 1024)
    decoder = Qwen38PlainLaneDecoder(32, paging=PAGING, shared_pools=True)  # block 64: every bucket fits
    assert decoder.pack_enabled and decoder.pack_buckets == (128, 256, 512, 1024)
    paging = Qwen38LanePaging(512, 20, 128)
    decoder = Qwen38PlainLaneDecoder(32, paging=paging, shared_pools=True)
    assert decoder.pack_enabled and decoder.pack_buckets == (512, 1024)
    decoder = Qwen38PlainLaneDecoder(32, paging=Qwen38LanePaging(2048, 20, 128), shared_pools=True, pack=True)
    assert decoder.pack_enabled is False and decoder.pack_buckets == ()
    assert Qwen38PlainLaneDecoder(32, paging=paging).pack_buckets == (128, 256, 512, 1024)  # unshared: unfiltered


def test_admit_pack_writes_the_flat_pack_row_before_the_replay(monkeypatch):
    import torch

    rig = Rig(monkeypatch)
    decoder = rig.decoder
    decoder.prepare(rig.chain)
    decoder.pack_enabled = True
    rig.events.clear()
    events = rig.events
    monkeypatch.setattr(batch.TraceAllocationTracker, "verify_before_replay", lambda mesh, tid: None)
    users = SimpleNamespace(users=16)
    decoder.pack_state = SimpleNamespace(buckets={128: users})
    decoder.pack_traces = {128: (1, "out")}
    decoder.lander = SimpleNamespace(land=lambda assignments, rows: events.append(("land", assignments, rows)))

    def run_pack(state, rows, chain_state, prompts, **kwargs):
        events.append(("run_pack", rows))
        kwargs["timing"].update(prepare_ms=0.0, upload_ms=0.0, replay_ms=0.0)
        kwargs["info"]["contexts"] = ["c"] * len(prompts)
        return torch.zeros(len(prompts), 8)

    model = rig.model
    model.run_pack = run_pack
    model.write_pack_page_table = lambda state, blocks, rows: events.append(("pack_table", state, blocks, rows))
    decoder.admit_pack([2, 0], [[1] * 5, [2] * 7], 128, pages=[[11, 12], [13]], users=[1, 0])
    kinds = [e[0] for e in events]
    assert kinds.index("pack_table") < kinds.index("run_pack") < kinds.index("land")
    ((_, state, blocks, rows),) = [e for e in events if e[0] == "pack_table"]
    assert state == "state" and rows == 128
    assert blocks == [[13], [11, 12]]  # user 0 sits in lane 0, user 1 in lane 2; 128 rows = two 64-row pages
    assert decoder.positions[2] == 7 and decoder.positions[0] == 5


def test_prepare_refuses_a_paging_mismatch(monkeypatch, expect_error):
    rig = Rig(monkeypatch, chain_paging=Qwen38LanePaging(64, 21, 128))
    with expect_error(ValueError, "paging"):
        rig.decoder.prepare(rig.chain)


def test_admit_passes_the_lane_page_table_to_the_prefill(monkeypatch):
    rig = Rig(monkeypatch)
    decoder = rig.decoder
    decoder.prepare(rig.chain)
    rig.events.clear()
    events = rig.events
    slot = SimpleNamespace(ple_context="ctx", committed=[1, 2, 3])
    import torch

    seen = {}

    def prefill_forward(**kwargs):
        events.append(("prefill",))
        seen.update(kwargs)
        return torch.tensor([7]), None

    decoder.vm = SimpleNamespace(
        prefill_forward=prefill_forward, slot=slot, release_request=lambda i: events.append(("release",))
    )
    assert decoder.admit(2, [1, 2, 3], page_count=2) == 7
    kinds = [e[0] for e in events]
    assert kinds.index("prefill") < kinds.index("import")
    assert "table" not in kinds
    ids = decoder.page_rows[2]
    assert len(ids) == 2 and 0 not in ids
    table = seen["page_table"]
    assert table.dtype == torch.int32 and tuple(table.shape) == (1, PAGING.max_blocks)
    assert table[0].tolist() == ids + [0] * (PAGING.max_blocks - 2)


def test_admit_releases_the_pages_when_the_prefill_fails(monkeypatch, expect_error):
    rig = Rig(monkeypatch)
    decoder = rig.decoder
    decoder.prepare(rig.chain)
    free = decoder.free_pages()

    def prefill_forward(**kwargs):
        raise RuntimeError("prefill failed")

    decoder.vm = SimpleNamespace(prefill_forward=prefill_forward)
    with expect_error(RuntimeError, "prefill failed"):
        decoder.admit(2, [1, 2, 3], page_count=2)
    assert decoder.page_rows[2] == []
    assert decoder.free_pages() == free
