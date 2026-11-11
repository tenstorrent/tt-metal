# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host tests of the PUP tt-d-gen prefill runner pieces (no device): config, GDN Snapshot geometry and host decoders,
the KV chunk address table, media from stored values, adapter descriptors."""

import types

import numpy as np
import pytest
import torch

from models.demos.blackhole.qwen36.tt.runners import gdn_state as gs
from models.demos.blackhole.qwen36.tt.runners import kv_table
from models.demos.blackhole.qwen36.tt.runners import media as pm
from models.demos.blackhole.qwen36.tt.runners.config import DEFAULT_CONFIG, PupPrefillConfig

G = gs.GdnGeometry()  # Qwen3.5-9B on a (1, 4) column


# --- config -------------------------------------------------------------------------------------------------------


class _Args:
    def __init__(self):
        self.profile_calls = 0
        self.cluster_shape = [1, 4]

    def set_column_prefill_profile(self):
        import ttnn

        self.profile_calls += 1
        self.mlp_math_fidelity = ttnn.MathFidelity.LoFi
        self.prefill_tokpar_mlp_cfg2 = True
        self.prefill_tokpar_mlp_l1 = True
        self.prefill_tokpar_res_l1 = True
        self.prefill_media_source = "in_prefill"


def test_config_applies_profile_then_mlp_fix():
    import ttnn

    a = _Args()
    DEFAULT_CONFIG.apply(a)
    assert a.profile_calls == 1
    assert a.mlp_math_fidelity == ttnn.MathFidelity.HiFi2
    assert a.prefill_tokpar_mlp_cfg2 is False
    assert a.prefill_tokpar_mlp_l1 == 1152 and a.prefill_tokpar_res_l1 == 3072
    assert a.prefill_media_source == "spliced"


def test_config_serving_and_validation():
    c = DEFAULT_CONFIG.with_serving(num_users=8, max_seq_len=8192)
    assert (c.num_users, c.max_seq_len, c.chunk_size, c.blocks_per_slot) == (8, 8192, 4096, 128)
    assert c.mlp_fidelity == "hifi2"
    with pytest.raises(ValueError):
        PupPrefillConfig(chunk_size=8192, max_seq_len=4096)
    with pytest.raises(ValueError):
        PupPrefillConfig(mlp_fidelity="hifi9")
    assert DEFAULT_CONFIG.fast and DEFAULT_CONFIG.with_serving(fast=False).fast is False
    with pytest.raises(ValueError):
        PupPrefillConfig(fast_buckets=(100,))


def test_fast_runtime_system_turn_boundary():
    """The snapshot boundary: the system turn (through the first <|im_end|>) floored to 128; no snapshot when the
    turn is shorter than 128 or longer than prefix_max, an image token sits before it, or the tail is < 3 tokens."""
    from models.demos.blackhole.qwen36.tt.runners.fast_runtime import TtQwen35PupFastPrefillRuntime

    rt = object.__new__(TtQwen35PupFastPrefillRuntime)
    rt.prefix_end_token, rt.prefix_align, rt.prefix_max, rt.image_token = 7, 128, 1024, 9
    ids = torch.zeros(1, 900, dtype=torch.int32)
    ids[0, 600] = 7  # system turn = [0, 601)
    assert rt._prefix_len(ids) == 512
    ids2 = ids.clone()
    ids2[0, 700] = 7  # only the first <|im_end|> counts
    assert rt._prefix_len(ids2) == 512
    ids3 = ids.clone()
    ids3[0, 100] = 9  # an image inside the system turn: no snapshot
    assert rt._prefix_len(ids3) == 0
    tail = torch.zeros(1, 513, dtype=torch.int32)
    tail[0, 511] = 7  # P = 512 leaves a 1-token tail
    assert rt._prefix_len(tail) == 0
    short = torch.zeros(1, 300, dtype=torch.int32)
    short[0, 100] = 7
    assert rt._prefix_len(short) == 0  # < 128
    rt.prefix_end_token = None
    assert rt._prefix_len(ids) == 0


@pytest.mark.parametrize(
    "flag, rows_per_chip, l1",
    [(False, 100, False), (True, 10_000, True), (1152, 288, True), (1152, 352, False), (3072, 768, True), (3072, 800, False)],
)
def test_l1_rows_switch(flag, rows_per_chip, l1):
    from models.demos.blackhole.qwen36.tt.layer import _l1_rows_ok

    assert _l1_rows_ok(flag, rows_per_chip, types.SimpleNamespace(cluster_shape=[1, 4])) is l1


# --- GDN Snapshot geometry ----------------------------------------------------------------------------------------


def test_geometry_counts():
    assert G.recurrent_shape == (1, 8, 128, 128)
    assert G.convolution_channels == 2048  # q 512 | k 512 | v 1024 per chip
    assert G.recurrent_segments_per_layer == 128 and G.convolution_segments_per_layer == 128
    assert G.recurrent_shards_per_layer == 32 and G.convolution_shards_per_layer == 32
    assert G.recurrent_segment_bytes == 16384 and G.convolution_segment_bytes == 384
    assert G.window == 4096
    assert G.chunk_n_tokens("gdn_recurrent") == 32 and G.chunk_n_tokens("gdn_convolution") == 32


@pytest.mark.parametrize("kind", ["gdn_recurrent", "gdn_convolution"])
def test_segments_partition_over_columns(kind):
    seen = {}
    for c in range(4):
        shards = set()
        for seg, shard in G.local_segments(kind, c):
            assert seg not in seen, f"segment {seg} on two columns"
            seen[seg] = c
            shards.add(shard)
        assert shards == set(range(G.shards_per_layer(kind)))
    assert sorted(seen) == list(range(G.segments_per_layer(kind)))


def test_segment_maps_roundtrip():
    for c in range(4):
        for h in range(8):
            for band in range(4):
                assert G.decompose_recurrent(G.recurrent_segment(c, h, band)) == (c, h, band)
        for branch in range(3):
            for h in range(G.branch_local_heads(branch)):
                for half in range(2):
                    assert G.decompose_convolution(G.convolution_segment(branch, c, h, half)) == (branch, c, h, half)


def _hf_conv_layout_channel(branch, g, d):
    """HF conv channel index of (branch, global head, dim) in [all q | all k | all v]."""
    return (G.branch_base(branch) + g) * G.head_dim + d


def test_convolution_native_to_hf_through_the_slab():
    """Per-chip native [3, 2048] rows -> slab rows -> 384 B segments -> assembled == the HF conv layout."""
    hf = torch.randn(3, (2 * 16 + 32) * 128).to(torch.bfloat16)
    segments = {}
    for c in range(4):
        # chip c's native row: [q heads 4c..4c+3 | k heads 4c.. | v heads 8c..8c+7]
        parts = []
        for branch in range(3):
            n = G.branch_local_heads(branch)
            g0 = c * n
            a = _hf_conv_layout_channel(branch, g0, 0)
            parts.append(hf[:, a : a + n * 128])
        native = torch.cat(parts, dim=1)
        slab = gs.native_convolution_to_slab_rows(native, G)
        assert tuple(slab.shape) == (32, 192)
        for seg, shard in G.local_segments("gdn_convolution", c):
            raw = slab[shard].contiguous().view(torch.int16).numpy().tobytes()
            segments[seg] = gs.convolution_segment_to_torch(raw, G)
    out = gs.assemble_convolution(segments, G)
    assert torch.equal(out, hf)
    st = gs.hf_conv_state(out)
    assert tuple(st.shape) == (8192, 4) and torch.equal(st[:, 3], hf[2]) and not st[:, 0].any()


def test_recurrent_tile_bytes_roundtrip():
    rec = torch.randn(32, 128, 128)  # HF recurrent_state [heads, Dk, Dv]
    segments = {}
    for c in range(4):
        for h in range(8):
            for band in range(4):
                raw = gs.tile_bytes_fp32(rec[c * 8 + h][:, band * 32 : (band + 1) * 32])
                segments[G.recurrent_segment(c, h, band)] = gs.recurrent_segment_to_torch(raw, G)
    assert torch.equal(gs.assemble_recurrent(segments, G), rec)


# --- KV chunk address table ---------------------------------------------------------------------------------------


class _Rec:
    """A stand-in for ttnn.experimental.disaggregation recording the table."""

    class KvChunkTensorType:
        Kv, Snapshot = "Kv", "Snapshot"

    class KvChunkAddressTableConfig:
        tensor_type = "Kv"

    class KvCacheLocation:
        pass

    class KvChunkAddressTable:
        def __init__(self, configs):
            self.configs, self.groups, self.entries, self.hosts = configs, [], {}, {}

        def add_device_group(self, members):
            self.groups.append(list(members))
            return len(self.groups) - 1

        def set_fabric_node_host(self, fid, host_name):
            self.hosts[(fid.mesh_id, fid.chip_id)] = host_name

        def set(self, layer, position, slot, loc, config_id):
            key = (config_id, layer, position, slot)
            assert key not in self.entries, f"duplicate entry {key}"
            self.entries[key] = (loc.noc_addr, loc.size_bytes, loc.device_group_index)


def _fid(chip):
    return types.SimpleNamespace(mesh_id=0, chip_id=chip)


ATTN = tuple(range(3, 32, 4))
GDN = tuple(i for i in range(32) if i not in ATTN)


def _build(num_users=2, max_seq_len=256):
    fnids = [[_fid(c) for c in range(4)]]
    bases = {"k": 1 << 20, "v": 2 << 20, "gdn_recurrent": 3 << 20, "gdn_convolution": 4 << 20, "slot_params": 5 << 20}
    counts = {"k": 8, "v": 8, "gdn_recurrent": 24, "gdn_convolution": 24, "slot_params": 32}
    layouts = [
        [dict(first_layer=0, count=counts[k], base_addr=bases[k], num_banks=8, host_name="lb", fnids=fnids)]
        for k in kv_table.KINDS
    ]
    t = kv_table.build_table(
        stage_layouts=layouts,
        num_users=num_users,
        max_seq_len=max_seq_len,
        head_dim=256,
        num_kv_heads=4,
        attn_layer_ids=ATTN,
        gdn_layer_ids=GDN,
        num_model_layers=32,
        geometry=G,
        disagg=_Rec,
    )
    return t, bases


def test_table_configs():
    t, _ = _build()
    names = sorted(t.configs)
    assert names == [f"{i:02d}" for i in range(11)]
    for i, name in enumerate(names):
        cfg = t.configs[name]
        assert cfg.num_layers == 32 and cfg.num_slots == 2
        if i < 8:
            assert (cfg.chunk_n_tokens, cfg.chunk_size_bytes, cfg.max_sequence_length) == (32, 8704, 256)
            assert cfg.tensor_type == "Kv"
        elif i < 10:
            assert cfg.tensor_type == "Snapshot" and cfg.chunk_n_tokens == 32 and cfg.max_sequence_length == 4096
    assert t.configs["08"].chunk_size_bytes == 16384 and t.configs["09"].chunk_size_bytes == 384
    c10 = t.configs["10"]
    assert (c10.tensor_type, c10.chunk_n_tokens, c10.chunk_size_bytes, c10.max_sequence_length) == (
        "Snapshot", 32, 64, 128)


def test_table_slot_record():
    """Config 10: every layer row, one 64 B chunk per chip (chunk c on chip c), page layer * slots + slot."""
    t, bases = _build()
    for chip in range(4):
        page = 17 * 2 + 1
        addr, size, grp = t.entries[(10, 17, chip * 32, 1)]
        assert size == 64 and addr == ((page % 8) << 32) | (bases["slot_params"] + (page // 8) * 64)
        assert [f.chip_id for f in t.groups[grp]] == [chip]
    assert {k[1] for k in t.entries if k[0] == 10} == set(range(32))
    assert sum(1 for k in t.entries if k[0] == 10) == 32 * 2 * 4
    rec = kv_table.slot_record(-2400, 1.5)
    assert int(rec[0]) == -2400 and rec[1:2].view(torch.float32).item() == 1.5
    assert kv_table.mrope_delta([[1, 32, 32]]) == -240 and kv_table.mrope_delta([[1, 64, 64]] ) == -992


def test_table_kv_addresses_follow_the_nd_round_robin():
    t, bases = _build()
    # v (config 4 + head), slot 1, attention layer 11 (attention position 2), position 64 (chunk 2 of 8)
    head = 3
    shard = (1 * 8 + 2) * (256 // 32) + 2
    addr, size, grp = t.entries[(4 + head, 11, 64, 1)]
    assert size == 8704 and addr == ((shard % 8) << 32) | (bases["v"] + (shard // 8) * 8704)
    assert [f.chip_id for f in t.groups[grp]] == [head]
    # attention configs publish attention layers only, every position of every slot
    rows = {k[1] for k in t.entries if k[0] < 8}
    assert rows == set(ATTN)
    assert sum(1 for k in t.entries if k[0] == 0) == 2 * 8 * 8


def test_table_gdn_addresses():
    t, bases = _build()
    c, h, band = 2, 3, 1
    seg = G.recurrent_segment(c, h, band)
    gdn_pos = 5  # layer GDN[5]
    s = (1 * 24 + gdn_pos) * 32 + G.recurrent_local_shard(h, band)
    addr, size, grp = t.entries[(8, GDN[gdn_pos], seg * 32, 1)]
    assert size == 16384 and addr == ((s % 8) << 32) | (bases["gdn_recurrent"] + (s // 8) * 16384)
    assert [f.chip_id for f in t.groups[grp]] == [c]
    branch, half = 2, 0  # a v head
    seg = G.convolution_segment(branch, c, h, half)
    s = (0 * 24 + 0) * 32 + G.convolution_local_shard(branch, h, half)
    addr, size, _ = t.entries[(9, GDN[0], seg * 32, 0)]
    assert size == 384 and addr == ((s % 8) << 32) | (bases["gdn_convolution"] + (s // 8) * 384)
    # one full window per (layer, slot): 128 segments of each kind
    for cid in (8, 9):
        assert sum(1 for k in t.entries if k[0] == cid and k[1] == GDN[3] and k[3] == 1) == 128
    assert {k[1] for k in t.entries if k[0] in (8, 9)} == set(GDN)


def test_cache_kind_and_position_range():
    from models.demos.blackhole.qwen36.tt.runners.adapter import Qwen35PupPrefillAdapter, layer_ids

    a = Qwen35PupPrefillAdapter()
    assert [a.cache_kind(i) for i in range(11)] == ["kv"] * 8 + ["gdn_recurrent", "gdn_convolution", "slot_params"]
    assert a.layer_position_range(3, 900) == (0, 900)
    assert a.layer_position_range(0, 900) == (0, 12288)  # the ring form (R.window)
    attn, gdn = layer_ids(32, ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8)
    assert attn == ATTN and gdn == GDN
    assert layer_ids(32) == (ATTN, GDN)


def test_adapter_registered():
    from models.demos.common.prefill.adapter import ADAPTER_PATHS

    assert ADAPTER_PATHS["qwen35_9b_pup"].endswith("runners.adapter:Qwen35PupPrefillAdapter")


# --- media from stored values -------------------------------------------------------------------------------------

IMG = 248056


def _value(n, grid, seed=0):
    rows = np.random.default_rng(seed).integers(0, 1 << 16, size=(n, 64), dtype=np.uint16)
    return rows, pm.stored_value(rows, grid)


def _ref(chunk, s, e, key):
    return types.SimpleNamespace(chunk_index=chunk, start_pos=s, end_pos=e, media_key=key)


def test_media_one_batched_read_rows_and_grids_in_place():
    r1, v1 = _value(64, (1, 16, 16), 1)
    r2, v2 = _value(16, (1, 8, 8), 2)
    reader = pm.DictEmbeddingReader({b"a" * 32: v1, b"b" * 32: v2})
    ids = [7] * 10 + [IMG] * 64 + [7] * 3 + [IMG] * 16 + [7] * 5
    refs = pm.check_refs([_ref(0, 77, 93, b"b" * 32), _ref(0, 10, 74, b"a" * 32)], 0, len(ids), ids, IMG)
    assert [int(r.start_pos) for r in refs] == [10, 77]
    out = pm.fetch_values(reader, refs)
    assert reader.fetches == 1
    assert [h.grid_thw for h, _ in out] == [(1, 16, 16), (1, 8, 8)]
    assert np.array_equal(out[0][1], r1) and np.array_equal(out[1][1], r2)


def test_media_refs_of_a_chunk():
    refs = [_ref(0, 0, 4, b"x"), _ref(1, 4, 8, b"y"), _ref(1, 9, 12, b"z")]
    assert [r.media_key for r in pm.refs_in_chunk(refs, 1)] == [b"y", b"z"]
    assert pm.refs_in_chunk(None) == []


@pytest.mark.parametrize(
    "ref, why",
    [
        (_ref(0, 60, 130, b"a" * 32), "not inside the chunk"),  # straddles the chunk end
        (_ref(0, 0, 64, b"a" * 32), "not a run of image tokens"),
    ],
)
def test_media_refused(ref, why):
    ids = [7] * 10 + [IMG] * 64 + [7] * 26
    with pytest.raises(pm.MediaSpanError, match=why):
        pm.check_refs([ref], 0, 100, ids, IMG)


def test_media_wrong_size_or_missing():
    _, v = _value(64, (1, 16, 16))
    reader = pm.DictEmbeddingReader({b"a" * 32: v})
    with pytest.raises(pm.MediaSpanError, match="stored 64 rows for a span of 32"):
        pm.fetch_values(reader, [_ref(0, 10, 42, b"a" * 32)])
    with pytest.raises(KeyError):
        pm.fetch_values(reader, [_ref(0, 10, 74, b"c" * 32)])


# --- the decode ring's forms (tt-blaze blaze.models.qwen35_9b.cache_migration) ------------------------------------

R = gs.GdnGeometry(conv_per_value_head=True)


def test_ring_form_geometry():
    """The ring runs one q / k head per value head: 192 conv chunks a layer, window lcm(128, 192) * 32 with strides
    96 / 64, the numbers the ring's table publishes."""
    assert R.convolution_channels == 3 * 1024 and R.native_convolution_channels == 2048
    assert R.recurrent_segments_per_layer == 128 and R.convolution_segments_per_layer == 192
    assert R.convolution_shards_per_layer == 48
    assert R.window == 12288
    assert R.chunk_n_tokens("gdn_recurrent") == 96 and R.chunk_n_tokens("gdn_convolution") == 64
    # ring numbering: conv segment (branch * 32 + g) * 2 + half, g = 8c + h, in every branch
    for branch in range(3):
        for c in range(4):
            for h in range(8):
                for half in range(2):
                    assert R.convolution_segment(branch, c, h, half) == (branch * 32 + 8 * c + h) * 2 + half


def test_ring_form_conv_segments_hold_the_key_head_of_each_value_head():
    """Native [3, 2048] rows per chip -> ring-form slab -> segments: q / k of value head g = HF key head g // 2."""
    hf = torch.randn(3, (2 * 16 + 32) * 128).to(torch.bfloat16)
    segments = {}
    for c in range(4):
        parts = []
        for branch in range(3):
            n = G.branch_local_heads(branch)
            a = _hf_conv_layout_channel(branch, c * n, 0)
            parts.append(hf[:, a : a + n * 128])
        slab = gs.native_convolution_to_slab_rows(torch.cat(parts, dim=1), R)
        assert tuple(slab.shape) == (48, 192)
        for seg, shard in R.local_segments("gdn_convolution", c):
            raw = slab[shard].contiguous().view(torch.int16).numpy().tobytes()
            segments[seg] = gs.convolution_segment_to_torch(raw, R)
    assert len(segments) == 192
    for seg, rect in segments.items():
        branch, c, h, half = R.decompose_convolution(seg)
        g = 8 * c + h
        src_head = g // 2 if branch < 2 else g
        a = _hf_conv_layout_channel(branch, src_head, half * 64)
        assert torch.equal(rect, hf[:, a : a + 64]), (branch, g, half)
    out = gs.per_value_head_to_native(gs.assemble_convolution(segments, R), R)
    assert torch.equal(out, hf)


def test_k_interleaved_order_is_the_ring_head_perm():
    """The published K is in the ring's interleaved-pair RoPE order: ring dim 2j = HF dim j, 2j + 1 = HF dim 32 + j
    for j < 32, the rest unchanged (tt-blaze ``blaze.weights.qwen35_9b.rope.head_perm``)."""
    perm = kv_table.interleaved_k_perm()
    assert perm[:6].tolist() == [0, 32, 1, 33, 2, 34] and perm[62:66].tolist() == [31, 63, 64, 65]
    assert perm[64:].tolist() == list(range(64, 256))
    k = torch.randn(2, 3, 256)
    assert torch.equal(kv_table.rotate_half_to_interleaved_host(k), k[..., perm])
