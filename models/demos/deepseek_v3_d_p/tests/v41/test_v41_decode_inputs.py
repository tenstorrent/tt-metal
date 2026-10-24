# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The batched decode step's packed host inputs (``V41_DECODE_HOST_INPUTS=packed``): the RoPE
rows from host tables equal ``cos_sin`` at the same positions bit for bit; ``StepPacker.pack`` + ``unpack`` on the
mesh gives every chip the tensors the per-input path gave it (shapes, dtypes, bits); an Engram layer's packed
shards with ``live_rows`` equal the full path's shards on the chips that hold a live row and the zero shard on the
others. No weights; ~20 s on the Galaxy."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tt.v41.batch_decode import StepPacker
from models.demos.deepseek_v3_d_p.tt.v41.decode import ROWS
from models.demos.deepseek_v3_d_p.tt.v41.engram import PACKED_WIDTH, TtV41Engram, V41EngramTable
from models.demos.deepseek_v3_d_p.tt.v41.rope import RopeTables, cos_sin


def test_v41_rope_tables():
    """``RopeTables.rows`` against ``cos_sin`` (bf16) at random positions, the chunk boundaries and the top."""
    length = 70000
    tables = RopeTables(C, length, chunk=65536)
    g = torch.Generator().manual_seed(59)
    positions = torch.cat(
        [torch.randint(0, length, (512,), generator=g), torch.tensor([0, 1, 65535, 65536, 65537, length - 1])]
    )
    for compressed in (False, True):
        cos, sin = tables.rows(compressed, positions)
        ref_cos, ref_sin = (t.view(len(positions), -1).to(torch.bfloat16) for t in cos_sin(C, compressed, positions))
        assert cos.dtype == torch.bfloat16 and cos.shape == (len(positions), C.QK_ROPE_HEAD_DIM)
        assert torch.equal(cos.view(torch.int16), ref_cos.view(torch.int16)), compressed
        assert torch.equal(sin.view(torch.int16), ref_sin.view(torch.int16)), compressed


class _Pool:
    """``V41DecodePool.per_rank`` alone."""

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.sp, self.tp = tuple(mesh_device.shape)

    def per_rank(self, host, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=True):
        assert host.shape[0] == self.sp, (tuple(host.shape), self.sp)
        return ttnn.from_torch(
            host,
            device=self.mesh_device if device else None,
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, (self.sp, self.tp), dims=(0, None)),
        )


def _chips(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(), indirect=True)
def test_v41_step_packer(mesh_device, device_params):
    """pack -> upload -> unpack on the mesh against the per-input path (``from_torch`` of each value)."""
    pool = _Pool(mesh_device)
    sp, tp = pool.sp, pool.tp
    g = torch.Generator().manual_seed(59)
    rope_dim = C.QK_ROPE_HEAD_DIM
    ints = {
        "token": torch.randint(0, C.VOCAB_SIZE, (sp, ROWS), generator=g, dtype=torch.int32),
        "k": torch.randint(1, 33, (sp, ROWS), generator=g, dtype=torch.int32),
        "ring": torch.full((sp, ROWS), 77, dtype=torch.int32),
        "newest1": torch.randint(-1, 1000, (sp, ROWS), generator=g, dtype=torch.int32),
        "comp1": torch.randint(-1, 1000, (sp, ROWS), generator=g, dtype=torch.int32),
        "newest2": torch.randint(-1, 1000, (sp, ROWS), generator=g, dtype=torch.int32),
        "comp2": torch.randint(-1, 1000, (sp, ROWS), generator=g, dtype=torch.int32),
    }
    floats = {"p": torch.rand(sp, ROWS, generator=g), "temp": torch.rand(sp, ROWS, generator=g) * 3}
    rope = {
        k: tuple(torch.randn(sp * ROWS, rope_dim, generator=g).to(torch.bfloat16) for _ in range(2))
        for k in ("rope0", "ropec", "ropeg")
    }
    visible = {"visible1": torch.randint(0, 1000, (sp, ROWS), generator=g).float()}
    visible["visible2"] = (visible["visible1"] + 1) // 2

    packer = StepPacker(pool, sp)
    host = packer.pack(ints, floats, rope, visible)
    assert set(host) == {StepPacker.INTS, StepPacker.FLOATS, StepPacker.ROPE, StepPacker.VISIBLE}
    buffers = {k: ttnn.to_device(v, mesh_device) for k, v in host.items()}
    # a second pack of the same keys lands in the same rows and columns (the capture reads fixed slices)
    host2 = packer.pack(ints, floats, rope, visible)
    for k, v in host2.items():
        ttnn.copy_host_to_device_tensor(v, buffers[k])
    out = packer.unpack(buffers)

    def expect(k, ref, dtype, layout, shape):
        got = out[k]
        assert got.dtype == dtype and got.layout == layout, (k, got.dtype, got.layout)
        assert tuple(got.shape) == shape, (k, tuple(got.shape), shape)
        for chip, (mine, theirs) in enumerate(zip(_chips(got), _chips(ref))):
            assert mine.shape == theirs.shape, (k, chip, mine.shape, theirs.shape)
            if mine.dtype.is_floating_point:
                mine, theirs = mine.view(torch.int16), theirs.view(torch.int16)
            assert torch.equal(mine, theirs), (k, chip)

    def flat(value, dtype):  # the sampler's per-row tensors: [sp * ROWS] sharded over the ranks
        shard = ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, None))
        return ttnn.from_torch(value.reshape(-1), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=shard)

    for k, v in ints.items():
        if k == "k":
            expect(k, flat(v, ttnn.uint32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (ROWS,))
        else:
            ref = pool.per_rank(v.view(sp, 1, 1, ROWS), ttnn.uint32, device=False)
            expect(k, ref, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (1, 1, 1, ROWS))
    for k, v in floats.items():
        expect(k, flat(v, ttnn.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (ROWS,))
    for k, (cos, sin) in rope.items():
        got = out[k]
        assert isinstance(got, tuple) and len(got) == 2, k
        for j, t in enumerate((cos, sin)):
            out[f"{k}.{j}"] = got[j]
            ref = pool.per_rank(t.view(sp, 1, ROWS, rope_dim), ttnn.bfloat16, ttnn.TILE_LAYOUT, device=False)
            expect(f"{k}.{j}", ref, ttnn.bfloat16, ttnn.TILE_LAYOUT, (1, 1, ROWS, rope_dim))
    for k, v in visible.items():
        ref = pool.per_rank(v.view(sp, 1, ROWS, 1), ttnn.float32, ttnn.TILE_LAYOUT, device=False)
        expect(k, ref, ttnn.float32, ttnn.TILE_LAYOUT, (1, 1, ROWS, 1))
    # the embedding's token shape
    token = ttnn.reshape(out["token"], (1, 1, ROWS))
    assert tuple(token.shape) == (1, 1, ROWS) and torch.equal(
        _chips(token)[tp].view(-1).to(torch.int64), ints["token"][1].to(torch.int64)
    )


@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(), indirect=True)
def test_v41_engram_live_rows(mesh_device, device_params):
    """``TtV41Engram._packed_shards`` with ``live_rows`` against the full path on a small random table."""
    sp, tp = tuple(mesh_device.shape)
    chips = sp * tp
    g = torch.Generator().manual_seed(59)
    rows, head_dim = 4096, C.ENGRAM_HEAD_DIM
    weight = torch.randn(rows, head_dim, generator=g).to(torch.float8_e4m3fn)
    scale = (torch.rand(rows, head_dim // 32, generator=g) * 4).to(torch.float8_e8m0fnu)
    e = TtV41Engram.__new__(TtV41Engram)
    e.table, e.chips, e.sp, e.tp, e._zero_shards = V41EngramTable(weight, scale), chips, sp, tp, {}
    cols = (C.ENGRAM_MAX_NGRAM_SIZE - 1) * C.ENGRAM_N_HEADS
    chunk = ROWS * sp
    lookups = chunk * cols
    per_chip = lookups // chips
    rows_per_chip = per_chip // cols
    hash_ids = torch.randint(0, rows, (chunk, cols), generator=g)
    live = torch.tensor(sorted({0, 3, ROWS + 1, 2 * ROWS, chunk - 1}))
    full, _ = e._packed_shards(hash_ids, lookups)
    part, keep = e._packed_shards(hash_ids, lookups, live_rows=live)
    assert keep.shape == (len({int(r) // rows_per_chip for r in live}) * per_chip, PACKED_WIDTH)
    live_chips = {int(r) // rows_per_chip for r in live}
    for c, (a, b) in enumerate(zip(_chips(full), _chips(part))):
        assert a.shape == b.shape == (1, 1, per_chip, PACKED_WIDTH), (c, a.shape, b.shape)
        if c in live_chips:
            assert torch.equal(a, b), c
        else:
            assert not b.any(), c
