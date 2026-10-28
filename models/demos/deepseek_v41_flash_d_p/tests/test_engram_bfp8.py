# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ENGRAM_ROWS=bfp8: the integer packer's bytes equal ttnn's float packer's, and the rows reach every chip as the
bfloat8_b that from_torch(dtype=bfloat8_b) of the exact rows gives, bit for bit (device; run_safe_pytest.sh)."""

import torch

import ttnn
from models.demos.common.bringup.reference.golden import hf_path
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
from models.demos.deepseek_v41_flash_d_p.reference.ngram import DEAD
from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint, EngramTable
from models.demos.deepseek_v41_flash_d_p.tt.engram import TtEngramRows, engram_hasher
from models.demos.deepseek_v41_flash_d_p.tt.engram_bfp8 import pack_rows, pack_rows_torch, tile_bytes

TOKENS, PAD = 2048, 9


def _rows(layer, mesh):
    path = str(hf_path(spec()))
    cfg = Config.from_dir(path)
    hasher = engram_hasher(path, cfg)
    r = TtEngramRows(mesh, hasher, EngramTable(Checkpoint(path), layer), layer, rows_format="bfp8")
    ids = torch.randint(0, cfg.vocab_size, (TOKENS,), generator=torch.Generator().manual_seed(layer))
    ids[-PAD:] = -1  # engine pad past actual_end
    return r, ids, torch.full((hasher.max_ngram_size - 1,), DEAD, dtype=torch.int64)


@mesh_parametrize
def test_bfp8_rows_on_device(mesh_device):
    for layer in (1, 14):
        r, ids, tail = _rows(layer, mesh_device)
        exps, mags, t_bfp8 = r.host_bfp8(ids, tail)
        exact, t_bf16 = r.host_rows(ids, tail)
        assert torch.equal(t_bfp8, t_bf16)
        ref_bytes = ttnn._ttnn.core.tilize_and_pack_bfp8_b(exact.float().numpy(), TOKENS, r.width)
        assert tile_bytes(exps, mags).numpy().tobytes() == ref_bytes, "integer packer != ttnn's packer"
        w, s = r.gather(torch.randint(0, r.table.num_rows, (4096,)))
        assert all(torch.equal(a, b) for a, b in zip(pack_rows(w, s, threads=4), pack_rows_torch(w, s)))
        got = r.to_device_bfp8(exps, mags)
        assert got.dtype == ttnn.bfloat8_b
        ref = ttnn.from_torch(
            exact.reshape(1, 1, TOKENS, -1),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, None)),
        )
        for g, e in zip(ttnn.get_device_tensors(got), ttnn.get_device_tensors(ref)):
            assert torch.equal(ttnn.to_torch(g), ttnn.to_torch(e)), f"layer {layer}: a chip's rows differ"
