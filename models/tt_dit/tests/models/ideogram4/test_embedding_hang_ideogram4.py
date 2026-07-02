# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# MINIMAL REPRO — the soak+watcher run halted device 0 with a NOC error whose
# "While running kernels" section named embedding_ind_tilized.cpp (the token
# embedding), NOT SDPA: NCRISC read 4096 bytes into an L1 circular buffer that is
# too small (NOC transaction overflows a circular buffer). The Ideogram 4 encoder
# embedding (layers/embeddings.py Embedding) is ttnn.embedding(ids, weight,
# layout=TILE_LAYOUT) with weight TP-sharded on the embedding dim: 4096/tp=2 =
# 2048 bf16 = the 4096-byte per-device row that over-reads.
#
# This isolates ttnn.embedding with no fabric/CCL (embedding is a per-device op),
# so the FULL watcher runs (no fabric-ERISC overflow -> no need to disable eth).
# Brackets the trigger: sharded vs replicated embedding dim, TILE vs ROW_MAJOR
# output, short vs padded-2048 index length. A NOC-CB-overflow -> watcher halts
# and names embedding_ind_tilized; a clean pass -> that config is safe.
#
#   TT_METAL_WATCHER=5 timeout 600 \
#     pytest .../test_embedding_hang_ideogram4.py -s -q -p no:cacheprovider --timeout=0
# =============================================================================

import pytest
import torch
from loguru import logger

import ttnn

from ....layers.embeddings import Embedding
from ....utils.tensor import bf16_tensor  # noqa: F401  (kept for parity with other tests)

HIDDEN = 4096  # Qwen3-VL text hidden size (matches the failing config)
VOCAB = 4096  # small vocab -> fast build; the overflow is about the ROW size, not vocab


def _log(m):
    logger.info(m)
    print(m, flush=True)


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis"),
    [pytest.param((4, 2), (1, 2), 1, id="tp2")],
    indirect=["mesh_device"],
)
# NO fabric_config: embedding is a per-device op (no CCL), so we can run the FULL
# watcher without the fabric-ERISC kernel-config overflow. Test both the default L1
# allocator layout and the pipeline's l1_small_size=65536 (the soak's setting) — the
# CB placement under that config is the prime suspect for the over-read.
@pytest.mark.parametrize("device_params", [{}, {"l1_small_size": 65536}], ids=["l1def", "l1small64k"], indirect=True)
@pytest.mark.parametrize("layout", ["TILE", "ROW_MAJOR"], ids=["tile", "rowmajor"])
@pytest.mark.parametrize("mesh_axis", [1, None], ids=["sharded", "replicated"])
@pytest.mark.parametrize("seq_len", [32, 2048], ids=["seq32", "seq2048"])
def test_embedding_noc_cb_overflow(*, mesh_device, submesh_shape, tp_axis, layout, mesh_axis, seq_len) -> None:
    torch.manual_seed(0)
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    emb = Embedding(VOCAB, HIDDEN, device=submesh, mesh_axis=mesh_axis)
    emb.load_torch_state_dict({"weight": torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)})

    ids = torch.randint(0, VOCAB, (1, seq_len), dtype=torch.int32)
    tt_ids = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=submesh)

    out_layout = ttnn.TILE_LAYOUT if layout == "TILE" else ttnn.ROW_MAJOR_LAYOUT
    per_dev = HIDDEN // (2 if mesh_axis is not None else 1)
    _log(
        f"[emb-repro] layout={layout} mesh_axis={mesh_axis} seq_len={seq_len} per_device_row={per_dev}bf16={per_dev*2}B"
    )

    out = ttnn.embedding(tt_ids, emb.weight.data, layout=out_layout)
    ttnn.synchronize_device(submesh)  # watcher NOC-sanitize check completes here
    _log(f"[emb-repro] OK layout={layout} mesh_axis={mesh_axis} seq_len={seq_len} out_shape={tuple(out.shape)}")
    ttnn.deallocate(out)
