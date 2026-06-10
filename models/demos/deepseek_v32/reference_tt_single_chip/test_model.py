# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TT (ttnn, single-device) tests for the DeepSeek-V3.2 MLA layer, validated
against the CPU reference (``reference_cpu``) via PCC.  See spec §7.

The reference is run with ``simulate_fp8=False`` so its latent KV cache stores
plain bf16 (matching the ttnn port, which does not simulate fp8).  At the tested
sequence lengths (S <= index_topk = 2048) the indexer's additive mask is all
zero, so MLA is exercised without the indexer (wired separately, spec §8 step 4).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch

import ttnn
from models.demos.deepseek_v32.reference_cpu.model import MLACPU, IndexerCPU, ModelArgs
from models.demos.deepseek_v32.reference_cpu.utils import precompute_freqs_cis
from models.demos.deepseek_v32.reference_cpu.weights import initialize_weights
from models.demos.deepseek_v32.reference_tt_single_chip.indexer import ttIndexer
from models.demos.deepseek_v32.reference_tt_single_chip.mla import ttMLA
from models.demos.deepseek_v32.reference_tt_single_chip.utils import RopeTables
from tests.ttnn.utils_for_testing import assert_with_pcc

PREFILL_PCC = 0.99
EQUIV_PCC = 0.999
DETERMINISM_PCC = 0.9999
INDEXER_PCC = 0.99


def _build_reference(seed: int = 42):
    args = ModelArgs()
    torch.manual_seed(seed)
    mla = MLACPU(args, simulate_fp8=False)
    initialize_weights(mla)  # random
    return args, mla


def _to_device(t, mesh_device):
    return ttnn.from_torch(
        t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("batch,seq", [(1, 8), (2, 8), (1, 100)], ids=["b1s8", "b2s8", "b1s100"])
def test_mla_prefill_pcc(mesh_device, batch, seq):
    args, mla_cpu = _build_reference()
    freqs_cis = precompute_freqs_cis(args)

    torch.manual_seed(123)
    x = torch.randn(batch, seq, args.dim, dtype=torch.bfloat16)
    mask = torch.full((seq, seq), float("-inf")).triu_(1)
    with torch.no_grad():
        ref = mla_cpu.forward(x, start_pos=0, freqs_cis=freqs_cis[:seq], mask=mask)

    tt = ttMLA(args, mla_cpu.state_dict(), mesh_device)
    rope = RopeTables(args, mesh_device)
    x_tt = _to_device(x.reshape(batch, 1, seq, args.dim), mesh_device)
    mask_tt = _to_device(mask.reshape(1, 1, seq, seq), mesh_device)

    out_tt = tt.forward(x_tt, start_pos=0, rope=rope, causal_mask=mask_tt)
    out = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    out = out[:batch].reshape(batch, seq, args.dim)

    assert_with_pcc(ref.float(), out.float(), PREFILL_PCC)


def _out_to_torch(out_tt, mesh_device, batch, seq, dim):
    out = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return out[:batch].reshape(batch, seq, dim)


def _causal_mask(p):
    return torch.full((p, p), float("-inf")).triu_(1).reshape(1, 1, p, p)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("batch,seq", [(1, 8), (2, 8)], ids=["b1s8", "b2s8"])
def test_mla_prefill_decode_equivalence(mesh_device, batch, seq):
    """
    TT-internal cache-correctness invariant (spec §7 test 2): prefilling ``seq``
    tokens then comparing position ``seq-1`` against prefill(seq-1)+decode(1).
    The two paths (MHA vs MQA-absorbed) are algebraically equal.
    """
    args, mla_cpu = _build_reference()
    tt = ttMLA(args, mla_cpu.state_dict(), mesh_device)
    rope = RopeTables(args, mesh_device)

    torch.manual_seed(123)
    x = torch.randn(batch, seq, args.dim, dtype=torch.bfloat16)

    # one-shot prefill
    x_tt = _to_device(x.reshape(batch, 1, seq, args.dim), mesh_device)
    mask_tt = _to_device(_causal_mask(seq), mesh_device)
    out_prefill = _out_to_torch(tt.forward(x_tt, 0, rope, mask_tt), mesh_device, batch, seq, args.dim)

    # prefill(P-1) then decode(1) at position P-1
    p = seq - 1
    xp_tt = _to_device(x[:, :p].reshape(batch, 1, p, args.dim), mesh_device)
    maskp_tt = _to_device(_causal_mask(p), mesh_device)
    tt.forward(xp_tt, 0, rope, maskp_tt)
    xd_tt = _to_device(x[:, p : p + 1].reshape(batch, 1, 1, args.dim), mesh_device)
    out_dec = _out_to_torch(tt.forward(xd_tt, p, rope, None), mesh_device, batch, 1, args.dim)

    assert_with_pcc(out_prefill[:, p : p + 1].float(), out_dec.float(), EQUIV_PCC)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("n_runs", [3])
def test_mla_determinism(mesh_device, n_runs):
    """Same input, repeated prefill forwards must match (spec §7 test 3)."""
    args, mla_cpu = _build_reference()
    tt = ttMLA(args, mla_cpu.state_dict(), mesh_device)
    rope = RopeTables(args, mesh_device)
    batch, seq = 1, 8

    torch.manual_seed(123)
    x = torch.randn(batch, seq, args.dim, dtype=torch.bfloat16)
    x_tt = _to_device(x.reshape(batch, 1, seq, args.dim), mesh_device)
    mask_tt = _to_device(_causal_mask(seq), mesh_device)

    ref = _out_to_torch(tt.forward(x_tt, 0, rope, mask_tt), mesh_device, batch, seq, args.dim)
    for _ in range(1, n_runs):
        out = _out_to_torch(tt.forward(x_tt, 0, rope, mask_tt), mesh_device, batch, seq, args.dim)
        assert_with_pcc(ref.float(), out.float(), DETERMINISM_PCC)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("batch,seq", [(1, 8), (2, 8), (1, 100)], ids=["b1s8", "b2s8", "b1s100"])
def test_indexer_index_score_pcc(mesh_device, batch, seq):
    """
    ttIndexer ``index_score`` vs the reference on its functional path
    (``use_fp8_path=False``), via PCC (spec §7 test 4).  topk_indices shape is
    also checked; the continuous index_score is the numeric ground truth.
    """
    args = ModelArgs()
    torch.manual_seed(42)
    idx_cpu = IndexerCPU(args, use_fp8_path=False)
    initialize_weights(idx_cpu)

    torch.manual_seed(123)
    x = torch.randn(batch, seq, args.dim, dtype=torch.bfloat16)
    qr = torch.randn(batch, seq, args.q_lora_rank, dtype=torch.bfloat16)
    freqs_cis = precompute_freqs_cis(args)
    with torch.no_grad():
        _, ref_score = idx_cpu.forward(x, qr, start_pos=0, freqs_cis=freqs_cis[:seq], mask=None)

    tt = ttIndexer(args, idx_cpu.state_dict(), mesh_device)
    x_tt = _to_device(x.reshape(batch, 1, seq, args.dim), mesh_device)
    qr_tt = _to_device(qr.reshape(batch, 1, seq, args.q_lora_rank), mesh_device)
    topk_indices, score = tt.forward(x_tt, qr_tt, start_pos=0, mask=None)

    assert topk_indices.shape == (batch, seq, min(args.index_topk, seq))
    assert_with_pcc(ref_score.float(), score.float(), INDEXER_PCC)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("batch,seq", [(1, 8), (2, 8)], ids=["b1s8", "b2s8"])
def test_mla_with_indexer_sparse_pcc(mesh_device, batch, seq):
    """
    Exercise the sparse-mask wiring (spec §8 step 4): with ``index_topk < seq``
    the indexer's additive {0,-inf} mask is non-trivial.  ttMLA(with_indexer) is
    compared to the reference MLA whose nested indexer runs the same functional
    (non-fp8) path, so both select the same top-k and outputs match via PCC.
    """
    args = ModelArgs(index_topk=4)  # < seq -> non-trivial sparse selection
    torch.manual_seed(42)
    mla_cpu = MLACPU(args, simulate_fp8=False)
    initialize_weights(mla_cpu)
    mla_cpu.indexer.use_fp8_path = False  # match the ttnn functional indexer path
    freqs_cis = precompute_freqs_cis(args)

    torch.manual_seed(123)
    x = torch.randn(batch, seq, args.dim, dtype=torch.bfloat16)
    mask = torch.full((seq, seq), float("-inf")).triu_(1)
    with torch.no_grad():
        ref = mla_cpu.forward(x, start_pos=0, freqs_cis=freqs_cis[:seq], mask=mask)

    tt = ttMLA(args, mla_cpu.state_dict(), mesh_device, with_indexer=True)
    rope = RopeTables(args, mesh_device)
    x_tt = _to_device(x.reshape(batch, 1, seq, args.dim), mesh_device)
    mask_tt = _to_device(mask.reshape(1, 1, seq, seq), mesh_device)
    out = _out_to_torch(tt.forward(x_tt, 0, rope, mask_tt), mesh_device, batch, seq, args.dim)

    assert_with_pcc(ref.float(), out.float(), PREFILL_PCC)


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("batch,seq", [(1, 8)], ids=["b1s8"])
def test_mla_prefill_pretrained_pcc(mesh_device, batch, seq):
    """
    Prefill PCC against the CPU reference loaded with real DeepSeek-V3.2 layer-0
    weights (spec §7 test 5).  Skipped if the ~5 GB shard is not cached.
    """
    from huggingface_hub.utils import LocalEntryNotFoundError

    args = ModelArgs()
    mla_cpu = MLACPU(args, simulate_fp8=False)
    try:
        initialize_weights(mla_cpu, layer=0, local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip("layer-0 shard not cached; run a pretrained load once (downloads ~5 GB)")
    freqs_cis = precompute_freqs_cis(args)

    torch.manual_seed(123)
    x = torch.randn(batch, seq, args.dim, dtype=torch.bfloat16)
    mask = torch.full((seq, seq), float("-inf")).triu_(1)
    with torch.no_grad():
        ref = mla_cpu.forward(x, start_pos=0, freqs_cis=freqs_cis[:seq], mask=mask)

    tt = ttMLA(args, mla_cpu.state_dict(), mesh_device)
    rope = RopeTables(args, mesh_device)
    x_tt = _to_device(x.reshape(batch, 1, seq, args.dim), mesh_device)
    mask_tt = _to_device(mask.reshape(1, 1, seq, seq), mesh_device)
    out = _out_to_torch(tt.forward(x_tt, 0, rope, mask_tt), mesh_device, batch, seq, args.dim)

    assert_with_pcc(ref.float(), out.float(), PREFILL_PCC)
