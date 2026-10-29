"""DFlash2Drafter (fixed-shape, KV-cached) vs the validated DFlash2Draft oracle on REAL target taps.

No 27B needed: uses the captured real taps (profiles/dflash2_real.npz, 128-token prompt) and the
golden draft_hidden (profiles/dflash2_golden_real.npz). Checks the three context paths agree with
each other and with the oracle/golden:

  B1  fill_context(all 128 rows)                                -> draft
  B2  fill_context(64) + extend_context x8 (8 rows, all valid)  -> draft   (== B1)
  B3  fill(64) + extends incl. PARTIAL rows (3 valid, then 5)   -> draft   (== B1; scratch path)

Pass: B1 == B2 == B3 (token for token), B1 vs golden selector tokens >= 6/7 (A6's bar), and
PCC(draft_hidden B1, golden) >= 0.97.

Run:  MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/dflash2_kv_equiv.py -v -s
"""
import os

import numpy as np
import pytest
import torch
from safetensors import safe_open

import ttnn
from models.demos.blackhole.qwen36.tt.dflash2 import DEFAULT_WEIGHTS, DFlash2Draft, DFlash2Drafter, H, select_path

TGT = os.environ.get("HF_MODEL", "/home/ttuser/experiments/qwen36_27b/model_volume/weights/Qwen3.6-27B")
REAL = os.environ.get("DFLASH_REAL", "/home/ttuser/experiments/qwen36_27b/profiles/dflash2_real.npz")
GOLD = os.environ.get("DFLASH_GOLD", "/home/ttuser/experiments/qwen36_27b/profiles/dflash2_golden_real.npz")
NB, BS = 8, 64  # 512 slots


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return float(((a - a.mean()) * (b - b.mean())).sum() / (a.std() * b.std() * (a.numel() - 1)))


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 24576,
            "num_command_queues": 2,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 1024 * 1024 * 1024,
        }
    ],
    indirect=True,
)
def test_dflash2_kv_equiv(mesh_device):
    md = mesh_device
    md.enable_program_cache()
    real = np.load(REAL)
    taps = torch.from_numpy(real["target_hidden_cat"]).float()  # (1,128,25600)
    anchor = int(real["anchor"])
    C = taps.shape[1]
    assert C == 128
    gold = np.load(GOLD)
    gh = torch.from_numpy(gold["draft_hidden"]).float()  # (1,7,5120) golden (fp32 reference)

    with safe_open(f"{TGT}/model-00001-of-00015.safetensors", framework="pt") as f:
        embed = f.get_tensor("model.language_model.embed_tokens.weight").to(torch.bfloat16)
    with safe_open(f"{TGT}/model-00008-of-00015.safetensors", framework="pt") as f:
        lm = f.get_tensor("lm_head.weight")  # (V,H) bf16
    lm_dev = ttnn.from_torch(
        lm.T.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    lm_head_fn = lambda x: ttnn.linear(x, lm_dev)  # [1,1,8,H] -> [1,1,8,V]

    # ---- oracle: validated recompute-everything drafter ----
    A = DFlash2Draft(md, DEFAULT_WEIGHTS, embed, None)
    BLOCK = A.block
    dh_A = A.propose_hidden(taps, anchor, C)  # (1,7,5120)
    lmf = lm.float()
    tok_A = A._select(dh_A, dh_A @ lmf.T, anchor)
    tok_G = A._select(gh, gh @ lmf.T, anchor)  # golden selector tokens (A6)
    print(f"[equiv] golden tokens {tok_G}")
    print(
        f"[equiv] oracle tokens {tok_A}  match_golden={sum(a == g for a, g in zip(tok_A, tok_G))}/7  "
        f"pcc(dh_A,golden)={_pcc(dh_A, gh):.4f}"
    )

    # ---- production drafter ----
    B = DFlash2Drafter(md, DEFAULT_WEIGHTS, embed_host=embed, lm_head_fn=lm_head_fn)
    pt = torch.arange(NB, dtype=torch.int32).reshape(1, NB)

    def run(fill_rows, extends):
        B.alloc_kv(pt, BS)
        B.fill_context(taps[:, :fill_rows], 0)
        for slot0, rows, n in extends:
            t8 = torch.zeros(1, BLOCK, taps.shape[-1])
            t8[:, : rows.shape[1]] = rows
            B.extend_context(t8, slot0, n)
        tok = B.draft(anchor, C)
        dh = B.last_hidden.clone()
        B.free_kv()
        return tok, dh

    tok_B1, dh_B1 = run(128, [])
    print(
        f"[equiv] B1 fill(128)              tokens {tok_B1}  pcc(dh,golden)={_pcc(dh_B1, gh):.4f} "
        f"pcc(dh,oracle)={_pcc(dh_B1, dh_A):.4f}"
    )

    ext2 = [(r, taps[:, r : r + 8], 8) for r in range(64, 128, 8)]
    tok_B2, dh_B2 = run(64, ext2)
    print(f"[equiv] B2 fill(64)+8x extend(8)  tokens {tok_B2}  pcc(dh,B1)={_pcc(dh_B2, dh_B1):.5f}")

    ext3 = [(r, taps[:, r : r + 8], 8) for r in range(64, 120, 8)]
    ext3.append((120, taps[:, 120:123], 3))  # rows 3..7 -> scratch
    ext3.append((123, taps[:, 123:128], 5))  # rows 5..7 -> scratch
    tok_B3, dh_B3 = run(64, ext3)
    print(f"[equiv] B3 partial extends (3,5)   tokens {tok_B3}  pcc(dh,B1)={_pcc(dh_B3, dh_B1):.5f}")

    # Selector on the oracle hidden through the DEVICE lm_head/topk path: isolates lm_head/topk
    # precision from the attention-path difference.
    dh_dev = ttnn.from_torch(
        dh_A.reshape(1, 1, BLOCK - 1, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    dh_dev8 = ttnn.pad(dh_dev, [1, 1, BLOCK, H], [0, 0, 0, 0], 0.0)
    lg = lm_head_fn(dh_dev8)
    vals, idx = ttnn.topk(lg, 16, dim=-1, largest=True, sorted=True)
    un = ttnn.to_torch(ttnn.get_device_tensors(vals)[0]).float().reshape(BLOCK, 16)[: BLOCK - 1]
    cd = ttnn.to_torch(ttnn.get_device_tensors(idx)[0]).long().reshape(BLOCK, 16)[: BLOCK - 1]
    hp = dh_A[0] @ A.hproj.T
    tok_A_dev = select_path(un, cd, hp, anchor, A.pred_cb, A.succ_cb)
    print(
        f"[equiv] oracle hidden via device lm_head+topk: {tok_A_dev}  match_oracle_host={sum(a == b for a, b in zip(tok_A_dev, tok_A))}/7"
    )

    m_gold = sum(a == g for a, g in zip(tok_B1, tok_G))
    m_orc = sum(a == g for a, g in zip(tok_B1, tok_A))
    print(f"[equiv] B1 match golden={m_gold}/7 match oracle={m_orc}/7")
    assert tok_B1 == tok_B2, f"fill vs extend paths disagree: {tok_B1} vs {tok_B2}"
    assert tok_B1 == tok_B3, f"full vs partial-extend paths disagree: {tok_B1} vs {tok_B3}"
    assert _pcc(dh_B1, gh) >= 0.97, f"draft_hidden PCC vs golden too low: {_pcc(dh_B1, gh):.4f}"
    assert m_gold >= 6, f"B1 tokens match golden only {m_gold}/7"
