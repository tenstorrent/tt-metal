"""Stage 04: the TTNN depth decoder vs the torch reference (unit goldens + teacher forcing over the readme golden)."""
import json
import os
import time

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.autoports.minimaxai_minimax_music3.config import Music3Config  # noqa: E402
from models.autoports.minimaxai_minimax_music3.reference.sampling import guided_depth_logits  # noqa: E402
from models.autoports.minimaxai_minimax_music3.tests.tt_common import Report, golden, pcc, thresholds  # noqa: E402
from models.autoports.minimaxai_minimax_music3.tt import weights as W  # noqa: E402
from models.autoports.minimaxai_minimax_music3.tt.depth_decoder import TTDepthDecoder  # noqa: E402
from models.autoports.minimaxai_minimax_music3.tt.device import open_mesh, parse_shape  # noqa: E402

REPORT = Report()


@pytest.fixture(scope="module")
def handle():
    h = open_mesh(parse_shape(os.environ.get("MUSIC3_MESH_SHAPE", "1x1")))
    yield h
    h.close()


@pytest.fixture(scope="module")
def depth(handle, snapshot):
    dt = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[os.environ.get("MUSIC3_DEPTH_DTYPE", "bf16")]
    cfg = Music3Config.from_snapshot(snapshot)
    full = W.load_lm_tables(snapshot)["model.embed_tokens.weight"]
    t0 = time.time()
    d = TTDepthDecoder(
        handle.mesh,
        W.load_depth_state(snapshot),
        cfg.depth,
        weights_dtype=dt,
        full_embed=full,
        use_trace=os.environ.get("MUSIC3_TRACE", "1") == "1",
    )
    d.warmup()
    REPORT.update(load_s=time.time() - t0, depth_dtype=os.environ.get("MUSIC3_DEPTH_DTYPE", "bf16"))
    yield d
    d.release()


@pytest.fixture(scope="module", autouse=True)
def _write_report():
    yield
    REPORT.write()


def test_unit_goldens(depth, golden_root):
    u = torch.load(golden_root / "unit" / "unit_goldens.pt")
    rows = {}
    for steps in range(2, 9):
        g = u[f"depth_steps{steps}"]
        assert "raw" in g, "unit goldens predate the raw-row format; re-run stage 01 unit-goldens"
        h, lg = depth.forward_rows(g["raw"])
        rows[steps] = {"hidden_pcc": pcc(h, g["hidden"]), "logits_pcc": pcc(lg, g["logits"])}
    REPORT["unit"] = rows
    print(json.dumps(rows, indent=1))
    assert min(r["hidden_pcc"] for r in rows.values()) >= 0.99 and min(r["logits_pcc"] for r in rows.values()) >= 0.99


def test_teacher_forced_vs_golden(depth, golden_root):
    g, tf = golden(golden_root, "readme")
    codes = g["codes_all"]
    N = min(codes.shape[0], int(os.environ.get("MUSIC3_DEPTH_TF_ROWS", 200)))
    lp, hp, t1, t5, nd = [], [], 0, 0, 0
    t0 = time.time()
    for i in range(N):
        rec = {}
        out_codes, dh = depth.frame(
            tf["hidden_all"][i], int(codes[i, 0]), None, forced=codes[i, 1:].tolist(), record=rec
        )
        assert torch.equal(out_codes, codes[i])
        L = torch.stack(rec["depth_logits"])  # [7, 2, 1024]
        lp.append(pcc(L, tf["depth_logits"][i]))
        hp.append(pcc(dh, tf["depth_hidden"][i]))
        for j in range(7):
            gd = guided_depth_logits(L[j]).reshape(-1)
            top = torch.topk(gd, 5).indices.tolist()
            cpu_t = int(guided_depth_logits(tf["depth_logits"][i, j]).reshape(-1).argmax())
            t1 += int(top[0] == cpu_t)
            t5 += int(cpu_t in top)
            nd += 1
    dt = time.time() - t0
    r = {
        "frames": N,
        "logits_pcc_min": min(lp),
        "logits_pcc_mean": sum(lp) / N,
        "hidden_pcc_min": min(hp),
        "hidden_pcc_mean": sum(hp) / N,
        "depth_top1": t1 / nd,
        "depth_top5": t5 / nd,
        "ms_per_frame": 1000 * dt / N,
    }
    REPORT["teacher_forced"] = r
    print(json.dumps(r, indent=1))
    th = thresholds()
    assert r["logits_pcc_min"] >= 0.99 and r["hidden_pcc_min"] >= 0.99
    assert r["depth_top5"] >= th["depth_top5"], f"depth top-5 {r['depth_top5']} < {th['depth_top5']}"
    print(f"depth top-1 {r['depth_top1']:.4f} (bar {th['depth_top1']}) - advisory until the dtype sweep")
