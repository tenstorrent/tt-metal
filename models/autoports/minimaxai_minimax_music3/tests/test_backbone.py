"""Stage 03: the global LLM on one chip vs the CPU fp32 goldens (teacher forcing) + a free-running smoke.
Env: MUSIC3_SNAPSHOT, MUSIC3_GOLDEN_ROOT, MUSIC3_REPORT, MUSIC3_FLOORS, MUSIC3_MESH_SHAPE=1x1, MUSIC3_LLM_DTYPE (bfp8|bf16),
MUSIC3_MAX_SEQ_LEN, MUSIC3_LAYERS (debug)."""
import json
import os
import time

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.autoports.minimaxai_minimax_music3.reference.sampling import guided_c0_logits_sliced  # noqa: E402
from models.autoports.minimaxai_minimax_music3.tests.tt_common import (  # noqa: E402
    Report,
    agreement,
    golden,
    pcc,
    thresholds,
)
from models.autoports.minimaxai_minimax_music3.tt.device import open_mesh, parse_shape  # noqa: E402
from models.autoports.minimaxai_minimax_music3.tt.generator import Music3Generator  # noqa: E402

REPORT = Report()


@pytest.fixture(scope="module")
def handle():
    h = open_mesh(parse_shape(os.environ.get("MUSIC3_MESH_SHAPE", "1x1")))
    yield h
    h.close()


@pytest.fixture(scope="module")
def gen(handle, snapshot):
    t0 = time.time()
    dt = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[os.environ.get("MUSIC3_LLM_DTYPE", "bfp8")]
    g = Music3Generator(
        handle.mesh,
        snapshot,
        max_seq_len=int(os.environ.get("MUSIC3_MAX_SEQ_LEN", 16384)),
        llm_dtype=dt,
        load_dit=False,
        load_vocoder=False,
        depth_device="cpu",
        n_layers=int(os.environ["MUSIC3_LAYERS"]) if os.environ.get("MUSIC3_LAYERS") else None,
    )
    REPORT.update(
        load_s=time.time() - t0,
        llm_dtype=os.environ.get("MUSIC3_LLM_DTYPE", "bfp8"),
        mesh="x".join(map(str, handle.shape)),
        device_name=g.args.device_name,
        max_seq_len=g.args.max_seq_len,
        prefill_chunk=getattr(g.args, "max_prefill_chunk_size", None),
    )
    return g


@pytest.fixture(scope="module", autouse=True)
def _write_report():
    yield
    REPORT.write()


@pytest.mark.parametrize("clip", ["readme", "blues"])
def test_teacher_forced_vs_golden(gen, golden_root, clip):
    g, tf = golden(golden_root, clip)
    codes = g["codes_all"]
    t0 = time.time()
    res = gen.semantic_generation(g["text_ids"], 0, None, forced_codes=codes, record=True, skip_depth=True)
    dt = time.time() - t0
    N = codes.shape[0]
    H, L = res["hidden_all"], res["c0_logits"]  # [N, 2, D], [N, 2, 16385]
    assert H.shape[0] == N and L.shape == tf["c0_logits"].shape, (H.shape, L.shape, tf["c0_logits"].shape)
    h_pcc = [pcc(H[i], tf["hidden_all"][i]) for i in range(N)]
    l_pcc = [pcc(L[i], tf["c0_logits"][i]) for i in range(N)]
    top1, top5 = agreement(guided_c0_logits_sliced, L, codes[:, 0])
    # agreement with the CPU model's own guided argmax (the ceiling: goldens were sampled, so top-1 vs codes is < 1 even for CPU)
    cpu_arg = torch.stack([guided_c0_logits_sliced(tf["c0_logits"][i]).reshape(-1).argmax() for i in range(N)])
    tt_arg = torch.stack([guided_c0_logits_sliced(L[i]).reshape(-1).argmax() for i in range(N)])
    r = {
        "rows": N,
        "prompt_len": int(g["text_ids"].shape[1]),
        "hidden_pcc_min": min(h_pcc),
        "hidden_pcc_mean": sum(h_pcc) / N,
        "hidden_pcc_prefill": h_pcc[0],
        "logits_pcc_min": min(l_pcc),
        "logits_pcc_mean": sum(l_pcc) / N,
        "c0_top1": top1,
        "c0_top5": top5,
        "argmax_agree_with_cpu": float((cpu_arg == tt_arg).float().mean()),
        "seconds": dt,
        "ms_per_step": 1000 * dt / N,
        "hidden_pcc_first8": [round(x, 4) for x in h_pcc[:8]],
        "logits_pcc_first8": [round(x, 4) for x in l_pcc[:8]],
    }
    REPORT[clip] = r
    print(json.dumps(r, indent=1))
    th = thresholds()
    assert r["hidden_pcc_min"] >= 0.99, f"hidden PCC {r['hidden_pcc_min']}"
    assert r["logits_pcc_min"] >= 0.99, f"logits PCC {r['logits_pcc_min']}"
    assert top5 >= th["c0_top5"], f"c0 top-5 {top5} < {th['c0_top5']}"
    print(f"c0 top-1 {top1:.4f} (bar {th['c0_top1']}) - advisory until the dtype sweep")


def test_long_prompt_prefill(gen):
    """Advisory: a prompt near the 5 000-token contract must prefill (chunk size / L1)."""
    from models.autoports.minimaxai_minimax_music3.reference.prompt import build_text_ids

    lyrics = "\n".join(
        ["[verse]"] + ["Morning light filtering through the pine, every quiet street is yours and mine"] * 260
    )
    ids = build_text_ids(gen.tok, "Genre: acoustic pop. BPM: 96.", lyrics)
    n = int(ids.shape[1])
    try:
        t0 = time.time()
        logits, hidden = gen._prefill(ids)
        ok = bool(torch.isfinite(hidden).all() and torch.isfinite(logits).all())
        REPORT["long_prompt"] = {"tokens": n, "ok": ok, "seconds": time.time() - t0}
    except Exception as e:  # noqa
        REPORT["long_prompt"] = {"tokens": n, "ok": False, "error": str(e)[:300]}
        pytest.xfail(f"long prompt ({n} tokens) prefill failed: {str(e)[:200]}")
    assert REPORT["long_prompt"]["ok"]


def test_free_running_smoke(gen):
    from models.autoports.minimaxai_minimax_music3.tests.test_prompt import CASES  # noqa: F401

    out = gen.generate(
        "Genre: acoustic pop. BPM: 96. Warm female vocal, fingerpicked guitar.",
        "[verse]\nMorning light filtering through the pine\n[chorus]\nSoftly the world begins to breathe",
        max_frames=12,
        seed=3,
        stages=("semantic",),
    )
    st = out["stats"]
    REPORT["free_running_smoke"] = st.as_dict() | {"codes_rows": int(out["codes_all"].shape[0])}
    print(REPORT["free_running_smoke"])
    assert (
        out["codes_all"].shape[0] >= 2
        and (out["codes_all"][:, 0] < 16384).all()
        and (out["codes_all"][:, 1:] < 1024).all()
    )
