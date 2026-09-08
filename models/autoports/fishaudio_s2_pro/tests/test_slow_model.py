"""Slow tower on TTNN vs the CPU fp32 goldens (stage 03/04). Needs hardware + stage-01 goldens.

Env: FISH_S2_MESH_SHAPE (default 1x1), FISH_S2_MAX_SEQ_LEN (default 4096), FISH_S2_REPORT (json out),
FISH_S2_FLOORS (tf_floors.json from stage 01; thresholds default to the framework bars otherwise).
"""
import json
import os
import time
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.autoports.fishaudio_s2_pro.tt.device import open_mesh, parse_shape  # noqa: E402
from models.autoports.fishaudio_s2_pro.tt.generator import S2Generator  # noqa: E402

REPORT = {}


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def thresholds():
    p = os.environ.get("FISH_S2_FLOORS")
    t = {"cb0_top1": 0.90, "cb0_top5": 0.98, "cbn_top1": 0.85, "cbn_top5": 0.97}
    if p and Path(p).exists():
        t.update(json.load(open(p)).get("thresholds", {}))
    return t


@pytest.fixture(scope="module")
def handle():
    h = open_mesh(parse_shape(os.environ.get("FISH_S2_MESH_SHAPE", "1x1")))
    yield h
    h.close()


@pytest.fixture(scope="module")
def gen(handle, snapshot):
    t0 = time.time()
    wdt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("FISH_S2_WEIGHTS_DTYPE", "bfp8")]
    g = S2Generator(handle.mesh, snapshot, max_seq_len=int(os.environ.get("FISH_S2_MAX_SEQ_LEN", 4096)), dtype=wdt)
    REPORT["weights_dtype"] = os.environ.get("FISH_S2_WEIGHTS_DTYPE", "bfp8")
    REPORT["load_s"] = time.time() - t0
    REPORT["mesh"] = "x".join(map(str, handle.shape))
    REPORT["device_name"] = g.args.device_name
    return g


@pytest.fixture(scope="module", autouse=True)
def _write_report():
    yield
    out = os.environ.get("FISH_S2_REPORT")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(REPORT, open(out, "w"), indent=2, default=str)


@pytest.mark.parametrize("prompt_id,variant", [("short", "no_ref"), ("short", "ref"), ("medium", "no_ref")])
def test_teacher_forced_vs_golden(gen, golden_root, prompt_id, variant):
    d = golden_root / prompt_id / variant / "greedy"
    if not (d / "frames.pt").exists():
        pytest.skip(f"golden {d} missing")
    meta = json.load(open(d / "meta.json"))
    ref = torch.load(d / "teacher_forced.pt")
    T = meta["prompt_len"]
    # golden frames.pt holds the GENERATED frames only (incl. the <|im_end|> frame); prompt.pt the prompt
    prompt = torch.load(d / "prompt.pt")
    assert prompt.shape[1] == T
    frames = torch.cat([prompt, torch.load(d / "frames.pt")], dim=1)
    G = frames.shape[1] - T
    t0 = time.time()
    res = gen.teacher_forced(frames, T)
    dt = time.time() - t0
    target = frames[0, T:]
    top1 = float((res["slow_argmax"] == target).float().mean())
    top5 = float(torch.tensor([int(target[g]) in res["slow_top5"][g].tolist() for g in range(G)]).float().mean())
    keep = target != gen.cfg.im_end_id
    fc = frames[2:, T:][:, keep].T
    fast_top1 = float((res["fast_argmax"][keep] == fc).float().mean())
    n = min(8, res["slow_logits_first"].shape[0], ref["slow_full_logits_first"].shape[0])
    # diagnostics: which tap row matched the reference hidden at the prefill step, and per-step PCCs
    tile = res.get("hidden_tile_first")
    if tile is not None:
        rows = [round(pcc(tile[r], ref["fast_hidden_first"][0]), 3) for r in range(tile.shape[0])]
        print(
            f"[diag] prompt_len {T}: prefill hidden PCC per tap row: best row {int(torch.tensor(rows).argmax())} = {max(rows)} (used row {(T - 1) % 32})"
        )
    print(
        "[diag] per-step logits PCC:",
        [round(pcc(res["slow_logits_first"][i], ref["slow_full_logits_first"][i]), 4) for i in range(n)],
    )
    print(
        "[diag] per-step hidden PCC:",
        [round(pcc(res["hidden_first"][i], ref["fast_hidden_first"][i]), 4) for i in range(n)],
    )
    logits_pcc = min(pcc(res["slow_logits_first"][i], ref["slow_full_logits_first"][i]) for i in range(n))
    hidden_pcc = min(pcc(res["hidden_first"][i], ref["fast_hidden_first"][i]) for i in range(n))
    # agreement with the CPU model's own argmax (ref["slow_argmax"] == target for greedy goldens, but keep both)
    r = {
        "prompt_len": T,
        "gen_len": G,
        "slow_top1": top1,
        "slow_top5": top5,
        "fast_top1": fast_top1,
        "logits_pcc_min_first8": logits_pcc,
        "hidden_pcc_min_first8": hidden_pcc,
        "seconds": dt,
        "frames_per_s": G / dt,
    }
    # tensors for cross-mesh comparison (stage 04 compares TP2/TP4 against these 1x1 outputs)
    tdir = os.environ.get("FISH_S2_TENSOR_OUT")
    if tdir:
        Path(tdir).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "slow_logits_first": res["slow_logits_first"],
                "hidden_first": res["hidden_first"],
                "slow_argmax": res["slow_argmax"],
            },
            Path(tdir) / f"{prompt_id}_{variant}.pt",
        )
    base = os.environ.get("FISH_S2_BASELINE_TENSORS")
    if base and (Path(base) / f"{prompt_id}_{variant}.pt").exists():
        b = torch.load(Path(base) / f"{prompt_id}_{variant}.pt")
        r["vs_baseline_logits_pcc_min"] = min(
            pcc(res["slow_logits_first"][i], b["slow_logits_first"][i]) for i in range(n)
        )
        r["vs_baseline_hidden_pcc_min"] = min(pcc(res["hidden_first"][i], b["hidden_first"][i]) for i in range(n))
        r["vs_baseline_argmax_agreement"] = float((res["slow_argmax"] == b["slow_argmax"]).float().mean())
    REPORT[f"{prompt_id}/{variant}"] = r
    print(json.dumps(r, indent=2))
    th = thresholds()
    calibrated = bool(os.environ.get("FISH_S2_FLOORS")) and Path(os.environ["FISH_S2_FLOORS"]).exists()
    assert hidden_pcc >= 0.99, f"hidden PCC {hidden_pcc}"
    assert logits_pcc >= 0.99, f"logits PCC {logits_pcc}"
    assert top5 >= th["cb0_top5"], f"cb0 top-5 {top5} < {th['cb0_top5']}"
    if calibrated:  # top-1 bars come from the bf16-vs-fp32 CPU floor (stage 01); uncalibrated defaults only report
        assert top1 >= th["cb0_top1"], f"cb0 top-1 {top1} < {th['cb0_top1']}"
        assert fast_top1 >= th["cbn_top1"], f"fast top-1 {fast_top1}"
    else:
        print(
            f"[uncalibrated] cb0 top-1 {top1:.4f} (framework default bar 0.90), fast top-1 {fast_top1:.4f} — reported only"
        )
    if "vs_baseline_logits_pcc_min" in r:
        assert r["vs_baseline_hidden_pcc_min"] >= 0.995, f"hidden vs 1-chip PCC {r['vs_baseline_hidden_pcc_min']}"
        assert r["vs_baseline_logits_pcc_min"] >= 0.995, f"logits vs 1-chip PCC {r['vs_baseline_logits_pcc_min']}"


def test_free_running_smoke(gen):
    codes, st = gen.generate("The quick brown fox jumps over the lazy dog.", greedy=True, max_new_tokens=40)
    REPORT["free_running_smoke"] = {
        "frames": st.frames,
        "stopped": st.stopped_on_im_end,
        "frames_per_s": st.frames_per_s,
        "prefill_s": st.prefill_s,
        "slow_s": st.slow_s,
        "fast_s": st.fast_s,
    }
    print(REPORT["free_running_smoke"])
    assert codes.shape[0] == 10 and st.frames > 0
    assert (codes[0] >= 0).all() and (codes[0] < 4096).all()
