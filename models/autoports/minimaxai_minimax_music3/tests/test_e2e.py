"""Stage 06: end-to-end generation on the chip (LLM + depth + DiT on TT; cond encoder + vocoder on CPU)."""
import json
import os
import time

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.autoports.minimaxai_minimax_music3.tests.tt_common import Report  # noqa: E402
from models.autoports.minimaxai_minimax_music3.tt.device import open_mesh, parse_shape  # noqa: E402
from models.autoports.minimaxai_minimax_music3.tt.generator import Music3Generator  # noqa: E402

REPORT = Report()
CLIPS = {"readme": 8.0, "blues": 12.0}


def _clip(name):
    import importlib.util

    p = os.environ.get("MUSIC3_CLIPS_PY", os.path.expanduser("~/music3-bringup/scripts/py/clips.py"))
    spec = importlib.util.spec_from_file_location("clips", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.clip(name)


def audio_metrics(wav, sr):
    x = wav.float().squeeze(0)
    mono = x.mean(0)
    win = int(sr * 0.05)
    n = mono.numel() // win
    frms = mono[: n * win].reshape(n, win).pow(2).mean(1).sqrt()
    spec = torch.stft(mono, 2048, 512, window=torch.hann_window(2048), return_complex=True).abs().mean(1)
    freqs = torch.linspace(0, sr / 2, spec.numel())
    return {
        "rms": float(x.pow(2).mean().sqrt()),
        "peak": float(x.abs().max()),
        "clip_frac": float((x.abs() > 0.999).float().mean()),
        "silence_frac": float((frms < 1e-3).float().mean()),
        "spectral_centroid_hz": float((spec * freqs).sum() / spec.sum().clamp_min(1e-9)),
        "nan": bool(torch.isnan(x).any()),
        "seconds": x.shape[-1] / sr,
    }


@pytest.fixture(scope="module")
def handle():
    h = open_mesh(parse_shape(os.environ.get("MUSIC3_MESH_SHAPE", "1x1")))
    yield h
    h.close()


@pytest.fixture(scope="module")
def gen(handle, snapshot):
    dts = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
    t0 = time.time()
    g = Music3Generator(
        handle.mesh,
        snapshot,
        max_seq_len=int(os.environ.get("MUSIC3_MAX_SEQ_LEN", 16384)),
        llm_dtype=dts[os.environ.get("MUSIC3_LLM_DTYPE", "bfp8")],
        depth_dtype=dts[os.environ.get("MUSIC3_DEPTH_DTYPE", "bf16")],
        dit_dtype=dts[os.environ.get("MUSIC3_DIT_DTYPE", "bf16")],
    )
    REPORT.update(load_s=time.time() - t0, impl=g.impl)
    yield g
    g.release()


@pytest.fixture(scope="module", autouse=True)
def _write_report():
    yield
    REPORT.write()


@pytest.mark.parametrize("clip", ["readme", "blues"])
def test_generate(gen, golden_root, clip):
    import soundfile as sf

    c = _clip(clip)
    out = gen.generate(c["caption"], c["lyrics"], audio_duration=c["duration_s"], seed=c["seed"])
    st = out["stats"]
    m = audio_metrics(out["audio"], out["sample_rate"])
    gm = json.load(open(golden_root / clip / "metrics.json"))
    codes = out["codes_all"][1:, 0]
    runs, best, cur = 1, 1, 1
    for a, b in zip(codes[:-1].tolist(), codes[1:].tolist()):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    r = {
        **st.as_dict(),
        "audio": m,
        "golden_audio": {k: gm[k] for k in ("rms", "silence_frac", "spectral_centroid_hz")},
        "c0_unique_ratio": float(codes.unique().numel() / max(codes.numel(), 1)),
        "c0_max_run": best,
        "codes_rows": int(out["codes_all"].shape[0]),
    }
    REPORT[clip] = r
    print(json.dumps(r, indent=1, default=str))
    o = os.environ.get("MUSIC3_E2E_OUT")
    if o:
        os.makedirs(o, exist_ok=True)
        sf.write(
            os.path.join(o, f"{clip}_tt.wav"), out["audio"].squeeze(0).T.numpy(), out["sample_rate"], subtype="PCM_16"
        )
        torch.save({"codes_all": out["codes_all"], "stats": st.as_dict()}, os.path.join(o, f"{clip}_tt.pt"))
    assert st.frames >= 40 and not m["nan"]
    assert m["rms"] > 0.25 * gm["rms"] and m["silence_frac"] < 0.4 and m["clip_frac"] < 0.01, m
    assert 0.4 * gm["spectral_centroid_hz"] < m["spectral_centroid_hz"] < 2.5 * gm["spectral_centroid_hz"], m
    assert r["c0_unique_ratio"] > 0.3 and r["c0_max_run"] < 25, r


def test_seed_determinism(gen):
    c = _clip("smoke")
    a = gen.generate(c["caption"], c["lyrics"], audio_duration=c["duration_s"], seed=11, stages=("semantic",))
    b = gen.generate(c["caption"], c["lyrics"], audio_duration=c["duration_s"], seed=11, stages=("semantic",))
    same = torch.equal(a["codes_all"], b["codes_all"])
    REPORT["seed_determinism"] = {"identical_codes": same, "rows": int(a["codes_all"].shape[0])}
    print(REPORT["seed_determinism"])
    assert same
