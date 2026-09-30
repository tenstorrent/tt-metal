"""CPU self-test for the eval scripts: run with the eval venv's pytest from this directory."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare  # noqa: E402


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("fx")
    subprocess.run(
        [
            sys.executable,
            str(HERE / "make_fixture.py"),
            str(out),
            "--seeds",
            "0,1",
            "--width",
            "256",
            "--height",
            "144",
            "--frames",
            "49",
            "--noise",
            "4",
        ],
        check=True,
        capture_output=True,
    )
    return out / "ref", out / "cand"


def test_stats_match_numpy():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=10_000), rng.normal(size=10_000)
    y = 0.7 * x + 0.3 * y
    stats = compare.Stats()
    for a, b in zip(np.split(x, 10), np.split(y, 10)):
        stats.add(a, b)
    assert stats.pcc() == pytest.approx(np.corrcoef(x, y)[0, 1], abs=1e-9)


def test_identical_is_perfect(runs):
    ref, _ = runs
    result = compare.compare_seed(ref, ref, "seed0", force_mp4=True)
    assert math.isinf(result["pixels_npy"]["psnr"]) and math.isinf(result["pixels_mp4"]["psnr"])
    assert result["latents"]["video_rows"]["pcc"] == pytest.approx(1.0)
    assert result["audio"]["pcc"] == pytest.approx(1.0)


def test_known_noise_psnr(runs):
    ref, cand = runs
    result = compare.compare_seed(ref, cand, "seed0", force_mp4=False)
    # Sigma 4 Gaussian noise, slightly reduced by clipping at 0/255.
    assert 35.5 < result["pixels_npy"]["psnr"] < 37.0
    assert result["pixels_npy"]["frames"] == 4
    assert result["timing"]["speedup"] == pytest.approx(1.5)
    assert 0.99 < result["latents"]["video_rows"]["pcc"] < 0.9995


def test_cli_thresholds_and_json(runs, tmp_path):
    ref, cand = runs
    out = tmp_path / "cmp.json"
    ok = subprocess.run([sys.executable, str(HERE / "compare.py"), str(ref), str(cand), "--json", str(out), "--strict"])
    assert ok.returncode == 0
    assert set(json.loads(out.read_text())["results"]) == {"seed0", "seed1"}
    bad = subprocess.run(
        [sys.executable, str(HERE / "compare.py"), str(ref), str(cand), "--min-psnr", "40", "--strict"],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1 and "FAIL pixels_npy.psnr" in bad.stdout


def test_latent_shape_mismatch(tmp_path):
    torch.save({"video_rows": torch.zeros(4, 4)}, tmp_path / "a.pt")
    torch.save({"video_rows": torch.zeros(4, 5)}, tmp_path / "b.pt")
    assert "error" in compare.compare_latents(tmp_path / "a.pt", tmp_path / "b.pt")["video_rows"]


def test_review_outputs(runs, tmp_path):
    ref, cand = runs
    subprocess.run(
        [sys.executable, str(HERE / "review.py"), str(ref), str(cand), "--out", str(tmp_path / "r"), "--cols", "3"],
        check=True,
        capture_output=True,
    )
    import cv2

    sheet = cv2.imread(str(tmp_path / "r_sheet.png"))
    assert sheet is not None and sheet.shape[1] == 3 * 320
    cap = cv2.VideoCapture(str(tmp_path / "r_grid.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 49
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 2 * 640
