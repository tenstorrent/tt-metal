# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for the LTX eval harness (tests/models/ltx/tools/ltx_eval.py), VBench excluded."""

import argparse
import json

import numpy as np
import pytest

from models.tt_dit.tests.models.ltx.tools import ltx_eval

FRAMES, H, W = 8, 32, 48


def clip(seed):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    return np.stack([np.roll(base, shift, axis=1) for shift in range(FRAMES)])


def write_video(path, frames):
    import av

    with av.open(str(path), mode="w") as out:
        stream = out.add_stream("libx264rgb", rate=24)
        stream.width, stream.height, stream.pix_fmt = W, H, "rgb24"
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def noisy(frames, sigma, seed=1):
    noise = np.random.default_rng(seed).normal(0, sigma, frames.shape)
    return np.clip(frames + noise, 0, 255).astype(np.uint8)


def test_pcc_matches_numpy_and_accumulates_across_chunks():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=1000), rng.normal(size=1000)
    y += x
    expected = np.corrcoef(x, y)[0, 1]
    moments = ltx_eval.Moments()
    for part in range(4):
        moments.add(x[part * 250 : (part + 1) * 250], y[part * 250 : (part + 1) * 250])
    assert moments.pcc() == pytest.approx(expected, abs=1e-12)
    assert ltx_eval.pcc(x, x) == pytest.approx(1.0)


def test_psnr_known_value():
    ref = np.zeros((4, 4))
    assert ltx_eval.psnr(ref, ref, 255.0) == ltx_eval.PSNR_IDENTICAL
    # mse = 1 -> 20 log10(255)
    assert ltx_eval.psnr(ref, ref + 1, 255.0) == pytest.approx(48.1308, abs=1e-4)


def test_identical_videos_are_lossless_parity(tmp_path):
    ref = write_video(tmp_path / "ref.mp4", clip(0))
    result = ltx_eval.compare_videos(ref, ref, tmp_path, name="same", stills=None)
    assert result["frames"] == FRAMES
    assert result["psnr_min"] == ltx_eval.PSNR_IDENTICAL
    assert result["pcc"] == pytest.approx(1.0)
    assert sorted(p.name for p in tmp_path.glob("same_cmp_*.png")) == [
        "same_cmp_f000.png",
        "same_cmp_f004.png",
        "same_cmp_f007.png",
    ]


def test_noise_lowers_psnr_and_fails_verdict(tmp_path):
    frames = clip(0)
    ref = write_video(tmp_path / "ref.mp4", frames)
    cand = write_video(tmp_path / "cand.mp4", noisy(frames, 20))
    opts = ltx_eval.parse_opts(
        argparse.Namespace(
            vbench="none",
            vbench_ref=False,
            vbench_tol=0.01,
            prompt=None,
            temporal_width=960,
            stills="0,-1",
            pcc_min=0.99,
            psnr_min=30.0,
            seeds=1,
        )
    )
    report = ltx_eval.evaluate_clip(cand, ref, tmp_path / "q", name="cand", opts=opts)
    assert 20 < report["parity"]["psnr_min"] < 30
    assert report["verdict"]["ok"] is False
    assert (tmp_path / "q" / "cand_f000.png").exists() and (tmp_path / "q" / "cand_f007.png").exists()
    assert "QUALITY FAIL" in ltx_eval.verdict_line(report)


def test_frame_count_mismatch_is_an_error(tmp_path, expect_error):
    ref = write_video(tmp_path / "ref.mp4", clip(0))
    short = write_video(tmp_path / "short.mp4", clip(0)[:-2])
    with expect_error(ValueError, "Frame count differs"):
        ltx_eval.compare_videos(ref, short, tmp_path, name="short", stills=None)


def test_batch_pairs_by_name_and_aggregates(tmp_path):
    for side in ("ref", "cand"):
        (tmp_path / side).mkdir()
        for seed in range(5):
            write_video(tmp_path / side / f"seed{seed}.mp4", clip(seed))
    code = ltx_eval.main(
        [
            "batch",
            "--cand-dir",
            str(tmp_path / "cand"),
            "--ref-dir",
            str(tmp_path / "ref"),
            "--out",
            str(tmp_path / "q"),
            "--vbench",
            "none",
            "--jobs",
            "2",
        ]
    )
    summary = json.loads((tmp_path / "q" / "summary.json").read_text())
    assert code == 0 and summary["ok"]
    assert summary["clips"] == [f"seed{seed}" for seed in range(5)]
    assert summary["psnr_worst"] == ltx_eval.PSNR_IDENTICAL
    assert "warning" not in summary


def test_vbench_regression_against_reference_flags(tmp_path):
    report = {"vbench": {"imaging_quality": 0.60}, "vbench_ref": {"imaging_quality": 0.65}}
    opts = {"pcc_min": 0.99, "psnr_min": 30.0, "vbench_tol": 0.01}
    assert ltx_eval.verdict(report, opts)["ok"] is False
    report["vbench"]["imaging_quality"] = 0.645
    assert ltx_eval.verdict(report, opts)["ok"] is True


def test_tensor_mode(tmp_path):
    import torch

    ref = torch.randn(1, 3, 4, 16, 16)
    torch.save(ref, tmp_path / "ref.pt")
    np.save(tmp_path / "cand.npy", (ref + 0.01 * torch.randn_like(ref)).numpy())
    assert ltx_eval.main(["tensor", "--ref", str(tmp_path / "ref.pt"), "--cand", str(tmp_path / "cand.npy")]) == 0
    np.save(tmp_path / "bad.npy", torch.randn_like(ref).numpy())
    assert ltx_eval.main(["tensor", "--ref", str(tmp_path / "ref.pt"), "--cand", str(tmp_path / "bad.npy")]) == 1


RAPPER, BOAT = "A confident rapper", "A red paper boat"


def score_video(tmp_path, *extra):
    argv = ["video", "--cand", str(tmp_path / "cand.mp4"), "--ref", str(tmp_path / "ref.mp4")]
    return ltx_eval.main(argv + ["--out", str(tmp_path / "q"), "--vbench", "none", *extra])


@pytest.fixture
def clip_pair(tmp_path):
    write_video(tmp_path / "ref.mp4", clip(0))
    write_video(tmp_path / "cand.mp4", clip(0))
    return tmp_path


@pytest.mark.parametrize(
    "cand_meta, field",
    [({"prompt": BOAT, "seed": 0}, "prompt"), ({"prompt": RAPPER, "seed": 1}, "seed")],
)
def test_sidecar_mismatch_refuses_to_score(clip_pair, capsys, cand_meta, field):
    ltx_eval.write_sidecar(clip_pair / "ref.mp4", prompt=RAPPER, seed=0)
    ltx_eval.write_sidecar(clip_pair / "cand.mp4", **cand_meta)
    assert score_video(clip_pair) == ltx_eval.EXIT_MISMATCH
    out = capsys.readouterr().out
    assert f"QUALITY MISMATCH {field} clip=cand" in out
    assert "QUALITY OK" not in out and "QUALITY FAIL" not in out
    assert not (clip_pair / "q" / "cand_report.json").exists()


def test_allow_mismatch_scores_anyway(clip_pair, capsys):
    ltx_eval.write_sidecar(clip_pair / "ref.mp4", prompt=RAPPER, seed=0)
    ltx_eval.write_sidecar(clip_pair / "cand.mp4", prompt=BOAT, seed=0)
    assert score_video(clip_pair, "--allow-mismatch") == 0
    out = capsys.readouterr().out
    assert "QUALITY MISMATCH prompt clip=cand" in out and "QUALITY OK clip=cand" in out


@pytest.mark.parametrize("sidecars", [("ref", "cand"), ("ref",), ()])
def test_matching_or_missing_sidecars_keep_output(clip_pair, capsys, sidecars):
    assert score_video(clip_pair) == 0
    baseline = capsys.readouterr().out
    for side in sidecars:
        ltx_eval.write_sidecar(clip_pair / f"{side}.mp4", prompt=RAPPER, seed=0, gen=int(side == "cand"))
    assert score_video(clip_pair) == 0
    captured = capsys.readouterr()
    assert captured.out == baseline
    assert ("WARNING: no sidecar" in captured.err) == (len(sidecars) < 2)


def test_batch_seed_mismatch_refuses_and_allow_scores(tmp_path, capsys):
    for side in ("ref", "cand"):
        (tmp_path / side).mkdir()
        for seed in range(2):
            write_video(tmp_path / side / f"seed{seed}.mp4", clip(seed))
            ltx_eval.write_sidecar(tmp_path / side / f"seed{seed}.mp4", prompt=RAPPER, seed=seed)
    ltx_eval.write_sidecar(tmp_path / "cand" / "seed1.mp4", prompt=RAPPER, seed=4)
    argv = ["batch", "--cand-dir", str(tmp_path / "cand"), "--ref-dir", str(tmp_path / "ref")]
    argv += ["--out", str(tmp_path / "q"), "--vbench", "none", "--jobs", "1", "--seeds", "2"]
    assert ltx_eval.main(argv) == ltx_eval.EXIT_MISMATCH
    out = capsys.readouterr().out
    assert out.splitlines() == ["QUALITY MISMATCH seed clip=seed1 cand=4 ref=1"]
    assert not (tmp_path / "q" / "summary.json").exists()
    assert ltx_eval.main(argv + ["--allow-mismatch"]) == 0
    assert "BATCH OK" in capsys.readouterr().out
