#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 07 before/after harness: one pipeline configuration -> AR frames/s, DiT chunk time, vocoder time, accuracy.

The same code path measures the functional baseline and every optimized candidate so the numbers in
``doc/optimize/perf.json`` are comparable: the golden 10 s replay (teacher-forced codes and noises, 30 steps: the
accuracy scores - frame-hidden PCC, latent PCC vs the golden windows, log-mel distance of the stitched wav) and a
free-running 10 s song (seed 7, 30 steps: the throughput numbers - warmed AR frames/s, per-window DiT and vocoder
seconds). Results land in ``doc/optimize/perf_runs/<label>.json``; ``--write-perf-json before|after`` also copies
the headline numbers into ``doc/optimize/perf.json`` under that key (the stage gate reads ``before`` / ``after``).

    source ~/mm3-bringup/common.sh && cd $MM3_WT
    with_hw_lock timeout 3000 $MM3_PY $MM3_MODEL_DIR/scripts/measure_perf.py --label before --policy functional --write-perf-json before
    with_hw_lock timeout 3000 $MM3_PY $MM3_MODEL_DIR/scripts/measure_perf.py --label after --policy optimized --write-perf-json after
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import soundfile as sf
import torch
from loguru import logger

import ttnn
from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.tt.audio_metrics import audio_stats, code_stats, log_mel_distance
from models.autoports.minimaxai_minimax_music3.tt.constants import AUDIO_CODE_OFFSET, NUM_CODEBOOKS
from models.common.utility_functions import comp_pcc

MODEL_DIR = Path(__file__).resolve().parents[1]
GENERATED = MODEL_DIR / "generated"
DOC = MODEL_DIR / "doc" / "optimize"
os.environ.setdefault("TT_DIT_CACHE_DIR", str(GENERATED / "tt_dit_cache"))


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(comp_pcc(a.float(), b.float(), 0.0)[1])


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=MODEL_DIR, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def load_golden(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    codes = torch.load(root / "sampled_codes.pt")
    raw = torch.load(root / "sampled_raw.pt")
    groups = raw[: (raw.numel() // NUM_CODEBOOKS) * NUM_CODEBOOKS].reshape(-1, NUM_CODEBOOKS).clone()
    groups[:, 0] -= AUDIO_CODE_OFFSET
    chunks = torch.load(root / "chunks.pt")
    wav, sr = sf.read(root / "audio.wav", dtype="float32")
    return {
        "manifest": manifest,
        "text_ids": torch.load(root / "text_ids.pt"),
        "codes": codes,
        "frame0_codes": groups[0],
        "frame_hiddens": torch.load(root / "frame_hiddens.pt"),
        "chunk_starts": list(chunks["chunk_starts"]),
        "noises": [n.float() for n in chunks["noises"]],
        "latents": [t.float() for t in chunks["latents"]],
        "wav": torch.from_numpy(wav.T.copy()),
        "sr": sr,
    }


def golden_replay(pipe, g: dict, steps: int) -> dict:
    m = g["manifest"]
    t0 = time.perf_counter()
    out = pipe.generate(
        m["prompt"],
        m["lyrics"],
        seed=m["seed"],
        num_inference_steps=steps,
        max_frames=g["codes"].shape[0],
        teacher_codes=g["codes"],
        teacher_frame0_codes=g["frame0_codes"],
        noises=g["noises"],
        text_ids=g["text_ids"],
    )
    wall = time.perf_counter() - t0
    assert torch.equal(out["codes"], g["codes"]) and out["chunk_starts"] == g["chunk_starts"], "replay diverged"
    fh = out["frame_hiddens"]
    gh = g["frame_hiddens"]
    per_frame = [_pcc(gh[0, f], fh[0, f]) for f in range(gh.shape[1])]
    latent_pcc = [_pcc(g["latents"][k], out["latents"][k]) for k in range(len(g["latents"]))]
    audio = torch.from_numpy(out["audio"])
    n = min(audio.shape[-1], g["wav"].shape[-1])
    lm = log_mel_distance(audio[..., :n], g["wav"][..., :n], g["sr"])
    wav_pcc = _pcc(g["wav"][..., :n], audio[..., :n])
    res = {
        "frames": out["frames"],
        "wall_s": wall,
        "frame_hiddens_pcc": _pcc(gh, fh),
        "frame_hiddens_pcc_min": min(per_frame),
        "frame_hiddens_backbone_pcc_min": min(_pcc(gh[0, f, :4096], fh[0, f, :4096]) for f in range(gh.shape[1])),
        "frame_hiddens_depth_pcc_min": min(_pcc(gh[0, f, 4096:], fh[0, f, 4096:]) for f in range(gh.shape[1])),
        "latent_pcc": latent_pcc,
        "log_mel_rms_db": lm["rms_db"],
        "log_mel_mean_abs_db": lm["mean_abs_db"],
        "wav_pcc": wav_pcc,
        "timings": out["timings"],
    }
    logger.info(
        f"golden replay: frame_hiddens PCC {res['frame_hiddens_pcc']:.5f} (min {res['frame_hiddens_pcc_min']:.5f}), "
        f"latents {['%.5f' % p for p in latent_pcc]}, log-mel {lm['rms_db']:.3f} dB, wav PCC {wav_pcc:.5f}; "
        f"AR {out['timings']['ar_frames_per_s']:.2f} frames/s, DiT {out['timings']['dit_per_chunk']}, "
        f"vocoder {out['timings']['vocoder_per_chunk']}"
    )
    return res


def free_running(pipe, g: dict, steps: int, seed: int, duration: float, label: str) -> dict:
    m = g["manifest"]
    out = pipe.generate(m["prompt"], m["lyrics"], audio_duration=duration, seed=seed, num_inference_steps=steps)
    ar = out["timings"]["ar_detail"]
    frames_run = out["frames"] + 1
    GENERATED.mkdir(parents=True, exist_ok=True)
    sf.write(
        GENERATED / f"perf_{label}_seed{seed}_{int(duration)}s.wav",
        out["audio"].T,
        out["sampling_rate"],
        subtype="FLOAT",
    )
    res = {
        "frames": out["frames"],
        "stopped_by": out["stopped_by"],
        "seconds": out["audio"].shape[-1] / out["sampling_rate"],
        "ar_frames_per_s": out["timings"]["ar_frames_per_s"],
        "ar_ms_per_frame": 1e3 / out["timings"]["ar_frames_per_s"],
        "llm_step_ms": 1e3 * ar["llm_step"] / max(1, ar["llm_steps"]),
        "depth_loop_ms": 1e3 * ar["depth"] / frames_run,
        "host_ms": 1e3 * ar["host"] / frames_run,
        "dit_per_chunk_s": out["timings"]["dit_per_chunk"],
        "dit_per_step_ms": out["timings"]["dit_per_step_ms"],
        "vocoder_per_chunk_s": out["timings"]["vocoder_per_chunk"],
        "total_s": out["timings"]["total"],
        "audio_stats": audio_stats(torch.from_numpy(out["audio"]), out["sampling_rate"]),
        "code_stats": code_stats(out["codes"]),
        "decode_stats": out.get("decode_stats", {}),
        "timings": {k: v for k, v in out["timings"].items()},
    }
    logger.info(
        f"free-running {duration:.0f} s seed {seed}: {res['ar_frames_per_s']:.2f} frames/s (LLM {res['llm_step_ms']:.1f} + depth "
        f"{res['depth_loop_ms']:.1f} + host {res['host_ms']:.1f} ms), DiT {res['dit_per_chunk_s']}, vocoder {res['vocoder_per_chunk_s']}, total {res['total_s']:.1f} s"
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--policy", default="optimized", help="pipeline dtype_policy preset")
    ap.add_argument("--llm-policy", default=None, help="override the backbone policy (sweep)")
    ap.add_argument("--dit-dtype", default=None, help="override the DiT weight dtype: bf16 | bfp8")
    ap.add_argument("--depth-dtype", default=None, help="override the depth decoder weight dtype: bf16 | bfp8")
    ap.add_argument("--dit-fidelity", default=None, help="override the DiT matmul fidelity: hifi2 | hifi2_fp16 | lofi")
    ap.add_argument("--vocoder", default=None, help="host | device")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--skip-free-running", action="store_true")
    ap.add_argument("--skip-golden", action="store_true")
    ap.add_argument("--write-perf-json", default=None, help="also write doc/optimize/perf.json[<key>]")
    ap.add_argument("--threads", type=int, default=max(8, (os.cpu_count() or 8) - 4))
    ap.add_argument("--trace-region", type=int, default=int(os.environ.get("MM3_TRACE_REGION_SIZE", 200_000_000)))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from models.autoports.minimaxai_minimax_music3.tt.pipeline import MiniMaxMusic3Pipeline

    g = load_golden(R.reference_dir())
    load_kwargs = {"dtype_policy": args.policy}
    for k in ("llm_policy", "dit_dtype", "dit_fidelity", "depth_dtype", "vocoder"):
        v = getattr(args, k)
        if v is not None:
            load_kwargs[k] = v
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=args.trace_region)
    mesh.enable_program_cache()
    result = {
        "label": args.label,
        "load_kwargs": load_kwargs,
        "steps": args.steps,
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": _commit(),
        "loadavg_1m": os.getloadavg()[0],
        "torch_threads": args.threads,
        "trace_region": args.trace_region,
    }
    try:
        t0 = time.perf_counter()
        pipe = MiniMaxMusic3Pipeline.load(mesh, **load_kwargs)
        result["load_s"] = time.perf_counter() - t0
        result["load_log"] = pipe.load_log
        result["policy_report"] = pipe.policy_report() if hasattr(pipe, "policy_report") else {}
        if not args.skip_golden:
            result["golden_replay"] = golden_replay(pipe, g, args.steps)
        if not args.skip_free_running:
            result["free_running"] = free_running(pipe, g, args.steps, args.seed, args.duration, args.label)
        result["dram_after"] = pipe.load_log.get("dram_resident")
        pipe.release()
    finally:
        ttnn.close_mesh_device(mesh)

    out_dir = DOC / "perf_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.label}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    logger.info(f"wrote {path}")

    if args.write_perf_json:
        perf_path = DOC / "perf.json"
        perf = json.loads(perf_path.read_text()) if perf_path.is_file() else {}
        fr = result.get("free_running") or {}
        gr = result.get("golden_replay") or {}
        entry = {
            "label": args.label,
            "load_kwargs": load_kwargs,
            "ar_frames_per_s": fr.get("ar_frames_per_s", gr.get("timings", {}).get("ar_frames_per_s")),
            "llm_step_ms": fr.get("llm_step_ms"),
            "depth_loop_ms": fr.get("depth_loop_ms"),
            "host_ms": fr.get("host_ms"),
            # Headline chunk time: the full 200-frame window (689 latents, S_pad 768), 30 steps.
            "dit_chunk_s": (fr.get("dit_per_chunk_s") or gr.get("timings", {}).get("dit_per_chunk"))[0],
            "dit_chunk_s_all": fr.get("dit_per_chunk_s"),
            "vocoder_chunk_s": (fr.get("vocoder_per_chunk_s") or gr.get("timings", {}).get("vocoder_per_chunk"))[0],
            "total_10s_clip_s": fr.get("total_s"),
            "golden_replay": {k: v for k, v in gr.items() if k != "timings"},
            "commit": result["commit"],
            "recorded_at": result["recorded_at"],
        }
        perf[args.write_perf_json] = entry
        perf_path.write_text(json.dumps(perf, indent=2, sort_keys=True, default=float) + "\n")
        logger.info(f"wrote {perf_path}[{args.write_perf_json}]")


if __name__ == "__main__":
    main()
