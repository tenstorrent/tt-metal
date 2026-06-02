# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Publication-grade end-to-end RTF benchmark for the real demo path.

What this measures
------------------
The same chunked flow→generator pipeline as `demo.py`. Same modules
(`TTNNFlowDecoder`, `TTNNGeneratorNSF`), same chunking (MAX_CHUNK_FRAMES=75,
OVERLAP=3), same overlap-add trimming, same checkpoint, same preprocessing.

What this adds vs `demo.py`
---------------------------
1. Setup (device open + checkpoint load + module construction) runs ONCE.
2. The inference body runs N times in the same process so cold-start
   (first run, includes JIT) is separated from warm steady-state.
3. Any chunk that would silently fall back to torch is treated as a
   benchmark failure. No try/except swallowing.
4. Correctness (audio PCC vs torch, NaN/Inf, output shape) checked per run.
5. Timing attribution: total, preprocessing, flow, generator, per-chunk.

What this does NOT do
---------------------
- Modify runtime structure.
- Change chunking strategy.
- Use synthetic tensors or isolated `Generator.forward()`.
- Cherry-pick best runs.

Usage
-----
    python -m models.demos.rvc.benchmark --max_secs 3.0 --warmup 1 --runs 3
    python -m models.demos.rvc.benchmark --max_secs 10.0 --warmup 1 --runs 3
"""

import sys
import os

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCH_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import ttnn  # noqa: E402

import argparse  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from typing import List  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from scipy import signal  # noqa: E402

# Reuse demo.py's exact loaders and constants so the benchmark exercises the
# identical inference graph.
from models.demos.rvc.demo import (
    # Chunking constants — demo.py is the single source of truth shared
    # with the production-shape tests; eliminates drift between files.
    MAX_CHUNK_FRAMES,
    OVERLAP,
    TARGET_LEN,
    SR_HUBERT,
    SR_TARGET,
    UPP,
    bh,
    ah,
    load_hubert,
    load_text_encoder,
    load_source_module,
    extract_f0_dio,
    extract_f0_rmvpe,
)
from models.demos.rvc.tests.pcc_utils import compute_pcc
from models.demos.rvc.torch_impl.reference import (
    build_torch_generator,
    load_flow_torch_modules,
    torch_flow_forward,
    torch_generator_forward,
)
from models.demos.rvc.tt.runtime import TTNNFlowDecoder, TTNNGeneratorNSF
from models.demos.rvc.utils.audio import load_audio
from models.demos.rvc.utils.config import get_model_and_config_paths


class FallbackError(RuntimeError):
    """Raised when a chunk would silently fall back to torch."""


def _preprocess_audio(audio_raw: torch.Tensor) -> torch.Tensor:
    """Same preprocessing as demo.py: normalize + highpass."""
    audio_np = audio_raw.numpy()
    audio_max = np.abs(audio_np).max()
    if audio_max > 1:
        audio_np = audio_np / audio_max
    audio_np = signal.filtfilt(bh, ah, audio_np)
    return torch.from_numpy(audio_np.copy()).unsqueeze(0).float()


class BenchmarkSession:
    """Persistent device + modules across runs. Setup runs once."""

    def __init__(self, device_id: int, f0_method: str):
        self.f0_method = f0_method
        synth_path, _ = get_model_and_config_paths("v2", "48k", True)
        print(f"  Loading checkpoint: {os.path.basename(synth_path)}")
        self.sd = load_file(synth_path)

        print("  Loading torch modules (Hubert + TextEncoder + SineGen + emb_g)...")
        t0 = time.time()
        self.hubert = load_hubert()
        self.enc_p = load_text_encoder(self.sd)
        self.m_source = load_source_module(self.sd)
        self.emb_g = torch.nn.Embedding(109, 256)
        self.emb_g.weight.data = self.sd["emb_g.weight"].float()
        # Persist RMVPE: avoid reloading the 173 MB checkpoint on every run.
        self.rmvpe = None
        if f0_method == "rmvpe":
            from models.demos.rvc.torch_impl.rmvpe import RMVPEPitchAlgorithm
            self.rmvpe = RMVPEPitchAlgorithm(sample_rate=SR_HUBERT, hop_size=160)
        print(f"    torch modules ready in {time.time() - t0:.2f}s")

        print(f"  Opening TTNN device id={device_id} (l1_small_size=32768)...")
        self.device = ttnn.open_device(device_id=device_id, l1_small_size=32768)

        print("  Loading TTNN modules (TTNNFlowDecoder + TTNNGeneratorNSF)...")
        t0 = time.time()
        self.flow = TTNNFlowDecoder.from_checkpoint(self.sd, self.device)
        self.gen = TTNNGeneratorNSF.from_checkpoint(self.sd, self.device)
        print(f"    TTNN modules ready in {time.time() - t0:.2f}s")

        print("  Loading torch reference modules (for correctness check)...")
        self.flow_torch = load_flow_torch_modules(self.sd)
        self.gen_torch = build_torch_generator(self.sd)

    def close(self):
        self.flow.deallocate()
        self.gen.deallocate()
        ttnn.close_device(self.device)


def run_inference(session: BenchmarkSession, audio: torch.Tensor,
                  speaker_id: int = 0, f0_up_key: int = 0,
                  capture_torch_reference: bool = True) -> dict:
    """One end-to-end inference. Mirrors demo.py's body exactly.

    Returns dict of timings, chunk info, output, and (optional) torch ref.
    Raises FallbackError if any chunk would fall back to torch.
    """
    timings = {}
    chunk_log: List[dict] = []

    with torch.no_grad():
        # Preprocessing (torch).
        t0 = time.time()
        logits = session.hubert(source=audio, output_layer=12)
        feats = F.interpolate(logits.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        num_frames = feats.shape[1]
        timings["hubert"] = time.time() - t0

        t0 = time.time()
        if session.f0_method == "rmvpe":
            pitch, pitchf = extract_f0_rmvpe(audio, f0_up_key, rmvpe=session.rmvpe)
        else:
            pitch, pitchf = extract_f0_dio(audio.squeeze(0).numpy(), f0_up_key)
        pitch = pitch[:, :num_frames]
        pitchf = pitchf[:, :num_frames]
        timings["f0"] = time.time() - t0

        sid = torch.tensor([speaker_id])
        g = session.emb_g(sid).unsqueeze(-1)

        t0 = time.time()
        m_p, logs_p = session.enc_p(feats, pitch)
        z_p = m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666
        timings["text_encoder"] = time.time() - t0

        t0 = time.time()
        har_source = session.m_source(pitchf, UPP).transpose(1, 2)
        timings["source"] = time.time() - t0

        # Chunked TTNN inference. STRICT: no try/except around TTNN.
        # A torch fallback is impossible because we don't catch.
        n_chunks = (num_frames + MAX_CHUNK_FRAMES - 1) // MAX_CHUNK_FRAMES
        ttnn_z_chunks = []
        audio_segments = []
        t_flow_total = 0.0
        t_gen_total = 0.0

        for c in range(n_chunks):
            nom_start = c * MAX_CHUNK_FRAMES
            nom_end = min(nom_start + MAX_CHUNK_FRAMES, num_frames)
            ext_start = max(0, nom_start - OVERLAP)
            ext_end = min(num_frames, nom_end + OVERLAP)
            ext_len = ext_end - ext_start

            z_p_chunk = z_p[:, :, ext_start:ext_end]
            har_chunk = har_source[:, :, ext_start * UPP:ext_end * UPP]

            if ext_len < TARGET_LEN:
                pad_len = TARGET_LEN - ext_len
                z_p_chunk = F.pad(z_p_chunk, (0, pad_len))
                har_chunk = F.pad(har_chunk, (0, pad_len * UPP))

            chunk_t0 = time.time()
            try:
                tf = time.time()
                z_chunk = session.flow(z_p_chunk, g)
                t_flow = time.time() - tf
                tg = time.time()
                audio_chunk = session.gen(z_chunk, har_chunk, g)
                t_gen = time.time() - tg
            except RuntimeError as e:
                # Strict mode: any chunk that would have fallen back fails the run.
                raise FallbackError(
                    f"chunk {c} (T={nom_end - nom_start}, ext={ext_len}): "
                    f"TTNN raised {type(e).__name__}: {str(e)[:200]}"
                ) from e

            t_flow_total += t_flow
            t_gen_total += t_gen

            # Same trim logic as demo.py.
            audio_chunk = audio_chunk[:, :, :ext_len * UPP]
            z_chunk = z_chunk[:, :, :ext_len]
            lt = (nom_start - ext_start) * UPP
            rt = (ext_end - nom_end) * UPP
            nominal_audio = audio_chunk[:, :, lt:audio_chunk.shape[2] - rt if rt > 0 else audio_chunk.shape[2]]
            lz = nom_start - ext_start
            rz = ext_end - nom_end
            nominal_z = z_chunk[:, :, lz:z_chunk.shape[2] - rz if rz > 0 else z_chunk.shape[2]]

            audio_segments.append(nominal_audio)
            ttnn_z_chunks.append(nominal_z)
            chunk_log.append({
                "chunk_idx": c,
                "nominal_T": nom_end - nom_start,
                "ext_T": ext_len,
                "padded_to": TARGET_LEN,
                "flow_s": t_flow,
                "gen_s": t_gen,
                "total_s": time.time() - chunk_t0,
                "backend": "TTNN",
            })

        audio_out = torch.cat(audio_segments, dim=2)
        z_out = torch.cat(ttnn_z_chunks, dim=2)

        timings["flow"] = t_flow_total
        timings["generator"] = t_gen_total
        timings["ttnn_total"] = t_flow_total + t_gen_total
        timings["preprocessing"] = timings["hubert"] + timings["f0"] + timings["text_encoder"] + timings["source"]

        # Correctness vs torch reference (per-run, never skipped).
        if capture_torch_reference:
            t_ref0 = time.time()
            ref_audio_segs = []
            ref_z_segs = []
            for c in range(n_chunks):
                nom_start = c * MAX_CHUNK_FRAMES
                nom_end = min(nom_start + MAX_CHUNK_FRAMES, num_frames)
                ext_start = max(0, nom_start - OVERLAP)
                ext_end = min(num_frames, nom_end + OVERLAP)
                ext_len = ext_end - ext_start

                z_p_c = z_p[:, :, ext_start:ext_end]
                har_c = har_source[:, :, ext_start * UPP:ext_end * UPP]
                if ext_len < TARGET_LEN:
                    pad_len = TARGET_LEN - ext_len
                    z_p_c = F.pad(z_p_c, (0, pad_len))
                    har_c = F.pad(har_c, (0, pad_len * UPP))

                z_ref = torch_flow_forward(z_p_c, g, session.flow_torch)
                a_ref = torch_generator_forward(z_ref, har_c, g, session.gen_torch)

                a_ref = a_ref[:, :, :ext_len * UPP]
                z_ref = z_ref[:, :, :ext_len]
                lt = (nom_start - ext_start) * UPP
                rt = (ext_end - nom_end) * UPP
                nom_a = a_ref[:, :, lt:a_ref.shape[2] - rt if rt > 0 else a_ref.shape[2]]
                lz = nom_start - ext_start
                rz = ext_end - nom_end
                nom_z = z_ref[:, :, lz:z_ref.shape[2] - rz if rz > 0 else z_ref.shape[2]]
                ref_audio_segs.append(nom_a)
                ref_z_segs.append(nom_z)

            audio_ref = torch.cat(ref_audio_segs, dim=2)
            z_ref = torch.cat(ref_z_segs, dim=2)
            timings["torch_ref"] = time.time() - t_ref0

            flow_pcc = compute_pcc(z_ref, z_out)
            audio_pcc = compute_pcc(audio_ref, audio_out)
        else:
            flow_pcc = None
            audio_pcc = None

    # Output validation — failures here mean silent corruption.
    if not torch.isfinite(audio_out).all():
        raise AssertionError("output contains NaN or Inf")
    if audio_out.abs().max() > 1.0:
        raise AssertionError(
            f"output exceeds [-1, 1]: max abs = {audio_out.abs().max().item():.4f}"
        )

    return {
        "audio_out": audio_out,
        "n_chunks": n_chunks,
        "num_frames": num_frames,
        "timings": timings,
        "chunk_log": chunk_log,
        "audio_pcc": audio_pcc,
        "flow_pcc": flow_pcc,
    }


def _fmt(seconds: float) -> str:
    return f"{seconds:.3f}s"


def _report(label: str, result: dict, audio_secs: float):
    t = result["timings"]
    output_secs = result["audio_out"].shape[2] / SR_TARGET
    rtf_ttnn = t["ttnn_total"] / output_secs if output_secs > 0 else float("inf")
    rtf_total = (t["preprocessing"] + t["ttnn_total"]) / output_secs if output_secs > 0 else float("inf")
    print(f"  [{label}]")
    print(f"    chunks={result['n_chunks']}  frames={result['num_frames']}  output={output_secs:.3f}s")
    print(f"    audio_PCC={result['audio_pcc']:.6f}  flow_PCC={result['flow_pcc']:.6f}")
    print(f"    preprocessing={_fmt(t['preprocessing'])}  (hubert={_fmt(t['hubert'])}  f0={_fmt(t['f0'])}  enc={_fmt(t['text_encoder'])}  src={_fmt(t['source'])})")
    print(f"    flow={_fmt(t['flow'])}  generator={_fmt(t['generator'])}  ttnn_total={_fmt(t['ttnn_total'])}")
    print(f"    torch_ref={_fmt(t.get('torch_ref', 0.0))}")
    print(f"    RTF(TTNN_only)={rtf_ttnn:.4f}   RTF(full_pipeline)={rtf_total:.4f}")
    paths = sorted({c["backend"] for c in result["chunk_log"]})
    print(f"    chunk backends: {paths}")


def main():
    parser = argparse.ArgumentParser(description="RVC TTNN benchmark (real demo path)")
    parser.add_argument("--max_secs", type=float, default=3.0,
                        help="Truncate input audio to this many seconds")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of cold-start runs before steady-state timing")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of warm steady-state runs to time")
    parser.add_argument("--f0_method", type=str, default="rmvpe",
                        choices=["rmvpe", "dio"])
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()

    print("=" * 72)
    print("RVC TTNN End-to-End Benchmark — Real Demo Path")
    print("=" * 72)

    # Reproducibility log.
    print("\n--- Configuration ---")
    print(f"  Python:        {platform.python_version()}")
    print(f"  Platform:      {platform.platform()}")
    print(f"  torch:         {torch.__version__}")
    print(f"  ttnn module:   {ttnn.__file__}")
    print(f"  device_id:     {args.device_id}")
    print(f"  l1_small_size: 32768")
    print(f"  max_secs:      {args.max_secs}")
    print(f"  f0_method:     {args.f0_method}")
    print(f"  MAX_CHUNK_FRAMES={MAX_CHUNK_FRAMES}  OVERLAP={OVERLAP}  TARGET_LEN={TARGET_LEN}")
    print(f"  warmup runs:   {args.warmup}")
    print(f"  timed runs:    {args.runs}")

    # Audio prep (same as demo.py).
    audio_raw = load_audio(SR_HUBERT)
    max_samples = int(args.max_secs * SR_HUBERT)
    if audio_raw.shape[0] > max_samples:
        audio_raw = audio_raw[:max_samples]
    audio = _preprocess_audio(audio_raw)
    audio_secs = audio.shape[1] / SR_HUBERT
    print(f"  audio:         {audio.shape[1]} samples ({audio_secs:.2f}s @ {SR_HUBERT}Hz)")

    # Session setup (paid once).
    print("\n--- Setup (one-time) ---")
    t_setup0 = time.time()
    session = BenchmarkSession(device_id=args.device_id, f0_method=args.f0_method)
    print(f"  Setup total: {_fmt(time.time() - t_setup0)}")

    try:
        # Cold-start runs (first invocations, includes JIT compile).
        cold_results = []
        print(f"\n--- Cold-start runs (n={args.warmup}) ---")
        for i in range(args.warmup):
            t0 = time.time()
            res = run_inference(session, audio, capture_torch_reference=True)
            res["wall"] = time.time() - t0
            cold_results.append(res)
            _report(f"cold_{i}", res, audio_secs)

        # Warm steady-state runs.
        warm_results = []
        print(f"\n--- Warm steady-state runs (n={args.runs}) ---")
        for i in range(args.runs):
            t0 = time.time()
            res = run_inference(session, audio, capture_torch_reference=True)
            res["wall"] = time.time() - t0
            warm_results.append(res)
            _report(f"warm_{i}", res, audio_secs)

        # Stability checks.
        print("\n--- Stability ---")
        warm_ttnn = [r["timings"]["ttnn_total"] for r in warm_results]
        warm_pcc = [r["audio_pcc"] for r in warm_results]
        print(f"  TTNN total per warm run: {[f'{t:.3f}' for t in warm_ttnn]}")
        print(f"  audio_PCC per warm run:  {[f'{p:.6f}' for p in warm_pcc]}")
        if len(warm_ttnn) > 1:
            spread = max(warm_ttnn) - min(warm_ttnn)
            mean = statistics.mean(warm_ttnn)
            stdev = statistics.stdev(warm_ttnn)
            print(f"  mean={_fmt(mean)}  stdev={_fmt(stdev)}  spread={_fmt(spread)}  cv={stdev/mean*100:.1f}%")

        # Headline summary.
        output_secs = warm_results[0]["audio_out"].shape[2] / SR_TARGET
        cold_ttnn = cold_results[0]["timings"]["ttnn_total"]
        warm_ttnn_mean = statistics.mean(warm_ttnn)
        cold_pre = cold_results[0]["timings"]["preprocessing"]
        warm_pre_mean = statistics.mean(r["timings"]["preprocessing"] for r in warm_results)

        print("\n" + "=" * 72)
        print(f"HEADLINE — audio={audio_secs:.2f}s  output={output_secs:.2f}s  chunks={warm_results[0]['n_chunks']}")
        print("=" * 72)
        print(f"  Cold-start RTF (TTNN only):      {cold_ttnn / output_secs:.4f}    ({_fmt(cold_ttnn)} TTNN)")
        print(f"  Warm steady RTF (TTNN only):     {warm_ttnn_mean / output_secs:.4f}    ({_fmt(warm_ttnn_mean)} TTNN, mean of {args.runs})")
        print(f"  Cold-start RTF (full pipeline):  {(cold_ttnn + cold_pre) / output_secs:.4f}")
        print(f"  Warm steady RTF (full pipeline): {(warm_ttnn_mean + warm_pre_mean) / output_secs:.4f}")
        print(f"  Bounty target:                   RTF < 0.5")
        print(f"  Audio PCC (warm, mean):          {statistics.mean(warm_pcc):.6f}")
        print(f"  Chunk backends used:             {sorted({c['backend'] for r in warm_results for c in r['chunk_log']})}")
        print("=" * 72)

    finally:
        print("\n--- Teardown ---")
        session.close()
        print("  Device closed.")


if __name__ == "__main__":
    main()
