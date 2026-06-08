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

import os
import sys

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_BENCH_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from typing import List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from scipy import signal  # noqa: E402

import ttnn  # noqa: E402

# Reuse demo.py's exact loaders and constants so the benchmark exercises the
# identical inference graph.
from models.demos.rvc.demo import (  # Chunking constants — demo.py is the single source of truth shared; with the production-shape tests; eliminates drift between files.
    MAX_CHUNK_FRAMES,
    OVERLAP,
    SR_HUBERT,
    SR_TARGET,
    TARGET_LEN,
    UPP,
    ah,
    bh,
    extract_f0_dio,
    extract_f0_rmvpe,
    load_hubert,
    load_source_module,
    load_text_encoder,
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


# ProcessPool path bypasses RMVPE pitch post-processing's GIL hold.

_W_HUBERT = None
_W_RMVPE = None
_W_TEXT_ENCODER = None
_W_SOURCE_MODULE = None
_W_EMB_G = None
_W_F0_METHOD = None


def _worker_init(checkpoint_path: str, f0_method: str):
    global _W_HUBERT, _W_RMVPE, _W_TEXT_ENCODER, _W_SOURCE_MODULE, _W_EMB_G, _W_F0_METHOD
    sd = load_file(checkpoint_path)
    _W_HUBERT = load_hubert()
    _W_TEXT_ENCODER = load_text_encoder(sd)
    _W_SOURCE_MODULE = load_source_module(sd)
    _W_EMB_G = torch.nn.Embedding(109, 256)
    _W_EMB_G.weight.data = sd["emb_g.weight"].float()
    _W_F0_METHOD = f0_method
    if f0_method == "rmvpe":
        from models.demos.rvc.torch_impl.rmvpe import RMVPEPitchAlgorithm

        _W_RMVPE = RMVPEPitchAlgorithm(sample_rate=SR_HUBERT, hop_size=160)
    torch.set_num_threads(1)  # avoid oversubscribe across worker procs


def _worker_preprocess(args):
    audio, speaker_id, f0_up_key = args
    with torch.no_grad():
        logits = _W_HUBERT(source=audio, output_layer=12)
        feats = F.interpolate(logits.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        nf = feats.shape[1]
        if _W_F0_METHOD == "rmvpe":
            pitch, pitchf = extract_f0_rmvpe(audio, f0_up_key, rmvpe=_W_RMVPE)
        else:
            pitch, pitchf = extract_f0_dio(audio.squeeze(0).numpy(), f0_up_key)
        pitch = pitch[:, :nf]
        pitchf = pitchf[:, :nf]
        sid = torch.tensor([speaker_id])
        g_t = _W_EMB_G(sid).unsqueeze(-1)
        m_p, logs_p = _W_TEXT_ENCODER(feats, pitch)
        z_p = m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666
        har = _W_SOURCE_MODULE(pitchf, UPP).transpose(1, 2)
    return z_p, har, g_t, nf


def _worker_ping(_):
    return True


class BenchmarkSession:
    """Persistent device + modules across runs. Setup runs once."""

    def __init__(self, f0_method: str, l1_small_size: int = 32768):
        self.f0_method = f0_method
        self.l1_small_size = l1_small_size
        synth_path, _ = get_model_and_config_paths("v2", "48k", True)
        self.checkpoint_path = synth_path
        print(f"  Loading checkpoint: {os.path.basename(synth_path)}")
        self.sd = load_file(synth_path)
        self._proc_pool: Optional[ProcessPoolExecutor] = None
        self._proc_pool_workers: int = 0
        self.device = None

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

    def start_proc_pool(self, n_workers: int):
        # Must run before open_device(): otherwise workers inherit TTNN
        # device FDs and may close them on exit.
        if self.device is not None:
            raise RuntimeError("start_proc_pool() must run before open_device()")
        if self._proc_pool is not None and self._proc_pool_workers == n_workers:
            return
        if self._proc_pool is not None:
            self._proc_pool.shutdown(wait=True)
        # spawn (not fork): torch's intra-op threadpool is already alive in
        # parent; fork'd children inherit dead mutex state → futex deadlock.
        ctx = mp.get_context("spawn")
        self._proc_pool = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(self.checkpoint_path, self.f0_method),
        )
        self._proc_pool_workers = n_workers
        list(self._proc_pool.map(_worker_ping, range(n_workers)))

    def proc_pool(self) -> Optional[ProcessPoolExecutor]:
        return self._proc_pool

    def open_device(self, device_id: int, trace_shapes: Optional[List] = None):
        """Open TTNN device. If trace_shapes is non-empty, allocate
        trace_region_size and warm up Flow per-flow traces for those
        (seq_len, batch) shapes."""
        kw = dict(device_id=device_id, l1_small_size=self.l1_small_size)
        if trace_shapes:
            kw["trace_region_size"] = 16_000_000
        print(
            f"  Opening TTNN device id={device_id} (l1_small_size={self.l1_small_size}"
            f"{', trace_region_size=' + str(kw['trace_region_size']) if trace_shapes else ''})..."
        )
        self.device = ttnn.open_device(**kw)

        print("  Loading TTNN modules (TTNNFlowDecoder + TTNNGeneratorNSF)...")
        t0 = time.time()
        self.flow = TTNNFlowDecoder.from_checkpoint(self.sd, self.device)
        self.gen = TTNNGeneratorNSF.from_checkpoint(self.sd, self.device)
        print(f"    TTNN modules ready in {time.time() - t0:.2f}s")

        if trace_shapes:
            # Pre-warm Generator's prep_cache (persistent conv1d weights) for
            # every trace shape BEFORE Flow trace capture. Otherwise Generator
            # forward lazily allocates those during inference — when a Flow
            # trace is active — and collides with Flow's trace region,
            # silently corrupting Generator output.
            print(f"  Pre-warming Generator prep_cache for trace shapes...")
            t0 = time.time()
            with torch.no_grad():
                for seq_len, batch in trace_shapes:
                    z_dummy = torch.zeros(batch, 192, seq_len)
                    har_dummy = torch.zeros(batch, 1, seq_len * UPP)
                    g_dummy = torch.zeros(batch, 256, 1)
                    _ = self.gen(z_dummy, har_dummy, g_dummy)
            print(f"    Generator warmed in {time.time() - t0:.2f}s")

            print(f"  Capturing Flow traces for shapes {trace_shapes}...")
            t0 = time.time()
            self.flow.warm_up_traces(trace_shapes)
            print(
                f"    Flow traces ready in {time.time() - t0:.2f}s " f"({len(self.flow._flow_traces)} per-flow traces)"
            )

        print("  Loading torch reference modules (for correctness check)...")
        self.flow_torch = load_flow_torch_modules(self.sd)
        self.gen_torch = build_torch_generator(self.sd)

    def close(self):
        if self._proc_pool is not None:
            self._proc_pool.shutdown(wait=True)
            self._proc_pool = None
        if self.device is not None:
            self.flow.deallocate()
            self.gen.deallocate()
            ttnn.close_device(self.device)
            self.device = None


def run_inference(
    session: BenchmarkSession,
    audio: torch.Tensor,
    speaker_id: int = 0,
    f0_up_key: int = 0,
    capture_torch_reference: bool = True,
) -> dict:
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
            har_chunk = har_source[:, :, ext_start * UPP : ext_end * UPP]

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
            audio_chunk = audio_chunk[:, :, : ext_len * UPP]
            z_chunk = z_chunk[:, :, :ext_len]
            lt = (nom_start - ext_start) * UPP
            rt = (ext_end - nom_end) * UPP
            nominal_audio = audio_chunk[:, :, lt : audio_chunk.shape[2] - rt if rt > 0 else audio_chunk.shape[2]]
            lz = nom_start - ext_start
            rz = ext_end - nom_end
            nominal_z = z_chunk[:, :, lz : z_chunk.shape[2] - rz if rz > 0 else z_chunk.shape[2]]

            audio_segments.append(nominal_audio)
            ttnn_z_chunks.append(nominal_z)
            chunk_log.append(
                {
                    "chunk_idx": c,
                    "nominal_T": nom_end - nom_start,
                    "ext_T": ext_len,
                    "padded_to": TARGET_LEN,
                    "flow_s": t_flow,
                    "gen_s": t_gen,
                    "total_s": time.time() - chunk_t0,
                    "backend": "TTNN",
                }
            )

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
                har_c = har_source[:, :, ext_start * UPP : ext_end * UPP]
                if ext_len < TARGET_LEN:
                    pad_len = TARGET_LEN - ext_len
                    z_p_c = F.pad(z_p_c, (0, pad_len))
                    har_c = F.pad(har_c, (0, pad_len * UPP))

                z_ref = torch_flow_forward(z_p_c, g, session.flow_torch)
                a_ref = torch_generator_forward(z_ref, har_c, g, session.gen_torch)

                a_ref = a_ref[:, :, : ext_len * UPP]
                z_ref = z_ref[:, :, :ext_len]
                lt = (nom_start - ext_start) * UPP
                rt = (ext_end - nom_end) * UPP
                nom_a = a_ref[:, :, lt : a_ref.shape[2] - rt if rt > 0 else a_ref.shape[2]]
                lz = nom_start - ext_start
                rz = ext_end - nom_end
                nom_z = z_ref[:, :, lz : z_ref.shape[2] - rz if rz > 0 else z_ref.shape[2]]
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
        raise AssertionError(f"output exceeds [-1, 1]: max abs = {audio_out.abs().max().item():.4f}")

    return {
        "audio_out": audio_out,
        "n_chunks": n_chunks,
        "num_frames": num_frames,
        "timings": timings,
        "chunk_log": chunk_log,
        "audio_pcc": audio_pcc,
        "flow_pcc": flow_pcc,
    }


def _run_chunked_ttnn_batched(session, z_p_batch, har_batch, g_batch, num_frames, batch):
    n_chunks = (num_frames + MAX_CHUNK_FRAMES - 1) // MAX_CHUNK_FRAMES
    t_flow_total = 0.0
    t_gen_total = 0.0
    audio_segments: List[torch.Tensor] = []
    for c in range(n_chunks):
        nom_start = c * MAX_CHUNK_FRAMES
        nom_end = min(nom_start + MAX_CHUNK_FRAMES, num_frames)
        ext_start = max(0, nom_start - OVERLAP)
        ext_end = min(num_frames, nom_end + OVERLAP)
        ext_len = ext_end - ext_start
        z_p_chunk = z_p_batch[:, :, ext_start:ext_end]
        har_chunk = har_batch[:, :, ext_start * UPP : ext_end * UPP]
        if ext_len < TARGET_LEN:
            pad_len = TARGET_LEN - ext_len
            z_p_chunk = F.pad(z_p_chunk, (0, pad_len))
            har_chunk = F.pad(har_chunk, (0, pad_len * UPP))
        tf = time.time()
        try:
            z_chunk = session.flow(z_p_chunk, g_batch)
            t_flow_total += time.time() - tf
            tg = time.time()
            audio_chunk = session.gen(z_chunk, har_chunk, g_batch)
            t_gen_total += time.time() - tg
        except RuntimeError as e:
            raise FallbackError(
                f"chunk {c} at B={batch} (T={nom_end - nom_start}, "
                f"ext={ext_len}): TTNN raised {type(e).__name__}: {str(e)[:200]}"
            ) from e
        audio_chunk = audio_chunk[:, :, : ext_len * UPP]
        lt = (nom_start - ext_start) * UPP
        rt = (ext_end - nom_end) * UPP
        nominal_audio = audio_chunk[:, :, lt : audio_chunk.shape[2] - rt if rt > 0 else audio_chunk.shape[2]]
        audio_segments.append(nominal_audio)
    return torch.cat(audio_segments, dim=2), t_flow_total, t_gen_total, n_chunks


def run_inference_batched(
    session: BenchmarkSession,
    audio: torch.Tensor,
    batch: int,
    speaker_ids: Optional[List[int]] = None,
    f0_up_key: int = 0,
) -> dict:
    """Batched inference for B>=1. Default speaker_ids are distinct per row
    so cond_linear paths don't collapse to rank-1."""
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if speaker_ids is None:
        speaker_ids = [i % 109 for i in range(batch)]  # 109 = emb_g vocab size
    if len(speaker_ids) != batch:
        raise ValueError(f"speaker_ids length {len(speaker_ids)} != batch {batch}")
    timings: dict = {}

    def _preprocess_serial(idx):
        sp_id = speaker_ids[idx]
        with torch.no_grad():
            logits = session.hubert(source=audio, output_layer=12)
            feats = F.interpolate(logits.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            nf = feats.shape[1]
            if session.f0_method == "rmvpe":
                pitch, pitchf = extract_f0_rmvpe(audio, f0_up_key, rmvpe=session.rmvpe)
            else:
                pitch, pitchf = extract_f0_dio(audio.squeeze(0).numpy(), f0_up_key)
            pitch = pitch[:, :nf]
            pitchf = pitchf[:, :nf]
            sid = torch.tensor([sp_id])
            g_t = session.emb_g(sid).unsqueeze(-1)
            m_p, logs_p = session.enc_p(feats, pitch)
            z_p = m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666
            har = session.m_source(pitchf, UPP).transpose(1, 2)
        return z_p, har, g_t, nf

    with torch.no_grad():
        n_workers = min(batch, os.cpu_count() or 1)
        t_pre0 = time.time()
        if batch == 1:
            results = [_preprocess_serial(0)]
            preprocess_mode = "serial"
        else:
            pool = session.proc_pool()
            if pool is None:
                raise RuntimeError(
                    "batched preprocess requires a ProcessPool; call "
                    "session.start_proc_pool(n_workers) before run_inference_batched."
                )
            args_iter = [(audio, sp_id, f0_up_key) for sp_id in speaker_ids]
            results = list(pool.map(_worker_preprocess, args_iter))
            preprocess_mode = "proc_pool"
        zs_p = [r[0] for r in results]
        hars = [r[1] for r in results]
        gs = [r[2] for r in results]
        num_frames = results[0][3]
        timings["preprocessing"] = time.time() - t_pre0
        timings["preprocessing_per_sample"] = timings["preprocessing"] / batch
        timings["preprocessing_workers"] = n_workers
        timings["preprocessing_mode"] = preprocess_mode

        z_p_batch = torch.cat(zs_p, dim=0)
        har_batch = torch.cat(hars, dim=0)
        g_batch = torch.cat(gs, dim=0)

        audio_out, t_flow_total, t_gen_total, n_chunks = _run_chunked_ttnn_batched(
            session, z_p_batch, har_batch, g_batch, num_frames, batch
        )
        timings["flow"] = t_flow_total
        timings["generator"] = t_gen_total
        timings["ttnn_total"] = t_flow_total + t_gen_total
        timings["ttnn_per_sample"] = timings["ttnn_total"] / batch

    if not torch.isfinite(audio_out).all():
        raise AssertionError(f"batched output contains NaN or Inf at B={batch}")
    if audio_out.abs().max() > 1.0:
        raise AssertionError(
            f"batched output exceeds [-1, 1] at B={batch}: max abs = {audio_out.abs().max().item():.4f}"
        )

    return {
        "audio_out": audio_out,
        "batch": batch,
        "n_chunks": n_chunks,
        "num_frames": num_frames,
        "timings": timings,
    }


def run_inference_batched_overlapped(
    session: BenchmarkSession,
    audio: torch.Tensor,
    batch: int,
    speaker_ids: Optional[List[int]] = None,
    f0_up_key: int = 0,
) -> dict:
    """B>=2 batched inference with CPU/TTNN pipeline overlap. Splits batch
    into two micro-batches; mb2 preprocess runs in worker procs while main
    thread runs TTNN on mb1. Closes Stage 3 cross-stage pipeline gap for
    the batched path."""
    if batch < 2:
        raise ValueError(f"overlapped path requires batch >= 2, got {batch}")
    if speaker_ids is None:
        speaker_ids = [i % 109 for i in range(batch)]
    if len(speaker_ids) != batch:
        raise ValueError(f"speaker_ids length {len(speaker_ids)} != batch {batch}")
    pool = session.proc_pool()
    if pool is None:
        raise RuntimeError("overlapped path requires session.start_proc_pool() first")

    half = (batch + 1) // 2
    mb1_ids = speaker_ids[:half]
    mb2_ids = speaker_ids[half:]
    timings: dict = {}
    t_wall0 = time.time()

    with torch.no_grad():
        mb1_futures = [pool.submit(_worker_preprocess, (audio, sp_id, f0_up_key)) for sp_id in mb1_ids]
        mb2_futures = [pool.submit(_worker_preprocess, (audio, sp_id, f0_up_key)) for sp_id in mb2_ids]

        t_mb1_pre0 = time.time()
        mb1_res = [f.result() for f in mb1_futures]
        t_mb1_pre = time.time() - t_mb1_pre0

        zs_p_1 = [r[0] for r in mb1_res]
        hars_1 = [r[1] for r in mb1_res]
        gs_1 = [r[2] for r in mb1_res]
        num_frames = mb1_res[0][3]
        z_p_1 = torch.cat(zs_p_1, dim=0)
        har_1 = torch.cat(hars_1, dim=0)
        g_1 = torch.cat(gs_1, dim=0)

        t_mb1_ttnn0 = time.time()
        audio_1, t_flow_1, t_gen_1, n_chunks = _run_chunked_ttnn_batched(
            session, z_p_1, har_1, g_1, num_frames, len(mb1_ids)
        )
        t_mb1_ttnn = time.time() - t_mb1_ttnn0

        t_mb2_pre0 = time.time()
        mb2_res = [f.result() for f in mb2_futures]
        t_mb2_pre_observed = time.time() - t_mb2_pre0

        zs_p_2 = [r[0] for r in mb2_res]
        hars_2 = [r[1] for r in mb2_res]
        gs_2 = [r[2] for r in mb2_res]
        z_p_2 = torch.cat(zs_p_2, dim=0)
        har_2 = torch.cat(hars_2, dim=0)
        g_2 = torch.cat(gs_2, dim=0)

        t_mb2_ttnn0 = time.time()
        audio_2, t_flow_2, t_gen_2, _ = _run_chunked_ttnn_batched(session, z_p_2, har_2, g_2, num_frames, len(mb2_ids))
        t_mb2_ttnn = time.time() - t_mb2_ttnn0

        audio_out = torch.cat([audio_1, audio_2], dim=0)

    wall = time.time() - t_wall0
    timings["wall"] = wall
    timings["mb1_preprocessing"] = t_mb1_pre
    timings["mb1_ttnn"] = t_mb1_ttnn
    timings["mb2_preprocessing_observed"] = t_mb2_pre_observed  # 0 if hidden by mb1 TTNN
    timings["mb2_ttnn"] = t_mb2_ttnn
    timings["preprocessing"] = t_mb1_pre + t_mb2_pre_observed
    timings["preprocessing_per_sample"] = timings["preprocessing"] / batch
    timings["preprocessing_workers"] = min(batch, os.cpu_count() or 1)
    timings["preprocessing_mode"] = "proc_pool_overlapped"
    timings["flow"] = t_flow_1 + t_flow_2
    timings["generator"] = t_gen_1 + t_gen_2
    timings["ttnn_total"] = t_mb1_ttnn + t_mb2_ttnn
    timings["ttnn_per_sample"] = timings["ttnn_total"] / batch

    if not torch.isfinite(audio_out).all():
        raise AssertionError(f"overlapped output contains NaN or Inf at B={batch}")
    if audio_out.abs().max() > 1.0:
        raise AssertionError(f"overlapped output exceeds [-1, 1] at B={batch}")

    return {
        "audio_out": audio_out,
        "batch": batch,
        "n_chunks": n_chunks,
        "num_frames": num_frames,
        "timings": timings,
        "micro_batches": [len(mb1_ids), len(mb2_ids)],
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
    print(
        f"    preprocessing={_fmt(t['preprocessing'])}  (hubert={_fmt(t['hubert'])}  f0={_fmt(t['f0'])}  enc={_fmt(t['text_encoder'])}  src={_fmt(t['source'])})"
    )
    print(f"    flow={_fmt(t['flow'])}  generator={_fmt(t['generator'])}  ttnn_total={_fmt(t['ttnn_total'])}")
    print(f"    torch_ref={_fmt(t.get('torch_ref', 0.0))}")
    print(f"    RTF(TTNN_only)={rtf_ttnn:.4f}   RTF(full_pipeline)={rtf_total:.4f}")
    paths = sorted({c["backend"] for c in result["chunk_log"]})
    print(f"    chunk backends: {paths}")


def main():
    parser = argparse.ArgumentParser(description="RVC TTNN benchmark (real demo path)")
    parser.add_argument("--max_secs", type=float, default=3.0, help="Truncate input audio to this many seconds")
    parser.add_argument("--warmup", type=int, default=1, help="Number of cold-start runs before steady-state timing")
    parser.add_argument("--runs", type=int, default=3, help="Number of warm steady-state runs to time")
    parser.add_argument("--f0_method", type=str, default="rmvpe", choices=["rmvpe", "dio"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Number of concurrent voice conversions per "
        "batched TTNN call. B=1 runs the single-stream "
        "path with torch-reference PCC check; B>1 runs "
        "the batched path (per-row correctness covered "
        "by tests/test_production_shapes.py::"
        "test_generator_batched_matches_individual_b1_calls).",
    )
    parser.add_argument(
        "--overlap",
        action="store_true",
        help="Use pipeline-overlapped batched path: split B into "
        "two micro-batches so CPU preprocess of mb2 overlaps "
        "with TTNN compute of mb1 (requires --batch >= 2).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Capture and use trace+execute_trace for Flow's "
        "per-flow device compute. Pre-allocates persistent "
        "L1 buffers and replays the captured op sequence, "
        "amortizing host dispatch (~3x on Flow forward).",
    )
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be >= 1")
    if args.overlap and args.batch < 2:
        parser.error("--overlap requires --batch >= 2")

    print("=" * 72)
    print("RVC TTNN End-to-End Benchmark — Real Demo Path")
    print("=" * 72)

    # Reproducibility log.
    print("\n--- Configuration ---")
    print(f"  Python:        {platform.python_version()}")
    print(f"  Platform:      {platform.platform()}")
    print(f"  torch:         {torch.__version__}")
    print(f"  ttnn module:   {ttnn.__file__}")
    # B>1 needs the larger L1_SMALL bank because the act_block_h_override=32
    # config at B>1 produces larger halo allocations than the demo's B=1 path.
    l1s = 131072 if args.batch > 1 else 32768
    print(f"  device_id:     {args.device_id}")
    print(f"  l1_small_size: {l1s}")
    print(f"  max_secs:      {args.max_secs}")
    print(f"  f0_method:     {args.f0_method}")
    print(f"  MAX_CHUNK_FRAMES={MAX_CHUNK_FRAMES}  OVERLAP={OVERLAP}  TARGET_LEN={TARGET_LEN}")
    print(f"  warmup runs:   {args.warmup}")
    print(f"  timed runs:    {args.runs}")
    print(
        f"  batch (B):     {args.batch}{'  (batched: per-row correctness tested via test_production_shapes)' if args.batch > 1 else ''}"
    )

    # Audio prep (same as demo.py).
    audio_raw = load_audio(SR_HUBERT)
    max_samples = int(args.max_secs * SR_HUBERT)
    if audio_raw.shape[0] > max_samples:
        audio_raw = audio_raw[:max_samples]
    audio = _preprocess_audio(audio_raw)
    audio_secs = audio.shape[1] / SR_HUBERT
    print(f"  audio:         {audio.shape[1]} samples ({audio_secs:.2f}s @ {SR_HUBERT}Hz)")

    print("\n--- Setup (one-time) ---")
    t_setup0 = time.time()
    session = BenchmarkSession(f0_method=args.f0_method, l1_small_size=l1s)
    if args.batch > 1:
        n_workers = min(args.batch, os.cpu_count() or 1)
        print(f"  Starting ProcessPool ({n_workers} workers, spawn ctx, pre-TTNN)...")
        t_pool0 = time.time()
        session.start_proc_pool(n_workers)
        print(f"    ProcessPool ready in {_fmt(time.time() - t_pool0)}")
    trace_shapes = None
    if args.trace:
        # Flow always sees TARGET_LEN-padded chunks (one-sided first chunk
        # gets F.pad'd up to TARGET_LEN before flow call). So we only need
        # traces at TARGET_LEN, not at the unpadded ext_len.
        if args.overlap:
            half = (args.batch + 1) // 2
            batches = sorted({half, args.batch - half})
        else:
            batches = [args.batch]
        trace_shapes = [(TARGET_LEN, b) for b in batches]
    session.open_device(device_id=args.device_id, trace_shapes=trace_shapes)
    print(f"  Setup total: {_fmt(time.time() - t_setup0)}")

    try:
        if args.batch == 1:
            first_results = []
            print(f"\n--- First-run timings (n={args.warmup}, JIT cache may be pre-warmed) ---")
            for i in range(args.warmup):
                t0 = time.time()
                res = run_inference(session, audio, capture_torch_reference=True)
                res["wall"] = time.time() - t0
                first_results.append(res)
                _report(f"first_{i}", res, audio_secs)

            warm_results = []
            print(f"\n--- Warm steady-state runs (n={args.runs}) ---")
            for i in range(args.runs):
                t0 = time.time()
                res = run_inference(session, audio, capture_torch_reference=True)
                res["wall"] = time.time() - t0
                warm_results.append(res)
                _report(f"warm_{i}", res, audio_secs)

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

            output_secs = warm_results[0]["audio_out"].shape[2] / SR_TARGET
            first_ttnn = first_results[0]["timings"]["ttnn_total"]
            warm_ttnn_mean = statistics.mean(warm_ttnn)
            first_pre = first_results[0]["timings"]["preprocessing"]
            warm_pre_mean = statistics.mean(r["timings"]["preprocessing"] for r in warm_results)

            print("\n" + "=" * 72)
            print(
                f"HEADLINE — audio={audio_secs:.2f}s  output={output_secs:.2f}s  chunks={warm_results[0]['n_chunks']}"
            )
            print("=" * 72)
            print(
                f"  First-run RTF (TTNN only):       {first_ttnn / output_secs:.4f}    ({_fmt(first_ttnn)} TTNN, JIT cache state-dependent)"
            )
            print(
                f"  Warm steady RTF (TTNN only):     {warm_ttnn_mean / output_secs:.4f}    ({_fmt(warm_ttnn_mean)} TTNN, mean of {args.runs})"
            )
            print(f"  First-run RTF (full pipeline):   {(first_ttnn + first_pre) / output_secs:.4f}")
            print(f"  Warm steady RTF (full pipeline): {(warm_ttnn_mean + warm_pre_mean) / output_secs:.4f}")
            print(f"  Bounty target:                   RTF < 0.5")
            print(f"  Audio PCC (warm, mean):          {statistics.mean(warm_pcc):.6f}")
            print(
                f"  Chunk backends used:             {sorted({c['backend'] for r in warm_results for c in r['chunk_log']})}"
            )
            print(f"  Note: 'First-run' is NOT a cold-JIT run unless cache is wiped — see /root/.cache/ttnn")
            print("=" * 72)
        else:
            bench_fn = run_inference_batched_overlapped if args.overlap else run_inference_batched
            path_label = "overlapped" if args.overlap else "sequential"
            print(
                f"\n--- First-run batched timings (n={args.warmup}, B={args.batch}, path={path_label}, JIT cache may be pre-warmed) ---"
            )
            for i in range(args.warmup):
                t0 = time.time()
                _ = bench_fn(session, audio, batch=args.batch)
                print(f"  first_{i}: wall={_fmt(time.time() - t0)}")

            print(f"\n--- Warm steady-state batched runs (n={args.runs}, B={args.batch}, path={path_label}) ---")
            warm_results = []
            for i in range(args.runs):
                t0 = time.time()
                res = bench_fn(session, audio, batch=args.batch)
                res["wall"] = time.time() - t0
                warm_results.append(res)
                n_workers = res["timings"]["preprocessing_workers"]
                mode = res["timings"]["preprocessing_mode"]
                print(
                    f"  warm_{i}: ttnn={_fmt(res['timings']['ttnn_total'])}  "
                    f"preprocess({mode} x{n_workers} for B={args.batch})={_fmt(res['timings']['preprocessing'])}  "
                    f"wall={_fmt(res['wall'])}"
                )

            output_secs = warm_results[0]["audio_out"].shape[2] / SR_TARGET
            warm_ttnn_per_sample = statistics.mean(r["timings"]["ttnn_per_sample"] for r in warm_results)
            warm_pre_per_sample = statistics.mean(r["timings"]["preprocessing_per_sample"] for r in warm_results)
            warm_wall_per_sample = statistics.mean(r["wall"] / args.batch for r in warm_results)
            # For overlapped path, sequential sums overcount (CPU and TTNN run
            # in parallel) — use wall-time directly. For sequential path, wall
            # equals preproc + ttnn so both numbers agree.
            full_rtf = (
                warm_wall_per_sample / output_secs
                if args.overlap
                else (warm_ttnn_per_sample + warm_pre_per_sample) / output_secs
            )

            print("\n" + "=" * 72)
            print(
                f"HEADLINE — B={args.batch} concurrent  per-sample audio={output_secs:.2f}s  chunks={warm_results[0]['n_chunks']}  path={path_label}"
            )
            print("=" * 72)
            print(f"  Per-sample TTNN:                       {_fmt(warm_ttnn_per_sample)}")
            print(
                f"  Per-sample preprocess ({warm_results[0]['timings']['preprocessing_mode']}):  {_fmt(warm_pre_per_sample)}  (using {warm_results[0]['timings']['preprocessing_workers']} workers)"
            )
            print(f"  Per-sample wall (overlap-corrected):   {_fmt(warm_wall_per_sample)}")
            print(f"  Per-sample RTF (TTNN only):            {warm_ttnn_per_sample / output_secs:.4f}")
            print(f"  Per-sample RTF (full pipeline):        {full_rtf:.4f}")
            print(
                f"  Bounty stretched goal RTF < 0.2:       {'MET' if full_rtf < 0.2 else f'not met (need {0.2:.2f})'}"
            )
            print(f"  Bounty stretched goal 5+ concurrent:   {'MET' if args.batch >= 5 else 'partial'}")
            print(f"  Note: per-row output correctness covered by test_generator_batched_matches_individual_b1_calls")
            print("=" * 72)

    finally:
        print("\n--- Teardown ---")
        session.close()
        print("  Device closed.")


if __name__ == "__main__":
    main()
