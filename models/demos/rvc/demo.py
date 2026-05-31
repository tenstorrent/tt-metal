# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RVC TTNN Hybrid Inference Demo.

End-to-end voice conversion using real checkpoint weights:
    WAV input → Hubert (torch) → Feature retrieval (FAISS, optional)
    → TextEncoder (torch) → F0/RMVPE (torch) → Flow decoder (TTNN)
    → Generator (TTNN) → WAV output

Usage:
    cd <repo_root>
    python -m models.demos.rvc.demo [--f0_method rmvpe] [--index_path path.index]
"""

import sys
import os

# Add repo root to path FIRST so 'models.demos.rvc...' imports work
# (our local ttnn/ package __init__.py uses absolute imports)
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_DEMO_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Now safe to import ttnn (system ttnn is already on sys.path via pip)
import ttnn  # noqa: E402

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from safetensors.torch import load_file
from scipy import signal

from models.demos.rvc.torch_impl.vc.hubert import HubertModel
from models.demos.rvc.torch_impl.vc.synthesizer import TextEncoder, SourceModuleHnNSF
from models.demos.rvc.ttnn.runtime import TTNNFlowDecoder, TTNNGeneratorNSF
from models.demos.rvc.utils.audio import load_audio
from models.demos.rvc.utils.config import (
    Config, HubertPretrainingConfig, HubertPretrainingTask,
    get_hubert_paths, get_model_and_config_paths,
)

# =====================================================================
# Config
# =====================================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SR_HUBERT = 16000
SR_TARGET = 48000
WINDOW = 160
UPP = 480  # product of upsample rates [12, 10, 2, 2]

# Butterworth highpass for preprocessing
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=SR_HUBERT)


def load_hubert(device="cpu"):
    """Load Hubert feature extractor from checkpoint."""
    cfg_path, model_path = get_hubert_paths()
    task = HubertPretrainingTask(HubertPretrainingConfig())
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = HubertModel(cfg=cfg["model"], task_cfg=task.cfg)
    model.load_state_dict(load_file(model_path), strict=True)
    return model.eval().float()


def load_text_encoder(state_dict):
    """Load TextEncoder from synthesizer checkpoint."""
    enc_p = TextEncoder(
        embedding_dims=768, out_channels=192, hidden_channels=192,
        filter_channels=768, num_heads=2, num_layers=6, kernel_size=3, f0=True,
    )
    enc_state = {k.replace("enc_p.", ""): v.float()
                  for k, v in state_dict.items() if k.startswith("enc_p.")}
    enc_p.load_state_dict(enc_state, strict=True)
    return enc_p.eval()


def load_source_module(state_dict):
    """Load SineGen/SourceModule from checkpoint."""
    # validation=False: enable noise + random phase for natural-sounding output
    m_source = SourceModuleHnNSF(sampling_rate=SR_TARGET, harmonic_num=0, validation=False)
    m_state = {k.replace("dec.m_source.", ""): v.float()
                for k, v in state_dict.items() if k.startswith("dec.m_source.")}
    m_source.load_state_dict(m_state, strict=True)
    return m_source.eval()


def extract_f0_dio(audio_np, f0_up_key=0):
    """Extract F0 using pyworld DIO (no extra model needed)."""
    import pyworld as pw

    audio_f64 = audio_np.astype(np.float64)
    frame_period = WINDOW / SR_HUBERT * 1000.0
    f0, t = pw.dio(audio_f64, SR_HUBERT, f0_floor=50, f0_ceil=1100,
                    frame_period=frame_period, allowed_range=0.1)
    f0 = pw.stonemask(audio_f64, f0, t, SR_HUBERT)
    f0 = torch.from_numpy(f0.astype(np.float32)).unsqueeze(0)

    # Apply pitch shift
    f0 *= pow(2, f0_up_key / 12)

    return _f0_to_coarse(f0)


def extract_f0_rmvpe(audio, f0_up_key=0, rmvpe=None):
    """Extract F0 using RMVPE neural pitch estimator (official RVC-Project model).

    If ``rmvpe`` is None a fresh ``RMVPEPitchAlgorithm`` is constructed, which
    reloads the 173 MB checkpoint (~440 ms). Callers that run repeated
    inference (e.g. ``BenchmarkSession``) should build it once and pass it in.
    """
    if rmvpe is None:
        from models.demos.rvc.torch_impl.rmvpe import RMVPEPitchAlgorithm
        rmvpe = RMVPEPitchAlgorithm(sample_rate=SR_HUBERT, hop_size=WINDOW)
    with torch.no_grad():
        f0 = rmvpe.extract_pitch(audio)  # [1, T]
    f0 *= pow(2, f0_up_key / 12)
    return _f0_to_coarse(f0)


def _f0_to_coarse(f0):
    """Convert continuous F0 to mel-scale quantized coarse pitch."""
    f0_continuous = f0.clone()
    f0_mel_min = 1127 * math.log(1 + 50 / 700)
    f0_mel_max = 1127 * math.log(1 + 1100 / 700)
    f0_mel = 1127 * torch.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel = torch.clamp(f0_mel, min=1, max=255)
    f0_coarse = torch.round(f0_mel).to(torch.int64)
    return f0_coarse, f0_continuous


def load_feature_index(index_path):
    """Load FAISS feature index for retrieval-based voice conversion."""
    if not index_path or not os.path.exists(index_path):
        return None, None
    try:
        import faiss
    except ImportError:
        print("  Warning: faiss-cpu not installed, skipping feature retrieval")
        return None, None
    index = faiss.read_index(index_path)
    big_npy = torch.from_numpy(index.reconstruct_n(0, index.ntotal)).float()
    return index, big_npy


# Module-level cache for torch fallback modules (used when TTNN hits L1 OOM)
_torch_fallback_cache = {}


def run_demo(speaker_id=0, f0_up_key=0, device_id=0, max_secs=5.0,
             f0_method="rmvpe", index_path=None, index_rate=0.0):
    """Run full hybrid inference pipeline."""
    print("=" * 60)
    print("RVC TTNN Hybrid Inference Demo")
    print("=" * 60)

    # --- Load audio ---
    t0 = time.time()
    audio_raw = load_audio(SR_HUBERT)
    # Truncate for manageable first demo
    max_samples = int(max_secs * SR_HUBERT)
    if audio_raw.shape[0] > max_samples:
        audio_raw = audio_raw[:max_samples]
        print(f"  Truncated to {max_secs}s ({max_samples} samples)")
    audio_np = audio_raw.numpy()

    # Normalize + highpass
    audio_max = np.abs(audio_np).max()
    if audio_max > 1:
        audio_np = audio_np / audio_max
    audio_np = signal.filtfilt(bh, ah, audio_np)
    audio = torch.from_numpy(audio_np.copy()).unsqueeze(0).float()
    audio_secs = audio.shape[1] / SR_HUBERT
    print(f"  Input: {audio.shape[1]} samples, {audio_secs:.2f}s @ {SR_HUBERT}Hz")

    # --- Load checkpoint ---
    synth_path, _ = get_model_and_config_paths("v2", "48k", True)
    sd = load_file(synth_path)
    print(f"  Checkpoint: {os.path.basename(synth_path)} loaded")

    # --- Load torch modules ---
    hubert = load_hubert()
    enc_p = load_text_encoder(sd)
    m_source = load_source_module(sd)
    emb_g = torch.nn.Embedding(109, 256)
    emb_g.weight.data = sd["emb_g.weight"].float()
    t_load = time.time() - t0
    print(f"  Torch modules loaded in {t_load:.2f}s")

    # --- Open TTNN device ---
    device = ttnn.open_device(device_id=device_id, l1_small_size=32768)

    # --- Load TTNN modules ---
    t0 = time.time()
    flow = TTNNFlowDecoder.from_checkpoint(sd, device)
    gen = TTNNGeneratorNSF.from_checkpoint(sd, device)
    t_ttnn_load = time.time() - t0
    print(f"  TTNN modules loaded in {t_ttnn_load:.2f}s")

    # === INFERENCE ===
    # Chunked processing: Hubert+TextEncoder run on full audio (torch),
    # then z_p is chunked for TTNN flow+generator to stay within L1 limits.
    # Generator L1 budget allows up to 75 frames per chunk on N300; 80+ OOMs.
    MAX_CHUNK_FRAMES = 75  # L1-safe maximum (~0.75s output per chunk)
    OVERLAP = 3  # boundary smoothing context per side; tuned with chunk=75

    print("\n--- Preprocessing (torch, full audio) ---")
    with torch.no_grad():
        # 1. Hubert
        t_hubert_start = time.time()
        logits = hubert(source=audio, output_layer=12)
        feats = logits
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        num_frames = feats.shape[1]
        t_hubert = time.time() - t_hubert_start
        print(f"  Hubert: {audio.shape} → feats {feats.shape}, {t_hubert:.3f}s")

        # 2. F0 extraction
        t_f0_start = time.time()
        if f0_method == "rmvpe":
            pitch, pitchf = extract_f0_rmvpe(audio, f0_up_key)
        else:
            pitch, pitchf = extract_f0_dio(audio_np, f0_up_key)
        pitch = pitch[:, :num_frames]
        pitchf = pitchf[:, :num_frames]
        t_f0 = time.time() - t_f0_start
        print(f"  F0 ({f0_method}): pitch {pitch.shape}, {t_f0:.3f}s")

        # 2b. Feature retrieval (FAISS, optional)
        t_retrieval_start = time.time()
        index, big_npy = load_feature_index(index_path)
        if index is not None and index_rate > 0:
            index_features = feats[0].detach().cpu().numpy()
            scores, indices_found = index.search(index_features, k=8)
            scores = torch.from_numpy(scores).float()
            indices_found = torch.from_numpy(indices_found).long()
            weights = torch.square(1.0 / torch.clamp(scores, min=1e-6))
            weights /= weights.sum(dim=1, keepdim=True)
            retrieved = torch.sum(big_npy[indices_found] * weights.unsqueeze(2), dim=1)
            retrieved = retrieved.unsqueeze(0).to(dtype=feats.dtype)
            feats = retrieved * index_rate + (1 - index_rate) * feats
            print(f"  Retrieval: index_rate={index_rate}, {time.time()-t_retrieval_start:.3f}s")
        else:
            print(f"  Retrieval: disabled (no index or rate=0)")

        # 3. Speaker embedding
        sid = torch.tensor([speaker_id])
        g = emb_g(sid).unsqueeze(-1)

        # 4. TextEncoder (full sequence)
        t_enc_start = time.time()
        m_p, logs_p = enc_p(feats, pitch)
        # Normal inference: add noise for natural-sounding output
        z_p = m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666
        t_enc = time.time() - t_enc_start
        print(f"  TextEncoder: → z_p {z_p.shape}, {t_enc:.3f}s")

        # 5. SineGen (full sequence)
        t_src_start = time.time()
        har_source = m_source(pitchf, UPP).transpose(1, 2)
        t_src = time.time() - t_src_start
        print(f"  SineGen: → har_source {har_source.shape}, {t_src:.3f}s")

    # === CHUNKED TTNN INFERENCE ===
    n_chunks = (num_frames + MAX_CHUNK_FRAMES - 1) // MAX_CHUNK_FRAMES
    print(f"\n--- TTNN Inference ({n_chunks} chunks of ≤{MAX_CHUNK_FRAMES} frames, overlap-add) ---")
    OVERLAP_SAMPLES = OVERLAP * UPP

    ttnn_z_chunks = []
    audio_segments = []
    t_flow_total = 0
    t_gen_total = 0

    with torch.no_grad():
        for c in range(n_chunks):
            # Nominal chunk boundaries
            nom_start = c * MAX_CHUNK_FRAMES
            nom_end = min(nom_start + MAX_CHUNK_FRAMES, num_frames)

            # Extended boundaries with overlap
            ext_start = max(0, nom_start - OVERLAP)
            ext_end = min(num_frames, nom_end + OVERLAP)
            ext_len = ext_end - ext_start

            z_p_chunk = z_p[:, :, ext_start:ext_end]
            har_chunk = har_source[:, :, ext_start * UPP:ext_end * UPP]

            # Pad to uniform shape so conv1d cache reuses entries (prevents OOM)
            target_len = MAX_CHUNK_FRAMES + 2 * OVERLAP
            if ext_len < target_len:
                pad_len = target_len - ext_len
                z_p_chunk = F.pad(z_p_chunk, (0, pad_len))
                har_chunk = F.pad(har_chunk, (0, pad_len * UPP))

            try:
                # Flow (TTNN)
                t0 = time.time()
                z_chunk = flow(z_p_chunk, g)
                t_flow_total += time.time() - t0

                # Generator (TTNN)
                t0 = time.time()
                audio_chunk = gen(z_chunk, har_chunk, g)
                t_gen_total += time.time() - t0
                backend = "TTNN"
            except RuntimeError as e:
                if "out of memory" not in str(e).lower() and "l1" not in str(e).lower():
                    raise  # Re-raise non-OOM errors
                from models.demos.rvc.torch_impl.reference import (
                    load_flow_torch_modules, torch_flow_forward,
                    build_torch_generator, torch_generator_forward,
                )
                if 'flow_mods' not in _torch_fallback_cache:
                    _torch_fallback_cache['flow_mods'] = load_flow_torch_modules(sd)
                    _torch_fallback_cache['gen'] = build_torch_generator(sd)
                z_chunk = torch_flow_forward(z_p_chunk, g, _torch_fallback_cache['flow_mods'])
                audio_chunk = torch_generator_forward(z_chunk, har_chunk, g, _torch_fallback_cache['gen'])
                backend = "torch"

            # Trim: first remove zero-padding, then remove overlap
            # Audio from padded region
            audio_chunk = audio_chunk[:, :, :ext_len * UPP]
            z_chunk = z_chunk[:, :, :ext_len]

            # Trim to nominal region (remove overlap)
            left_trim = (nom_start - ext_start) * UPP
            right_trim = (ext_end - nom_end) * UPP
            nominal_audio = audio_chunk[:, :, left_trim:audio_chunk.shape[2] - right_trim if right_trim > 0 else audio_chunk.shape[2]]

            left_z_trim = nom_start - ext_start
            right_z_trim = ext_end - nom_end
            nominal_z = z_chunk[:, :, left_z_trim:z_chunk.shape[2] - right_z_trim if right_z_trim > 0 else z_chunk.shape[2]]
            ttnn_z_chunks.append(nominal_z)

            audio_segments.append(nominal_audio)
            print(f"    Chunk {c+1}/{n_chunks}: T={nom_end-nom_start} (ext={ext_end-ext_start}) [{backend}] → audio {nominal_audio.shape}")

    # Concatenate (overlap already trimmed, segments are contiguous)
    audio_out = torch.cat(audio_segments, dim=2)
    z_out = torch.cat(ttnn_z_chunks, dim=2)
    print(f"  Total TTNN output: {audio_out.shape}")

    # === TORCH REFERENCE (same overlap-add for fair comparison) ===
    print("\n--- Torch Reference ---")
    with torch.no_grad():
        from models.demos.rvc.torch_impl.reference import (
            load_flow_torch_modules, torch_flow_forward,
            build_torch_generator, torch_generator_forward,
        )
        flow_mods = load_flow_torch_modules(sd)
        gen_torch = build_torch_generator(sd)

        ref_audio_segments = []
        ref_z_chunks = []
        t_ref_start = time.time()
        for c in range(n_chunks):
            nom_start = c * MAX_CHUNK_FRAMES
            nom_end = min(nom_start + MAX_CHUNK_FRAMES, num_frames)
            ext_start = max(0, nom_start - OVERLAP)
            ext_end = min(num_frames, nom_end + OVERLAP)
            ext_len = ext_end - ext_start

            z_p_chunk = z_p[:, :, ext_start:ext_end]
            har_chunk = har_source[:, :, ext_start * UPP:ext_end * UPP]

            # Pad to uniform shape (same as TTNN)
            target_len = MAX_CHUNK_FRAMES + 2 * OVERLAP
            if ext_len < target_len:
                pad_len = target_len - ext_len
                z_p_chunk = F.pad(z_p_chunk, (0, pad_len))
                har_chunk = F.pad(har_chunk, (0, pad_len * UPP))

            z_ref_c = torch_flow_forward(z_p_chunk, g, flow_mods)
            audio_ref_c = torch_generator_forward(z_ref_c, har_chunk, g, gen_torch)

            # Remove padding then overlap
            audio_ref_c = audio_ref_c[:, :, :ext_len * UPP]
            z_ref_c = z_ref_c[:, :, :ext_len]

            left_trim = (nom_start - ext_start) * UPP
            right_trim = (ext_end - nom_end) * UPP
            nominal = audio_ref_c[:, :, left_trim:audio_ref_c.shape[2] - right_trim if right_trim > 0 else audio_ref_c.shape[2]]
            ref_audio_segments.append(nominal)

            left_z = nom_start - ext_start
            right_z = ext_end - nom_end
            nom_z = z_ref_c[:, :, left_z:z_ref_c.shape[2] - right_z if right_z > 0 else z_ref_c.shape[2]]
            ref_z_chunks.append(nom_z)
        t_ref = time.time() - t_ref_start

        audio_ref = torch.cat(ref_audio_segments, dim=2)
        z_ref = torch.cat(ref_z_chunks, dim=2)
        print(f"  Torch: → audio {audio_ref.shape}, {t_ref:.3f}s")

    # === COMPARISON ===
    print("\n--- Comparison ---")
    from models.demos.rvc.tests.pcc_utils import compute_pcc
    flow_pcc = compute_pcc(z_ref, z_out)
    audio_pcc = compute_pcc(audio_ref, audio_out)
    max_err = (audio_ref - audio_out).abs().max().item()
    print(f"  Flow PCC:  {flow_pcc:.6f}")
    print(f"  Audio PCC: {audio_pcc:.6f}")
    print(f"  Max error: {max_err:.6f}")

    # === SAVE OUTPUT ===
    output_dir = os.path.join(DATA_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    audio_np_out = audio_out[0, 0].numpy()
    audio_np_out = audio_np_out / max(np.abs(audio_np_out).max(), 1e-6) * 0.95
    ttnn_path = os.path.join(output_dir, "ttnn_output.wav")
    sf.write(ttnn_path, audio_np_out, SR_TARGET)

    audio_np_ref = audio_ref[0, 0].numpy()
    audio_np_ref = audio_np_ref / max(np.abs(audio_np_ref).max(), 1e-6) * 0.95
    ref_path = os.path.join(output_dir, "torch_reference.wav")
    sf.write(ref_path, audio_np_ref, SR_TARGET)
    print(f"\n  Saved: {ttnn_path}")
    print(f"  Saved: {ref_path}")

    # === TIMING SUMMARY ===
    output_secs = audio_out.shape[2] / SR_TARGET
    t_ttnn_total = t_flow_total + t_gen_total
    t_preprocessing = t_hubert + t_f0 + t_enc + t_src
    rtf = t_ttnn_total / output_secs if output_secs > 0 else float('inf')

    print(f"\n{'=' * 60}")
    print(f"TIMING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Input audio:      {audio_secs:.2f}s")
    print(f"  Output audio:     {output_secs:.2f}s @ {SR_TARGET}Hz")
    print(f"  Chunks:           {n_chunks} × {MAX_CHUNK_FRAMES} frames")
    print(f"  Preprocessing:    {t_preprocessing:.3f}s")
    print(f"  TTNN flow:        {t_flow_total:.3f}s")
    print(f"  TTNN generator:   {t_gen_total:.3f}s")
    print(f"  TTNN total:       {t_ttnn_total:.3f}s")
    print(f"  Torch ref total:  {t_ref:.3f}s")
    print(f"  RTF (TTNN only):  {rtf:.4f}")
    print(f"  Audio PCC:        {audio_pcc:.6f}")
    print(f"{'=' * 60}")

    # Cleanup
    flow.deallocate()
    gen.deallocate()
    ttnn.close_device(device)
    print("  Device closed. Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RVC TTNN Hybrid Inference Demo")
    parser.add_argument("--speaker_id", type=int, default=0, help="Speaker embedding index")
    parser.add_argument("--key", type=int, default=0, help="Pitch shift in semitones")
    parser.add_argument("--device_id", type=int, default=0, help="TTNN device ID")
    parser.add_argument("--max_secs", type=float, default=5.0, help="Max input audio seconds")
    parser.add_argument("--f0_method", type=str, default="rmvpe",
                        choices=["dio", "rmvpe"], help="Pitch extraction method")
    parser.add_argument("--index_path", type=str, default=None,
                        help="Path to FAISS .index file for feature retrieval")
    parser.add_argument("--index_rate", type=float, default=0.0,
                        help="Feature retrieval blending rate (0=disabled, 1=full retrieval)")
    args = parser.parse_args()

    run_demo(speaker_id=args.speaker_id, f0_up_key=args.key,
             device_id=args.device_id, max_secs=args.max_secs,
             f0_method=args.f0_method, index_path=args.index_path,
             index_rate=args.index_rate)
