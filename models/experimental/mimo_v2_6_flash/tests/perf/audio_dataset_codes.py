# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Code-frame agreement with the fp32 reference on real eval audio: the first 300 items of LibriSpeech test-clean and
of MMAU test-mini (lmms-lab-audio parquet shards, in shard order), each resampled by vLLM's resampler.

    # host: fp32 reference codes and margins, and the fp64 encoder's codes (cached in --out)
    python .../audio_dataset_codes.py --data <dir with the parquet shards> --out codes.pt --mode reference
    # the device tokenizer encoder (tt/audio/tokenizer.py) on a 1x8 row, and the device patch encoder's embeddings
    TT_VISIBLE_DEVICES=<the row's 8 chips> python .../audio_dataset_codes.py --data <dir> --out codes.pt --mode device
    # device matmuls with host elementwise ops (audio_device_matmul_codes.py; one chip; slow)
    TT_VISIBLE_DEVICES=<one chip> python .../audio_dataset_codes.py --data <dir> --out codes.pt --mode hybrid \\
        --dataset librispeech [--scheme split]

Reports per dataset: items, audio seconds, code frames, frames that differ from the fp32 reference (fp64, device), the
items with any change, and the RVQ decision margins of the changed frames against all frames' margins.
"""

import argparse
import glob
import io
import os
import time

import pyarrow.parquet as pq
import soundfile as sf
import torch

from models.experimental.mimo_v2_6_flash.reference import audio as ref
from models.experimental.mimo_v2_6_flash.reference.checkpoint import default_model_path
from models.experimental.mimo_v2_6_flash.tests.perf.audio_codes_margin import margins

ITEMS = 300
SOURCES = {
    "librispeech": "librispeech_test_clean/librispeech_test_clean-*.parquet",
    "mmau": "data/test_mini-*.parquet",
}


def load_items(data_dir, name, n=ITEMS):
    """[(id, wav fp32 24 kHz mono)] for the first ``n`` rows of the dataset's shards."""
    from vllm.multimodal.audio import resample_audio_torchaudio

    out = []
    for path in sorted(glob.glob(os.path.join(data_dir, SOURCES[name]))):
        table = pq.read_table(path)
        ids = table.column("id").to_pylist() if "id" in table.column_names else list(range(table.num_rows))
        for i, audio in zip(ids, table.column("audio").to_pylist()):
            wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            out.append((str(i), torch.from_numpy(resample_audio_torchaudio(wav, orig_sr=sr, target_sr=24000)).float()))
            if len(out) == n:
                return out
    raise RuntimeError(f"{name}: only {len(out)} items under {data_dir}")


def codes_of(enc, mel, cfg, dtype=torch.float32):
    lens = torch.tensor(ref.segment_lengths(mel.shape[0], cfg.segment_size))
    h, _ = enc.features(mel.to(dtype), lens)
    return h, enc.quantize(h)


def summary(name, rows, key):
    frames = sum(r["frames"] for r in rows)
    changed = sum(int(r[key].sum()) for r in rows)
    items = sum(bool(r[key].any()) for r in rows)
    mm = torch.cat([r["margins"][r[key]] for r in rows]) if changed else torch.zeros(0)
    allm = torch.cat([r["margins"] for r in rows])
    q = lambda t, p: t.quantile(p).item() if t.numel() else 0.0
    return (
        f"{name} {key}: {changed}/{frames} frames changed ({100 * changed / frames:.3f} %), {items}/{len(rows)} items; "
        f"changed frames' margins p50 {q(mm, 0.5):.1e} p90 {q(mm, 0.9):.1e} max {mm.max().item() if mm.numel() else 0:.1e} "
        f"(all frames: p10 {q(allm, 0.1):.1e} p50 {q(allm, 0.5):.1e})"
    )


def device_tokenizer(args, cache, weights, cfg):
    """Codes from tt/audio/tokenizer.py on a (1, 8) row, embeddings from tt/audio/patch.py, against the cached fp32
    reference (codes; embeddings from its codes through the fp32 patch encoder on the host)."""
    import ttnn

    from models.experimental.mimo_v2_6_flash.tt.audio.patch import TtAudioPatchEncoder
    from models.experimental.mimo_v2_6_flash.tt.audio.tokenizer import TtAudioTokenizerEncoder

    p32 = ref.AudioPatchEncoder(ref.load_audio_weights(args.model, tokenizer=False), cfg, torch.float32)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=65536, trace_region_size=0)
    try:
        cache_root = os.environ.get("MIMO_AUDIO_CACHE")
        sub = (
            lambda what: os.path.join(cache_root, os.path.basename(os.path.normpath(args.model)), what)
            if cache_root
            else None
        )
        tok = TtAudioTokenizerEncoder.from_checkpoint(mesh, args.model, cache_dir=sub("tokenizer"))
        patch = TtAudioPatchEncoder.from_checkpoint(mesh, args.model, cache_dir=sub("patch"))
        tok.warm()
        patch.warm()
        for name in args.dataset:
            rows, t, t_dev, pccs = cache[name], time.time(), 0.0, []
            for row, (item_id, wav) in zip(rows, load_items(args.data, name)):
                assert row["id"] == item_id
                mel = ref.log_mel_spectrogram(wav, cfg)
                t0 = time.perf_counter()
                codes = tok.codes(mel)
                emb = patch.embed_audio_codes(codes)
                t_dev += time.perf_counter() - t0
                row["device"] = (codes.t() != row["codes"]).any(0)
                with torch.inference_mode():
                    want = p32(ref.pad_and_group_codes(row["codes"].t(), cfg))
                a, b = emb.double().flatten(), want.double().flatten()
                a, b = a - a.mean(), b - b.mean()
                pccs.append((a @ b / (a.norm() * b.norm())).item())
            torch.save(cache, args.out)
            p = torch.tensor(pccs)
            print(
                f"{name}: {len(rows)} items, {sum(r['seconds'] for r in rows):.0f} s of audio; tokenizer + patch encoder "
                f"{t_dev:.1f} s on the device ({time.time() - t:.0f} s with mels and checks); embeddings PCC vs fp32: "
                f"min {p.min().item():.5f}, p1 {p.quantile(0.01).item():.5f}, median {p.median().item():.5f}"
            )
            print(summary(name, rows, "fp64"))
            print(summary(name, rows, "device"), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["reference", "device", "hybrid"], required=True)
    parser.add_argument("--dataset", nargs="+", default=list(SOURCES))
    parser.add_argument("--scheme", default="split", choices=["split", "bf16"])
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    weights = ref.load_audio_weights(args.model, patch=False)
    cfg = ref.MiMoAudioConfig.from_model_path(args.model)
    cache = torch.load(args.out) if os.path.exists(args.out) else {}
    base = ref.AudioTokenizerEncoder(weights, cfg)
    if args.mode == "reference":
        f64 = ref.AudioTokenizerEncoder(weights, cfg, torch.float64)
        for name in args.dataset:
            rows, t = [], time.time()
            for item_id, wav in load_items(args.data, name):
                mel = ref.log_mel_spectrogram(wav, cfg)
                with torch.inference_mode():
                    h0, c0 = codes_of(base, mel, cfg)
                    _, c64 = codes_of(f64, mel, cfg, torch.float64)
                rows.append(
                    {
                        "id": item_id,
                        "seconds": wav.numel() / cfg.sampling_rate,
                        "frames": c0.shape[1],
                        "codes": c0,
                        "margins": margins(base, h0).float(),
                        "fp64": (c64 != c0).any(0),
                    }
                )
            cache[name] = rows
            torch.save(cache, args.out)
            print(
                f"{name}: {len(rows)} items, {sum(r['seconds'] for r in rows):.0f} s of audio, {time.time() - t:.0f} s"
            )
            print(summary(name, rows, "fp64"), flush=True)
        return

    import ttnn

    if args.mode == "device":
        device_tokenizer(args, cache, weights, cfg)
        return

    from models.experimental.mimo_v2_6_flash.tests.perf.audio_device_matmul_codes import (
        DeviceAttentionEncoder,
        DeviceMatmul,
        on_device,
    )

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        dm = DeviceMatmul(mesh, args.scheme == "split")
        enc = DeviceAttentionEncoder(weights, cfg)
        enc.dm = dm
        key = f"device_{args.scheme}"
        for name in args.dataset:
            rows, t = cache[name], time.time()
            for row, (item_id, wav) in zip(rows, load_items(args.data, name)):
                assert row["id"] == item_id
                mel = ref.log_mel_spectrogram(wav, cfg)
                with torch.inference_mode(), on_device(dm):
                    _, c = codes_of(enc, mel, cfg)
                row[key] = (c != row["codes"]).any(0)
            torch.save(cache, args.out)
            print(f"{name}: {len(rows)} items on the device in {time.time() - t:.0f} s")
            print(summary(name, rows, "fp64"))
            print(summary(name, rows, key), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
