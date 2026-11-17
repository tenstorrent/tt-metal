# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Speech test clips for the audio tests: LibriSpeech test-clean (public domain, LibriVox) cut to ~5 / 30 / 65 /
120 / 300 / 600 s.

The ~5 s clip is the single utterance closest to 5 s; the longer ones are consecutive utterances of the longest
chapter (one speaker reading in order), then of the whole shard in id order once the chapter runs out, joined with
0.3 s of silence. Written as 16 kHz mono 16-bit WAV, LibriSpeech's native rate, so the serving path has to resample
to 24 kHz as it would for a user's file. The 65 s clip is over 6000 mel frames (60 s): it exercises the segmenting.

    curl -L -o test.clean.parquet \\
      https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/all/test.clean/0000.parquet
    python models/experimental/mimo_v2_6_flash/reference/make_audio_clips.py test.clean.parquet <out_dir>

Then point the tests at it with ``MIMO_AUDIO_CLIPS=<out_dir>``.
"""

import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SR = 16000
TARGETS = (
    ("speech_30s", 30.0),
    ("speech_65s", 65.5),
    ("speech_120s", 120.0),
    ("speech_300s", 300.0),
    ("speech_600s", 600.0),
)


def main(src, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = pq.read_table(src, columns=["audio", "text", "speaker_id", "chapter_id", "id"]).to_pylist()
    for r in rows:
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        assert sr == SR and wav.ndim == 1, (sr, wav.shape)
        r["wav"] = wav
    chapters = {}
    for r in rows:
        chapters.setdefault((r["speaker_id"], r["chapter_id"]), []).append(r)
    longest = max(chapters.values(), key=lambda rs: sum(len(r["wav"]) for r in rs))
    longest.sort(key=lambda r: r["id"])
    taken = {r["id"] for r in longest}
    pool = longest + sorted((r for r in rows if r["id"] not in taken), key=lambda r: r["id"])
    gap = np.zeros(int(0.3 * SR), dtype=np.float32)

    manifest = []
    one = min(rows, key=lambda r: abs(len(r["wav"]) / SR - 5.0))
    sf.write(os.path.join(out_dir, "speech_5s.wav"), one["wav"], SR, subtype="PCM_16")
    manifest.append({"name": "speech_5s", "seconds": len(one["wav"]) / SR, "ids": [one["id"]], "text": one["text"]})
    for name, target in TARGETS:
        parts, ids, texts, n = [], [], [], 0
        for r in pool:
            if n + len(r["wav"]) > target * SR:
                break
            parts += [r["wav"], gap]
            n += len(r["wav"]) + len(gap)
            ids.append(r["id"])
            texts.append(r["text"])
        wav = np.concatenate(parts)
        assert len(wav) > (target - 30.0) * SR, (name, len(wav) / SR)
        if name == "speech_65s":
            assert len(wav) > 60.5 * SR, "the 65 s clip must be over one 6000-frame segment"
        sf.write(os.path.join(out_dir, f"{name}.wav"), wav, SR, subtype="PCM_16")
        manifest.append({"name": name, "seconds": len(wav) / SR, "ids": ids, "text": " ".join(texts)})
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    for m in manifest:
        print(f"{m['name']}: {m['seconds']:.2f} s, {len(m['ids'])} utterances")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
