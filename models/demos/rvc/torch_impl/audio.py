# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from io import BytesIO

import av
import librosa
import numpy as np
import torch

BASE_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
FIXED_AUDIO_FILE = "sample-speech.wav"


def audio2(i, o, format, sr):
    inp = av.open(i, "r")
    out = av.open(o, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "f32le":
        format = "pcm_f32le"

    ostream = out.add_stream(format, rate=sr)
    try:
        ostream.layout = "mono"
    except Exception:
        pass

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    out.close()
    inp.close()


def load_audio(sr):
    file = os.path.abspath(os.path.join(BASE_DIRECTORY, FIXED_AUDIO_FILE))
    if not file.startswith(BASE_DIRECTORY):
        raise RuntimeError(f"Audio file must be located under {BASE_DIRECTORY}")
    if not os.path.exists(file):
        raise RuntimeError("You input a wrong audio path that does not exists, please fix it!")
    try:
        with open(file, "rb") as f:
            with BytesIO() as out:
                audio2(f, out, "f32le", sr)
                out = np.frombuffer(out.getvalue(), np.float32).flatten()
                return torch.from_numpy(out)

    except AttributeError:
        audio = file[1] / 32768.0
        if len(audio.shape) == 2:
            audio = np.mean(audio, -1)
            out = librosa.resample(audio, orig_sr=file[0], target_sr=16000)
            return torch.from_numpy(out)

    # except Exception:
    #     raise RuntimeError(traceback.format_exc())
