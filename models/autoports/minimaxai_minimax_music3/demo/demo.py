"""CLI: caption + lyrics -> WAV on Tenstorrent.
  python -m models.autoports.minimaxai_minimax_music3.demo.demo --caption "..." --lyrics-file lyrics.txt --duration 30 --seed 7 --out /tmp/song
"""
import argparse
import json
import os

import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lyrics")
    g.add_argument("--lyrics-file")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-rate", type=int, default=32000, choices=(32000, 44100))
    a = ap.parse_args()
    lyrics = a.lyrics if a.lyrics is not None else open(a.lyrics_file).read()
    from models.autoports.minimaxai_minimax_music3.server import audio_io
    from models.autoports.minimaxai_minimax_music3.tt.device import open_mesh
    from models.autoports.minimaxai_minimax_music3.tt.generator import Music3Generator

    os.makedirs(a.out, exist_ok=True)
    h = open_mesh()
    try:
        gen = Music3Generator(h.mesh)
        out = gen.generate(a.caption, lyrics, audio_duration=a.duration, seed=a.seed, num_steps=a.steps)
        audio = audio_io.resample(out["audio"].squeeze(0).numpy(), out["sample_rate"], a.sample_rate)
        sf.write(os.path.join(a.out, "song.wav"), audio.T, a.sample_rate, subtype="PCM_16")
        json.dump(out["stats"].as_dict(), open(os.path.join(a.out, "stats.json"), "w"), indent=2)
        print(json.dumps(out["stats"].as_dict(), indent=2))
    finally:
        h.close()


if __name__ == "__main__":
    main()
