"""t430 step 2 (CPU only): time assemble_clip_parts of one tree on 5 s 768p yuv420 planar clips.

Run with PYTHONPATH=<tree>; prints min/median over REPS. Parts are built here (same slicing as
clip_parts in b0f45278d72) so both trees assemble identical inputs. Prints a checksum of the result.
"""
import os
import statistics
import sys
import time
import zlib

import numpy as np

from models.tt_dit.models.vae.minimax_h3 import vae_minimax_h3 as m

reps = int(os.environ.get("REPS", "7"))
seconds = int(os.environ.get("SECONDS_", "5"))
num_clips = {5: 7, 15: 21}[seconds]
config = m.MiniMaxH3VaeConfig()
chunk = config.tokens_chunk_size * config.temporal_compression_ratio
clip_len = chunk + config.token_overlap * config.temporal_compression_ratio
rng = np.random.default_rng(0)
clips = [rng.integers(0, 256, size=(clip_len, 1152, 1344), dtype=np.uint8) for _ in range(num_clips)]


def split(clip):
    pieces = []
    for j in range(int(config.token_drop > 0) + 1):
        piece = clip[j * chunk : (j + 1) * chunk]
        pieces.append(piece[config.frame_pre_padding :])
    return pieces[0], (pieces[1] if len(pieces) > 1 else None)


parts, overlap = [], None
for clip in clips:
    segment, tail = split(clip)
    parts.append((segment, overlap))
    if tail is not None:
        overlap = tail
if overlap is not None:
    parts.append((overlap, None))

times = []
for _ in range(reps + 1):
    t = time.perf_counter()
    out = m.assemble_clip_parts(parts, config.frame_overlap)
    times.append(time.perf_counter() - t)
times = times[1:]
print(
    f"ASSEMBLE tree={sys.argv[1]} seconds={seconds} affinity={len(os.sched_getaffinity(0))} "
    f"shape={out.shape} min={min(times) * 1000:.1f}ms median={statistics.median(times) * 1000:.1f}ms "
    f"crc={zlib.crc32(out.tobytes()):08x}"
)
