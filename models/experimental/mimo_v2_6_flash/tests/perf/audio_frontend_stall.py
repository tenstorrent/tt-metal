# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Does audio tokenization in the front end (tt/audio/processor.py) stall the API server's streaming?

vLLM 0.30.0's renderer, built as an API server builds it (no engine, no device), processes a long clip through its
own async path (``_process_multimodal_async``: the single-worker ``_mm_executor`` thread) while a coroutine on the
same event loop "streams" a text request: every ``--interval`` ms it detokenizes one token and serializes an SSE
chunk, as the output handler does per token. Reported: the gaps between the stream's chunks before, during and
(for comparison) with the same processing called synchronously on the loop, plus the latency of a short clip queued
behind the long one.

    python models/experimental/mimo_v2_6_flash/tests/perf/audio_frontend_stall.py --clips <dir> [--clip speech_120s]
"""

import argparse
import asyncio
import json
import os
import statistics
import time

import soundfile as sf

from models.experimental.mimo_v2_6_flash.reference.checkpoint import default_model_path
from models.experimental.mimo_v2_6_flash.tests.test_audio_processor_host import audio_prompt, build_renderer


class Stream:
    """A text stream on the event loop: one detokenized token + one SSE chunk every ``interval`` s."""

    def __init__(self, tokenizer, interval):
        self.tokenizer, self.interval = tokenizer, interval
        self.stamps = []
        self.running = True

    async def run(self):
        ids = self.tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 8)
        i = 0
        while self.running:
            text = self.tokenizer.decode(ids[: i % len(ids) + 1])[-16:]
            json.dumps({"choices": [{"delta": {"content": text}, "index": 0}], "object": "chat.completion.chunk"})
            self.stamps.append(time.perf_counter())
            i += 1
            await asyncio.sleep(self.interval)

    def gaps_ms(self, start, end):
        """Gaps between the chunks in [start, end], counting from the last chunk before ``start``."""
        before = [t for t in self.stamps if t < start][-1:]
        s = before + [t for t in self.stamps if start <= t <= end]
        return [1e3 * (b - a) for a, b in zip(s, s[1:])]


def summary(gaps):
    if not gaps:
        return "no chunks"
    g = sorted(gaps)
    return f"{len(g) + 1} chunks, gap median {statistics.median(g):.1f} ms, p99 {g[int(0.99 * (len(g) - 1))]:.1f} ms, max {g[-1]:.1f} ms"


async def main_async(args):
    renderer = build_renderer(args.model)
    wav, sr = sf.read(os.path.join(args.clips, f"{args.clip}.wav"), dtype="float32")
    short, short_sr = sf.read(os.path.join(args.clips, "speech_5s.wav"), dtype="float32")
    stream = Stream(renderer.tokenizer, args.interval / 1e3)
    task = asyncio.create_task(stream.run())

    async def process(audio, label):
        t = time.perf_counter()
        await renderer._process_multimodal_async(audio_prompt(renderer, 1), {"audio": [audio]}, None, None)
        return label, t, time.perf_counter()

    await process((short, short_sr), "load")  # first audio item: loads the front-end tokenizer encoder
    await asyncio.sleep(1.0)
    t0 = time.perf_counter()
    await asyncio.sleep(3.0)
    print(f"baseline (3 s): {summary(stream.gaps_ms(t0, time.perf_counter()))}", flush=True)

    renderer.clear_mm_cache()
    _, a, b = await process((wav, sr), args.clip)
    print(
        f"{args.clip} in the mm executor: {b - a:.2f} s; stream meanwhile: {summary(stream.gaps_ms(a, b))}", flush=True
    )

    renderer.clear_mm_cache()
    a = time.perf_counter()
    results = await asyncio.gather(process((wav, sr), args.clip), process((short, short_sr), "speech_5s"))
    for label, s, e in results:
        print(f"concurrent: {label} done {e - a:.2f} s after both were submitted", flush=True)

    renderer.clear_mm_cache()
    a = time.perf_counter()
    renderer._process_multimodal(audio_prompt(renderer, 1), {"audio": [(wav, sr)]}, None, None)  # on the loop
    await asyncio.sleep(0.5)
    print(
        f"for comparison, {args.clip} processed on the event loop itself: {summary(stream.gaps_ms(a, time.perf_counter()))}",
        flush=True,
    )
    stream.running = False
    await task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument("--clips", default=os.environ.get("MIMO_AUDIO_CLIPS", "/data/ttuser-agentx/mm/audio/clips"))
    parser.add_argument("--clip", default="speech_120s")
    parser.add_argument("--interval", type=float, default=10.0, help="ms between the stream's chunks")
    args = parser.parse_args()
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} load average {os.getloadavg()}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
