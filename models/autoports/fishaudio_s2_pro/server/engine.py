"""S2Engine: one worker thread owns the mesh, the TT generator and the codec; HTTP handlers enqueue jobs and
consume a per-job event queue (header / segment / final / error). One request at a time (single device owner),
like fish-speech's llama queue. Cancellation is checked per generated frame."""
from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from models.autoports.fishaudio_s2_pro.config import SAMPLE_RATE
from models.autoports.fishaudio_s2_pro.server.schemas import ServeTTSRequest


@dataclass
class Job:
    req: ServeTTSRequest
    out: "queue.Queue[tuple]" = field(default_factory=queue.Queue)
    cancel: threading.Event = field(default_factory=threading.Event)
    submitted: float = field(default_factory=time.time)


class Cancelled(Exception):
    pass


class _StreamDecoder:
    """Streams audio without stalling the AR loop. The generation thread only records a snapshot of the frames at
    each chunk boundary; this thread re-decodes the codec prefix (exact: the codec is causal, so a longer prefix
    reproduces the samples already emitted) and puts ("segment", new_samples) on the job queue, in order. When the
    codec falls behind, only the newest snapshot is decoded (coalescing is exact for the same reason). Torch's conv
    kernels release the GIL, so the CPU codec overlaps with device generation instead of serialising with it."""

    def __init__(self, codec, out: "queue.Queue[tuple]", first_chunk_frames: int, chunk_frames: int):
        self.codec, self.out = codec, out
        self.chunk = max(1, int(chunk_frames))
        self.next_emit = max(1, int(first_chunk_frames))
        self.cv = threading.Condition()
        self.pending: Optional[List[List[int]]] = None  # newest snapshot not yet taken by the worker
        self.taken_len = 0  # frames in the snapshot the worker is decoding / has decoded
        self.done = False
        self.stop = False
        self.emitted = 0  # samples already put on the queue
        self.wav = np.zeros(0, dtype=np.float32)  # last decoded prefix
        self.error: Optional[BaseException] = None
        self.decodes = 0
        self.codec_s = 0.0
        self.thread = threading.Thread(target=self._run, name="s2-codec", daemon=True)

    def start(self):
        self.thread.start()

    def on_frame(self, frames: List[List[int]]):
        """Generation thread: enqueue a snapshot when a chunk boundary is reached. Never blocks on the codec."""
        if len(frames) >= self.next_emit:
            with self.cv:
                self.pending = list(frames)
                self.cv.notify()
            self.next_emit = len(frames) + self.chunk

    def _decode(self, snap: List[List[int]]):
        t0 = time.time()
        wav = self.codec.decode(torch.tensor(snap, dtype=torch.int64).T[1:])
        self.codec_s += time.time() - t0
        self.decodes += 1
        self.wav = wav
        if len(wav) > self.emitted and not self.stop:
            self.out.put(("segment", wav[self.emitted :]))
            self.emitted = len(wav)

    def _run(self):
        try:
            while True:
                with self.cv:
                    while self.pending is None and not self.done and not self.stop:
                        self.cv.wait()
                    if self.stop:
                        return
                    snap, self.pending = self.pending, None
                    if snap is None:  # done, nothing left
                        return
                    self.taken_len = len(snap)
                self._decode(snap)
        except BaseException as e:  # noqa - re-raised on the generation thread by finish()
            self.error = e

    def finish(self, frames: List[List[int]]) -> np.ndarray:
        """Generation thread, after the last frame: make sure the FULL clip is decoded exactly once more than needed,
        wait for the worker (all its segments are on the queue when this returns) and return the full-clip waveform."""
        n = len(frames)
        with self.cv:
            already = (self.pending is not None and len(self.pending) == n) or (
                self.pending is None and self.taken_len == n
            )
            if n and not already:
                self.pending = list(frames)
            self.done = True
            self.cv.notify()
        self.thread.join()
        if self.error is not None:
            raise self.error
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        return self.wav

    def abort(self):
        with self.cv:
            self.stop = True
            self.cv.notify()
        self.thread.join()


class S2Engine:
    def __init__(self, log=print):
        self.log = log
        self.mesh = None
        self.gen = None
        self.codec = None
        self.refs = None
        self.ready = threading.Event()
        self.dead: Optional[str] = None
        self.busy = False
        self.served = 0
        self.started = time.time()
        self.q: "queue.Queue[Job]" = queue.Queue(maxsize=int(os.environ.get("FISH_S2_MAX_QUEUE", 8)))
        self.chunk_frames = int(os.environ.get("FISH_S2_STREAM_CHUNK_FRAMES", 32))
        self.first_chunk_frames = int(os.environ.get("FISH_S2_FIRST_CHUNK_FRAMES", 8))
        self.codec_device = os.environ.get("FISH_S2_CODEC_DEVICE", "cpu")
        self.max_seq_len = int(os.environ.get("FISH_S2_MAX_SEQ_LEN", 8192))
        self.max_text_length = int(os.environ.get("FISH_S2_MAX_TEXT_LENGTH", 0))
        self.impl = {
            "slow": "ttnn/tt_transformers",
            "fast": "ttnn" if os.environ.get("FISH_S2_FAST_DEVICE", "cpu") == "tt" else "torch-cpu",
            "codec": "ttnn" if self.codec_device == "tt" else "torch-cpu",
        }
        self._thread = threading.Thread(target=self._run, name="s2-worker", daemon=True)

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        self._thread.start()

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        while not self.ready.wait(1.0):
            if self.dead:
                raise RuntimeError(f"engine failed to start: {self.dead}")
            if timeout is not None and time.time() - self.started > timeout:
                return False
        return True

    def _load(self):
        from models.autoports.fishaudio_s2_pro.server.references import ReferenceStore
        from models.autoports.fishaudio_s2_pro.tt.codec.codec_decoder import CPUCodec
        from models.autoports.fishaudio_s2_pro.tt.device import open_mesh
        from models.autoports.fishaudio_s2_pro.tt.generator import S2Generator

        self.log("Opening mesh device ...")
        self.handle = open_mesh()
        self.mesh = self.handle.mesh
        self.log(f"mesh {self.handle.shape} open ({self.handle.num_devices} devices)")
        self.log("Loading weights ...")
        self.gen = S2Generator(self.mesh, max_seq_len=self.max_seq_len, log=self.log)
        self.log("Loading codec ...")
        self.codec = CPUCodec(self.gen.snapshot)
        cache_root = Path(os.environ.get("TT_DIT_CACHE_DIR", Path.home() / ".cache" / "fish_s2_pro"))
        self.refs = ReferenceStore(
            Path(os.environ.get("FISH_S2_REFERENCES_DIR", cache_root / "fish_s2_pro" / "references")), self.codec.encode
        )
        if os.environ.get("FISH_S2_WARMUP", "1") not in ("0", "false", "no"):
            self.log("Warming up ... (one short synthesis compiles the device programs)")
            t0 = time.time()
            codes, st = self.gen.generate("Hello world.", greedy=True, max_new_tokens=24)
            self.codec.decode(codes)
            self.log(f"warmup done in {time.time() - t0:.1f}s ({st.frames} frames, {st.frames_per_s:.2f} frames/s)")

    def _run(self):
        try:
            self._load()
        except Exception as e:  # noqa
            self.dead = f"{e}\n{traceback.format_exc()}"
            self.log(f"ENGINE LOAD FAILED: {self.dead}")
            return
        self.ready.set()
        self.log("engine ready")
        while True:
            job = self.q.get()
            self.busy = True
            try:
                self._serve(job)
                self.served += 1
            except Cancelled:
                job.out.put(("cancelled", None))
            except Exception as e:  # noqa
                self.log(f"job failed: {e}\n{traceback.format_exc()}")
                job.out.put(("error", str(e)))
                if "TT_THROW" in str(e) or "TIMEOUT" in str(e):
                    self.dead = str(e)
                    self.ready.clear()
                    self.log("device fault: engine marked dead (health -> 503)")
                    return
            finally:
                self.busy = False

    # ------------------------------------------------------------------ requests
    def submit(self, req: ServeTTSRequest) -> Job:
        if not self.ready.is_set():
            raise RuntimeError(self.dead or "engine not ready")
        if self.max_text_length and len(req.text) > self.max_text_length:
            raise ValueError(f"text too long ({len(req.text)} > {self.max_text_length})")
        job = Job(req)
        self.q.put_nowait(job)  # raises queue.Full -> 429 in the app
        return job

    def _references(self, req: ServeTTSRequest):
        if req.reference_id:
            codes, text = self.refs.get(req.reference_id)
            return [codes], [text]
        if req.references:
            pairs = [self.refs.inline(r.audio, r.text, cache=req.use_memory_cache == "on") for r in req.references]
            return [c for c, _ in pairs], [t for _, t in pairs]
        return None, None

    def _serve(self, job: Job):
        req = job.req
        ref_codes, ref_texts = self._references(req)
        frames: List[List[int]] = []
        t0 = time.time()
        streamer: Optional[_StreamDecoder] = None
        if req.streaming:
            job.out.put(("header", None))
            streamer = _StreamDecoder(self.codec, job.out, self.first_chunk_frames, self.chunk_frames)
            streamer.start()

        def on_frame(i, frame):
            if job.cancel.is_set():
                raise Cancelled()
            frames.append(frame)
            if streamer is not None:
                streamer.on_frame(frames)  # only records a snapshot: the codec runs on its own thread

        try:
            codes, st = self.gen.generate(
                req.text,
                ref_codes=ref_codes,
                ref_texts=ref_texts,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                greedy=req.greedy,
                seed=req.seed,
                on_frame=on_frame,
            )
            if job.cancel.is_set():
                raise Cancelled()
        except BaseException:
            if streamer is not None:
                streamer.abort()
            raise
        t_codec = time.time()
        if streamer is not None:
            wav = streamer.finish(frames)  # emits the remaining tail; returns the full-clip decode
        else:
            wav = self.codec.decode(codes) if codes.shape[1] else np.zeros(0, dtype=np.float32)
        stats = {
            "frames": st.frames,
            "prompt_len": st.prompt_len,
            "stopped_on_im_end": st.stopped_on_im_end,
            "seconds": time.time() - t0,
            "prefill_s": st.prefill_s,
            "decode_s": st.decode_s,
            "audio_s": len(wav) / SAMPLE_RATE,
            "frames_per_s": st.frames_per_s,
            "codec_tail_s": time.time() - t_codec,
        }
        if streamer is not None:
            stats["stream_decodes"] = streamer.decodes
            stats["stream_codec_s"] = round(streamer.codec_s, 3)
        self.log(f"tts done: {stats}")
        job.out.put(("final", (wav, stats)))

    # ------------------------------------------------------------------ status
    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.ready.is_set() else ("degraded" if self.dead else "starting"),
            "model": os.environ.get("HF_MODEL", "fishaudio/s2-pro"),
            "mesh": "x".join(map(str, self.handle.shape)) if self.mesh is not None else "",
            "device_name": getattr(getattr(self.gen, "args", None), "device_name", ""),
            "codec_device": self.codec_device,
            "busy": self.busy,
            "queue_depth": self.q.qsize(),
            "uptime_s": time.time() - self.started,
            "requests_served": self.served,
            "impl": self.impl,
            "error": self.dead,
        }
