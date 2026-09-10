"""Music3Engine: one worker thread owns the chip and the generator; HTTP handlers enqueue jobs and poll/await them.
One request at a time (single device owner). Cancellation is checked per generated frame. Device faults mark the
engine dead (health -> 503) so the launcher/operator restarts the container."""
from __future__ import annotations

import os
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from models.autoports.minimaxai_minimax_music3.config import HF_REPO_ID, VOCODER_SAMPLE_RATE
from models.autoports.minimaxai_minimax_music3.server.schemas import SpeechRequest


class Cancelled(Exception):
    pass


@dataclass
class Job:
    req: SpeechRequest
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"
    frames_done: int = 0
    stage: str = "queued"
    submitted: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0
    result: Optional[tuple] = None  # (stereo float32 [2, N] @ 44.1k, stats dict)
    error: Optional[str] = None
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)

    def to_status(self) -> dict:
        el = (self.finished or time.time()) - (self.started or self.submitted)
        return {
            "id": self.id,
            "status": self.status,
            "frames_done": self.frames_done,
            "frames_max": self.req.frames,
            "stage": self.stage,
            "elapsed_s": round(el, 2),
            "stats": self.result[1] if self.result else None,
            "error": self.error,
            "audio_url": f"/v1/music/jobs/{self.id}/audio" if self.status == "done" else None,
        }


class Music3Engine:
    def __init__(self, log=print):
        self.log = log
        self.handle = self.gen = None
        self.ready = threading.Event()
        self.dead: Optional[str] = None
        self.busy = False
        self.served = 0
        self.started = time.time()
        self.q: "queue.Queue[Job]" = queue.Queue(maxsize=int(os.environ.get("MUSIC3_MAX_QUEUE", 8)))
        self.jobs: Dict[str, Job] = {}
        self.max_seq_len = int(os.environ.get("MUSIC3_MAX_SEQ_LEN", 16384))
        self.max_frames = int(os.environ.get("MUSIC3_MAX_FRAMES", 9000))
        self.warmup_frames = int(os.environ.get("MUSIC3_WARMUP_FRAMES", 200))
        self.impl: Dict[str, str] = {}
        self._thread = threading.Thread(target=self._run, name="music3-worker", daemon=True)

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
        import ttnn
        from models.autoports.minimaxai_minimax_music3.tt.device import open_mesh
        from models.autoports.minimaxai_minimax_music3.tt.generator import Music3Generator

        dts = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
        defaults = self._defaults()
        for comp in ("dit", "depth"):
            if defaults.get(f"{comp}_fidelity"):
                os.environ.setdefault(f"MUSIC3_{comp.upper()}_FIDELITY", defaults[f"{comp}_fidelity"])
        self.log(f"defaults: {defaults}")
        self.log("Opening mesh device ...")
        self.handle = open_mesh()
        self.log(f"mesh {self.handle.shape} open ({self.handle.num_devices} device)")
        self.gen = Music3Generator(
            self.handle.mesh,
            max_seq_len=self.max_seq_len,
            llm_dtype=dts[defaults["llm_dtype"]],
            depth_dtype=dts[defaults["depth_dtype"]],
            dit_dtype=dts[defaults["dit_dtype"]],
            log=self.log,
        )
        self.impl = self.gen.impl
        if os.environ.get("MUSIC3_WARMUP", "1") not in ("0", "false", "no") and self.warmup_frames > 0:
            self.log(f"Warming up ({self.warmup_frames} frames: compiles + captures the LLM/depth/DiT traces) ...")
            t0 = time.time()
            out = self.gen.generate(
                "Genre: acoustic pop. BPM: 96. Warm female vocal, fingerpicked guitar and soft piano.",
                "[verse]\nMorning light filtering through the pine\n[chorus]\nSoftly the world begins to breathe",
                max_frames=self.warmup_frames,
                seed=0,
            )
            self.log(f"warmup done in {time.time() - t0:.1f}s: {out['stats'].as_dict()}")

    @staticmethod
    def _defaults() -> dict:
        import json
        from pathlib import Path

        d = {
            "llm_dtype": os.environ.get("MUSIC3_LLM_DTYPE", "bfp8"),
            "depth_dtype": os.environ.get("MUSIC3_DEPTH_DTYPE", "bf16"),
            "dit_dtype": os.environ.get("MUSIC3_DIT_DTYPE", "bf16"),
            "dit_fidelity": os.environ.get("MUSIC3_DIT_FIDELITY"),
            "depth_fidelity": os.environ.get("MUSIC3_DEPTH_FIDELITY"),
        }
        p = Path(__file__).resolve().parents[1] / "tt" / "config_defaults.json"
        if p.exists():
            saved = json.load(open(p))
            for k in d:
                if k not in os.environ.get("MUSIC3_ENV_OVERRIDES", "") and f"MUSIC3_{k.upper()}" not in os.environ:
                    d[k] = saved.get(k, d[k])
        return d

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
            if job.cancel.is_set():
                job.status, job.finished = "cancelled", time.time()
                job.done.set()
                continue
            self.busy = True
            job.status, job.started, job.stage = "running", time.time(), "semantic"
            try:
                self._serve(job)
                job.status = "done"
                self.served += 1
            except Cancelled:
                job.status = "cancelled"
            except Exception as e:  # noqa
                self.log(f"job {job.id} failed: {e}\n{traceback.format_exc()}")
                job.status, job.error = "error", str(e)
                if "TT_THROW" in str(e) or "TIMEOUT" in str(e) or "device timeout" in str(e).lower():
                    self.dead = str(e)
                    self.ready.clear()
                    self.log("device fault: engine marked dead (health -> 503)")
                    job.finished = time.time()
                    job.done.set()
                    return
            finally:
                job.finished = time.time()
                job.done.set()
                self.busy = False

    # ------------------------------------------------------------------ requests
    def submit(self, req: SpeechRequest) -> Job:
        if not self.ready.is_set():
            raise RuntimeError(self.dead or "engine not ready")
        if req.frames > self.max_frames:
            raise ValueError(f"max_new_tokens {req.frames} exceeds this server's limit {self.max_frames}")
        job = Job(req)
        self.jobs[job.id] = job
        self.q.put_nowait(job)  # raises queue.Full -> 429
        self._gc_jobs()
        return job

    def _gc_jobs(self, keep: int = 64):
        done = [j for j in self.jobs.values() if j.done.is_set()]
        for j in sorted(done, key=lambda j: j.finished)[:-keep] if len(done) > keep else []:
            self.jobs.pop(j.id, None)

    def _serve(self, job: Job):
        req = job.req

        def on_frame(i, codes):
            if job.cancel.is_set():
                raise Cancelled()
            job.frames_done = i + 1

        t0 = time.time()
        out = self.gen.generate(
            req.instructions,
            req.input,
            max_frames=req.frames,
            seed=req.seed if req.seed is not None else int(time.time() * 1000) % (2**31),
            num_steps=req.num_inference_steps,
            on_frame=on_frame,
        )
        if job.cancel.is_set():
            raise Cancelled()
        audio = (
            out["audio"].squeeze(0).numpy().astype(np.float32) if "audio" in out else np.zeros((2, 0), dtype=np.float32)
        )
        st = out["stats"].as_dict()
        st.update(seconds=time.time() - t0, sample_rate_native=VOCODER_SAMPLE_RATE, seed=req.seed)
        job.stage = "done"
        self.log(
            f"job {job.id} done: frames {st['frames']} audio {st['audio_s']:.1f}s in {st['seconds']:.1f}s (RTF {st['rtf']:.2f})"
        )
        job.result = (audio, st)

    # ------------------------------------------------------------------ status
    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.ready.is_set() else ("degraded" if self.dead else "starting"),
            "model": os.environ.get("HF_MODEL_ID", HF_REPO_ID),
            "mesh": "x".join(map(str, self.handle.shape)) if self.handle is not None else "",
            "device_name": getattr(getattr(self.gen, "args", None), "device_name", ""),
            "busy": self.busy,
            "queue_depth": self.q.qsize(),
            "uptime_s": round(time.time() - self.started, 1),
            "requests_served": self.served,
            "impl": self.impl,
            "limits": {"max_frames": self.max_frames, "max_seq_len": self.max_seq_len, "max_prompt_tokens": 5000},
            "error": self.dead,
        }
