# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""lmms-eval model `mimo_openrouter`: OpenRouter-shaped chat requests to any OpenAI-compatible endpoint.

Why not lmms-eval's own `openai` backends: they send video as a file:// URL plus vLLM's `media_io_kwargs` (or as
extracted frames), and audio only from local mp3/wav paths (an HF audio column becomes a "video" part). OpenRouter
accepts neither, so the TT endpoint and a reference endpoint would see different requests. This model sends exactly
the documented OpenRouter forms, the same bytes to every endpoint:

  image  image_url, data:image/png;base64 (or jpeg with image_format=jpeg)
  audio  input_audio {data: base64 WAV (16-bit mono, resampled to audio_rate), format: "wav"}
  video  video_url, data:video/mp4;base64 of the file with its audio track (re-encoded once, same resolution, audio
         kept as AAC, when above video_max_mb; the transcode is cached next to the journal so every endpoint gets the
         same bytes); an http(s) video is sent by URL, as is

Media come first in the user turn, then the task's text. Reasoning is switched per `thinking`: "off" / "on" / "default"
(send nothing). On target "vllm" the switch is chat_template_kwargs.enable_thinking; on "openrouter" it is
OpenRouter's `reasoning.enabled`, and `provider` (comma-separated slugs) pins providers with no fallbacks.

Resume: every successful response is appended (fsync) to `journal` (JSONL), keyed by task, doc and request content
(not by endpoint); rerunning the same command answers journaled requests from it. Failures are not journaled, so a
rerun retries them. 4xx answers other than 408 / 429 are not retried (the request itself is bad), except OpenRouter's
relayed provider errors ("Provider returned error": transient under load, accepted on a resend); 5xx, 408, 429 and
connection errors are retried `max_retries` times with backoff. A request that fails for good returns
"[MIMO_REQUEST_FAILED <status>] ..." as its text and is counted in `<journal>.stats.json`. That text is scored wrong
without going through answer extraction (mimo_lmms/failures.py; the plugin's tasks check for it, compare_runs.py
does for lmms-eval's own tasks).

The journal is one endpoint's: every record carries the endpoint it came from (target and base URL), and a journal
is resumed only from the records of the endpoint being run (records of an older journal without the field are
taken as they are; `journal_any_endpoint=True` takes all). A dry run keeps its "A" answers in
`<journal>.dryrun-journal.jsonl`, never in the journal, so a real run into the same directory resumes nothing from
it. `<journal>.keys.json` lists the request keys of the last run (answered from the journal or sent), which is what
compare_runs.py counts: a journal also holds the answers of earlier runs with other settings (temperature,
thinking, token limit), under other keys.

Raw log: every request that reaches the network is also appended (fsync) to `<journal>.raw.jsonl`, one record per
request after its last attempt, kept forever (never rewritten, also for failures): the request body with each media
payload replaced by a reference (dataset doc id, source path when there is one, MIME type, size and sha256 of the data
URL), the full response body (content, reasoning, usage incl. OpenRouter's cost, finish_reason, provider), the
response headers, every attempt's status / latency / error, and the wall-clock latency. Retries back off
exponentially with full jitter (and honour Retry-After); `<journal>.stats.json` also reports the effective
concurrency (time-averaged in-flight HTTP requests) and 429 counts. Each invocation's stats are appended to
`<journal>.stats.jsonl` too.

`dry_run=True` sends nothing: an in-process httpx transport captures each request, checks it against the OpenRouter
request shape (content-part types, MIME types, audio formats, no vLLM-only fields) and answers "A". The captured
requests go to `<journal>.dryrun.jsonl` (media replaced by their sizes, or kept with MM_DRYRUN_FULL=1).
"""

import base64
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import GenerationResult, TokenCounts
from lmms_eval.api.model import lmms

from ..failures import failed_text

# Retry jitter only spreads resends in time (nothing is derived from it); the OS source, so no seeded generator
# stands where a security scanner, or a later use, would take it for one.
_JITTER = random.SystemRandom()

OPENROUTER_PART_TYPES = {"text", "image_url", "input_audio", "video_url", "input_video"}
OPENROUTER_IMAGE_MIME = {"image/png", "image/jpeg", "image/webp", "image/gif"}
OPENROUTER_AUDIO_FORMATS = {"wav", "mp3", "flac", "m4a", "ogg", "aiff", "aac", "pcm16", "pcm24"}
OPENROUTER_VIDEO_MIME = {"video/mp4", "video/mpeg", "video/mov", "video/quicktime", "video/webm"}
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".wmv", ".mpeg", ".mpg", ".m4v")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")
NO_RETRY_4XX = lambda status: 400 <= status < 500 and status not in (408, 429)  # noqa: E731


def _truthy(value):
    return str(value).lower() in ("1", "true", "yes", "on")


def _wav_bytes(samples, rate):
    """float [-1, 1] mono samples -> 16-bit PCM WAV bytes."""
    import wave

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm)
    return buf.getvalue()


def _resample(samples, src_rate, dst_rate):
    """Polyphase resampling (scipy; the server resamples to 24 kHz again, so this only sizes the payload)."""
    src_rate, dst_rate = int(src_rate), int(dst_rate)
    if src_rate == dst_rate:
        return samples
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(src_rate, dst_rate)
    return resample_poly(samples, dst_rate // g, src_rate // g).astype(np.float32)


def _audio_samples(visual):
    """An HF audio value (dict with array / bytes / path, or a torchcodec AudioDecoder) or a file path ->
    (mono float32 samples, sample rate)."""
    if hasattr(visual, "get_all_samples"):  # datasets >= 4 decodes audio to torchcodec AudioDecoder
        s = visual.get_all_samples()
        data = s.data.numpy()
        return (data.mean(axis=0) if data.ndim == 2 else data).astype(np.float32), int(s.sample_rate)
    if isinstance(visual, dict) and visual.get("array") is not None:
        arr = np.asarray(visual["array"], dtype=np.float32)
        return (arr.mean(axis=0) if arr.ndim == 2 else arr), int(visual["sampling_rate"])
    import soundfile as sf

    source = io.BytesIO(visual["bytes"]) if isinstance(visual, dict) and visual.get("bytes") else visual
    if isinstance(visual, dict) and not visual.get("bytes"):
        source = visual["path"]
    data, rate = sf.read(source, dtype="float32", always_2d=True)
    return data.mean(axis=1), int(rate)


def _validate_openrouter(req, target):
    """Problems that would make this request differ from what OpenRouter sends (empty list = conforms)."""
    problems = []
    allowed_top = {
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "seed",
        "stop",
        "stream",
        "reasoning",
        "provider",
        "usage",
    }
    if target == "vllm":
        allowed_top.add("chat_template_kwargs")  # our endpoint's reasoning switch (OpenRouter maps its own to it)
    for key in req:
        if key not in allowed_top:
            problems.append(f"top-level field {key!r} is not an OpenRouter field")
    for msg in req.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            continue
        for part in content:
            t = part.get("type")
            if t not in OPENROUTER_PART_TYPES:
                problems.append(f"content part type {t!r}")
            elif t == "image_url":
                url = part["image_url"]["url"]
                if url.startswith("data:") and url[5:].split(";", 1)[0] not in OPENROUTER_IMAGE_MIME:
                    problems.append(f"image MIME {url[5:40]!r}")
                elif not url.startswith(("data:", "http://", "https://")):
                    problems.append(f"image URL scheme {url[:20]!r}")
            elif t == "input_audio":
                fmt = part["input_audio"].get("format")
                if fmt not in OPENROUTER_AUDIO_FORMATS:
                    problems.append(f"audio format {fmt!r}")
                if str(part["input_audio"].get("data", "")).startswith("data:"):
                    problems.append("input_audio.data must be bare base64")
            elif t in ("video_url", "input_video"):
                url = part["video_url"]["url"]
                if url.startswith("data:") and url[5:].split(";", 1)[0] not in OPENROUTER_VIDEO_MIME:
                    problems.append(f"video MIME {url[5:40]!r}")
                elif not url.startswith(("data:", "http://", "https://")):
                    problems.append(f"video URL scheme {url[:20]!r} (OpenRouter takes http(s) or data: only)")
    return problems


def _redact(req):
    """The request with media payloads replaced by their sizes (for the dry-run log)."""
    out = json.loads(json.dumps(req))
    for msg in out.get("messages", []):
        if isinstance(msg.get("content"), list):
            for part in msg["content"]:
                for key in ("image_url", "video_url"):
                    if key in part and part[key]["url"].startswith("data:"):
                        url = part[key]["url"]
                        part[key]["url"] = f"{url.split(',', 1)[0]},<{len(url)} chars>"
                if "input_audio" in part:
                    part["input_audio"]["data"] = f"<{len(part['input_audio']['data'])} chars>"
    return out


def _media_refs(req, sources):
    """The request with each media payload replaced by a reference, and the list of references (sha256 of the URL /
    base64 string, size, MIME type, the visual's source path when it came from a file). `sources` lists the media
    parts' sources in order (None for in-memory dataset values: those are identified by the record's doc id)."""
    out = json.loads(json.dumps(req))
    refs = []
    for msg in out.get("messages", []):
        if not isinstance(msg.get("content"), list):
            continue
        for part in msg["content"]:
            payload = None
            if part.get("type") in ("image_url", "video_url", "input_video"):
                key = "image_url" if part["type"] == "image_url" else "video_url"
                url = part[key]["url"]
                if url.startswith("data:"):
                    payload, mime = url, url[5:].split(";", 1)[0]
                    holder, field = part[key], "url"
            elif part.get("type") == "input_audio":
                payload, mime = part["input_audio"]["data"], "audio/" + str(part["input_audio"].get("format"))
                holder, field = part["input_audio"], "data"
            if payload is None:
                continue
            i = len(refs)
            ref = {
                "index": i,
                "type": part["type"],
                "mime": mime,
                "chars": len(payload),
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "source": sources[i] if i < len(sources) else None,
            }
            refs.append(ref)
            holder[field] = f"<media #{i} {mime} sha256:{ref['sha256'][:16]} {len(payload)} chars>"
    return out, refs


class OpenRouterChat(lmms):
    is_simple = True

    def __init__(
        self,
        model_version="XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        target="vllm",
        thinking="off",
        provider="",
        num_concurrent=16,
        timeout=3600,
        max_retries=3,
        max_new_tokens=None,
        temperature=None,
        top_p=None,
        journal=None,
        image_format="png",
        audio_rate=24000,
        video_max_mb=32,
        dry_run=False,
        retry_cap_s=120,
        raw_log=True,
        journal_any_endpoint=False,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            eval_logger.warning(f"mimo_openrouter: ignoring unknown model_args {sorted(kwargs)}")
        if target not in ("vllm", "openrouter"):
            raise ValueError("target must be vllm or openrouter")
        if thinking not in ("off", "on", "default"):
            raise ValueError("thinking must be off, on or default")
        self.model_version = model_version
        self.target = target
        self.thinking = thinking
        self.provider = [p for p in str(provider).replace("+", ",").split(",") if p]
        self.num_concurrent = int(num_concurrent)
        self.max_retries = int(max_retries)
        self.retry_cap_s = float(retry_cap_s)
        self.max_new_tokens = int(max_new_tokens) if max_new_tokens not in (None, "", "None") else None
        self.temperature = float(temperature) if temperature not in (None, "", "None") else None
        self.top_p = float(top_p) if top_p not in (None, "", "None") else None
        self.image_format = image_format.upper()
        self.audio_rate = int(audio_rate)
        self.video_max_bytes = int(float(video_max_mb) * 1024 * 1024)
        self.dry_run = _truthy(dry_run)
        # MM_DRYRUN_FULL=1 keeps the media in the dry-run log (for conformance_offline.py --requests token counts).
        self.dry_run_full = _truthy(os.environ.get("MM_DRYRUN_FULL", "0"))
        self.journal = journal or os.path.join(os.getcwd(), "mimo_openrouter_journal.jsonl")
        os.makedirs(os.path.dirname(os.path.abspath(self.journal)), exist_ok=True)
        self.video_cache = os.path.splitext(self.journal)[0] + "_video_cache"
        self.raw_log = os.path.splitext(self.journal)[0] + ".raw.jsonl" if _truthy(raw_log) else None
        # The responses a rerun resumes from (module docstring): the journal, of this endpoint; a dry run's own file.
        self.endpoint = "dry-run" if self.dry_run else f"{target}|{(base_url or '').rstrip('/')}"
        self.journal_any_endpoint = _truthy(journal_any_endpoint)
        self.responses = os.path.splitext(self.journal)[0] + ".dryrun-journal.jsonl" if self.dry_run else self.journal
        self._run_keys = set()
        self._lock = threading.Lock()
        self._transcode_locks = {}  # one lock per cached transcode: Video-MME asks 3 questions per video concurrently
        self._done = self._load_journal()
        self.stats = {"requests": 0, "from_journal": 0, "ok": 0, "failed": 0, "retried": 0, "status": {}}
        self.stats.update(input_tokens=0, output_tokens=0, reasoning_tokens=0, finish_length=0, empty_content=0)
        self.stats.update(cost_usd=0.0, attempts=0, attempt_status={}, http_busy_s=0.0, max_in_flight=0)
        self._latencies = []
        self._in_flight = 0

        import httpx
        from openai import OpenAI

        http_client = None
        if self.dry_run:
            http_client = httpx.Client(transport=httpx.MockTransport(self._dry_run_handler))
            self._dry_log = open(os.path.splitext(self.journal)[0] + ".dryrun.jsonl", "a")
        api_key = os.environ.get(api_key_env) or "none"
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or "http://dry-run.invalid/v1",
            timeout=float(timeout),
            max_retries=0,
            http_client=http_client,
        )

    # ------------------------------------------------------------------------------------------------ journal

    def _load_journal(self):
        done, other = {}, 0
        if os.path.exists(self.responses):
            with open(self.responses) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue  # a torn last line from a killed run
                    if not self.journal_any_endpoint and rec.get("endpoint", self.endpoint) != self.endpoint:
                        other += 1  # another endpoint's answer to the same request
                        continue
                    done[rec["key"]] = rec
            eval_logger.info(f"mimo_openrouter: {len(done)} responses in journal {self.responses}")
            if other:
                eval_logger.warning(
                    f"mimo_openrouter: {other} journaled responses of another endpoint than {self.endpoint} are not "
                    "resumed from (journal_any_endpoint=True takes them)"
                )
        return done

    def _append_journal(self, rec, path=None):
        with self._lock:
            with open(path or self.responses, "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())

    # ------------------------------------------------------------------------------------------------ media

    def _image_part(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        if self.image_format == "JPEG" or image.mode not in ("RGB", "RGBA", "L", "P"):
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, self.image_format)
        mime = "image/jpeg" if self.image_format == "JPEG" else f"image/{self.image_format.lower()}"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()},
        }

    def _audio_part(self, visual):
        samples, rate = _audio_samples(visual)
        samples = _resample(samples, rate, self.audio_rate)
        data = base64.b64encode(_wav_bytes(samples, self.audio_rate)).decode()
        return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}

    def _video_bytes(self, path):
        size = os.path.getsize(path)
        ext = os.path.splitext(path)[1].lower()
        if size <= self.video_max_bytes and ext in (".mp4", ".webm", ".mov", ".mpeg", ".mpg"):
            with open(path, "rb") as f:
                return f.read(), {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}.get(
                    ext, "video/mpeg"
                )
        os.makedirs(self.video_cache, exist_ok=True)
        key = f"{os.path.abspath(path)}:{size}:{self.video_max_bytes}:with-audio"  # older caches dropped the audio
        key = hashlib.sha256(key.encode()).hexdigest()[:24]
        out = os.path.join(self.video_cache, key + ".mp4")
        with self._lock:
            transcode_lock = self._transcode_locks.setdefault(out, threading.Lock())
        with transcode_lock:
            if not os.path.exists(out):
                self._transcode(path, size, out)
        with open(out, "rb") as f:
            return f.read(), "video/mp4"

    def _transcode(self, path, size, out):
        """Re-encode `path` (same resolution and frame rate, H.264, audio kept as AAC) into the cache file `out`."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            try:  # the static binary of the imageio-ffmpeg wheel
                import imageio_ffmpeg

                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:  # noqa: BLE001 - no ffmpeg anywhere: raise below
                ffmpeg = None
        if ffmpeg is None:
            raise RuntimeError(f"{path} is {size} bytes (> video_max_mb) or not mp4/webm/mov, and ffmpeg is missing")
        # A per-process / per-thread temporary name: other processes (another endpoint's run sharing the cache)
        # may encode the same video at the same time; the last os.replace wins with identical bytes.
        tmp = f"{out}.{os.getpid()}.{threading.get_ident()}.tmp.mp4"
        # No shell: every argument reaches ffmpeg as it is. The input is an absolute path of an existing file (never
        # read as an option or a protocol URL), from the benchmark's local dataset.
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"video to transcode is not a file: {path}")
        # Same resolution and frame rate, H.264 at a quality setting, the audio track kept: OpenRouter's providers
        # (Xiaomi) feed a video's audio to the model, and some benchmarks ask what was said.
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            path,
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
        ]
        cmd += ["-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(tmp, out)

    def _video_part(self, path):
        if path.startswith(("http://", "https://", "file://")):
            return {"type": "video_url", "video_url": {"url": path}}
        payload, mime = self._video_bytes(path)
        return {"type": "video_url", "video_url": {"url": f"data:{mime};base64," + base64.b64encode(payload).decode()}}

    def _media_parts(self, visuals, sources=None):
        parts = []
        for visual in visuals:
            if visual is None:
                continue
            if sources is not None:
                sources.append(
                    visual if isinstance(visual, str) else (visual.get("path") if isinstance(visual, dict) else None)
                )
            if isinstance(visual, Image.Image):
                parts.append(self._image_part(visual))
            elif isinstance(visual, str):
                low = visual.lower()
                if low.endswith(VIDEO_EXTS) or (low.startswith("http") and not low.endswith(IMAGE_EXTS)):
                    parts.append(self._video_part(visual))
                elif low.endswith(AUDIO_EXTS):
                    parts.append(self._audio_part(visual))
                elif low.endswith(IMAGE_EXTS):
                    parts.append(self._image_part(visual))
                else:
                    raise ValueError(f"cannot tell the modality of visual {visual[:120]!r}")
            elif hasattr(visual, "get_all_samples") or (
                isinstance(visual, dict) and ("array" in visual or "bytes" in visual)
            ):
                parts.append(self._audio_part(visual))
            else:
                raise ValueError(f"unsupported visual type {type(visual)}")
        return parts

    # ------------------------------------------------------------------------------------------------ requests

    def _build(self, context, gen_kwargs, visuals, sources=None):
        content = self._media_parts(visuals, sources) + [{"type": "text", "text": context}]
        req = {"model": self.model_version, "messages": [{"role": "user", "content": content}]}
        max_tokens = self.max_new_tokens or gen_kwargs.get("max_new_tokens") or gen_kwargs.get("max_gen_toks")
        if max_tokens:
            req["max_tokens"] = int(max_tokens)
        temperature = self.temperature if self.temperature is not None else gen_kwargs.get("temperature", 0)
        req["temperature"] = float(temperature or 0)
        top_p = self.top_p if self.top_p is not None else gen_kwargs.get("top_p")
        if top_p is not None and req["temperature"] > 0:
            req["top_p"] = float(top_p)
        if self.thinking != "default":
            on = self.thinking == "on"
            if self.target == "openrouter":
                req["reasoning"] = {"enabled": on}
            else:
                req["chat_template_kwargs"] = {"enable_thinking": on}
        if self.target == "openrouter" and self.provider:
            req["provider"] = {"order": self.provider, "allow_fallbacks": False}
        if self.target == "openrouter":
            req["usage"] = {"include": True}  # OpenRouter's cost accounting in the response usage
        return req

    def _key(self, task, split, doc_id, req):
        stable = dict(req)
        stable.pop("provider", None)
        stable.pop("usage", None)
        digest = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
        return f"{task}|{split}|{doc_id}|{digest[:32]}"

    def _dry_run_handler(self, request):
        import httpx

        req = json.loads(request.content)
        problems = _validate_openrouter(req, self.target)
        with self._lock:
            logged = req if self.dry_run_full else _redact(req)
            self._dry_log.write(json.dumps({"problems": problems, "request": logged}) + "\n")
            self._dry_log.flush()
        if problems:
            return httpx.Response(400, json={"error": {"message": "; ".join(problems), "type": "dry_run_shape"}})
        body = {
            "id": "dry-run",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "A"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(request.content) // 4, "completion_tokens": 1, "total_tokens": 0},
        }
        return httpx.Response(200, json=body)

    def _backoff(self, attempt, retry_after=None):
        """Full-jitter exponential backoff, at least the server's Retry-After."""
        wait = _JITTER.uniform(0, min(self.retry_cap_s, 5 * 2**attempt))
        try:
            wait = max(wait, min(self.retry_cap_s, float(retry_after)))
        except (TypeError, ValueError):
            pass
        time.sleep(wait)

    def _attempt_done(self, t0, status):
        with self._lock:
            self._in_flight -= 1
            self.stats["http_busy_s"] += time.time() - t0
            self.stats["attempts"] += 1
            self.stats["attempt_status"][str(status)] = self.stats["attempt_status"].get(str(status), 0) + 1

    def _provider_error(self, body):
        """OpenRouter's relay of an upstream provider's 4xx ("Provider returned error"). Under load the MiMo providers
        answer "Invalid request parameters" to requests they accept on a resend, so these are retried like a 5xx."""
        if self.target != "openrouter" or not isinstance(body, dict):
            return False
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        meta = err.get("metadata") or {}
        if "size limit" in str(meta.get("raw", "")):  # e.g. Xiaomi: "exceeded maximum size limit for video url data"
            return False
        return err.get("message") == "Provider returned error" or bool(meta.get("provider_name"))

    def _send(self, req):
        """dict(text, usage, finish, status, body, headers, attempts). Retries 5xx / 408 / 429 / transport errors and
        200s without a choice."""
        import openai

        last = "unknown error"
        status = None
        body, headers, attempts = None, {}, []
        # The SDK's typed arguments; everything else (reasoning switch, provider pinning) goes in the body as is.
        standard = {k: v for k, v in req.items() if k in ("model", "messages", "max_tokens", "temperature", "top_p")}
        extra = {k: v for k, v in req.items() if k not in standard}
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            with self._lock:
                self._in_flight += 1
                self.stats["max_in_flight"] = max(self.stats["max_in_flight"], self._in_flight)
            retry_after = None
            try:
                raw = self.client.chat.completions.with_raw_response.create(**standard, extra_body=extra or None)
                status = raw.status_code
                headers = {k: v for k, v in raw.headers.items() if k.lower() != "set-cookie"}
                try:
                    body = raw.http_response.json()
                except ValueError:
                    body = {"_text": raw.http_response.text[:20000]}
                choices = body.get("choices") or []
                attempts.append({"status": status, "latency_s": round(time.time() - t0, 3)})
                self._attempt_done(t0, status)
                if choices:
                    message = choices[0].get("message") or {}
                    usage = body.get("usage") or {}
                    text = message.get("content") or ""
                    return dict(
                        text=text,
                        usage=usage,
                        finish=choices[0].get("finish_reason"),
                        status=200,
                        body=body,
                        headers=headers,
                        attempts=attempts,
                    )
                status = "no_choices"
                last = f"200 without choices: {json.dumps(body)[:300]}"
                attempts[-1]["error"] = last
            except openai.APIStatusError as exc:
                status = exc.status_code
                last = f"{status} {str(exc)[:300]}"
                headers = {k: v for k, v in exc.response.headers.items() if k.lower() != "set-cookie"}
                body = exc.body if isinstance(exc.body, (dict, list)) else {"_text": str(exc.body)[:20000]}
                retry_after = exc.response.headers.get("retry-after")
                attempts.append(
                    {"status": status, "latency_s": round(time.time() - t0, 3), "error": last, "headers": headers}
                )
                self._attempt_done(t0, status)
                if NO_RETRY_4XX(status) and not self._provider_error(body):
                    break
            except (openai.APIConnectionError, openai.APITimeoutError) as exc:
                status = "conn"
                last = f"{type(exc).__name__}: {str(exc)[:300]}"
                attempts.append({"status": status, "latency_s": round(time.time() - t0, 3), "error": last})
                self._attempt_done(t0, status)
            if attempt < self.max_retries:
                with self._lock:
                    self.stats["retried"] += 1
                self._backoff(attempt, retry_after)
        return dict(
            text=failed_text(status, last),
            usage={},
            finish=None,
            status=status,
            body=body,
            headers=headers,
            attempts=attempts,
        )

    def generate_until(self, requests):
        results = [None] * len(requests)
        todo = []
        for i, request in enumerate(requests):
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            visuals = doc_to_visual(self.task_dict[task][split][doc_id]) or []
            if not isinstance(visuals, (list, tuple)):
                visuals = [visuals]
            todo.append((i, context, dict(gen_kwargs or {}), visuals, doc_id, task, split))

        def run(item):
            i, context, gen_kwargs, visuals, doc_id, task, split = item
            sources = []
            req = self._build(context, gen_kwargs, visuals, sources)
            key = self._key(task, split, doc_id, req)
            with self._lock:
                self.stats["requests"] += 1
                self._run_keys.add(key)
                rec = self._done.get(key)
            if rec is not None:
                with self._lock:
                    self.stats["from_journal"] += 1
                return i, rec["text"], rec.get("usage") or {}
            t0 = time.time()
            out = self._send(req)
            text, usage, finish, status = out["text"], out["usage"], out["finish"], out["status"]
            latency = time.time() - t0
            if self.raw_log:
                logged, refs = _media_refs(req, sources)
                raw = {"key": key, "task": task, "split": split, "doc_id": doc_id, "status": status}
                raw.update(start=round(t0, 3), latency_s=round(latency, 3), attempts=out["attempts"])
                raw.update(request=logged, media=refs, response=out["body"], response_headers=out["headers"])
                self._append_journal(raw, self.raw_log)
            with self._lock:
                self.stats["status"][str(status)] = self.stats["status"].get(str(status), 0) + 1
                if status == 200:
                    self.stats["ok"] += 1
                    self._latencies.append(latency)
                    self.stats["input_tokens"] += usage.get("prompt_tokens") or 0
                    self.stats["output_tokens"] += usage.get("completion_tokens") or 0
                    details = usage.get("completion_tokens_details") or {}
                    self.stats["reasoning_tokens"] += details.get("reasoning_tokens") or 0
                    self.stats["finish_length"] += finish == "length"
                    self.stats["empty_content"] += not text.strip()
                    self.stats["cost_usd"] += float(usage.get("cost") or 0)
                else:
                    self.stats["failed"] += 1
            if status == 200:
                rec = {"key": key, "task": task, "doc_id": doc_id, "text": text, "usage": usage, "finish": finish}
                rec["latency_s"] = round(latency, 3)
                rec["endpoint"] = self.endpoint
                self._append_journal(rec)
                with self._lock:
                    self._done[key] = rec
            else:
                eval_logger.warning(f"mimo_openrouter: {task} doc {doc_id} failed: {text[:300]}")
            return i, text, usage

        wall0 = time.time()
        with ThreadPoolExecutor(max_workers=self.num_concurrent) as pool:
            futures = [pool.submit(run, item) for item in todo]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="mimo_openrouter"):
                i, text, usage = fut.result()
                counts = TokenCounts(
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                )
                results[i] = GenerationResult(text=text, token_counts=counts)
        self._write_stats(time.time() - wall0)
        return results

    def _write_stats(self, wall_s):
        lat = sorted(self._latencies)
        stats = dict(
            self.stats, wall_s=round(wall_s, 1), dry_run=self.dry_run, target=self.target, thinking=self.thinking
        )
        stats.update(provider=self.provider, num_concurrent=self.num_concurrent, finished_at=time.time())
        stats["effective_concurrency"] = round(self.stats["http_busy_s"] / wall_s, 2) if wall_s > 0 else None
        if lat:
            stats["latency_p50_s"] = round(lat[len(lat) // 2], 2)
            stats["latency_p95_s"] = round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2)
        path = os.path.splitext(self.journal)[0] + ".stats.json"
        with open(path, "w") as f:
            json.dump(stats, f, indent=1)
        with open(os.path.splitext(self.journal)[0] + ".keys.json", "w") as f:  # the last run's requests
            json.dump({"endpoint": self.endpoint, "dry_run": self.dry_run, "keys": sorted(self._run_keys)}, f)
        self._append_journal(stats, os.path.splitext(self.journal)[0] + ".stats.jsonl")
        eval_logger.info(f"mimo_openrouter stats: {stats}")

    def loglikelihood(self, requests):
        raise NotImplementedError("mimo_openrouter is a chat endpoint client: generate_until tasks only")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("mimo_openrouter: multi-round tasks are not supported")
