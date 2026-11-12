# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""OpenRouter-shaped multimodal chat requests: media fixtures and the API conformance case table.

Shared by test_api_conformance.py (online, against a running endpoint) and conformance_offline.py (vLLM's chat front
end and the upstream MiMo processor, no engine). Only the standard library, Pillow and the ffmpeg CLI are needed.

Request forms follow OpenRouter's chat-completions docs (2026-11):
  image  {"type": "image_url", "image_url": {"url": http(s) URL or data:image/{png,jpeg,webp,gif};base64,..., "detail"}}
  audio  {"type": "input_audio", "input_audio": {"data": <base64>, "format": wav|mp3|flac|m4a|ogg|aiff|aac|pcm16|pcm24}}
  video  {"type": "video_url", "video_url": {"url": URL or data:video/{mp4,mpeg,mov,webm};base64,...}}
         and the legacy {"type": "input_video", "video_url": {...}}

Each case states what an OpenRouter provider must do with it: "ok" (200 and a sane answer) or "4xx" (a client error
with a message). Anything else -- a 5xx, a dropped connection, a dead engine -- is a failure, since 5xx and mid-stream
errors count against an endpoint's uptime on OpenRouter while 400/413 do not.
"""

import base64
import dataclasses
import io
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable

IMAGE_URL = os.environ.get(
    "MM_IMAGE_URL",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
)
VIDEO_URL = os.environ.get(
    "MM_VIDEO_URL", "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"
)
SPEECH_URL = os.environ.get("MM_SPEECH_URL", "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
MISSING_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/no-such-file-mimo.png"

# OpenRouter input_audio formats and the ffmpeg arguments that produce each from the speech clip. pcm16 / pcm24 are
# headerless little-endian mono at 24 kHz (OpenAI's realtime convention, which OpenRouter's names follow).
AUDIO_FORMATS = {
    "wav": ["-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", "-f", "wav"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"],
    "flac": ["-c:a", "flac", "-f", "flac"],
    "m4a": ["-c:a", "aac", "-b:a", "64k", "-f", "ipod"],
    "ogg": ["-c:a", "libvorbis", "-f", "ogg"],
    "aiff": ["-c:a", "pcm_s16be", "-f", "aiff"],
    "aac": ["-c:a", "aac", "-b:a", "64k", "-f", "adts"],
    "pcm16": ["-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", "-f", "s16le"],
    "pcm24": ["-c:a", "pcm_s24le", "-ar", "24000", "-ac", "1", "-f", "s24le"],
}
AUDIO_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
    "aiff": "audio/aiff",
    "aac": "audio/aac",
}
VIDEO_FORMATS = {
    "mp4": (["-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4"], "video/mp4"),
    "webm": (["-c:v", "libvpx-vp9", "-b:v", "200k", "-f", "webm"], "video/webm"),
    "mov": (["-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mov"], "video/quicktime"),
    "mpeg": (["-c:v", "mpeg1video", "-q:v", "4", "-f", "mpeg"], "video/mpeg"),
}
IMAGE_MIME = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp", "gif": "image/gif"}


def data_uri(mime, payload):
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


def b64(payload):
    return base64.b64encode(payload).decode("ascii")


def _ffmpeg(args, input_bytes=None):
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found on PATH (needed to build the audio / video fixtures)")
    proc = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", "-y", *args], input=input_bytes, capture_output=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg {' '.join(args)} failed: {proc.stderr.decode(errors='replace')[-400:]}")
    return proc.stdout


class Media:
    """Deterministic test media. Shapes on white for images, a solid-colour clip for video, a real speech clip (MLK,
    "I have a dream") for audio, converted to every OpenRouter format. Downloads are cached in `cache_dir`."""

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir or os.environ.get("MM_ASSETS_DIR", Path.home() / ".cache" / "mimo_mm_evals"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memo = {}

    def _download(self, url):
        path = self.cache_dir / ("dl_" + url.rsplit("/", 1)[-1])
        if not path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "mimo-mm-evals/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp, tempfile.NamedTemporaryFile(
                dir=self.cache_dir, delete=False
            ) as tmp:
                tmp.write(resp.read())
            os.replace(tmp.name, path)
        return path.read_bytes()

    def image(self, fmt="png", size=(448, 448), shape="circle", color=(220, 20, 20)):
        key = ("image", fmt, size, shape, color)
        if key not in self._memo:
            from PIL import Image, ImageDraw

            img = Image.new("RGB", size, (255, 255, 255))
            draw = ImageDraw.Draw(img)
            w, h = size
            box = [w // 5, h // 5, 4 * w // 5, 4 * h // 5]
            if shape == "circle":
                draw.ellipse(box, fill=color)
            elif shape == "square":
                draw.rectangle(box, fill=color)
            else:
                draw.polygon([(w // 2, h // 5), (w // 5, 4 * h // 5), (4 * w // 5, 4 * h // 5)], fill=color)
            buf = io.BytesIO()
            img.save(buf, {"jpeg": "JPEG", "png": "PNG", "webp": "WEBP", "gif": "GIF", "bmp": "BMP"}[fmt])
            self._memo[key] = buf.getvalue()
        return self._memo[key]

    def animated_gif(self):
        """Three frames, each a coloured shape on white (red circle first). Shapes, not uniform frames: MiMo-V2.6-Flash
        names the colour of a shape, but answers 'Black' / 'White' for a uniformly coloured picture, and so does the
        fp32 host reference (reference/answer_image_fp32.py), so a uniform fixture tests the checkpoint, not the API."""
        from PIL import Image

        shapes = (("circle", (220, 20, 20)), ("square", (20, 20, 220)), ("triangle", (20, 160, 20)))
        frames = [Image.open(io.BytesIO(self.image("png", (256, 256), s, c))).convert("RGB") for s, c in shapes]
        buf = io.BytesIO()
        frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:], duration=300, loop=0)
        return buf.getvalue()

    def speech_source(self):
        return self._download(SPEECH_URL)

    def audio(self, fmt, seconds=None, sample_rate=None, channels=None, streamed=False):
        """The speech clip in `fmt`. Written to a seekable file, as an encoder normally does (the FLAC / MP4 headers
        then carry the length); `streamed=True` encodes to a pipe instead (FLAC without a total-samples field)."""
        key = ("audio", fmt, seconds, sample_rate, channels, streamed)
        if key not in self._memo:
            with tempfile.TemporaryDirectory() as tmp:
                src = os.path.join(tmp, "src.flac")
                Path(src).write_bytes(self.speech_source())
                args = ["-i", src]
                if seconds is not None:
                    # Loop the clip to the requested length (long-audio cases).
                    args = ["-stream_loop", "-1", "-i", src, "-t", str(seconds)]
                extra = []
                if sample_rate is not None:
                    extra += ["-ar", str(sample_rate)]
                if channels is not None:
                    extra += ["-ac", str(channels)]
                if streamed:
                    self._memo[key] = _ffmpeg(args + AUDIO_FORMATS[fmt] + extra + ["pipe:1"])
                else:
                    out = os.path.join(tmp, f"out.{fmt}")
                    _ffmpeg(args + AUDIO_FORMATS[fmt] + extra + [out])
                    self._memo[key] = Path(out).read_bytes()
        return self._memo[key]

    def video(self, fmt="mp4", seconds=4, size="320x240", color=(220, 20, 20), with_audio=False):
        """A still red circle on white (``image``'s shape) for ``seconds`` at 24 fps, optionally with a 440 Hz audio
        track. A shape, not a uniform colour: see ``animated_gif``."""
        key = ("video", fmt, seconds, size, color, with_audio)
        if key not in self._memo:
            codec_args, _ = VIDEO_FORMATS[fmt]
            w, h = (int(v) for v in size.split("x"))
            with tempfile.TemporaryDirectory() as tmp:
                still = os.path.join(tmp, "still.png")
                Path(still).write_bytes(self.image("png", (w, h), "circle", color))
                args = ["-loop", "1", "-framerate", "24", "-i", still]
                if with_audio:
                    args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
                args += ["-t", str(seconds), "-pix_fmt", "yuv420p"]
                if fmt in ("mp4", "mov"):
                    codec_args = codec_args + ["-movflags", "+faststart"]
                # mp4 / mov muxers need a seekable output, so every format goes through a temporary file.
                out = os.path.join(tmp, f"v.{fmt}")
                _ffmpeg(args + codec_args + [out])
                self._memo[key] = Path(out).read_bytes()
        return self._memo[key]


# ----------------------------------------------------------------------------------------------- request builders


def text(t):
    return {"type": "text", "text": t}


def image_part(url, detail=None):
    part = {"type": "image_url", "image_url": {"url": url}}
    if detail is not None:
        part["image_url"]["detail"] = detail
    return part


def audio_part(data_b64, fmt):
    return {"type": "input_audio", "input_audio": {"data": data_b64, "format": fmt}}


def video_part(url, legacy=False):
    return {"type": "input_video" if legacy else "video_url", "video_url": {"url": url}}


def user(*parts):
    return {"role": "user", "content": list(parts)}


def body(*messages, max_tokens=512, **extra):
    out = {"messages": list(messages), "max_tokens": max_tokens, "temperature": 0}
    out.update(extra)
    return out


COLOR_Q = "What color is the shape in the image? Answer with one word."
VIDEO_Q = "What color is the shape in this video? Answer with one word."
ASR_Q = "Transcribe the speech in this audio."
# OpenRouter's input-audio-transcription check: this system prompt and a user turn holding only the clip.
TRANSCRIBE_SYSTEM = (
    "You are a transcription engine. Reply with only the exact words spoken in the audio, with no commentary."
)
SPEECH_WORDS = ("dream", "nation", "children", "character", "judged")

GET_COLOR_TOOL = {
    "type": "function",
    "function": {
        "name": "record_color",
        "description": "Record the color of the shape shown in the image.",
        "parameters": {
            "type": "object",
            "properties": {"color": {"type": "string", "description": "a basic color name such as red or blue"}},
            "required": ["color"],
        },
    },
}


@dataclasses.dataclass
class Case:
    id: str
    build: Callable[["Media"], dict]
    want: str = "ok"  # "ok": 200 + sane answer; "4xx": client error with a message; "ok|4xx": either
    expect_any: tuple = ()  # lower-case substrings; the answer must contain at least one
    kind: str = "chat"  # chat | stream | tools | reasoning_on | reasoning_off
    openrouter: bool = True  # a documented OpenRouter input form (False: extra robustness probe)
    thinking: bool = False  # False: the harness turns reasoning off (fast, short answers)
    note: str = ""


def _img(fmt, **kw):
    return lambda m: data_uri(IMAGE_MIME.get(fmt, f"image/{fmt}"), m.image(fmt, **kw))


def _cases():
    c = []
    # ---- images
    c.append(
        Case("text_only", lambda m: body(user(text("What is the capital of France? One word."))), expect_any=("paris",))
    )
    c.append(
        Case(
            "image_url_https",
            lambda m: body(user(text("What animal is in this picture? One word."), image_part(IMAGE_URL))),
            expect_any=("cat", "kitten"),
        )
    )
    for fmt in ("png", "jpeg", "webp", "gif"):
        c.append(
            Case(
                f"image_data_{fmt}",
                lambda m, fmt=fmt: body(user(text(COLOR_Q), image_part(_img(fmt)(m)))),
                expect_any=("red",),
            )
        )
    c.append(
        Case(
            "image_gif_animated",
            lambda m: body(
                user(
                    text("What color is the shape in this image? One word."),
                    image_part(data_uri("image/gif", m.animated_gif())),
                )
            ),
            expect_any=("red", "blue", "green"),
            note="first frame is red",
        )
    )
    c.append(
        Case(
            "image_first_then_text",
            lambda m: body(user(image_part(_img("png")(m)), text(COLOR_Q))),
            expect_any=("red",),
        )
    )
    for detail in ("low", "high"):
        c.append(
            Case(
                f"image_detail_{detail}",
                lambda m, d=detail: body(user(text(COLOR_Q), image_part(_img("jpeg")(m), detail=d))),
                expect_any=("red",),
            )
        )
    c.append(
        Case(
            "image_multi_3",
            lambda m: body(
                user(
                    text("Three images follow. List the color of the shape in each, in order, as three words."),
                    image_part(_img("png", shape="circle", color=(220, 20, 20))(m)),
                    image_part(_img("png", shape="square", color=(20, 20, 220))(m)),
                    image_part(_img("png", shape="triangle", color=(20, 160, 20))(m)),
                )
            ),
            expect_any=("blue",),
        )
    )
    c.append(
        Case(
            "image_multi_turn",
            lambda m: body(
                user(text("Look at this image."), image_part(_img("png")(m))),
                {"role": "assistant", "content": "I see the image."},
                user(text(COLOR_Q)),
            ),
            expect_any=("red",),
        )
    )
    c.append(
        Case(
            "image_large_4096",
            lambda m: body(user(text(COLOR_Q), image_part(_img("jpeg", size=(4096, 4096))(m)))),
            expect_any=("red",),
            note="above the processor's max pixels: resized, ~8K tokens",
        )
    )
    c.append(
        Case(
            "image_tiny_16",
            lambda m: body(user(text(COLOR_Q), image_part(_img("png", size=(16, 16))(m)))),
            want="ok|4xx",
            openrouter=False,
        )
    )
    c.append(
        Case(
            "image_extreme_aspect",
            lambda m: body(user(text(COLOR_Q), image_part(_img("png", size=(6400, 16))(m)))),
            want="ok|4xx",
            openrouter=False,
            note="aspect 400:1; Qwen-style resize rejects > 200",
        )
    )
    c.append(
        Case(
            "image_bad_base64",
            lambda m: body(user(text(COLOR_Q), image_part("data:image/png;base64,not-an-image!!"))),
            want="4xx",
        )
    )
    c.append(
        Case(
            "image_truncated_png",
            lambda m: body(user(text(COLOR_Q), image_part(data_uri("image/png", m.image("png")[:200])))),
            want="4xx",
        )
    )
    c.append(Case("image_url_404", lambda m: body(user(text(COLOR_Q), image_part(MISSING_URL))), want="4xx"))
    c.append(
        Case(
            "image_bmp_data",
            lambda m: body(user(text(COLOR_Q), image_part(data_uri("image/bmp", m.image("bmp"))))),
            want="ok|4xx",
            openrouter=False,
        )
    )
    # ---- audio
    for fmt in AUDIO_FORMATS:
        c.append(
            Case(
                f"audio_{fmt}",
                lambda m, fmt=fmt: body(user(text(ASR_Q), audio_part(b64(m.audio(fmt)), fmt))),
                expect_any=SPEECH_WORDS,
            )
        )
    c.append(
        Case(
            "audio_flac_streamed",
            lambda m: body(user(text(ASR_Q), audio_part(b64(m.audio("flac", streamed=True)), "flac"))),
            want="ok|4xx",
            note="FLAC encoded to a pipe: no total-samples field in STREAMINFO",
        )
    )
    c.append(
        Case(
            "audio_wav_data_uri_in_data",
            lambda m: body(user(text(ASR_Q), audio_part(data_uri("audio/wav", m.audio("wav")), "wav"))),
            want="ok|4xx",
            openrouter=False,
            note="a data: URI inside input_audio.data (some clients send this)",
        )
    )
    c.append(
        Case(
            "audio_wav_48k_stereo",
            lambda m: body(user(text(ASR_Q), audio_part(b64(m.audio("wav", sample_rate=48000, channels=2)), "wav"))),
            expect_any=SPEECH_WORDS,
        )
    )
    c.append(
        Case(
            "audio_wav_8k",
            lambda m: body(user(text(ASR_Q), audio_part(b64(m.audio("wav", sample_rate=8000)), "wav"))),
            expect_any=SPEECH_WORDS,
        )
    )
    c.append(
        Case(
            "audio_mp3_10min",
            lambda m: body(
                user(text("Summarize this audio in one sentence."), audio_part(b64(m.audio("mp3", seconds=600)), "mp3"))
            ),
            expect_any=SPEECH_WORDS + ("king", "speech", "freedom", "equal"),
            note="~3.75K audio tokens",
        )
    )
    c.append(
        Case(
            "audio_format_mismatch",
            lambda m: body(user(text(ASR_Q), audio_part(b64(m.audio("mp3")), "wav"))),
            want="ok|4xx",
            note="mp3 bytes labelled wav",
        )
    )
    c.append(
        Case(
            "audio_bad_base64",
            lambda m: body(user(text(ASR_Q), audio_part("bm90IGF1ZGlv", "wav"))),
            want="4xx",
        )
    )
    c.append(
        Case(
            "audio_url_vllm_native",
            lambda m: body(
                user(text(ASR_Q), {"type": "audio_url", "audio_url": {"url": data_uri("audio/wav", m.audio("wav"))}})
            ),
            expect_any=SPEECH_WORDS,
            openrouter=False,
            note="vLLM's own audio form",
        )
    )
    # ---- video
    for fmt in VIDEO_FORMATS:
        c.append(
            Case(
                f"video_data_{fmt}",
                lambda m, fmt=fmt: body(user(text(VIDEO_Q), video_part(data_uri(VIDEO_FORMATS[fmt][1], m.video(fmt))))),
                expect_any=("red",),
            )
        )
    c.append(
        Case(
            "video_data_mov_alt_mime",
            lambda m: body(user(text(VIDEO_Q), video_part(data_uri("video/mov", m.video("mov"))))),
            expect_any=("red",),
            note="OpenRouter lists the MIME as video/mov",
        )
    )
    c.append(
        Case(
            "video_url_https",
            lambda m: body(user(text("Describe this video in one sentence."), video_part(VIDEO_URL))),
        )
    )
    c.append(
        Case(
            "video_input_video_legacy",
            lambda m: body(user(text(VIDEO_Q), video_part(data_uri("video/mp4", m.video("mp4")), legacy=True))),
            expect_any=("red",),
        )
    )
    c.append(
        Case(
            "video_with_audio_track",
            lambda m: body(user(text(VIDEO_Q), video_part(data_uri("video/mp4", m.video("mp4", with_audio=True))))),
            expect_any=("red",),
        )
    )
    c.append(
        Case(
            "video_60s_720p",
            lambda m: body(
                user(text(VIDEO_Q), video_part(data_uri("video/mp4", m.video("mp4", seconds=60, size="1280x720"))))
            ),
            expect_any=("red",),
            note="token count at the default frame sampling",
        )
    )
    c.append(
        Case(
            "video_bad_base64",
            lambda m: body(user(text(VIDEO_Q), video_part("data:video/mp4;base64,bm90IGEgdmlkZW8="))),
            want="4xx",
        )
    )
    # ---- mixed
    c.append(
        Case(
            "mixed_text_image_audio",
            lambda m: body(
                user(
                    text("First say the color of the shape in the image, then transcribe the audio."),
                    image_part(_img("png")(m)),
                    audio_part(b64(m.audio("wav")), "wav"),
                )
            ),
            expect_any=("red",),
        )
    )
    c.append(
        Case(
            "mixed_image_video",
            lambda m: body(
                user(
                    text(
                        "Name the color of the shape in the image and the color of the shape in the video, two words."
                    ),
                    image_part(_img("png", color=(20, 20, 220))(m)),
                    video_part(data_uri("video/mp4", m.video("mp4"))),
                )
            ),
            expect_any=("blue", "red"),
        )
    )
    # ---- API features with media in context
    c.append(
        Case(
            "stream_image",
            lambda m: body(
                user(text(COLOR_Q), image_part(_img("png")(m))), stream=True, stream_options={"include_usage": True}
            ),
            expect_any=("red",),
            kind="stream",
        )
    )
    c.append(
        Case(
            "tools_with_image",
            lambda m: body(
                user(text("Call record_color with the color of the shape in this image."), image_part(_img("png")(m))),
                tools=[GET_COLOR_TOOL],
                tool_choice="auto",
                max_tokens=1024,
            ),
            expect_any=("red",),
            kind="tools",
        )
    )
    c.append(
        Case(
            "reasoning_on_image",
            lambda m: body(user(text(COLOR_Q), image_part(_img("png")(m))), max_tokens=4096),
            expect_any=("red",),
            kind="reasoning_on",
            thinking=True,
        )
    )
    c.append(
        Case(
            "reasoning_off_effort_none",
            lambda m: body(user(text(COLOR_Q), image_part(_img("png")(m))), reasoning_effort="none"),
            expect_any=("red",),
            kind="reasoning_off",
            thinking=True,  # the case carries its own switch; the harness adds none
        )
    )
    c.append(
        Case(
            "reasoning_off_openrouter_field",
            lambda m: body(user(text(COLOR_Q), image_part(_img("png")(m))), reasoning={"enabled": False}),
            expect_any=("red",),
            kind="reasoning_off",
            thinking=True,
            note="OpenRouter's own field; a provider may ignore it",
        )
    )
    c.append(
        Case(
            "max_tokens_cut_in_reasoning",
            lambda m: body(user(text(COLOR_Q), image_part(_img("png")(m))), max_tokens=8),
            thinking=True,
            note="finish_reason length while reasoning; must not be an error",
        )
    )
    # ---- OpenRouter's provider checks, as it sends them (streamed, temperature 1, top_p 0.95, no reasoning field)
    sent = dict(max_tokens=131072, temperature=1, top_p=0.95, stream=True, stream_options={"include_usage": True})
    c.append(
        Case(
            "openrouter_audio_transcription",
            lambda m: body(
                {"role": "system", "content": TRANSCRIBE_SYSTEM}, user(audio_part(b64(m.audio("wav")), "wav")), **sent
            ),
            expect_any=SPEECH_WORDS,
            kind="stream",
            thinking=True,  # as sent: thinking at the server's default (on)
            note="the checkpoint answers <think><think>the words<|im_end|>: the words must also be the content",
        )
    )
    c.append(
        Case(
            "openrouter_youtube_url",
            lambda m: body(
                user(
                    text("What animal is shown in this video?"),
                    video_part("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
                ),
                **sent,
            ),
            want="4xx",
            thinking=True,
            note="a YouTube link is a web page, not a video file: 400 with a message, never a decoder error",
        )
    )
    return c


CASES = _cases()
CASE_IDS = [case.id for case in CASES]


def apply_target(req, case, model, target):
    """Finish a case body for a target: the model name, and reasoning off unless the case controls it.

    target "vllm": our server (chat_template_kwargs.enable_thinking); "openrouter": OpenRouter's reasoning field plus
    optional provider pinning from MM_PROVIDER (comma-separated provider slugs, no fallbacks)."""
    req = dict(req)
    req["model"] = model
    if not case.thinking:
        if target == "openrouter":
            req["reasoning"] = {"enabled": False}
        else:
            req["chat_template_kwargs"] = {"enable_thinking": False}
    if target == "openrouter" and os.environ.get("MM_PROVIDER"):
        req["provider"] = {"order": os.environ["MM_PROVIDER"].split(","), "allow_fallbacks": False}
    return req
