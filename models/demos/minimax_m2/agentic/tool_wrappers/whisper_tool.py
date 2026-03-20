# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
WhisperTool: Wraps WhisperGenerator for transcription and translation.

Uses a single TTNN-accelerated pipeline for both transcribe and translate.
The decoder trace is captured lazily on the first call and reused for all
subsequent calls across agentic turns — no re-capture between turns.

Note: distil-whisper/distil-large-v3 is an English-only model, so transcribe
and translate produce the same English output.  A single pipeline is reused
for both tasks to avoid allocating two generators on the same device.
"""

from loguru import logger

from models.demos.audio.whisper.demo.demo import create_functional_whisper_for_conditional_generation_inference_pipeline
from models.demos.audio.whisper.tt.whisper_generator import GenerationParams

MODEL_REPO = "distil-whisper/distil-large-v3"
BATCH_SIZE_PER_DEVICE = 1  # single-stream for agentic use


def _load_audio_file(path: str):
    """Load a .wav/.mp3/.flac file → (sampling_rate, numpy array)."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return sr, data


def _decode_pipeline_output(results) -> str:
    """
    Extract the first transcription string from pipeline output.

    The pipeline returns (list_of_strings, log_probs, cum_log_probs).
    We take the first element of the list (corresponding to the first audio
    in the batch — the real input; subsequent items are padding duplicates).
    """
    if isinstance(results, tuple) and len(results) >= 1:
        transcriptions = results[0]
    elif isinstance(results, list):
        transcriptions = results
    else:
        return str(results).strip()

    if isinstance(transcriptions, list) and len(transcriptions) > 0:
        return str(transcriptions[0]).strip()
    return str(transcriptions).strip()


class WhisperTool:
    """
    TTNN-accelerated Whisper STT/translate wrapper.

    Accepts a path to an audio file and returns the transcript as a string.
    The decoder trace is captured lazily on the first call and reused for all
    subsequent calls across agentic turns.
    """

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self._num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
        logger.info(f"Loading Whisper model on {self._num_devices} device(s)...")
        self._pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
            mesh_device=mesh_device,
            model_repo=MODEL_REPO,
            generation_params=GenerationParams(),
            language="en",
            task="transcribe",
            use_trace=True,
            batch_size_per_device=BATCH_SIZE_PER_DEVICE,
        )
        self._whisper_generator = getattr(self._pipeline, "whisper_generator", None)
        logger.info("Whisper ready.")

    def _pad_batch(self, sr, data):
        """Repeat a single audio sample to fill all devices (batch must be divisible by num_devices)."""
        return [(sr, data)] * self._num_devices

    def transcribe(self, path: str) -> str:
        """Transcribe an audio file to text (English)."""
        sr, data = _load_audio_file(path)
        results = self._pipeline(self._pad_batch(sr, data))
        return _decode_pipeline_output(results)

    def release_decoder_trace(self):
        """Release persistent decoder trace so other models can allocate on the same mesh.

        Whisper's first transcribe captures a persistent trace; Metal warns that further
        device allocations are unsafe while that trace is active. Call this after Whisper
        warmup if loading additional models; the next transcribe will re-capture the trace.
        """
        gen = getattr(self, "_whisper_generator", None)
        if gen is not None:
            gen.cleanup()

    def translate(self, path: str) -> str:
        """Translate audio to English text.

        distil-whisper/distil-large-v3 is English-only; this reuses the
        transcription pipeline which already produces English output.
        """
        return self.transcribe(path)

    def close(self):
        """Explicitly release the pipeline generator before device close."""
        try:
            if getattr(self, "_whisper_generator", None) is not None:
                self._whisper_generator.cleanup()
        except Exception:
            pass
        self._whisper_generator = None
        self._pipeline = None
