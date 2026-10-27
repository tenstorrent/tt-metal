# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-TTS Code2Wav as a vLLM-Omni generation stage, decoded on TT.

vLLM-Omni's own `Qwen3TTSCode2Wav` reads each step's chunks, keeps a state dict per request,
frees it when the request finishes, and returns the audio. This class keeps all of that and
swaps its PyTorch decoder for one that runs `StreamingCodecDecoder` on the device, a request
at a time. Streamed chunks only (`async_chunk: true`): without it, Omni hands the stage a
whole utterance at once, longer than one decode window.
"""

from vllm.config import get_current_vllm_config
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import Qwen3TTSCode2Wav
from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import parse_chunk_ramp

from models.demos.audio.qwen3_tts.tt.generator_vllm import _point_checkpoint_at
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_codec import LENGTH_BUCKET, STREAM_WINDOW, StreamingCodecDecoder


def stream_window(connector_config):
    """Frames each decode covers: `STREAM_WINDOW`, or more if a chunk the stage's connector
    settings allow would not fit.

    The longest chunk is the first one of an in-context clone: the reference frames, which
    vLLM-Omni trims to `ref_code_context_frames` (or `codec_left_context_frames`, default 25),
    ahead of the largest chunk size the schedule can emit.
    """
    config = connector_config or {}
    config = config.get("extra", config) if isinstance(config, dict) else getattr(config, "extra", None) or {}
    steady = int(config.get("codec_chunk_frames") or 25)
    reference = int(config.get("ref_code_context_frames") or config.get("codec_left_context_frames", 25))
    if reference <= 0:
        raise ValueError(
            "ref_code_context_frames and codec_left_context_frames are 0, so vLLM-Omni sends an in-context "
            "clone's whole reference in its first chunk, of no bounded length; set either"
        )
    sizes = [steady, int(config.get("initial_codec_chunk_frames") or 0)]
    sizes += parse_chunk_ramp(config) or []
    longest = reference + max(sizes)
    return max(STREAM_WINDOW, -(-longest // LENGTH_BUCKET) * LENGTH_BUCKET)


class _DeviceDecoder:
    """The one decoder method `Qwen3TTSCode2Wav.forward` calls, over `StreamingCodecDecoder`."""

    def __init__(self, device, window=STREAM_WINDOW):
        self.stream = StreamingCodecDecoder(device, window=window)

    def batched_chunked_decode(self, codes, lengths, caches=None, **_):
        """codes [B, 16, F], zero-padded past each row's length -> one waveform per row."""
        if caches is None:
            raise ValueError("the TT Code2Wav stage decodes streamed chunks only; set async_chunk: true")
        waveforms = []
        for row, (length, state) in enumerate(zip(lengths, caches, strict=True)):
            # Omni records an in-context clone's reference frames on its first chunk.
            skip = 0 if "codes" in state else int(state.get("prefix_frames", 0))
            waveforms.append(self.stream.decode(codes[row : row + 1, :, :length], state, skip))
        return waveforms


class Qwen3TTSCode2WavForTT(Qwen3TTSCode2Wav):
    model_capabilities = {}

    def __init__(self, *, vllm_config, mesh_device):
        super().__init__(vllm_config=vllm_config)
        if not self._async_chunk:
            raise ValueError("the TT Code2Wav stage decodes streamed chunks only; set async_chunk: true")
        # The parent builds a PyTorch decoder from the config; its weights are never loaded.
        del self.decoder
        window = stream_window(getattr(vllm_config.model_config, "stage_connector_config", None))
        self.decoder = _DeviceDecoder(mesh_device, window)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1:
            raise ValueError(
                f"the Qwen3-TTS Code2Wav stage runs on one device; got tt_data_parallel={tt_data_parallel}"
            )
        _point_checkpoint_at(hf_config)
        model = cls(vllm_config=get_current_vllm_config(), mesh_device=mesh_device)
        model.decoder.stream.warmup()
        return model
