# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The Qwen3-TTS talker as a vLLM stage, served by the TT vLLM plugin under vLLM-Omni.

One vLLM decode step is one frame of `Qwen3TTSPipeline`'s loop:

    prefill_forward   the prompt from the request's `stage_inputs`  -> codebook-0 logits, frame 0
    decode_forward    codebook 0 of frame k, sampled by vLLM        -> codebooks 1-15 of frame k,
                      then the talker steps over frame k            -> codebook-0 logits, frame k+1
    pop_stage_outputs frame k, all 16 codes, for the next stage

vLLM's host sampler owns codebook 0: temperature, top-k, the repetition penalty,
`min_tokens` and the end-of-speech stop id. This adapter owns the logits mask and all of
the code predictor's sampling, which reads the checkpoint's `subtalker_*` settings.

Up to `max_num_seqs` requests decode together, one per talker row: decode row i is device
slot i. Host sampling, no plugin trace: the pipeline captures its own traces at warmup,
the step's and a prefill's per prompt length, and keeps them, so a request arriving
mid-decode stalls the others only for its prefill's replay. Plugin capabilities used:

    supports_stage_inputs   prefill_forward(stage_inputs=[additional_information])
    supports_stage_outputs  pop_stage_outputs() after every decode_forward
"""

import os

import torch

import ttnn
from models.demos.audio.qwen3_tts import stage_inputs as stage
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import Qwen3TTSPipeline, ReferenceEncoders

# The talker's KV dtype. The plugin passes vLLM's torch dtype; the talker is bf16 throughout.
KV_DTYPE = ttnn.bfloat16

# The shortest prompt length the talker's prefill is traced at. Each next one doubles, so a
# prompt is padded by less than itself, up to the context length.
MIN_PREFILL_LENGTH = 128


def _prefill_lengths(max_seq_len):
    longest = -(-max_seq_len // 32) * 32
    lengths = [min(MIN_PREFILL_LENGTH, longest)]
    while lengths[-1] < longest:
        lengths.append(min(lengths[-1] * 2, longest))
    return lengths


def _point_checkpoint_at(hf_config):
    """Make `weights.checkpoint_dir` resolve to the checkpoint vLLM loaded the config from."""
    source = str(getattr(hf_config, "_name_or_path", "") or "")
    if not source or os.environ.get("QWEN3_TTS_CKPT", "").strip():
        return
    os.environ["QWEN3_TTS_CKPT" if os.path.isdir(source) else "HF_MODEL"] = source


class _Row:
    """One slot's request between frames: the talker position its next frame goes to, and
    its text track still to feed (None once the whole text is in the prompt)."""

    def __init__(self, position, feed):
        self.position = position
        self.feed = feed


class Qwen3TTSTalkerForConditionalGeneration:
    model_capabilities = {
        "supports_stage_inputs": True,
        "supports_stage_outputs": True,
        "supports_sample_on_device": False,
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,
    }

    def __init__(self, device, max_seq_len, max_batch_size=1):
        self.device = device
        self.max_seq_len = max_seq_len
        self.batch = max_batch_size
        self.pipeline = None  # built with the KV cache, which the plugin allocates later
        self._rows = [None] * max_batch_size
        self._references = stage.ReferenceCache()
        self._frames = []

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1:
            raise ValueError(f"the Qwen3-TTS talker stage runs on one device; got tt_data_parallel={tt_data_parallel}")
        _point_checkpoint_at(hf_config)
        return cls(mesh_device, max_seq_len, max_batch_size)

    @classmethod
    def get_max_tokens_all_users(cls, max_model_len, max_num_seqs, **kwargs):
        """KV positions to allocate: one full context per sequence, nothing spare."""
        return max_model_len * max_num_seqs

    # ── allocation and lifecycle ────────────────────────────────────────────

    def allocate_kv_cache(self, shape, dtype, num_layers):
        """Allocate the talker's paged cache, then build the pipeline around it."""
        empty = torch.zeros(tuple(shape))
        make = lambda: ttnn.from_torch(empty, dtype=KV_DTYPE, layout=ttnn.TILE_LAYOUT, device=self.device)
        kv_cache = [(make(), make()) for _ in range(num_layers)]
        # The pipeline sizes its talker for prompt room plus frames; this covers the context.
        self.pipeline = Qwen3TTSPipeline(
            self.device,
            max_frames=self.max_seq_len,
            kv_cache=kv_cache,
            codec=False,
            batch=self.batch,
            prefill_lengths=_prefill_lengths(self.max_seq_len),
        )
        # On the host: a clip length the device encoders have not seen compiles for seconds,
        # stalling every request on this device.
        self._encoders = ReferenceEncoders()
        return kv_cache

    def warmup_model_prefill(self, kv_cache, can_sample_on_device, enable_trace):
        """Nothing to do: `warmup_model_decode` captures the prefill traces with the steps'."""

    def warmup_model_decode(self, kv_cache, max_batch_size, num_blocks, can_sample_on_device, enable_trace, **kwargs):
        """Capture the step and prefill traces, before any request: every trace must exist
        before any replays, and eager work beside a live trace hangs the device."""
        self.pipeline.capture()

    def release_request(self, slot):
        self._rows[slot] = None

    def release_persistent_capture(self):
        if self.pipeline is not None:
            self.pipeline.release()

    # ── execution ───────────────────────────────────────────────────────────

    def _logits(self):
        rows = self.pipeline.codec_logits_rows()
        # vLLM-Omni's talker allows codec ids [1, codebook size) and end-of-speech; id 0 too
        # is never emitted.
        rows[:, 0] = -float("inf")
        return rows.reshape(self.batch, 1, -1)

    def prefill_forward(
        self,
        tokens,
        start_pos,
        prompt_lens,
        page_table,
        kv_cache,
        enable_trace=False,
        stage_inputs=None,
        empty_slots=None,
        **kwargs,
    ):
        prompt_lens = [int(length) for length in torch.as_tensor(prompt_lens).reshape(-1)]
        if any(int(start) != 0 for start in torch.as_tensor(start_pos).reshape(-1)):
            raise ValueError("the talker stage takes whole prompts; chunked prefill and prefix caching are off")
        stage_inputs = list(stage_inputs or [None] * len(prompt_lens))
        slots = list(range(len(prompt_lens))) if empty_slots is None else [int(slot) for slot in empty_slots]
        page_table = torch.as_tensor(page_table)

        for row, (length, info, slot) in enumerate(zip(prompt_lens, stage_inputs, slots, strict=True)):
            if not isinstance(info, dict):
                raise ValueError("a talker prefill needs the request's additional_information as stage_inputs")
            reference = self._references.reference(info, self.device, self._encoders)
            embeddings, feed = stage.build_stage_prefill(info, reference, length, self.pipeline.tables)
            self.pipeline.talker.set_page_table(page_table[row], slot)
            # vLLM ends the request by the context length, which bounds its frames.
            position = self.pipeline.start_row(slot, embeddings, self.max_seq_len - length)
            self._rows[slot] = _Row(position, feed)
        self.pipeline.capture()
        # The talker's config declares M-RoPE, so the plugin reads a position delta per row.
        # Its three axes carry the same positions throughout, so the delta is zero.
        return self._logits()[slots], torch.zeros(len(slots), dtype=torch.long)

    def decode_forward(self, tokens, start_pos, page_table, kv_cache, enable_trace=False, slot_remap=None, **kwargs):
        if slot_remap is not None:
            order = [int(slot) for slot in torch.as_tensor(slot_remap).reshape(-1)]
            if order != list(range(len(order))):
                self._rows = [self._rows[slot] for slot in order]
                self.pipeline.move_rows(order)
        positions = [int(position) for position in torch.as_tensor(start_pos).reshape(-1)]
        if len(positions) != self.batch:
            raise ValueError(f"{len(positions)} decode rows for a talker of {self.batch}")
        firsts = [int(code) for code in torch.as_tensor(tokens).reshape(-1)]
        active = [position >= 0 for position in positions]
        for slot, (position, live) in enumerate(zip(positions, active)):
            request = self._rows[slot]
            if live and (request is None or position != request.position):
                raise ValueError(
                    f"slot {slot} decodes at position {position}, but its request is at "
                    f"{None if request is None else request.position}"
                )
        # An inactive row steps too, at position 0 of block 0, vLLM's null block, which the
        # plugin's zeroed page-table row points it at; its output is dropped.
        self.pipeline.talker.set_page_tables(page_table)
        frames = self.pipeline.predict_rows([first if live else 0 for first, live in zip(firsts, active)])
        self._frames = [torch.tensor(frame, dtype=torch.long) if live else None for frame, live in zip(frames, active)]
        self.pipeline.feed_back_rows(
            frames,
            [position if live else 0 for position, live in zip(positions, active)],
            [request.feed if live else None for request, live in zip(self._rows, active)],
        )
        for request, live in zip(self._rows, active):
            if live:
                request.position += 1
        return self._logits()

    def pop_stage_outputs(self):
        frames, self._frames = self._frames, []
        return frames
