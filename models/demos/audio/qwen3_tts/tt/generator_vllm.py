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

Batch 1, host sampling, no plugin trace: the pipeline captures its own step traces for each
request, after that request's prefill. Plugin capabilities used:

    supports_stage_inputs   prefill_forward(stage_inputs=[additional_information])
    supports_stage_outputs  pop_stage_outputs() after every decode_forward
"""

import os

import torch

import ttnn
from models.demos.audio.qwen3_tts import stage_inputs as stage
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import Qwen3TTSPipeline

# The talker's KV dtype. The plugin passes vLLM's torch dtype; the talker is bf16 throughout.
KV_DTYPE = ttnn.bfloat16


def _point_checkpoint_at(hf_config):
    """Make `weights.checkpoint_dir` resolve to the checkpoint vLLM loaded the config from."""
    source = str(getattr(hf_config, "_name_or_path", "") or "")
    if not source or os.environ.get("QWEN3_TTS_CKPT", "").strip():
        return
    os.environ["QWEN3_TTS_CKPT" if os.path.isdir(source) else "HF_MODEL"] = source


class Qwen3TTSTalkerForConditionalGeneration:
    model_capabilities = {
        "supports_stage_inputs": True,
        "supports_stage_outputs": True,
        "supports_sample_on_device": False,
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,
    }

    def __init__(self, device, max_seq_len):
        self.device = device
        self.max_seq_len = max_seq_len
        self.pipeline = None  # built with the KV cache, which the plugin allocates later
        self.request = None
        self.reference = None
        self._references = stage.ReferenceCache()
        self._frames = []

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1 or max_batch_size != 1:
            raise ValueError(
                "the Qwen3-TTS talker stage runs batch 1 on one device; got "
                f"max_batch_size={max_batch_size}, tt_data_parallel={tt_data_parallel}"
            )
        _point_checkpoint_at(hf_config)
        return cls(mesh_device, max_seq_len)

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
        self.pipeline = Qwen3TTSPipeline(self.device, max_frames=self.max_seq_len, kv_cache=kv_cache, codec=False)
        self._inner = self.pipeline._inner_pick()
        return kv_cache

    def warmup_model_prefill(self, kv_cache, can_sample_on_device, enable_trace):
        """Nothing to do: each prefill compiles what it needs and captures its own traces."""

    def warmup_model_decode(self, kv_cache, max_batch_size, num_blocks, can_sample_on_device, enable_trace, **kwargs):
        """Nothing to do: the step traces are captured per request, after its prefill."""

    def release_request(self, slot):
        self.request = None
        self.reference = None

    def release_persistent_capture(self):
        if self.pipeline is not None:
            self.pipeline.release()

    # ── execution ───────────────────────────────────────────────────────────

    def _logits(self):
        row = self.pipeline.codec_logits(self.request)
        # vLLM-Omni's talker allows codec ids [1, codebook size) and end-of-speech; id 0 too
        # is never emitted.
        row[0] = -float("inf")
        return row.reshape(1, 1, -1)

    def prefill_forward(
        self, tokens, start_pos, prompt_lens, page_table, kv_cache, enable_trace=False, stage_inputs=None, **kwargs
    ):
        prompt_lens = [int(length) for length in torch.as_tensor(prompt_lens).reshape(-1)]
        if len(prompt_lens) != 1:
            raise ValueError(f"the talker stage prefills one request at a time, got {len(prompt_lens)}")
        if int(torch.as_tensor(start_pos).reshape(-1)[0]) != 0:
            raise ValueError("the talker stage takes whole prompts; chunked prefill and prefix caching are off")
        info = (stage_inputs or [None])[0]
        if not isinstance(info, dict):
            raise ValueError("a talker prefill needs the request's additional_information as stage_inputs")

        # The encoders run eagerly, and nothing eager may run beside a live trace.
        self.pipeline.release()
        self.reference = self._references.reference(info, self.device)
        embeddings, feed = stage.build_stage_prefill(info, self.reference, prompt_lens[0], self.pipeline.tables)
        self.pipeline.talker.set_page_table(torch.as_tensor(page_table)[0])
        # vLLM ends the request by the context length, which bounds the frames this prefill
        # must make room for.
        self.request, _ = self.pipeline.start(embeddings, self.max_seq_len - prompt_lens[0], feed)
        # The talker's config declares M-RoPE, so the plugin reads a position delta per row.
        # Its three axes carry the same positions throughout, so the delta is zero.
        return self._logits(), torch.zeros(1, dtype=torch.long)

    def decode_forward(self, tokens, start_pos, page_table, kv_cache, enable_trace=False, slot_remap=None, **kwargs):
        if slot_remap is not None and torch.as_tensor(slot_remap).reshape(-1)[0] != 0:
            raise ValueError(f"the talker stage holds one request in slot 0; slot_remap {slot_remap}")
        position = int(torch.as_tensor(start_pos).reshape(-1)[0])
        if self.request is None or position != self.request.position:
            raise ValueError(
                f"decode at position {position}, but the talker's request is at "
                f"{None if self.request is None else self.request.position}"
            )
        self.pipeline.talker.set_page_table(torch.as_tensor(page_table)[0])
        first = int(torch.as_tensor(tokens).reshape(-1)[0])
        frame = self.pipeline.predict(self.request, first, self._inner)
        self._frames = [torch.tensor(frame, dtype=torch.long)]
        self.pipeline.feed_back(self.request, frame)
        return self._logits()

    def pop_stage_outputs(self):
        frames, self._frames = self._frames, []
        return frames
