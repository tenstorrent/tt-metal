# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""vLLM model class for Qwen3-ASR on Tenstorrent.

Prompt construction and audio preprocessing stay in upstream vLLM's Qwen3-ASR
processor; this class registers that processor and serves the model through
``Qwen3ASRGenerator``.
"""
import torch
from huggingface_hub import snapshot_download
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRDummyInputsBuilder,
    Qwen3ASRMultiModalProcessor,
    Qwen3ASRProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from models.demos.audio.qwen3_asr.demo.transcribe import find_text_decoder
from models.demos.audio.qwen3_asr.tt.generator import Qwen3ASRGenerator
from models.tt_transformers.tt.model_config import DecodersPrecision


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3ASRMultiModalProcessor, info=Qwen3ASRProcessingInfo, dummy_inputs=Qwen3ASRDummyInputsBuilder
)
class Qwen3ASRForConditionalGeneration(Qwen3ASRGenerator, SupportsMultiModal):
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1:
            raise ValueError(f"Qwen3-ASR runs on one submesh, got tt_data_parallel={tt_data_parallel}")
        model_id = getattr(hf_config, "_name_or_path", None) or hf_config.name_or_path
        snap_dir = snapshot_download(model_id)
        opt = None
        if optimizations is not None:
            precision = DecodersPrecision.from_string(optimizations)
            opt = lambda model_args: precision(model_args.n_layers, model_args.model_name)
        return cls.build(
            mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            snap_dir=snap_dir,
            text_decoder_dir=find_text_decoder(snap_dir),
            optimizations=opt,
        )

    def prefill_forward(self, *args, **kwargs):
        """vLLM runs Qwen3-ASR with M-RoPE, so it takes one position delta per request. Audio
        and text positions are both linear, so every delta is 0."""
        tokens = kwargs["tokens"] if "tokens" in kwargs else args[0]
        return super().prefill_forward(*args, **kwargs), torch.zeros(tokens.shape[0], dtype=torch.long)

    def decode_forward(self, *args, rope_deltas_all_users=None, **kwargs):
        if rope_deltas_all_users is not None and any(int(d) != 0 for d in rope_deltas_all_users):
            raise ValueError(f"Qwen3-ASR positions are linear; got M-RoPE deltas {rope_deltas_all_users}")
        return super().decode_forward(*args, **kwargs)
