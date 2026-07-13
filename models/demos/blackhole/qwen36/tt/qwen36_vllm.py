# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM wrapper for Qwen3.5/3.6 on Blackhole — a thin tt_transformers Generator subclass.

Hybrid model: 8 paged-KV attention + 24 GDN recurrent-state layers. GDN forbids token-padding and
isn't position-general, so prefill is model-owned (masked-bucket for short prompts / chunk-outer
trace for long, via prefill_dispatch) while Generator drives decode only. GDN + KV state is
model-bound, so the kv_cache contract param is accepted but unused.
"""
import math
import os
from typing import Mapping, Optional

import torch
from loguru import logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

import ttnn
from models.demos.blackhole.qwen36.tt.common import create_tt_model
from models.demos.blackhole.qwen36.tt.generator_interface import prefill_dispatch
from models.tt_transformers.tt.generator import Generator

_PREFILL_WARMUP_CHUNK = 2048
_PREFILL_WARMUP_BUCKET = 4096
_BLOCK_SIZE = 64


class TT_Qwen3_5ProcessingInfo(Qwen3_5ProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": 0, "video": 0}


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=TT_Qwen3_5ProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen36ForCausalLM(Generator, SupportsMultiModal):
    """vLLM-compatible wrapper for Qwen3.5-9B on Blackhole P150."""

    # supports_async_decode=False: async decode assumes on-device token/position continuity, which
    # corrupts Qwen's GDN scan. supports_sample_on_device=True: on-device sampling is decode-only.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
    }

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs,
    ) -> int:
        """All-user KV capacity = max_model_len × max_num_seqs (B=1 serving), not the inherited
        131072 fallback — so these models serve the full requested ISL (e.g. 256K)."""
        if max_model_len is not None:
            return int(max_model_len) * int(max_num_seqs or 1)
        return super().get_max_tokens_all_users(
            model_name=model_name,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            max_num_seqs=max_num_seqs,
            **kwargs,
        )

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ):
        # Weights dir: MODEL_WEIGHTS_DIR → HF_MODEL → hf_config._name_or_path; a hub id resolves to a local snapshot.
        name_or_path = os.environ.get("MODEL_WEIGHTS_DIR") or os.environ.get("HF_MODEL") or hf_config._name_or_path
        if name_or_path and not os.path.isdir(os.path.expanduser(name_or_path)):
            from huggingface_hub import snapshot_download

            # When offline/CI, resolve from the local cache only so snapshot_download
            # reads the cached refs instead of reaching the HF API (refused by HF_HUB_OFFLINE=1).
            offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
            name_or_path = snapshot_download(name_or_path, local_files_only=offline)
        args, model, _ = create_tt_model(
            mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, hf_model=name_or_path
        )
        return cls([model], [args], mesh_device)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate paged KV (8 attn layers) + external GDN state; returns the 8 KV pairs."""
        return self.model[0].allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """All prefill is model-owned (Generator drives decode only)."""
        model = self.model[0]
        if model.num_devices > 1:
            return self._prefill_forward_tp(model, tokens, page_table, prompt_lens)
        seq_len = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
        logger.info(f"Prefilling User 1 up to {seq_len} tokens")
        logits = prefill_dispatch(model, tokens, page_table, prompt_lens, use_trace=kwargs.get("enable_trace", False))
        logits = ttnn.to_torch(logits)
        # The vLLM runner unpacks (logits, rope_deltas) because the HF config has mrope_section;
        # zero deltas are correct for our text-only port.
        rope_deltas = torch.zeros(logits.shape[0], dtype=torch.long)
        logger.info(f"Finished prefill up to {seq_len} tokens, starting decode...")
        return logits, rope_deltas

    def _prefill_forward_tp(self, model, tokens, page_table, prompt_lens):
        """TP (B=1) paged prefill: masked fixed-bucket for <=2048 prompts, chunk-outer trace beyond.
        Returns host logits [1, 1, vocab] from one replica."""
        T = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
        if tokens.shape[1] > T:
            tokens = tokens[:, :T]
        logger.info(f"Prefilling User 1 up to {T} tokens (TP masked-bucket/chunked)")
        logits = model.prefill_traced_chunked(tokens, page_table, actual_len=T)  # [1,1,vocab] replicated
        logits = (
            ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
            .reshape(-1, model.args.vocab_size)[:1]
            .float()
            .view(1, 1, -1)
        )
        logger.info(f"Finished prefill up to {T} tokens, starting decode...")
        return logits, torch.zeros(1, dtype=torch.long)

    def decode_forward(self, *args, **kwargs):
        # Traced decode (single-device and TP): trace captured at pos 0 in warmup, replayed here.
        # Valid for TP — GDN state is in fixed in-place buffers, and prefill only replays pre-warmed programs.
        if not getattr(self, "_decode_logged", False):
            self._decode_logged = True
            logger.info("Decode trace replay active (Qwen)")
        return super().decode_forward(*args, **kwargs)

    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        # Capture the chunk-prefill trace + warm the masked-bucket set so requests only replay
        # pre-compiled programs (compile-clobbers-trace fix). Guard name must match the plugin's reset.
        if not enable_trace:
            return
        if getattr(self, "already_warmed_up_prefill", False):
            return
        self.already_warmed_up_prefill = True
        # Size the chunk-trace page table to the full KV cache (not a hardcoded 4096) so served ISL
        # isn't capped; still captures one chunk — just a bigger page-table tensor.
        if kv_cache:
            # Round to a multiple of 32: paged/chunked SDPA needs the page-table stick % 32 == 0.
            num_blocks = math.ceil(int(kv_cache[0][0].shape[0]) / 32) * 32
        else:
            num_blocks = math.ceil(_PREFILL_WARMUP_BUCKET / _BLOCK_SIZE)
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        logger.info(
            f"Starting Qwen prefill warmup: capturing chunk-prefill trace "
            f"(chunk={_PREFILL_WARMUP_CHUNK}, page_table_blocks={num_blocks})..."
        )
        self.model[0].capture_prefill_trace_chunked(self.mesh_device, page_table, chunk_size=_PREFILL_WARMUP_CHUNK)

    def warmup_model_decode(self, *args, **kwargs):
        # Defer to WarmupForwardMixin, which captures the paged-SDPA + GDN decode trace at pos 0.
        # Drop stale `non_greedy_decoding_on_device` from the old vLLM plugin; no-op for Qwen.
        kwargs.pop("non_greedy_decoding_on_device", None)
        return super().warmup_model_decode(*args, **kwargs)
