# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM (vllm-tt-plugin) adapter for Kimi-Linear-48B-A3B-Instruct.

Registered through EXTRA_MODELS_DIR/kimi_linear/vllm_metadata.json as ``TTKimiLinearForCausalLM``. Follows the Qwen3.6
Blackhole pattern: prefill is model-owned (one request at a time into its decode slot, KDA state + MLA latent cache),
decode goes through models.tt_transformers.tt.generator.Generator (traced decode with persistent device inputs). The
KDA recurrent/conv state is model-owned and indexed by decode slot; the plugin's slot_remap is mirrored onto it.
Host sampling first (``supports_sample_on_device`` False); on-device sampling is an optimisation-stage item.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.config import KimiLinearConfig
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.weights import KimiCheckpoint
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.layer import PrecisionPolicy
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.model import KimiLinearModel
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.model_args import KimiModelArgs
from models.tt_transformers.tt.generator import Generator

BLOCK_SIZE = 64


def _resolve_snapshot(hf_config) -> Path:
    for key in ("MODEL_WEIGHTS_DIR", "KIMI_SNAPSHOT"):
        v = os.environ.get(key)
        if v and Path(v).is_dir():
            return Path(v)
    name = (
        os.environ.get("HF_MODEL")
        or getattr(hf_config, "_name_or_path", None)
        or "moonshotai/Kimi-Linear-48B-A3B-Instruct"
    )
    if Path(name).is_dir():
        return Path(name)
    from huggingface_hub import snapshot_download

    offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
    return Path(snapshot_download(name, local_files_only=offline))


def _precision_from_env() -> PrecisionPolicy:
    pol = PrecisionPolicy()
    name = os.environ.get("KIMI_PRECISION", "").lower()
    if name in ("bfp4", "bfp4_experts"):
        pol.experts = ttnn.bfloat4_b
    return pol


class KimiLinearModelForGenerator(KimiLinearModel):
    """KimiLinearModel + the hooks tt_transformers' Generator drives for decode."""

    sampling = None  # host sampling
    sampling_dp = 1

    def __init__(self, *a, args: KimiModelArgs, **kw):
        super().__init__(*a, **kw)
        self.args = args
        self.mesh_device = args.mesh_device

    def switch_mode(self, mode):
        return None

    # --- decode hooks -------------------------------------------------------------------------------------------------
    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        B = self.max_batch_size
        tok = tokens.reshape(-1)[:B]
        if isinstance(current_pos, torch.Tensor):
            pos = current_pos.reshape(-1)[:B]
        else:
            pos = torch.full((B,), int(current_pos))
        pt = page_table if page_table is not None else torch.zeros(B, 1, dtype=torch.int32)
        # idle rows carry position -1 in the plugin: keep them (the paged ops skip negative positions)
        tok_h, pos_h, pt_h = self._host_decode_inputs(tok, pos, pt)
        return tok_h, pos_h, None, pt_h

    def prepare_inputs_decode(self, tokens, current_pos, page_table=None):
        from models.tt_transformers.tt.common import copy_host_to_device

        host = self.prepare_decode_inputs_host(tokens, current_pos, page_table=page_table)
        return copy_host_to_device(host, mesh_device=self.mesh_device)

    def ttnn_decode_forward(
        self, tokens, current_pos, rot_mat_idxs=None, page_table=None, kv_cache=None, on_device_logits=False, **kwargs
    ):
        logits = self.decode_device(tokens, current_pos, page_table, gather=True)  # [1,1,B,vocab]
        return logits, None

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
        full = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
        rows = full.reshape(-1, full.shape[-1])[: B * S, : self.cfg.vocab_size]
        return rows.reshape(B, S, self.cfg.vocab_size)


class KimiLinearForCausalLM(Generator):
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": False,
        "supports_chunked_prefill": False,
    }
    # host tokens/positions are authoritative every step (no device-side token feedback)
    _tt_vllm_always_refresh_decode_trace_inputs = True

    def __init__(self, model, model_args, mesh_device, tokenizer=None):
        super().__init__(model, model_args, mesh_device, tokenizer=tokenizer)
        self._num_blocks = None

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None, **kwargs
    ):
        assert tt_data_parallel == 1, "data parallel not supported"
        snapshot = _resolve_snapshot(hf_config)
        cfg = KimiLinearConfig.from_snapshot(snapshot)
        cfg.validate()
        cache_root = os.environ.get("TT_CACHE_PATH") or os.environ.get("TT_DIT_CACHE_DIR")
        cache_path = Path(cache_root) / "kimi_linear_48b" / f"tp{tuple(mesh_device.shape)[1]}" if cache_root else None
        args = KimiModelArgs(mesh_device, cfg, max_batch_size=max_batch_size, max_seq_len=max_seq_len)
        t0 = time.time()
        ck = KimiCheckpoint(snapshot, cfg)
        model = KimiLinearModelForGenerator(
            mesh_device,
            cfg,
            ck,
            max_batch_size=max_batch_size,
            cache_path=cache_path,
            precision=_precision_from_env(),
            block_size=BLOCK_SIZE,
            args=args,
        )
        ck.close()
        logger.info(
            f"Kimi-Linear model initialised in {time.time()-t0:.0f}s (B={max_batch_size}, max_seq_len={max_seq_len}, tp={tuple(mesh_device.shape)[1]})"
        )
        return cls([model], [args], mesh_device)

    @classmethod
    def get_max_tokens_all_users(
        cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None, max_num_seqs=None, **kwargs
    ):
        """Shared paged-KV token pool. Only the 7 MLA layers use it (8 KB/token bf16 replicated per chip), so a large pool is
        cheap; cap at 1M tokens (8.4 GB/chip)."""
        override = os.environ.get("KIMI_MAX_TOKENS_ALL_USERS")
        if override:
            return int(override)
        if max_model_len is not None:
            return int(min(max_model_len * (max_num_seqs or 1), 1_048_576))
        return 131072

    # --- caches / state -------------------------------------------------------------------------------------------------
    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Legacy API: the plugin computes a per-layer shape from its config; we only take num_blocks and allocate the
        7 latent caches + the KDA slot state. Returns the latent caches (one per MLA layer)."""
        num_blocks = int(kv_cache_shape[0])
        self._num_blocks = num_blocks
        caches = self.model[0].allocate_state(num_blocks)
        return caches

    # --- prefill (model-owned) ------------------------------------------------------------------------------------------
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, empty_slots=None, **kwargs):
        model = self.model[0]
        N = tokens.shape[0]
        slots = list(empty_slots) if empty_slots is not None else list(range(N))
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        out = []
        for u in range(N):
            n = int(prompt_lens[u])
            t0 = time.time()
            logits = model.prefill(tokens[u, :n], pt[u : u + 1], slot=int(slots[u]))
            logger.info(f"prefill user {u} -> slot {slots[u]}: {n} tokens in {time.time()-t0:.2f}s")
            out.append(logits.reshape(1, 1, -1))
        return torch.cat(out, dim=0)  # [N, 1, vocab]

    # --- decode: Generator.decode_forward drives prepare_inputs_decode / ttnn_decode_forward / process_output_decode -----
    def decode_forward(self, *args, **kwargs):
        slot_remap = kwargs.get("slot_remap")
        if slot_remap is not None and self.model[0].max_batch_size > 1:
            self.model[0].remap_slots(slot_remap)
        return super().decode_forward(*args, **kwargs)

    # --- warm-up ------------------------------------------------------------------------------------------------------------
    def warmup_model_prefill(self, kv_cache=None, enable_trace=False, *args, **kwargs):
        if getattr(self, "already_warmed_up_prefill", False) or enable_trace:
            self.already_warmed_up_prefill = True
            return
        self.already_warmed_up_prefill = True
        model = self.model[0]
        n = 64
        pt = torch.arange(max(1, n // BLOCK_SIZE + 1), dtype=torch.int32).reshape(1, -1)
        toks = torch.full((n,), model.cfg.pad_token_id, dtype=torch.long)
        t0 = time.time()
        model.prefill(toks, pt, slot=0)
        model.reset_slot(0)
        logger.info(f"prefill warm-up ({n} tokens) in {time.time()-t0:.1f}s")

    def warmup_model_decode(self, kv_cache=None, enable_trace=False, *args, **kwargs):
        model = self.model[0]
        B = model.max_batch_size
        blocks = self._num_blocks or 1
        max_blocks = int(kwargs.get("max_num_blocks_per_req", 0)) or min(blocks, 8)
        pt = torch.zeros(B, max_blocks, dtype=torch.int32)
        toks = torch.full((B, 1), model.cfg.pad_token_id, dtype=torch.long)
        pos = torch.full((B,), -1, dtype=torch.int32)  # idle rows: skipped by the paged ops
        t0 = time.time()
        self.decode_forward(
            toks, pos, page_table=pt, kv_cache=kv_cache, enable_trace=enable_trace, read_from_device=True
        )
        for s in range(B):
            model.reset_slot(s)
        logger.info(f"decode warm-up (trace={enable_trace}) in {time.time()-t0:.1f}s")
