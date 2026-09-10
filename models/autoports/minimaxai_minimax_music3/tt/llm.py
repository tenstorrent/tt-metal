# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-8B backbone of MiniMax-Music3 on one Blackhole chip, wrapped around ``models.tt_transformers``.

The autoregressive stage of MiniMax-Music3 drives its ``Qwen3ForCausalLM`` language model in a
way the stock tt_transformers ``Generator`` does not cover:

* batch 2 always (row 0 conditional prompt, row 1 unconditional / CFG prompt),
* prefill from **embeddings** of a prompt of up to 5000 tokens,
* every decode step consumes a pre-embedded input (the *sum* of the semantic-code embedding and
  the depth decoder's residual-code embeddings, scaled by ``8**-0.5``), not token ids,
* every step needs **both** the final-norm hidden state (it conditions the depth decoder and the
  flow-matching DiT) and the logits (CFG + top-k sampling of the next semantic code).

``MusicTransformer`` is a small subclass of ``tt_transformers.Transformer`` whose ``forward``
returns ``(normed_hidden, logits)``; ``MusicLLM`` owns the ``ModelArgs``, the paged KV cache and
the traced decode step. Nothing under ``models/tt_transformers`` is modified.

Dtype policy ``"functional"`` = ``DecodersPrecision.accuracy`` for an unknown model name:
bf16 WQKV / WO / KV cache with HiFi4 attention matmuls and SDPA, bfp8 MLP weights (HiFi2),
bf16 embedding, norms and LM head (the LM-head *output* dtype is forced to bf16 here, the stock
default is bfp8 which is too coarse for the CFG top-k over 16384 semantic codes).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
from loguru import logger

import ttnn
from models.autoports.minimaxai_minimax_music3.tt.constants import (
    AUDIO_CODE_OFFSET,
    LLM_BATCH,
    LLM_HIDDEN,
    LLM_MAX_POSITION_EMBEDDINGS,
    NUM_CODEBOOKS,
)
from models.tt_transformers.tt.common import (
    Mode,
    PagedAttentionConfig,
    get_max_prefill_chunk_size,
    get_padded_prefill_len,
    num_blocks_in_seq,
)
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelArgs,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)

TILE = 32

# Per-decoder precision settings by policy name (tt_transformers ModelOptimizations vocabulary).
# "functional" reproduces ModelOptimizations.accuracy for an unknown model name exactly (bf16
# attention weights and KV cache at HiFi4, bfp8 MLP weights at HiFi2/fp16-acc, activations follow
# the op defaults: bfp8 Q in the prefill SDPA and a bfp8 MLP intermediate). The other policies are
# precision probes used to localize the long-prompt accuracy loss (doc/llm/README.md); they are
# not the shipped default. (A further probe with fp32-accumulating HiFi2 MLP matmuls does not
# fit: the prefill w1 matmul's circular buffers grow to 1.60 MB > 1.5 MB L1 on P150.)
_BF16_ATTENTION = {
    TensorGroup.WQKV: PrecisionSetting.BF16,
    TensorGroup.KV_CACHE: PrecisionSetting.BF16,
    TensorGroup.WO: PrecisionSetting.BF16,
}
_HIFI4_ATTENTION = {
    OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI4,
    OpGroup.LI_QKV_PREFILL: MathFidelitySetting.HIFI4,
    OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI4,
    OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4,
    OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI4,
    OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI4,
}
_PREC = {"bf16": PrecisionSetting.BF16, "bfp8": PrecisionSetting.BFP8, "bfp4": PrecisionSetting.BFP4}


_LOFI_DECODE = {
    OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI,
    OpGroup.LI_FF2: MathFidelitySetting.LOFI,
    OpGroup.LI_QKV_DECODE: MathFidelitySetting.LOFI,
    OpGroup.LI_O_DECODE: MathFidelitySetting.LOFI,
}


def _policy(attn: str, mlp: str, kv: str, lm_head: str = "bfp8", decode_fidelity: str = "hifi2") -> dict:
    """A stage-07 policy: per-group weight dtypes; fidelities follow tt_transformers' defaults (HiFi2 with fp32
    accumulation) except bf16 attention (HiFi4, as in "functional") and bfp4 MLP weights (LoFi, as in the stock
    ``performance`` policy). ``lm_head`` is the LM-head weight dtype (the embedding stays bf16, the LM-head
    output bf16)."""
    tp = {
        TensorGroup.WQKV: _PREC[attn],
        TensorGroup.WO: _PREC[attn],
        TensorGroup.KV_CACHE: _PREC[kv],
        TensorGroup.FF1_FF3: _PREC[mlp],
        TensorGroup.FF2: _PREC[mlp],
    }
    of = {}
    if attn == "bf16":
        of.update(_HIFI4_ATTENTION)
    if mlp == "bfp4":
        of[OpGroup.LI_FF1_FF3] = MathFidelitySetting.LOFI
    if decode_fidelity == "lofi":
        # The DRAM-sharded decode matmuls run on 12 cores (one per DRAM bank) and are compute-bound at HiFi2 with
        # bfp8 weights (53 % of DRAM bandwidth, doc/optimize/README.md); LoFi halves the math passes. Note
        # LI_FF1_FF3 / LI_FF2 also cover the prefill MLP matmuls (tt_transformers has no separate prefill group).
        of.update(_LOFI_DECODE)
    return {
        "TensorPrecision": tp,
        "OpFidelity": of,
        "lm_head": lm_head,
        "lm_head_fidelity": decode_fidelity,
        "groups": {"attn": attn, "mlp": mlp, "kv": kv, "decode_fidelity": decode_fidelity},
    }


DTYPE_POLICIES = {
    # Stage 02-06 default: bf16 attention weights + KV cache at HiFi4, bfp8 MLP, bf16 LM head.
    "functional": {**_policy("bf16", "bfp8", "bf16", lm_head="bf16")},
    # + bf16 activations everywhere (prefill SDPA Q, MLP intermediate) instead of the bfp8 op defaults.
    "functional_bf16_act": {
        "TensorPrecision": {**_BF16_ATTENTION, TensorGroup.ACTIVATION: PrecisionSetting.BF16},
        "OpFidelity": dict(_HIFI4_ATTENTION),
        "lm_head": "bf16",
        "groups": {"attn": "bf16", "mlp": "bfp8", "kv": "bf16", "act": "bf16"},
    },
    # Stage 07 default: bfp8 attention + MLP + LM-head weights, bfp8 KV cache, bf16 embedding / norms / activations,
    # LoFi decode matmuls + LM head (HiFi2 stays for the prefill attention matmuls and SDPA), see doc/optimize/README.md.
    "optimized": _policy("bfp8", "bfp8", "bfp8", decode_fidelity="lofi"),
    # The same weights at HiFi2 everywhere (the stage-07 first candidate; 17.9 vs 22.5 teacher-forced frames/s).
    "optimized_hifi2": _policy("bfp8", "bfp8", "bfp8"),
}
DTYPE_POLICIES["optimized_lofi"] = DTYPE_POLICIES["optimized"]  # name used by the first sweep runs
# Stage 07 datatype sweep: {MLP bfp4 / bfp8 / bf16} x {KV bfp8 / bf16} with bfp8 attention + LM head, LoFi decode.
for _mlp in ("bfp4", "bfp8", "bf16"):
    for _kv in ("bfp8", "bf16"):
        DTYPE_POLICIES[f"opt_mlp-{_mlp}_kv-{_kv}"] = _policy("bfp8", _mlp, _kv, decode_fidelity="lofi")

_TT_DTYPE = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def policy_cache_key(policy: str) -> str:
    """Converted-weight cache directory for a policy: keyed by the weight dtypes only (the KV-cache dtype does not
    change any stored weight), so sweep policies that share weight dtypes share one conversion."""
    p = DTYPE_POLICIES[policy]
    g = p["groups"]
    key = f"attn-{g['attn']}_mlp-{g['mlp']}_lmhead-{p['lm_head']}"  # fidelity does not change stored weights
    if g.get("act"):
        key += f"_act-{g['act']}"
    return key


def decoders_precision_for(policy: str, num_decoders: int, model_name: str) -> DecodersPrecision:
    p = DTYPE_POLICIES[policy]
    conf = ModelOptimizations({"TensorPrecision": p["TensorPrecision"], "OpFidelity": p["OpFidelity"]})
    conf.__name__ = policy
    return DecodersPrecision(num_decoders, model_name, conf)


class MusicTransformer(Transformer):
    """``tt_transformers.Transformer`` whose forward returns the normed hidden state *and* the logits.

    Behaviour is identical to the parent for the decoder stack; only the tail differs:

    * PREFILL with ``get_last_token == -1`` returns the raw (un-normed) hidden states of the
      whole chunk, exactly like the parent (used for the non-final chunks of a chunked prefill and
      for layer tests).
    * PREFILL with ``get_last_token >= 0`` and DECODE return ``(hidden, logits)`` where ``hidden``
      is the final RMSNorm output ``[1, 1, 32, dim]`` (bf16, DRAM interleaved, tile layout) and
      ``logits`` the LM-head output ``[1, 1, 32, padded_vocab]`` (bf16, DRAM interleaved).

    The prefetcher / Galaxy / hybrid per-layer page-table paths of the parent are not needed on
    one chip and are rejected up front instead of being half-supported.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.prefetcher is None, "MusicTransformer does not support the prefetcher"
        assert not self.args.is_galaxy, "MusicTransformer is a single-chip model"

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode: Mode = Mode.DECODE,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        page_tables_per_layer=None,
    ):
        assert page_tables_per_layer is None, "per-layer page tables are a vLLM hybrid-attention feature"
        assert rot_mats_local is None, "Qwen3 has no local (sliding) RoPE"

        if mode == Mode.PREFILL:
            rot_mats_global = self._slice_prefill_rot_mats(rot_mats_global, chunk_start_idx)

        for i, layer in enumerate(self.layers):
            activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
                decoder_id=i, tensor=TensorGroup.ACTIVATION
            )
            if mode == Mode.DECODE:
                x = ttnn.to_memory_config(x, self.args.get_residual_mem_config(mode, None), activation_dtype)
            elif activation_dtype is not None and x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)
            x = layer(
                x,
                current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=None,
                user_id=user_id,
                mode=mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
                batch_size=batch_size,
            )

        if mode == Mode.PREFILL and get_last_token == -1:
            return x

        if get_last_token != -1:
            # Nearest tile row block that contains the last token; the caller picks the row.
            x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + TILE, x.shape[-1]))

        x = self.norm(x, mode=mode, norm_config=self.args.get_norm_config("lm_head", mode, None))

        # The LM head deallocates its input, so keep an interleaved DRAM copy of the normed hidden.
        if x.is_sharded():
            hidden = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        else:
            hidden = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        lm_head_input_mem_cfg = self.args.get_lm_head_input_mem_config(mode, None)
        if mode == Mode.PREFILL and lm_head_input_mem_cfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_input_mem_cfg)
        logits = self.lm_head(x)
        logits = ttnn.to_memory_config(logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return hidden, logits


def _default_weight_cache_root() -> Path:
    root = os.environ.get("TT_CACHE_PATH")
    if root:
        return Path(root)
    base = os.environ.get("TT_METAL_CACHE") or str(Path.home() / ".cache" / "tt-metal-cache-mm3")
    return Path(base) / "minimax_music3_llm"


class MusicLLM:
    """The MiniMax-Music3 language model on one chip: prefill from embeddings, traced decode, KV cache.

    Args:
        mesh_device: an open 1x1 ``ttnn.MeshDevice`` (the chip is opened by the caller).
        max_batch_size: 2 (conditional + unconditional rows). tt_transformers pads the batch to a
            32-row tile internally; rows >= ``max_batch_size`` are padding.
        max_seq_len: context in tokens (prompt + frames). The checkpoint advertises 10240.
        dtype_policy: ``"functional"`` (see module docstring) or one of the probes in ``DTYPE_POLICIES``.
        block_size: paged KV-cache block size in tokens.
        hf_model_dir: the Qwen3 backbone directory; defaults to ``$HF_MODEL``. ``ModelArgs`` reads
            ``HF_MODEL`` from the environment, so it is set here when a directory is passed.
        logits_window: optional logical vocabulary range ``[start, end)``. When set, the decode trace
            slices that (tile-aligned) column window out of the logits on device and untilizes only
            it, so :meth:`decode_windowed` reads back ``B x (end - start)`` values instead of the whole
            200k vocabulary (the AR loop only needs the end token and the 16384 semantic codes).
            Can also be set later with :meth:`set_logits_window`, before the first decode.
    """

    def __init__(
        self,
        mesh_device,
        *,
        max_batch_size: int = LLM_BATCH,
        max_seq_len: int = LLM_MAX_POSITION_EMBEDDINGS,
        dtype_policy: str = "functional",
        block_size: int = 32,
        hf_model_dir: Optional[str] = None,
        weight_cache_root: Optional[str] = None,
        logits_window: Optional[Tuple[int, int]] = None,
        num_layers: Optional[int] = None,
    ):
        if dtype_policy not in DTYPE_POLICIES:
            raise ValueError(f"unknown dtype_policy {dtype_policy!r}; supported: {sorted(DTYPE_POLICIES)}")
        assert list(mesh_device.shape) == [1, 1], f"MusicLLM runs on a 1x1 mesh, got {list(mesh_device.shape)}"
        if hf_model_dir is not None:
            os.environ["HF_MODEL"] = str(hf_model_dir)
        if not os.environ.get("HF_MODEL"):
            raise RuntimeError("HF_MODEL must point at the MiniMax-Music3 language_model directory")
        # ModelArgs joins "model_cache" with HF_MODEL; with an absolute HF_MODEL that would drop the
        # converted-weight cache *inside* the HF snapshot. Pin it to a dedicated directory instead.
        cache_root = Path(weight_cache_root) if weight_cache_root else _default_weight_cache_root()
        # Stage 02-06 caches keep their policy-named directories; the stage-07 policies share one directory per
        # weight-dtype combination (the KV-cache dtype does not change any stored weight).
        cache_dir = dtype_policy if dtype_policy.startswith("functional") else policy_cache_key(dtype_policy)
        os.environ["TT_CACHE_PATH"] = str(cache_root / cache_dir)

        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype_policy = dtype_policy
        self.block_size = block_size
        # LM-head weight dtype (tt_transformers' Transformer(dtype=...) reaches the LM head only; the embedding is
        # always bf16 and the decoder groups follow the policy above).
        self.weight_dtype = _TT_DTYPE[DTYPE_POLICIES[dtype_policy]["lm_head"]]
        self.num_layers_override = num_layers

        t0 = time.time()
        self.args = ModelArgs(
            mesh_device,
            instruct=False,
            dummy_weights=False,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=lambda model_args: decoders_precision_for(
                dtype_policy, model_args.n_layers, model_args.model_name
            ),
            cache_hf=False,
        )
        assert self.args.dim == LLM_HIDDEN, self.args.dim
        if num_layers is not None:
            # Reduced-layer variant for profiling only (tt-perf-report on one decoder layer + norm + LM head);
            # numerically meaningless, never used by the pipeline.
            assert 1 <= num_layers <= self.args.n_layers, (num_layers, self.args.n_layers)
            self.args.n_layers = num_layers
        # bf16 logits: the stock bfp8 LM-head output is too coarse for CFG over the 16384 semantic codes.
        self.args.lm_head_dtype = ttnn.bfloat16
        self.vocab_size = self.args.vocab_size
        self.hidden_size = self.args.dim

        blocks_per_user = num_blocks_in_seq(max_seq_len, block_size)
        # Chunked prefill pads the prompt to a power of two and fills whole 4096-token chunks, so a
        # prompt longer than the largest power of two <= max_seq_len (8192 here) touches cache
        # positions beyond max_seq_len. Those padded positions go to shared scratch blocks (only
        # one user prefills at a time and causal attention never reads them from a real position)
        # instead of spilling into the next user's blocks. 64 blocks = 2048 tokens for 10240.
        chunk = self.args.max_prefill_chunk_size
        max_chunk_end = -(-max_seq_len // chunk) * chunk  # end of the chunk holding position max_seq_len-1
        scratch_blocks = max(0, num_blocks_in_seq(max_chunk_end, block_size) - blocks_per_user)
        self.paged_attention_config = PagedAttentionConfig(
            block_size=block_size, max_num_blocks=max_batch_size * blocks_per_user + scratch_blocks
        )
        # Identity page table: user u owns blocks [u*blocks_per_user, (u+1)*blocks_per_user);
        # blocks >= max_batch_size*blocks_per_user are the shared prefill scratch.
        self.page_table = torch.arange(max_batch_size * blocks_per_user, dtype=torch.int32).reshape(
            max_batch_size, blocks_per_user
        )
        self.scratch_blocks = torch.arange(
            max_batch_size * blocks_per_user, self.paged_attention_config.max_num_blocks, dtype=torch.int32
        )
        self.page_table_tt = ttnn.from_torch(
            self.page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        logger.info(f"MusicLLM: loading HF state dict from {self.args.CKPT_DIR}")
        state_dict = self.args.load_state_dict()
        logger.info(f"MusicLLM: state dict loaded in {time.time() - t0:.0f}s; building device model")
        t1 = time.time()
        self.model = MusicTransformer(
            args=self.args,
            mesh_device=mesh_device,
            dtype=self.weight_dtype,
            state_dict=state_dict,
            weight_cache_path=self.args.weight_cache_path(self.weight_dtype),
            paged_attention_config=self.paged_attention_config,
        )
        del state_dict
        logger.info(f"MusicLLM: device model built in {time.time() - t1:.0f}s")
        self.kv_cache = [layer.attention.layer_past for layer in self.model.layers]
        if DTYPE_POLICIES[dtype_policy].get("lm_head_fidelity") == "lofi":
            # tt_transformers' LMHead hard-codes HiFi2; the LM head is the largest single decode matmul group.
            self.model.lm_head.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
            )

        # Per-slot logical prompt lengths (bookkeeping for callers; the KV cache itself is positional).
        self.prefill_lens: List[Optional[int]] = [None] * max_batch_size

        # Traced decode state (allocated lazily on the first decode call).
        self._dec_x: Optional[ttnn.Tensor] = None
        self._dec_pos: Optional[ttnn.Tensor] = None
        self._dec_rot: Optional[ttnn.Tensor] = None
        self._dec_pos_shadow: Optional[torch.Tensor] = None  # what the device position tensor holds now
        self._trace_id = None
        self._trace_out: Optional[Tuple[ttnn.Tensor, ttnn.Tensor, Optional[ttnn.Tensor]]] = None
        self.logits_window: Optional[Tuple[int, int]] = None
        self._win_tiles: Optional[Tuple[int, int]] = None  # tile-aligned [start, end) actually sliced
        if logits_window is not None:
            self.set_logits_window(*logits_window)
        self.decode_stats = {"trace_captures": 0, "trace_replays": 0, "input_refreshes": 0, "position_refreshes": 0}
        # Optional host callback run after every prefill forward (one row, or one chunk of a row).
        # The perf harness uses it to drain the device profiler between ~1000-op chunks; None = off.
        self.after_prefill_chunk = None

    # ------------------------------------------------------------------ dtype report
    def dtype_report(self) -> dict:
        """The per-tensor-group dtypes actually configured (for the work log / context contract)."""
        opt = self.args.decoders_optimizations
        groups = {g.value: str(opt.get_tensor_dtype(decoder_id=0, tensor=g)) for g in TensorGroup}
        return {
            "policy": self.dtype_policy,
            "decoder_tensor_groups": groups,
            "embedding_weight": "bfloat16",
            "norm_weights": "bfloat16",
            "lm_head_weight": str(self.weight_dtype),
            "groups": DTYPE_POLICIES[self.dtype_policy]["groups"],
            "weight_cache_dir": os.environ.get("TT_CACHE_PATH"),
            "lm_head_output": str(self.args.lm_head_dtype),
            "rope_tables": "bfloat16",
            "activations": "bfloat16 (ACTIVATION group unset -> input dtype)",
            "kv_cache_tensor_dtype": str(self.kv_cache[0][0].dtype),
            "math_fidelity": opt.decoder_optimizations[0]._names["OpFidelity"],
        }

    # ------------------------------------------------------------------ embeddings
    def embed_tokens(self, token_ids: torch.Tensor) -> ttnn.Tensor:
        """``[B, S]`` token ids -> ``[1, B, S, 4096]`` bf16 tile-layout embeddings on device."""
        assert token_ids.dim() == 2, token_ids.shape
        b, s = token_ids.shape
        ids = ttnn.from_torch(
            token_ids.to(torch.int32).reshape(1, 1, 1, b * s),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        emb = self.model.embd(ids, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        emb = ttnn.reshape(ttnn.unsqueeze_to_4D(emb), (1, b, s, self.hidden_size))
        return emb

    def embed_frame(self, semantic_codes, residual_embeds_sum) -> ttnn.Tensor:
        """The AR feedback embedding of one frame for the decode step, as a decode input tensor.

        ``(embed_tokens(semantic + AUDIO_CODE_OFFSET) + residual_embeds_sum) * NUM_CODEBOOKS**-0.5``,
        i.e. diffusers ``_embed_audio_frame`` with the depth decoder's 7-way residual embedding sum
        supplied by the caller (stage 03/04 computes it on device from ``audio_embeddings``).

        Args:
            semantic_codes: ``[B]`` semantic codes (0..16383) as a torch tensor.
            residual_embeds_sum: ``[B, 4096]`` torch tensor, or a device ttnn tensor of shape
                ``[1, 1, 32, 4096]`` (bf16, tile layout, rows >= B are ignored).
        Returns:
            ``[1, 1, 32, 4096]`` bf16 tile-layout DRAM tensor accepted by :meth:`decode`.
        """
        codes = torch.as_tensor(semantic_codes).reshape(-1)
        assert codes.numel() <= self.max_batch_size, codes.shape
        ids = torch.zeros(TILE, dtype=torch.int32)
        ids[: codes.numel()] = codes.to(torch.int32) + AUDIO_CODE_OFFSET
        tt_ids = ttnn.from_torch(
            ids.reshape(1, 1, 1, TILE),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        sem = self.model.embd(tt_ids, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sem = ttnn.reshape(ttnn.unsqueeze_to_4D(sem), (1, 1, TILE, self.hidden_size))
        if isinstance(residual_embeds_sum, torch.Tensor):
            res = self._decode_input_host(residual_embeds_sum)
            res = ttnn.to_device(res, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            res = residual_embeds_sum
            assert list(res.shape) == [1, 1, TILE, self.hidden_size], list(res.shape)
            if res.layout != ttnn.TILE_LAYOUT:
                res = ttnn.to_layout(res, ttnn.TILE_LAYOUT)
        out = ttnn.add(sem, res, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.multiply(out, NUM_CODEBOOKS**-0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out

    # ------------------------------------------------------------------ prefill
    def prefill(self, inputs_embeds, *, user_id: Union[int, Sequence[int], None] = None):
        """Prefill one or more batch rows from embeddings and fill their KV cache.

        Args:
            inputs_embeds: ``[B, S, 4096]`` (or ``[S, 4096]`` for one row) torch tensor, or a
                ttnn tensor of shape ``[1, B, S, 4096]`` (bf16 tile layout, as :meth:`embed_tokens`
                returns). Any logical ``1 <= S <= max_seq_len``; padding / chunking is internal
                (padded positions beyond ``max_seq_len`` land in shared scratch blocks).
            user_id: the cache slot(s) the rows go to. ``None`` = rows ``0..B-1``.
        Returns:
            ``(hidden, logits)`` torch fp32 tensors of shape ``[B, 4096]`` (final-norm output at the
            last prompt position) and ``[B, vocab_size]``.
        """
        if isinstance(inputs_embeds, ttnn.Tensor):
            host = ttnn.to_torch(inputs_embeds)
            assert host.dim() == 4 and host.shape[0] == 1, host.shape
            inputs_embeds = host[0]
        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        assert inputs_embeds.dim() == 3 and inputs_embeds.shape[-1] == self.hidden_size, inputs_embeds.shape
        b, s, _ = inputs_embeds.shape
        if user_id is None:
            slots = list(range(b))
        elif isinstance(user_id, int):
            slots = [user_id + i for i in range(b)]
        else:
            slots = list(user_id)
        assert len(slots) == b and all(0 <= u < self.max_batch_size for u in slots), slots
        if not 1 <= s <= self.max_seq_len:
            raise ValueError(f"prompt length {s} outside 1..{self.max_seq_len}")

        hiddens, logits = [], []
        for row, slot in zip(inputs_embeds, slots):
            h, l = self._prefill_one(row, slot)
            hiddens.append(h)
            logits.append(l)
            self.prefill_lens[slot] = s
        return torch.stack(hiddens, 0), torch.stack(logits, 0)

    def _prefill_one(self, embeds: torch.Tensor, slot: int):
        s = embeds.shape[0]
        last_idx = s - 1
        s_pad = get_padded_prefill_len(s)
        x = torch.zeros(1, 1, s_pad, self.hidden_size, dtype=torch.bfloat16)
        x[0, 0, :s] = embeds.to(torch.bfloat16)

        chunk_max = self.args.max_prefill_chunk_size
        blocks_per_user = self.page_table.shape[1]
        if s_pad <= chunk_max:
            # Single chunk: the full [B, blocks] page table, this slot's row is selected by user_id.
            x_tt = self._prefill_input(x)
            rot = self._prefill_rot_mats(0, s_pad)
            hidden, logits = self.model.forward(
                x_tt,
                None,
                rot_mats_global=rot,
                user_id=slot,
                mode=Mode.PREFILL,
                page_table=self.page_table_tt,
                get_last_token=(last_idx // TILE) * TILE,
            )
            result = self._read_last_row(hidden, logits, last_idx % TILE)
            if self.after_prefill_chunk is not None:
                self.after_prefill_chunk()
            return result

        # Chunked prefill (same constraints as tt_transformers Generator.prefill_forward_single_user_text):
        # the page table must be this user's single row (SDPA checks its batch dim against the input's),
        # padded to the number of blocks of the padded prompt, and user_id must be 0 within the chunk.
        chunk_size = get_max_prefill_chunk_size(s_pad, chunk_max)
        last_chunk_start = (last_idx // chunk_size) * chunk_size
        needed_blocks = num_blocks_in_seq(last_chunk_start + chunk_size, self.block_size)
        page_table_user = self.page_table[slot : slot + 1, :]
        if needed_blocks > blocks_per_user:
            extra = needed_blocks - blocks_per_user
            assert extra <= self.scratch_blocks.numel(), (needed_blocks, blocks_per_user, self.scratch_blocks.numel())
            page_table_user = torch.cat([page_table_user, self.scratch_blocks[:extra].unsqueeze(0)], dim=1)
        page_table_user_tt = ttnn.from_torch(
            page_table_user,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        result = None
        for chunk_start in range(0, s_pad, chunk_size):
            chunk_end = chunk_start + chunk_size
            x_tt = self._prefill_input(x[:, :, chunk_start:chunk_end])
            rot = self._prefill_rot_mats(chunk_start, chunk_end)
            chunk_pt = page_table_user[:, chunk_start // self.block_size : chunk_end // self.block_size]
            chunk_pt_tt = ttnn.from_torch(
                chunk_pt,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            is_last = chunk_start == last_chunk_start
            out = self.model.forward(
                x_tt,
                None,
                rot_mats_global=rot,
                user_id=0,
                mode=Mode.PREFILL,
                page_table=page_table_user_tt,
                chunk_page_table=chunk_pt_tt,
                chunk_start_idx=chunk_start,
                get_last_token=((last_idx - chunk_start) // TILE) * TILE if is_last else -1,
            )
            if is_last:
                hidden, logits = out
                result = self._read_last_row(hidden, logits, (last_idx - chunk_start) % TILE)
            else:
                ttnn.deallocate(out)
            if self.after_prefill_chunk is not None:
                self.after_prefill_chunk()
            if is_last:
                break
        assert result is not None
        return result

    def _prefill_input(self, x: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            x,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _prefill_rot_mats(self, start: int, end: int):
        """cos/sin rows for positions [start, end); rows past max_seq_len (padding only) are zero-filled.

        A padded chunk can reach past the context (10240 -> padded 16384, third chunk ends at
        12288). Those positions only hold padding tokens that no real token attends to, so their
        rotation is irrelevant; tt_transformers zero-pads them the same way.
        """
        rs = self.model.rope_setup
        mat_len = rs.cos_matrix_prefill.shape[2]
        assert start < mat_len, (start, mat_len)
        stop = min(end, mat_len)
        cos = rs.cos_matrix_prefill[:, :, start:stop, :]
        sin = rs.sin_matrix_prefill[:, :, start:stop, :]
        if end > mat_len:
            padding = [(0, 0), (0, 0), (0, end - mat_len), (0, 0)]
            cos = ttnn.pad(cos, padding=padding, value=0.0)
            sin = ttnn.pad(sin, padding=padding, value=0.0)
        return [cos, sin]

    def _read_last_row(self, hidden: ttnn.Tensor, logits: ttnn.Tensor, row: int):
        h = ttnn.to_torch(hidden).float()[0, 0, row, : self.hidden_size]
        l = ttnn.to_torch(logits).float()[0, 0, row, : self.vocab_size]
        ttnn.deallocate(hidden)
        ttnn.deallocate(logits)
        return h, l

    # ------------------------------------------------------------------ decode
    def _decode_input_host(self, x: torch.Tensor) -> ttnn.Tensor:
        """``[B, 4096]`` / ``[B, 1, 4096]`` torch -> host ttnn ``[1, 1, 32, 4096]`` bf16 tile tensor."""
        x = x.reshape(-1, self.hidden_size)
        assert x.shape[0] <= self.max_batch_size, x.shape
        padded = torch.zeros(1, 1, TILE, self.hidden_size, dtype=torch.bfloat16)
        padded[0, 0, : x.shape[0]] = x.to(torch.bfloat16)
        return ttnn.from_torch(
            padded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _positions(self, current_pos) -> torch.Tensor:
        pos = torch.as_tensor(current_pos, dtype=torch.int64).reshape(-1)
        if pos.numel() == 1:
            pos = pos.repeat(self.max_batch_size)
        assert pos.numel() == self.max_batch_size, pos.shape
        assert int(pos.min()) >= 0 and int(pos.max()) < self.max_seq_len, pos
        return pos

    def _alloc_decode_inputs(self, x_host: ttnn.Tensor, pos: torch.Tensor):
        self._dec_x = ttnn.to_device(x_host, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._dec_pos = ttnn.from_torch(
            pos.to(torch.int32),
            device=self.mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        rot_host = self.model.rope_setup.get_rot_idxs(pos, on_host=True)
        self._dec_rot = ttnn.to_device(rot_host, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._dec_pos_shadow = pos.clone()

    def _write_positions(self, pos: torch.Tensor):
        pos_host = ttnn.from_torch(
            pos.to(torch.int32), dtype=ttnn.int32, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)
        )
        ttnn.copy_host_to_device_tensor(pos_host, self._dec_pos)
        rot_host = self.model.rope_setup.get_rot_idxs(pos, on_host=True)
        ttnn.copy_host_to_device_tensor(rot_host, self._dec_rot)
        self._dec_pos_shadow = pos.clone()

    def _write_input(self, inputs_embeds):
        if isinstance(inputs_embeds, torch.Tensor):
            ttnn.copy_host_to_device_tensor(self._decode_input_host(inputs_embeds), self._dec_x)
        else:
            x = inputs_embeds
            assert list(x.shape) == [1, 1, TILE, self.hidden_size], list(x.shape)
            assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT, (x.dtype, x.layout)
            if x.storage_type() == ttnn.StorageType.HOST:
                ttnn.copy_host_to_device_tensor(x, self._dec_x)
            else:
                ttnn.copy(x, self._dec_x)
        self.decode_stats["input_refreshes"] += 1

    def set_logits_window(self, start: int, end: int):
        """Restrict the untilized logits read-back to the logical columns ``[start, end)`` (before the first decode)."""
        assert self._trace_id is None, "the decode trace is already captured; set the window before the first decode"
        assert 0 <= start < end <= self.vocab_size, (start, end, self.vocab_size)
        self.logits_window = (int(start), int(end))
        self._win_tiles = ((start // TILE) * TILE, -(-end // TILE) * TILE)

    def prepare_decode_inputs(self):
        """Allocate the persistent decode input tensors now (position 0) instead of on the first decode.

        Call this before other components capture their own traces (stage 04: the depth decoder's
        step traces): every device buffer that lives across a trace replay has to exist before that
        trace is captured, otherwise it may be placed where the trace's intermediates were and be
        overwritten by a replay. The trace itself is still captured lazily on the first decode.
        """
        if self._dec_x is None:
            pos = torch.zeros(self.max_batch_size, dtype=torch.int64)
            self._alloc_decode_inputs(self._decode_input_host(torch.zeros(self.max_batch_size, self.hidden_size)), pos)

    def _decode_graph(self):
        """Device-only decode step over the persistent inputs; this is what gets traced."""
        rot_mats = self.model.rope_setup.get_rot_mats(self._dec_rot)
        hidden, logits = self.model.forward(
            self._dec_x,
            self._dec_pos,
            rot_mats_global=rot_mats,
            mode=Mode.DECODE,
            page_table=self.page_table_tt,
        )
        if self._win_tiles is None:
            logits = ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            window = None
        else:
            # Only the window is untilized and read back; the full logits stay tiled on device (still
            # readable through decode(read_back=True), just slower on the host side).
            ws, we = self._win_tiles
            window = ttnn.slice(logits, [0, 0, 0, ws], [1, 1, TILE, we], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            window = ttnn.untilize(window, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Advance the device-owned positions so consecutive steps need no host refresh.
        ttnn.plus_one(self._dec_pos, skip_negative_entries=True)
        ttnn.plus_one(self._dec_rot)
        return hidden, logits, window

    def _capture_decode_trace(self, pos: torch.Tensor):
        # Compile run with the exact shapes/args of the capture; it also mutates the positions.
        h, l, w = self._decode_graph()
        ttnn.deallocate(h)
        ttnn.deallocate(l)
        if w is not None:
            ttnn.deallocate(w)
        ttnn.synchronize_device(self.mesh_device)
        assert self._dec_x.is_allocated(), "the persistent decode input was deallocated by the graph"
        # Restore the intended capture state (the compile run advanced the positions by one).
        self._write_positions(pos)
        ttnn.synchronize_device(self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._trace_out = self._decode_graph()
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        self._trace_id = trace_id
        self.decode_stats["trace_captures"] += 1
        logger.info("MusicLLM: decode trace captured")

    def decode(self, inputs_embeds, current_pos, *, read_back: bool = True):
        """One traced decode step for all batch rows from pre-embedded input.

        Args:
            inputs_embeds: ``[B, 4096]`` / ``[B, 1, 4096]`` torch tensor, or a ttnn tensor of shape
                ``[1, 1, 32, 4096]`` bf16 tile layout (host or device), e.g. from :meth:`embed_frame`.
            current_pos: the position this input occupies for each row: ``[B]`` ints (or one int
                for all rows). The KV cache is written at that position and attention covers
                ``0..current_pos``.
            read_back: return torch tensors (default). With ``False`` the device output tensors
                ``(hidden [1,1,32,4096] tile, logits [1,1,32,padded_vocab] row-major)`` of the trace
                are returned; they are overwritten by the next step.
        Returns:
            ``(hidden, logits)`` torch fp32 ``[B, 4096]`` and ``[B, vocab_size]`` (or device tensors).
        """
        pos = self._positions(current_pos)
        if self._dec_x is None:
            self._alloc_decode_inputs(self._decode_input_host(torch.zeros(self.max_batch_size, self.hidden_size)), pos)
        if not torch.equal(pos, self._dec_pos_shadow):
            self._write_positions(pos)
            self.decode_stats["position_refreshes"] += 1
        self._write_input(inputs_embeds)
        if self._trace_id is None:
            self._capture_decode_trace(pos)
        ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)
        self.decode_stats["trace_replays"] += 1
        self._dec_pos_shadow = pos + 1  # the graph advanced the device positions
        hidden, logits, _ = self._trace_out
        if not read_back:
            return hidden, logits
        h = ttnn.to_torch(hidden).float()[0, 0, : self.max_batch_size, : self.hidden_size]
        l = ttnn.to_torch(logits).float()[0, 0, : self.max_batch_size, : self.vocab_size]
        return h, l

    def decode_windowed(self, inputs_embeds, current_pos):
        """One traced decode step that reads back only the logits window set by :meth:`set_logits_window`.

        Returns ``(hidden_device, hidden, window)``: the persistent device hidden tensor
        (``[1, 1, 32, 4096]`` bf16 tile, overwritten by the next step; the depth decoder seeds from
        it without a host round-trip), the host fp32 hidden ``[B, 4096]`` and the host fp32 logits
        ``[B, end - start]`` of the window (column ``i`` = vocabulary id ``start + i``).
        """
        assert self.logits_window is not None, "set_logits_window() first"
        hidden, _ = self.decode(inputs_embeds, current_pos, read_back=False)
        window = self._trace_out[2]
        start, end = self.logits_window
        off = start - self._win_tiles[0]
        h = ttnn.to_torch(hidden).float()[0, 0, : self.max_batch_size, : self.hidden_size]
        w = ttnn.to_torch(window).float()[0, 0, : self.max_batch_size, off : off + (end - start)]
        return hidden, h, w

    def decode_replay_only(self):
        """Replay the captured decode trace once more without touching any input (perf harness)."""
        assert self._trace_id is not None, "capture the trace with a decode() call first"
        ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)
        self.decode_stats["trace_replays"] += 1
        self._dec_pos_shadow = self._dec_pos_shadow + 1

    # ------------------------------------------------------------------ cache
    def reset_cache(self):
        """Start a new song: forget the prompt lengths and the device-side position shadow.

        The paged KV cache is positional (prefill fills ``0..S-1`` and decode writes exactly at
        ``current_pos``; attention never reads beyond ``current_pos``), so stale entries from a
        previous song are never observed and zeroing 36 layers x 2 x 168 MB of cache is not needed.
        The decode trace stays valid: it only references the persistent input tensors.
        """
        self.prefill_lens = [None] * self.max_batch_size
        if self._dec_pos_shadow is not None:
            # Force the next decode() to write its positions; no assumption about continuity.
            self._dec_pos_shadow = torch.full((self.max_batch_size,), -1, dtype=torch.int64)

    def kv_cache_bytes(self) -> dict:
        """KV-cache footprint from the allocated tensors (per layer and total)."""
        k, v = self.kv_cache[0]
        per_layer = 0
        for t in (k, v):
            per_layer += t.volume() * _bytes_per_element(t.dtype)
        return {
            "layers": len(self.kv_cache),
            "block_size": self.block_size,
            "max_num_blocks": self.paged_attention_config.max_num_blocks,
            "dtype": str(k.dtype),
            "shape_per_tensor": list(k.shape),
            "bytes_per_layer": per_layer,
            "bytes_total": per_layer * len(self.kv_cache),
            "bytes_per_token_per_layer": per_layer // (self.paged_attention_config.max_num_blocks * self.block_size),
        }

    def release(self):
        if self._trace_id is not None:
            ttnn.release_trace(self.mesh_device, self._trace_id)
            self._trace_id = None


def _bytes_per_element(dtype) -> float:
    if dtype == ttnn.bfloat16:
        return 2.0
    if dtype == ttnn.bfloat8_b:
        return 1088 / 1024  # 32x32 tile: 1024 mantissa bytes + 64 shared exponents
    if dtype == ttnn.bfloat4_b:
        return 576 / 1024
    if dtype in (ttnn.float32, ttnn.int32, ttnn.uint32):
        return 4.0
    raise ValueError(f"unknown dtype {dtype}")
