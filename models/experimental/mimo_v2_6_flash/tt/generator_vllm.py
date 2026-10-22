# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM serving adapter for MiMo-V2.6-Flash on a Blackhole Galaxy (vllm-tt-plugin, tenstorrent/vllm-tt-plugin#216).

Data parallelism: ``--data-parallel-size 4`` runs as four in-process TT lanes
(``model_capabilities["folds_dp_into_lanes"]``): one engine drives MiMoGalaxyGenerator's four TP=8 replicas, one
per row of the 4x8 mesh. Lane ``l`` owns the plugin's slots [l * per_lane, (l + 1) * per_lane), which are decode
slots 0 .. per_lane - 1 of replica ``l``. Each lane has its own vLLM scheduler and block pool, so block ids are
lane-local and each replica has its own KV storage.

KV cache: vLLM owns the blocks. ``get_kv_cache_spec`` reports a FullAttentionSpec for the global layers and a
SlidingWindowSpec (window 128) for the SWA layers, both K 192 / V 128 at block size 64. vLLM's hybrid KV manager
splits them into KV cache groups of equal size (for 9 global and 39 SWA layers: one global group and five SWA
groups of up to 9 layers), every group with its own block table and all of them drawing block ids from one pool.
The model gets one page table per group (TtMiMoModel.setup_vllm_kv_cache). A SWA group's table is indexed by
absolute position with the blocks no window can read set to vLLM's null block 0, the layout the model's own
kv.SwaBlockTable keeps.

Storage follows vLLM's sharing rule: the i-th layer of every group shares one buffer (``tensor_idx``), which is
safe because a block id belongs to one group at a time. On a chip every layer's K cache is
[num_blocks, 1, 64, 192] and every layer's V cache [num_blocks, 1, 64, 128] (attention.py keeps V's real dims on
the SWA layers too), each shared that way: 9 K and 9 V buffers for 9 global and 39 SWA layers. The global layers
keep both copies of their KV head on a chip pair (setup_vllm_kv_cache): a block can be shared by requests in slots
of either parity.

Sampling: prefill returns last-token logits [B, 1, vocab] for the host to sample. Decode replays one of two traced
steps per replica (captured in ``warmup_model_decode``) that also computes each slot's token and writes it, with the
slot's position plus one, into the trace's own inputs (TtMiMoModel._feed_back): the greedy step (GreedyHead), or,
while a lane holds a row with temperature > 0, the sampled step (DeviceSampler: temperature, top_k up to
``max_device_top_k``, top_p; its random stream is on device, seeded with the parameters, a seeded row's by (seed,
position)). With ``sample_on_device_mode`` the plugin samples on device unless a row asks for what the sampler lacks
(a top_k past its candidates, pure temperature sampling without top_p or top_k, penalties, logprobs or structured
output send the whole batch to host sampling, which reads the logit rows) and overlaps the steps (``decode_input_update_contract`` 1,
``supports_async_decode``): a steady step copies nothing (the trace reads its RoPE
cos / sin from device tables at its own positions, at every context length; with MIMO_DEVICE_ROPE=0 it copies those
of the adapter's own position counter), and its tokens are read back without blocking. Prefill runs
eagerly in chunks of up to 4096 tokens, between the replays. A program first compiled after the capture would keep buffers the replays overwrite, so
``warmup_model_prefill`` compiles every prefill program a served chunk can run first: they key only on the chunk's
padded length and the alignment of its start (attention.py), a bounded set.

Speculative decoding (MIMO_SPEC_DECODE=mtp; the plugin's model-owned drafter contract, docs/SPEC_DECODE_CONTRACT.md,
with ``--speculative-config '{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter",
"num_speculative_tokens": 3}'``): the MTP layers draft up to MAX_DRAFTS tokens for a lane that ran one request in its
last step (MIMO_MTP_LAYERS=3, the default: draft i + 1 from MTP layer i; 1: layer MIMO_MTP_LAYER drafts every
token), and a verify step runs that request in every decode slot of its replica (reference/spec_decode.py). Lanes
with several requests decline to draft and decode as usual; a verify block that still carries drafts beside other
rows (a request joined after the proposal) is packed into passes of SPEC_SLOTS slots. A verify answers the
plugin's argmax ids when every request of the step is greedy. A step with a sampled request is accepted on device
(``fused_sample``, tt/verify_sampler.py) when every request's sampling is one the device sampler serves (the plugin
decides from ``max_device_top_k`` and ``supports_device_top_p_without_top_k``, and sends no penalties, logprobs,
grammar or token filters there): the plugin sends the requests' temperature / top_k / top_p and the uniforms it drew
from their generators, the device runs the plugin's accept walk on each slot's candidates, and the step returns each
request's committed tokens and count. Otherwise the verify returns its logits and the plugin walks acceptance on the
host with each request's own sampling (temperature, top-p, penalties, ...). Either way sampled requests speculate
losslessly. A verify and a draft each copy one packed int32 tensor of their tokens and positions (plus the RoPE cos /
sin with MIMO_DEVICE_ROPE=0); their traces derive every other input from it on device
(tt/spec_inputs.py). A step that verifies nothing decodes as above, a sampled request's on the sampled decode trace.

The draft is chained behind the verify (tt/spec_chain.py, reference/spec_chain.py): an argmax_ids verify accepts on
device too, every request greedy (its committed tokens are the plugin's greedy walk over the ids), and on a lane that
speculates the host enqueues, without waiting, the verify trace, the accept trace (SampleVerify, then the chain step,
which writes the MTP draft's inputs from the accepted count and committed tokens on device) and the MTP draft trace,
then reads one 320-byte result page per pass: the argmax ids or the committed tokens and count, and the drafts.
propose_draft_tokens returns those drafts when the plugin committed what the device did; it drafts from the host plan
after a plain decode step, a logits verify, without the decode RoPE tables (MIMO_DEVICE_ROPE=0), or when the commits differ.

The MTP rings only hold what the lane's drafts wrote. With MIMO_MTP_BACKFILL (default on, with the decode RoPE tables)
each decode slot that may speculate keeps a history of the target's final normed hidden (tt/mtp_backfill.py: written
by every decode step, verify pass and prefill chunk), and a draft whose request's rows the rings do not continue
(``mtp_ring``: after the request's prompt, after the lane served several requests) is preceded by the backfill trace,
which fills every MTP layer's ring from that history: its drafts are those of an MTP that saw every position.
"""

import math
import os
import time

import numpy as np
import torch
from loguru import logger

import ttnn

from ..reference.checkpoint import MiMoCheckpoint, default_model_path
from ..reference.config import MiMoConfig
from ..reference.sample_verify import commit_pass, pass_inputs
from ..reference.spec_chain import chain_row, chained_draft
from ..reference.spec_decode import MAX_DRAFTS, SPEC_SLOTS, DraftSource, draft_source, plan_mtp, plan_verify_passes
from .attention import page_table_width
from .device_sampler import CANDIDATES_PER_CHIP, stream_states
from .generator import MiMoGalaxyGenerator
from .model import prefill_padded_len, prefill_warmup_batches, prefill_warmup_chunks

TP = 8
BLOCK_SIZE = 64
# Per-chip bytes per token of the vLLM KV layout above for MiMo-V2.6-Flash (9 global, 39 SWA layers, so 9 shared K
# buffers and 9 shared V buffers), BFP8: (9 * 192 + 9 * 128) values * 1088 / 1024 bytes = 3060.
KV_BYTES_PER_TOKEN = (9 * 192 + 9 * 128) * 1088 / 1024
# Per-chip DRAM given to the KV pool when MIMO_KV_GB_PER_CHIP is unset (weights take 23.7 of 34 GB per chip).
DEFAULT_KV_GB_PER_CHIP = 6.0
# Blocks vLLM's sliding-window groups can hold per request while a prefill chunk of up to 4096 tokens is in flight
# (SlidingWindowSpec.max_admission_blocks_per_request), in tokens: 5 SWA groups x (127 + 4096 tokens + 1 block).
SWA_TOKENS_PER_USER = 5 * (math.ceil((127 + 4096) / BLOCK_SIZE) + 1) * BLOCK_SIZE
# MIMO_VLLM_TIMING=1 logs each prefill's host enqueue and total time, decode's host time every 128 steps, and every 128
# speculative steps where a step's time goes (verify, MTP draft, and the host gap between them; _spec_timing).
TIMING = os.environ.get("MIMO_VLLM_TIMING") == "1"
# MIMO_SAMPLER_CHECK=1 (debug): each device-sampled decode step blocks, recomputes every row's draw on the host from the
# step's logit rows and the row's uniform (DeviceSampler.check) and logs the counts every 64 steps.
SAMPLER_CHECK = os.environ.get("MIMO_SAMPLER_CHECK") == "1"
# MIMO_PREFILL_BATCH=0: prefill every row on its own (TtMiMoModel.prefill_enqueue_batch is not used).
PREFILL_BATCH = os.environ.get("MIMO_PREFILL_BATCH", "1") == "1"
# MIMO_PREFILL_TRACE=0: run prefill batches eagerly instead of replaying their traces.
PREFILL_TRACE = os.environ.get("MIMO_PREFILL_TRACE", "1") == "1"
# Decode, verify and MTP draft steps read their RoPE cos / sin from device tables of every position
# (TtMiMoModel.setup_decode_rope_tables); MIMO_DEVICE_ROPE=0 uploads them every step instead.
DEVICE_ROPE = os.environ.get("MIMO_DEVICE_ROPE", "1") == "1"
# Up to this context the tables hold whole 192-wide rows (201 MB per chip at 128K); past it only the 64 rotated dims
# (``compact``: 537 MB per chip at 1M instead of 1.61 GB), each read appending the constant 1 / 0 of the other dims.
ROPE_TABLE_FULL_WIDTH_MAX_LEN = 131072
# MIMO_SPEC_DECODE=mtp: load the MTP layers and serve the plugin's model-owned drafter contract (module docstring).
# MIMO_MTP_LAYERS=3 (default): draft i + 1 from MTP layer i (vLLM 0.31's MultiModuleMTPSpeculator, which the GPU
# references run with the num_nextn_predict_layers patch; needs the decode RoPE tables); 1: MTP layer MIMO_MTP_LAYER
# (default 0) drafts every token, feeding itself its own output (vLLM's MTP proposer, upstream vLLM for MiMo).
SPEC_DECODE = os.environ.get("MIMO_SPEC_DECODE", "")
MTP_LAYERS = int(os.environ.get("MIMO_MTP_LAYERS", "3"))
MTP_LAYER = int(os.environ.get("MIMO_MTP_LAYER", "0"))
# MIMO_MTP_BACKFILL=0: no target-hidden history, no ring backfill (the rings start empty for each prompt and keep the
# gaps of the steps the lane did not draft); it needs the decode RoPE tables.
MTP_BACKFILL = os.environ.get("MIMO_MTP_BACKFILL", "1") == "1"
# MIMO_SPEC_CHAIN=0: no draft chained behind a verify (every draft host-planned after the plugin's walk, and an
# argmax_ids verify reads its greedy tokens without an accept step), the flow before tt/spec_chain.py, for comparison.
SPEC_CHAIN = os.environ.get("MIMO_SPEC_CHAIN", "1") == "1"
# MIMO_SPEC_CHAIN_CHECK=1 (debug): every draft returned from the chain is drafted again from the host plan (the same
# MTP inputs, rewriting the same ring entries) and compared; mismatches are logged (``chain_stats``).
SPEC_CHAIN_CHECK = os.environ.get("MIMO_SPEC_CHAIN_CHECK") == "1"


def _model_path(hf_config):
    path = getattr(hf_config, "_name_or_path", None)
    if path and os.path.exists(os.path.join(path, "config.json")):
        return path
    return default_model_path()


def _cache_dir(model_path):
    """The converted TTNN weights, as the tests keep them (tests/common.py cache_dir("model_bfp4experts"))."""
    root = os.environ.get("MIMO_TT_CACHE", "/home/zni/models/mimo-tt-cache")
    return os.path.join(root, os.path.basename(os.path.normpath(model_path)), "model_bfp4experts")


class MiMoV2ForCausalLM:
    model_capabilities = {
        "supports_chunked_prefill": True,
        "supports_prefix_caching": True,
        "folds_dp_into_lanes": True,
        "fabric_config": {"config": ttnn.FabricConfig.FABRIC_1D_RING},
        # Device sampling: the decode trace's greedy token or DeviceSampler's draw (module docstring). Its candidates
        # are each chip's CANDIDATES_PER_CHIP largest logits, which bound top_k and stand in for the vocabulary when
        # a top_p < 1 row sets no top_k; a row with neither (pure temperature sampling) samples on host.
        "supports_sample_on_device": True,
        "supports_async_decode": True,
        "max_device_top_k": TP * CANDIDATES_PER_CHIP,
        "supports_device_top_p_without_top_k": True,
        "supports_device_penalties": False,
        "supports_device_logprobs": False,
        # The MTP drafter (spec_plan, propose_draft_tokens): it keeps the target hidden on device, so it needs no
        # hidden feed. A verify, with the draft chained behind it, is read back before the step returns (no async
        # speculative decode).
        "supports_spec_decode": True,
        "spec_requirements": ("device_propose",),
        "output_tokens_per_step": 1,
        "supports_async_spec_decode": False,
    }
    # The four reload commands of the plugin's decode reload contract (docs/DECODE_RELOAD_CONTRACT.md).
    decode_input_update_contract = 1
    # Emits SlidingWindowSpec groups: the plugin's KV budget adds its sliding-window headroom.
    _HYBRID_KV_CACHE_GROUPS_ENABLED = True
    # (vllm_config, specs) recorded by get_kv_cache_spec, which the plugin calls in the worker process that later
    # allocates the KV cache. allocate_kv_cache_per_layer groups the specs with vLLM's own function, as the engine
    # did: vLLM resolves the KV cache layout the grouping reads only after collecting the specs.
    _kv_cache_spec_source = None
    # vLLM's KV cache group of each layer, when set directly (tests driving the adapter without vLLM).
    _layer_groups = None
    # Tokens vLLM schedules per step (--max-num-batched-tokens), which bounds a prefill chunk: None reads it from the
    # vLLM config get_kv_cache_spec recorded (set directly when driving the adapter without vLLM).
    prefill_token_budget = None

    def __init__(self, generator, max_seq_len):
        self.gen = generator
        self.models = generator.models
        self.dp = generator.dp
        self.per_lane = generator.users_per_replica
        self.max_seq_len = max_seq_len
        self.already_warmed_up_prefill = False
        self.prefill_programs_compiled = False
        self.layer_groups = None
        self.kv_caches = None
        self.decode_recaptures = 0  # decode traces captured again after a prefill compiled new programs
        # Per lane: int32 [max_batch] positions the next decode step reads (-1: idle slot), set by a reloading step
        # and advanced with every step, as the trace advances its own (a steady step reads no host positions).
        self.decode_positions = [None] * self.dp
        # Per lane: whether its rows' sampling parameters (set by a reloading step) include a sampled row, which
        # selects the sampled decode trace.
        self.lane_sampled = [False] * self.dp
        # Per lane: where the last step left its one request's committed tokens (reference.spec_decode.DraftSource),
        # None when that step ran no request or several; propose_draft_tokens drafts from it.
        self.spec_sources = [None] * self.dp
        # Per lane: the draft its last verify chained on device for that request (reference.spec_chain.ChainedDraft),
        # None when it chained none; propose_draft_tokens returns it when the plugin committed what the device did.
        self.chained = [None] * self.dp
        self.chain_check = SPEC_CHAIN_CHECK
        # Drafts returned from the chain and drafted from the host plan; with chain_check, chained drafts checked
        # against the host plan's and the mismatches; ring backfills.
        self.chain_stats = {"chained": 0, "host": 0, "checked": 0, "mismatched": 0, "backfills": 0}
        # Per lane: (lane slot, last row) of the request whose rows the MTP rings hold up to that row (the last draft's
        # B - 1), None after a prompt reset them. A draft for another request, or past a gap, backfills first.
        self.mtp_ring = [None] * self.dp
        # Per lane and slot: where the slot's last prefill chunk ended (a chunk starting there continues its request).
        self.prefill_ends = [{} for _ in range(self.dp)]

    # ------------------------------------------------------------------ construction and sizing
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None, **kwargs
    ):
        if optimizations not in (None, "performance"):
            raise ValueError(f"MiMo-V2.6-Flash serves one datatype policy; optimizations={optimizations!r}")
        dp = tt_data_parallel
        num_devices = mesh_device.get_num_devices()
        if num_devices != TP * dp:
            raise ValueError(
                f"MiMo runs one TP={TP} replica per lane: {dp} lanes need {TP * dp} chips, got {num_devices}"
            )
        if max_batch_size % dp:
            raise ValueError(f"max_batch_size {max_batch_size} does not split over {dp} lanes")
        if max_seq_len % BLOCK_SIZE:
            raise ValueError(f"max_model_len {max_seq_len} must be a multiple of the block size {BLOCK_SIZE}")
        path = _model_path(hf_config)
        cfg = MiMoConfig.from_model_path(path)
        num_layers = hf_config.num_hidden_layers
        if num_layers > cfg.num_hidden_layers:
            raise ValueError(f"num_hidden_layers {num_layers} > the checkpoint's {cfg.num_hidden_layers}")
        mtp_layers = None
        if SPEC_DECODE:
            if SPEC_DECODE != "mtp":
                raise ValueError(f"MIMO_SPEC_DECODE={SPEC_DECODE!r}: MiMo drafts with its MTP layer (mtp)")
            if max_batch_size // dp < SPEC_SLOTS:
                raise ValueError(
                    f"a verify runs one request in {SPEC_SLOTS} decode slots: launch with --max-num-seqs >= "
                    f"{SPEC_SLOTS} per lane (got {max_batch_size // dp})"
                )
            if MTP_LAYERS not in (1, MAX_DRAFTS):
                raise ValueError(f"MIMO_MTP_LAYERS={MTP_LAYERS}: 1, or {MAX_DRAFTS} (one MTP layer per draft)")
            if MTP_LAYERS > cfg.num_nextn_predict_layers:
                raise ValueError(f"MIMO_MTP_LAYERS={MTP_LAYERS}: the checkpoint has {cfg.num_nextn_predict_layers}")
            if MTP_LAYERS > 1 and not DEVICE_ROPE:
                raise ValueError("MIMO_MTP_LAYERS=3 plans its passes on device: it needs MIMO_DEVICE_ROPE=1")
            mtp_layers = (MTP_LAYER,) if MTP_LAYERS == 1 else tuple(range(MAX_DRAFTS))
            if MTP_BACKFILL and not DEVICE_ROPE:
                logger.warning("MiMo vLLM: MIMO_MTP_BACKFILL needs the decode RoPE tables (MIMO_DEVICE_ROPE=1): off")
        logger.info(f"MiMo vLLM: {dp} replicas x {max_batch_size // dp} users, {num_layers} layers, {path}")
        generator = MiMoGalaxyGenerator(
            mesh_device,
            cfg,
            MiMoCheckpoint(path, cfg),
            _cache_dir(path),
            dp=dp,
            users_per_replica=max_batch_size // dp,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            block_size=BLOCK_SIZE,
            setup_kv=False,
            mtp_layers=mtp_layers,
            mtp_backfill=MTP_BACKFILL and DEVICE_ROPE,
        )
        return cls(generator, max_seq_len)

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """MTP drafts up to MAX_DRAFTS tokens: the last committed token and its drafts fill one verify pass."""
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        if SPEC_DECODE != "mtp":
            return SpecReject(
                reason="MiMo drafts with its MTP layer: launch with MIMO_SPEC_DECODE=mtp", supported_k=(MAX_DRAFTS,)
            )
        if requested_k < 1:
            return SpecReject(reason=f"num_speculative_tokens {requested_k} < 1", supported_k=(MAX_DRAFTS,))
        k = min(requested_k, MAX_DRAFTS)
        return SpecPlan(
            effective_k=k,
            lanes_per_request=k + 1,
            # The MTP ring cache is per lane, not per request; its weights are the replica's.
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            # argmax_ids for all-greedy steps; fused_sample (acceptance on device) when a sampled request shares
            # the step and the device sampler serves every request's sampling; logits otherwise (the plugin's host
            # accept walk then samples losslessly, ``_verify``).
            accept_modes=("argmax_ids", "logits", "fused_sample"),
            drafter_state="internal",
            supports_narrow_decode=True,
        )

    @classmethod
    def get_max_tokens_all_users(cls, model_name="", num_devices=TP, tt_data_parallel=1, **kwargs):
        """KV tokens of one lane's pool: enough for ``max_num_seqs`` requests of ``max_model_len`` (plus what vLLM's
        SWA groups hold per request), capped by MIMO_KV_GB_PER_CHIP of DRAM per chip."""
        budget = float(os.environ.get("MIMO_KV_GB_PER_CHIP", DEFAULT_KV_GB_PER_CHIP)) * 1e9
        capacity = int(budget // KV_BYTES_PER_TOKEN)
        max_model_len, max_num_seqs = kwargs.get("max_model_len"), kwargs.get("max_num_seqs")
        if not max_model_len or not max_num_seqs:
            return capacity
        return min(capacity, max_num_seqs * (max_model_len + SWA_TOKENS_PER_USER))

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        hf = vllm_config.model_config.hf_config
        block_size = vllm_config.cache_config.block_size
        if block_size != BLOCK_SIZE:
            raise ValueError(
                f"MiMo's paged caches use {BLOCK_SIZE}-token blocks; launch with --block-size {BLOCK_SIZE}"
            )
        full = dict(num_kv_heads=hf.num_key_value_heads, head_size=hf.head_dim, head_size_v=hf.v_head_dim)
        swa = dict(num_kv_heads=hf.swa_num_key_value_heads, head_size=hf.swa_head_dim, head_size_v=hf.swa_v_head_dim)
        dtype = torch.bfloat16
        # vLLM's hybrid manager needs one page size for every group; pad the global layers' page to the SWA page.
        page = max(block_size * d["num_kv_heads"] * (d["head_size"] + d["head_size_v"]) * 2 for d in (full, swa))
        window = getattr(hf, "sliding_window_size", None) or hf.sliding_window
        specs = {}
        for i in range(hf.num_hidden_layers):
            name = f"model.layers.{i}.self_attn"
            if hf.hybrid_layer_pattern[i] == 1:
                specs[name] = SlidingWindowSpec(
                    block_size=block_size, dtype=dtype, sliding_window=window, page_size_padded=page, **swa
                )
            else:
                specs[name] = FullAttentionSpec(block_size=block_size, dtype=dtype, page_size_padded=page, **full)
        cls._kv_cache_spec_source = (vllm_config, specs)
        return specs

    @classmethod
    def _vllm_layer_groups(cls):
        """vLLM's KV cache group of each layer, from the specs get_kv_cache_spec returned."""
        if cls._layer_groups is not None:
            return cls._layer_groups
        if cls._kv_cache_spec_source is None:
            raise RuntimeError("allocate_kv_cache_per_layer before get_kv_cache_spec: no KV cache groups")
        from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

        vllm_config, specs = cls._kv_cache_spec_source
        groups = [None] * len(specs)
        for g, group in enumerate(get_kv_cache_groups(vllm_config, dict(specs))):
            for name in group.layer_names:
                groups[int(name.split(".")[2])] = g
        if None in groups:
            raise RuntimeError(f"layers without a KV cache group: {groups}")
        logger.info(f"MiMo vLLM KV cache groups per layer: {groups}")
        return groups

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """per_layer_specs: (shape, dtype, tensor_idx) per layer, shape[0] the lane's block count. Returns, per lane,
        [K, V] per layer, with buffers shared as in the module docstring."""
        groups = self._vllm_layer_groups()
        if len(groups) != len(per_layer_specs):
            raise RuntimeError(f"{len(per_layer_specs)} layer KV specs but {len(groups)} layer groups")
        num_blocks, block_size = per_layer_specs[0][0][0], per_layer_specs[0][0][2]
        assert block_size == BLOCK_SIZE and all(s[0][0] == num_blocks for s in per_layer_specs)
        # Layers sharing a buffer must come from different groups: their block ids never overlap.
        seen = set()
        for i, (_, _, t) in enumerate(per_layer_specs):
            assert (groups[i], t) not in seen, f"layer {i}: two layers of group {groups[i]} share buffer {t}"
            seen.add((groups[i], t))
        # vLLM trims its tables to the pool; the padding entries stay 0 (its null block, set_vllm_page_tables).
        table_width = page_table_width(min(math.ceil(self.max_seq_len / block_size), num_blocks))
        self.layer_groups = groups
        compact_rope = self.max_seq_len > ROPE_TABLE_FULL_WIDTH_MAX_LEN
        # The replicas' RoPE tables are equal: computed once on the host (~2.5 s at 1M), uploaded to each.
        rope_host = self.models[0].decode_rope_tables_host(compact_rope) if DEVICE_ROPE else None
        self.kv_caches = []
        for model in self.models:
            buffers, caches = {}, []
            for layer, (_, _, t) in zip(model.layers, per_layer_specs):
                attn = layer.attn
                caches.append(
                    [
                        self._buffer(model, buffers, ("k", t, attn.head_dim), num_blocks, attn.head_dim),
                        self._buffer(model, buffers, ("v", t, attn.v_head_dim), num_blocks, attn.v_head_dim),
                    ]
                )
            model.setup_vllm_kv_cache(self.per_lane, caches, groups, table_width)
            if DEVICE_ROPE:
                model.setup_decode_rope_tables(compact_rope, rope_host)
            if model.mtp is not None:
                model.mtp.setup_cache()  # after the RoPE tables: the backfill reads them
            self.kv_caches.append(caches)
        logger.info(
            f"MiMo vLLM KV: {num_blocks} blocks of {block_size} per lane, {len(buffers)} buffers per replica, "
            f"page tables {table_width} wide for groups {sorted(set(groups))}"
        )
        if rope_host is not None:
            rows, width = rope_host[0].shape
            logger.info(
                f"MiMo vLLM: decode RoPE tables 4 x {rows} x {width} ({4 * rows * width * 2 / 1e6:.0f} MB per chip)"
            )
        return self.kv_caches

    @staticmethod
    def _buffer(model, buffers, key, num_blocks, dim):
        if key not in buffers:
            cache = ttnn.allocate_tensor_on_device(
                ttnn.Shape((num_blocks, 1, BLOCK_SIZE, dim)),
                model.kv_dtype,
                ttnn.TILE_LAYOUT,
                model.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.fill(cache, 0.0, output_tensor=cache)
            buffers[key] = cache
        return buffers[key]

    # ------------------------------------------------------------------ page tables
    def _group_rows(self, tables, row):
        """{group: vLLM block table row ``row``}, one per KV cache group (the layers of a group share a table)."""
        out = {}
        for i, g in enumerate(self.layer_groups):
            if g not in out:
                out[g] = torch.as_tensor(tables[i][row])
        return out

    def _layer_tables(self, page_table, page_tables_per_layer):
        """One block table per layer. The plugin sends ``page_tables_per_layer`` when vLLM made several KV cache
        groups; with one group, ``page_table`` serves every layer."""
        if page_tables_per_layer is None and len(set(self.layer_groups)) == 1:
            return [page_table] * len(self.layer_groups)
        if page_tables_per_layer is None or len(page_tables_per_layer) != len(self.layer_groups):
            raise ValueError(
                f"MiMo needs a page table per attention layer ({len(self.layer_groups)}); got "
                f"{None if page_tables_per_layer is None else len(page_tables_per_layer)}"
            )
        return page_tables_per_layer

    # ------------------------------------------------------------------ prefill
    def prefill_forward(
        self,
        tokens,
        start_pos,
        prompt_lens,
        page_table,
        kv_cache,
        enable_trace=False,
        empty_slots=None,
        page_tables_per_layer=None,
        sampling_params=None,
        **kwargs,
    ):
        """Row ``i`` computes tokens[i, start_pos[i]:prompt_lens[i]] for the request in slot ``empty_slots[i]``.
        Returns its last-token logits, float32 [B, 1, vocab] (host sampling)."""
        if sampling_params is not None:
            raise ValueError("MiMo samples on the host; got device sampling_params")
        tables = self._layer_tables(page_table, page_tables_per_layer)
        rows = tokens.shape[0]
        starts = np.asarray(start_pos).reshape(-1).astype(np.int64)
        ends = np.asarray(prompt_lens).reshape(-1).astype(np.int64)
        slots = list(range(rows)) if empty_slots is None else [int(s) for s in empty_slots]
        placed = []
        table_rows = [{} for _ in range(self.dp)]
        continues = {}
        for i in range(rows):
            lane, slot = divmod(slots[i], self.per_lane)
            # vLLM starts chunks and prefix hits on block boundaries; a chunk that does not is moved back to one and
            # recomputes the positions in front of it (their K / V are rewritten with the same values).
            start = int(starts[i]) // BLOCK_SIZE * BLOCK_SIZE
            cached = 0
            if (
                start > 0
                and PREFILL_BATCH
                and prefill_padded_len(int(ends[i])) <= self.models[0].PREFILL_BATCH_MAX_PADDED
            ):
                # A short prompt whose first blocks are cached (a prefix hit) is recomputed from position 0 in a batch
                # pass (with the lane's other first chunks, traced) instead of an eager later chunk. Its cached blocks
                # are not written: another request may read them, and a pass of another length could compute their
                # K / V slightly differently. Their fill entries go to the null block.
                cached, start = start // BLOCK_SIZE, 0
            placed.append((i, lane, slot, start, int(ends[i]), cached))
            table_rows[lane][slot] = self._group_rows(tables, i)
            # The MTP history: a chunk starting where the slot's last one ended continues its request.
            continues[i] = start > 0 and self.prefill_ends[lane].get(slot) == start
            self.prefill_ends[lane][slot] = int(ends[i])
        for lane, rows_of_lane in enumerate(table_rows):
            if rows_of_lane:
                self.models[lane].set_vllm_page_tables(rows_of_lane)
                # A lane taking a prompt drafts again only after a decode step of one request; its MTP ring starts
                # empty for the new request.
                self.spec_sources[lane] = None
                self.chained[lane] = None
                self.mtp_ring[lane] = None
                if self.models[lane].mtp is not None:
                    self.models[lane].mtp.reset()
        # A lane's first chunks of one padded length (at most PREFILL_BATCH_MAX_PADDED) go in one pass
        # (TtMiMoModel.prefill_enqueue_batch, which pads every chunk to the longest); the other rows one by one. Every
        # pass is enqueued before any is read, the lanes interleaved so all replicas have work.
        # A lane's batch passes go before its other rows: a batch reads no cached KV (it starts at 0), and a traced
        # batch's replay must not run while an eager pass's output of that lane is alive.
        by_lane = []
        for lane in range(self.dp):
            groups = {}
            for p in placed:
                if p[1] == lane:
                    key = prefill_padded_len(p[4]) if self._batchable(p) else ("row", p[0])
                    groups.setdefault(key, []).append(p)
            batches = [g for key, g in groups.items() if not isinstance(key, tuple)]
            by_lane.append(batches + [g for key, g in groups.items() if isinstance(key, tuple)])
        order = [passes[k] for k in range(max(map(len, by_lane))) for passes in by_lane if k < len(passes)]
        programs = [model.mesh_device.num_program_cache_entries() for model in self.models]
        t0 = time.perf_counter()
        pending = []
        for rows_of_pass in order:
            lane = rows_of_pass[0][1]
            chunks = [
                (torch.as_tensor(tokens[i, start:end]).to(torch.int32), slot, cached)
                for i, _, slot, start, end, cached in rows_of_pass
            ]
            if self._batchable(rows_of_pass[0]):
                handle = self.models[lane].prefill_enqueue_batch(chunks)
            else:
                (chunk, slot, _), start = chunks[0], rows_of_pass[0][3]
                handle = self.models[lane].prefill_enqueue(
                    chunk, slot, start_pos=start, history_continues=continues[rows_of_pass[0][0]]
                )
            pending.append((rows_of_pass, lane, handle))
        t1 = time.perf_counter()
        logits = [None] * rows
        for rows_of_pass, lane, handle in pending:
            out = self.models[lane].prefill_read(handle)[0].reshape(len(rows_of_pass), -1)
            for k, p in enumerate(rows_of_pass):
                logits[p[0]] = out[k]
        if TIMING:
            logger.info(
                f"MiMo vLLM timing: prefill {rows} rows in {len(pending)} passes "
                f"{[[p[4] - p[3] for p in rows_of] for rows_of, _, _ in pending]}: enqueue {1e3 * (t1 - t0):.0f} ms, "
                f"total {1e3 * (time.perf_counter() - t0):.0f} ms"
            )
        self._recapture_after_new_programs(programs)
        return torch.stack(logits).unsqueeze(1)

    def _batchable(self, placed_row):
        """A first chunk (start 0) whose padded length a prefill batch takes."""
        start, end = placed_row[3], placed_row[4]
        return PREFILL_BATCH and start == 0 and prefill_padded_len(end) <= self.models[0].PREFILL_BATCH_MAX_PADDED

    def _recapture_after_new_programs(self, programs):
        """``programs``: each replica's program-cache size before the prefill. A prefill that compiled new programs
        (a chunk length or offset first seen while serving) put their program-cache buffers in memory the replica's
        decode trace may write on replay (tech_reports/.../TraceCorrectness.md, unsafe allocations), so that replica's
        decode trace is captured again, around them. Every slot is idle during the capture, so no KV is written."""
        for lane, model in enumerate(self.models):
            if model.decode_trace is None or model.mesh_device.num_program_cache_entries() == programs[lane]:
                continue
            new = model.mesh_device.num_program_cache_entries() - programs[lane]
            # Every trace of the replica, in warmup_model_decode's order: the greedy and sampled decode steps, then the
            # verify step and the MTP draft program.
            model.release_spec_traces()
            model.release_decode_trace()
            model.capture_decode_trace()
            if model.mtp is not None:
                model.capture_spec_traces()
            self.decode_recaptures += 1
            logger.info(
                f"MiMo vLLM: replica {lane} decode trace recaptured after {new} new prefill programs "
                f"({self.decode_recaptures} recaptures so far)"
            )

    # ------------------------------------------------------------------ decode
    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=True,
        read_from_device=True,
        page_tables_per_layer=None,
        sampling_params=None,
        slot_remap=None,
        prompt_tokens=None,
        output_tokens=None,
        *,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        spec_mode=None,
        num_valid_drafts=None,
        accepted_counts=None,
        **kwargs,
    ):
        """One decode step for every lane's slots: tokens [B, 1], start_pos [B] (-1: idle slot), B = lanes x per_lane.
        With ``spec_mode`` it is a verify of the plugin's [B, 1 + K] candidate block (``_verify``); a plain decode
        ignores ``accepted_counts`` (it keeps no per-candidate state).

        ``sampling_params`` None: host sampling, every step reloads the inputs and the step returns logits
        [B, 1, vocab]. Otherwise device sampling: the step returns the tokens [B], greedy or drawn by DeviceSampler with
        the rows' parameters, which a step with ``reload_inputs`` or ``reload_sampling_params`` uploads (the rows'
        order is the one ``tokens`` has then). ``reload_inputs`` False (steady device decode) leaves ``tokens`` and
        ``start_pos`` unread: the
        trace consumes the tokens and positions its last replay wrote, and the adapter copies the RoPE inputs of its
        own position counter (``decode_positions``); ``reload_page_table`` then copies the page tables.

        The model keeps no other slot state (no penalty or RNG state), so a reloading step needs no ``slot_remap``:
        it rewrites every slot's tokens and positions. A steady step with a non-identity remap raises."""
        if "reset_batch" in kwargs:
            raise ValueError("MiMo implements decode_input_update_contract 1; got the legacy reset_batch keyword")
        if kwargs.get("defer_device_sampling"):
            raise ValueError("MiMo has no deferred device sampling (no device grammar)")
        if spec_mode is not None:
            return self._verify(
                tokens,
                start_pos,
                page_table,
                page_tables_per_layer,
                num_valid_drafts,
                spec_mode,
                kwargs.pop("spec_sampling", None),
            )
        if kwargs.get("spec_sampling") is not None:
            raise ValueError("spec_sampling comes with a fused_sample verify; got it on a plain decode")
        if reload_inputs and reload_page_table:
            raise ValueError("reload_inputs already copies the page tables; got reload_page_table too")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("reset_sampling_state without reload_inputs")
        on_device = sampling_params is not None
        if not reload_inputs:
            if not on_device:
                raise ValueError("host sampling decodes from host inputs: reload_inputs must be True")
            if not enable_trace:
                raise ValueError("a steady decode (reload_inputs False) replays the decode trace: enable_trace False")
            if slot_remap is not None and any(int(j) != i for i, j in enumerate(slot_remap)):
                raise ValueError(f"a steady decode cannot move resident slot state: slot_remap {list(slot_remap)}")
        t0 = time.perf_counter()
        tables = self._layer_tables(page_table, page_tables_per_layer)
        rows = self.dp * self.per_lane
        if reload_inputs:
            tokens = torch.as_tensor(tokens).reshape(-1)
            positions = torch.as_tensor(start_pos).reshape(-1)
            rows = positions.numel()
            if rows > self.dp * self.per_lane:
                raise ValueError(f"{rows} decode rows > {self.dp} lanes x {self.per_lane} slots")
        handles = []
        self.spec_sources = [None] * self.dp
        self.chained = [None] * self.dp
        for lane, model in enumerate(self.models):
            first = lane * self.per_lane
            last = min(first + self.per_lane, rows)
            if first >= rows:
                break
            if reload_inputs:
                tok = torch.zeros(model.max_batch, dtype=torch.int32)
                pos = torch.full((model.max_batch,), -1, dtype=torch.int32)
                tok[: last - first] = tokens[first:last].to(torch.int32)
                pos[: last - first] = positions[first:last].to(torch.int32)
            else:
                pos = self.decode_positions[lane]
                if pos is None:
                    raise RuntimeError(f"lane {lane}: a steady decode before any reloading decode")
            active = pos >= 0
            if on_device and (reload_inputs or reload_sampling_params):
                self._set_sampling(lane, model, sampling_params, first, last, pos)
            sample = on_device and self.lane_sampled[lane]
            if not bool(active.any()):
                self.decode_positions[lane] = pos
                handles.append(None)
                continue
            if reload_inputs or reload_page_table:
                model.set_vllm_page_tables(
                    {s: self._group_rows(tables, first + s) for s in range(last - first) if active[s]}
                )
            if enable_trace and model.decode_trace is None:
                raise RuntimeError("decode trace requested before warmup_model_decode captured it")
            rng = model.sampler.read_rng() if sample and SAMPLER_CHECK else None
            if reload_inputs:
                handle = model.decode_enqueue(tok, pos, sample=sample)
            else:
                handle = model.decode_enqueue_resident(pos, sample=sample)
            if rng is not None:
                self._check_sampling(lane, model, handle, rng, active)
            self.decode_positions[lane] = torch.where(active, pos + 1, pos)
            handles.append((lane, last - first, handle))
            live = [s for s in range(last - first) if active[s]]
            if len(live) == 1:
                self.spec_sources[lane] = DraftSource(row=first + live[0], slots=live)
        out = {"rows": rows, "handles": handles, "tokens": on_device}
        if TIMING:
            self._decode_timing(t0, reload_inputs, on_device)
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(out), is_tokens=on_device)
        return out

    def _decode_timing(self, t0, reload_inputs, on_device):
        """Every 128 decode steps: the mean host time of decode_forward's submission, and the share of steps that
        reloaded their inputs or sampled on device."""
        now = time.perf_counter()
        stats = getattr(self, "_timing", None) or {"steps": 0, "host": 0.0, "reload": 0, "device": 0, "since": now}
        stats["steps"] += 1
        stats["host"] += now - t0
        stats["reload"] += int(reload_inputs)
        stats["device"] += int(on_device)
        if stats["steps"] == 128:
            logger.info(
                f"MiMo vLLM timing: decode {stats['steps']} steps in {1e3 * (now - stats['since']):.0f} ms, submit "
                f"{1e3 * stats['host'] / stats['steps']:.2f} ms/step, {stats['reload']} reloads, "
                f"{stats['device']} device-sampled"
            )
            stats = None
        self._timing = stats

    # ------------------------------------------------------------------ speculative decoding
    def _verify(
        self, tokens, start_pos, page_table, page_tables_per_layer, num_valid_drafts, spec_mode, spec_sampling=None
    ):
        """A verify of the plugin's candidate block tokens / start_pos [B, 1 + K] (-1: no position): each lane's
        active rows packed into verify passes (reference.spec_decode.plan_verify_passes), a lane's passes one after
        another, the lanes side by side. Returns argmax_ids [B, 1 + K] (column j the target's greedy token after
        column j, unused columns 0), or with ``spec_mode`` "logits" the fp32 logits [B, 1 + K, vocab] of the same
        cells for the plugin's host accept walk: every answered cell holds its slot's full logit row; the others
        (gap rows, columns past a row's drafts) are never read and keep a finite earlier value (``_spec_logits``).

        With "fused_sample" each pass's acceptance runs on device right after it (VerifySampler; its inputs from
        reference.sample_verify.pass_inputs) and the verify returns accepted_token_ids [B, 1 + K] int32 (each row's
        committed tokens, -1 past its count) and accepted_counts [B] int32 in [1, 1 + K]; a row without a position (a
        gap) commits token 0 with count 1, what its argmax_ids answer commits. ``spec_sampling``
        (``_check_spec_sampling``) carries the sampled rows' parameters and the uniforms the plugin drew; None: every
        row greedy.

        An argmax_ids verify runs the same accept trace, every request greedy (its committed tokens are the plugin's
        greedy walk over the ids), so both modes read one result page per pass (tt/spec_chain.py): the argmax ids of
        the pass's slots, its accept page. On a lane whose step is one request in one pass (its DraftSource), with the
        decode RoPE tables, the accept trace's chain step also writes the MTP draft's inputs from what the device
        committed, and the draft trace is enqueued right behind it (``mtp.enqueue_chained``), before the host reads:
        the page then carries the drafts, kept in ``chained`` for propose_draft_tokens. The draft is declined on
        device when the bonus position reaches max_seq_len - K. MIMO_SPEC_CHAIN=0: no chain, and an argmax_ids verify
        reads the verify's greedy tokens. A logits verify chains nothing."""
        from vllm_tt_plugin.spec_decode import VerifyOutput

        t0 = time.perf_counter()
        self._verify_tables_s = 0.0
        if spec_mode not in ("argmax_ids", "logits", "fused_sample"):
            raise ValueError(f"MiMo verifies argmax ids, logits or fused_sample; got spec_mode {spec_mode!r}")
        if any(model.verify_trace is None for model in self.models):
            raise RuntimeError("a verify before warmup_model_decode captured the verify traces (MIMO_SPEC_DECODE=mtp)")
        fused = spec_mode == "fused_sample"
        accept = fused or (spec_mode == "argmax_ids" and SPEC_CHAIN)  # the pass's accept trace and result page
        if spec_sampling is not None and not fused:
            raise ValueError(f"spec_sampling is only valid with spec_mode 'fused_sample', got {spec_mode!r}")
        tables = self._layer_tables(page_table, page_tables_per_layer)
        tokens = torch.as_tensor(tokens).to(torch.int64)
        positions = torch.as_tensor(start_pos).to(torch.int64)
        counts = torch.as_tensor(num_valid_drafts).reshape(-1)
        rows = tokens.shape[0]
        if rows > self.dp * self.per_lane:
            raise ValueError(f"{rows} verify rows > {self.dp} lanes x {self.per_lane} slots")
        argmax = torch.zeros(tokens.shape, dtype=torch.int32)
        logits = self._spec_logits(tuple(tokens.shape)) if spec_mode == "logits" else None
        if fused:
            spec_sampling = self._check_spec_sampling(spec_sampling, rows, tokens.shape[1] - 1)
            # A row no pass answers (a gap) keeps token 0 and count 1, its argmax_ids answer.
            committed = torch.full(tokens.shape, -1, dtype=torch.int32)
            committed[:, 0] = 0
            accepted_counts = torch.ones(rows, dtype=torch.int32)
        # The next verify writes the bonus and K drafts at positions bonus .. bonus + K: the chain drafts below that.
        limit = self.max_seq_len - max(tokens.shape[1] - 1, 1)
        lanes, chains = [], []
        for lane, model in enumerate(self.models):
            first, last = lane * self.per_lane, min((lane + 1) * self.per_lane, rows)
            active = [
                (r, tokens[r].tolist(), positions[r].tolist(), int(counts[r]))
                for r in range(first, last)
                if int(positions[r, 0]) >= 0
            ]
            passes = plan_verify_passes(active)
            lanes.append(passes)
            source = self.spec_sources[lane] = draft_source(passes)
            # A draft source runs one pass: its chain row goes with it.
            chained = accept and SPEC_CHAIN and source is not None and model.mtp is not None
            chained = chained and model.rope_tables is not None
            chains.append((source, limit) if chained else None)
            # The decode trace's resident tokens and positions are stale now: the plugin reloads after a verify.
            self.decode_positions[lane] = None
        self.chained = [None] * self.dp
        waited = draft_wait = 0.0
        for k in range(max(map(len, lanes))):
            pending = []
            for lane, p in enumerate(lanes):
                if k >= len(p):
                    continue
                model = self.models[lane]
                handle = self._enqueue_verify_pass(lane, p[k], tables, chains[lane])
                event = None
                if accept:
                    model.verify_accept_enqueue(*pass_inputs(p[k], spec_sampling))
                    if chains[lane] is not None:
                        # MIMO_VLLM_TIMING splits the wait at the accept trace's end: verify vs chained draft.
                        event = ttnn.record_event(model.mesh_device, 0) if TIMING else None
                        model.mtp.enqueue_chained()
                pending.append((lane, p[k], handle, event))
            for lane, vp, handle, event in pending:
                model = self.models[lane]
                t = time.perf_counter()
                if accept:
                    if event is not None:
                        ttnn.event_synchronize(event)
                        waited += time.perf_counter() - t
                        t = time.perf_counter()
                    result = model.spec_chain.read()
                    if event is not None:
                        draft_wait += time.perf_counter() - t
                    else:
                        waited += time.perf_counter() - t
                    if chains[lane] is not None:
                        layers = len(model.mtp.layers)
                        self.chained[lane] = chained_draft(result, self.spec_sources[lane], vp.positions, layers)
                        if self.chained[lane] is not None:  # the chained draft wrote its rows up to B - 1
                            slot = self.spec_sources[lane].row - lane * self.per_lane
                            self.mtp_ring[lane] = (slot, result.bonus - 1)
                    if fused:
                        model.verify_sampler.tally(result.accept)
                        commit_pass(vp, result.accept, committed, accepted_counts)
                    else:
                        for s, row, col in vp.answers:
                            argmax[row, col] = int(result.argmax[s])
                    continue
                out = model.read_next_tokens(handle)
                rows = model.decode_read_rows(handle) if logits is not None else None
                waited += time.perf_counter() - t
                for s, row, col in vp.answers:
                    argmax[row, col] = int(out[s])
                    if rows is not None:
                        logits[row, col] = rows[s]
        if TIMING:
            chained = sum(c is not None for c in chains)
            self._spec_timing(
                "verify",
                t0,
                waited,
                sum(map(len, lanes)),
                self._verify_tables_s,
                draft_wait=draft_wait,
                chained=chained,
            )
        if fused:
            self._log_fused()
            return VerifyOutput(spec_mode="fused_sample", accepted_token_ids=committed, accepted_counts=accepted_counts)
        if logits is not None:
            return VerifyOutput(spec_mode="logits", logits=logits)
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=argmax)

    def _check_spec_sampling(self, spec_sampling, rows, num_drafts):
        """The plugin's ``spec_sampling`` of a fused_sample verify (vllm-tt-plugin docs/SPEC_DECODE_CONTRACT.md):
        host tensors over the block's B rows: temperature fp32 [B] (0: greedy), top_k int32 [B] (below 1 or at least
        the vocabulary: none), top_p fp32 [B], greedy bool [B], uniforms fp32 [B, K + 2] in [0, 1) (column 0 the
        bonus, 1..K the accept draws, K + 1 the correction). None: every row greedy. A sampled row must be one the
        device sampler serves, as the plugin routes it (``model_capabilities``)."""
        if spec_sampling is None:
            return None
        want = {
            "temperature": ((rows,), torch.float32),
            "top_k": ((rows,), torch.int32),
            "top_p": ((rows,), torch.float32),
            "greedy": ((rows,), torch.bool),
            "uniforms": ((rows, num_drafts + 2), torch.float32),
        }
        if set(spec_sampling) != set(want):
            raise ValueError(f"spec_sampling keys {sorted(spec_sampling)} != {sorted(want)}")
        for name, (shape, dtype) in want.items():
            t = spec_sampling[name]
            if not isinstance(t, torch.Tensor) or tuple(t.shape) != shape or t.dtype != dtype:
                got = (tuple(t.shape), t.dtype) if isinstance(t, torch.Tensor) else type(t).__name__
                raise ValueError(f"spec_sampling[{name!r}] must be a {dtype} tensor of shape {shape}, got {got}")
        u = spec_sampling["uniforms"]
        if bool(((u < 0) | (u >= 1.0)).any()):
            raise ValueError("spec_sampling['uniforms'] must lie in [0, 1)")
        vocab = self.gen.cfg.vocab_size
        max_k = self.model_capabilities["max_device_top_k"]
        for r in torch.nonzero(~spec_sampling["greedy"]).reshape(-1).tolist():
            t = float(spec_sampling["temperature"][r])
            k, p = int(spec_sampling["top_k"][r]), float(spec_sampling["top_p"][r])
            unbounded = k < 1 or k >= vocab
            if t <= 0.0 or (not unbounded and k > max_k) or (unbounded and p >= 1.0):
                raise ValueError(
                    f"row {r}: device acceptance takes a temperature > 0 with top_k <= {max_k}, or none with "
                    f"top_p < 1; got temperature {t}, top_k {k}, top_p {p}"
                )
        return spec_sampling

    def _log_fused(self):
        """Every 1024 fused verifies: the requests accepted on device, and how many of them on truncated candidates
        (a nucleus reaching past the candidates: accepted on them, as the decode sampler draws)."""
        self._fused_steps = getattr(self, "_fused_steps", 0) + 1
        if self._fused_steps % 1024 == 0:
            groups = sum(m.verify_sampler.groups for m in self.models)
            truncated = sum(m.verify_sampler.truncated for m in self.models)
            logger.info(
                f"MiMo vLLM fused accept: {self._fused_steps} verifies, {groups} requests accepted on device, "
                f"{truncated} on truncated candidates"
            )

    def _spec_logits(self, shape):
        """The host buffer a logits verify of candidate block ``shape`` [B, 1 + K] fills, kept across steps: a
        fresh [B, 1 + K, vocab] fp32 tensor would be zeroed every step (16 x 4 x vocab: ~40 MB). Lane DP is
        synchronous, so the plugin has finished reading a step's logits before the next verify writes them."""
        buffers = getattr(self, "_spec_logit_buffers", None)
        if buffers is None:
            buffers = self._spec_logit_buffers = {}
        if shape not in buffers:
            buffers[shape] = torch.zeros(*shape, self.gen.cfg.vocab_size, dtype=torch.float32)
        return buffers[shape]

    def _enqueue_verify_pass(self, lane, vp, tables, chain=None):
        """``chain``: (DraftSource, limit) of the draft chained behind the pass, None when none does. The pass's chain
        row (reference.spec_chain.chain_row) carries it and, with the MTP history, each slot's request slot (the
        verify trace writes the slots' final norm rows into their requests' rings)."""
        model = self.models[lane]
        hslots = None
        if model.mtp is not None and model.mtp.history is not None:
            first = lane * self.per_lane
            hslots = [None if row is None else row - first for row in vp.table_rows]
        if chain is not None or hslots is not None:
            chain = chain_row(model.max_batch, *(chain or (None, 0)), hslots)
        tok = torch.zeros(model.max_batch, dtype=torch.int32)
        pos = torch.full((model.max_batch,), -1, dtype=torch.int32)
        tok[:SPEC_SLOTS] = torch.tensor(vp.tokens, dtype=torch.int32)
        pos[:SPEC_SLOTS] = torch.tensor(vp.positions, dtype=torch.int32)
        t = time.perf_counter()
        # A request's candidates fill several slots: build its group rows once, not once per slot.
        by_row = {}
        for row in vp.table_rows:
            if row is not None and row not in by_row:
                by_row[row] = self._group_rows(tables, row)
        model.set_vllm_page_tables({s: by_row[row] for s, row in enumerate(vp.table_rows) if row is not None})
        self._verify_tables_s += time.perf_counter() - t
        return model.verify_enqueue(tok, pos, None if chain is None else torch.tensor(chain, dtype=torch.int32))

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        """Drafts for the next step: the MTP layer drafts for each lane whose last step ran one request, from that
        step's resident final hidden; every other row offers none. After a verify that chained the lane's draft on
        device (``_verify``), the drafts are already on the host: they are returned when the plugin committed what the
        device committed (the row's count, tokens and positions). Otherwise (a plain decode step, a logits verify, no
        decode RoPE tables, MIMO_SPEC_CHAIN=0, a draft the device declined at max_seq_len - K that ``num_drafts``
        allows, or a different commit) the MTP drafts now from the host plan (reference.spec_decode.plan_mtp), or with
        every MTP layer from the commit's description (the device plans the passes: their rows depend on its stash)."""
        from vllm_tt_plugin.spec_decode import DraftOutput

        t0 = time.perf_counter()
        committed_tokens = torch.as_tensor(committed_tokens)
        committed_positions = torch.as_tensor(committed_positions)
        counts = torch.as_tensor(accepted_counts).reshape(-1)
        rows = committed_tokens.shape[0]
        drafts = torch.zeros(rows, num_drafts, dtype=torch.int32)
        valid = torch.zeros(rows, dtype=torch.int32)
        jobs, inputs_s, hits = [], 0.0, 0
        for lane, source in enumerate(self.spec_sources):
            mtp, chained = self.models[lane].mtp, self.chained[lane]
            if source is None or mtp is None or source.row >= rows or not 1 <= num_drafts <= MAX_DRAFTS:
                continue
            r, m = source.row, int(counts[source.row])
            # The next verify writes the bonus token and the drafts at positions bonus .. bonus + num_drafts.
            if int(committed_positions[r, m - 1]) + num_drafts >= self.max_seq_len:
                continue
            hit = chained is not None and chained.row == r
            hit = hit and chained.matches(committed_tokens[r].tolist(), committed_positions[r].tolist(), m)
            if hit:
                drafts[r] = torch.tensor(chained.drafts[:num_drafts], dtype=torch.int32)
                valid[r] = num_drafts
                hits += 1
                if not self.chain_check:
                    continue
            # The rings hold this request's rows up to a - 1 when its last draft (or the chained one being checked)
            # wrote them; otherwise (after its prompt, after a gap without drafts) they are backfilled first.
            slot, a = source.row - lane * self.per_lane, int(committed_positions[r, 0]) - 1
            ring = self.mtp_ring[lane]
            if mtp.history is not None and slot < SPEC_SLOTS and (ring is None or ring[0] != slot or ring[1] < a - 1):
                mtp.enqueue_backfill(slot, a)
                self.chain_stats["backfills"] += 1
            self.mtp_ring[lane] = (slot, int(committed_positions[r, m - 1]) - 1)
            if mtp.multi:
                mtp.enqueue_layers(source, committed_tokens[r].tolist(), committed_positions[r].tolist(), m)
                anchor = SPEC_SLOTS - 1
            else:
                plan = plan_mtp(source, committed_positions[r].tolist(), m)
                slot_tokens = [0] * SPEC_SLOTS
                for c in range(m):
                    slot_tokens[source.slots[c]] = int(committed_tokens[r, c])
                mtp.enqueue(slot_tokens, plan)
                anchor = plan.anchor
            inputs_s += mtp.last_inputs_s
            jobs.append((lane, r, anchor, hit))
        waited = 0.0
        for lane, r, anchor, hit in jobs:
            t = time.perf_counter()
            host = torch.tensor(self.models[lane].mtp.read_drafts(anchor)[:num_drafts], dtype=torch.int32)
            waited += time.perf_counter() - t
            if hit:
                self._check_chained(lane, drafts[r], host)
                continue
            drafts[r] = host
            valid[r] = num_drafts
        planned = sum(not hit for *_, hit in jobs)
        self.chain_stats["chained"] += hits
        self.chain_stats["host"] += planned
        self.spec_sources = [None] * self.dp
        self.chained = [None] * self.dp
        if TIMING:
            self._spec_timing("propose", t0, waited, planned, inputs_s, chained=hits)
        return DraftOutput(draft_token_ids=drafts, num_valid=valid)

    def _check_chained(self, lane, chained, host):
        """MIMO_SPEC_CHAIN_CHECK: a draft returned from the chain against the host plan's draft of the same commit."""
        stats = self.chain_stats
        stats["checked"] += 1
        if not torch.equal(chained, host):
            stats["mismatched"] += 1
            logger.warning(
                f"MiMo spec chain check: lane {lane}: chained drafts {chained.tolist()}, host {host.tolist()}"
            )
        if stats["checked"] % 64 == 0:
            logger.info(
                f"MiMo spec chain check: {stats['checked']} chained drafts checked, {stats['mismatched']} differ"
            )

    def _spec_timing(self, kind, t0, waited, count, part=0.0, draft_wait=0.0, chained=0):
        """Every 128 verifies: per verify step, the mean host time of the verify (its enqueues: page tables, uploads,
        the verify, accept and chained draft traces) and its time blocked reading back, split at the accept trace's
        end on a lane that chained its draft (the verify's device share, then the draft's); the plugin's accept walk
        (from the verify's return to propose_draft_tokens); propose_draft_tokens' host time (with the host plan's
        inputs) and its time blocked on host-planned drafts; and the gap from its return to the next verify (the
        plugin's commit and vLLM's scheduling and output). ``count``: the verify's passes or the lanes the propose
        drafted from the host plan; ``chained``: the lanes whose draft the verify chained, or the propose returned."""
        now = time.perf_counter()
        stats = getattr(self, "_spec_timing_stats", None) or {
            "verifies": 0,
            "passes": 0,
            "chained": 0,
            "hits": 0,
            "drafts": 0,
            "verify_host": 0.0,
            "verify_wait": 0.0,
            "draft_wait": 0.0,
            "propose_host": 0.0,
            "propose_wait": 0.0,
            "verify_tables": 0.0,
            "draft_inputs": 0.0,
            "accept": 0.0,
            "accepts": 0,
            "gap": 0.0,
            "gaps": 0,
            "verified": None,
            "last": None,
            "since": t0,
        }
        if kind == "verify":
            if stats["last"] is not None:
                stats["gap"] += t0 - stats["last"]
                stats["gaps"] += 1
            stats["verifies"] += 1
            stats["passes"] += count
            stats["chained"] += chained
            stats["verify_host"] += now - t0 - waited - draft_wait
            stats["verify_wait"] += waited
            stats["draft_wait"] += draft_wait
            stats["verify_tables"] += part
            stats["verified"] = now
        else:
            if stats["verified"] is not None:
                stats["accept"] += t0 - stats["verified"]
                stats["accepts"] += 1
                stats["verified"] = None
            stats["drafts"] += count
            stats["hits"] += chained
            stats["propose_host"] += now - t0 - waited
            stats["propose_wait"] += waited
            stats["draft_inputs"] += part
            stats["last"] = now
        if kind == "propose" and stats["verifies"] >= 128:
            n = stats["verifies"]

            def ms(key, per=n):
                return f"{1e3 * stats[key] / max(per, 1):.2f}"

            logger.info(
                f"MiMo vLLM timing: spec {n} verifies in {1e3 * (now - stats['since']):.0f} ms "
                f"({1e3 * (now - stats['since']) / n:.2f} ms/step), {stats['passes']} passes, {stats['chained']} lane "
                f"drafts chained on device ({stats['hits']} returned), {stats['drafts']} host-planned; per step: "
                f"verify host {ms('verify_host')} (tables {ms('verify_tables')}) + wait {ms('verify_wait')} ms, "
                f"chained draft wait {ms('draft_wait')} ms, accept walk {ms('accept', stats['accepts'])} ms, "
                f"propose host {ms('propose_host')} (inputs {ms('draft_inputs')}) + wait {ms('propose_wait')} ms, "
                f"gap {ms('gap', stats['gaps'])} ms"
            )
            stats = None
        self._spec_timing_stats = stats

    def _check_sampling(self, lane, model, handle, rng, active):
        """MIMO_SAMPLER_CHECK: the step's tokens against the host's draw on its logit rows (blocking)."""
        logits = model.plan.to_torch_tp_concat(handle["logits_rows"], dim=-1).float().reshape(model.decode_users, -1)
        tokens = ttnn.to_torch(ttnn.get_device_tensors(handle["next_tokens"])[0]).reshape(-1)
        checked, exact, bad = model.sampler.check(logits, tokens, rng, active)
        stats = getattr(self, "_check_stats", None) or {"steps": 0, "rows": 0, "exact": 0, "bad": 0}
        stats["steps"] += 1
        stats["rows"] += checked
        stats["exact"] += exact
        stats["bad"] += len(bad)
        for r, got, want in bad:
            logger.warning(f"MiMo sampler check: lane {lane} row {r}: device {got}, host {want}")
        if stats["steps"] % 64 == 0:
            logger.info(
                f"MiMo sampler check: {stats['steps']} lane-steps, {stats['rows']} rows, {stats['exact']} with the "
                f"nucleus in the candidates, {stats['bad']} mismatches"
            )
        self._check_stats = stats

    def _set_sampling(self, lane, model, sampling_params, first, last, pos):
        """Uploads the sampling parameters of the lane's rows [first, last) (idle rows, position -1, greedy) and their
        random streams, and records whether any row samples. A seeded row's stream is keyed by (seed, position)
        (device_sampler.stream_states), so its draws do not depend on which steps reload it. A sampled row must be
        within what the plugin routes to the device (``model_capabilities``): top_k up to ``max_device_top_k``, or
        none with top_p < 1."""
        n = last - first
        active = pos[:n] >= 0
        temperature = torch.tensor([float(t) for t in sampling_params.temperature[first:last]])
        top_k = torch.tensor([int(k) for k in sampling_params.top_k[first:last]])
        top_p = torch.tensor([float(p) for p in sampling_params.top_p[first:last]])
        seeds = list(sampling_params.seed[first:last]) if sampling_params.seed is not None else [None] * n
        temperature = torch.where(active, temperature, torch.zeros_like(temperature))
        vocab = self.gen.cfg.vocab_size
        max_k = self.model_capabilities["max_device_top_k"]
        for s in range(n):
            if temperature[s] == 0 or top_k[s] == 1:
                continue
            k, p = int(top_k[s]), float(top_p[s])
            unbounded = k < 1 or k >= vocab
            if (not unbounded and k > max_k) or (unbounded and p >= 1.0):
                raise ValueError(
                    f"row {first + s}: device sampling takes top_k <= {max_k}, or none with top_p < 1; got "
                    f"temperature {float(temperature[s])}, top_k {k}, top_p {p}"
                )
        rows = model.max_batch
        states = stream_states(seeds + [None] * (rows - n), pos.tolist())
        hosts, sampled = model.sampler.host_params(temperature, top_k, top_p, states=states)
        model.sampler.set_params(hosts)
        self.lane_sampled[lane] = sampled

    def read_decode_output(self, out, async_read=False, **kwargs):
        """Starts reading a decode step's output: its greedy tokens (device sampling), without blocking when
        ``async_read`` (returns the per-lane host tensors and their read events), or its logit rows (host sampling;
        read now, returned with no events)."""
        if kwargs.get("sample_rows") is not None:
            raise ValueError("MiMo declares no selective host readback; got sample_rows")
        if not out["tokens"]:
            logits = self._read_logits(out)
            return (logits, []) if async_read else logits
        reads, events = [], []
        for entry in out["handles"]:
            if entry is None:
                continue
            lane, n, handle = entry
            reads.append((lane, n, handle["next_tokens"].cpu(blocking=not async_read)))
            if async_read:
                events.append(ttnn.record_event(self.models[lane].mesh_device, 0))
        host = {"rows": out["rows"], "reads": reads}
        return (host, events) if async_read else host

    def _read_logits(self, out):
        vocab = self.gen.cfg.vocab_size
        logits = torch.zeros(out["rows"], 1, vocab, dtype=torch.float32)
        for entry in out["handles"]:
            if entry is None:
                continue
            lane, n, handle = entry
            logits[lane * self.per_lane : lane * self.per_lane + n, 0] = self.models[lane].decode_read_rows(handle)[:n]
        return logits

    def process_decode_output_host(self, out, is_tokens=False, **kwargs):
        """Host output of ``read_decode_output``: greedy tokens int32 [B] in slot order (``is_tokens``; idle slots
        0), else the logits it read."""
        if kwargs.get("sample_rows") is not None:
            raise ValueError("MiMo declares no compact host logits; got sample_rows")
        if not is_tokens:
            if not isinstance(out, torch.Tensor):
                raise ValueError("host sampling: read_decode_output returns the logits")
            return out
        if isinstance(out, torch.Tensor) or "reads" not in out:
            raise ValueError("device sampling: expected the token reads of read_decode_output")
        tokens = torch.zeros(out["rows"], dtype=torch.int32)
        for lane, n, host in out["reads"]:
            row = ttnn.to_torch(ttnn.get_device_tensors(host)[0]).reshape(-1)[:n]
            tokens[lane * self.per_lane : lane * self.per_lane + n] = row.to(torch.int32)
        return tokens

    # ------------------------------------------------------------------ warmup and lifetime
    def _max_prefill_tokens(self):
        """The most tokens one prefill chunk of the model can hold: vLLM's per-step token budget plus the
        BLOCK_SIZE - 1 positions prefill_forward recomputes in front of an unaligned start, at most the model's
        max_prefill_chunk (prefill_enqueue splits longer chunks) and the context."""
        budget = self.prefill_token_budget
        if budget is None and self._kv_cache_spec_source is not None:
            budget = self._kv_cache_spec_source[0].scheduler_config.max_num_batched_tokens
        model = self.models[0]
        if budget is None:
            budget = model.max_prefill_chunk
        return min(budget + BLOCK_SIZE - 1, model.max_prefill_chunk, self.max_seq_len)

    def warmup_model_prefill(self, kv_cache=None, enable_trace=False, can_sample_on_device=False, **kwargs):
        """Compile every prefill program a served chunk can run, on every replica, before any decode trace is
        captured (prefill runs eagerly, so ``enable_trace`` changes nothing). One chunk per padded length at start 0
        and per later start that changes what the programs key on (model.prefill_warmup_chunks), each through one
        layer of each kind (TtMiMoModel.prefill_warmup_layers). Writes only KV blocks no request owns yet."""
        if self.already_warmed_up_prefill or self.prefill_programs_compiled:
            self.already_warmed_up_prefill = True
            return
        t0 = time.time()
        # Decode makes persistent L1 buffers on its first step (the MoE fabric row gather's among them). Layernorm and
        # some reductions size their circular buffers from the free L1 and key the program on that plan
        # (ttnn/cpp/ttnn/operations/core/program_cache_l1.hpp), so prefill is compiled with those buffers in place, as
        # served prefills run. The plugin warms prefill before decode and, with trace_mode decode_only, not again after.
        idle = torch.full((self.models[0].max_batch,), -1, dtype=torch.int32)  # every slot idle: no KV is written
        handles = [(m, m.decode_enqueue(torch.zeros_like(idle), idle)) for m in self.models if m.decode_trace is None]
        for m, handle in handles:
            m.decode_read(handle)
        model0 = self.models[0]
        groups = sorted(set(self.layer_groups))
        width = model0.vllm_tables[groups[0]].shape[1]
        num_blocks = self.kv_caches[0][0][0].shape[0]
        window = next((l.attn.sliding_window for l in model0.layers if l.is_swa), BLOCK_SIZE)
        max_tokens = self._max_prefill_tokens()
        chunks = prefill_warmup_chunks(max_tokens, self.max_seq_len, BLOCK_SIZE, window)
        layers = model0.prefill_warmup_layers()
        # Slot 0 of every group maps positions from 0 to blocks 1, 2, ..: free while vLLM has not started.
        blocks = torch.arange(1, min(width, num_blocks - 1) + 1, dtype=torch.int32)
        if blocks.numel() * BLOCK_SIZE < max(start for _, start in chunks):  # the later chunks' window blocks
            raise RuntimeError(f"{num_blocks} KV blocks cannot back the prefill warmup's chunks {chunks}")
        for model in self.models:
            model.set_vllm_page_tables({0: {g: blocks for g in groups}})
        programs = [m.mesh_device.num_program_cache_entries() for m in self.models]
        tokens = torch.arange(max_tokens, dtype=torch.int32) % 1000 + 1000
        for n, start in chunks:
            handles = [(m, m.prefill_enqueue(tokens[:n], 0, start_pos=start, layers=layers)) for m in self.models]
            for m, h in handles:
                m.prefill_read(h)
        # Batches of first chunks (prefill_forward): every user count and padded length a lane's step can hold. Slots
        # 1 .. users - 1 write the blocks after slot 0's.
        batches = prefill_warmup_batches(max_tokens, self.per_lane, model0.PREFILL_BATCH_MAX_PADDED, self.max_seq_len)
        batches = batches if PREFILL_BATCH else []  # prefill_forward runs every batchable first chunk as a batch
        span = -(-model0.PREFILL_BATCH_MAX_PADDED // BLOCK_SIZE)
        if self.per_lane * span > blocks.numel():
            raise RuntimeError(f"{num_blocks} KV blocks cannot back the prefill warmup's batches {batches}")
        for model in self.models:
            model.set_vllm_page_tables({s: {g: blocks[s * span :] for g in groups} for s in range(self.per_lane)})
        for users, n in batches:
            chunks_of = [(tokens[:n], s) for s in range(users)]
            handles = [(m, m.prefill_enqueue_batch(chunks_of, layers=layers)) for m in self.models]
            for m, h in handles:
                m.prefill_read(h)
        for model in self.models:
            empty = torch.zeros(width, dtype=torch.int32)
            model.set_vllm_page_tables({s: {g: empty for g in groups} for s in range(self.per_lane)})
        # Traced batches: their persistent buffers now, before any capture, and one eager pass each through every
        # layer (compiling whatever the capture will record); warmup_model_decode captures them after the decode trace.
        if PREFILL_TRACE and PREFILL_BATCH:
            for model in self.models:
                for users, n in batches:
                    padded = prefill_padded_len(n)
                    if (users, padded) not in model.prefill_traces:
                        model.setup_prefill_batch_trace(users, padded)
                        model.warm_prefill_batch_trace(users, padded)
            for model in self.models:
                ttnn.synchronize_device(model.mesh_device)
        self.already_warmed_up_prefill = self.prefill_programs_compiled = True
        new = [m.mesh_device.num_program_cache_entries() - p for m, p in zip(self.models, programs)]
        logger.info(
            f"MiMo vLLM: prefill warmed up in {time.time() - t0:.1f}s: {len(chunks)} chunks of up to {max_tokens} "
            f"tokens and {len(batches)} batches through layers {layers}, {new} new programs per replica"
        )

    def warmup_model_decode(self, kv_cache=None, enable_trace=False, max_batch_size=None, **kwargs):
        """``enable_trace``: capture each replica's decode trace (every slot idle, so no KV is written); otherwise
        compile the eager step. The plugin always runs the eager pass first; with MTP the verify and draft programs
        compile in their own captures, so that pass is the same as without it."""
        for lane, model in enumerate(self.models):
            if enable_trace:
                if model.decode_trace is None:
                    model.capture_decode_trace()
                    logger.info(f"MiMo vLLM: replica {lane} decode trace captured")
                pending = [key for key, t in model.prefill_traces.items() if t["id"] is None]
                for k, (users, padded) in enumerate(pending):
                    logger.debug(f"MiMo vLLM: replica {lane} capturing prefill trace {k + 1}/{len(pending)}")
                    model.capture_prefill_batch_trace(users, padded)
                if pending:
                    logger.info(f"MiMo vLLM: replica {lane} prefill traces captured for (users, length) {pending}")
                if model.mtp is not None and model.verify_trace is None:
                    model.capture_spec_traces()
                    logger.info(f"MiMo vLLM: replica {lane} verify and MTP draft traces captured")
            else:
                idle = torch.full((model.max_batch,), -1, dtype=torch.int32)
                model.decode_read(model.decode_enqueue(torch.zeros(model.max_batch, dtype=torch.int32), idle))

    def release_persistent_capture(self):
        for model in self.models:
            model.release_decode_trace()
            model.release_prefill_traces()
            model.release_spec_traces()
