# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer-30B full text model on a 1 x TP mesh: embedding, decoder stack, final norm, softcapped lm_head."""

import gc
import time

import torch
from loguru import logger

import ttnn
from models.demos.muse_glimmer_30b.tt.attention import allocate_kv_cache
from models.demos.muse_glimmer_30b.tt.ccl import MuseCCL
from models.demos.muse_glimmer_30b.tt.common import (
    RopeTables,
    as_tt,
    matmul_compute_config,
    replicate_mapper,
    shard_mapper,
)
from models.demos.muse_glimmer_30b.tt.decoder import MuseGlimmerDecoderLayer
from models.demos.muse_glimmer_30b.tt.rms_norm import RMSNorm
from models.demos.muse_glimmer_30b.tt.weight_packing import pack_lm_head

DECODE_B_PAD = 32  # decode batch is always padded to one tile of rows


class MuseGlimmerModel:
    def __init__(self, args, mesh_device, ccl=None, use_weight_cache=True, kv_num_blocks=None):
        self.args = args
        self.mesh_device = mesh_device
        self.ccl = ccl if ccl is not None else MuseCCL(mesh_device)
        self.matmul_cfg = matmul_compute_config(mesh_device)
        self.tp = args.tp

        def cache(name, dtype, tag="v1"):
            return args.cache_file(name, dtype, tag) if use_weight_cache else None

        # Embedding table [1,1,V,H] bf16 ROW_MAJOR, sharded on hidden (each rank holds [V, H/TP]).
        emb = args.load_embedding_weight()
        self.embedding_weight = as_tt(
            emb.reshape(1, 1, *emb.shape),
            mesh_device,
            dtype=args.embed_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=shard_mapper(mesh_device, -1),
            cache_file_name=cache("embed_tokens", args.embed_dtype),
        )
        del emb
        self.embed_norm = RMSNorm(mesh_device, None, args.rms_norm_eps)
        self.rope = RopeTables(mesh_device, args.head_dim, args.rope_theta, args.max_seq_len)

        self.layers = []
        t0 = time.time()
        for i in range(args.n_layers):
            sd = args.load_layer_state_dict(i)
            self.layers.append(MuseGlimmerDecoderLayer(args, mesh_device, self.ccl, i, sd, use_weight_cache))
            del sd
            gc.collect()
            logger.info(f"Built layer {i + 1}/{args.n_layers} ({time.time() - t0:.0f}s)")

        self.norm = RMSNorm(mesh_device, args.load_final_norm_weight(), args.rms_norm_eps, add_unit_offset=False)
        lm = pack_lm_head(args.load_lm_head_weight(), args.padded_vocab_size)
        self.lm_head = as_tt(
            lm,
            mesh_device,
            dtype=args.lm_head_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=shard_mapper(mesh_device, -1),
            cache_file_name=cache("lm_head", args.lm_head_dtype),
        )
        del lm
        gc.collect()
        self.kv_caches = [allocate_kv_cache(args, mesh_device, kv_num_blocks) for _ in self.layers]
        self.sampling = None
        self._init_device_sampling()

    def _init_device_sampling(self):
        """Build a SamplingGenerator (gemma4-style). tp == 1 is not supported (single-device vocab-split path)."""
        if self.tp <= 1:
            logger.info("On-device sampling disabled for tp == 1; using host sampling")
            return
        from models.common.sampling.generator import SamplingGenerator
        from models.tt_transformers.tt.ccl import TT_CCL

        class _SamplingArgs:
            pass

        a = _SamplingArgs()
        a.vocab_size = self.args.vocab_size
        a.padded_vocab_size = self.args.padded_vocab_size  # == lm_head width (vocab_local * tp)
        a.cluster_shape = tuple(self.mesh_device.shape)
        a.sampling_all_gather_axis = 1  # gather across the TP (column) axis of the (1, tp) mesh
        a.sampling_dp = 1
        a.num_devices = self.mesh_device.get_num_devices()
        a.max_batch_size = DECODE_B_PAD
        a.max_top_k = 32
        a.use_topk_logprobs = False
        a.salt_duplicate_seeds = False
        a.model_config = {
            "SAMPLING_AG_CONFIG": {
                "allow_force_argmax": True,
                "num_links": self.ccl.num_links,
                "topology": ttnn.Topology.Linear,  # ring wraparound (D0->D28) is not routable on a 1x8 submesh under FABRIC_1D
            }
        }
        assert a.padded_vocab_size == self.args.vocab_local * self.tp, "sampling padded vocab must match lm_head"
        self.sampling = SamplingGenerator(args=a, mesh_device=self.mesh_device, tt_ccl=TT_CCL(self.mesh_device))
        # Same single-untilize override as gemma4 (avoids wide-row ttnn.concat L1 overflow on the chunked path).
        self.sampling.tt_sampling._untilize_chunk_count = lambda width: 1
        logger.info(
            f"On-device sampling initialized (vocab={a.vocab_size}, padded={a.padded_vocab_size}, "
            f"topology={self.ccl.topology}, links={self.ccl.num_links})"
        )

    @property
    def supports_device_sampling(self) -> bool:
        return self.sampling is not None

    def sample_on_device(self, logits_tt, sampling_params=None):
        """Eager (untraced) on-device sampling of sharded logits -> token tensor (per-device replicated).

        sampling_params: SamplingParams; if given, parameters are (re)loaded on the device first
        (do this once per generate(), not every step -- pass None on later steps).
        """
        if sampling_params is not None:
            self.sampling.reset_sampling_params(sampling_params)
        self.sampling.seed_manager.get_new_values()
        out = self.sampling.sample(logits=logits_tt, enable_trace=False)
        return out[0] if isinstance(out, tuple) else out

    def tokens_to_torch(self, tok_tt, num_rows: int) -> torch.Tensor:
        """Read sampled tokens from device 0 only -> int64 [num_rows]."""
        t = ttnn.to_torch(ttnn.get_device_tensors(tok_tt)[0]).reshape(-1)
        return t[:num_rows].to(torch.int64)

    def embed(self, tokens_tt):
        """tokens_tt uint32 ROW_MAJOR replicated [1, M] -> [1, 1, M, H] bf16 TILE (M multiple of 32)."""
        m = tokens_tt.shape[-1]
        e = ttnn.embedding(tokens_tt, self.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        e = ttnn.reshape(e, (1, 1, m, self.args.hidden_local))
        if self.tp > 1:
            g = self.ccl.all_gather(e, dim=3)
            ttnn.deallocate(e)
            e = g
        out = self.embed_norm(e)
        ttnn.deallocate(e)
        return out

    def head(self, h):
        """h [1,1,M,H] -> softcapped logits [1,1,M,vocab_local] (sharded per rank). Does not free h."""
        a = self.args
        n = self.norm(h)
        logits = ttnn.linear(
            n, self.lm_head, compute_kernel_config=self.matmul_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(n)
        x = ttnn.mul(logits, a.output_multiplier / a.final_logit_softcapping)
        ttnn.deallocate(logits)
        t = ttnn.tanh(x, fast_and_approximate_mode=False)
        ttnn.deallocate(x)
        out = ttnn.mul(t, a.final_logit_softcapping)
        ttnn.deallocate(t)
        return out

    def logits_to_torch(self, logits_tt, num_rows: int) -> torch.Tensor:
        """Sharded logits [1,1,M,vocab_local] -> [num_rows, vocab_size] float32 (padded vocab cols dropped)."""
        full = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))
        return full[0, 0, :num_rows, : self.args.vocab_size].float()

    def _prefill_hidden(self, tokens: torch.Tensor, user_id: int, page_table_tt):
        """Run embedding + all layers for one prompt -> (hidden [1,1,S,H], s_real, S)."""
        a = self.args
        tokens = tokens.reshape(-1)
        s_real = tokens.shape[0]
        assert 0 < s_real <= a.max_prefill_len, f"prompt length {s_real} > max_prefill_len {a.max_prefill_len}"
        # Multiple of the 256-row prefill SDPA Q chunk: a partial last Q chunk gives wrong sliding-window rows.
        s = 128 if s_real <= 128 else -(-s_real // 256) * 256
        assert s <= a.max_seq_len, f"padded prefill length {s} exceeds max_seq_len {a.max_seq_len}"
        pad_id = a.pad_token_id if a.pad_token_id is not None else 0
        padded = torch.full((1, s), pad_id, dtype=torch.int64)
        padded[0, :s_real] = tokens.to(torch.int64)
        tokens_tt = ttnn.from_torch(
            padded,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=replicate_mapper(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = self.embed(tokens_tt)
        ttnn.deallocate(tokens_tt)
        rope = self.rope.prefill(0, s)
        for i, layer in enumerate(self.layers):
            nx = layer.forward_prefill(x, rope if a.uses_rope(i) else None, self.kv_caches[i], page_table_tt, user_id)
            ttnn.deallocate(x)
            x = nx
        return x, s_real, s

    def prefill_forward_logits_tt(self, tokens: torch.Tensor, user_id: int, page_table_tt):
        """Prefill one prompt -> softcapped logits [1,1,32,vocab_local] on device; the last real token is row 0."""
        a = self.args
        x, s_real, _ = self._prefill_hidden(tokens, user_id, page_table_tt)
        last = ttnn.slice(x, (0, 0, s_real - 1, 0), (1, 1, s_real, a.hidden_size))
        ttnn.deallocate(x)
        hs = ttnn.pad(last, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
        ttnn.deallocate(last)
        logits = self.head(hs)
        ttnn.deallocate(hs)
        return logits

    def prefill_forward(self, tokens: torch.Tensor, user_id: int, page_table_tt, return_all_logits=False):
        a = self.args
        x, s_real, s = self._prefill_hidden(tokens, user_id, page_table_tt)

        if not return_all_logits:
            t0 = ((s_real - 1) // 32) * 32  # 32-row tile containing the last real row
            hs = ttnn.slice(x, (0, 0, t0, 0), (1, 1, t0 + 32, a.hidden_size))
            ttnn.deallocate(x)
            logits = self.head(hs)
            ttnn.deallocate(hs)
            out = self.logits_to_torch(logits, 32)[(s_real - 1) - t0]
            ttnn.deallocate(logits)
            return out

        outs = []
        for c in range(0, s, 128):
            hs = ttnn.slice(x, (0, 0, c, 0), (1, 1, c + 128, a.hidden_size))
            logits = self.head(hs)
            ttnn.deallocate(hs)
            outs.append(self.logits_to_torch(logits, 128))
            ttnn.deallocate(logits)
        ttnn.deallocate(x)
        return torch.cat(outs, dim=0)[:s_real]

    def prepare_decode_inputs_host(self, tokens: torch.Tensor, positions: torch.Tensor, page_table: torch.Tensor):
        """Host-side (not on device) replicated ttnn tensors for one decode step.

        Batch is padded to 32 rows: pad slots get token 0, position -1 and an all-zero page-table row.
        Returns (tokens uint32 [1,32], current_pos int32 [32], rope_idx uint32 [1,32], page_table int32 [32,n]).
        positions == -1 marks an inactive slot: paged_update_cache and paged SDPA decode skip it (the kernel
        reads int32 -1 as UINT32_MAX == "skip user"); RoPE embedding cannot take it so rope_idx is clamped to 0.
        Output rows of inactive slots are unspecified; ignore them.
        """
        tokens = tokens.reshape(-1).to(torch.int64)
        positions = positions.reshape(-1).to(torch.int64)
        b = tokens.shape[0]
        assert b <= DECODE_B_PAD and positions.shape[0] == b and page_table.shape[0] == b
        bp = DECODE_B_PAD
        tok = torch.zeros(1, bp, dtype=torch.int64)
        tok[0, :b] = tokens
        pos = torch.full((bp,), -1, dtype=torch.int64)
        pos[:b] = positions
        rope_idx = pos.clamp(min=0).reshape(1, bp)
        pt = torch.zeros(bp, page_table.shape[1], dtype=torch.int32)
        pt[:b] = page_table.to(torch.int32)
        mapper = replicate_mapper(self.mesh_device)

        def host(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper)

        return (
            host(tok, ttnn.uint32),
            host(pos.to(torch.int32), ttnn.int32),
            host(rope_idx, ttnn.uint32),
            host(pt, ttnn.int32),
        )

    def allocate_decode_inputs(self, B: int, n_blocks_per_user: int):
        """Persistent device tensors with the same specs as prepare_decode_inputs_host (for trace capture)."""
        dummy = self.prepare_decode_inputs_host(
            torch.zeros(B, dtype=torch.int64),
            torch.zeros(B, dtype=torch.int64),
            torch.zeros(B, n_blocks_per_user, dtype=torch.int32),
        )
        return tuple(ttnn.to_device(t, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in dummy)

    def copy_decode_inputs(self, host_inputs, device_inputs, copy_page_table=True):
        """Copy host decode inputs (tokens, pos, rope_idx, page_table) to the persistent device tensors."""
        n = len(host_inputs) if copy_page_table else len(host_inputs) - 1
        for h, d in list(zip(host_inputs, device_inputs))[:n]:
            ttnn.copy_host_to_device_tensor(h, d)

    def decode_forward(self, tokens_tt, current_pos_tt, rope_idx_tt, page_table_tt):
        """One decode step for the padded batch; returns sharded logits [1,1,32,vocab_local] (no host reads)."""
        a = self.args
        x = self.embed(tokens_tt)
        rope = self.rope.decode(rope_idx_tt)
        for i, layer in enumerate(self.layers):
            nx = layer.forward_decode(
                x, current_pos_tt, rope if a.uses_rope(i) else None, self.kv_caches[i], page_table_tt
            )
            ttnn.deallocate(x)
            x = nx
        logits = self.head(x)
        ttnn.deallocate(x)
        return logits
