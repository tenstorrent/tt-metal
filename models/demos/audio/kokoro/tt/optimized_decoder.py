# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Optimized TTNN decoder for hexgrad/Kokoro-82M (plbert / ALBERT encoder).

This is the optimized counterpart of ``tt/functional_decoder.py``. It preserves
the exact same public contract (see that file and
``doc/optimized_decoder/README.md``) - a non-autoregressive, stateless,
bidirectional ALBERT encoder with one weight-tied layer kind (``AlbertLayer``)
applied ``num_hidden_layers`` times - but replaces the functional-stage op
topology with a much cheaper one:

Topology changes vs the functional decoder
------------------------------------------
* **Packed QKV projection.** The functional path ran three separate
  ``q``/``k``/``v`` matmuls on the same post-norm activation. Here the three
  weight matrices are concatenated at load time into one ``[hidden, 3*hidden]``
  matmul (OPT-001), then split into heads with the fused
  ``ttnn.experimental.nlp_create_qkv_heads`` op - eliminating the expensive
  per-head ``reshape``/``permute``/``transpose`` chain that dominated the
  functional decode window.
* **Fused scaled-dot-product-attention.** The functional path built attention by
  hand (``matmul`` -> ``* scale`` -> ``+ mask`` (a full broadcast add that was
  the single largest device op) -> ``softmax`` -> ``matmul``). Here that is one
  ``ttnn.transformer.scaled_dot_product_attention`` (FlashAttention-2) call with
  a broadcastable additive mask (OPT-002).
* **Fused head concat.** ``ttnn.experimental.nlp_concat_heads`` replaces the
  post-attention ``permute``/``reshape``.
* **Precision policy.** Activations move fp32 -> bf16; linear weights move
  bf16 -> BFP8 (attention) / BFP4 (MLP FF1/FF2) by default; math fidelity moves
  HiFi4 -> LoFi. All are configurable via :class:`PrecisionPolicy` so the policy
  can be swept on real weights against the >= 0.995 PCC bar.

Workload-shape note (prefill vs "decode")
------------------------------------------
Kokoro is non-autoregressive: "decode" is *not* an autoregressive M=1 step, it is
the same full-sequence encode captured into a TTNN trace and replayed for a fixed
``(batch, padded_seq_len)`` shape. The activation M dimension is therefore the
whole (tile-padded) sequence length, i.e. this workload is *prefill-shaped*, not
small-M decode-shaped. Consequently the ``$optimize`` skill's small-M
"decode activations width-sharded in L1 / DRAM-sharded decode matmul" guidance
does **not** map to this model: the correct architecture-appropriate choice is
DRAM-interleaved activations with 2D-grid compute-bound matmul program configs,
which is what both the prefill and traced-decode paths use here. This mapping is
recorded per the skill's "map each requirement to the nearest equivalent" rule.
See ``doc/optimized_decoder/README.md`` for the full accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _fidelity(name: str) -> "ttnn.MathFidelity":
    return {
        "LoFi": ttnn.MathFidelity.LoFi,
        "HiFi2": ttnn.MathFidelity.HiFi2,
        "HiFi4": ttnn.MathFidelity.HiFi4,
    }[name]


def _dtype(name: str) -> "ttnn.DataType":
    return {
        "bf16": ttnn.bfloat16,
        "bfp8": ttnn.bfloat8_b,
        "bfp4": ttnn.bfloat4_b,
        "fp32": ttnn.float32,
    }[name]


@dataclass
class PrecisionPolicy:
    """Per-tensor-group precision/fidelity policy for the optimized decoder.

    Defaults are the selected optimized policy (see README candidate table).
    Everything is tunable so the sweep harness can move one group at a time.
    """

    # Selected optimized policy (see doc/optimized_decoder/sweeps/*.json + work_log.md).
    # Precision was swept on real weights at the exact test seeds. Findings:
    #   * BF16 activations are MANDATORY - ttnn SDPA rejects fp32 inputs, so the
    #     functional stage's fp32-activation choice cannot be carried forward.
    #   * fp32 dest accumulation is REQUIRED for the reduced-precision weights: with
    #     fp32_dest_acc=False, BFP8 weights drop T=16 prefill PCC to 0.9909 < bar;
    #     with fp32_dest_acc=True the same BFP8 policy passes every tested length
    #     (worst 0.99674 > 0.995, above the functional fp32 floor of 0.9958). It
    #     costs ~0.3% latency, so it is kept.
    #   * BFP8 weights are the selected policy: they pass all real-weight PCC tests,
    #     are ~4% faster on traced decode than BF16 weights (2.95 vs 3.07 ms @T=512),
    #     and halve weight memory. BFP4 weights and LoFi ARE rejected - BFP4 drops
    #     PCC to ~0.95 and LoFi to ~0.991 with no latency benefit (matmuls are not
    #     the bottleneck), a perf-AND-correctness rejection.
    #   * HiFi4 adds only +0.0008 PCC for +9% latency vs HiFi2 and is rejected.
    activation: str = "bf16"  # matmul/activation compute dtype (bf16 required for SDPA)
    attn_weight: str = "bfp8"  # QKV + attention-output-dense weights
    mlp_weight: str = "bfp8"  # FFN FF1 (ffn) + FF2 (ffn_output) weights
    map_weight: str = "bfp8"  # embedding->hidden projection weight
    norm_weight: str = "bf16"  # LayerNorm gamma/beta
    embedding: str = "bf16"  # embedding tables (ttnn.embedding needs bf16)
    matmul_fidelity: str = "HiFi2"  # math fidelity for linear matmuls
    sdpa_fidelity: str = "HiFi2"  # math fidelity for SDPA
    fp32_dest_acc: bool = True  # fp32 dest accumulation (required for BFP8 short-length PCC)

    def label(self) -> str:
        return (
            f"act={self.activation},attn_w={self.attn_weight},mlp_w={self.mlp_weight},"
            f"mm_fid={self.matmul_fidelity},sdpa_fid={self.sdpa_fidelity},"
            f"fp32acc={int(self.fp32_dest_acc)}"
        )


class OptimizedDecoder(LightweightModule):
    """Optimized TTNN implementation of the Kokoro-82M plbert (ALBERT) encoder.

    Public forward signatures match :class:`FunctionalDecoder` exactly:
        prefill_forward(input_ids, position_ids, token_type_ids, attention_mask=None, *, batch, seq_len)
        decode_forward(input_ids, position_ids, token_type_ids, attention_mask=None, *, batch, seq_len)
        prepare_inputs(input_ids, mesh_device, *, attention_mask=None)
        from_state_dict(state_dict, *, hf_config, mesh_device, policy=None, ...)
    """

    def __init__(
        self, *, mesh_device, hf_config, weights, policy: Optional[PrecisionPolicy] = None, layer_idx: int = 0
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.config = hf_config
        self.layer_idx = layer_idx
        self.num_layers = int(hf_config.num_hidden_layers)
        self.num_heads = int(hf_config.num_attention_heads)
        self.hidden_size = int(hf_config.hidden_size)
        self.head_dim = self.hidden_size // self.num_heads
        self.eps = float(hf_config.layer_norm_eps)
        self.max_position_embeddings = int(hf_config.max_position_embeddings)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.policy = policy or PrecisionPolicy()
        self.w = weights
        self.activation_dtype = _dtype(self.policy.activation)

        self.matmul_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=_fidelity(self.policy.matmul_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=self.policy.fp32_dest_acc,
            packer_l1_acc=True,
        )
        # Norms/gelu benefit from a bit more fidelity and are cheap; keep HiFi4/fp32-acc.
        self.norm_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.sdpa_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=_fidelity(self.policy.sdpa_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            exp_approx_mode=False,
            q_chunk_size=128,
            k_chunk_size=128,
        )
        # Explicit 2D matmul core grid. The default ttnn.linear heuristic assigned
        # only ~24 cores to the projection/FFN matmuls (all marked SLOW); a
        # (y=8, x=10) = 80-core 2D multicast grid is near-optimal and robust across
        # every matmul shape here (see doc/optimized_decoder/sweeps/matmul_grid.json):
        # ffn_out 98->27us, dense 42->14us, ffn 51->24us, qkv 52->26us, map 24->9us.
        cg = mesh_device.compute_with_storage_grid_size()
        self.mm_core_grid = ttnn.CoreGrid(y=min(8, cg.y), x=min(10, cg.x))
        self._ffn_out_pc_cache: dict = {}
        self._intermediate_ktiles = int(hf_config.intermediate_size) // TILE  # K tiles for ffn_output
        self._hidden_ntiles = self.hidden_size // TILE  # N tiles for ffn_output
        # (batch, padded_seq_len, has_mask) -> captured-trace record
        self._traces: dict = {}

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_state_dict(
        cls, state_dict, *, hf_config, layer_idx=0, mesh_device, policy: Optional[PrecisionPolicy] = None, **kwargs
    ):
        """Build the optimized decoder from a real HF ALBERT state dict.

        All host->device weight conversion happens here (setup-time boundary);
        nothing in the forward path touches torch. Q/K/V weights and biases are
        concatenated into a single packed projection at load time (OPT-001).
        """
        policy = policy or PrecisionPolicy()
        sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state_dict.items()}

        emb_dt = _dtype(policy.embedding)
        norm_dt = _dtype(policy.norm_weight)
        attn_dt = _dtype(policy.attn_weight)
        mlp_dt = _dtype(policy.mlp_weight)
        map_dt = _dtype(policy.map_weight)

        def to_tt(tensor, transpose=False, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
            t = tensor.t().contiguous() if transpose else tensor.contiguous()
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh_device)

        p = "encoder.albert_layer_groups.0.albert_layers.0."

        # Packed QKV: concat [q; k; v] along the output dim so nlp_create_qkv_heads
        # can slice Q[0:H], K[H:2H], V[2H:3H] and split into heads in one fused op.
        qkv_w_torch = torch.cat(
            [sd[p + "attention.query.weight"], sd[p + "attention.key.weight"], sd[p + "attention.value.weight"]], dim=0
        )
        qkv_b_torch = torch.cat(
            [sd[p + "attention.query.bias"], sd[p + "attention.key.bias"], sd[p + "attention.value.bias"]], dim=0
        )

        weights = {
            # factorized embeddings
            "word_emb": to_tt(sd["embeddings.word_embeddings.weight"], dtype=emb_dt),
            "pos_emb": to_tt(sd["embeddings.position_embeddings.weight"], dtype=emb_dt),
            "tt_emb": to_tt(sd["embeddings.token_type_embeddings.weight"], dtype=emb_dt),
            "emb_ln_w": to_tt(sd["embeddings.LayerNorm.weight"], dtype=norm_dt),
            "emb_ln_b": to_tt(sd["embeddings.LayerNorm.bias"], dtype=norm_dt),
            # embedding -> hidden projection
            "map_w": to_tt(sd["encoder.embedding_hidden_mapping_in.weight"], transpose=True, dtype=map_dt),
            "map_b": to_tt(sd["encoder.embedding_hidden_mapping_in.bias"], dtype=ttnn.bfloat16),
            # single shared AlbertLayer - packed QKV
            "qkv_w": to_tt(qkv_w_torch, transpose=True, dtype=attn_dt),
            "qkv_b": to_tt(qkv_b_torch, dtype=ttnn.bfloat16),
            "dense_w": to_tt(sd[p + "attention.dense.weight"], transpose=True, dtype=attn_dt),
            "dense_b": to_tt(sd[p + "attention.dense.bias"], dtype=ttnn.bfloat16),
            "attn_ln_w": to_tt(sd[p + "attention.LayerNorm.weight"], dtype=norm_dt),
            "attn_ln_b": to_tt(sd[p + "attention.LayerNorm.bias"], dtype=norm_dt),
            "ffn_w": to_tt(sd[p + "ffn.weight"], transpose=True, dtype=mlp_dt),
            "ffn_b": to_tt(sd[p + "ffn.bias"], dtype=ttnn.bfloat16),
            "ffn_out_w": to_tt(sd[p + "ffn_output.weight"], transpose=True, dtype=mlp_dt),
            "ffn_out_b": to_tt(sd[p + "ffn_output.bias"], dtype=ttnn.bfloat16),
            "full_ln_w": to_tt(sd[p + "full_layer_layer_norm.weight"], dtype=norm_dt),
            "full_ln_b": to_tt(sd[p + "full_layer_layer_norm.bias"], dtype=norm_dt),
        }
        return cls(mesh_device=mesh_device, hf_config=hf_config, weights=weights, policy=policy, layer_idx=layer_idx)

    # -------------------------------------------------------------- input prep
    @staticmethod
    def prepare_inputs(input_ids: torch.Tensor, mesh_device, *, attention_mask: Optional[torch.Tensor] = None):
        """Host-side input construction (the allowed torch boundary).

        Accepts any logical sequence length. Pads ids/positions/type-ids to a
        tile multiple and builds the additive SDPA mask of shape
        ``(batch, 1, padded, padded)`` (broadcast over heads). The mask masks
        both tile padding and the optional user padding mask. Returns ``None``
        for the mask only when the sequence is tile-aligned and fully valid, so
        the common max-context path pays no masking cost.
        """
        assert input_ids.dim() == 2, "input_ids must be (batch, seq_len)"
        batch, seq_len = input_ids.shape
        padded = _round_up(seq_len, TILE)

        ids = torch.zeros((batch, padded), dtype=torch.int32)
        ids[:, :seq_len] = input_ids.to(torch.int32)
        pos = torch.zeros((batch, padded), dtype=torch.int32)
        pos[:, :seq_len] = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        tok_type = torch.zeros((batch, padded), dtype=torch.int32)

        # valid mask over keys: 1 for real, 0 for padded (tile pad or user pad)
        valid = torch.zeros((batch, padded), dtype=torch.float32)
        valid[:, :seq_len] = 1.0
        if attention_mask is not None:
            valid[:, :seq_len] *= attention_mask.to(torch.float32)

        need_mask = bool((valid[:, :padded] == 0).any().item())

        def dev(t, dtype, layout):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh_device)

        mask_dev = None
        if need_mask:
            # SDPA additive mask: (batch, 1, padded, padded), broadcast over heads.
            # Per-key padding mask is identical across query rows.
            key_bias = (1.0 - valid) * -1.0e9  # (batch, padded)
            additive = key_bias.view(batch, 1, 1, padded).expand(batch, 1, padded, padded).contiguous()
            mask_dev = dev(additive, ttnn.bfloat16, ttnn.TILE_LAYOUT)

        return {
            "input_ids": dev(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "position_ids": dev(pos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "token_type_ids": dev(tok_type, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "attention_mask": mask_dev,
            "seq_len": seq_len,
            "padded_seq_len": padded,
            "batch": batch,
        }

    def _ffn_out_program_config(self, m_tiles: int):
        """Explicit 2D program config for the ffn_output (down) matmul.

        This was the single largest decode matmul. An explicit
        MatmulMultiCoreReuseMultiCast config with in0_block_w=8 and a 1x3 output
        subblock beats the core_grid heuristic at every measured seq length, and
        crucially avoids the heuristic's small-M cliff (M=32/64 were ~53us on the
        heuristic vs ~13-17us here). Grid rows are chosen to divide M-tiles evenly
        so the config is valid for any (tile-padded) sequence length. K=intermediate
        tiles, N=hidden tiles. See doc/optimized_decoder/sweeps/matmul_grid.json.
        """
        pc = self._ffn_out_pc_cache.get(m_tiles)
        if pc is not None:
            return pc
        gy = max(g for g in range(1, 9) if m_tiles % g == 0)
        per_core_m = m_tiles // gy
        n_tiles = self._hidden_ntiles  # 24 for hidden=768
        gx = 8
        per_core_n = (n_tiles + gx - 1) // gx  # 3 for N=24
        # in0_block_w must divide K tiles; 8 divides 64 (intermediate=2048).
        in0_block_w = math.gcd(8, self._intermediate_ktiles)
        out_subblock_w = per_core_n if per_core_n <= 4 else 1
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
        )
        self._ffn_out_pc_cache[m_tiles] = pc
        return pc

    # ------------------------------------------------------------- computation
    def _embed(self, input_ids, position_ids, token_type_ids):
        ck = self.norm_kernel_config
        adt = self.activation_dtype
        we = ttnn.embedding(input_ids, self.w["word_emb"], layout=ttnn.TILE_LAYOUT, dtype=adt)
        pe = ttnn.embedding(position_ids, self.w["pos_emb"], layout=ttnn.TILE_LAYOUT, dtype=adt)
        te = ttnn.embedding(token_type_ids, self.w["tt_emb"], layout=ttnn.TILE_LAYOUT, dtype=adt)
        emb = ttnn.add(ttnn.add(we, te), pe)
        emb = ttnn.layer_norm(
            emb, weight=self.w["emb_ln_w"], bias=self.w["emb_ln_b"], epsilon=self.eps, compute_kernel_config=ck
        )
        hidden = ttnn.linear(
            emb,
            self.w["map_w"],
            bias=self.w["map_b"],
            compute_kernel_config=self.matmul_kernel_config,
            core_grid=self.mm_core_grid,
            dtype=adt,
        )
        ttnn.deallocate(we)
        ttnn.deallocate(pe)
        ttnn.deallocate(te)
        ttnn.deallocate(emb)
        return hidden

    def _albert_layer(self, hidden, attention_mask, batch, seq_len):
        """One AlbertLayer: packed-QKV + fused SDPA attention, then FFN, post-LN."""
        mm = self.matmul_kernel_config
        adt = self.activation_dtype
        nh, hd = self.num_heads, self.head_dim

        cg = self.mm_core_grid
        # Packed QKV projection -> fused head split (no manual reshape/permute).
        qkv = ttnn.linear(
            hidden, self.w["qkv_w"], bias=self.w["qkv_b"], compute_kernel_config=mm, core_grid=cg, dtype=adt
        )
        qkv = ttnn.reshape(qkv, (batch, 1, seq_len, 3 * nh * hd))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=nh, num_kv_heads=nh, transpose_k_heads=False)
        ttnn.deallocate(qkv)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=False,
            scale=self.scale,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_kernel_config,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.experimental.nlp_concat_heads(attn)  # (b, 1, s, nh*hd)
        attn = ttnn.reshape(attn, (batch, seq_len, nh * hd))

        attn = ttnn.linear(
            attn, self.w["dense_w"], bias=self.w["dense_b"], compute_kernel_config=mm, core_grid=cg, dtype=adt
        )
        hidden = ttnn.layer_norm(
            ttnn.add(attn, hidden),
            weight=self.w["attn_ln_w"],
            bias=self.w["attn_ln_b"],
            epsilon=self.eps,
            compute_kernel_config=self.norm_kernel_config,
        )

        # HF plbert uses gelu_new; ttnn's accurate (erf) gelu tracks it best. The
        # fused matmul activation "gelu" is the accurate-erf variant (bit-identical
        # to a separate ttnn.gelu(fast_and_approximate_mode=False); see
        # doc/optimized_decoder/work_log.md), so we fuse it into FF1 and drop the
        # separate ~23us/layer gelu op with no accuracy cost.
        ff = ttnn.linear(
            hidden,
            self.w["ffn_w"],
            bias=self.w["ffn_b"],
            compute_kernel_config=mm,
            core_grid=cg,
            dtype=adt,
            activation="gelu",
        )
        ff = ttnn.linear(
            ff,
            self.w["ffn_out_w"],
            bias=self.w["ffn_out_b"],
            compute_kernel_config=mm,
            program_config=self._ffn_out_program_config(batch * seq_len // TILE),
            dtype=adt,
        )
        hidden = ttnn.layer_norm(
            ttnn.add(ff, hidden),
            weight=self.w["full_ln_w"],
            bias=self.w["full_ln_b"],
            epsilon=self.eps,
            compute_kernel_config=self.norm_kernel_config,
        )
        return hidden

    def _encode(self, input_ids, position_ids, token_type_ids, attention_mask, batch, padded_seq_len):
        hidden = self._embed(input_ids, position_ids, token_type_ids)
        for _ in range(self.num_layers):
            hidden = self._albert_layer(hidden, attention_mask, batch, padded_seq_len)
        return hidden

    # ------------------------------------------------------------------ prefill
    def prefill_forward(
        self, input_ids, position_ids, token_type_ids, attention_mask=None, *, batch=None, seq_len=None
    ):
        """Eager bidirectional encode over a (possibly tile-padded) sequence."""
        b = batch if batch is not None else input_ids.shape[0]
        tp = seq_len if seq_len is not None else input_ids.shape[-1]
        return self._encode(input_ids, position_ids, token_type_ids, attention_mask, b, tp)

    # ------------------------------------------------------------------- decode
    def capture_decode_trace(self, prepared):
        """Compile + capture a TTNN trace for the (batch, padded_seq_len) shape."""
        batch = prepared["batch"]
        tp = prepared["padded_seq_len"]
        has_mask = prepared["attention_mask"] is not None
        key = (batch, tp, has_mask)
        if key in self._traces:
            return self._traces[key]

        dev = self.mesh_device
        in_ids = ttnn.clone(prepared["input_ids"])
        in_pos = ttnn.clone(prepared["position_ids"])
        in_tt = ttnn.clone(prepared["token_type_ids"])
        in_mask = ttnn.clone(prepared["attention_mask"]) if has_mask else None

        # warm-up compile (populates program cache) before capture
        warm = self._encode(in_ids, in_pos, in_tt, in_mask, batch, tp)
        ttnn.deallocate(warm)
        ttnn.synchronize_device(dev)

        trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
        out = self._encode(in_ids, in_pos, in_tt, in_mask, batch, tp)
        ttnn.end_trace_capture(dev, trace_id, cq_id=0)
        ttnn.synchronize_device(dev)

        record = {
            "trace_id": trace_id,
            "in_ids": in_ids,
            "in_pos": in_pos,
            "in_tt": in_tt,
            "in_mask": in_mask,
            "out": out,
        }
        self._traces[key] = record
        return record

    def decode_forward(self, input_ids, position_ids, token_type_ids, attention_mask=None, *, batch=None, seq_len=None):
        """Traced replay of the encoder for a fixed (batch, padded_seq_len)."""
        b = batch if batch is not None else input_ids.shape[0]
        tp = seq_len if seq_len is not None else input_ids.shape[-1]
        prepared = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "batch": b,
            "padded_seq_len": tp,
        }
        record = self.capture_decode_trace(prepared)
        dev = self.mesh_device
        ttnn.copy(input_ids, record["in_ids"])
        ttnn.copy(position_ids, record["in_pos"])
        ttnn.copy(token_type_ids, record["in_tt"])
        if attention_mask is not None and record["in_mask"] is not None:
            ttnn.copy(attention_mask, record["in_mask"])
        ttnn.execute_trace(dev, record["trace_id"], cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        return record["out"]

    def release_traces(self):
        for record in self._traces.values():
            ttnn.release_trace(self.mesh_device, record["trace_id"])
        self._traces.clear()
