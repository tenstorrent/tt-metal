# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN functional decoder for hexgrad/Kokoro-82M.

Architecture note (important)
-----------------------------
Kokoro-82M is a StyleTTS2 / ISTFTNet **text-to-speech** model. It is
*non-autoregressive*: ``KModel.forward_with_tokens`` runs a single forward pass
(bert -> bert_encoder -> prosody predictor -> text encoder -> ISTFTNet decoder)
and there is **no causal transformer decoder, no KV cache, and no token-by-token
decode step** anywhere in the model.

The only attention-based transformer component - and therefore the only piece
that this "functional decoder" bringup stage (which targets HF *transformer
decoder layers*: attention, MLP, norms, head reshapes) can meaningfully map to -
is the ``plbert`` module, a HuggingFace :class:`transformers.AlbertModel`
(``CustomAlbert`` in ``kokoro/modules.py``). This file brings that transformer
up in TTNN, end to end, with real weights.

Because ALBERT ties parameters across layers, the entire encoder is a single
repeating layer kind (``AlbertLayer``) applied ``num_hidden_layers`` times. That
one layer kind is the target of this bringup; :class:`FunctionalDecoder`
represents the full parameter-shared encoder stack so it can be validated
against HF ``last_hidden_state``.

Prefill / decode contract
--------------------------
The transformer is bidirectional and stateless, so the two "modes" differ only
in *execution strategy*, not in autoregressive semantics (there is none):

* ``prefill_forward`` - eager, accepts any logical sequence length
  ``1..max_position_embeddings`` (512). It owns tile padding and masking, so
  non-tile-aligned lengths are valid public inputs.
* ``decode_forward`` - the same computation captured into a TTNN trace and
  replayed for a fixed ``(batch, seq_len)`` shape. This satisfies the
  traced-execution requirement and provides the fast repeated-inference path a
  TTS server would use for fixed-length phoneme windows.

KV-cache / paged-cache / current-position concepts from the LLM decoder template
are architecturally **N/A** for this model; see
``doc/context_contract.json`` and ``doc/functional_decoder/README.md``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class FunctionalDecoder(LightweightModule):
    """TTNN implementation of the Kokoro-82M plbert (ALBERT) encoder.

    Forward signatures (recorded in the README):
        prefill_forward(input_ids, position_ids, token_type_ids, attention_mask=None, seq_len=None)
        decode_forward(input_ids, position_ids, token_type_ids, attention_mask=None)

    All forward inputs are device tensors produced by :meth:`prepare_inputs`
    (host-side input construction, the allowed torch boundary). The forward
    passes themselves are pure TTNN.
    """

    def __init__(self, *, mesh_device, hf_config, weights, layer_idx: int = 0, activation_dtype=ttnn.float32):
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
        self.w = weights
        # Weights are bf16 (ttnn.embedding requires bf16 tables; bf16 linear weights
        # are standard). Activations run in fp32: with bf16 activations the
        # HF-vs-TTNN PCC occasionally dips just below 0.995 on atypical inputs at
        # short lengths, while fp32 activations keep the worst case >= 0.9958 over a
        # 16-seed representative sweep (real IPA sentences: ~0.9994). Fidelity is
        # raised here per the functional-decoder skill; the optimize stage may lower
        # it. See doc/functional_decoder/pcc_results.json and README.md.
        self.activation_dtype = activation_dtype
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # (batch, seq_len) -> captured-trace record
        self._traces: dict = {}

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx=0, mesh_device, **kwargs):
        """Build the decoder from a real HF ALBERT state dict.

        ``state_dict`` may be the raw Kokoro checkpoint ``bert`` sub-dict (keys
        optionally prefixed with ``module.``) or an already-stripped ALBERT
        state dict. All host->device weight conversion happens here (setup-time
        boundary); nothing in the forward path touches torch.
        """
        sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state_dict.items()}

        def to_tt(tensor, transpose=False, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
            t = tensor.t().contiguous() if transpose else tensor.contiguous()
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh_device)

        p = "encoder.albert_layer_groups.0.albert_layers.0."
        weights = {
            # factorized embeddings
            "word_emb": to_tt(sd["embeddings.word_embeddings.weight"]),
            "pos_emb": to_tt(sd["embeddings.position_embeddings.weight"]),
            "tt_emb": to_tt(sd["embeddings.token_type_embeddings.weight"]),
            "emb_ln_w": to_tt(sd["embeddings.LayerNorm.weight"]),
            "emb_ln_b": to_tt(sd["embeddings.LayerNorm.bias"]),
            # embedding -> hidden projection
            "map_w": to_tt(sd["encoder.embedding_hidden_mapping_in.weight"], transpose=True),
            "map_b": to_tt(sd["encoder.embedding_hidden_mapping_in.bias"]),
            # single shared AlbertLayer
            "q_w": to_tt(sd[p + "attention.query.weight"], transpose=True),
            "q_b": to_tt(sd[p + "attention.query.bias"]),
            "k_w": to_tt(sd[p + "attention.key.weight"], transpose=True),
            "k_b": to_tt(sd[p + "attention.key.bias"]),
            "v_w": to_tt(sd[p + "attention.value.weight"], transpose=True),
            "v_b": to_tt(sd[p + "attention.value.bias"]),
            "dense_w": to_tt(sd[p + "attention.dense.weight"], transpose=True),
            "dense_b": to_tt(sd[p + "attention.dense.bias"]),
            "attn_ln_w": to_tt(sd[p + "attention.LayerNorm.weight"]),
            "attn_ln_b": to_tt(sd[p + "attention.LayerNorm.bias"]),
            "ffn_w": to_tt(sd[p + "ffn.weight"], transpose=True),
            "ffn_b": to_tt(sd[p + "ffn.bias"]),
            "ffn_out_w": to_tt(sd[p + "ffn_output.weight"], transpose=True),
            "ffn_out_b": to_tt(sd[p + "ffn_output.bias"]),
            "full_ln_w": to_tt(sd[p + "full_layer_layer_norm.weight"]),
            "full_ln_b": to_tt(sd[p + "full_layer_layer_norm.bias"]),
        }
        return cls(mesh_device=mesh_device, hf_config=hf_config, weights=weights, layer_idx=layer_idx)

    # -------------------------------------------------------------- input prep
    @staticmethod
    def prepare_inputs(input_ids: torch.Tensor, mesh_device, *, attention_mask: Optional[torch.Tensor] = None):
        """Host-side input construction (the allowed torch boundary).

        Accepts any logical sequence length. Pads token / position / type ids to
        a tile multiple, and builds the additive attention mask that (a) masks
        the tile padding and (b) applies the optional user padding mask. Returns
        device tensors plus ``seq_len`` (the logical length).
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
        additive = (1.0 - valid) * -1.0e9  # (batch, padded)
        additive = additive.view(batch, 1, 1, padded)

        def dev(t, dtype, layout):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh_device)

        return {
            "input_ids": dev(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "position_ids": dev(pos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "token_type_ids": dev(tok_type, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "attention_mask": dev(additive, ttnn.float32, ttnn.TILE_LAYOUT),
            "seq_len": seq_len,
            "padded_seq_len": padded,
            "batch": batch,
        }

    # ------------------------------------------------------------- computation
    def _embed(self, input_ids, position_ids, token_type_ids):
        ck = self.compute_kernel_config
        adt = self.activation_dtype
        we = ttnn.embedding(input_ids, self.w["word_emb"], layout=ttnn.TILE_LAYOUT, dtype=adt)
        pe = ttnn.embedding(position_ids, self.w["pos_emb"], layout=ttnn.TILE_LAYOUT, dtype=adt)
        te = ttnn.embedding(token_type_ids, self.w["tt_emb"], layout=ttnn.TILE_LAYOUT, dtype=adt)
        emb = ttnn.add(ttnn.add(we, te), pe)
        emb = ttnn.layer_norm(
            emb, weight=self.w["emb_ln_w"], bias=self.w["emb_ln_b"], epsilon=self.eps, compute_kernel_config=ck
        )
        hidden = ttnn.linear(emb, self.w["map_w"], bias=self.w["map_b"], compute_kernel_config=ck, dtype=adt)
        ttnn.deallocate(we)
        ttnn.deallocate(pe)
        ttnn.deallocate(te)
        ttnn.deallocate(emb)
        return hidden

    def _albert_layer(self, hidden, attention_mask, batch, seq_len):
        """One AlbertLayer (the single shared layer kind): post-LN attention + FFN."""
        ck = self.compute_kernel_config
        adt = self.activation_dtype
        nh, hd = self.num_heads, self.head_dim

        q = ttnn.linear(hidden, self.w["q_w"], bias=self.w["q_b"], compute_kernel_config=ck, dtype=adt)
        k = ttnn.linear(hidden, self.w["k_w"], bias=self.w["k_b"], compute_kernel_config=ck, dtype=adt)
        v = ttnn.linear(hidden, self.w["v_w"], bias=self.w["v_b"], compute_kernel_config=ck, dtype=adt)

        q = ttnn.permute(ttnn.reshape(q, (batch, seq_len, nh, hd)), (0, 2, 1, 3))
        k = ttnn.permute(ttnn.reshape(k, (batch, seq_len, nh, hd)), (0, 2, 3, 1))
        v = ttnn.permute(ttnn.reshape(v, (batch, seq_len, nh, hd)), (0, 2, 1, 3))

        scores = ttnn.matmul(q, k, compute_kernel_config=ck)
        scores = ttnn.multiply(scores, self.scale)
        if attention_mask is not None:
            scores = ttnn.add(scores, attention_mask)
        probs = ttnn.softmax(scores, dim=-1)
        context = ttnn.matmul(probs, v, compute_kernel_config=ck)
        context = ttnn.permute(context, (0, 2, 1, 3))
        context = ttnn.reshape(context, (batch, seq_len, nh * hd))

        attn = ttnn.linear(context, self.w["dense_w"], bias=self.w["dense_b"], compute_kernel_config=ck, dtype=adt)
        hidden = ttnn.layer_norm(
            ttnn.add(attn, hidden),
            weight=self.w["attn_ln_w"],
            bias=self.w["attn_ln_b"],
            epsilon=self.eps,
            compute_kernel_config=ck,
        )

        ff = ttnn.linear(hidden, self.w["ffn_w"], bias=self.w["ffn_b"], compute_kernel_config=ck, dtype=adt)
        # HF plbert uses gelu_new (tanh approximation). Empirically ttnn's accurate
        # erf gelu (fast_and_approximate_mode=False) tracks gelu_new far better than
        # ttnn's fast tanh mode: worst-case PCC over T=32/128/512 x 4 seeds is 0.9986
        # (erf) vs 0.9942 (fast tanh). See doc/functional_decoder/README.md.
        ff = ttnn.gelu(ff, fast_and_approximate_mode=False)
        ff = ttnn.linear(ff, self.w["ffn_out_w"], bias=self.w["ffn_out_b"], compute_kernel_config=ck, dtype=adt)
        hidden = ttnn.layer_norm(
            ttnn.add(ff, hidden),
            weight=self.w["full_ln_w"],
            bias=self.w["full_ln_b"],
            epsilon=self.eps,
            compute_kernel_config=ck,
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
        """Eager bidirectional encode over a (possibly tile-padded) sequence.

        Returns the tile-padded hidden state ``(batch, padded_seq_len, hidden)``.
        Callers slice to the logical ``seq_len`` at the torch boundary.
        """
        b = batch if batch is not None else input_ids.shape[0]
        tp = seq_len if seq_len is not None else input_ids.shape[-1]
        return self._encode(input_ids, position_ids, token_type_ids, attention_mask, b, tp)

    # ------------------------------------------------------------------- decode
    def capture_decode_trace(self, prepared):
        """Compile + capture a TTNN trace for the (batch, padded_seq_len) shape.

        Follows the standard pattern: warm-up compile, allocate persistent input
        tensors, capture, and stash. The persistent input tensors are the only
        buffers whose contents change on replay.
        """
        batch = prepared["batch"]
        tp = prepared["padded_seq_len"]
        key = (batch, tp)
        if key in self._traces:
            return self._traces[key]

        dev = self.mesh_device
        # persistent inputs (device-resident, addresses baked into the trace)
        in_ids = ttnn.clone(prepared["input_ids"])
        in_pos = ttnn.clone(prepared["position_ids"])
        in_tt = ttnn.clone(prepared["token_type_ids"])
        in_mask = ttnn.clone(prepared["attention_mask"]) if prepared["attention_mask"] is not None else None

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
        """Traced replay of the encoder for a fixed (batch, padded_seq_len).

        Captures the trace on first use for a shape, then updates the persistent
        device inputs and replays. The measured window is ``execute_trace``
        only - no torch, from_torch, or to_torch inside it.
        """
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
        # refresh persistent inputs with the new (already device-resident) values
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
