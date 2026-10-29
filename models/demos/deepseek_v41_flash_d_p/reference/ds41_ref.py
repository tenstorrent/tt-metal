# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone torch CPU reference for DeepSeek-V4.1-Flash (text decoder), chunked prefill.

Ground truth is DeepSeek's inference code shipped with the checkpoint (``inference/model.py`` + ``kernel.py`` +
``engram.py``); this module re-implements its prefill chunk by chunk (the inference code prefills one chunk from
position 0 and then decodes token by token), with the GPU kernels' fp8 / fp4 quantize-dequantize steps
(``kernels.py``) and the checkpoint's exact dequantization (``weights.py``). Each block runs through ``run_block``, so
the block graphs below are the code. Compute dtype fp32; ``dtype=torch.bfloat16`` (official numerics only) keeps every
activation in bf16 where DeepSeek's GPU code does (module docstring "Numerics").

Residual: mHC with 4 streams, "single pass": a sublayer collapses the streams with the ``pre`` weights the PREVIOUS
sublayer computed (attention: the previous layer's FFN mixes; FFN: this layer's attention mixes, computed from the block
input; layer 0: one-hot on stream 0). So the block boundary carries the streams and the pending pre weights, packed:

    in / x / out   [S, 4 * H + 4]   columns [0, 4H): the 4 streams, token-major, stream-minor (HF's [S, 4, H]
                                    flattened); columns [4H, 4H + 4): the pre weights the next sublayer collapses with
    h_mid          [S, 4 * H]       the streams after the attention residual (no pending pre: the FFN uses attn_hc's)

Block (every layer; H = 5120, D = 512 head / latent dim, R = 64 rope dims at the END of a vector, interleaved pairs):
  engram layers (1, 14) first:
    engram_rows  = table rows [S, 24 * 256] (host)   n-gram hashes of the chunk's token ids (ctx.extra tokens; the last
                                                     3 compressed ids of the previous chunk in state engram_tail) ->
                                                     memory-mapped fp8 table rows times their e8m0 scales
    engram_kv    = wkv(rows) [S, 5 H]                fp8 linear (activations fp8-quantized per 32, ue8m0)
    x            = engram_add(in, engram_kv)         key = kv[:, :4H] per stream, value = kv[:, 4H:]; per stream
                                                     dot = <h, q_w k_w key> rstd(h) rstd(key) / sqrt(H),
                                                     gate = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))); h + gate value
  attn_hc      = mHC mixes of x [S, 24] fp32         [pre 4 | post 4 | comb 16 (row j = input stream, col k = output)]
                                                     from rms(flat x) @ hc_attn_fn, scale / base, 20 Sinkhorn rounds
  attn_in      = sum_n x.pre[n] x[n]   [S, H]        the PENDING pre (packed in x), not attn_hc's
  attn_norm    = rms(attn_in) w        (eps 1e-20)
  q_resid      = q_norm(wq_a(attn_norm)) [S, 1280]   fp8 linear
  kv           = window KV [S, D]                    kv_norm(wkv(attn_norm)), RoPE at the token's position, then
                                                     fp8 quantize-dequantize per 32 (RoPE tail included)
  compressed   (KV sources 2, 8, 14, 20) [m, D + 128] [stateful] the m groups of `ratio` tokens completed in this
                                                     chunk (groups start at multiples of ratio from position 0; an
                                                     unfinished group waits in compress_partial): ratio 2: softmax(
                                                     wgate x) weighted sum of wkv x per channel (fp32); ratio 1: wkv x;
                                                     latent = compressor norm; columns [0, D): latent with RoPE at the
                                                     group's first position, fp4 (16 per e4m3 scale); columns [D, D+128):
                                                     index key = k_norm(wk latent), RoPE, fp4 (32 per ue8m0 scale)
  topk         (index sources 2, 8, 14, 20, 24, 28, 32, 36) [S, 512] int32 [stateful]   compressed positions per query,
                                                     ascending, -1 = none: q = wq_b(q_resid) [32 heads x 128], RoPE,
                                                     fp4; score[t] = sum_h w_h relu(q_h . k_t), w = weights_proj(
                                                     attn_norm) / 64; position t visible iff t < (pos + 1) // ratio;
                                                     layer 20 also keeps per query the 2048 best blocks of 8 positions
                                                     (block score = max, the newest block pinned): state candidates;
                                                     layers > 20 score only inside layer 20's candidate blocks; top 512
                                                     (ties: lowest position first)
  attn_out     = attention [S, H] [stateful]         q = RoPE(wq_b(q_resid)) [64 heads x D]; keys = values = the
                                                     window (positions pos - 127 .. pos: the previous chunk's tail in
                                                     state window_kv, a ring indexed pos % 128) + (ratio > 0) the
                                                     compressed rows topk selects (own, or the source layer's state);
                                                     softmax(q.k / sqrt(D)) with a per-head sink logit in the
                                                     denominator; inverse RoPE on the output; fp8 q-dq; grouped wo_a
                                                     (8 groups of 8 heads -> 1024 each, bf16-exact weights); fp8 wo_b
  h_mid        = attn_residual(x, attn_hc, attn_out) [S, 4H]   post[k] attn_out + sum_j comb[j, k] x[j]
  ffn_hc       = mHC mixes of h_mid (hc_ffn_*)
  ffn_in       = sum_n attn_hc.pre[n] h_mid[n]       (the attention mixes' pre)
  ffn_norm     = rms(ffn_in) w
  router       = dense routing [S, 384] fp32          sqrt(softplus(x @ gate.T)); top 6 of score + bias; weights =
                                                     chosen scores / sum * 1.5
  experts_out  = sum_e (router e) expert_e           fp4 experts: w2(r * silu(min(w1 x, 10)) * clamp(w3 x, +-10)),
                                                     the routing weight applied before w2, every product fp8-quantized
  shared_out   = shared expert (fp8, no routing weight, same clamps)
  mlp_out      = experts_out + shared_out
  out          = ffn_residual(h_mid, ffn_hc, mlp_out)   streams post[k] mlp_out + sum_j comb[j, k] h_mid[j], packed with
                                                     ffn_hc's pre (the next layer's attention collapses with it)
Final: final_norm = rms(sum_n out.pre[n] out[n]) w; logits = final_norm @ head.T (fp32).

RoPE: ratio-0 layers theta 10000, no YaRN; compressed layers (ratio 1, 2) theta 160000 with YaRN (factor 16,
original 65536, beta 32 / 1), for every rope in the layer (q, window KV, compressed KV, index q / k).

State per layer (spec state.by_block_type; "fixed" ones are the current value, the others rows [0, length)):
    window_kv [128, D]             ring: slot p % 128 holds position p's window KV (zeros before the first write)
    compress_kv [length // r, D]   the KV source's compressed rows (RoPE'd, fp4); readers use their source's
    compress_partial [2, r, D]     fp32 [kv; score] of the unfinished group (slots 0 .. length % r - 1), ratio 2 only
    index_k [length // r, 128]     the KV source's index keys
    topk_idx [length, 512] int32   the index source's selection per query (rows [0, length): readers slice their chunk)
    candidates [length, 2048] int32   layer 20's candidate block ids per query (ascending, -1 = none)
    engram_tail [3] int64          compressed token ids of the 3 positions before `length` (-1 = none)
Readers (hooks.state_sources) keep only window_kv and read their sources' rows as of the end of the chunk.

Numerics (F112 / F128, reference/interface.py; one switch for the whole model, ``DS41Reference(numerics=)``):
    official   (default) the released code's arithmetic as above, bit for bit: the goldens and R.2 / R.3.
    native     full precision: every activation quantize-dequantize that only emulates the GPU's formats is the
               identity: the fp8 act q-dq of every fp8 linear's input (q_a, kv, wq_b, wo_b, indexer wq_b, engram wkv,
               shared / routed experts x and h), the fp8 q-dq of the window KV and of the attention output before wo_a,
               the fp4 q-dq of the compressed KV, the index key and the index query. Everything stays fp32. The
               checkpoint's weights keep their exact dequantized values (they are the model). There is no TF32
               rounding to remove, and no bf16 cast: every ``_r`` / ``_mm16`` bf16 point is official-at-bf16 only,
               and kernels.attend's bf16 P only applies to bf16 q (official at bf16, the oracle). kernels.py itself
               stays the faithful port the oracle (hf_oracle.py) runs; the switch is at the call sites here (``_q8`` /
               ``_q4``), so a native
               forward never calls a kernels.py quantizer (reference/test_numerics.py). Always fp32 (``dtype`` must
               be fp32): the golden target.
    official at dtype bf16 (F128 / F134 / F136: the margin source, spec reference.shipped_dtype): the GPU configuration
               (generate.py sets torch's default dtype to bf16), as the oracle runs it with ``build_oracle(dtype=
               torch.bfloat16)``. Arithmetic stays fp32 (the kernels' and the GEMMs' accumulation); a value is rounded
               to bf16 (``_r``) exactly where model.py holds it in bf16 (bringup/official_taps.yaml): the block's
               streams (at every unpack, so an fp32 golden input is read as the bf16 stream the GPU would hold), the
               engram rows, every fp8 / fp4 / bf16 GEMM's output (fp8_gemm / fp4_gemm / F.linear write the default
               dtype), every RMSNorm output, the hc_pre collapses, the RoPE'd q / kv / index q / latent / index key
               (apply_rotary_emb copies back into the bf16 tensor), the index-score tile, its relu x weights product
               and its head sum (bf16 tensors), weights_proj x scale, sparse_attn's bf16 P and output, the
               inverse-RoPE'd output, wo_a, the hc_post outputs, the experts' h before w2 (Expert.forward:
               x.to(dtype)), moe_add.
               fp32 where model.py computes in fp32: the mHC mixes and Sinkhorn, the ratio-2 compressor's wkv / wgate
               and pooling, the router, the routed experts' sum, the attention scores and softmax statistics, the
               pending pre weights, the logits. The GEMMs model.py runs on bf16 x and a bf16 weight (wo_a, ratio-1
               compressor wkv, indexer wk / weights_proj) and the index-score einsum with its relu x weights and head
               sum run as bf16 tensors (``_mm16``), so the summation order is the oracle's: CPU bf16 GEMMs and fp32
               GEMMs rounded once differ on ~0.03 % of outputs, and the fp8 quantizers grew that to rel 1.1e-2 by
               layer 3 (DS41-91). With them the reference equals the oracle at bf16 bit for bit (layers 0-3, 512
               tokens; reference/test_official_bf16.py). The fp8 / fp4 q-dq values are exact in bf16. State storage
               stays fp32 (every stored value is bf16-exact). Top-k ties (frequent on bf16 index scores) break toward
               the lowest position here; model.py's ``topk(sorted=False)`` leaves them to the kernel.

Determinism: every GEMM runs in the fixed tiles of kernels.py ("GEMM tiling": 256-row dense tiles, expert groups
padded to 32-row multiples (K-24), 64-query x 512-key index-score tiles), as the oracle's do, so a row's result never
depends on the chunk and the reference matches the oracle to the bit; top-k ties break toward the lowest position.
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F

from models.demos.common.bringup.reference.interface import Ctx, Step, noop, run_block
from models.demos.deepseek_v41_flash_d_p.reference import kernels as K
from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint, EngramTable, ExpertPrefetcher

Q_BLOCK = 64  # queries per attention block (== kernels.INDEX_Q_BLOCK == kernels.SPARSE_Q_BLOCK)
NUMERICS = ("official", "native")  # the reference's arithmetic (module docstring "Numerics")
PACK_PRE = 4  # pre weights packed after the streams


# ============================================================================================= config
# HF ``config.json`` (text_config) key -> inference ``config.json`` key, for the counts a scaled checkpoint
# (plan/scale.py, bringup/scaling.yaml) may cut. scale.py rewrites only the top-level config.json; the inference
# config of a scaled dir is the source's with these counts taken from the scaled config.json.
HF_TO_INFERENCE = {
    "num_attention_heads": "n_heads",
    "o_groups": "o_groups",
    "n_routed_experts": "n_routed_experts",
    "num_experts_per_tok": "n_activated_experts",
    "moe_intermediate_size": "moe_inter_dim",
    "hidden_size": "dim",
    "vocab_size": "vocab_size",
    "num_hidden_layers": "n_layers",
    "q_lora_rank": "q_lora_rank",
    "o_lora_rank": "o_lora_rank",
    "head_dim": "head_dim",
    "index_n_heads": "index_n_heads",
    "index_head_dim": "index_head_dim",
    "index_topk": "index_topk",
    "engram_n_heads": "engram_n_heads",
    "engram_head_dim": "engram_head_dim",
}
SCALE_MARKER = "bringup_scale.json"  # plan/scale.py MARKER


def _text_config(path: str) -> dict:
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def inference_dir(model_path: str) -> str:
    """The ``inference/`` code dir of a checkpoint; a scaled checkpoint's is its source's (bringup_scale.json)."""
    d = os.path.join(model_path, "inference")
    marker = os.path.join(model_path, SCALE_MARKER)
    if not os.path.isdir(d) and os.path.exists(marker):
        with open(marker) as f:
            d = os.path.join(json.load(f)["source"], "inference")
    return d


def inference_config(model_path: str) -> dict:
    """The inference ModelArgs dict of a checkpoint dir: its ``inference/config.json``; for a scaled checkpoint (no
    ``inference/``, a ``bringup_scale.json`` naming its source), the source's with every count the cut changed in
    ``config.json`` applied (an unmapped changed key fails loudly)."""
    own = os.path.join(model_path, "inference", "config.json")
    if os.path.exists(own):
        with open(own) as f:
            return json.load(f)
    marker = os.path.join(model_path, SCALE_MARKER)
    if not os.path.exists(marker):
        raise FileNotFoundError(f"{own} (and no {SCALE_MARKER}: not a scaled checkpoint)")
    with open(marker) as f:
        src = json.load(f)["source"]
    with open(os.path.join(src, "inference", "config.json")) as f:
        raw = json.load(f)
    hf_src, hf_dst = _text_config(src), _text_config(model_path)
    for k in sorted(set(hf_src) | set(hf_dst)):
        if hf_src.get(k) == hf_dst.get(k):
            continue
        assert k in HF_TO_INFERENCE, f"scaled config.json changes {k!r}, which has no inference config key"
        ik = HF_TO_INFERENCE[k]
        assert raw[ik] == hf_src[k], (k, ik, raw[ik], hf_src[k])  # the two configs of the source agree
        raw[ik] = hf_dst[k]
    return raw


class Config:
    """The inference ModelArgs (``inference/config.json``) plus the per-layer roles derived from them."""

    def __init__(self, raw: dict):
        self.raw = raw
        self.vocab_size = raw["vocab_size"]
        self.dim = raw["dim"]
        self.n_layers = raw["n_layers"]
        self.n_heads = raw["n_heads"]
        self.head_dim = raw["head_dim"]
        self.rope_dim = raw["rope_head_dim"]
        self.q_lora_rank = raw["q_lora_rank"]
        self.o_groups = raw["o_groups"]
        self.o_lora_rank = raw["o_lora_rank"]
        self.window = raw["window_size"]
        self.eps = raw["norm_eps"]
        self.ratios = list(raw["compress_ratios"])
        self.kv_sources = list(raw["kv_source_layers"])
        self.index_sources = list(raw["index_source_layers"])
        self.cand_source = raw.get("candidate_source_layer", -1)
        self.cand_topk_blocks = raw.get("candidate_topk_blocks", 0)
        self.cand_block = raw.get("candidate_block_size", 0)
        self.index_heads = raw["index_n_heads"]
        self.index_dim = raw["index_head_dim"]
        self.index_topk = raw["index_topk"]
        self.n_experts = raw["n_routed_experts"]
        self.topk_experts = raw["n_activated_experts"]
        self.route_scale = raw.get("route_scale", 1.0)
        self.swiglu_limit = raw.get("swiglu_limit", 0.0)
        self.inter = raw["moe_inter_dim"]
        self.hc = raw["hc_mult"]
        self.sinkhorn_iters = raw["hc_sinkhorn_iters"]
        self.hc_eps = raw["hc_eps"]
        self.engram_layers = list(raw.get("engram_layer_ids", []))
        self.engram_heads = raw.get("engram_n_heads", 0)
        self.engram_dim = raw.get("engram_head_dim", 0)
        self.engram_ngram = raw.get("engram_max_ngram_size", 1)
        self.rope = dict(
            theta=raw["rope_theta"],
            compress_theta=raw["compress_rope_theta"],
            original=raw.get("original_seq_len", 0),
            factor=raw.get("rope_factor", 1.0),
            beta_fast=raw.get("beta_fast", 32),
            beta_slow=raw.get("beta_slow", 1),
        )
        # what this reference implements; anything else in a new checkpoint must fail loudly
        assert raw.get("score_func", "sqrtsoftplus") == "sqrtsoftplus" and raw.get("gate_temp", 1.0) == 1.0
        assert raw.get("norm_topk_prob", True) and raw["n_shared_experts"] == 1
        assert raw.get("dtype", "fp8") == "fp8" and raw.get("expert_dtype", "fp4") == "fp4"

    @classmethod
    def from_dir(cls, model_path: str) -> "Config":
        return cls(inference_config(model_path))

    # roles
    def ratio(self, i: int) -> int:
        return self.ratios[i]

    def is_kv_source(self, i: int) -> bool:
        return i in self.kv_sources

    def is_index_source(self, i: int) -> bool:
        return i in self.index_sources

    def is_cand_source(self, i: int) -> bool:
        return i == self.cand_source

    def uses_candidates(self, i: int) -> bool:
        return 0 <= self.cand_source < i and self.is_index_source(i)

    def is_engram(self, i: int) -> bool:
        return i in self.engram_layers

    def kv_source_of(self, i: int) -> int | None:
        """The layer whose compressed KV and index keys layer i reads (the latest KV source at or before it)."""
        if not self.ratio(i):
            return None
        return max(j for j in self.kv_sources if j <= i)

    def index_source_of(self, i: int) -> int | None:
        """The layer whose top-k layer i attends with (the latest index source at or before it)."""
        if not self.ratio(i):
            return None
        return max(j for j in self.index_sources if j <= i)

    def sources(self, i: int) -> list[int]:
        """Earlier layers whose state layer i reads (hooks.state_sources)."""
        out = set()
        if self.ratio(i):
            out.add(self.kv_source_of(i))
            out.add(self.index_source_of(i))
        if self.uses_candidates(i):
            out.add(self.cand_source)
        out.discard(i)
        return sorted(out)

    def state_names(self, i: int) -> list[str]:
        """State tensors layer i owns (must equal the spec's state.by_block_type entry for its block type)."""
        names = ["window_kv"]
        if self.is_kv_source(i):
            names += ["compress_kv"] + (["compress_partial"] if self.ratio(i) > 1 else []) + ["index_k"]
        if self.is_index_source(i):
            names.append("topk_idx")
        if self.is_cand_source(i):
            names.append("candidates")
        if self.is_engram(i):
            names.append("engram_tail")
        return names


# ============================================================================================= building blocks
def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    """inference RMSNorm: fp32 x * rsqrt(mean(x^2) + eps), times the weight."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return y if w is None else w * y


mm = K.linear  # x [S, K] @ w[N, K].T in fixed-row GEMM tiles (kernels.py "GEMM tiling")


def fp8_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """inference linear() with an fp8 weight: activations fp8-quantized per 32 (ue8m0), then one fp32 GEMM."""
    return mm(K.act_quant_dequant(x, 32), w)


def yarn_freqs(dim: int, theta: float, original: int, factor: float, beta_fast: float, beta_slow: float):
    """inference precompute_freqs_cis without the position table: the [dim / 2] fp32 frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original > 0:

        def corrected_dim(rotations):
            return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(theta))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def rope_table(freqs: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Complex rotations [len(pos), dim / 2], the rows of inference's torch.polar(1, outer(arange, freqs))."""
    ang = torch.outer(pos.to(torch.int64), freqs)
    return torch.polar(torch.ones_like(ang), ang)


def rope(x: torch.Tensor, rot: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate the last R dims of x [S, ..., D] (adjacent pairs as complex numbers) by rot [S, R / 2]; new tensor."""
    r = rot.shape[-1] * 2
    head, tail = x[..., :-r], x[..., -r:]
    c = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)).contiguous())
    rr = rot.conj() if inverse else rot
    rr = rr.view(rr.shape[0], *([1] * (c.dim() - 2)), rr.shape[-1])
    out = torch.view_as_real(c * rr).flatten(-2).to(x.dtype)
    return torch.cat([head, out], dim=-1)


def hc_mixes(stream: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, cfg: Config):
    """Block.hc_mixes: streams [S, 4, H] -> [S, 24] fp32 = [pre | post | comb row-major (j input, k output)]."""
    flat = stream.flatten(1).float()
    rs = torch.rsqrt(flat.square().mean(-1, keepdim=True) + cfg.eps)
    mixes = mm(flat, fn) * rs
    pre, post, comb = K.sinkhorn_split(mixes, scale, base, cfg.hc, cfg.sinkhorn_iters, cfg.hc_eps)
    return torch.cat([pre, post, comb.flatten(-2)], dim=-1)


def hc_pre(stream: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Block.hc_pre: sum_n pre[:, n] stream[:, n] -> [S, H]."""
    return torch.sum(pre.unsqueeze(-1) * stream.float(), dim=1)


HC_ROW_BLOCK = 512  # rows per hc_post evaluation (elementwise; bounds the [rows, 4, 4, H] temporary)


def hc_post(y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """Block.hc_post: out[k] = post[k] y + sum_j comb[j, k] residual[j]; y [S, H], residual [S, 4, H] -> [S, 4, H]."""
    out = torch.empty_like(residual, dtype=torch.float32)
    for r in range(0, y.shape[0], HC_ROW_BLOCK):
        sl = slice(r, r + HC_ROW_BLOCK)
        out[sl] = post[sl].unsqueeze(-1) * y[sl].unsqueeze(-2) + torch.sum(
            comb[sl].unsqueeze(-1) * residual[sl].unsqueeze(-2), dim=1
        )
    return out


def window_rows(pos: torch.Tensor, win: int) -> torch.Tensor:
    """The window positions each query attends to, in the inference order (get_window_topk_idxs at prefill): query q
    reads max(0, q - win + 1) + [0, win); positions after q are not visible."""
    return (pos - win + 1).clamp_min(0).unsqueeze(-1) + torch.arange(win)


def pack(stream: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """streams [S, 4, H] + pending pre [S, 4] -> [S, 4H + 4]."""
    return torch.cat([stream.flatten(1).float(), pre.float()], dim=-1)


def unpack(x: torch.Tensor, hc: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[S, 4H + 4] -> (streams [S, 4, H], pending pre [S, 4])."""
    return x[:, :-PACK_PRE].unflatten(1, (hc, -1)), x[:, -PACK_PRE:]


def lowest_index_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices [B, k], ascending, of the k largest entries per row; among equal values the lowest index wins (so the
    choice does not depend on the row's length or on torch.topk's tie order)."""
    kth = scores.topk(k, dim=-1).values[:, -1:]
    greater = scores > kth
    eq = scores == kth
    need = k - greater.sum(-1, keepdim=True)
    sel = greater | (eq & (eq.cumsum(-1) <= need))
    return sel.nonzero()[:, 1].view(scores.shape[0], k)


def select_candidate_blocks(logits: torch.Tensor, compress_lens: torch.Tensor, topk_blocks: int, block: int):
    """inference select_candidate_blocks as block ids: logits [B, T] (-inf where unreachable), compress_lens [B, 1] ->
    [B, topk_blocks] int64 ascending block ids, -1 = none. A block scores its best position; the block holding the
    query's newest position is pinned; blocks scoring -inf are dropped."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block), value=float("-inf")).unflatten(-1, (-1, block)).amax(dim=-1)
    nb = scores.size(-1)
    last = (compress_lens - 1) // block
    scores = scores.masked_fill(torch.arange(nb) == last, float("inf"))
    kk = min(topk_blocks, nb)
    ids = lowest_index_topk(scores, kk)
    keep = scores.gather(-1, ids) > float("-inf")
    ids = torch.where(keep, ids, torch.full_like(ids, -1))
    ids = torch.sort(torch.where(ids < 0, torch.full_like(ids, nb), ids), dim=-1).values  # -1 last
    ids = torch.where(ids == nb, torch.full_like(ids, -1), ids)
    if kk < topk_blocks:
        ids = torch.cat([ids, ids.new_full((ids.shape[0], topk_blocks - kk), -1)], dim=-1)
    return ids


def candidate_mask(blocks: torch.Tensor, width: int, block: int) -> torch.Tensor:
    """Block ids [B, n] (-1 = none) -> bool [B, width] over compressed positions."""
    nb = -(-width // block)
    m = torch.zeros(blocks.shape[0], nb + 1, dtype=torch.bool)
    m.scatter_(1, torch.where(blocks < 0, nb, blocks).long(), True)
    return m[:, :nb].repeat_interleave(block, dim=-1)[:, :width]


# ============================================================================================= weights
class LayerWeights(dict):
    """Name (checkpoint name below ``layers.{i}.``) -> fp32 tensor, fp8 blocks dequantized; experts stay packed."""


def load_layer(ck: Checkpoint, cfg: Config, i: int) -> LayerWeights:
    p = f"layers.{i}."
    w = LayerWeights()
    for full in ck.weight_map:
        if not full.startswith(p):
            continue
        n = full[len(p) :]
        if n.endswith(".scale") or ".experts." in n or n == "ffn.gate.bias_vl" or n.startswith("engram.embed."):
            continue
        w[n] = ck.weight(full, torch.float32)
    w["attn.wo_a.weight"] = w["attn.wo_a.weight"].view(cfg.o_groups, cfg.o_lora_rank, -1)
    return w


# ============================================================================================= state
class State:
    """Per-layer prefill state (see the module docstring); created for the requested layers and their sources."""

    def __init__(self, cfg: Config, layers: list[int], max_seq: int):
        self.cfg, self.max_seq = cfg, max_seq
        self.window, self.ckv, self.index_k, self.partial, self.topk, self.cand, self.tail = ({} for _ in range(7))
        for i in sorted(set(layers) | {j for i in layers for j in cfg.sources(i)}):
            self.window[i] = torch.zeros(cfg.window, cfg.head_dim)
            r = cfg.ratio(i)
            if cfg.is_kv_source(i):
                self.ckv[i] = torch.zeros(max_seq // r, cfg.head_dim)
                self.index_k[i] = torch.zeros(max_seq // r, cfg.index_dim)
                if r > 1:
                    self.partial[i] = torch.zeros(2, r, cfg.head_dim)
            if cfg.is_index_source(i):
                self.topk[i] = torch.full((max_seq, cfg.index_topk), -1, dtype=torch.int32)
            if cfg.is_cand_source(i):
                self.cand[i] = torch.full((max_seq, cfg.cand_topk_blocks), -1, dtype=torch.int32)
            if cfg.is_engram(i):
                self.tail[i] = torch.full((cfg.engram_ngram - 1,), -1, dtype=torch.int64)


# ============================================================================================= graph
def block_steps(cfg: Config, i: int) -> list[Step]:
    steps, x = [], "in"
    if cfg.is_engram(i):
        steps += [
            Step("engram_rows", (), "engram_rows", "embedding", stateful=True),
            Step("engram_kv", ("engram_rows",), "engram_kv", "op"),
            Step("engram_add", ("in", "engram_kv"), "x", "residual"),
        ]
        x = "x"
    steps += [
        Step("attn_hc", (x,), "attn_hc", "op"),
        Step("attn_collapse", (x,), "attn_in", "op"),
        Step("attn_norm", ("attn_in",), "attn_norm", "norm"),
        Step("q_a", ("attn_norm",), "q_resid", "op"),
        Step("kv", ("attn_norm",), "kv", "op"),
    ]
    attn_in = ["q_resid", "kv"]
    if cfg.is_kv_source(i):
        steps.append(Step("compressor", ("attn_norm",), "compressed", "op", stateful=True))
        attn_in.append("compressed")
    if cfg.is_index_source(i):
        idx_in = ("attn_norm", "q_resid") + (("compressed",) if cfg.is_kv_source(i) else ())
        steps.append(Step("indexer", idx_in, "topk", "op", stateful=True))
        attn_in.append("topk")
    steps += [
        Step("attention", tuple(attn_in), "attn_out", "attention", stateful=True),
        Step("attn_residual", (x, "attn_hc", "attn_out"), "h_mid", "residual"),
        Step("ffn_hc", ("h_mid",), "ffn_hc", "op"),
        Step("ffn_collapse", ("h_mid", "attn_hc"), "ffn_in", "op"),
        Step("ffn_norm", ("ffn_in",), "ffn_norm", "norm"),
        Step("router", ("ffn_norm",), "router", "router"),
        Step("experts", ("ffn_norm", "router"), "experts_out", "moe"),
        Step("shared_expert", ("ffn_norm",), "shared_out", "mlp"),
        Step("moe_add", ("experts_out", "shared_out"), "mlp_out", "residual"),
        Step("ffn_residual", ("h_mid", "ffn_hc", "mlp_out"), "out", "residual"),
    ]
    return steps


# ============================================================================================= model
class DS41Reference:
    """Implements models/demos/common/bringup/reference/interface.py; keeps the requested layers' dense weights
    resident (fp32) and reads each routed expert from the checkpoint when it runs."""

    def __init__(
        self,
        model_path: str,
        layers=None,
        dtype=torch.float32,
        lm_head: bool = True,
        tokenizer=None,
        numerics: str = "official",
    ):
        assert numerics in NUMERICS, f"numerics {numerics!r} not in {NUMERICS}"
        dtype = torch.float32 if dtype is None else dtype
        assert dtype in (torch.float32, torch.bfloat16), f"dtype {dtype}: fp32, or bf16 (official, the GPU's)"
        assert numerics == "official" or dtype == torch.float32, "native is fp32 (the golden target)"
        self.numerics = numerics
        self.cfg = cfg = Config.from_dir(model_path)
        self.ck = Checkpoint(model_path)
        self.dtype = dtype  # official's activation dtype (bf16: the GPU configuration); the arithmetic stays fp32
        self.layer_ids = list(range(cfg.n_layers)) if layers is None else sorted(layers)
        self.embed_w = self.ck.weight("embed.weight", torch.float32)
        self.final_norm_w = self.ck.weight("norm.weight", torch.float32)
        self.head_w = self.ck.weight("head.weight", torch.float32) if lm_head else None
        self.w = {i: load_layer(self.ck, cfg, i) for i in self.layer_ids}
        self.tables = {i: EngramTable(self.ck, i) for i in self.layer_ids if cfg.is_engram(i)}
        self.hasher = None
        if self.tables:
            from models.demos.deepseek_v41_flash_d_p.reference.ngram import EngramHasher

            if tokenizer is None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.hasher = EngramHasher.build(cfg.raw, tokenizer)
        r = cfg.rope
        self.freqs = {
            0: yarn_freqs(cfg.rope_dim, r["theta"], 0, r["factor"], r["beta_fast"], r["beta_slow"]),
            1: yarn_freqs(
                cfg.rope_dim, r["compress_theta"], r["original"], r["factor"], r["beta_fast"], r["beta_slow"]
            ),
        }

    # ---------------------------------------------------------------- numerics
    @property
    def native(self) -> bool:
        return self.numerics == "native"

    @property
    def bf16(self) -> bool:
        """official at dtype bf16: activations rounded to bf16 where the GPU code holds them in bf16."""
        return self.numerics == "official" and self.dtype == torch.bfloat16

    def with_numerics(self, numerics: str, dtype=None) -> "DS41Reference":
        """The same reference (weights, tables, hasher shared, not copied) with the other arithmetic; native is fp32,
        official keeps this one's dtype unless ``dtype`` is given."""
        import copy

        assert numerics in NUMERICS, f"numerics {numerics!r} not in {NUMERICS}"
        dtype = torch.float32 if numerics == "native" else (dtype or self.dtype)
        assert numerics == "official" or dtype == torch.float32, "native is fp32 (the golden target)"
        other = copy.copy(self)
        other.numerics, other.dtype = numerics, dtype
        return other

    def _r(self, x: torch.Tensor) -> torch.Tensor:
        """A value model.py holds in bf16 (official at bf16): x rounded to bf16 (round to nearest even), kept in an fp32
        tensor; otherwise x itself (fp32 official and native are unchanged)."""
        return x.to(torch.bfloat16).float() if self.bf16 else x

    def _mm16(self, x: torch.Tensor, w: torch.Tensor, fn=mm) -> torch.Tensor:
        """A GEMM model.py runs on bf16 x and a bf16 weight (wo_a, ratio-1 compressor wkv, indexer wk / weights_proj;
        fn = mm or K.grouped_out): at bf16 the same bf16 GEMM (fp32 accumulation, output rounded to bf16; the oracle's
        summation order), returned in fp32; otherwise the fp32 GEMM of today."""
        if not self.bf16:
            return fn(x.float(), w)
        return fn(x.to(torch.bfloat16), w.to(torch.bfloat16)).float()

    def _unpack(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """unpack, the streams as model.py holds them (bf16 at bf16; the pending pre stays fp32)."""
        s, pre = unpack(x, self.cfg.hc)
        return self._r(s), pre

    def _q8(self, x: torch.Tensor) -> torch.Tensor:
        """The fp8 act quant-dequant (per 32, ue8m0) an fp8 GEMM input / the window KV gets; native: x in fp32."""
        return x.float() if self.native else K.act_quant_dequant(x, 32)

    def _q4(self, x: torch.Tensor, block: int, e4m3_scale: bool = False) -> torch.Tensor:
        """The fp4 quant-dequant of the compressed KV / index key / index query; native: x in fp32 (nudged inside its
        fp4 cell in a noise-floor run of a step marked nudge_inner, F224)."""
        if self.native:
            from models.demos.common.bringup.testing import noise_floor as NF

            return NF.nudge_inner(x.float(), f"fp4/{block}" if e4m3_scale else f"fp4e8/{block}")
        return K.fp4_quant_dequant(x, block, e4m3_scale=e4m3_scale)

    def _fp8_linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """fp8_linear (official) or the plain fp32 GEMM of x (native)."""
        return mm(self._q8(x), w)

    # ---------------------------------------------------------------- interface
    def new_state(self, max_seq: int) -> State:
        return State(self.cfg, self.layer_ids, max_seq)

    def state_tensors(self, state: State, layer: int, length: int) -> dict:
        cfg, out = self.cfg, {}
        for n in cfg.state_names(layer):
            if n == "window_kv":
                out[n] = state.window[layer].clone()
            elif n == "compress_kv":
                out[n] = state.ckv[layer][: length // cfg.ratio(layer)].clone()
            elif n == "index_k":
                out[n] = state.index_k[layer][: length // cfg.ratio(layer)].clone()
            elif n == "compress_partial":
                out[n] = state.partial[layer].clone()
            elif n == "topk_idx":
                out[n] = state.topk[layer][:length].clone()
            elif n == "candidates":
                out[n] = state.cand[layer][:length].clone()
            elif n == "engram_tail":
                out[n] = state.tail[layer].clone()
        return out

    def load_state(self, state: State, layer: int, tensors: dict, length: int) -> None:
        cfg = self.cfg
        for n in cfg.state_names(layer):
            t = tensors[n]
            if n == "window_kv":
                state.window[layer].copy_(t)
            elif n == "compress_kv":
                g = length // cfg.ratio(layer)
                state.ckv[layer][:g] = t[:g].float()
            elif n == "index_k":
                g = length // cfg.ratio(layer)
                state.index_k[layer][:g] = t[:g].float()
            elif n == "compress_partial":
                state.partial[layer].copy_(t)
            elif n == "topk_idx":
                state.topk[layer][:length] = t[:length].to(torch.int32)
            elif n == "candidates":
                state.cand[layer][:length] = t[:length].to(torch.int32)
            elif n == "engram_tail":
                state.tail[layer].copy_(t)

    def block_graph(self, layer: int) -> list[Step]:
        return block_steps(self.cfg, layer)

    def chunk_context(self, layer: int, start: int, length: int, state) -> Ctx:
        return Ctx(layer, start, length, state)

    def component(self, layer: int, name: str):
        cfg, w, hc = self.cfg, self.w[layer], self.cfg.hc

        r = self._r  # model.py's bf16 points (identity unless official at bf16)

        def attn_hc(ctx, x):
            s, _ = self._unpack(x)
            return hc_mixes(s, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"], cfg)

        def attn_collapse(ctx, x):
            s, pre = self._unpack(x)
            return r(hc_pre(s, pre))

        def attn_residual(ctx, x, mix, y):
            s, _ = self._unpack(x)
            return r(hc_post(r(y), s, mix[:, hc : 2 * hc], mix[:, 2 * hc :].unflatten(-1, (hc, hc))).flatten(1))

        def ffn_hc(ctx, h):
            return hc_mixes(r(h).unflatten(1, (hc, -1)), w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"], cfg)

        def ffn_collapse(ctx, h, attn_mix):
            return r(hc_pre(r(h).unflatten(1, (hc, -1)), attn_mix[:, :hc]))

        def ffn_residual(ctx, h, mix, y):
            s = hc_post(
                r(y), r(h).unflatten(1, (hc, -1)), mix[:, hc : 2 * hc], mix[:, 2 * hc :].unflatten(-1, (hc, hc))
            )
            return pack(r(s), mix[:, :hc])

        table = {
            "engram_rows": lambda ctx: self._engram_rows(layer, ctx),
            "engram_kv": lambda ctx, rows: r(self._fp8_linear(rows, w["engram.wkv.weight"])),
            "engram_add": lambda ctx, x, kv: self._engram_add(layer, x, kv),
            "attn_hc": attn_hc,
            "attn_collapse": attn_collapse,
            "attn_norm": lambda ctx, x: r(rms_norm(r(x), w["attn_norm.weight"], cfg.eps)),
            "q_a": lambda ctx, x: r(
                rms_norm(r(self._fp8_linear(r(x), w["attn.wq_a.weight"])), w["attn.q_norm.weight"], cfg.eps)
            ),
            "kv": lambda ctx, x: self._window_kv(layer, ctx, x),
            "compressor": lambda ctx, x: self._compressor(layer, ctx, x),
            "indexer": lambda ctx, x, qr, *c: self._indexer(layer, ctx, x, qr, *c),
            "attention": lambda ctx, qr, kv, *rest: self._attention(layer, ctx, qr, kv, *rest),
            "attn_residual": attn_residual,
            "ffn_hc": ffn_hc,
            "ffn_collapse": ffn_collapse,
            "ffn_norm": lambda ctx, x: r(rms_norm(r(x), w["ffn_norm.weight"], cfg.eps)),
            "router": lambda ctx, x: self._route(layer, r(x)),
            "experts": lambda ctx, x, rt: self._experts(layer, r(x), rt),
            "shared_expert": lambda ctx, x: self._swiglu(
                r(x), *(w[f"ffn.shared_experts.{n}.weight"] for n in ("w1", "w3", "w2")), None, lin=mm
            ),
            "moe_add": lambda ctx, a, b: r(a + r(b)),
            "ffn_residual": ffn_residual,
        }
        return table[name]

    # ---------------------------------------------------------------- engram
    def _engram_rows(self, layer: int, ctx: Ctx) -> torch.Tensor:
        st = ctx.state
        if ctx.start == 0:
            st.tail[layer].fill_(-1)
        ids, st.tail[layer] = self.hasher.hashes(ctx.extra["tokens"], layer, st.tail[layer])
        return self._r(self.tables[layer].rows(ids).flatten(1))  # ParallelEngramEmbedding: .to(bf16)

    def _engram_add(self, layer: int, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        cfg, w = self.cfg, self.w[layer]
        s, pre = self._unpack(x)
        kv = self._r(kv)
        key = kv[:, : cfg.hc * cfg.dim].float().unflatten(-1, (cfg.hc, cfg.dim))
        value = kv[:, cfg.hc * cfg.dim :].float()
        weight = w["engram.q_weight"].float() * w["engram.k_weight"].float()
        h, eps = s.float(), cfg.eps
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * cfg.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return pack(self._r(h + gate.unsqueeze(-1) * value.unsqueeze(-2)), pre)  # .to(x.dtype)

    # ---------------------------------------------------------------- attention parts
    def _rot(self, layer: int, pos: torch.Tensor) -> torch.Tensor:
        return rope_table(self.freqs[1 if self.cfg.ratio(layer) else 0], pos)

    def _window_kv(self, layer: int, ctx: Ctx, x: torch.Tensor) -> torch.Tensor:
        cfg, w = self.cfg, self.w[layer]
        r = self._r
        kv = r(rms_norm(r(self._fp8_linear(r(x), w["attn.wkv.weight"])), w["attn.kv_norm.weight"], cfg.eps))
        kv = r(rope(kv, self._rot(layer, torch.arange(ctx.start, ctx.start + x.shape[0]))))
        return r(self._q8(kv))

    def _compressor(self, layer: int, ctx: Ctx, x: torch.Tensor) -> torch.Tensor:
        cfg, w, st = self.cfg, self.w[layer], ctx.state
        r, s0, b = cfg.ratio(layer), ctx.start, self._r
        x = b(x)
        if s0 == 0 and layer in st.partial:  # a new sequence (each step resets only the state it owns)
            st.partial[layer].zero_()
        if r == 1:  # bf16 wkv on bf16 x at bf16 (model.py: self.norm(self.wkv(x)))
            latent = b(
                rms_norm(self._mm16(x, w["attn.compressor.wkv.weight"]), w["attn.compressor.norm.weight"], cfg.eps)
            )
        else:
            kv = mm(x.float(), w["attn.compressor.wkv.weight"])
            sc = mm(x.float(), w["attn.compressor.wgate.weight"])
            p = s0 % r  # tokens of the unfinished group carried in the state
            kv = torch.cat([st.partial[layer][0, :p], kv])
            sc = torch.cat([st.partial[layer][1, :p], sc])
            n = kv.shape[0] // r
            rem = kv.shape[0] - n * r
            st.partial[layer].zero_()
            st.partial[layer][0, :rem] = kv[n * r :]
            st.partial[layer][1, :rem] = sc[n * r :]
            pooled = (kv[: n * r].view(n, r, -1) * sc[: n * r].view(n, r, -1).softmax(dim=1)).sum(dim=1)
            latent = b(rms_norm(b(pooled), w["attn.compressor.norm.weight"], cfg.eps))  # self.norm(kv.to(dtype))
        g0 = s0 // r
        n = latent.shape[0]
        rot = self._rot(layer, (g0 + torch.arange(n)) * r)  # a group sits at its first token's position
        k = b(rms_norm(self._mm16(latent, w["attn.indexer.wk.weight"]), w["attn.indexer.k_norm.weight"], cfg.eps))
        k = b(self._q4(b(rope(k, rot)), 32))
        ckv = b(self._q4(b(rope(latent, rot)), 16, e4m3_scale=True))
        st.ckv[layer][g0 : g0 + n] = ckv
        st.index_k[layer][g0 : g0 + n] = k
        return torch.cat([ckv, k], dim=-1)

    def _indexer(self, layer: int, ctx: Ctx, x: torch.Tensor, qr: torch.Tensor, compressed=None) -> torch.Tensor:
        cfg, w, st = self.cfg, self.w[layer], ctx.state
        r, s0, s = cfg.ratio(layer), ctx.start, x.shape[0]
        end, nh, hd = s0 + s, cfg.index_heads, cfg.index_dim
        n_keys = end // r
        if compressed is not None:  # this layer's own index keys: the prefix from the state, this chunk's from input
            keys = torch.cat([st.index_k[layer][: s0 // r], compressed[:, cfg.head_dim :].float()])
        else:
            keys = st.index_k[cfg.kv_source_of(layer)][:n_keys]
        assert keys.shape[0] == n_keys
        pos = torch.arange(s0, end)
        rb = self._r
        q = rb(self._fp8_linear(rb(qr), w["attn.indexer.wq_b.weight"])).view(s, nh, hd)
        q = rb(self._q4(rb(rope(q, self._rot(layer, pos))), 32))
        wts = rb(self._mm16(rb(x), w["attn.indexer.weights_proj.weight"]) * (hd**-0.5 * nh**-0.5))
        topk = min(cfg.index_topk, n_keys)
        out = torch.full((s, cfg.index_topk), -1, dtype=torch.int32)
        if self.bf16:  # model.py: einsum('bshd,btd->bsht') of bf16 q and keys, relu x weights and the head sum in bf16
            q, keys, wts = q.to(torch.bfloat16), keys.to(torch.bfloat16), wts.to(torch.bfloat16)
        for t0 in range(0, s, K.INDEX_Q_BLOCK):
            t1 = min(s, t0 + K.INDEX_Q_BLOCK)
            b = t1 - t0
            lens = ((pos[t0:t1] + 1) // r).unsqueeze(-1)  # visible compressed positions per query
            width = int(lens[-1])  # keys any query of the block can see (scores beyond are -inf for all)
            qb = K._pad_rows(q[t0:t1], K.INDEX_Q_BLOCK)
            wb = wts[t0:t1].unsqueeze(-1)
            score = torch.full((b, n_keys), float("-inf"))
            for k0 in range(0, width, K.INDEX_K_BLOCK):
                sc = K.index_score_tile(qb, keys, k0)[:b]  # [b, nh, nk]
                score[:, k0 : k0 + sc.shape[-1]] = (sc.relu_() * wb).sum(dim=1)
            score = score.masked_fill(torch.arange(n_keys) >= lens, float("-inf"))
            if cfg.is_cand_source(layer):
                blocks = select_candidate_blocks(score, lens, cfg.cand_topk_blocks, cfg.cand_block)
                st.cand[layer][s0 + t0 : s0 + t1] = blocks.to(torch.int32)
            elif cfg.uses_candidates(layer):
                blocks = st.cand[cfg.cand_source][s0 + t0 : s0 + t1]
                score = score.masked_fill(~candidate_mask(blocks, n_keys, cfg.cand_block), float("-inf"))
            if topk:
                idx = lowest_index_topk(score, topk)
                out[t0:t1, :topk] = torch.where(idx < lens, idx, -1).to(torch.int32)
        st.topk[layer][s0:end] = out
        return out

    def _attention(self, layer: int, ctx: Ctx, qr: torch.Tensor, kv: torch.Tensor, *rest) -> torch.Tensor:
        cfg, w, st = self.cfg, self.w[layer], ctx.state
        s0, s, win, d = ctx.start, qr.shape[0], cfg.window, cfg.head_dim
        r = cfg.ratio(layer)
        if s0 == 0:
            st.window[layer].zero_()
        pos = torch.arange(s0, s0 + s)
        rot = self._rot(layer, pos)
        b = self._r
        q = b(self._fp8_linear(b(qr), w["attn.wq_b.weight"])).view(s, cfg.n_heads, d)
        q = b(rope(q, rot))
        # window keys: ext holds positions s0 - win .. s0 + s - 1 (the previous chunk's tail from the ring)
        prev_pos = torch.arange(s0 - win, s0)
        ext = torch.cat([st.window[layer][prev_pos.remainder(win)], b(kv).float()])
        wpos = window_rows(pos, win)  # [S, win] positions, inference order
        wvalid = (wpos <= pos.unsqueeze(-1)) & (wpos < s0 + s)
        wrow = (wpos - (s0 - win)).clamp(max=ext.shape[0] - 1)
        if r:
            rest = list(rest)
            compressed = rest.pop(0) if cfg.is_kv_source(layer) else None
            topk = rest.pop(0) if cfg.is_index_source(layer) else st.topk[cfg.index_source_of(layer)][s0 : s0 + s]
            if compressed is not None:
                ckv = torch.cat([st.ckv[layer][: s0 // r], b(compressed[:, :d]).float()])
            else:
                ckv = st.ckv[cfg.kv_source_of(layer)][: (s0 + s) // r]
            topk = topk[:, : min(cfg.index_topk, (s0 + s) // r)].long()  # inference width: min(topk, end // r)
        sink = w["attn.attn_sink"].float()
        qa = q.to(torch.bfloat16) if self.bf16 else q  # kernels.attend: bf16 q -> P rounded to bf16 before P @ V
        o = torch.empty(s, cfg.n_heads, d)
        for t0 in range(0, s, Q_BLOCK):
            t1 = min(s, t0 + Q_BLOCK)
            keys = ext[wrow[t0:t1]]  # [B, win, d]
            valid = wvalid[t0:t1]
            if r:
                idx = topk[t0:t1]
                keys = torch.cat([keys, ckv[idx.clamp_min(0)]], dim=1)
                valid = torch.cat([valid, idx >= 0], dim=1)
            o[t0:t1] = b(K.attend(qa[t0:t1], keys, valid, sink, d**-0.5))  # sparse_attn: .to(q.dtype)
        # the ring keeps the last `win` positions
        tail = min(s, win)
        st.window[layer][pos[-tail:].remainder(win)] = b(kv[-tail:]).float()
        o = b(rope(o, rot, inverse=True)).view(s, cfg.o_groups, -1)
        o = b(self._q8(o))
        y = self._mm16(o, w["attn.wo_a.weight"], K.grouped_out)  # wo_a [groups, o_lora_rank, heads_per_group * d]
        return b(self._fp8_linear(y.flatten(1), w["attn.wo_b.weight"]))

    # ---------------------------------------------------------------- MoE
    def _route(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        cfg, w = self.cfg, self.w[layer]
        scores = F.softplus(mm(x.float(), w["ffn.gate.weight"])).sqrt()
        idx = (scores + w["ffn.gate.bias"].float()).topk(cfg.topk_experts, dim=-1)[1]
        wts = scores.gather(1, idx)
        wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-20) * cfg.route_scale
        return torch.zeros_like(scores).scatter_(1, idx, wts)

    def _swiglu(self, x, w1, w3, w2, weights, lin=K.expert_linear) -> torch.Tensor:
        """inference Expert.forward: w2(r * silu(min(w1 x, lim)) * clamp(w3 x, +-lim)), fp8-quantized GEMM inputs.
        ``lin``: the routed-expert GEMM (one token group) or mm (the shared expert, fixed row tiles)."""
        xq = self._q8(x)
        lim = self.cfg.swiglu_limit
        gate, up = self._r(lin(xq, w1)), self._r(lin(xq, w3))  # fp8 / fp4 GEMMs write bf16, then .float()
        if lim > 0:
            up = up.clamp(min=-lim, max=lim)
            gate = gate.clamp(max=lim)
        h = F.silu(gate) * up
        if weights is not None:
            h = weights * h
        return self._r(lin(self._q8(self._r(h)), w2))  # self.w2(x.to(dtype))

    def _experts(self, layer: int, x: torch.Tensor, routing: torch.Tensor) -> torch.Tensor:
        """sum over the selected experts of routing[t, e] * expert_e(x[t]), in expert-id order (the inference MoE
        order), tokens grouped per expert (kernels.expert_linear pads a group to 32-row multiples)."""
        out = torch.zeros(x.shape, dtype=torch.float32)
        tok_all, exp_all = (routing != 0).nonzero(as_tuple=True)
        order = torch.argsort(exp_all, stable=True)
        tok_all, exp_all = tok_all[order], exp_all[order]
        experts, counts = torch.unique_consecutive(exp_all, return_counts=True)
        experts, counts = experts.tolist(), counts.tolist()
        pf = ExpertPrefetcher(self.ck, layer, experts)
        off = 0
        for e, n in zip(experts, counts):
            tok = tok_all[off : off + n]
            off += n
            ws = pf.get(e)
            y = self._swiglu(x[tok].float(), ws["w1"][2], ws["w3"][2], ws["w2"][2], routing[tok, e, None].float())
            out.index_add_(0, tok, y)
            del ws
        return out

    # ---------------------------------------------------------------- forward
    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.embedding(tokens.long(), self.embed_w)

    def first_block_input(self, e: torch.Tensor) -> torch.Tensor:
        """Layer 0's input: the embedding on all 4 streams, pending pre one-hot on stream 0."""
        pre = torch.zeros(e.shape[0], self.cfg.hc)
        pre[:, 0] = 1.0
        return pack(e.unsqueeze(1).expand(-1, self.cfg.hc, -1), pre)

    def final_norm(self, h: torch.Tensor) -> torch.Tensor:
        s, pre = self._unpack(h)
        return self._r(rms_norm(self._r(hc_pre(s, pre)), self.final_norm_w, self.cfg.eps))

    @torch.no_grad()
    def forward_chunk(self, tokens, start, state, rec=noop, logits_last_n=0):
        """tokens [S] at absolute positions [start, start + S). Returns (final_norm [S, H], logits [n, V] | None)."""
        tokens = tokens.long()
        e = self.embed(tokens)
        rec("embed", e)
        h = self.first_block_input(e)
        for i in self.layer_ids:
            ctx = self.chunk_context(i, start, tokens.shape[0], state)
            ctx.extra["tokens"] = tokens
            h = run_block(self.block_graph(i), lambda name, i=i: self.component(i, name), ctx, h, rec, prefix=f"L{i}.")
        out = self.final_norm(h)
        rec("final_norm", out)
        logits = None
        if logits_last_n and self.head_w is not None:
            logits = self.logits(out[-logits_last_n:])
            rec("logits", logits)
        return out, logits

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return mm(hidden.float(), self.head_w)
