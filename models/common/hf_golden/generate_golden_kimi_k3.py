#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for moonshotai/Kimi-K3, layer by layer (engine.py), from the checkpoint's remote code
(modeling_kimi_linear.py: KimiLinearForCausalLM inside KimiK3ForConditionalGeneration; trust_remote_code).

Stored tensors (N = isl, i = layer; 93 layers. The config's kda_layers / full_attn_layers are 1-based: KDA on 69
layers, MLA on 3, 7, ..., 87, 91, 92 (0-based). Dense MLP on layer 0, latent MoE (896 routed experts, top 16,
2 shared) on 1-92. AttnRes seals a block at every layer i % 12 == 0):

  hidden/embed.safetensors
    model.embed_tokens.output                                [1, N, 7168]     bf16  embeddings (no scale)
  hidden/layer_XX.safetensors
    model.layers.{i}.output                                  [1, N, 7168]     bf16  all   the layer's prefix_sum
                                                                                    (AttnRes, below)
    model.layers.{i}.input_layernorm.input                   [1, N, 7168]     bf16  all   --sublayers: the AttnRes
                                                                                    read attention sees
    model.layers.{i}.post_attention_layernorm.input          [1, N, 7168]     bf16  all   --sublayers: the AttnRes
                                                                                    read the MLP sees
    model.layers.{i}.self_attn.output                        [1, N, 7168]     bf16  all   --sublayers: KDA / MLA
                                                                                    after o_proj
    model.layers.{i}.mlp.output                              [1, N, 7168]     bf16  0     --sublayers: dense MLP
    model.layers.{i}.block_sparse_moe.output                 [1, N, 7168]     bf16  1-92  --sublayers: routed +
                                                                                    shared experts
    model.layers.{i}.block_sparse_moe.routed_expert_up_proj.output
                                                             [N, 7168]        bf16  1-92  --sublayers: routed experts
                                                                                    (latent, normed, up-projected)
    model.layers.{i}.block_sparse_moe.shared_experts.output  [1, N, 7168]     bf16  1-92  --sublayers
  cache/layer_XX.safetensors
    past_key_values.layers.{i}.recurrent_state               [1, 96, 128, 128] fp32 KDA   final KDA state, V-first
                                                                                    [V, K] (state_v_first)
    past_key_values.layers.{i}.conv_state_{q,k,v}            [1, 12288, W]    bf16  KDA   short-conv caches, as
                                                                                    fla's ShortConvolution keeps them
    model.layers.{i}.self_attn.kv_a_proj_with_mqa.output     [1, N, 576]      bf16  MLA   [c_kv 512 | k_rot 64], raw
    model.layers.{i}.self_attn.kv_a_layernorm.output         [1, N, 512]      bf16  MLA   the MLA latent cache:
                                                                                    kv_a_layernorm(c_kv)
    model.layers.{i}.block_sparse_moe.gate.output            [N, 896]         fp32  1-92  router logits (fp32)
    model.layers.{i}.block_sparse_moe.gate.top_k_index       [N, 16]          int32 1-92  chosen experts, ascending
    model.layers.{i}.block_sparse_moe.gate.top_k_weights     [N, 16]          fp32  1-92  their weights, same order
  hidden/final.safetensors
    model.last_hidden_state                                  [1, N, 7168]     bf16  final norm of the output AttnRes
    logits                                                   [1, 1, 163840]   fp32  last position, untied lm_head

REQUIRES fla-core (the remote code imports it, and KDA, its short convolutions and its gated norm run through it;
CUDA only), einops, and blobfile for the remote tiktoken tokenizer. Run through a uv overlay on top of python_env:

    uv run --no-project --python python_env/bin/python --with fla-core==0.6.0 --with einops --with blobfile \\
        --with compressed-tensors==0.20.0 python models/common/hf_golden/generate_golden_kimi_k3.py --isl 56320

What is model-specific, and why it is handled here:

  * Remote code written for transformers 4.56+, run on 5.12 with shims: OutputRecorder is
    aliased where it used to live (utils.generic); KimiLinearModel.__init__ forces flash_attention_2 whatever was
    asked, so the requested --attn is put back after every build (only the forced FA2 path pads V from 128 to 192);
    and create_causal_mask, which the remote model calls with 4.x arguments, is a plain causal mask (only the
    --selftest reference's whole-model forward calls it: the layer-by-layer run builds its masks itself) (these
    two in _patch_remote); and KimiK3ForConditionalGeneration.tie_weights(), which 5.12's post_init calls with
    recompute_mapping=, takes 5.12's arguments and forwards them to language_model.tie_weights as upstream's override
    does (tie_word_embeddings is false), and KimiLinearForCausalLM's 4.x _tied_weights_keys list ["lm_head.weight"]
    becomes 5.x's {target: source} dict, {"lm_head.weight": "model.embed_tokens.weight"} (save_pretrained reads it).
    KimiK3PreTrainedModel declares _supports_sdpa = False (from the LLaVA port it was written from) and
    KimiPreTrainedModel (the text model's) declares only flash_attention_2, which 5.12 enforces at construction; the
    text model's MLA calls ALL_ATTENTION_FUNCTIONS[_attn_implementation], so both are set True for the default
    --attn sdpa. These run in the constructor, so they go on the classes as transformers loads them
    (_shim_remote_classes), before any build.
  * KDA in torch (HF_GOLDEN_KDA=torch, the default): the remote layer calls fla's Triton chunk_kda; torch_chunk_kda
    replaces it with fla's own pure-torch references, the ones fla's tests grade chunk_kda against in this exact
    configuration (tests/ops/test_kda.py: gate in kernel, lower bound -5, qk L2 norm): l2-normalized q / k, sigmoid
    beta, naive_kda_lowerbound_gate, naive_recurrent_kda, everything fp32 and token by token (state [96, 128, 128]
    fp32, so memory is ~O(N), no [N/64, 64, 64, K] intermediates). On CUDA it is first checked against fla's
    chunk_kda with K3's arguments at fla's own tolerance (check_torch_kda; result in metadata.json).
    HF_GOLDEN_KDA=triton keeps the remote code's chunk_kda instead (bf16 dots inside, TF32 in its triangular solve).
  * MLA without RoPE (mla_use_nope): [nope 128 | rope 64] q / k heads, but the "rope" dims are never rotated, so
    there are no positions anywhere in the model. The HF cache holds K / V expanded to all 96 heads (~3.5 GB per
    layer at N = 56320); the device holds the latent, so that is what is stored. Output gate: sigmoid(g_proj(x)).
  * AttnRes (attn_res_block_size 12): each layer reads its input as a softmax mix over [sealed blocks..., the live
    prefix_sum] (scores = RMS-normalized candidate . (norm.weight * proj.weight), fp32), once before attention
    (self_attention_res_*) and once before the MLP (mlp_res_*). At i % 12 == 0 the incoming prefix_sum is sealed
    as a block and prefix_sum restarts from that layer's own attention output, so model.layers.{i}.output is the
    sum of attention + MLP outputs since the last seal, not a residual stream. The sealed blocks are the embeddings
    and model.layers.{11, 23, ..., 83}.output: nothing more needs storing. After the last layer an output AttnRes
    (output_attn_res_*) mixes them once more, then the final norm (model.last_hidden_state). The blocks are carried
    between layers (state.carry), which --prefill-chunk does not support.
  * Latent MoE: the router (fp32 GEMM of the bf16 weight; sigmoid; top-16 of sigmoid + e_score_correction_bias,
    one group; renormalized, x routed_scaling_factor 1.0) reads the hidden size; the routed experts run in a 3584
    latent (routed_expert_down_proj, experts, routed_expert_norm, routed_expert_up_proj); SiTU activation
    (4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25), fp32) everywhere. e_score_correction_bias, A_log, dt_bias and the
    short-conv weights are fp32 in the checkpoint and cast to the compute dtype, as from_pretrained casts them.
  * Routed experts are MXFP4 (compressed-tensors mxfp4-pack-quantized: E2M1 pairs, low nibble first, x 2^(e - 127)
    per 32 inputs); convert() dequantizes them, bit-identical to compressed_tensors' own decompress (checked on
    the real layer 1 expert 0, and at run time on the first expert when compressed_tensors imports: metadata.json
    "dequant_check"). Everything else is bf16. A dequantized MoE layer is ~59 GB in bf16; experts are not fused,
    so that is the only copy, plus the packed bytes while they are read.
  * Memory at N = 56320 (TOKEN_CHUNK, lean_moe_infer, HF_GOLDEN_EXPERTS). Upstream's _apply_attn_res makes ~4 fp32
    copies of [N, blocks + 1, 7168] (~51 GB with 8 blocks) and its moe_infer ~5 copies of [N x 16, 3584] (~32 GB);
    both are per token, so _apply_attn_res runs TOKEN_CHUNK tokens at a time and moe_infer is replaced by
    lean_moe_infer: the same expert calls on the same rows and the same weighted sum, without the extra copies.
    HF_GOLDEN_EXPERTS=host keeps the dequantized routed experts in host RAM (~59 GB per layer) and copies each to
    the device for its one call: same GEMMs, ~2 GB of weights on the device instead of ~61 GB. auto (default):
    host on a GPU under 120 GiB (H100), device otherwise (H200), host under --selftest so the path is tested.
  * A_log is stored zero-padded to 128 (num_heads 96): the padding is checked to be zero and dropped.
  * Tokenizer: remote code (TikTokenTokenizer); it adds no BOS ([BOS] = 163584). The prefix is [BOS], as for
    Kimi-K2.7-Code (where it scored best); not yet measured on K3.

    python generate_golden_kimi_k3.py --selftest --device cuda               # first: tiny model, every path above
    python generate_golden_kimi_k3.py --isl 1024 --num-layers 5 --root /tmp/smoke   # dense KDA, MoE, first MLA
    python generate_golden_kimi_k3.py --isl 56320 --attn-chunk 1024
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import LayerwiseGolden, Recipe, RunState  # noqa: E402

NOTES = {
    "model.embed_tokens.output": "Token embeddings (no scale, no BOS added). AttnRes's first sealed block.",
    "model.layers.{i}.output": (
        "The layer's prefix_sum: its attention + MLP outputs added to the incoming prefix_sum, except at i % 12 == 0, "
        "where the incoming one is sealed as a block and prefix_sum restarts from this layer's attention output. "
        "Not the next layer's input: that is the AttnRes read (input_layernorm.input)."
    ),
    "model.layers.{i}.input_layernorm.input": (
        "AttnRes read before attention (--sublayers): softmax over [sealed blocks, prefix_sum] of the RMS-normalized "
        "candidates . (self_attention_res_norm.weight * self_attention_res_proj.weight), fp32, mixing the raw "
        "candidates. Equal to the incoming prefix_sum on layer 0 (no block sealed yet)."
    ),
    "model.layers.{i}.post_attention_layernorm.input": (
        "AttnRes read before the MLP (--sublayers), as input_layernorm.input with mlp_res_*, over the blocks "
        "including any sealed by this layer and the prefix_sum after attention."
    ),
    "model.layers.{i}.self_attn.output": (
        "Attention after o_proj (--sublayers). KDA: q / k / v projections, short causal conv + SiLU, KDA "
        "(l2-normalized q / k, sigmoid beta, gate -5 sigmoid(exp(A_log) (f_b(f_a(x)) + dt_bias))), gated RMSNorm "
        "with sigmoid(g_proj(x)), o_proj. MLA: no RoPE, x sigmoid(g_proj(x)) before o_proj."
    ),
    "model.layers.{i}.mlp.output": "Dense SiTU MLP of layer 0 (--sublayers).",
    "model.layers.{i}.block_sparse_moe.output": "Routed experts (up-projected from the latent) + shared experts.",
    "model.layers.{i}.block_sparse_moe.routed_expert_up_proj.output": (
        "The routed experts' contribution [N, hidden] (--sublayers): routed_expert_down_proj to the 3584 latent, the "
        "16 chosen experts weighted by top_k_weights and summed, routed_expert_norm, routed_expert_up_proj."
    ),
    "model.layers.{i}.block_sparse_moe.shared_experts.output": "The 2 shared experts as one SiTU MLP (--sublayers).",
    "past_key_values.layers.{i}.recurrent_state": (
        "KDA state after the last position [1, num_heads, V, K], fp32, V-first (the remote code's "
        "transpose_state_layout): S[v, k], read as o = q . S."
    ),
    "past_key_values.layers.{i}.conv_state_q": "q short-conv cache (the last inputs fla's ShortConvolution keeps).",
    "past_key_values.layers.{i}.conv_state_k": "k short-conv cache, as conv_state_q.",
    "past_key_values.layers.{i}.conv_state_v": "v short-conv cache, as conv_state_q.",
    "model.layers.{i}.self_attn.kv_a_proj_with_mqa.output": (
        "MLA [c_kv (kv_lora_rank 512) | k_rot (qk_rope_head_dim 64)], raw: k_rot is never rotated (mla_use_nope) and "
        "is shared by all heads."
    ),
    "model.layers.{i}.self_attn.kv_a_layernorm.output": (
        "MLA latent cache kv_a_layernorm(c_kv) [1, N, 512]; kv_b_proj expands it to 96 heads of [k_nope 128 | v 128]."
    ),
    "model.layers.{i}.block_sparse_moe.gate.output": (
        "Router logits [N, num_experts] = post_attention_layernorm output @ gate.weight.T, both in fp32, before the "
        "sigmoid. Selection: top-16 of sigmoid(logits) + e_score_correction_bias."
    ),
    "model.layers.{i}.block_sparse_moe.gate.top_k_index": "The 16 chosen experts per token, ascending by id. int32.",
    "model.layers.{i}.block_sparse_moe.gate.top_k_weights": (
        "Weights of top_k_index, same order: sigmoid(logits) of the chosen (no bias) / (their sum + 1e-20), x "
        "routed_scaling_factor (1.0). fp32."
    ),
    "model.last_hidden_state": (
        "Final RMSNorm of the output AttnRes read (output_attn_res_*) over [sealed blocks, last prefix_sum]."
    ),
    "logits": "Last position only: lm_head (not tied to embed_tokens), fp32.",
}

FLA_VERSION = "0.6.0"  # the fla-core the torch KDA references and the remote code's kernels are taken from
GROUP = 32  # quantization_config: mxfp4-pack-quantized, group_size 32
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # magnitudes by the low 3 bits; bit 3 is the sign
TOKEN_CHUNK = 4096  # tokens per _apply_attn_res call and per moe_infer weighted sum (header: memory); 16 in --selftest
HOST_EXPERTS_BELOW = 120 * 2**30  # HF_GOLDEN_EXPERTS=auto: experts in host RAM on a GPU with less memory than this
# --selftest's tiny model: routing top-k, and the seed of its biases / norms / experts. Seed 2 left one bf16 GPU
# near-tie (layer 3, token 33: a margin of 1e-4 against 1e-3 of gate noise) that flipped a pick; see compare().
TINY_TOP_K = 4
TINY_SEED = 3
EXPERT = re.compile(r"block_sparse_moe\.experts\.(\d+)\.(w[123]\.weight)")


def _alias_output_recorder() -> None:
    """transformers 5 moved OutputRecorder to utils.output_capturing; the remote modeling imports it from
    utils.generic. Before any remote import."""
    from transformers.utils import generic

    if not hasattr(generic, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder

        generic.OutputRecorder = OutputRecorder


def _shim_remote_classes() -> None:
    """Wrap transformers' loader of remote classes for the tie_weights / _tied_weights_keys / sdpa shims (header);
    every remote import, the hub's and --selftest's tiny checkpoint's, goes through it."""
    from transformers import dynamic_module_utils as dmu

    load = dmu.get_class_in_module

    def get_class_in_module(*args, **kwargs):
        cls = load(*args, **kwargs)
        module = sys.modules.get(cls.__module__)
        wrapper = getattr(module, "KimiK3ForConditionalGeneration", None)
        if wrapper is not None:
            wrapper.tie_weights = lambda self, *a, **kw: self.language_model.tie_weights(*a, **kw)
            module.KimiK3PreTrainedModel._supports_sdpa = True
        linear = sys.modules.get(module.__name__.rsplit(".", 1)[0] + ".modeling_kimi_linear")
        lm = getattr(linear, "KimiLinearForCausalLM", None)
        if lm is not None and isinstance(lm._tied_weights_keys, list):
            lm._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
            sys.modules[lm.__module__].KimiPreTrainedModel._supports_sdpa = True
        return cls

    dmu.get_class_in_module = get_class_in_module


_alias_output_recorder()
_shim_remote_classes()


def dequantize_mxfp4(packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """compressed-tensors mxfp4-pack-quantized -> ``dtype``: E2M1 pairs [rows, cols / 2] uint8 (low nibble first,
    unpacked by transformers' own table) x 2^(scale - 127) per group of 32 inputs. Every product is exact in bf16."""
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    values = Fp8Dequantize(None)._unpack_fp4(packed)
    group = values.shape[-1] // scale.shape[-1]
    return (values * (scale.to(torch.float32) - 127).exp2().repeat_interleave(group, dim=-1)).to(dtype)


def check_dequant(packed: torch.Tensor, scale: torch.Tensor) -> str:
    """dequantize_mxfp4 against compressed_tensors' MXFP4PackedCompressor.decompress, on one real tensor."""
    try:
        import compressed_tensors
        from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
        from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    except ImportError:
        return "not run (compressed_tensors not installed)"
    args = QuantizationArgs(num_bits=4, type="float", strategy="group", group_size=GROUP, symmetric=True)
    scheme = QuantizationScheme(targets=["Linear"], weights=args)
    state = {"weight_packed": packed.cpu(), "weight_scale": scale.cpu()}
    ref = MXFP4PackedCompressor.decompress(state, scheme)["weight"]
    if not torch.equal(dequantize_mxfp4(packed.cpu(), scale.cpu(), ref.dtype), ref):
        raise SystemExit("MXFP4 dequantization differs from compressed_tensors: update dequantize_mxfp4")
    return f"bit-identical to compressed_tensors {compressed_tensors.__version__}"


def pack_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A weight on the MXFP4 grid (E2M1 value x a power of two per group of 32) -> (weight_packed, weight_scale),
    the inverse of dequantize_mxfp4. For --selftest's tiny checkpoint."""
    groups = w.float().unflatten(-1, (-1, GROUP))
    exponent = groups.abs().amax(-1).clamp(min=2.0**-126).log2().floor() - 2  # max |value| in [4, 8): E2M1 4 or 6
    values = groups / exponent.exp2()[..., None]
    table = torch.tensor(E2M1)
    index = (values.abs()[..., None] == table).float().argmax(-1)
    assert torch.equal(table[index] * values.sign().abs(), values.abs()), "weight is not on the MXFP4 grid"
    nibbles = (index + 8 * (values < 0)).to(torch.uint8).flatten(-2)
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    return packed, (exponent + 127).to(torch.uint8)


def torch_chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    A_log=None,
    dt_bias=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    safe_gate=False,
    lower_bound=None,
    transpose_state_layout=False,
    state_v_first=False,
    cu_seqlens=None,
    **kwargs,
):
    """fla's chunk_kda contract (prefill, one sequence) computed by fla's own torch references, composed as fla's
    tests compose them to grade chunk_kda (tests/ops/test_kda.py): F.normalize q / k, sigmoid beta, the gate by
    naive_kda_lowerbound_gate when lower_bound is set (chunk_kda's rule; safe_gate only picks its numerically safer
    kernel) else naive_kda_gate, then naive_recurrent_kda. All in fp32; o in v's dtype, the state fp32."""
    from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate
    from fla.ops.kda.naive import naive_recurrent_kda

    if cu_seqlens is not None or kwargs:
        raise SystemExit(f"torch_chunk_kda: unsupported arguments (cu_seqlens / {sorted(kwargs)}): see the header")
    if use_qk_l2norm_in_kernel:
        q, k = F.normalize(q.float(), p=2, dim=-1), F.normalize(k.float(), p=2, dim=-1)
    if use_beta_sigmoid_in_kernel:
        beta = beta.float().sigmoid()
    if use_gate_in_kernel:
        g = (
            naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound)
            if lower_bound is not None
            else naive_kda_gate(g, A_log, dt_bias)
        )
    v_first = state_v_first or transpose_state_layout
    if initial_state is not None and v_first:
        initial_state = initial_state.transpose(-1, -2)
    o, state = naive_recurrent_kda(
        q, k, v, g, beta, scale=scale, initial_state=initial_state, output_final_state=output_final_state
    )
    if state is not None and v_first:
        state = state.transpose(-1, -2).contiguous()
    return o, state


def check_torch_kda(device: torch.device) -> str:
    """torch_chunk_kda against fla's Triton chunk_kda with the remote layer's arguments, on random inputs like its
    (bf16 q / k / v after conv + SiLU, bf16 g, fp32 beta, A_log / dt_bias in bf16 as the Loader leaves them), at the
    tolerance fla's tests grade chunk_kda against its naive reference with (RMSE ratio 0.005)."""
    import fla
    from fla.ops.kda import chunk_kda

    gen = torch.Generator().manual_seed(0)
    batch, length, heads, dim = 1, 512, 4, 128

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen).to(device, dtype)

    q, k, v = (F.silu(randn(batch, length, heads, dim)) for _ in range(3))
    g, beta = randn(batch, length, heads, dim), randn(batch, length, heads, dtype=torch.float32)
    a_log = torch.empty(heads).uniform_(1, 16, generator=gen).log().to(device, torch.bfloat16)
    kw = dict(
        A_log=a_log,
        dt_bias=randn(heads * dim),
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        transpose_state_layout=True,
    )
    with torch.no_grad():
        ref_o, ref_state = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, **kw)
        o, state = torch_chunk_kda(q=q, k=k, v=v, g=g, beta=beta, **kw)

    def ratio(a, b):
        a, b = a.float(), b.float()
        return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()

    errs = {"o": ratio(o, ref_o), "state": ratio(state, ref_state)}
    if max(errs.values()) > 0.005:
        raise SystemExit(f"torch KDA differs from fla {fla.__version__} chunk_kda beyond fla's tolerance: {errs}")
    return f"matches fla {fla.__version__} chunk_kda (RMSE ratio o {errs['o']:.2e}, state {errs['state']:.2e})"


def _patch_remote(modeling, attn: str, kda: str) -> None:
    """The shims of the header, on one import of the remote modeling module (the hub's, or --selftest's tiny
    checkpoint's): the causal mask, and with kda == "torch" the KDA function the layer calls by name."""

    def create_causal_mask(config, input_embeds, attention_mask=None, cache_position=None, **_):
        if attention_mask is not None and not torch.all(attention_mask == 1):
            raise SystemExit("create_causal_mask shim: padded batches are not supported")
        return causal_mask(input_embeds.shape[1], input_embeds.dtype, input_embeds.device, attn)

    modeling.create_causal_mask = create_causal_mask
    if kda == "torch":
        modeling.chunk_kda = torch_chunk_kda


def _chunk_attn_res(modeling, chunk: int) -> None:
    """Make the remote module's _apply_attn_res (which the layers and the output AttnRes call by name) run ``chunk``
    tokens at a time; 0 calls it whole. Every op in it is per token, so chunking changes only the batch of each
    kernel. The --selftest reference sets 0: it is upstream's whole-model forward."""
    if not hasattr(modeling, "_apply_attn_res_whole"):
        whole = modeling._apply_attn_res_whole = modeling._apply_attn_res

        def _apply_attn_res(prefix_sum, block_residual, proj, norm):
            n, step = prefix_sum.shape[0], modeling.attn_res_chunk
            if not step or n <= step:
                return whole(prefix_sum, block_residual, proj, norm)
            parts = [
                whole(prefix_sum[s : s + step], block_residual[s : s + step], proj, norm) for s in range(0, n, step)
            ]
            return torch.cat(parts)

        modeling._apply_attn_res = _apply_attn_res
    modeling.attn_res_chunk = chunk


def lean_moe_infer(moe, x, topk_ids, topk_weight, chunk: int, host=None, slot=None):
    """KimiSparseMoeBlock.moe_infer in less memory, same arithmetic: each expert gets the same rows (argsort of the
    flat top-k ids) and writes its output straight into its rows of new_x, where upstream first gathers all N x top_k
    rows and then concatenates the outputs (two more copies); the fp32 weighted sum over top_k then runs ``chunk``
    tokens at a time, where upstream casts all of new_x to fp32 at once. With ``host`` (HF_GOLDEN_EXPERTS=host) expert
    e's weights host[e] are copied to the device for its one call, through ``slot``, a meta expert module."""
    top_k = topk_ids.shape[1]
    idxs = topk_ids.view(-1).argsort()
    counts = torch.bincount(topk_ids.view(-1), minlength=moe.num_experts).tolist()
    new_x = x.new_empty(idxs.numel(), x.shape[-1])
    start = 0
    for e, n in enumerate(counts):
        if n == 0:
            continue
        rows = idxs[start : start + n]
        tokens = x[rows // top_k]
        if host is None:
            new_x[rows] = moe.experts[e](tokens)
        else:
            weights = {name: w.to(x.device) for name, w in host[e].items()}
            new_x[rows] = torch.func.functional_call(slot, weights, (tokens,))
        start += n
    new_x = new_x.view(*topk_ids.shape, -1)
    out = x.new_empty(topk_ids.shape[0], x.shape[-1])
    for s in range(0, out.shape[0], chunk):
        w = topk_weight[s : s + chunk].unsqueeze(dim=-1)
        out[s : s + chunk] = new_x[s : s + chunk].type(topk_weight.dtype).mul_(w).sum(dim=1).type(new_x.dtype)
    return out


def _restore_attn(model: torch.nn.Module, attn: str) -> None:
    """KimiLinearModel.__init__ overwrote _attn_implementation with flash_attention_2: put the requested one back on
    every config the modules read it from."""
    for config in {id(m.config): m.config for m in model.modules() if hasattr(m, "config")}.values():
        config._attn_implementation = attn


def causal_mask(n: int, dtype: torch.dtype, device: torch.device, attn: str) -> torch.Tensor | None:
    """Prefill from position 0, no padding: sdpa takes is_causal (None); eager adds a [1, 1, N, N] mask."""
    if attn == "sdpa":
        return None
    keep = torch.ones(n, n, dtype=torch.bool, device=device).tril()
    return torch.zeros(n, n, dtype=dtype, device=device).masked_fill(~keep, torch.finfo(dtype).min)[None, None]


class OutputAttnResNorm(torch.nn.Module):
    """The last stage before lm_head: KimiLinearModel's output AttnRes read over the carried blocks, then the final
    norm. Its weights are the backbone's norm and output_attn_res_{norm,proj}, read under one tuple prefix."""

    def __init__(self, backbone):
        super().__init__()
        self.norm = backbone.norm
        self.output_attn_res_norm = backbone.output_attn_res_norm
        self.output_attn_res_proj = backbone.output_attn_res_proj
        self.apply_output_attn_res = backbone._apply_output_attn_res  # a bound method, not a submodule
        self.block_residual = None  # set by KimiK3Golden.after_layer

    def forward(self, hidden_states):
        return self.norm(self.apply_output_attn_res(hidden_states, self.block_residual))


class KimiK3Golden(LayerwiseGolden):
    hf_repo = "moonshotai/Kimi-K3"
    trust_remote_code = True
    remote_tokenizer = True
    checkpoint_prefix = "language_model.model."  # the VLM stores the text backbone under language_model.model
    lm_head_prefix = "language_model.lm_head."
    prefix = "[BOS]"  # the tokenizer adds none; Kimi-K2.7-Code's measured best, not yet measured on K3
    recipe = Recipe(
        modules=("self_attn.kv_a_proj_with_mqa", "self_attn.kv_a_layernorm"),  # MLA only; + the gate (register_hooks)
        sublayers=(
            "self_attn",
            "mlp",
            "block_sparse_moe",
            "block_sparse_moe.routed_expert_up_proj",
            "block_sparse_moe.shared_experts",
        ),  # + the AttnRes reads (register_hooks)
        cache=(),  # KimiDynamicCache is not a transformers Cache: see cache_tensors
        # Routing is a top-k: rounding the logits can flip an expert at the cut, so they are never stored rounded; the
        # KDA state is what the next token starts from.
        keep_fp32=("block_sparse_moe.gate.output", "block_sparse_moe.gate.top_k_weights", "recurrent_state"),
        config=(
            "linear_attn_config",
            "attn_res_block_size",
            "first_k_dense_replace",
            "moe_layer_freq",
            "kv_lora_rank",
            "q_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "mla_use_nope",
            "mla_use_output_gate",
            "num_attention_heads",
            "num_key_value_heads",
            "rms_norm_eps",
            "hidden_act",
            "activation_situ_beta",
            "activation_situ_linear_beta",
            "intermediate_size",
            "moe_intermediate_size",
            "routed_expert_hidden_size",
            "latent_moe_use_norm",
            "num_experts",
            "num_experts_per_token",
            "num_shared_experts",
            "moe_router_activation_func",
            "moe_renormalize",
            "num_expert_group",
            "topk_group",
            "routed_scaling_factor",
            "tie_word_embeddings",
            "quantization_config",
        ),
        notes=NOTES,
    )

    def __init__(self):
        self.kda = os.environ.get("HF_GOLDEN_KDA", "torch")
        if self.kda not in ("torch", "triton"):
            raise SystemExit(f"HF_GOLDEN_KDA={self.kda!r}: 'torch' (fla's references, default) or 'triton'")
        self.experts = os.environ.get("HF_GOLDEN_EXPERTS", "auto")
        if self.experts not in ("auto", "device", "host"):
            raise SystemExit(f"HF_GOLDEN_EXPERTS={self.experts!r}: 'auto' (default), 'device' or 'host'")
        self.kda_check = self.dequant_check = None
        self.a_log_padding = set()

    def generate(self, args):
        if args.prefill_chunk:
            raise SystemExit("--prefill-chunk: the AttnRes blocks are carried between layers (see the header)")
        self.attn = args.attn
        device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
        if self.kda == "torch" and self.kda_check is None:
            self.kda_check = check_torch_kda(device) if device.type == "cuda" else "not run (needs CUDA for fla)"
            print(f"[kda] torch (fla's references): {self.kda_check}", flush=True)
        self.token_chunk = 16 if args.selftest else TOKEN_CHUNK  # the selftest's 64 tokens: several chunks
        small = device.type == "cuda" and torch.cuda.get_device_properties(device).total_memory < HOST_EXPERTS_BELOW
        self.experts_on_host = self.experts == "host" or (self.experts == "auto" and (args.selftest or small))
        print(f"[experts] routed experts on {'host' if self.experts_on_host else device} ({self.experts})", flush=True)
        if args.selftest:  # the tiny checkpoint, its experts packed and A_log padded as the real one's
            args.hf_repo = str(self.pack_checkpoint(Path(args.hf_repo)))
        out = super().generate(args)
        meta = json.loads((out / "metadata.json").read_text())
        import fla

        meta["kda"] = {
            "implementation": (
                "fla torch references (naive_kda_lowerbound_gate + naive_recurrent_kda), fp32"
                if self.kda == "torch"
                else "fla Triton chunk_kda (the remote code's)"
            ),
            "fla_version": fla.__version__,
            "check": self.kda_check,
        }
        meta["memory"] = {
            "token_chunk": self.token_chunk,
            "routed_experts": "host RAM, copied per call" if self.experts_on_host else "device",
        }
        meta["dequant_check"] = self.dequant_check
        meta["a_log_padding"] = sorted(self.a_log_padding)
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        return out

    def build(self, config, dtype):
        model = super().build(config, dtype)
        self.modeling = sys.modules[type(model.get_decoder()).__module__]
        _restore_attn(model, self.attn)
        _patch_remote(self.modeling, self.attn, self.kda)
        _chunk_attn_res(self.modeling, self.token_chunk)
        return model

    def layer_types(self, text_config) -> list[str]:
        """Attention kind (KDA / MLA, from the 1-based kda_layers) and MLP kind (dense before first_k_dense_replace)."""
        k = text_config.first_k_dense_replace
        return [
            f"{'linear_attention' if text_config.is_kda_layer(i) else 'full_attention'}/{'dense' if i < k else 'moe'}"
            for i in range(text_config.num_hidden_layers)
        ]

    def make_cache(self, text_config):
        return self.modeling.KimiDynamicCache(config=text_config)

    def prepare(self, backbone, text_config, hidden, position_ids) -> RunState:
        """What KimiLinearModel.forward computes before its layer loop: a KimiDynamicCache, the MLA causal mask (KDA
        layers get none: no padding), and an empty block_residual [N, 0, hidden]. No RoPE."""
        kinds = set(self.layer_types(text_config))
        mask = causal_mask(hidden.shape[1], hidden.dtype, hidden.device, self.attn)
        return RunState(
            position_ids=position_ids,
            cache=self.make_cache(text_config),
            masks={kind: None if kind.startswith("linear_attention") else mask for kind in kinds},
            rope={},
            carry={
                "cache_position": position_ids[0],
                "block_residual": hidden.new_zeros(hidden.shape[0] * hidden.shape[1], 0, hidden.shape[2]),
            },
        )

    def layer_kwargs(self, i, kind, state):
        """Exactly what KimiLinearModel.forward passes each decoder layer."""
        return dict(attention_mask=state.masks[kind], past_key_values=state.cache, **state.carry)

    def after_layer(self, i, output, state):
        prefix_sum, state.carry["block_residual"] = output
        self.final.block_residual = state.carry["block_residual"]
        return prefix_sum

    def stages(self, model, backbone, text_config):
        stages = super().stages(model, backbone, text_config)
        self.layers = backbone.layers
        self.final = OutputAttnResNorm(backbone)
        stages[-2] = (
            "norm",
            self.final,
            (f"{self.checkpoint_prefix}norm.", f"{self.checkpoint_prefix}output_attn_res_"),
        )
        return stages

    def convert(self, sd, prefix, module):
        """MXFP4 routed experts -> each expert's w1 / w2 / w3 weight in the compute dtype, packed bytes dropped as they
        are read; A_log's zero padding dropped; bf16 / fp32 tensors otherwise unchanged. MoE layers get lean_moe_infer;
        with experts on host, each expert weight moves to host RAM as it is dequantized and the layer's (meta) experts
        are taken out of the module, so the Loader neither expects nor moves them."""
        if not isinstance(module, self.modeling.KimiDecoderLayer):
            return sd
        moe = getattr(module, "block_sparse_moe", None)
        host = [{} for _ in moe.experts] if moe is not None and self.experts_on_host else None
        for packed in sorted(k for k in sd if k.endswith(".weight_packed")):
            weight = packed.removesuffix("_packed")
            if self.dequant_check is None:
                self.dequant_check = check_dequant(sd[packed], sd[f"{weight}_scale"])
                print(f"[dequant] {self.dequant_check}", flush=True)
            dtype = module.get_parameter(weight).dtype
            sd[weight] = dequantize_mxfp4(sd.pop(packed), sd.pop(f"{weight}_scale"), dtype)
            if host is not None and (m := EXPERT.fullmatch(weight)):
                host[int(m[1])][m[2]] = sd.pop(weight).cpu()
        if host is not None:
            for k in [k for k in sd if EXPERT.fullmatch(k)]:  # any expert weight not stored packed
                m = EXPERT.fullmatch(k)
                host[int(m[1])][m[2]] = sd.pop(k).to("cpu", module.get_parameter(k).dtype)
            if missing := [e for e, h in enumerate(host) if len(h) != 3]:
                raise SystemExit(f"{prefix}: routed experts {missing[:8]} lack w1 / w2 / w3 weights")
        if moe is not None:
            slot = moe.experts[0]
            if host is not None:
                moe.experts = torch.nn.ModuleList()
            moe.moe_infer = functools.partial(lean_moe_infer, moe, chunk=self.token_chunk, host=host, slot=slot)
        if "self_attn.A_log" in sd:
            heads = module.self_attn.A_log.numel()
            a_log = sd["self_attn.A_log"]
            if a_log.numel() > heads:
                if a_log[heads:].any():
                    raise SystemExit(f"{prefix}self_attn.A_log: padding past {heads} heads is not zero")
                self.a_log_padding.add(f"{a_log.numel()} -> {heads}")
                sd["self_attn.A_log"] = a_log[:heads].clone()
        return sd

    def register_hooks(self, i, layer, captured, sublayers):
        handles = super().register_hooks(i, layer, captured, sublayers)
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is not None:

            def routing(gate, args, out, key=f"model.layers.{i}.block_sparse_moe.gate"):
                # KimiMoEGate returns (topk_idx, topk_weight) only; its logits are this same fp32 GEMM.
                hidden = args[0].reshape(-1, args[0].shape[-1])
                captured[f"{key}.output"] = F.linear(hidden.float(), gate.weight.float())
                index, order = out[0].sort(dim=-1)  # torch.topk(sorted=False): order by expert id
                captured[f"{key}.top_k_index"] = index.to(torch.int32)
                captured[f"{key}.top_k_weights"] = out[1].gather(-1, order)

            handles.append(moe.gate.register_forward_hook(routing))
        if sublayers:
            for norm in ("input_layernorm", "post_attention_layernorm"):

                def read(_m, args, key=f"model.layers.{i}.{norm}.input"):
                    captured[key] = args[0]

                handles.append(getattr(layer, norm).register_forward_pre_hook(read))
        return handles

    def is_sublayer(self, key):
        return super().is_sublayer(key) or key.endswith((".input_layernorm.input", ".post_attention_layernorm.input"))

    def cache_tensors(self, i, cache):
        """KDA layers: the final recurrent state and the three short-conv caches. MLA layers: nothing here (HF holds
        K / V expanded per head; the latent is stored from the hooks)."""
        if cache is None or cache.recurrent_states[i] is None:
            return {}
        out = {f"past_key_values.layers.{i}.recurrent_state": cache.recurrent_states[i]}
        for name, t in zip("qkv", cache.conv_states[i], strict=True):
            out[f"past_key_values.layers.{i}.conv_state_{name}"] = t
        return out

    def free_cache(self, i, cache):
        if cache is not None:
            cache.recurrent_states[i] = cache.conv_states[i] = cache.key_cache[i] = cache.value_cache[i] = None
        # The engine's per-layer cleanup: the layer's lean_moe_infer holds its host experts (convert), ~59 GB.
        moe = getattr(self.layers[i], "block_sparse_moe", None)
        if moe is not None:
            moe.__dict__.pop("moe_infer", None)

    # ---- --selftest ----------------------------------------------------------------------------------------------

    def reference(self, ckpt, ids, device, dtype, attn):
        """One forward of the remote text model from from_pretrained, captured under the same names. The VLM
        wrapper's forward takes no logits_to_keep and builds the vision path, so the backbone and lm_head are called
        as KimiLinearForCausalLM.forward calls them."""
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(str(ckpt), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(str(ckpt), config=config, dtype=dtype, trust_remote_code=True)
        model = model.to(device).eval()
        backbone = model.get_decoder()
        _restore_attn(model, attn)
        modeling = sys.modules[type(backbone).__module__]
        _patch_remote(modeling, attn, self.kda)
        _chunk_attn_res(modeling, 0)  # upstream's whole _apply_attn_res (and its moe_infer: convert never ran here)
        captured: dict[str, torch.Tensor] = {}

        def put(key):
            return lambda _m, _a, o: captured.__setitem__(key, o[0] if isinstance(o, tuple) else o)

        backbone.get_input_embeddings().register_forward_hook(put("model.embed_tokens.output"))
        backbone.norm.register_forward_hook(put("model.last_hidden_state"))
        for i, layer in enumerate(backbone.layers):
            layer.register_forward_hook(put(f"model.layers.{i}.output"))
            self.register_hooks(i, layer, captured, sublayers=True)
        # This import's own class: KimiLinearModel.forward asserts isinstance against it.
        cache = sys.modules[type(backbone).__module__].KimiDynamicCache(config=config.get_text_config())
        with torch.no_grad():
            out = backbone(input_ids=ids[None].long().to(device), past_key_values=cache, use_cache=True)
            captured["logits"] = model.get_output_embeddings()(out.last_hidden_state[:, -1:])
        for i in range(len(backbone.layers)):
            captured.update(self.cache_tensors(i, cache))
        return {k: v.cpu() for k, v in captured.items()}

    def pack_checkpoint(self, ckpt: Path) -> Path:
        """A copy of the tiny checkpoint in the real one's layout: routed experts as weight_packed / weight_scale
        (tiny_init put them on the MXFP4 grid, so pack_mxfp4 recovers them exactly) and A_log zero-padded to twice
        its heads, so convert() runs every path it runs on the real checkpoint."""
        from safetensors.torch import load_file, save_file

        out = ckpt.parent / "ckpt_mxfp4"
        shutil.copytree(ckpt, out)
        index = json.loads((out / "model.safetensors.index.json").read_text())
        for f in sorted(set(index["weight_map"].values())):
            sd = load_file(str(out / f))
            for k in [k for k in sd if re.search(r"\.block_sparse_moe\.experts\.\d+\.w[123]\.weight$", k)]:
                packed, scale = pack_mxfp4(sd.pop(k))
                sd[f"{k}_packed"], sd[f"{k}_scale"] = packed, scale
                index["weight_map"][f"{k}_packed"] = index["weight_map"][f"{k}_scale"] = f
                del index["weight_map"][k]
            for k in [k for k in sd if k.endswith(".self_attn.A_log")]:
                sd[k] = torch.cat([sd[k], torch.zeros_like(sd[k])])
            save_file(sd, str(out / f), metadata={"format": "pt"})
        (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        return out

    def tiny_init(self, model):
        super().tiny_init(model)
        g = torch.Generator().manual_seed(TINY_SEED)
        self.tiny_bias = {}  # layer -> its router bias, for the routing margin compare() prints
        for name, t in model.named_parameters():
            if name.endswith("e_score_correction_bias"):
                t.copy_(torch.randn(t.shape, generator=g) * 0.1)
                self.tiny_bias[int(re.search(r"layers\.(\d+)\.", name)[1])] = t.detach().float().clone()
            elif name.endswith("norm.weight"):  # O(1) norms, like real ones
                t.copy_(1 + torch.randn(t.shape, generator=g) * 0.1)
            elif name.endswith("_res_proj.weight"):  # AttnRes scores large enough that the softmax is not uniform
                t.copy_(torch.randn(t.shape, generator=g))
            elif name.endswith("A_log"):  # decay rates of the real model's range (exp(A_log) in 1..16)
                t.copy_(torch.empty(t.shape).uniform_(1, 16, generator=g).log())
            elif re.search(r"experts\.\d+\.w[123]\.weight$", name):
                # MXFP4 grid: E2M1 values with a 4 or 6 in every group of 32 (so pack_mxfp4 finds the exponent), x a
                # power of two per group: exact in bf16.
                values = torch.tensor(E2M1)[
                    torch.randint(0, 8, (*t.shape[:-1], t.shape[-1] // GROUP, GROUP), generator=g)
                ]
                values[..., 0] = 4.0
                signs = torch.randint(0, 2, values.shape, generator=g) * 2 - 1
                scale = 2.0 ** torch.randint(-9, -5, (*values.shape[:-1], 1), generator=g)
                t.copy_((values * signs * scale).flatten(-2))

    def compare(self, name, got, ref, exact):
        """Also prints each MoE layer's routing margin: the smallest gap, over tokens, between the top_k-th and the next
        selection score (sigmoid(logits) + e_score_correction_bias) of the reference, against the gate noise between
        the two runs. A gap under the noise is a near-tie whose pick can flip on GPU, cascading through every later
        layer; TINY_SEED is chosen so that none occurs (the tolerance is unchanged)."""
        m = re.fullmatch(r"model\.layers\.(\d+)\.block_sparse_moe\.gate\.output", name)
        if m and int(m[1]) in getattr(self, "tiny_bias", {}):
            k = TINY_TOP_K
            scores = ref.float().sigmoid() + self.tiny_bias[int(m[1])].to(ref.device)
            top = scores.sort(dim=-1, descending=True).values
            gap = (top[:, k - 1] - top[:, k]).min().item()
            noise = (got.float().sigmoid() - ref.float().sigmoid()).abs().max().item()
            tie = "  NEAR-TIE: a pick can flip" if gap < 2 * noise else ""
            print(f"[routing] layer {m[1]}: smallest top-{k} margin {gap:.2e}, gate noise {noise:.2e}{tie}", flush=True)
        return super().compare(name, got, ref, exact)

    def tiny_config(self):
        """The real config cut down: every layer kind (KDA dense, KDA MoE, MLA MoE), AttnRes seals every 2 layers, no
        quantization_config (the tiny checkpoint is saved unquantized; pack_checkpoint packs a copy)."""
        from common import hub_kwargs, resolve_revision
        from transformers import AutoConfig

        hub = hub_kwargs(self.hf_repo, resolve_revision(self.hf_repo, "main"))
        config = AutoConfig.from_pretrained(self.hf_repo, trust_remote_code=True, **hub)
        text = config.text_config
        for c in (config, text):
            c.__dict__.pop("quantization_config", None)
        text.update(
            dict(
                vocab_size=512,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=6,
                num_attention_heads=4,
                num_key_value_heads=4,
                q_lora_rank=32,
                kv_lora_rank=32,
                qk_nope_head_dim=16,
                qk_rope_head_dim=8,
                v_head_dim=16,
                moe_intermediate_size=32,
                routed_expert_hidden_size=32,
                num_experts=8,
                num_experts_per_token=TINY_TOP_K,
                num_shared_experts=2,
                routed_scaling_factor=2.5,  # 1.0 in the checkpoint: != 1 so applying it is tested
                first_k_dense_replace=1,
                attn_res_block_size=2,
                linear_attn_config=dict(
                    text.linear_attn_config,
                    num_heads=2,
                    head_dim=16,
                    kda_layers=[1, 2, 3, 5],  # 1-based: MLA on 0-based 3 and 5
                    full_attn_layers=[4, 6],
                ),
                pad_token_id=None,
                bos_token_id=None,
                eos_token_id=None,
            )
        )
        config.vision_config.update(
            dict(
                vt_hidden_size=32,
                vt_intermediate_size=64,
                vt_num_attention_heads=2,
                vt_num_hidden_layers=1,
                qkv_hidden_size=32,
                mm_hidden_size=32,
                text_hidden_size=64,
                init_pos_emb_height=8,
                init_pos_emb_width=8,
            )
        )
        config.pad_token_id = None
        return config


if __name__ == "__main__":
    sys.exit(KimiK3Golden().main())
