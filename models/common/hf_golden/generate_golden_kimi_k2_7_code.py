#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for moonshotai/Kimi-K2.7-Code, layer by layer (engine.py), from transformers' kimi_k25 code (the
text model is DeepseekV3).

Stored tensors (N = isl, i = layer; 61 layers, MLA attention on all of them; dense MLP on layer 0, MoE (384 routed
experts, top 8, 1 shared) on 1-60):

  hidden/embed.safetensors
    model.embed_tokens.output                              [1, N, 7168]     bf16  embeddings (no scale)
  hidden/layer_XX.safetensors
    model.layers.{i}.output                                [1, N, 7168]     bf16  all   residual after attention
                                                                                  and MLP
    model.layers.{i}.self_attn.output                      [1, N, 7168]     bf16  all   --sublayers: MLA after o_proj
    model.layers.{i}.mlp.output                            [1, N, 7168]     bf16  all   --sublayers: dense MLP (0) /
                                                                                  routed + shared experts (1-60)
  cache/layer_XX.safetensors
    model.layers.{i}.self_attn.kv_a_proj_with_mqa.output   [1, N, 576]      bf16  all   MLA latent [c_kv 512 |
                                                                                  k_pe 64], raw: before
                                                                                  kv_a_layernorm and RoPE
    past_key_values.layers.{i}.keys                        [1, 1, N, 512]   bf16  all   MLA latent cache:
                                                                                  kv_a_layernorm(c_kv)
    past_key_values.layers.{i}.values                      [1, 1, N, 64]    bf16  all   k_pe after YaRN RoPE
                                                                                  (interleaved pairs in, half-split
                                                                                  out), one head shared by all 64
    model.layers.{i}.mlp.gate.output                       [N, 384]         fp32  1-60  router logits, before
                                                                                  sigmoid / correction bias
    model.layers.{i}.mlp.experts.top_k_index               [N, 8]           int32 1-60  chosen experts, ascending
    model.layers.{i}.mlp.experts.top_k_weights             [N, 8]           fp32  1-60  their weights, same order:
                                                                                  normalized sigmoid, x 2.827
  hidden/final.safetensors
    model.last_hidden_state                                [1, N, 7168]     bf16  after the final norm
    logits                                                 [1, 1, 163840]   fp32  last position, untied lm_head

REQUIRES transformers >= 5.17 (kimi_k25 is in-library; validated with 5.19.0) and, for the remote-code tiktoken
tokenizer, blobfile (tiktoken reads even local files through it). Run through a uv overlay on top of python_env:

    uv run --no-project --python python_env/bin/python --with transformers==5.19.0 --with blobfile \\
        python models/common/hf_golden/generate_golden_kimi_k2_7_code.py --isl 10240 --prefill-chunk 5120

What is model-specific, and why it is handled here:

  * The checkpoint is the Kimi-K2.5-style VLM (Kimi_K25ForConditionalGeneration: MoonViT tower + projector +
    DeepseekV3 text model; config text model_type "kimi_k2" is mapped to deepseek_v3). AutoModelForCausalLM has no
    kimi_k25 entry, so the VLM class is registered for it, as the GLM / Qwen scripts do. The text backbone is
    stored under language_model.model.* and the head under language_model.lm_head.*; the vision tower and projector
    are built on meta and never read. tie_word_embeddings is False in the checkpoint (Kimi_K25Config defaults True).
  * DeepseekV3Model.forward before its layer loop is the engine's default prepare(): one causal mask, YaRN RoPE
    (factor 64 over 4096, theta 5e4, mscale = mscale_all_dim = 1, so attention_factor 1) and a DynamicCache.
  * MLA: q = q_b_proj(q_a_layernorm(q_a_proj(x))), 64 heads of [nope 128 | rope 64]; kv_a_proj_with_mqa gives
    [c_kv 512 | k_pe 64]; rope_interleave (the config default): pairs (x0, x1), (x2, x3), ... rotate together and
    are written half-split, for q and k alike. Scaling 192**-0.5 * (0.1 * ln 64 + 1)**2. In this transformers the
    cache holds the latent itself (keys = kv_a_layernorm(c_kv), values = roped k_pe) and expand_kv applies
    kv_b_proj every step, so the cache is stored as is: ~66 MB per layer at N = 56320, not HF-expanded K / V.
  * MoE gate: fp32 logits, sigmoid, top-8 of sigmoid + e_score_correction_bias (fp32; n_group = topk_group = 1, so
    the group stage selects everything), weights from the uncorrected sigmoid renormalized and x 2.827. SiLU
    SwiGLU everywhere, no clamp. The routed experts sum in the compute dtype, then the shared expert is added.
  * Routed experts ship INT4 (compressed-tensors pack-quantized: 8 nibbles per int32, lowest first, offset 8,
    symmetric, one bf16 scale per group of 32 inputs); everything else bf16. convert() dequantizes them as
    compressed_tensors does, bf16(q * scale), straight into the fused gate_up_proj [384, 4096, 7168] / down_proj
    [384, 7168, 2048] (gate rows first, as the qwen2_moe mapping fuses them), so the peak is the packed bytes plus
    one bf16 copy (~34 GB). The first MoE layer's expert 0 is compared with compressed_tensors itself, bit for bit,
    when it is importable (metadata.json "dequant_check"). --selftest packs its tiny checkpoint's experts the same
    way, so the same convert() path runs there.
  * Tokenizer: remote code (TikTokenTokenizer), so the engine loads it with trust_remote_code (remote_tokenizer).
    It adds no BOS ([BOS] = 163584).

    python generate_golden_kimi_k2_7_code.py --isl 10240 --prefill-chunk 5120
    python generate_golden_kimi_k2_7_code.py --isl 1024 --num-layers 3 --root /tmp/smoke   # dense + 2 MoE
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import LayerwiseGolden, Recipe  # noqa: E402

NOTES = {
    "model.embed_tokens.output": "Token embeddings (no scale, no BOS added).",
    "model.layers.{i}.output": "Decoder layer output: the residual stream after attention and MLP. The next layer's input.",
    "model.layers.{i}.self_attn.output": "MLA attention output after o_proj, before the residual add (--sublayers).",
    "model.layers.{i}.mlp.output": (
        "Dense SwiGLU MLP (layers < first_k_dense_replace), or routed experts (weighted, summed in the compute dtype) "
        "+ the shared expert; before the residual add (--sublayers)."
    ),
    "model.layers.{i}.self_attn.kv_a_proj_with_mqa.output": (
        "MLA latent [c_kv (kv_lora_rank) | k_pe (qk_rope_head_dim)], the raw projection: before kv_a_layernorm on "
        "c_kv and before RoPE on k_pe."
    ),
    "past_key_values.layers.{i}.keys": (
        "MLA latent cache [1, 1, N, kv_lora_rank] = kv_a_layernorm(c_kv); kv_b_proj expands it to num_attention_heads "
        "of [k_nope qk_nope_head_dim | v v_head_dim] on every step."
    ),
    "past_key_values.layers.{i}.values": (
        "k_pe after RoPE [1, 1, N, qk_rope_head_dim], one rope head shared by all heads (HF's MLA cache keeps it in "
        "the values slot). YaRN RoPE, rope_interleave: input pairs (2m, 2m+1) rotate by frequency m and the output is "
        "laid out half-split [rotated evens | rotated odds]; q's rope part gets the same permutation."
    ),
    "model.layers.{i}.mlp.gate.output": (
        "Router logits [N, n_routed_experts] = hidden (post_attention_layernorm output) @ gate.weight.T, both in fp32; "
        "before the sigmoid. Selection: top-k of sigmoid(logits) + e_score_correction_bias (fp32), n_group = 1. fp32."
    ),
    "model.layers.{i}.mlp.experts.top_k_index": "The num_experts_per_tok chosen experts per token, ascending by expert id. int32.",
    "model.layers.{i}.mlp.experts.top_k_weights": (
        "Weights of top_k_index, same order: sigmoid(logits) of the chosen experts (no bias) divided by their sum + "
        "1e-20 (norm_topk_prob), times routed_scaling_factor. fp32; the shared expert is added unweighted."
    ),
    "model.last_hidden_state": "Final RMSNorm of the last layer's output.",
    "logits": "Last position only: lm_head (not tied to embed_tokens), fp32.",
}

GROUP = 32  # quantization_config: group_size 32, num_bits 4, symmetric
PROJS = ("gate", "up", "down")


def dequantize_int4(packed: torch.Tensor, scale: torch.Tensor, cols: int) -> torch.Tensor:
    """compressed-tensors pack-quantized INT4 -> scale.dtype. ``packed`` int32 [..., rows, cols / 8], nibble j of
    word w is column 8w + j (lowest bits first) as an unsigned value offset by 8; ``scale`` [..., rows, cols / 32].
    The product is formed in the scale's dtype, as compressed_tensors' dequantize does (checked: check_dequant)."""
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    q = (((packed[..., None] >> shifts) & 0xF).flatten(-2)[..., :cols] - 8).to(torch.int8)
    return (q.unflatten(-1, (-1, GROUP)).to(scale.dtype) * scale[..., None]).flatten(-2)


def check_dequant(packed: torch.Tensor, scale: torch.Tensor, shape: list[int]) -> str:
    """dequantize_int4 against compressed_tensors' own unpack + dequantize, on one real tensor."""
    try:
        import compressed_tensors
        from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
        from compressed_tensors.quantization.lifecycle.forward import dequantize
    except ImportError:
        return "not run (compressed_tensors not installed)"
    ref = dequantize(unpack_from_int32(packed, 4, torch.Size(shape)), scale)
    if not torch.equal(dequantize_int4(packed, scale, shape[1]), ref):
        raise SystemExit("INT4 dequantization differs from compressed_tensors: update dequantize_int4")
    return f"bit-identical to compressed_tensors {compressed_tensors.__version__}"


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """int8 values in [-8, 7], [rows, cols] -> the checkpoint's int32 [rows, cols / 8] (the inverse of the unpack)."""
    u = (q.to(torch.int32) + 8).unflatten(-1, (-1, 8))
    return (u << torch.arange(0, 32, 4, dtype=torch.int32)).sum(-1, dtype=torch.int32)


class KimiK27CodeGolden(LayerwiseGolden):
    hf_repo = "moonshotai/Kimi-K2.7-Code"
    remote_tokenizer = True
    checkpoint_prefix = "language_model.model."  # the VLM stores the text backbone under language_model.model
    lm_head_prefix = "language_model.lm_head."
    # [BOS]: the tokenizer adds none, and the model reads text best after it. Wikitext ppl over the same 10100 tokens
    # at isl 10240 (chunked 5120): 2.90 (top-1 73.7%) with it, 2.94 raw, 3.20 framed as its own reply by the chat
    # template ("<|im_user|>user<|im_middle|>Continue this WikiText-2 dump verbatim: ...<|im_end|><|im_assistant|>
    # assistant<|im_middle|><think></think>"), like GLM-5.3 and unlike DeepSeek-V4 / MiniMax-M3 / Qwen3.8.
    prefix = "[BOS]"
    recipe = Recipe(
        modules=("self_attn.kv_a_proj_with_mqa", "mlp.gate"),
        routing="mlp.experts",
        cache=("keys", "values"),  # the MLA latent and the roped k_pe, see NOTES
        # Routing is a top-k: rounding the logits can flip an expert at the cut, so they are never stored rounded.
        keep_fp32=("mlp.gate.output", "mlp.experts.top_k_weights"),
        config=(
            "first_k_dense_replace",
            "kv_lora_rank",
            "q_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "rope_parameters",
            "rope_interleave",
            "max_position_embeddings",
            "rms_norm_eps",
            "hidden_act",
            "intermediate_size",
            "moe_intermediate_size",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "n_group",
            "topk_group",
            "norm_topk_prob",
            "routed_scaling_factor",
            "tie_word_embeddings",
        ),
        notes=NOTES,
    )

    def __init__(self):
        from transformers import AutoModelForCausalLM, Kimi_K25Config, Kimi_K25ForConditionalGeneration

        AutoModelForCausalLM.register(Kimi_K25Config, Kimi_K25ForConditionalGeneration, exist_ok=True)
        self.dequant_check = None

    def generate(self, args):
        if args.selftest:  # the tiny checkpoint, its experts packed as the real one's (see pack_checkpoint)
            args.hf_repo = str(self.pack_checkpoint(Path(args.hf_repo)))
        out = super().generate(args)
        meta = json.loads((out / "metadata.json").read_text())
        meta["dequant_check"] = self.dequant_check
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        return out

    def layer_types(self, text_config) -> list[str]:
        """Attention is the same everywhere; the MLP is dense before first_k_dense_replace, MoE after."""
        n, k = text_config.num_hidden_layers, text_config.first_k_dense_replace
        return ["full_attention/dense" if i < k else "full_attention/moe" for i in range(n)]

    def convert(self, sd, prefix, module):
        """Packed INT4 experts -> the fused gate_up_proj / down_proj in the compute dtype; bf16 tensors unchanged."""
        experts = {int(m[1]) for k in sd if (m := re.fullmatch(r"mlp\.experts\.(\d+)\.\w+_proj\.weight_packed", k))}
        if not experts:
            return sd
        n = len(experts)
        assert experts == set(range(n)), f"expert ids not 0..{n - 1}"
        dtype = module.mlp.experts.gate_up_proj.dtype
        fused = {}
        for p in PROJS:
            rows, cols = (int(x) for x in sd[f"mlp.experts.0.{p}_proj.weight_shape"])
            fused[p] = (rows, cols)
        (inter, hidden), _ = fused["gate"], fused["up"]
        device = sd["mlp.experts.0.gate_proj.weight_packed"].device
        gate_up = torch.empty(n, 2 * inter, hidden, dtype=dtype, device=device)
        down = torch.empty(n, *fused["down"], dtype=dtype, device=device)
        if self.dequant_check is None:
            base = "mlp.experts.0.gate_proj."
            shape = [int(x) for x in sd[base + "weight_shape"]]
            self.dequant_check = check_dequant(sd[base + "weight_packed"], sd[base + "weight_scale"], shape)
            print(f"[dequant] {self.dequant_check}", flush=True)
        for e in range(n):
            for p, dst in (("gate", gate_up[e, :inter]), ("up", gate_up[e, inter:]), ("down", down[e])):
                base = f"mlp.experts.{e}.{p}_proj."
                w = dequantize_int4(sd.pop(base + "weight_packed"), sd.pop(base + "weight_scale"), fused[p][1])
                dst.copy_(w)
                del sd[base + "weight_shape"]
        sd["mlp.experts.gate_up_proj"], sd["mlp.experts.down_proj"] = gate_up, down
        return sd

    # ---- --selftest ----------------------------------------------------------------------------------------------

    def pack_checkpoint(self, ckpt: Path) -> Path:
        """A copy of the tiny checkpoint in the real one's layout. Routed experts as weight_packed (int32),
        weight_scale (bf16, per 32 inputs), weight_shape: tiny_init put every expert weight on that grid with an
        exact |q| = 8 in each group, so scale = max|w| / 8 and q = w / scale recover it exactly. And text layers under
        language_model.model.layers: save_pretrained reverses kimi_k25's unanchored "blocks" -> "layers" rename (meant
        for the vision encoder) on every key, writing language_model.model.blocks; from_pretrained reads either."""
        from safetensors.torch import load_file, save_file

        out = ckpt.parent / "ckpt_int4"
        shutil.copytree(ckpt, out)
        index = json.loads((out / "model.safetensors.index.json").read_text())
        real = {k: k.replace("language_model.model.blocks.", "language_model.model.layers.") for k in index["weight_map"]}
        index["weight_map"] = {real[k]: f for k, f in index["weight_map"].items()}
        for f in sorted(set(index["weight_map"].values())):
            sd = {real.get(k, k): v for k, v in load_file(str(out / f)).items()}
            for k in [k for k in sd if ".mlp.experts." in k and k.endswith("_proj.weight")]:
                w = sd.pop(k).float()
                scale = w.unflatten(-1, (-1, GROUP)).abs().amax(-1) / 8
                q = (w.unflatten(-1, (-1, GROUP)) / scale[..., None]).flatten(-2)
                assert torch.equal(q, q.round()) and q.abs().max() <= 8, f"{k} is not on the INT4 grid"
                base = k.removesuffix("weight")
                sd[base + "weight_packed"] = pack_int4(q.to(torch.int8))
                sd[base + "weight_scale"] = scale.to(torch.bfloat16)
                sd[base + "weight_shape"] = torch.tensor(list(w.shape), dtype=torch.int64)
                for s in ("weight_packed", "weight_scale", "weight_shape"):
                    index["weight_map"][base + s] = f
                del index["weight_map"][k]
            save_file(sd, str(out / f), metadata={"format": "pt"})
        (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        return out

    def tiny_init(self, model):
        super().tiny_init(model)
        g = torch.Generator().manual_seed(2)
        for name, t in [*model.named_parameters(), *model.named_buffers()]:
            if name.endswith("e_score_correction_bias"):  # zero from from_config: randomize so loading it is tested
                t.copy_(torch.randn(t.shape, generator=g) * 0.1)
            elif name.endswith(("norm.weight", "layernorm.weight")):  # O(1) norms, like real ones
                t.copy_(1 + torch.randn(t.shape, generator=g) * 0.1)
            elif name.endswith(("experts.gate_up_proj", "experts.down_proj")):
                # INT4 grid: q in [-8, 7] with -8 in every group, scale a power of two, so q * scale is exact in bf16
                # and pack_checkpoint recovers (q, scale) exactly.
                q = torch.randint(-8, 8, (*t.shape[:-1], t.shape[-1] // GROUP, GROUP), generator=g)
                q[..., 0] = -8
                scale = 2.0 ** torch.randint(-9, -5, (*q.shape[:-1], 1), generator=g)
                t.copy_((q * scale).flatten(-2))

    def tiny_config(self):
        from transformers import Kimi_K25Config

        return Kimi_K25Config(
            text_config=dict(
                model_type="deepseek_v3",
                vocab_size=512,
                hidden_size=64,
                intermediate_size=128,
                moe_intermediate_size=32,
                num_hidden_layers=3,  # dense + 2 MoE
                first_k_dense_replace=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                q_lora_rank=32,
                kv_lora_rank=32,
                qk_nope_head_dim=16,
                qk_rope_head_dim=16,
                v_head_dim=16,
                n_routed_experts=16,
                num_experts_per_tok=4,
                n_shared_experts=1,
                n_group=1,
                topk_group=1,
                norm_topk_prob=True,
                routed_scaling_factor=2.827,
                rope_parameters=dict(
                    rope_type="yarn",
                    rope_theta=50000.0,
                    factor=64.0,
                    original_max_position_embeddings=4096,
                    beta_fast=32.0,
                    beta_slow=1.0,
                    mscale=1.0,
                    mscale_all_dim=1.0,
                ),
                max_position_embeddings=262144,
                tie_word_embeddings=False,
            ),
            vision_config=dict(num_hidden_layers=1, hidden_size=32, num_attention_heads=2, intermediate_size=32),
            projection_hidden_size=32,
            tie_word_embeddings=False,
        )


if __name__ == "__main__":
    sys.exit(KimiK27CodeGolden().main())
