#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16, layer by layer (engine.py), from transformers'
nemotron_h code with the Mamba-2 CUDA kernels.

Each of the 108 blocks is ONE mixer behind a pre-norm and the residual: 48 Mamba-2, 48 latent MoE, 12 attention
(layers_block_type; attention on 7, 14, 23, 32, 39, 48, 57, 64, 73, 82, 89, 98). Stored tensors (N = isl, i = block):

  hidden/embed.safetensors
    model.embed_tokens.output                  [1, N, 8192]       bf16  token embeddings (no scale)
  hidden/layer_XX.safetensors
    model.layers.{i}.output                    [1, N, 8192]       bf16  all        x + mixer(norm(x))
    model.layers.{i}.mixer.output              [1, N, 8192]       bf16  all        --sublayers: the mixer, before the
                                                                                   residual
  cache/layer_XX.safetensors
    past_key_values.layers.{i}.keys            [1, 2, N, 128]     bf16  attention  k_proj, NO positional encoding
    past_key_values.layers.{i}.values          [1, 2, N, 128]     bf16  attention  v_proj
    past_key_values.layers.{i}.conv_states     [1, 18432, 4]      bf16  mamba      the last 4 conv1d inputs (x|B|C)
    past_key_values.layers.{i}.recurrent_states [1, 256, 64, 128] fp32  mamba      SSM state after the last token
    model.layers.{i}.mixer.gate.output         [N, 512]           fp32  moe        router logits x @ W.T in fp32
    model.layers.{i}.mixer.experts.top_k_index [N, 22]            int32 moe        chosen experts, ascending id
    model.layers.{i}.mixer.experts.top_k_weights [N, 22]          fp32  moe        their weights, same order
  hidden/final.safetensors
    model.last_hidden_state                    [1, N, 8192]       bf16  final norm (norm_f)
    logits                                     [1, 1, 131072]     fp32  last position, untied lm_head

What is model-specific:

  * Attention (64 query heads, 2 KV heads of 128, scaling 128**-0.5) has no RoPE: the config's rope_theta is unused
    by the code, so keys are plain k_proj outputs. Position comes from the Mamba layers.
  * Mamba-2 (NemotronHMamba2Mixer, kernel path): in_proj -> [gate 16384 | xBC 18432 | dt 256]; causal depthwise
    conv1d (kernel 4, bias) + SiLU on xBC = [x 16384 | B 8x128 | C 8x128]; SSD scan (256 heads of 64, state 128,
    8 groups, chunk_size 128, dt softplus(dt + dt_bias), A = -exp(A_log), + D skip); gated RMSNorm (8 groups) with
    silu(gate); out_proj.
  * Latent MoE: router sigmoid + e_score_correction_bias (fp32, n_group 1) for a top-22 of 512, weights from the
    uncorrected sigmoid renormalized x routed_scaling_factor 5.0; tokens go 8192 -> 2048 (fc1_latent_proj) into
    non-gated relu^2 experts (up 2048 -> 5120, down 5120 -> 2048), back 2048 -> 8192 (fc2_latent_proj); the shared
    expert (relu^2, 8192 -> 10240 -> 8192) runs on the un-projected input and is added unweighted.
  * Chunked prefill: transformers treats any Mamba call after the first as ONE decode token (both its kernel and
    its torch path), so --prefill-chunk runs every chunk through mamba_prefill below: the same kernels and calls as
    the upstream prefill branch, plus the cached state (the last 3 conv inputs prepended, the SSM state passed as
    initial_states). The SSM state is carried in fp32 (upstream's cache would round it to bf16). The scan kernel
    resumed at a 128-multiple is bit-identical to one pass; --selftest checks the whole path against one upstream
    forward. CUDA only (the kernels); the CPU selftest runs unchunked through the torch path.
  * Kernels: mamba-ssm / causal-conv1d from the kernels hub, which transformers 5.12.1 loads with kernels 0.12.x
    (newer kernels break its loader). Without them the naive torch scan needs ~85 GB of intermediates at isl 10240.
  * Checkpoint: "backbone." prefix, routed experts per expert (stacked into experts.{up,down}_proj here, as
    transformers' nemotron_h conversion does). The MTP layer (mtp.*) is never built nor read.

    uv run --no-project --python <venv python> --with kernels==0.12.3 python generate_golden_nemotron_3_ultra.py \\
        --isl 10240 --prefill-chunk 5120
    ... --isl 256 --num-layers 8 --root /tmp/smoke   # block 7 is the first attention block (needed by the cache)
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import LayerwiseGolden, Recipe, RunState  # noqa: E402
from transformers.models.nemotron_h import modeling_nemotron_h as nh  # noqa: E402

NOTES = {
    "model.embed_tokens.output": "Token embeddings (no scale).",
    "model.layers.{i}.output": "Block output, residual included: x + mixer(norm(x)); the next block's input.",
    "model.layers.{i}.mixer.output": "The block's mixer (Mamba-2, latent MoE or attention), before the residual add (--sublayers).",
    "past_key_values.layers.{i}.keys": "Keys [1, num_key_value_heads, N, head_dim]: k_proj only, no RoPE (the model has no positional encoding in attention).",
    "past_key_values.layers.{i}.values": "Values [1, num_key_value_heads, N, head_dim]: v_proj.",
    "past_key_values.layers.{i}.conv_states": (
        "Mamba conv state [1, conv_dim = 16384 + 2*8*128, conv_kernel 4]: the last 4 inputs of the depthwise conv1d "
        "(the xBC slice of in_proj, before conv and SiLU), oldest first."
    ),
    "past_key_values.layers.{i}.recurrent_states": (
        "Mamba SSM state [1, mamba_num_heads, mamba_head_dim, ssm_state_size] after the last token, fp32 as the scan "
        "kernel returns it (transformers' cache would hold it in the compute dtype)."
    ),
    "model.layers.{i}.mixer.gate.output": "Router logits [N, n_routed_experts] = x @ gate.weight.T in fp32, before the sigmoid.",
    "model.layers.{i}.mixer.experts.top_k_index": (
        "The num_experts_per_tok (22) chosen experts per token, ascending by expert id: top-k of sigmoid(logits) + "
        "e_score_correction_bias (fp32). int32."
    ),
    "model.layers.{i}.mixer.experts.top_k_weights": (
        "Weights of top_k_index, same order: sigmoid(logits) of the chosen experts (no bias) / (their sum + 1e-20) x "
        "routed_scaling_factor (5.0), fp32."
    ),
    "model.last_hidden_state": "Final RMSNorm (norm_f) of the last block's output.",
    "logits": "Last position only: lm_head (not tied to the embeddings) in the compute dtype, stored fp32.",
}

# The engine reads the backbone's final norm as ``.norm``; NemotronHModel calls it norm_f. A property, so no module is
# registered twice (named_modules, state_dict and the checkpoint prefix stay norm_f).
nh.NemotronHModel.norm = property(lambda self: self.norm_f)


def mamba_prefill(mixer, hidden_states: torch.Tensor, cache) -> torch.Tensor:
    """cuda_kernels_forward's prefill branch (no padding mask), resumed from ``cache``'s state for every chunk after
    the first. The conv input is built channel-last, as upstream passes it, so the same causal_conv1d kernel runs."""
    b, n, _ = hidden_states.shape
    layer = cache.layers[mixer.layer_idx]
    resume = layer.has_previous_state
    k = mixer.conv_kernel_size
    if n < k:
        raise SystemExit(f"--prefill-chunk: a chunk of {n} tokens is shorter than the conv kernel ({k})")
    gate, xbc, dt = torch.split(mixer.in_proj(hidden_states), [mixer.intermediate_size, mixer.conv_dim, mixer.num_heads], -1)

    # The cache keeps the last k conv inputs, as upstream's update_conv_state does with the prefill's padded tail. A copy:
    # the update below overwrites the cache buffer in place.
    previous = layer.conv_states[..., 1:].transpose(1, 2).clone() if resume else None  # [b, k - 1, conv_dim]
    cache.update_conv_state(nn.functional.pad(xbc.transpose(1, 2), (k - n, 0)), mixer.layer_idx)
    x = xbc if previous is None else torch.cat([previous, xbc], dim=1)
    x = nh.causal_conv1d_fn(
        x=x.transpose(1, 2), weight=mixer.conv1d.weight.squeeze(1), bias=mixer.conv1d.bias, activation=mixer.activation
    ).transpose(1, 2)[:, x.shape[1] - n :]

    gs = mixer.n_groups * mixer.ssm_state_size
    x, B, C = torch.split(x, [mixer.intermediate_size, gs, gs], dim=-1)
    dt_limit = {} if mixer.time_step_limit is None else {"dt_limit": mixer.time_step_limit}
    y, state = nh.mamba_chunk_scan_combined(
        x.view(b, n, -1, mixer.head_dim),
        dt,
        -torch.exp(mixer.A_log.float()),
        B.view(b, n, mixer.n_groups, -1),
        C.view(b, n, mixer.n_groups, -1),
        chunk_size=mixer.chunk_size,
        D=mixer.D,
        z=None,
        seq_idx=None,
        initial_states=layer.recurrent_states if resume else None,
        return_final_states=True,
        dt_bias=mixer.dt_bias,
        dt_softplus=True,
        **dt_limit,
    )
    layer.recurrent_states, layer.is_recurrent_states_initialized = state, True  # fp32, not cast into the cache
    y = mixer.norm(y.view(b, n, -1), gate)
    return mixer.out_proj(y.to(mixer.out_proj.weight.dtype))


_upstream_mixer_forward = nh.NemotronHMamba2Mixer.forward


def _mixer_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    if Nemotron3UltraGolden.chunked_mamba and cache_params is not None and hidden_states.shape[1] > 1:
        return mamba_prefill(self, hidden_states, cache_params)
    return _upstream_mixer_forward(self, hidden_states, cache_params, attention_mask, **kwargs)


nh.NemotronHMamba2Mixer.forward = _mixer_forward


class Nemotron3UltraGolden(LayerwiseGolden):
    hf_repo = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
    checkpoint_prefix = "backbone."
    # None: the tokenizer adds no BOS by itself. Wikitext ppl over the same 10100 tokens at isl 10240: 5.073 (top-1
    # 62.5%) raw, 5.075 after <s>, 5.238 framed by the chat template as its own reply ("<|im_start|>system\n<|im_end|>
    # \n<|im_start|>user\nContinue this WikiText-2 dump verbatim: ...<|im_end|>\n<|im_start|>assistant\n<think></think>").
    prefix = ""
    chunked_mamba = False  # set by chunked_layer: only the layer-by-layer run leaves the upstream Mamba path
    recipe = Recipe(
        modules=("mixer.gate",),
        sublayers=("mixer",),
        routing="mixer.experts",
        cache=("keys", "values", "conv_states", "recurrent_states"),
        # Routing is a top-k: rounding the logits can flip an expert at the cut, so they are never stored rounded.
        keep_fp32=("mixer.gate.output", "mixer.experts.top_k_weights", "recurrent_states"),
        config=(
            "layers_block_type",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "mamba_num_heads",
            "mamba_head_dim",
            "n_groups",
            "ssm_state_size",
            "conv_kernel",
            "chunk_size",
            "use_conv_bias",
            "mamba_hidden_act",
            "time_step_limit",
            "mamba_ssm_cache_dtype",
            "layer_norm_epsilon",
            "mlp_hidden_act",
            "moe_intermediate_size",
            "moe_latent_size",
            "moe_shared_expert_intermediate_size",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "n_group",
            "topk_group",
            "norm_topk_prob",
            "routed_scaling_factor",
            "rope_theta",
            "partial_rotary_factor",
            "tie_word_embeddings",
            "max_position_embeddings",
        ),
        notes=NOTES,
    )

    def generate(self, args):
        if args.selftest:  # the tiny checkpoint in the real one's layout (see real_layout)
            args.hf_repo = str(self.real_layout(Path(args.hf_repo)))
        return super().generate(args)

    def prepare(self, backbone, text_config, hidden, position_ids) -> RunState:
        """NemotronHModel.forward: a causal mask for attention blocks only (Mamba gets None when nothing is padded,
        MoE nothing); no RoPE anywhere."""
        from transformers.masking_utils import create_causal_mask

        cache = self.make_cache(text_config)
        mask = create_causal_mask(
            config=text_config, inputs_embeds=hidden, attention_mask=None, past_key_values=cache, position_ids=position_ids
        )
        return RunState(position_ids=position_ids, cache=cache, masks={"attention": mask}, rope={})

    def layer_kwargs(self, i, kind, state) -> dict:
        kwargs = dict(position_ids=state.position_ids, past_key_values=state.cache, use_cache=True)
        if kind == "attention":  # only attention reads a mask; a Mamba block takes a 2D padding mask, here none
            kwargs["attention_mask"] = state.masks["attention"]
        return kwargs

    def chunked_layer(self, i, layer, hidden, kind, state, captured, chunk):
        if kind == "mamba" and not hidden.is_cuda:
            raise SystemExit("--prefill-chunk: the chunked Mamba prefill runs on the CUDA kernels only")
        type(self).chunked_mamba = True
        try:
            return super().chunked_layer(i, layer, hidden, kind, state, captured, chunk)
        finally:
            type(self).chunked_mamba = False

    def convert(self, sd, prefix, module):
        """Routed experts per expert -> experts.up_proj [E, I, latent] / experts.down_proj [E, latent, I]
        (transformers' nemotron_h MergeModulelist)."""
        groups: dict[str, dict[int, str]] = {}
        for k in list(sd):
            if m := re.fullmatch(r"(mixer\.experts)\.(\d+)\.((?:up|down)_proj)\.weight", k):
                groups.setdefault(f"{m[1]}.{m[3]}", {})[int(m[2])] = k
        for name, by_id in groups.items():
            sd[name] = torch.stack([sd.pop(by_id[e]) for e in range(len(by_id))])
        return sd

    # ---- --selftest ----------------------------------------------------------------------------------------------

    def real_layout(self, ckpt: Path) -> Path:
        """A copy of the tiny checkpoint with the embedding under backbone.embeddings, as the real checkpoint has it:
        save_pretrained reverses nemotron_h's "embedding.weight" -> "embeddings.weight" renaming (meant for older
        checkpoints) and writes backbone.embedding.weight; from_pretrained reads either."""
        from safetensors.torch import load_file, save_file

        out = ckpt.parent / "ckpt_real_layout"
        shutil.copytree(ckpt, out)
        index = json.loads((out / "model.safetensors.index.json").read_text())
        real = {k: k.replace("backbone.embedding.weight", "backbone.embeddings.weight") for k in index["weight_map"]}
        index["weight_map"] = {real[k]: f for k, f in index["weight_map"].items()}
        for f in sorted(set(index["weight_map"].values())):
            save_file({real.get(k, k): v for k, v in load_file(str(out / f)).items()}, str(out / f), metadata={"format": "pt"})
        (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        return out

    def tiny_init(self, model):
        super().tiny_init(model)
        g = torch.Generator().manual_seed(2)
        for name, t in [*model.named_parameters(), *model.named_buffers()]:
            if name.endswith("e_score_correction_bias"):  # zero from from_config: randomize so loading it is tested
                t.copy_(torch.randn(t.shape, generator=g) * 0.1)
            elif name.endswith("norm.weight") or name.endswith("norm_f.weight"):  # O(1) norms, like real ones
                t.copy_(1 + torch.randn(t.shape, generator=g) * 0.1)
            elif name.endswith("A_log"):  # A = -exp(A_log) in [-16, -1], as Mamba initializes it
                t.copy_(torch.log(torch.empty(t.shape).uniform_(1, 16, generator=g)))

    def tiny_config(self):
        from transformers import NemotronHConfig

        return NemotronHConfig(
            vocab_size=512,
            hidden_size=64,
            layers_block_type=["mamba", "moe", "attention", "moe", "mamba", "attention"],
            mamba_num_heads=8,
            mamba_head_dim=16,
            n_groups=2,
            ssm_state_size=16,
            conv_kernel=4,
            chunk_size=16,  # < selftest isl: the scan crosses its own chunk boundaries
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            n_routed_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            moe_shared_expert_intermediate_size=48,
            moe_latent_size=32,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=5.0,
            num_nextn_predict_layers=0,
            tie_word_embeddings=False,
            pad_token_id=0,
        )


if __name__ == "__main__":
    sys.exit(Nemotron3UltraGolden().main())
