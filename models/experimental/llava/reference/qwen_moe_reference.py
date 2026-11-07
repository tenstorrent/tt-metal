#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference dump tool for MoE-LLaVA-Qwen-1.8B-4e.

Generates cached activation tensors (RMSNorm, attention, dense MLP, MoE layer,
single decoder block, full model prefill logits) so the equivalent TTNN ports
can compare PCC against a stable golden without re-loading the HF model on
every pytest run. The MoE wrapper from MoE-LLaVA relies on DeepSpeed for its
`MoE` layer; to keep this dump CPU-only and free of DeepSpeed, we re-bind
`EvalMoELLaVAQWenForCausalLM.__init__`'s MoE constructor to a small pure
PyTorch top-2 routing block that matches the eval-time math.

All dumps are saved under ``./reference_cache/qwen_moe/`` by default.

Usage:
    python models/demos/llava/reference/qwen_moe_reference.py \\
        --ckpt LanguageBind/MoE-LLaVA-Qwen-1.8B-4e \\
        --seq-len 32 --batch 1 --out-dir ./reference_cache/qwen_moe

Validation is CPU-only; this tool imports neither ttnn nor the NPU.

Reference:
    MoE-LLaVA/moellava/model/language_model/llava_qwen_moe.py
    MoE-LLaVA/moellava/model/language_model/qwen/modeling_qwen.py
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger


def _stub_deepspeed_moe():
    """Inject a tiny pure-PyTorch top-2 MoE before importing the HF model.

    The MoE-LLaVA `EvalMoELLaVAQWenForCausalLM` calls ``deepspeed.moe.layer.MoE``
    to replace each `QWenMLP` on `moe_layers_idx`. DeepSpeed is unavailable in
    this CPU-only context, and we don't need its capacity/aux-loss machinery to
    produce golden activations — only the top-2 routing math. So we register a
    minimal `MoE` class in `sys.modules['deepspeed.moe.layer']` that mirrors the
    inference-time behaviour:
        1. Linear gate (wg) over hidden_size -> num_experts
        2. Top-2 softmax over gate logits
        3. Weighted sum of the top-2 experts' MLP outputs
        4. Returns ``(out, moe_loss=0.0)`` so MoEQWenBlock_forward's 3-tuple
           unpack succeeds.
    """
    import sys
    import types

    if "deepspeed" in sys.modules:
        return

    ds = types.ModuleType("deepspeed")
    moe_pkg = types.ModuleType("deepspeed.moe")
    layer_pkg = types.ModuleType("deepspeed.moe.layer")

    class MoE(torch.nn.Module):
        def __init__(
            self,
            hidden_size,
            expert,
            num_experts,
            ep_size=1,
            k=2,
            capacity_factor=1.0,
            eval_capacity_factor=1.0,
            min_capacity=0,
            use_residual=False,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.k = k
            self.num_experts = num_experts
            # Replicate the expert class instance-wise so each expert keeps its
            # own copy of the dense QWenMLP weights loaded from the checkpoint.
            # The HF state-dict has per-expert weight keys, so we re-bind the
            # expert module's weights after the wrapper is constructed (handled
            # by the normal HF from_pretrained loader).
            self.experts = torch.nn.ModuleList(
                [type(expert)(expert._config_for_dump) for _ in range(num_experts)]
                if hasattr(expert, "_config_for_dump")
                else [expert for _ in range(num_experts)]
            )
            self.wg = torch.nn.Linear(hidden_size, num_experts, bias=False)

        def forward(self, x):
            # x: [B, S, H]
            gate_logits = self.wg(x)
            topk_values, topk_indices = torch.topk(gate_logits, self.k, dim=-1)
            topk_weights = F.softmax(topk_values, dim=-1)
            out = torch.zeros_like(x)
            for tok_idx in range(x.shape[0] * x.shape[1]):
                # Per-token top-2 selection (vectorised version would be faster
                # but this is golden-generation only — clarity over speed).
                b = tok_idx // x.shape[1]
                s = tok_idx % x.shape[1]
                for k_slot in range(self.k):
                    expert_id = int(topk_indices[b, s, k_slot].item())
                    weight = topk_weights[b, s, k_slot]
                    expert_out = self.experts[expert_id](x[b : b + 1, s : s + 1, :])
                    out[b, s, :] = out[b, s, :] + weight * expert_out.squeeze(0).squeeze(0)
            return out, torch.tensor(0.0, device=x.device)

    layer_pkg.MoE = MoE
    moe_pkg.layer = layer_pkg
    ds.moe = moe_pkg
    sys.modules["deepspeed"] = ds
    sys.modules["deepspeed.moe"] = moe_pkg
    sys.modules["deepspeed.moe.layer"] = layer_pkg

    # The Qwen-1_8B remote modeling code imports transformers_stream_generator
    # (a generation-only helper). The published 0.0.5 release is incompatible
    # with recent transformers (it imports names removed from transformers),
    # and we never HF-generate here — the stub satisfies the remote-code import
    # scan with a permissive module that fabricates any requested symbol.
    if "transformers_stream_generator" not in sys.modules:
        try:
            import transformers_stream_generator  # noqa: F401
        except Exception:

            class _StubStreamGeneratorModule(types.ModuleType):
                def __getattr__(self, name):
                    if name.startswith("__"):
                        raise AttributeError(name)
                    return type(name, (), {})

            sys.modules["transformers_stream_generator"] = _StubStreamGeneratorModule("transformers_stream_generator")


def load_model(ckpt_name: str, dtype: torch.dtype = torch.float32):
    """Load MoE-LLaVA-Qwen-1.8B-4e on CPU with DeepSpeed stubbed out."""
    _stub_deepspeed_moe()

    from transformers import AutoConfig, AutoModelForCausalLM

    logger.info(f"Loading MoE-LLaVA-Qwen config: {ckpt_name}")
    config = AutoConfig.from_pretrained(ckpt_name, trust_remote_code=True)

    logger.info(f"Loading MoE-LLaVA-Qwen weights: {ckpt_name}")
    try:
        # transformers >= 5 renamed torch_dtype -> dtype; the old kwarg is
        # silently ignored there, leaving the model in the checkpoint's fp16.
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_name,
            config=config,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_name,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    model.eval()
    # Belt-and-suspenders: some load paths keep the checkpoint dtype (fp16)
    # despite the dtype kwarg; force the requested dtype everywhere.
    if next(model.parameters()).dtype != dtype:
        model = model.to(dtype)

    # transformers-version compat: the frozen Qwen-1_8B remote code predates
    # the API move of get_head_mask (used inside QWenModel.forward). Bind a
    # no-op head-mask provider so the CPU reference forward runs on modern
    # transformers (head masking is irrelevant for golden logits).
    if not hasattr(model.transformer, "get_head_mask"):
        model.transformer.get_head_mask = lambda head_mask, num_layers: [None] * num_layers

    # The hub config's auto_map resolves AutoModelForCausalLM to the PLAIN
    # Qwen-1_8B class, whose MLPs are dense — the serialized checkpoint's
    # DeepSpeed-style MoE keys (mlp.deepspeed_moe.*) never load and the
    # experts stay randomly initialized. Rebuild each MoE layer's MLP as a
    # checkpoint-faithful top-2 module exposing the same key paths, then
    # (re)load the checkpoint into the fixed-up module tree.
    moe_layer_ids = list(getattr(config, "moe", {}).get("moe_layers_idx", []) or [])
    if moe_layer_ids:
        _install_checkpoint_moe(model, ckpt_name, moe_layer_ids, dtype)

    return model, config


def _install_checkpoint_moe(model, ckpt_name: str, moe_layer_ids, dtype):
    """Swap plain-Qwen MLPs for a top-2 MoE mirroring the serialized checkpoint layout:

        mlp.deepspeed_moe.gate.wg.weight            [num_experts, hidden]
        mlp.deepspeed_moe.experts.deepspeed_experts.{e}.{w1,w2,c_proj}.weight

    Qwen1 SwiGLU per expert (w1 = up, w2 = gate/silu, c_proj = down):
        c_proj(w1(x) * silu(w2(x)))
    with routing = softmax over the top-2 gate logits.
    """
    import torch.nn.functional as F
    from torch import nn

    class _QwenSwiGLU(nn.Module):
        """One Qwen1 expert MLP: c_proj(w1(x) * silu(w2(x)))."""

        def __init__(self, hidden: int, inter: int):
            super().__init__()
            self.w1 = nn.Linear(hidden, inter, bias=False)
            self.w2 = nn.Linear(hidden, inter, bias=False)
            self.c_proj = nn.Linear(inter, hidden, bias=False)

        def forward(self, x):
            return self.c_proj(self.w1(x) * F.silu(self.w2(x)))

    class _DSMoE(nn.Module):
        def __init__(self, hidden: int, inter: int, num_experts: int):
            super().__init__()
            self.gate = nn.Module()
            self.gate.wg = nn.Linear(hidden, num_experts, bias=False)
            self.experts = nn.Module()
            self.experts.deepspeed_experts = nn.ModuleList(_QwenSwiGLU(hidden, inter) for _ in range(num_experts))

        def forward(self, x):
            experts = self.experts.deepspeed_experts
            logits = self.gate.wg(x)  # [..., num_experts]
            vals, idx = torch.topk(logits, 2, dim=-1)
            weights = torch.zeros_like(logits).scatter(-1, idx, F.softmax(vals, dim=-1))
            outs = torch.stack([ex(x) for ex in experts], dim=-2)  # [..., E, H]
            return (outs * weights.unsqueeze(-1)).sum(dim=-2)

    class _DSMoEWrapper(nn.Module):
        def __init__(self, hidden, inter, num_experts):
            super().__init__()
            self.deepspeed_moe = _DSMoE(hidden, inter, num_experts)

        def forward(self, x):
            return self.deepspeed_moe(x)

    from transformers import AutoConfig as _AutoConfig

    ckpt_config = _AutoConfig.from_pretrained(ckpt_name, trust_remote_code=True)
    hidden = ckpt_config.hidden_size
    inter = ckpt_config.intermediate_size // 2  # QWenMLP halves the config value
    num_experts = ckpt_config.moe.get("num_experts", [4])
    num_experts = num_experts[0] if isinstance(num_experts, (list, tuple)) else int(num_experts)

    for l in moe_layer_ids:
        model.transformer.h[l].mlp = _DSMoEWrapper(hidden, inter, num_experts)

    # Reload the checkpoint into the fixed-up tree (dense layers, attention,
    # norms, embeddings, lm_head and the rebuilt MoE weights all match now).
    import glob as _glob
    import os as _os

    from huggingface_hub import hf_hub_download

    state_dict = {}
    for f in sorted(
        _glob.glob(
            _os.path.join(
                hf_hub_download(ckpt_name, "pytorch_model.bin.index.json").rsplit("/", 1)[0], "pytorch_model*.bin"
            )
        )
    ):
        state_dict.update(torch.load(f, map_location="cpu", weights_only=True))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if "rotary" not in k and "masked_bias" not in k and "kv_cache" not in k]
    unexpected = [k for k in unexpected if "rotary" not in k and "masked_bias" not in k]
    if missing or unexpected:
        logger.warning(f"Checkpoint fix-up load: missing={missing[:5]} unexpected={unexpected[:5]}")
    else:
        logger.info("Checkpoint fix-up load: complete (no missing/unexpected keys)")


def dump_component_activations(model, config, out_dir: Path, seq_len: int, batch: int):
    """Dump per-component activations: embed, RMSNorm, attn, MLP, MoE, block, full."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a tiny deterministic input
    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), dtype=torch.long)
    torch.save(input_ids, out_dir / "input_ids.pt")
    logger.info(f"Saved input_ids with shape {input_ids.shape}")

    # ---- Embedding (wte) ----
    embeds = model.transformer.wte(input_ids)
    torch.save(embeds, out_dir / "embedding_out.pt")
    logger.info(f"Saved embedding_out with shape {embeds.shape}")

    # ---- RMSNorm ln_1 of layer 0 ----
    ln_1_out = model.transformer.h[0].ln_1(embeds)
    torch.save(ln_1_out, out_dir / "layer0_ln1_out.pt")

    # ---- Attention layer 0 ----
    # Build rotary embeddings the way the Qwen model expects
    rotary_emb = model.transformer.rotary_emb
    rotary_pos_emb = rotary_emb(seq_len, ntk_alpha=1.0)
    attn_out, _ = model.transformer.h[0].attn(embeds, rotary_pos_emb_list=[rotary_pos_emb])
    torch.save(attn_out, out_dir / "layer0_attn_out.pt")

    # ---- Dense MLP on a non-MoE layer (e.g., layer 1) ----
    dense_layer = model.transformer.h[1]
    dense_mlp_out = dense_layer.mlp(embeds)
    torch.save(dense_mlp_out, out_dir / "layer1_dense_mlp_out.pt")

    # ---- MoE layer 0's MoE block ----
    moe_layer = model.transformer.h[0]
    if hasattr(moe_layer.mlp, "wg"):
        moe_out, moe_loss = moe_layer.mlp(embeds)
        torch.save(moe_out, out_dir / "layer0_moe_out.pt")
        torch.save(moe_loss, out_dir / "layer0_moe_loss.pt")
        logger.info(f"Saved layer0_moe_out with shape {moe_out.shape} (MoE loss={moe_loss.item():.6f})")
    else:
        logger.warning(f"Layer 0 mlp is not MoE (type={type(moe_layer.mlp).__name__}), skipping MoE dump")

    # ---- Full decoder block (layer 0) ----
    block_out = model.transformer.h[0](embeds, rotary_pos_emb_list=[rotary_pos_emb])
    if isinstance(block_out, tuple):
        block_out = block_out[0]
    torch.save(block_out, out_dir / "layer0_block_out.pt")

    # ---- Full model prefill logits ----
    with torch.no_grad():
        full_out = model(input_ids=input_ids)
    logits = full_out.logits
    torch.save(logits, out_dir / "full_prefill_logits.pt")
    logger.info(f"Saved full_prefill_logits with shape {logits.shape}")

    # ---- Weight exports for TTNN ports ----
    state_dict = model.state_dict()
    torch.save(state_dict, out_dir / "state_dict.pt")
    logger.info(f"Saved state_dict with {len(state_dict)} keys")

    return {
        "input_ids": input_ids,
        "embedding_out": embeds,
        "layer0_ln1_out": ln_1_out,
        "layer0_attn_out": attn_out,
        "layer1_dense_mlp_out": dense_mlp_out,
        "layer0_block_out": block_out,
        "full_prefill_logits": logits,
    }


def main():
    parser = argparse.ArgumentParser(description="MoE-LLaVA-Qwen-1.8B-4e PyTorch reference dump tool")
    parser.add_argument(
        "--ckpt",
        default="LanguageBind/MoE-LLaVA-Qwen-1.8B-4e",
        help="HF checkpoint name",
    )
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length for dump")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--out-dir",
        default="./reference_cache/qwen_moe",
        help="Directory to save cached tensors",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype to load the model in",
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    logger.info("=" * 80)
    logger.info("MoE-LLaVA-Qwen-1.8B-4e PyTorch Reference Dump")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info(f"Seq len: {args.seq_len}, Batch: {args.batch}")
    logger.info(f"Output dir: {args.out_dir}")
    logger.info(f"Dtype: {args.dtype}")

    out_dir = Path(args.out_dir)

    model, config = load_model(args.ckpt, dtype=dtype_map[args.dtype])

    logger.info(f"Model config:")
    logger.info(f"  hidden_size = {config.hidden_size}")
    logger.info(f"  num_hidden_layers = {config.num_hidden_layers}")
    logger.info(f"  num_attention_heads = {config.num_attention_heads}")
    logger.info(f"  intermediate_size = {config.intermediate_size}")
    logger.info(f"  vocab_size = {config.vocab_size}")
    logger.info(f"  moe_layers_idx = {getattr(config, 'moe', {}).get('moe_layers_idx', None)}")

    dump_component_activations(model, config, out_dir, args.seq_len, args.batch)

    logger.info("=" * 80)
    logger.info("Dump complete. Use these tensors as PyTorch goldens for:")
    logger.info("  - tt/qwen1_attention.py (layer0_attn_out.pt)")
    logger.info("  - tt/qwen1_mlp.py dense (layer1_dense_mlp_out.pt)")
    logger.info("  - tt/mixtral_moe.py generalized (layer0_moe_out.pt)")
    logger.info("  - decoder block integration (layer0_block_out.pt)")
    logger.info("  - end-to-end prefill (full_prefill_logits.pt)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
