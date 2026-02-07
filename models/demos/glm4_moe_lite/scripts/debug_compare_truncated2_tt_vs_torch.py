# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn

from models.demos.glm4_moe_lite.tt.config import Glm4MoeLiteHParams
from models.demos.glm4_moe_lite.tt.decoder_layer_tt import (
    prepare_decode_rope_and_positions_tt,
    run_decoder_layer_decode_one_step_update_cache_tt,
)
from models.demos.glm4_moe_lite.tt.layer0_tt import _alloc_contiguous_page_table, _alloc_paged_kvpe_cache, _round_up
from models.demos.glm4_moe_lite.tt.model_tt import Glm4MoeLiteDenseOnlyTT
from models.demos.glm4_moe_lite.tt.reference_layer0 import _build_causal_mask, _build_minimal_config
from models.demos.glm4_moe_lite.tt.reference_moe import run_layer_moe_reference_from_hidden_states
from models.demos.glm4_moe_lite.tt.tt_embedding import run_tt_embedding
from models.demos.glm4_moe_lite.tt.weights import find_missing_shards, resolve_best_effort_snapshot_dir

from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMLP, Glm4MoeRMSNorm, Glm4MoeRotaryEmbedding
from transformers import AutoTokenizer


def _load_hparams(snapshot_dir: Path) -> Glm4MoeLiteHParams:
    cfg = json.loads((Path(snapshot_dir) / "config.json").read_text())
    hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()
    return hparams


def _default_golden_path(snapshot_dir: Path, *, num_layers: int, max_new_tokens: int) -> Path:
    snap_name = Path(snapshot_dir).name
    root = Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/golden"))
    return root / f"golden_tokens_{snap_name}_layers{num_layers}_new{max_new_tokens}.json"


def _load_layer_state_dict(*, state: dict[str, torch.Tensor], layer_idx: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{int(layer_idx)}."
    out: dict[str, torch.Tensor] = {}
    for full_key, value in state.items():
        if not full_key.startswith(prefix):
            continue
        out[full_key[len(prefix) :]] = value
    return out


@dataclass(frozen=True)
class TorchTruncated2Ref:
    snapshot_dir: Path
    hparams: Glm4MoeLiteHParams
    device: torch.device

    # Weights
    embed_w: torch.Tensor  # [V,H]
    lm_head_w: torch.Tensor  # [V,H]

    # Modules
    rotary: Glm4MoeRotaryEmbedding
    final_norm: Glm4MoeRMSNorm
    layer0_in_norm: Glm4MoeRMSNorm
    layer0_attn: DeepseekV3Attention
    layer0_post_norm: Glm4MoeRMSNorm
    layer0_mlp: Glm4MoeMLP
    layer1_in_norm: Glm4MoeRMSNorm
    layer1_attn: DeepseekV3Attention
    layer1_post_norm: Glm4MoeRMSNorm

    @classmethod
    def create(cls, *, snapshot_dir: Path, device: torch.device = torch.device("cpu")) -> "TorchTruncated2Ref":
        snapshot_dir = Path(snapshot_dir)
        hparams = _load_hparams(snapshot_dir)

        # Minimal config sufficient for DeepseekV3Attention + GLM RMSNorm/MLP/Rotary.
        config = _build_minimal_config(snapshot_dir)

        # Load only the needed weights via the same lazy safetensors state dict.
        # Note: load_glm_lazy_state_dict is a LazyStateDict, but we only need a
        # small subset, so materialize as a regular dict of torch tensors here.
        from models.demos.glm4_moe_lite.tt.weights import load_glm_lazy_state_dict

        lazy_state = load_glm_lazy_state_dict(snapshot_dir, num_layers=int(hparams.num_hidden_layers))
        needed: dict[str, torch.Tensor] = {}
        for k in lazy_state.keys():
            if (
                k.startswith("model.layers.0.")
                or k.startswith("model.layers.1.")
                or k in {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
            ):
                needed[k] = lazy_state[k]

        embed_w = needed["model.embed_tokens.weight"].to(device=device)
        lm_head_w = needed["lm_head.weight"].to(device=device)
        final_norm = Glm4MoeRMSNorm(config.hidden_size, config.rms_norm_eps).to(device=device)
        final_norm.load_state_dict({"weight": needed["model.norm.weight"].to(device=device)}, strict=True)

        rotary = Glm4MoeRotaryEmbedding(config=config).to(device=device)

        # Layer 0 modules.
        layer0_in_norm = Glm4MoeRMSNorm(config.hidden_size, config.rms_norm_eps).to(device=device)
        layer0_attn = DeepseekV3Attention(config, layer_idx=0).to(device=device)
        layer0_post_norm = Glm4MoeRMSNorm(config.hidden_size, config.rms_norm_eps).to(device=device)
        layer0_mlp = Glm4MoeMLP(config).to(device=device)

        layer0_state = _load_layer_state_dict(state=needed, layer_idx=0)
        # Keys match the module hierarchy: {input_layernorm, self_attn, post_attention_layernorm, mlp}.
        layer0_in_norm.load_state_dict({"weight": layer0_state["input_layernorm.weight"].to(device=device)}, strict=True)
        layer0_post_norm.load_state_dict(
            {"weight": layer0_state["post_attention_layernorm.weight"].to(device=device)}, strict=True
        )
        layer0_attn.load_state_dict(
            {k[len("self_attn.") :]: v.to(device=device) for k, v in layer0_state.items() if k.startswith("self_attn.")},
            strict=True,
        )
        layer0_mlp.load_state_dict({k[len("mlp.") :]: v.to(device=device) for k, v in layer0_state.items() if k.startswith("mlp.")}, strict=True)

        # Layer 1 attention modules (MLP is MoE, computed via reference_moe).
        layer1_in_norm = Glm4MoeRMSNorm(config.hidden_size, config.rms_norm_eps).to(device=device)
        layer1_attn = DeepseekV3Attention(config, layer_idx=1).to(device=device)
        layer1_post_norm = Glm4MoeRMSNorm(config.hidden_size, config.rms_norm_eps).to(device=device)

        layer1_state = _load_layer_state_dict(state=needed, layer_idx=1)
        layer1_in_norm.load_state_dict({"weight": layer1_state["input_layernorm.weight"].to(device=device)}, strict=True)
        layer1_post_norm.load_state_dict(
            {"weight": layer1_state["post_attention_layernorm.weight"].to(device=device)}, strict=True
        )
        layer1_attn.load_state_dict(
            {k[len("self_attn.") :]: v.to(device=device) for k, v in layer1_state.items() if k.startswith("self_attn.")},
            strict=True,
        )

        return cls(
            snapshot_dir=snapshot_dir,
            hparams=hparams,
            device=device,
            embed_w=embed_w,
            lm_head_w=lm_head_w,
            rotary=rotary,
            final_norm=final_norm,
            layer0_in_norm=layer0_in_norm,
            layer0_attn=layer0_attn,
            layer0_post_norm=layer0_post_norm,
            layer0_mlp=layer0_mlp,
            layer1_in_norm=layer1_in_norm,
            layer1_attn=layer1_attn,
            layer1_post_norm=layer1_post_norm,
        )

    @torch.no_grad()
    def forward_last_logits_and_hiddens(self, *, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x0_last, x1_last, logits_last) for a 2-layer forward on CPU.

        Shapes:
        - x0_last: [H] float32
        - x1_last: [H] float32
        - logits_last: [V] float32
        """
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError(f"expected input_ids [1,S], got {tuple(input_ids.shape)}")
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        _, seq_len = input_ids.shape

        x_embed = torch.nn.functional.embedding(input_ids, self.embed_w)  # [1,S,H]
        position_ids = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0)
        cos, sin = self.rotary(x_embed, position_ids)
        attn_mask = _build_causal_mask(1, seq_len, device=self.device)

        # ---- Layer 0 ----
        residual = x_embed
        x = self.layer0_in_norm(x_embed)
        attn_out, _ = self.layer0_attn(
            hidden_states=x,
            attention_mask=attn_mask,
            position_embeddings=(cos, sin),
            past_key_values=None,
            cache_position=None,
        )
        x_attn_out = residual + attn_out
        residual = x_attn_out
        x = self.layer0_post_norm(x_attn_out)
        x = self.layer0_mlp(x)
        x0_out = residual + x  # [1,S,H]
        x0_last = x0_out[0, -1].to(dtype=torch.float32)

        # ---- Layer 1 attention ----
        residual = x0_out
        x = self.layer1_in_norm(x0_out)
        attn_out, _ = self.layer1_attn(
            hidden_states=x,
            attention_mask=attn_mask,
            position_embeddings=(cos, sin),
            past_key_values=None,
            cache_position=None,
        )
        x_attn_out = residual + attn_out
        residual = x_attn_out
        x = self.layer1_post_norm(x_attn_out)

        # ---- Layer 1 MoE MLP (last token only) ----
        x_last = x[0, -1].reshape(1, -1).to(dtype=torch.bfloat16).to(dtype=torch.float32)
        moe = run_layer_moe_reference_from_hidden_states(self.snapshot_dir, layer_idx=1, hidden_states=x_last)
        x1_last = residual[0, -1].to(dtype=torch.float32) + moe.moe_out.reshape(-1).to(dtype=torch.float32)

        # ---- Head ----
        x1_last_norm = self.final_norm(x1_last.reshape(1, -1)).reshape(-1).to(dtype=torch.float32)
        logits = torch.nn.functional.linear(x1_last_norm.reshape(1, -1), self.lm_head_w.to(dtype=torch.float32)).reshape(-1)
        return x0_last, x1_last, logits.to(dtype=torch.float32)


@torch.no_grad()
def _tt_decode_step_with_intermediates(
    *,
    runner: Glm4MoeLiteDenseOnlyTT,
    tokens: torch.Tensor,  # [1,1] int32
    positions: torch.Tensor,  # [1] int32
    page_table_tt: ttnn.Tensor,
    kv_cache: list[ttnn.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Decode one step on TT, returning per-layer hidden + logits (host torch).

    Returns:
    - layer_hiddens: list of [H] float32, after each layer (post-MLP residual)
    - logits: [1,1,V] float32
    """
    device = runner.device
    is_mesh_device = device.__class__.__name__ == "MeshDevice"
    mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None

    tt_positions, cos_batch, sin_batch = prepare_decode_rope_and_positions_tt(device=device, rope=runner.rope, positions=positions)

    x = run_tt_embedding(device=device, token_ids=tokens.to(torch.int32), tt_weight=runner.embed_w)
    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    x = ttnn.reshape(x, (1, 1, 1, int(runner.hparams.hidden_size)))
    # Decode convention: batch lives in dim=2 => [1,1,B,H]
    x = ttnn.permute(x, (0, 2, 1, 3))  # [1,1,1,H]

    layer_hiddens: list[torch.Tensor] = []
    for layer_idx in range(runner.num_layers_to_run):
        w = runner._ensure_layer_weights(layer_idx)
        x_next = run_decoder_layer_decode_one_step_update_cache_tt(
            device=device,
            x_embed_tok=x,
            tt_positions=tt_positions,
            page_table_tt=page_table_tt,
            kvpe_cache=kv_cache[layer_idx],
            cos_batch=cos_batch,
            sin_batch=sin_batch,
            trans_matrix=runner.rope["trans_matrix"],
            w=w,
            hparams=runner.hparams,
            moe_runtime=runner.moe_runtime,
            profile=None,
        )
        ttnn.deallocate(x)
        x = x_next

        # Read back hidden for this layer.
        dev_tensor = ttnn.get_device_tensors(x)[0] if is_mesh_device else x
        x_torch = ttnn.to_torch(dev_tensor).reshape(-1).to(dtype=torch.float32).cpu()
        layer_hiddens.append(x_torch)

    x = runner.final_norm(x, mode="decode")
    logits_tt = ttnn.linear(x, runner.lm_head_w)  # [1,1,1,V]
    logits_dev = ttnn.get_device_tensors(logits_tt)[0] if is_mesh_device else logits_tt
    logits = ttnn.to_torch(logits_dev).to(dtype=torch.float32).cpu()

    ttnn.deallocate(logits_tt)
    ttnn.deallocate(x)
    ttnn.deallocate(tt_positions)
    ttnn.deallocate(cos_batch)
    ttnn.deallocate(sin_batch)
    return layer_hiddens, logits


def _topk(logits_1d: torch.Tensor, k: int = 5) -> list[tuple[int, float]]:
    v = logits_1d.to(dtype=torch.float32)
    topv, topi = torch.topk(v, k=min(k, int(v.numel())))
    return [(int(i), float(val)) for i, val in zip(topi.tolist(), topv.tolist())]


def _l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.norm((a - b).to(dtype=torch.float32)).item())


def _greedy_generate_reference(
    *, ref: TorchTruncated2Ref, prompt_ids: torch.Tensor, max_new_tokens: int
) -> tuple[list[int], torch.Tensor]:
    """Greedy-generate tokens from the Torch reference (no caching).

    Returns:
    - generated_ids: list[int] of length max_new_tokens
    - prompt_logits_last: [V] float32 logits for the first generated token (prompt last position)
    """
    if prompt_ids.ndim != 2 or int(prompt_ids.shape[0]) != 1:
        raise ValueError(f"expected prompt_ids [1,S], got {tuple(prompt_ids.shape)}")
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens <= 0:
        return [], torch.empty((0,), dtype=torch.float32)

    _, _, prompt_logits = ref.forward_last_logits_and_hiddens(input_ids=prompt_ids)
    prompt_logits = prompt_logits.reshape(-1).to(dtype=torch.float32)

    generated: list[int] = []
    seq = prompt_ids
    for _ in range(max_new_tokens):
        _, _, logits = ref.forward_last_logits_and_hiddens(input_ids=seq)
        logits_1d = logits.reshape(-1).to(dtype=torch.float32)
        next_id = int(torch.argmax(logits_1d).item())
        generated.append(next_id)
        seq = torch.cat([seq, torch.tensor([[next_id]], dtype=torch.int32)], dim=1)

    return generated, prompt_logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="zai-org/GLM-4.7-Flash")
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--golden-record-idx", type=int, default=0)
    ap.add_argument(
        "--prompt",
        default="Hello.",
        help="Used when no offline golden file is present (or when golden generation is unavailable).",
    )
    ap.add_argument("--print-every", type=int, default=1)
    args = ap.parse_args()

    snap = Path(resolve_best_effort_snapshot_dir(args.model_id))
    missing = find_missing_shards(snap)
    if missing:
        raise SystemExit(f"Snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")

    golden_path = _default_golden_path(snap, num_layers=int(args.num_layers), max_new_tokens=int(args.max_new_tokens))
    prompt_ids: torch.Tensor
    expected: list[int]
    prompt_logits_ref: torch.Tensor
    if golden_path.is_file():
        golden = json.loads(golden_path.read_text())
        record = golden["records"][int(args.golden_record_idx)]
        prompt_ids = torch.tensor([record["prompt_input_ids"]], dtype=torch.int32)
        expected = list(record["generated_ids"])
        prompt_logits_ref = torch.empty((0,), dtype=torch.float32)
        prompt_len = int(prompt_ids.shape[1])
        print(f"Using offline golden record: {golden_path} idx={int(args.golden_record_idx)} prompt_len={prompt_len}")
    else:
        tok = AutoTokenizer.from_pretrained(snap, local_files_only=True, use_fast=True)
        enc = tok(str(args.prompt), return_tensors="pt", add_special_tokens=True)
        prompt_ids = enc["input_ids"].to(dtype=torch.int32)
        prompt_len = int(prompt_ids.shape[1])
        print(f"No golden file found at {golden_path}; generating reference tokens from prompt={args.prompt!r}")

    # Match the golden test environment.
    os.environ["GLM4_MOE_LITE_ENABLE_MOE"] = "1"
    os.environ["GLM4_MOE_LITE_NUM_LAYERS"] = str(int(args.num_layers))
    os.environ["GLM4_MOE_LITE_DEBUG_ALLOW_PARTIAL_LAYERS"] = "1"
    os.environ["GLM4_MOE_LITE_EXPERTS_TT_DTYPE"] = "bf16"
    os.environ["GLM4_MOE_LITE_MOE_FP32_ACC"] = "1"

    hparams = _load_hparams(snap)
    ref = TorchTruncated2Ref.create(snapshot_dir=snap, device=torch.device("cpu"))
    if not golden_path.is_file():
        expected, prompt_logits_ref = _greedy_generate_reference(
            ref=ref, prompt_ids=prompt_ids, max_new_tokens=int(args.max_new_tokens)
        )

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[0],
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    try:
        block_size = 64
        total_len = prompt_len + int(args.max_new_tokens)
        blocks_per_seq = max(1, _round_up(total_len, block_size) // block_size)
        min_blocks_per_seq = max(1, _round_up(128, block_size) // block_size)
        blocks_per_seq = max(blocks_per_seq, min_blocks_per_seq)

        kvpe_dim = int(hparams.kv_lora_rank + hparams.qk_rope_head_dim)
        kv_cache = [
            _alloc_paged_kvpe_cache(
                device=mesh_device,
                max_num_blocks=int(1 * blocks_per_seq),
                block_size=block_size,
                kvpe_dim=kvpe_dim,
                dtype=ttnn.bfloat16,
            )
            for _ in range(int(args.num_layers))
        ]
        page_table = _alloc_contiguous_page_table(batch=1, blocks_per_seq=blocks_per_seq)
        page_table_tt = ttnn.from_torch(
            page_table.to(torch.int32),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        runner = Glm4MoeLiteDenseOnlyTT.create(
            device=mesh_device,
            snapshot_dir=snap,
            cache_dir=Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/truncated2_tt_cache")),
            max_seq_len=int(blocks_per_seq * block_size),
            hparams=hparams,
        )

        # Initialize KV caches for the prompt using iterative decode.
        #
        # Why:
        # - The FlashMLA prefill path can be slow to compile during bring-up.
        # - For debugging correctness drift, we only need a small prompt and a
        #   consistent KV cache state for subsequent decode steps.
        tt_prefill_logits = None
        for t in range(prompt_len):
            tok = prompt_ids[:, t : t + 1].contiguous()
            pos = torch.tensor([t], dtype=torch.int32)
            tt_prefill_logits = runner.decode(tokens=tok, start_pos=pos, page_table=page_table, kv_cache=kv_cache)
        assert tt_prefill_logits is not None
        tt_prefill_logits_1d = tt_prefill_logits.reshape(-1).to(dtype=torch.float32).cpu()
        _, _, ref_prefill_logits = ref.forward_last_logits_and_hiddens(input_ids=prompt_ids)
        ref_prefill_logits_1d = ref_prefill_logits.reshape(-1).to(dtype=torch.float32).cpu()
        print(
            "prefill: "
            f"tt_pred={int(torch.argmax(tt_prefill_logits_1d).item())} "
            f"ref_pred={int(torch.argmax(ref_prefill_logits_1d).item())} "
            f"l2_logits={_l2(tt_prefill_logits_1d, ref_prefill_logits_1d):.3f} "
            f"tt_top5={_topk(tt_prefill_logits_1d, k=5)} ref_top5={_topk(ref_prefill_logits_1d, k=5)}"
        )

        # Decode loop (teacher forced).
        token_in = int(expected[0])
        for step in range(int(args.max_new_tokens) - 1):
            pos = int(prompt_len + step)
            tokens = torch.tensor([[token_in]], dtype=torch.int32)
            positions = torch.tensor([pos], dtype=torch.int32)

            # TT decode step (with intermediates).
            tt_hiddens, tt_logits = _tt_decode_step_with_intermediates(
                runner=runner,
                tokens=tokens,
                positions=positions,
                page_table_tt=page_table_tt,
                kv_cache=kv_cache,
            )
            tt_logits_1d = tt_logits.reshape(-1)

            # Torch reference for the same prefix (prompt + generated so far incl current token_in).
            seq = torch.cat([prompt_ids, torch.tensor([expected[: step + 1]], dtype=torch.int32)], dim=1)
            ref_x0_last, ref_x1_last, ref_logits = ref.forward_last_logits_and_hiddens(input_ids=seq)

            exp_next = int(expected[step + 1])
            tt_top = _topk(tt_logits_1d, k=5)
            ref_top = _topk(ref_logits, k=5)

            if step % int(args.print_every) == 0:
                l2_l0 = _l2(tt_hiddens[0], ref_x0_last)
                l2_l1 = _l2(tt_hiddens[1], ref_x1_last)
                tt_pred = int(torch.argmax(tt_logits_1d).item())
                ref_pred = int(torch.argmax(ref_logits).item())
                tt_exp_logit = float(tt_logits_1d[exp_next].item())
                tt_pred_logit = float(tt_logits_1d[tt_pred].item())
                print(
                    f"step={step:02d} pos={pos} token_in={token_in} exp_next={exp_next} "
                    f"tt_pred={tt_pred} ref_pred={ref_pred} "
                    f"tt_gap={tt_pred_logit - tt_exp_logit:+.4f} "
                    f"l2_layer0={l2_l0:.3f} l2_layer1={l2_l1:.3f} "
                    f"tt_top5={tt_top} ref_top5={ref_top}"
                )

            token_in = int(expected[step + 1])

        ttnn.deallocate(page_table_tt)
        for t in kv_cache:
            ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh_device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
