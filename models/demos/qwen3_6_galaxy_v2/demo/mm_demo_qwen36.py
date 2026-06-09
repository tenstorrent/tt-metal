# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B VLM demo: image/video prefill + multimodal DECODE on BH_GLX.

Stage-3 of the VLM bring-up. Composes:
  - Qwen36MMGenerator's vision pipeline (vision encoder seq-parallel across all
    32 chips, CPU splice) -> fused vision+text embeddings + 3D M-RoPE positions.
  - The proven paged prefill+traced-decode machinery from text_demo_qwen36.py.

The only differences vs the text demo:
  - PREFILL consumes the fused embeddings (vision spliced) and M-RoPE built from
    the REAL position_ids_3d (vision tokens compress positions).
  - DECODE tracks TWO counters the text demo conflates: the KV/sequence index
    (cur_pos = real fused length) and the rope position (max(pos_3d)+1, smaller
    because vision tokens share spatial positions). Both advance by 1 per step.
    Text-after-image positions are axes-equal, so M-RoPE degenerates to the
    model's 1D partial-RoPE — the existing decode rope path is exact.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=BH_GLX
    source python_env/bin/activate
    python -m pytest --noconftest models/demos/qwen3_6_galaxy_v2/demo/mm_demo_qwen36.py -v -s
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.demos.qwen3_6_galaxy_v2.demo.text_demo_qwen36 import (  # noqa: F401  direct-open mesh fixture (avoids GetNumAvailableDevices); run with --noconftest
    _SNAPSHOT,
    _build_paged_page_table,
    _build_tt_model_paged,
    _gather_prefill_logits_to_cpu,
    _load_full_state_dict,
    _send_col_sharded_hidden,
    bh_glx_mesh,
)

_N_LAYERS = 64
_PATTERN = (["linear_attention"] * 3 + ["full_attention"]) * 16
_DECODE_STEPS = int(os.environ.get("QWEN36_MM_DECODE_STEPS", "40"))
_IMAGE_PATH = os.environ.get("QWEN36_MM_IMAGE", "models/demos/multimodal/gemma3/dog.jpg")
_PROMPT = os.environ.get("QWEN36_MM_PROMPT", "<|vision_start|><|image_pad|><|vision_end|>What is in this image?")


def _build_mrope_prefill_cos_sin(mesh, position_ids_3d: torch.Tensor):
    """M-RoPE cos/sin for prefill from the REAL 3D positions.

    Uses the same `build_mrope_cos_sin` convention as the text demo's prefill
    helper and the model's decode rope_setup, so prefill-written KV and decode
    queries share one rope convention. position_ids_3d: [3, S].
    """
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    cos_ref, sin_ref = build_mrope_cos_sin(
        positions_3d=position_ids_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    upload = lambda t: ttnn.from_torch(  # noqa: E731
        t.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return upload(cos_ref), upload(sin_ref)


@pytest.mark.hardware
def test_mm_demo_qwen36(bh_glx_mesh):  # noqa: F811  (direct-open mesh fixture from text_demo_qwen36)
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig
    from models.demos.qwen3_6_galaxy_v2.tt.generator import get_padded_prefill_len
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mm_generator import Qwen36MMGenerator
    from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
    from models.tt_dit.parallel.manager import CCLManager

    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = "Qwen/Qwen3.6-27B"

    # --- Load weights + build the paged text model (decode-capable) ---
    full_sd = _load_full_state_dict(_SNAPSHOT)
    block_size = 32
    # Small multimodal prompts pad to the 4096 prefill bucket (pad-everything rule).
    # max_batch_size defaults to 1, so page_table=[1, max_blocks] and the per-seq KV
    # capacity is block_size * max_blocks — size it to cover 4096 + decode steps.
    _prefill_cap = int(os.environ.get("QWEN36_MM_PREFILL_CAP", "4096"))
    max_blocks = max(64, (_prefill_cap + _DECODE_STEPS + block_size - 1) // block_size + 8)
    paged_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_blocks)
    model, args = _build_tt_model_paged(bh_glx_mesh, full_sd, _PATTERN, _N_LAYERS, paged_cfg)
    page_table_tt = _build_paged_page_table(bh_glx_mesh, args, paged_cfg)

    # --- Vision pipeline -> fused embeddings + 3D positions ---
    ccl_manager = CCLManager(bh_glx_mesh, num_links=1, topology=ttnn.Topology.Linear)
    vision_args = Qwen36VisionModelArgs(bh_glx_mesh, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
    text_embed_weight = full_sd["model.language_model.embed_tokens.weight"].float()
    gen = Qwen36MMGenerator(
        bh_glx_mesh, ccl_manager, vision_args, text_model=model, text_embed_weight=text_embed_weight
    )
    tok = gen.tokenizer

    img = Image.open(_IMAGE_PATH).convert("RGB").resize((224, 224))
    inputs, fused_unpadded = gen.prepare_inputs(_PROMPT, images=[img])  # fused: [1, S_unpadded, 5120]
    S_unpadded = fused_unpadded.shape[1]
    S = get_padded_prefill_len(S_unpadded)
    logger.info(f"[mm-demo] prompt tokens (incl. vision)={S_unpadded} -> padded prefill bucket={S}")
    _cap = paged_cfg.block_size * (paged_cfg.max_num_blocks // args.max_batch_size)
    assert S + _DECODE_STEPS <= _cap, (
        f"prefill bucket {S} + decode {_DECODE_STEPS} exceeds paged KV capacity {_cap}; "
        f"raise QWEN36_MM_PREFILL_CAP (currently {_prefill_cap})"
    )

    # Pad fused embeddings + 3D positions to the prefill bucket.
    pos3d = inputs.position_ids_3d  # [3, 1, S_unpadded]
    rope_pos_next = int(pos3d[:, :, :S_unpadded].max().item()) + 1  # first decode rope position
    if S > S_unpadded:
        pad = S - S_unpadded
        fused = torch.cat(
            [fused_unpadded, torch.zeros(*fused_unpadded.shape[:-2], pad, fused_unpadded.shape[-1])], dim=-2
        )
        last = pos3d[:, :, -1:].max().item()
        pad_pos = torch.arange(last + 1, last + 1 + pad, dtype=pos3d.dtype).view(1, 1, pad).expand(3, 1, pad)
        pos3d = torch.cat([pos3d, pad_pos], dim=-1)
    else:
        fused = fused_unpadded
    pos3d_2d = pos3d[:, 0, :]  # [3, S]

    # --- PREFILL (single pass) ---
    if hasattr(model, "tt_ccl") and hasattr(model.tt_ccl, "reset_gather_and_buffer_idx"):
        model.tt_ccl.reset_gather_and_buffer_idx()
    x_tt = _send_col_sharded_hidden(fused.to(torch.bfloat16), bh_glx_mesh, args)
    cos_tt, sin_tt = _build_mrope_prefill_cos_sin(bh_glx_mesh, pos3d_2d)
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    prefill_hidden = model.forward(
        x_tt,
        current_pos=None,
        rot_mats=(cos_tt, sin_tt),
        user_id=0,
        mode="prefill",
        page_table=page_table_tt,
        chunk_page_table=None,
        chunk_start_idx=chunk_start_idx_tt,
        start_pos=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    )
    last_logits = _gather_prefill_logits_to_cpu(prefill_hidden, bh_glx_mesh, args, model, last_token_idx=S_unpadded - 1)
    first_tok = int(last_logits.reshape(-1)[: args.vocab_size].float().argmax().item())
    logger.info(f"[mm-demo] first decode token = {first_tok} ({tok.decode([first_tok])!r})")

    # --- DECODE: KV/seq index = S_unpadded; rope position = max(pos_3d)+1 ---
    cur_pos_int = S_unpadded
    generated = [first_tok]
    cur_pos_tt = ttnn.from_torch(
        torch.tensor([cur_pos_int] * args.max_batch_size, dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    rot_idxs_tt = model.rope_setup.get_qwen36_rm_rot_idxs(rope_pos_next, on_host=False)
    tt_out_tok = ttnn.from_torch(
        torch.full((1, 1, 1, 32), first_tok, dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )

    _use_sampling = os.environ.get("QWEN36_SAMPLE", "0") != "0"
    tt_sampling = None
    if _use_sampling:
        from models.common.sampling.tt_sampling import TTSampling

        tt_sampling = TTSampling(
            mesh_device=bh_glx_mesh,
            tt_ccl=model.tt_ccl,
            args=args,
            k=torch.tensor([int(os.environ.get("QWEN36_TOP_K", "20"))] * 32, dtype=torch.int32),
            p=torch.tensor([float(os.environ.get("QWEN36_TOP_P", "0.95"))] * 32, dtype=torch.float32),
            temp=torch.tensor([float(os.environ.get("QWEN36_TEMP", "1.0"))] * 32, dtype=torch.float32),
        )

    model.set_trace_decode_mode(True)
    eos_ids = set(getattr(tok, "all_special_ids", []) or [])

    def _decode_step() -> int:
        cos, sin = model.rope_setup.get_qwen36_rm_rot_mats(rot_idxs_tt)
        x_emb_flat = ttnn.embedding(
            tt_out_tok,
            model.embd.weights,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        if os.environ.get("QWEN36_DECODE_L1_RESIDUAL", "1") == "1":
            x_emb = ttnn.reshape(x_emb_flat, ttnn.Shape([1, 1, x_emb_flat.shape[-2], x_emb_flat.shape[-1]]))
        else:
            x_emb_3d = ttnn.slice(
                x_emb_flat, [0, 0, 0], [1, 1, x_emb_flat.shape[-1]], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            x_emb_flat.deallocate(True)
            x_emb = ttnn.unsqueeze_to_4D(x_emb_3d)
        lm_out = model.forward(
            x_emb,
            current_pos=cur_pos_tt,
            rot_mats=(cos, sin),
            user_id=0,
            mode="decode",
            page_table=page_table_tt,
            chunk_page_table=None,
            chunk_start_idx=None,
            start_pos=0,
            get_last_token=-1,
            kv_cache=None,
            batch_size=1,
        )
        logits = lm_out[0] if isinstance(lm_out, list) else lm_out
        if tt_sampling is not None:
            tt_sampling(logits, tt_out_tok=tt_out_tok)
            tok_id = int(ttnn.to_torch(ttnn.get_device_tensors(tt_out_tok)[0]).reshape(-1)[0].item())
        else:
            num_links = min(3, model.model_config["GALAXY_NUM_LINKS"])
            logits_bf16 = ttnn.typecast(logits, dtype=ttnn.bfloat16)
            logits_full = model.tt_ccl.line_all_gather(
                logits_bf16, dim=3, num_links=num_links, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            logits_bf16.deallocate(True)
            logits_unt = ttnn.untilize(logits_full, use_multicore=True)
            logits_full.deallocate(True)
            Vg = logits_unt.shape[-1]
            row0 = ttnn.slice(logits_unt, [0, 0, 0, 0], [1, 1, 1, Vg], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            logits_unt.deallocate(True)
            tok_1x1 = ttnn.argmax(row0, dim=3, keepdim=True, use_multicore=True)
            row0.deallocate(True)
            if isinstance(tok_1x1, list):
                tok_1x1 = tok_1x1[0]
            tok_id = int(ttnn.to_torch(ttnn.get_device_tensors(tok_1x1)[0]).reshape(-1)[0].item())
            tok_b = ttnn.repeat(tok_1x1, ttnn.Shape((1, 1, 1, 32)))
            tok_1x1.deallocate(True)
            ttnn.copy(input_a=tok_b, input_b=tt_out_tok)
            tok_b.deallocate(True)
        ttnn.plus_one(
            cur_pos_tt,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
            skip_negative_entries=True,
        )
        ttnn.plus_one(
            rot_idxs_tt, sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))])
        )
        return tok_id

    for _ in range(_DECODE_STEPS):
        nxt = _decode_step()
        generated.append(nxt)
        if nxt in eos_ids:
            break

    text = tok.decode(generated, skip_special_tokens=True)
    logger.info(f"[mm-demo] generated {len(generated)} tokens")
    logger.info(f"[mm-demo] OUTPUT: {text!r}")
    # Coherence gate: non-empty, not a single repeated token.
    assert len(text.strip()) > 0, "empty generation"
    assert len(set(generated)) > 2, f"degenerate generation (repeated token): {generated}"
