# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Video decode parity: HuggingFace reference vs TTNN ``forward_decode`` with teacher forcing.

After identical video prefill (same processor settings as ``preprocess_video``), we compare
last-token logits (prefill), then run greedy decode steps driven by **HF argmax** so both
paths see the same prefix tokens. At each step we compare the next-token logit vectors
(PCC) between HF and TTNN.

**Inputs:** Same fixtures/CLI as ``test_video_pcc_reference_vs_ttnn`` (``--molmo2-video``,
``--molmo2-prompt``, ``MOLMO2_TEST_VIDEO``, ``MOLMO2_VIDEO_PROMPT``). Prompt should include
``<|video|>`` or it is normalized like the other video PCC test.

**Hardware:** Opens an 8-device mesh (same class of setup as ``run_video_demo``).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.molmo2.demo.demo import Molmo2Generator, create_model, load_model_weights, load_processor
from models.demos.molmo2.reference.model import Molmo2Reference
from models.demos.molmo2.tests.test_video_pcc_reference_vs_ttnn import (
    _hf_video_batch_from_processor,
    _video_demo_max_fps,
    _video_pcc_num_frames,
)
from models.demos.molmo2.tt.hf_processor import preprocess_video


def _decode_compare_steps() -> int:
    raw = os.environ.get("MOLMO2_VIDEO_DECODE_COMPARE_STEPS", "4")
    try:
        n = int(raw)
    except ValueError:
        return 4
    return max(1, n)


def _prefill_pcc_threshold() -> float:
    raw = os.environ.get("MOLMO2_VIDEO_DECODE_PREFILL_PCC", "0.95")
    try:
        return float(raw)
    except ValueError:
        return 0.95


def _decode_step_pcc_threshold() -> float:
    raw = os.environ.get("MOLMO2_VIDEO_DECODE_STEP_PCC", "0.90")
    try:
        return float(raw)
    except ValueError:
        return 0.90


def comp_pcc(a: torch.Tensor, b: torch.Tensor, pcc_threshold: float = 0.99) -> Tuple[float, bool]:
    """Pearson correlation between two 1-D logit vectors."""
    a = a.flatten().float()
    b = b.flatten().float()
    if a.shape != b.shape:
        return 0.0, False
    a_mean = a.mean()
    b_mean = b.mean()
    a_centered = a - a_mean
    b_centered = b - b_mean
    numerator = (a_centered * b_centered).sum()
    denominator = torch.sqrt((a_centered**2).sum() * (b_centered**2).sum())
    if denominator == 0:
        pcc = 1.0 if numerator == 0 else 0.0
    else:
        pcc = (numerator / denominator).item()
    return pcc, pcc >= pcc_threshold


def _hf_prefill_batch_for_decode(
    processor,
    video_path: str,
    prompt: str,
    num_frames: int,
    max_fps: Optional[float],
) -> Dict[str, Any]:
    """Video prefill kwargs with ``use_cache=True`` and no hidden-state overhead."""
    base = _hf_video_batch_from_processor(
        processor,
        video_path,
        prompt,
        num_frames=num_frames,
        max_fps=max_fps,
    )
    base.pop("output_hidden_states", None)
    base["use_cache"] = True
    return base


def _ttnn_prefill_last_logits(
    logits_ttnn: ttnn.Tensor,
    mesh_device: ttnn.Device,
    prefill_timing: Dict[str, Any],
    input_seq_len: int,
) -> torch.Tensor:
    """Extract the last real prompt position logit vector (handles chunked prefill)."""
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    logits_torch = ttnn.to_torch(logits_ttnn, mesh_composer=mesh_composer)[0].squeeze()

    original_seq_len = prefill_timing.get("original_seq_len", input_seq_len)
    if logits_torch.dim() == 2:
        if prefill_timing.get("chunked_prefill", False):
            logits_idx = prefill_timing["last_token_idx_in_chunk"]
        else:
            logits_idx = original_seq_len - 1
        return logits_torch[logits_idx, :].float().cpu()
    return logits_torch.float().cpu()


def _ttnn_decode_logits_vec(logits_ttnn: ttnn.Tensor, mesh_device: ttnn.Device) -> torch.Tensor:
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    lt = ttnn.to_torch(logits_ttnn, mesh_composer=mesh_composer)[0]
    if lt.dim() == 4:
        vec = lt[0, 0, 0, :]
    elif lt.dim() == 3:
        vec = lt[0, 0, :]
    else:
        vec = lt[0].flatten()
    return vec.float().cpu()


@pytest.mark.parametrize("num_frames", [_video_pcc_num_frames()])
def test_video_decode_pcc_reference_vs_ttnn_teacher_forced(
    molmo2_video_path: str,
    molmo2_video_prompt: str,
    model_location: str,
    num_frames: int,
):
    """
    Compare HF vs TTNN decode logits on the same video + prompt after shared prefill.

    Teacher forcing uses HF greedy tokens so both models always see identical discrete prefixes.
    """
    max_fps = _video_demo_max_fps()
    n_decode = _decode_compare_steps()
    t_pref = _prefill_pcc_threshold()
    t_step = _decode_step_pcc_threshold()

    video_prompt = """<|video|>
What will the person do next?
\nA. Put down the laptop.
\nB. Take the phone/camera.
\nC. Take the clothes.
\nD. Open the laptop.
\nPlease respond with only the letter of the correct answer."""
    logger.info(
        "Video decode PCC — video={}\nprompt={!r}\nnum_frames={}\nmax_fps={}\ndecode_steps={}",
        molmo2_video_path,
        video_prompt,
        num_frames,
        max_fps,
        n_decode,
    )

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_location, trust_remote_code=True)

    # --- HuggingFace: prefill with cache ---
    hf_batch = _hf_prefill_batch_for_decode(
        processor,
        molmo2_video_path,
        video_prompt,
        num_frames=num_frames,
        max_fps=max_fps,
    )
    ref = Molmo2Reference(model_location, torch_dtype=torch.bfloat16)
    hf_model = ref.model
    hf_model.eval()
    dev = next(hf_model.parameters()).device
    hf_batch_dev = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in hf_batch.items()}

    with torch.no_grad():
        hf_prefill_out = hf_model(**hf_batch_dev)
    hf_prefill_logits = hf_prefill_out.logits[0, -1, :].float().cpu()
    hf_past = hf_prefill_out.past_key_values
    del hf_prefill_out

    # --- TTNN: preprocess + prefill (same kwargs as demo path) ---
    video_inputs = preprocess_video(
        molmo2_video_path,
        video_prompt,
        num_frames=num_frames,
        max_fps=max_fps,
        processor=processor,
    )
    input_ids = video_inputs["input_ids"]
    n_fr = video_inputs["n_frames"]
    n_tok = video_inputs["n_tokens"]
    k_pool = video_inputs["k_pool"]
    n_out = n_tok // n_fr
    pooled_patches_idx = video_inputs["image_token_pooling"]
    if pooled_patches_idx.dim() == 2 and pooled_patches_idx.shape[0] == n_tok:
        pooled_patches_idx = pooled_patches_idx.reshape(n_fr, n_out, k_pool)
    pixel_values = video_inputs["pixel_values"]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, 8)
    mesh_device = ttnn.open_mesh_device(mesh_shape)
    try:
        state_dict = load_model_weights()
        model = create_model(
            mesh_device,
            state_dict,
            num_layers=None,
            max_batch_size=1,
            max_seq_len=65536,
        )
        tokenizer = load_processor()
        text_num_layers = 36
        generator = Molmo2Generator(
            mesh_device=mesh_device,
            model=model,
            tokenizer=tokenizer,
            num_layers=text_num_layers,
            batch_size=1,
            max_seq_len=65536,
            use_paged_attention=True,
            repetition_penalty=1.0,
        )

        logits_ttnn, prefill_timing = generator.run_prefill(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pooled_patches_idx=pooled_patches_idx,
            use_trace=False,
            use_vision_trace=False,
            use_unified_trace=False,
            use_dp_vision_trace=False,
            use_data_parallel=True,
            frames_per_device=8,
            token_type_ids=video_inputs.get("token_type_ids"),
            hf_attention_mask=video_inputs.get("attention_mask"),
        )

        ttnn_prefill_vec = _ttnn_prefill_last_logits(
            logits_ttnn,
            mesh_device,
            prefill_timing,
            input_ids.shape[1],
        )
        ttnn.deallocate(logits_ttnn)

        pcc_prefill, ok_prefill = comp_pcc(hf_prefill_logits, ttnn_prefill_vec, pcc_threshold=t_pref)
        logger.info("Prefill last-token logits PCC: {:.6f} (threshold {:.4f})", pcc_prefill, t_pref)
        assert ok_prefill, f"Prefill PCC {pcc_prefill:.6f} < {t_pref}"

        effective_page_table = generator.page_table if generator.use_paged_attention else None

        # Teacher token after prefill (HF greedy)
        current = int(torch.argmax(hf_prefill_logits).item())

        for step in range(n_decode):
            with torch.no_grad():
                hf_step = hf_model(
                    input_ids=torch.tensor([[current]], device=dev, dtype=torch.long),
                    past_key_values=hf_past,
                    use_cache=True,
                )
            hf_step_logits = hf_step.logits[0, -1, :].float().cpu()
            hf_past = hf_step.past_key_values
            del hf_step

            token_u32 = torch.tensor([[current]], dtype=torch.long)
            input_ids_ttnn = ttnn.from_torch(
                token_u32,
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=generator.mesh_mapper,
            )
            hidden_states = generator.model.text_model.embed_tokens(input_ids_ttnn)

            logits_dec = generator.model.text_model.forward_decode(
                hidden_states=hidden_states,
                kv_caches=generator.kv_caches,
                current_pos=generator.current_pos,
                rot_mat_idxs=generator.rot_mat_idxs,
                page_table=effective_page_table,
            )
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(hidden_states)

            ttnn_vec = _ttnn_decode_logits_vec(logits_dec, mesh_device)
            ttnn.deallocate(logits_dec)

            ttnn.plus_one(generator.current_pos)
            ttnn.plus_one(generator.rot_mat_idxs)
            generator.decode_position += 1

            pcc_d, ok_d = comp_pcc(hf_step_logits, ttnn_vec, pcc_threshold=t_step)
            logger.info(
                "Decode step {}: PCC={:.6f} (threshold {:.4f}), HF next greedy token={}",
                step + 1,
                pcc_d,
                t_step,
                int(torch.argmax(hf_step_logits).item()),
            )
            assert ok_d, f"Decode step {step + 1} PCC {pcc_d:.6f} < {t_step}"

            current = int(torch.argmax(hf_step_logits).item())

    finally:
        ttnn.close_mesh_device(mesh_device)
