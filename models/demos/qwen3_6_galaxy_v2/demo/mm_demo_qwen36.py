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
import time

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
# Video branch (opt-in): QWEN36_MM_VIDEO=<path to .mp4>. When unset the demo runs the
# existing image test, untouched.
_VIDEO_PATH = os.environ.get("QWEN36_MM_VIDEO", "")
_IMAGE_PROMPT = os.environ.get("QWEN36_MM_PROMPT", "<|vision_start|><|image_pad|><|vision_end|>What is in this image?")
_VIDEO_PROMPT = os.environ.get(
    "QWEN36_MM_VIDEO_PROMPT", "<|vision_start|><|video_pad|><|vision_end|>Describe this video."
)


def _load_video_frames(path: str):
    """Decode a video file into (frames [T, C, H, W] uint8 tensor, source_fps).

    Returns the FULL decoded frame sequence channels-first (the layout the HF
    Qwen3VLVideoProcessor expects for tensor inputs) + the source fps; the
    processor then uniformly samples to its stock 2 fps default (using the
    metadata we attach), so the on-device frame count matches HF exactly.
    """
    import av
    import numpy as np

    container = av.open(path)
    stream = container.streams.video[0]
    src_fps = float(stream.average_rate) if stream.average_rate else 24.0
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))  # [H, W, 3]
    container.close()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    arr = np.stack(frames, axis=0)  # [T, H, W, 3] uint8
    arr = np.transpose(arr, (0, 3, 1, 2))  # [T, C, H, W]
    return torch.from_numpy(np.ascontiguousarray(arr)), src_fps


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

    # --- Build the modality-specific prepare_inputs call (image by default; video
    # when QWEN36_MM_VIDEO is set). Both run the SAME prefill + sampled-decode flow.
    if _VIDEO_PATH:
        from transformers.video_utils import VideoMetadata

        frames, src_fps = _load_video_frames(_VIDEO_PATH)  # [T, C, H, W] uint8, source fps
        T_src = int(frames.shape[0])
        metadata = VideoMetadata(
            total_num_frames=T_src,
            fps=src_fps,
            width=int(frames.shape[3]),
            height=int(frames.shape[2]),
            duration=T_src / src_fps,
            video_backend="pyav",
            frames_indices=list(range(T_src)),
        )
        # HF processor samples to its stock 2 fps default using this metadata.
        prompt = _VIDEO_PROMPT
        _prepare = lambda: gen.prepare_inputs(prompt, videos=[frames], video_metadata=[metadata])  # noqa: E731
        logger.info(f"[mm-demo] VIDEO branch: {_VIDEO_PATH} ({T_src} src frames @ {src_fps:.2f} fps)")
    else:
        img = Image.open(_IMAGE_PATH).convert("RGB").resize((224, 224))
        prompt = _IMAGE_PROMPT
        _prepare = lambda: gen.prepare_inputs(prompt, images=[img])  # noqa: E731
        logger.info(f"[mm-demo] IMAGE branch: {_IMAGE_PATH}")

    # --- TIMING: vision encoder + preprocessing (HF processor + 27-layer seq-parallel
    # vision encoder on device + CPU splice). Run twice; report the 2nd (warm, kernels
    # compiled) so the number reflects steady-state vision-feature extraction.
    _prepare()  # warmup (compile vision kernels)
    ttnn.synchronize_device(bh_glx_mesh)
    _t = time.perf_counter()
    inputs, fused_unpadded = _prepare()  # fused: [1, S_unpadded, 5120]
    ttnn.synchronize_device(bh_glx_mesh)
    t_vision = time.perf_counter() - _t
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
    ttnn.synchronize_device(bh_glx_mesh)
    _t = time.perf_counter()
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
    ttnn.synchronize_device(bh_glx_mesh)
    # NOTE: single-pass prefill -> COLD (includes one-time kernel compilation), same
    # caveat as text_demo_qwen36. The warm/served prefill routes through the generator's
    # warmup-compiled + traced path.
    t_prefill = time.perf_counter() - _t
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

    # --- DECODE timing. Default = TRACED (capture the decode step once, replay it):
    # VL decode == text decode after the image (axes-equal positions -> 1D partial-RoPE),
    # so the proven text-demo traced path is exact here. QWEN36_MM_EAGER=1 keeps the old
    # per-step-dispatch loop (slower, for debugging). Mirrors text_demo_qwen36's
    # _run_decode_intrace; the ONLY difference is the rope-idx reset uses rope_pos_next
    # (the VL path's separate rope counter), not cur_pos_int.
    # Default = EAGER (correct, coherent). The inline single-buffer traced path below doubles/
    # degenerates on readback cadence (a single reused tt_out_tok buffer can't be pipelined safely);
    # the PROVEN traced VL decode is the generator async one-deep path in mm_perf_qwen36.py. Opt into
    # the inline traced path with QWEN36_MM_EAGER=0 only for experimentation.
    step_times = []
    if os.environ.get("QWEN36_MM_EAGER", "1") == "1":
        for i in range(_DECODE_STEPS):
            ttnn.synchronize_device(bh_glx_mesh)
            _t = time.perf_counter()
            nxt = _decode_step()
            ttnn.synchronize_device(bh_glx_mesh)
            step_times.append(time.perf_counter() - _t)
            generated.append(nxt)
            if nxt in eos_ids:
                break
    else:

        def _run_decode_intrace():
            # Same ops as _decode_step but NO host read (traces can't capture to_torch);
            # the sampler/argmax writes the next token in-place into tt_out_tok.
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
                rot_idxs_tt,
                sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
            )

        # compile pass, then reset the decode input buffers to the first-token state.
        _run_decode_intrace()
        ttnn.synchronize_device(bh_glx_mesh)
        cur_pos_reset = ttnn.from_torch(
            torch.tensor([cur_pos_int] * args.max_batch_size, dtype=torch.int32),
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
        )
        ttnn.copy_host_to_device_tensor(cur_pos_reset, cur_pos_tt)
        rot_idxs_reset = model.rope_setup.get_qwen36_rm_rot_idxs(rope_pos_next, on_host=True)  # VL: rope counter
        ttnn.copy_host_to_device_tensor(rot_idxs_reset, rot_idxs_tt)
        tt_out_tok_reset = ttnn.from_torch(
            torch.full((1, 1, 1, 32), first_tok, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
        )
        ttnn.copy_host_to_device_tensor(tt_out_tok_reset, tt_out_tok)
        if hasattr(model, "tt_ccl") and hasattr(model.tt_ccl, "reset_gather_and_buffer_idx"):
            model.tt_ccl.reset_gather_and_buffer_idx()
        ttnn.synchronize_device(bh_glx_mesh)
        # capture
        trace_id = ttnn.begin_trace_capture(bh_glx_mesh, cq_id=0)
        _run_decode_intrace()
        ttnn.end_trace_capture(bh_glx_mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(bh_glx_mesh)
        ttnn.copy_host_to_device_tensor(cur_pos_reset, cur_pos_tt)
        ttnn.copy_host_to_device_tensor(rot_idxs_reset, rot_idxs_tt)
        ttnn.copy_host_to_device_tensor(tt_out_tok_reset, tt_out_tok)
        ttnn.synchronize_device(bh_glx_mesh)
        # replay — match text_demo_qwen36's exact pattern (NO per-step sync; the reused
        # tt_out_tok buffer must be read right after each blocking execute_trace, else the
        # readback cadence is off and tokens double). Time the whole loop, not per-step.
        ttnn.synchronize_device(bh_glx_mesh)
        _loop_t0 = time.perf_counter()
        n_steps = 0
        for step in range(_DECODE_STEPS):
            ttnn.execute_trace(bh_glx_mesh, trace_id, cq_id=0, blocking=True)
            tok_t = ttnn.to_torch(ttnn.get_device_tensors(tt_out_tok)[0])
            nxt = int(tok_t.reshape(-1)[0].item())
            generated.append(nxt)
            n_steps += 1
            if nxt in eos_ids:
                break
        ttnn.synchronize_device(bh_glx_mesh)
        _avg = (time.perf_counter() - _loop_t0) / max(n_steps, 1)
        step_times = [_avg] * max(n_steps, 1)  # uniform per-step for the perf summary
        try:
            ttnn.release_trace(bh_glx_mesh, trace_id)
        except Exception:
            pass

    text = tok.decode(generated, skip_special_tokens=True)
    logger.info(f"[mm-demo] generated {len(generated)} tokens")
    logger.info(f"[mm-demo] OUTPUT: {text!r}")

    # --- Perf summary (VL) ---
    steady = step_times[1:] if len(step_times) > 1 else step_times
    avg_decode_s = sum(steady) / max(len(steady), 1)
    logger.info("=" * 70)
    logger.info("[mm-demo] VL PERF SUMMARY (BH_GLX, batch=1)")
    logger.info(f"  vision encoder + preprocess (warm) : {t_vision * 1e3:8.1f} ms  ({S_unpadded} tokens incl. vision)")
    logger.info(f"  prefill (COLD, S={S} bucket)        : {t_prefill * 1e3:8.1f} ms  (incl. 1-time compile)")
    logger.info(f"  decode step 0 (compile)            : {step_times[0] * 1e3:8.1f} ms")
    logger.info(
        f"  decode steady-state (eager)        : {avg_decode_s * 1e3:8.1f} ms/tok  = {1.0 / avg_decode_s:6.2f} tok/s"
    )
    logger.info("  (eager loop; production traced decode is faster — VL decode == text decode post-prefill)")
    logger.info("=" * 70)
    # Coherence gate: non-empty, not a single repeated token.
    assert len(text.strip()) > 0, "empty generation"
    assert len(set(generated)) > 2, f"degenerate generation (repeated token): {generated}"
