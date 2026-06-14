# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B VLM SERVER-PATH perf: image prefill via Generator + traced decode.

Measures server-equivalent VL perf by routing through the Generator class (the
path tt-inference-server/vLLM uses):
  - PREFILL: Generator.prefill_forward_text_embeds(inputs_embeds=fused, rot_mats=M-RoPE)
    — the new qwen3_vl-style pre-embedded prefill (skips token embedding, populates
    paged KV + DeltaNet state). Timed COLD (1st, incl. compile) and WARM (re-run after
    decode, discarded — a 2nd in-place prefill corrupts state but the kernels are warm).
  - DECODE: Generator.decode_forward(enable_trace=True, on-device sampling) — the server
    traced decode. tok/s is valid timing (same kernels regardless of position).
  - VISION: the seq-parallel 27-layer encoder, timed warm.

Decode coherence: generator.set_decode_rope_offset() decouples the decode RoPE
position from the KV index (rope = cur_pos + offset, offset = (max(pos_3d)+1) -
S_unpadded <= 0, since vision tokens compress positions), so the Generator-path decode
is COHERENT — the benchmark reads back and prints the generated text. tok/s is
unaffected by the offset (identical kernels) and equals text decode.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=BH_GLX
    source python_env/bin/activate
    python -m pytest --noconftest models/demos/qwen3_6_galaxy_v2/demo/mm_perf_qwen36.py -v -s
"""
from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.demos.qwen3_6_galaxy_v2.demo.text_demo_qwen36 import (  # noqa: F401  direct-open mesh fixture
    _SNAPSHOT,
    _build_tt_model_paged_kv,
    _load_full_state_dict,
    bh_glx_mesh,
)

_N_LAYERS = 64
_PATTERN = (["linear_attention"] * 3 + ["full_attention"]) * 16
_DECODE_STEPS = int(os.environ.get("QWEN36_MM_DECODE_STEPS", "32"))
_IMAGE_PATH = os.environ.get("QWEN36_MM_IMAGE", "models/demos/multimodal/gemma3/dog.jpg")
_PROMPT = os.environ.get("QWEN36_MM_PROMPT", "<|vision_start|><|image_pad|><|vision_end|>What is in this image?")
# VIDEO branch (opt-in): set QWEN36_MM_VIDEO=<path to .mp4>. When unset the harness runs the
# existing IMAGE path, byte-for-byte unchanged. Video runs the SAME prefill + traced-decode flow.
_VIDEO_PATH = os.environ.get("QWEN36_MM_VIDEO", "")
_VIDEO_PROMPT = os.environ.get(
    "QWEN36_MM_VIDEO_PROMPT", "<|vision_start|><|video_pad|><|vision_end|>Describe this video."
)


def _load_video_frames(path: str):
    """Decode a video file -> (frames [T, C, H, W] uint8 tensor, source_fps).

    Channels-first is the layout the HF Qwen3VLVideoProcessor expects for tensor
    inputs; it then uniformly samples to its stock 2 fps default using the metadata
    we attach, so the on-device frame count matches HF exactly. (Copied from the
    mm_demo_qwen36.py video branch so the perf harness decodes clips identically.)
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


def _make_synthetic_clip(path: str, n_frames: int = 8, hw: int = 256, fps: int = 4) -> None:
    """Write a tiny moving-gradient mp4 (256x256) so the harness is self-contained
    when no real clip is supplied. Kept short so prompt+video tokens stay in the
    4096 prefill bucket."""
    import av
    import numpy as np

    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = hw
    stream.height = hw
    stream.pix_fmt = "yuv420p"
    for t in range(n_frames):
        img = np.zeros((hw, hw, 3), dtype=np.uint8)
        x = np.linspace(0, 255, hw, dtype=np.uint8)
        img[:, :, 0] = (x[None, :] + t * 16) % 256  # horizontal sweep, moves with t
        img[:, :, 1] = (x[:, None] + t * 8) % 256  # vertical sweep
        img[:, :, 2] = (t * 32) % 256
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


@pytest.mark.hardware
def test_mm_perf_qwen36(bh_glx_mesh):  # noqa: F811
    from models.common.sampling import SamplingParams
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin
    from models.demos.qwen3_6_galaxy_v2.tt.generator import Generator, get_padded_prefill_len
    from models.demos.qwen3_6_galaxy_v2.tt.generator_vllm import allocate_vllm_kv_cache
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mm_generator import Qwen36MMGenerator
    from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
    from models.tt_dit.parallel.manager import CCLManager

    # Tracy signposts bracket the three stages so aggregate_tracy_csv.py can split
    # the per-op device-kernel breakdown into vision / prefill / decode. No-op when
    # not running under `python -m tracy`.
    try:
        from tracy import signpost
    except ImportError:
        signpost = lambda *_a, **_k: None  # noqa: E731

    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
    }.items():
        os.environ.setdefault(_k, _v)
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = "Qwen/Qwen3.6-27B"

    # --- Paged model + Generator + paged KV (exactly like test_qwen36_demo_generator_batch1) ---
    state_dict = _load_full_state_dict(_SNAPSHOT)
    block_size = 32
    max_blocks = max(64, (4096 + _DECODE_STEPS + block_size - 1) // block_size + 8)
    paged_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_blocks)
    model, args = _build_tt_model_paged_kv(bh_glx_mesh, state_dict, _PATTERN, _N_LAYERS, paged_cfg)

    import torch as _torch

    permutation = _torch.randperm(paged_cfg.max_num_blocks)
    page_table = _torch.argsort(permutation).reshape(
        args.max_batch_size, paged_cfg.max_num_blocks // args.max_batch_size
    )
    _kv_shape = (paged_cfg.max_num_blocks, 1, paged_cfg.block_size, args.head_dim)
    tt_kv_cache = allocate_vllm_kv_cache(
        _kv_shape, torch.bfloat16, args.n_layers, model, args.weight_cache_path(ttnn.bfloat8_b)
    )

    generator = Generator(model, args, bh_glx_mesh)
    generator._disable_prefill_tracing = True
    generator.prefill_warmup_completed = True

    # --- Vision pipeline -> fused embeds + 3D positions (warm-timed) ---
    ccl_manager = CCLManager(bh_glx_mesh, num_links=1, topology=ttnn.Topology.Linear)
    vision_args = Qwen36VisionModelArgs(bh_glx_mesh, dummy_weights=False, max_batch_size=1, max_seq_len=2048)
    text_embed_weight = state_dict["model.language_model.embed_tokens.weight"].float()
    mmgen = Qwen36MMGenerator(
        bh_glx_mesh, ccl_manager, vision_args, text_model=model, text_embed_weight=text_embed_weight
    )
    tok = mmgen.tokenizer

    # --- Build the modality-specific prepare_inputs callable. IMAGE by default;
    # VIDEO when QWEN36_MM_VIDEO is set. Both run the identical prefill + traced-
    # decode flow below; only the vision-token source differs.
    if _VIDEO_PATH:
        from transformers.video_utils import VideoMetadata

        video_path = _VIDEO_PATH
        if video_path == "synthetic":
            video_path = os.path.join(os.getcwd(), "generated", "qwen36_mm_synthetic_clip.mp4")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            _make_synthetic_clip(video_path)
            logger.info(f"[mm-perf] generated synthetic clip at {video_path}")
        frames, src_fps = _load_video_frames(video_path)  # [T, C, H, W] uint8, source fps
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
        prompt = _VIDEO_PROMPT
        modality = "VIDEO"
        _prepare = lambda: mmgen.prepare_inputs(prompt, videos=[frames], video_metadata=[metadata])  # noqa: E731
        logger.info(f"[mm-perf] VIDEO branch: {video_path} ({T_src} src frames @ {src_fps:.2f} fps)")
    else:
        img = Image.open(_IMAGE_PATH).convert("RGB").resize((224, 224))
        prompt = _PROMPT
        modality = "IMAGE"
        # Optional chat-template wrapping (instruct-model assistant-turn priming + vision markers).
        if os.environ.get("QWEN36_MM_CHAT_TEMPLATE", "0") == "1":
            _question = os.environ.get("QWEN36_MM_QUESTION", "What is in this image?")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _question}]}]
            prompt = mmgen.pipeline.preprocessor.apply_chat_template(messages)
            logger.info(f"[mm-perf] chat-templated prompt: {prompt!r}")
        _prepare = lambda: mmgen.prepare_inputs(prompt, images=[img])  # noqa: E731
        logger.info(f"[mm-perf] IMAGE branch: {_IMAGE_PATH}")

    _prepare()  # warmup (compile vision kernels) — NOT signposted
    ttnn.synchronize_device(bh_glx_mesh)
    # Stage 1 of 3: vision encoder + preprocessing (HF processor + seq-parallel encoder
    # on device + HOST splice). NOTE: the host splice + a device->host->device roundtrip
    # are INCLUDED in t_vision (the on-device splice exists but is not wired into this
    # harness yet) — this inflates the vision-stage wall-clock; see PERF.md VL-PERF note.
    signpost("start")
    _t = time.perf_counter()
    inputs, fused_unpadded = _prepare()
    ttnn.synchronize_device(bh_glx_mesh)
    t_vision = time.perf_counter() - _t
    signpost("vision_done")

    S_unpadded = fused_unpadded.shape[1]
    S = get_padded_prefill_len(S_unpadded)
    pos3d = inputs.position_ids_3d  # [3, 1, S_unpadded]
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
    # Padded token ids (real ids + zero pad) for shape/page/last-token bookkeeping.
    ids = inputs.input_ids.to(torch.long)
    ids_padded = torch.cat([ids, torch.zeros(1, S - S_unpadded, dtype=torch.long)], dim=1) if S > S_unpadded else ids

    # M-RoPE cos/sin (real 3D positions), uploaded replicated — same convention as decode rope_setup.
    cos_ref, sin_ref = build_mrope_cos_sin(
        positions_3d=pos3d[:, 0, :],
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    upload = lambda t: ttnn.from_torch(  # noqa: E731
        t.unsqueeze(0),
        device=bh_glx_mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    rot_mats = (upload(cos_ref), upload(sin_ref))

    logger.info(f"[mm-perf] S_unpadded={S_unpadded} -> bucket S={S}; vision+preproc(warm)={t_vision*1e3:.1f} ms")

    # --- PREFILL via the server path (cold = incl. compile) ---
    ttnn.synchronize_device(bh_glx_mesh)
    _t = time.perf_counter()
    prefill_logits = generator.prefill_forward_text_embeds(
        ids_padded,
        inputs_embeds=fused,
        rot_mats=rot_mats,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[S_unpadded],
    )
    ttnn.synchronize_device(bh_glx_mesh)
    t_prefill_cold = time.perf_counter() - _t
    signpost("prefill_done")  # vision_done->prefill_done = prefill stage (single-pass COLD)
    first_tok = int(torch.as_tensor(prefill_logits).float().reshape(-1)[: args.vocab_size].argmax().item())
    logger.info(
        f"[mm-perf] first token = {first_tok} ({tok.decode([first_tok])!r}); prefill(cold)={t_prefill_cold*1e3:.1f} ms"
    )

    # --- DECODE via the server traced path ---
    # Decouple decode RoPE position from the KV index: rope_pos = cur_pos + offset,
    # offset = (max(pos_3d)+1) - S_unpadded (<= 0). Makes generator decode COHERENT.
    rope_pos_next = int(inputs.position_ids_3d[:, :, :S_unpadded].max().item()) + 1
    decode_rope_offset = rope_pos_next - S_unpadded
    generator.set_decode_rope_offset(decode_rope_offset)
    logger.info(
        f"[mm-perf] decode rope offset = {decode_rope_offset} (rope_pos0={rope_pos_next}, kv_pos0={S_unpadded})"
    )

    # DIAGNOSTIC (QWEN36_MM_HOST_SAMPLE=1): decode with host-argmax (sampling_params=None)
    # instead of the on-device sampler — isolates sampler-bug vs decode-forward/CCL-bug.
    if os.environ.get("QWEN36_MM_HOST_SAMPLE", "0") == "1":
        gen_ids = [first_tok]
        cur_in = first_tok
        current_pos = torch.tensor([S_unpadded], dtype=torch.long)
        for it in range(_DECODE_STEPS):
            out = generator.decode_forward(
                torch.tensor([cur_in], dtype=torch.long).reshape(1, 1),
                current_pos,
                enable_trace=False,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                read_from_device=True,
                sampling_params=None,
                reset_inputs=True,
            )
            _logits = out[0] if isinstance(out, (tuple, list)) else out
            cur_in = int(torch.as_tensor(_logits).float().reshape(-1)[: args.vocab_size].argmax().item())
            gen_ids.append(cur_in)
            current_pos = current_pos + 1
        logger.info(f"[mm-perf][HOST-ARGMAX] OUTPUT: {tok.decode(gen_ids, skip_special_tokens=True)!r}")
        return

    # Model-config sampling params (tt-inference-server llm.yaml override_generation_config):
    # temperature=0.5, top_k=50, top_p=0.95. These are the params the qwen3.6 on-device
    # sampler is validated against (text decode is coherent with them).
    sampling_params = SamplingParams(
        temperature=float(os.environ.get("QWEN36_TEMP", "0.5")),
        top_k=int(os.environ.get("QWEN36_TOP_K", "50")),
        top_p=float(os.environ.get("QWEN36_TOP_P", "0.95")),
    )
    out_tok = torch.tensor([first_tok], dtype=torch.long).reshape(1, 1)
    current_pos = torch.tensor([S_unpadded], dtype=torch.long)
    gen_ids = [first_tok]
    read_events, tt_out_toks = [], []
    _loop_t0 = None
    # Interleaved one-deep readback (mirrors text_demo_qwen36's fast path): issue step N,
    # then read step N-1. Batch-reading all tt_out_toks after the loop reads the reused
    # persistent buffer at the wrong cadence (caused token doubling).
    for it in range(_DECODE_STEPS):
        if it == 1:
            ttnn.synchronize_device(bh_glx_mesh)
            _loop_t0 = time.perf_counter()
        tt_tok, read_event = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=True,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            read_from_device=True,
            async_read=True,
            sampling_params=sampling_params,
            reset_inputs=(it == 0),
        )
        read_events.append(read_event)
        tt_out_toks.append(tt_tok)
        current_pos = current_pos + 1
        if it > 0:
            ttnn.event_synchronize(read_events.pop(0)[0])
            _tt_tok, _ = generator.process_decode_output_host(tt_out_toks.pop(0))
            gen_ids.append(int(torch.as_tensor(_tt_tok).reshape(-1)[0].item()))
    ttnn.synchronize_device(bh_glx_mesh)
    t_decode_steady = time.perf_counter() - _loop_t0
    n_steady = _DECODE_STEPS - 1
    decode_tok_s = n_steady / t_decode_steady
    # Drain the final in-flight step.
    ttnn.event_synchronize(read_events.pop(0)[0])
    _tt_tok, _ = generator.process_decode_output_host(tt_out_toks.pop(0))
    gen_ids.append(int(torch.as_tensor(_tt_tok).reshape(-1)[0].item()))
    signpost("stop")  # prefill_done->stop = decode stage (traced, on-device sampling)
    decoded_text = tok.decode(gen_ids, skip_special_tokens=True)
    logger.info(f"[mm-perf] generator-decode OUTPUT: {decoded_text!r}")

    # --- Warm prefill (re-run AFTER decode, OUTSIDE the signpost window; discarded) ---
    ttnn.synchronize_device(bh_glx_mesh)
    _t = time.perf_counter()
    generator.prefill_forward_text_embeds(
        ids_padded,
        inputs_embeds=fused,
        rot_mats=rot_mats,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[S_unpadded],
    )
    ttnn.synchronize_device(bh_glx_mesh)
    t_prefill_warm = time.perf_counter() - _t

    logger.info("=" * 72)
    logger.info(f"[mm-perf] QWEN3.6-27B VL SERVER-PATH PERF (BH_GLX, batch=1, {modality})")
    logger.info(f"  prompt tokens (incl. vision)   : {S_unpadded}  (prefill bucket {S})")
    logger.info(f"  vision encoder + preproc (warm): {t_vision*1e3:9.1f} ms")
    logger.info(f"  prefill TTFT (cold, w/ compile): {t_prefill_cold*1e3:9.1f} ms")
    logger.info(f"  prefill TTFT (warm)            : {t_prefill_warm*1e3:9.1f} ms")
    logger.info(
        f"  decode (traced, on-dev sample) : {1e3*t_decode_steady/n_steady:9.2f} ms/tok = {decode_tok_s:6.2f} tok/s"
    )
    logger.info("=" * 72)
    assert decode_tok_s > 0
