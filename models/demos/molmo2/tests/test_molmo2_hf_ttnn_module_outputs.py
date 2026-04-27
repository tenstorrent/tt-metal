# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
``Molmo2-8B``: HuggingFace vs TTNN **module output** agreement (PCC), aligned with
``models/demos/molmo2/ARCHITECTURE.md``.

**Text stack** (from ``ForConditionalGeneration.model``)
    ``wte`` → 36× ``transformer.blocks.*`` → ``ln_f`` → (separately) ``lm_head`` logits

    - HF: ``output_hidden_states=True`` yields 37 tensors: embed, one per block exit (through
      block 35), then post-``ln_f`` last hidden state.
    - TT: ``TextModel.forward_collect_hidden_states_torch`` yields 38 tensors: embed, one after
      each of 36 blocks, then after ``ln_f`` (compare HF[36] to TT[37]).
    - Logits: last-token slice of full forward vs ``hf_for_cond`` logits (same as
      ``test_hf_ttnn_text_module_pcc``).

    **Video + text** (set ``MOLMO2_PCC_VIDEO_PATH``): ``preprocess_video`` with your prompt; PCC for
    vision (HF ``vision_backbone`` vs TT ``embed_image``), then the same 36 text boundaries + ``ln_f``
    with HF ``output_hidden_states`` vs TT hiddens from **fused** embeds, then last-position logits
    (HF top-level forward vs full TT ``Molmo2Model.forward``; second TT vision pass for logits only).

**Vision stack** (concatenated backbone: ViT multi-scale + ``image_pooling_2d`` + ``image_projector``)
    - HF: ``model.model.vision_backbone(pixel_values, pooled_patches_idx)``.
    - TT: ``Molmo2Model.embed_image`` (single-image / small-``B`` path), tensor compared after
      mesh de-replication.

Finer steps (per-ViT-layer, pool-only, projector-only) are covered in
``test_vision_transformer.py`` and ``test_vision_pooling_pcc.py``.

Usage::

    pytest models/demos/molmo2/tests/test_molmo2_hf_ttnn_module_outputs.py -v -s

Env::

    HF_MODEL=allenai/Molmo2-8B
    MOLMO2_PCC_TEXT_PROMPT=...
    MOLMO2_PCC_VIDEO_PATH=...          # e.g. path to .mp4 — enables ``test_multimodal_video_modules_hf_vs_ttnn_pcc``
    MOLMO2_PCC_VIDEO_PROMPT=...      # e.g. ``<|video|>`` + question (default: demo-style)
    MOLMO2_PCC_VIDEO_NUM_FRAMES=...  # optional, passed to ``preprocess_video`` as ``num_frames``
    MOLMO2_PCC_VIDEO_MAX_FPS=...     # optional, passed to ``preprocess_video`` as ``max_fps``
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import List, Optional, Tuple

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.molmo2.tt.hf_processor import preprocess_video
from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors
from models.demos.molmo2.tt.molmo2_model import Molmo2Model
from models.demos.molmo2.tt.prefill_attention_mask import build_molmo2_prefill_attention_bias_ttnn
from models.demos.molmo2.tt.text_model import TextModel


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().contiguous()
    b = b.float().contiguous()
    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {a.shape} vs {b.shape}")
    _ok, pcc_val = comp_pcc(a, b, pcc=0.0)
    return float(pcc_val)


def _mesh_1x8():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(ttnn.MeshShape(1, 8))


@pytest.fixture(scope="module")
def mesh_device():
    d = _mesh_1x8()
    yield d
    ttnn.close_mesh_device(d)


@pytest.fixture(scope="module")
def hf_molmo2():
    from transformers import AutoModelForImageTextToText

    model_id = os.environ.get("HF_MODEL", "allenai/Molmo2-8B")
    m = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def hf_processor_tokenizer():
    from transformers import AutoProcessor, AutoTokenizer

    model_id = os.environ.get("HF_MODEL", "allenai/Molmo2-8B")
    return (
        AutoProcessor.from_pretrained(model_id, trust_remote_code=True),
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True),
    )


def _text_module_names_for_first_hidden_state_rows() -> List[str]:
    """
    One name per (HF[i], TT[i]) for i in 0..35.

    HF[0] = embed, HF[1..35] = after blocks 0..33 is wrong — per ``test_hf_ttnn_text_module_pcc``,
    index ``i`` uses name ``after block {i-1}`` for i>0, so HF[35] is after block 34.
    """
    names = ["model.transformer.wte (inputs_embed)"]
    # HF hidden_states[1]..[35] align with TT[1]..[35] (after blocks 0..34); see ``test_hf_ttnn_text_module_pcc``.
    names += [f"model.transformer.blocks.{i}" for i in range(35)]
    return names


def test_text_modules_hf_vs_ttnn_pcc(mesh_device, hf_molmo2, hf_processor_tokenizer):
    """
    For each text module boundary in ARCHITECTURE.md, report PCC vs ``TextModel`` (TT).
    """
    _processor, tokenizer = hf_processor_tokenizer
    prompt = os.environ.get("MOLMO2_PCC_TEXT_PROMPT", "What is the capital of France?")
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    with torch.no_grad():
        o = hf_molmo2.model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    hf_h = o.hidden_states
    assert len(hf_h) == 37, f"expected 37 HF hidden states, got {len(hf_h)}"

    state_dict = load_state_dict_from_safetensors(os.environ.get("HF_MODEL", "allenai/Molmo2-8B"))
    ttnn_text = TextModel(
        mesh_device=mesh_device,
        state_dict=state_dict,
        dtype=ttnn.bfloat8_b,
    )
    tt_h = ttnn_text.forward_collect_hidden_states_torch(input_ids)
    assert len(tt_h) == 38, f"expected 38 TT hiddens, got {len(tt_h)}"

    mod_names_36 = _text_module_names_for_first_hidden_state_rows()
    assert len(mod_names_36) == 36
    rows: List[Tuple[str, float]] = []

    for i in range(36):
        p = _pcc(hf_h[i].float(), tt_h[i].float())
        rows.append((mod_names_36[i], p))
        logger.info(f"  {mod_names_36[i]}: PCC={p:.5f}")
        if p < 0.90:
            logger.warning(f"  LOW PCC < 0.90: {mod_names_36[i]} pcc={p:.5f}")

    ln_f_name = "model.transformer.ln_f (out, last_hidden_state vs HF[36] / TT[37])"
    p = _pcc(hf_h[36].float(), tt_h[37].float())
    rows.append((ln_f_name, p))
    logger.info(f"  {ln_f_name}: PCC={p:.5f}")
    if p < 0.90:
        logger.warning(f"  LOW PCC < 0.90: {ln_f_name} pcc={p:.5f}")

    is_mesh = mesh_device.__class__.__name__ == "MeshDevice"
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    input_ids_ttnn = ttnn.from_torch(
        input_ids,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    h = ttnn_text.embed_tokens(input_ids_ttnn)
    ttnn.deallocate(input_ids_ttnn)
    logits, _ = ttnn_text.forward(h)
    if is_mesh:
        t_logits = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    else:
        t_logits = ttnn.to_torch(logits)
    t_logits = t_logits.float()
    if t_logits.dim() == 4 and t_logits.shape[0] == 1 and t_logits.shape[1] == 1:
        t_logits = t_logits.squeeze(0)

    with torch.no_grad():
        hf_logits = hf_molmo2(input_ids).logits.float()
    lm_name = "lm_head (logits, last pos)"
    p_log = _pcc(hf_logits[0, -1, :], t_logits[0, -1, :])
    rows.append((lm_name, p_log))
    logger.info(f"  {lm_name}: PCC={p_log:.5f}")
    if p_log < 0.85:
        logger.warning(f"  LOW logits PCC: {p_log:.5f}")

    # Sanity: embedding quality (module-level gate)
    assert rows[0][1] > 0.98, f"wte/inputs_embed PCC should be > 0.98, got {rows[0][1]}"


def _load_test_image(processor, torch_dtype: torch.dtype):
    import requests
    from PIL import Image

    try:
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/800px-Camponotus_flavomarginatus_ant.jpg"
        r = requests.get(url, timeout=10)
        image = Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not download test image: {e}; using blank 378x378")
        image = Image.open("models/demos/molmo2/demo/dog.jpg")
    out = processor(images=[image], text="Describe this image.", return_tensors="pt")
    images = out.pixel_values if hasattr(out, "pixel_values") and out.pixel_values is not None else out.images
    pooled = out.pooled_patches_idx
    if images is None or pooled is None:
        pytest.skip("Processor did not return pixel_values / pooled_patches_idx")
    return images.to(dtype=torch_dtype), pooled


def _tt_visual_to_torch(visual, mesh_device) -> torch.Tensor:
    is_mesh = mesh_device.__class__.__name__ == "MeshDevice"
    if is_mesh:
        t = ttnn.to_torch(visual, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        if t.shape[0] == mesh_device.get_num_devices():
            t = t[0]
    else:
        t = ttnn.to_torch(visual)
    # Common TT layout [1, 1, N, H] -> [1, N, H]
    if t.dim() == 4 and t.shape[0] == 1 and t.shape[1] == 1:
        t = t.squeeze(1)
    return t.float()


def test_vision_backbone_end_to_end_hf_vs_ttnn_pcc(mesh_device, hf_molmo2, hf_processor_tokenizer):
    """
    **Vision backbone** as one block in ARCHITECTURE (ViT + pool + projector): compare HF
    module output to ``Molmo2Model.embed_image`` (TT).
    """
    processor, _ = hf_processor_tokenizer
    pixel_values, pooled_patches_idx = _load_test_image(processor, torch_dtype=next(hf_molmo2.parameters()).dtype)
    with torch.no_grad():
        hf_out = hf_molmo2.model.vision_backbone(pixel_values, pooled_patches_idx).float()
    assert hf_out.dim() in (2, 3), f"unexpected HF vision shape: {hf_out.shape}"

    state_dict = load_state_dict_from_safetensors(os.environ.get("HF_MODEL", "allenai/Molmo2-8B"))
    molmo2 = Molmo2Model(
        mesh_device=mesh_device,
        state_dict=state_dict,
        dtype=ttnn.bfloat8_b,
        use_async_ccl=False,
    )
    vis_ttnn, _valid = molmo2.embed_image(pixel_values, pooled_patches_idx, use_data_parallel=False)
    tt_out = _tt_visual_to_torch(vis_ttnn, mesh_device)
    ttnn.deallocate(vis_ttnn)

    # Align leading dims for PCC (HF may be [N, H] with N = valid rows)
    h_dim = int(hf_out.shape[-1])
    if hf_out.dim() == 2:
        hf_2d = hf_out
    else:
        hf_2d = hf_out.reshape(-1, h_dim)
    if tt_out.dim() == 3:
        tt_2d = tt_out.reshape(-1, h_dim)
    else:
        tt_2d = tt_out
    n = min(hf_2d.shape[0], tt_2d.shape[0])
    if hf_2d.shape[0] != tt_2d.shape[0]:
        logger.warning(
            f"vision_backbone row count differs: HF {hf_2d.shape[0]} vs TT {tt_2d.shape[0]}; comparing first {n} rows"
        )
    p = _pcc(hf_2d[:n], tt_2d[:n])
    logger.info(
        f"  model.vision_backbone (full stack per ARCHITECTURE.md): PCC={p:.5f} "
        f"(HF {hf_2d.shape} vs TT {tt_2d.shape}, compared rows={n})"
    )
    if p < 0.85:
        logger.warning(f"LOW full vision-backbone PCC: {p:.5f}")


def _env_int(name: str) -> Optional[int]:
    v = os.environ.get(name, "").strip()
    if not v:
        return None
    return int(v)


def _hf_vision_backbone_video(
    inner,
    input_ids: torch.Tensor,
    pixel_values_videos: torch.Tensor,
    video_token_pooling: torch.Tensor,
    video_grids: torch.Tensor,
) -> torch.Tensor:
    """
    HuggingFace video path: ``merge_visual_inputs`` (``pixel_values_videos=``) then
    ``vision_backbone`` on the batched patch tensor.
    """
    dev = next(inner.parameters()).device
    dtype = next(inner.parameters()).dtype
    input_ids = input_ids.to(dev)
    pixel_values_videos = pixel_values_videos.to(dev, dtype=dtype)
    video_token_pooling = video_token_pooling.to(dev, dtype=torch.long)
    video_grids = video_grids.to(dev)
    with torch.no_grad():
        images, token_pooling = inner.merge_visual_inputs(
            input_ids=input_ids,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
        )
    assert images is not None and token_pooling is not None
    with torch.no_grad():
        return inner.vision_backbone(images, token_pooling)


def _hf_inner_forward_video_hidden_states(
    inner,
    *,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    token_type_ids: Optional[torch.Tensor],
    pixel_values_videos: torch.Tensor,
    video_token_pooling: torch.Tensor,
    video_grids: torch.Tensor,
) -> object:
    """
    ``ForConditionalGeneration.model`` forward with **video** kwargs
    (``pixel_values_videos``, ``video_token_pooling``, ``video_grids``), not the image path.
    """
    dev = next(inner.parameters()).device
    dtype = next(inner.parameters()).dtype
    input_ids = input_ids.to(dev)
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(dev)
    pixel_values_videos = pixel_values_videos.to(dev, dtype=dtype)
    video_token_pooling = video_token_pooling.to(dev, dtype=torch.long)
    video_grids = video_grids.to(dev)

    base = dict(
        input_ids=input_ids,
        pixel_values_videos=pixel_values_videos,
        video_token_pooling=video_token_pooling,
        video_grids=video_grids,
        output_hidden_states=True,
        use_cache=False,
    )
    if attention_mask is not None:
        base["attention_mask"] = attention_mask
    if token_type_ids is not None:
        base["token_type_ids"] = token_type_ids
    return inner(**base)


def _hf_for_cond_video_logits(
    for_cond,
    *,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    token_type_ids: Optional[torch.Tensor],
    pixel_values_videos: torch.Tensor,
    video_token_pooling: torch.Tensor,
    video_grids: torch.Tensor,
) -> torch.Tensor:
    """Top-level ``ForConditionalGeneration`` forward; returns logits with **video** kwargs."""
    dev = next(for_cond.parameters()).device
    dtype = next(for_cond.parameters()).dtype
    input_ids = input_ids.to(dev)
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(dev)
    pixel_values_videos = pixel_values_videos.to(dev, dtype=dtype)
    video_token_pooling = video_token_pooling.to(dev, dtype=torch.long)
    video_grids = video_grids.to(dev)

    base = dict(
        input_ids=input_ids,
        pixel_values_videos=pixel_values_videos,
        video_token_pooling=video_token_pooling,
        video_grids=video_grids,
    )
    if attention_mask is not None:
        base["attention_mask"] = attention_mask
    if token_type_ids is not None:
        base["token_type_ids"] = token_type_ids
    return for_cond(**base).logits


def test_multimodal_video_modules_hf_vs_ttnn_pcc(mesh_device, hf_molmo2):
    """
    When ``MOLMO2_PCC_VIDEO_PATH`` is set, preprocess a video with ``MOLMO2_PCC_VIDEO_PROMPT`` and
    report PCC for:

    1) **Vision backbone** — HF ``model.vision_backbone`` vs TT ``Molmo2Model.embed_image`` (as in
       ``test_vision_backbone_end_to_end``), and
    2) **Text stack** — HF ``output_hidden_states`` vs
       ``TextModel.forward_collect_hidden_states_from_inputs_embeds_torch`` on **fused** embeds, and
    3) **lm_head** — last-position logits: HF top-level forward vs full ``Molmo2Model.forward``.

    Skips if ``MOLMO2_PCC_VIDEO_PATH`` is unset or the file is missing. Uses two TT vision+prepare
    passes: one to collect per-layer hidden states, one full ``forward`` for logits.
    """
    video_path = os.environ.get("MOLMO2_PCC_VIDEO_PATH", "").strip()
    if not video_path or not os.path.isfile(video_path):
        pytest.skip("Set MOLMO2_PCC_VIDEO_PATH to a readable video file to run this test " f"(current: {video_path!r})")

    default_prompt = """<|video|> Why Lily's art teacher was mad when he
    saw the paint?
      A. Because the paint is blurry
      B. Because Lily's paint has too much red.
      C. Because Lily's paint is too dark.
      D. Because paint spilled on the painting
      E. Because Lily's paint was bad.
      Please respond with only the letter of the correct answer.."""
    prompt = os.environ.get("MOLMO2_PCC_VIDEO_PROMPT", default_prompt).strip() or default_prompt
    n_frames = _env_int("MOLMO2_PCC_VIDEO_NUM_FRAMES")
    max_fps_env = os.environ.get("MOLMO2_PCC_VIDEO_MAX_FPS", "").strip()
    max_fps: Optional[float] = float(max_fps_env) if max_fps_env else None

    pvw: dict = {"apply_template": True}
    if n_frames is not None:
        pvw["num_frames"] = n_frames
    if max_fps is not None:
        pvw["max_fps"] = max_fps
    try:
        vi = preprocess_video(video_path, prompt, **pvw)
    except TypeError:
        if n_frames is not None and max_fps is not None:
            try:
                vi = preprocess_video(video_path, prompt, num_frames=n_frames, max_fps=max_fps)
            except TypeError:
                if n_frames is not None:
                    try:
                        vi = preprocess_video(video_path, prompt, num_frames=n_frames)
                    except TypeError:
                        vi = preprocess_video(video_path, prompt)
                else:
                    vi = preprocess_video(video_path, prompt)
        elif n_frames is not None:
            try:
                vi = preprocess_video(video_path, prompt, num_frames=n_frames)
            except TypeError:
                vi = preprocess_video(video_path, prompt)
        else:
            vi = preprocess_video(video_path, prompt)

    n_frames = int(vi["n_frames"])
    k_pool = int(vi["k_pool"])
    n_tokens = int(vi["n_tokens"])
    n_out = n_tokens // n_frames
    itp2 = vi["image_token_pooling"]
    pooled_3d = itp2.reshape(n_frames, n_out, k_pool)
    dtype = next(hf_molmo2.parameters()).dtype

    input_ids = vi["input_ids"]
    attention_mask = vi.get("attention_mask")
    token_type_ids = vi.get("token_type_ids")
    # Image layout [F,3,H,W] for TT ``embed_image``; patch layout [F,729,588] for HF ``build_batched_videos``.
    pixel_values = vi["pixel_values"].to(dtype=dtype)
    pvv = vi["pixel_values_videos"].to(dtype=dtype)
    video_grids = vi["video_grids"].to(dtype=torch.long)

    inner = hf_molmo2.model
    with torch.no_grad():
        o = _hf_inner_forward_video_hidden_states(
            inner,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values_videos=pvv,
            video_token_pooling=itp2,
            video_grids=video_grids,
        )
    assert o.hidden_states is not None, "HF inner forward must return hidden_states"
    hf_h = o.hidden_states
    assert len(hf_h) == 37, f"expected 37 HF hidden states, got {len(hf_h)}"

    with torch.no_grad():
        hf_vision = _hf_vision_backbone_video(inner, input_ids, pvv, itp2, video_grids).float()
    h_dim = int(hf_vision.shape[-1])
    if hf_vision.dim() == 2:
        hf_vision_2d = hf_vision
    else:
        hf_vision_2d = hf_vision.reshape(-1, h_dim)

    state_dict = load_state_dict_from_safetensors(os.environ.get("HF_MODEL", "allenai/Molmo2-8B"))
    molmo2 = Molmo2Model(
        mesh_device=mesh_device,
        state_dict=state_dict,
        dtype=ttnn.bfloat8_b,
        use_async_ccl=False,
    )
    is_mesh = mesh_device.__class__.__name__ == "MeshDevice"
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None

    # One embed pass: same outputs for vision PCC and for multimodal text collection
    vis_ttnn, valid_tok = molmo2.embed_image(pixel_values, pooled_3d)
    tt_vision = _tt_visual_to_torch(vis_ttnn, mesh_device)
    h_dim_tt = int(tt_vision.shape[-1])
    if h_dim_tt != h_dim:
        logger.warning(f"vision hidden size HF {h_dim} vs TT {h_dim_tt}; truncating to min for PCC")
    if tt_vision.dim() == 3:
        tt_vision_2d = tt_vision.reshape(-1, h_dim_tt)
    else:
        tt_vision_2d = tt_vision
    nv = min(hf_vision_2d.shape[0], tt_vision_2d.shape[0])
    h_common = min(hf_vision_2d.shape[1], tt_vision_2d.shape[1])
    if hf_vision_2d.shape[0] != tt_vision_2d.shape[0]:
        logger.warning(
            f"Video vision row count: HF {hf_vision_2d.shape[0]} vs TT {tt_vision_2d.shape[0]}; comparing {nv} rows"
        )
    p_vis = _pcc(hf_vision_2d[:nv, :h_common], tt_vision_2d[:nv, :h_common])
    logger.info(f"  [video] model.vision_backbone vs TT embed_image: PCC={p_vis:.5f}")

    input_ids_cpu = input_ids.cpu() if input_ids.device.type != "cpu" else input_ids
    fused_ttnn = molmo2.prepare_inputs_for_multimodal(input_ids_cpu, vis_ttnn, valid_tok)
    ttnn.deallocate(vis_ttnn)

    seq_len = int(fused_ttnn.shape[-2])
    attn_mask_ttnn = None
    if token_type_ids is not None and seq_len > 1:
        ttid = token_type_ids.cpu() if token_type_ids.device.type != "cpu" else token_type_ids
        am = None
        if attention_mask is not None:
            am = attention_mask.cpu() if attention_mask.device.type != "cpu" else attention_mask
        attn_mask_ttnn = build_molmo2_prefill_attention_bias_ttnn(
            ttid,
            mesh_device,
            mesh_mapper,
            am,
        )

    tt_h = molmo2.text_model.forward_collect_hidden_states_from_inputs_embeds_torch(
        fused_ttnn,
        attn_mask=attn_mask_ttnn,
    )
    if attn_mask_ttnn is not None:
        ttnn.deallocate(attn_mask_ttnn)
    assert len(tt_h) == 38, f"expected 38 TT hiddens, got {len(tt_h)}"

    mod_names_36 = list(_text_module_names_for_first_hidden_state_rows())
    mod_names_36[0] = "fused text+vision inputs (HF: model[0] inputs_embeds, TT: after prepare)"
    for i in range(36):
        p = _pcc(hf_h[i].float(), tt_h[i].float())
        logger.info(f"  [video] {mod_names_36[i]}: PCC={p:.5f}")
        if p < 0.90:
            logger.warning(f"  LOW PCC < 0.90: {mod_names_36[i]} pcc={p:.5f}")
    ln_f_name = "model.transformer.ln_f (out, last_hidden_state vs HF[36] / TT[37])"
    p = _pcc(hf_h[36].float(), tt_h[37].float())
    logger.info(f"  [video] {ln_f_name}: PCC={p:.5f}")
    if p < 0.90:
        logger.warning(f"  LOW PCC < 0.90: {ln_f_name} pcc={p:.5f}")

    with torch.no_grad():
        hf_logits = _hf_for_cond_video_logits(
            hf_molmo2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values_videos=pvv,
            video_token_pooling=itp2,
            video_grids=video_grids,
        )
    hf_last = hf_logits[0, -1, :].float()

    am_cpu = attention_mask.cpu() if attention_mask is not None else None
    tid_cpu = token_type_ids.cpu() if token_type_ids is not None else None
    logits_ttnn, _ = molmo2.forward(
        input_ids_cpu,
        pixel_values=pixel_values,
        pooled_patches_idx=pooled_3d,
        attention_mask=am_cpu,
        token_type_ids=tid_cpu,
    )
    if is_mesh:
        t_logits = ttnn.to_torch(logits_ttnn, mesh_composer=mesh_composer)[0]
    else:
        t_logits = ttnn.to_torch(logits_ttnn)
    ttnn.deallocate(logits_ttnn)
    t_logits = t_logits.float()
    if t_logits.dim() == 4 and t_logits.shape[0] == 1 and t_logits.shape[1] == 1:
        t_logits = t_logits.squeeze(0)
    tt_last = t_logits[0, -1, :]
    p_log = _pcc(hf_last, tt_last)
    logger.info(f"  [video] lm_head (logits, last pos): PCC={p_log:.5f}")
    if p_log < 0.85:
        logger.warning(f"  LOW logits PCC: {p_log:.5f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
