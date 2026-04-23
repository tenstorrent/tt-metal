# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.demos.dots_ocr.reference.hf_utils import HFLoadSpec
from models.demos.dots_ocr.reference.model import DotsOCRReference
from models.demos.dots_ocr.reference.pcc import comp_pcc


@pytest.mark.skipif(torch.cuda.is_available(), reason="This PCC test is intended for CPU+TTNN workflows.")
def test_dots_vision_tower_ttnn_pcc(tmp_path):
    """
    Compare HF ``vision_tower`` output vs the full TTNN vision stack (``DropInVisionTransformer``:
    TTNN QKV/MLP, torch RoPE+SDPA, TTNN patch merger). PCC is stricter than the old hybrid
    path; relax the bar if bring-up requires it.

    Requires TTNN device runtime (T3K recommended).
    """
    try:
        import ttnn
    except Exception:
        pytest.skip("TTNN not available")

    from models.demos.dots_ocr.tt.mesh import close_dots_mesh_device, open_mesh_device
    from models.demos.dots_ocr.tt.model import DropInVisionTransformer
    from models.demos.dots_ocr.tt.vision_model_config import DotsVisionModelArgs

    # TTNN vision uses bfloat16; use the same dtype for the HF oracle so attention/MLP PCC
    # is not dominated by f32 vs bf16 numerical gaps (Block0 ``attn`` is especially sensitive).
    ref = DotsOCRReference(HFLoadSpec(model_id="rednote-hilab/dots.mocr", dtype=torch.bfloat16))
    # Use the repo's demo image path if present; otherwise skip.
    from PIL import Image

    img_path = "models/demos/dots_ocr/demo/test12.png"
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception:
        pytest.skip(f"Missing demo image {img_path}")

    inputs = ref.preprocess_image_and_prompt(image, "OCR: transcribe the text in the image exactly.")
    if inputs.pixel_values is None or inputs.image_grid_thw is None:
        pytest.skip("Processor did not produce pixel_values/image_grid_thw")

    try:
        mesh = open_mesh_device()
    except Exception as e:
        pytest.skip(f"TT device unavailable ({type(e).__name__}): {e}")

    try:
        vision_args = DotsVisionModelArgs(
            mesh_device=mesh, hf_config=ref.model.config, max_batch_size=1, max_seq_len=2048
        )
        visual = DropInVisionTransformer(ref.model, vision_args, debug=False)

        # Optional intermediate: patch_embed PCC (helps localize mismatches).
        # This does not change the TT implementation; it only provides a higher-signal failure mode.
        try:
            hf_vt = getattr(ref.model, "vision_tower", None)
            hf_pe_mod = getattr(hf_vt, "patch_embed", None) if hf_vt is not None else None
            if callable(hf_pe_mod) and hasattr(getattr(visual, "tt_model", None), "patch_embed"):
                with torch.no_grad():
                    hf_pe = hf_pe_mod(inputs.pixel_values, inputs.image_grid_thw)
                    if isinstance(hf_pe, (tuple, list)):
                        hf_pe = hf_pe[0]
                    # Some HF remote-code implementations may return a TTNN tensor in mixed environments.
                    if not isinstance(hf_pe, torch.Tensor):
                        try:
                            hf_pe = ttnn.to_torch(hf_pe)
                        except Exception:
                            pass
                    # Normalize HF patch embed to [S, D]
                    if isinstance(hf_pe, torch.Tensor):
                        if hf_pe.dim() == 4:
                            hf_pe = hf_pe.squeeze(0).squeeze(0)
                        elif hf_pe.dim() == 3:
                            hf_pe = hf_pe.reshape(-1, hf_pe.shape[-1])

                    tt_pe = visual.tt_model.patch_embed(inputs.pixel_values, inputs.image_grid_thw)
                    # TT patch_embed should return ttnn.Tensor [1,1,S,D], but fall back safely if it returns torch.
                    if isinstance(tt_pe, torch.Tensor):
                        tt_pe_t = tt_pe.to(torch.float32)
                    else:
                        composer = ttnn.ConcatMeshToTensor(mesh, dim=0)
                        tt_pe_t = ttnn.to_torch(tt_pe, mesh_composer=composer).to(torch.float32)
                    if tt_pe_t.dim() == 4:
                        tt_pe_t = tt_pe_t.squeeze(0).squeeze(0)
                    elif tt_pe_t.dim() == 3:
                        tt_pe_t = tt_pe_t.reshape(-1, tt_pe_t.shape[-1])

                    n0 = min(int(hf_pe.shape[0]), int(tt_pe_t.shape[0]))
                    d0 = min(int(hf_pe.shape[1]), int(tt_pe_t.shape[1]))
                    pe_pcc = comp_pcc(hf_pe[:n0, :d0].float(), tt_pe_t[:n0, :d0].float())
                    print(f"PatchEmbed PCC: {pe_pcc:.6f} (hf={tuple(hf_pe.shape)} tt={tuple(tt_pe_t.shape)})")
                    # Try spatial-merge permutation on TT patch tokens to see if order mismatch is the cause.
                    try:
                        g = inputs.image_grid_thw
                        t = int(g[0, 0].item())
                        h = int(g[0, 1].item())
                        w = int(g[0, 2].item())
                        m = 2
                        if t == 1 and h % m == 0 and w % m == 0 and (h * w) == tt_pe_t.shape[0]:
                            tt_perm = tt_pe_t.reshape(h, w, -1)
                            tt_perm = (
                                tt_perm.reshape(h // m, m, w // m, m, -1).permute(0, 2, 1, 3, 4).reshape(h * w, -1)
                            )
                            pe_pcc_perm = comp_pcc(hf_pe[:n0, :d0].float(), tt_perm[:n0, :d0].float())
                            print(f"PatchEmbed PCC (permute m=2): {pe_pcc_perm:.6f}")
                    except Exception:
                        pass
        except Exception as exc:
            print(f"PatchEmbed PCC skipped: {type(exc).__name__}: {exc}")

        # Layer-0 PCC: compare HF first vision block output vs TT first block output.
        try:
            hf_vt = getattr(ref.model, "vision_tower", None)
            if hf_vt is not None and hasattr(getattr(visual, "tt_model", None), "build_rot_mats_and_cu"):
                with torch.no_grad():
                    # HF patch embed output [S,D]
                    hf_pe = hf_vt.patch_embed(inputs.pixel_values, inputs.image_grid_thw)
                    if isinstance(hf_pe, (tuple, list)):
                        hf_pe = hf_pe[0]
                    if hf_pe.dim() == 4:
                        hf_pe = hf_pe.squeeze(0).squeeze(0)
                    elif hf_pe.dim() == 3:
                        hf_pe = hf_pe.reshape(-1, hf_pe.shape[-1])

                    # HF rotary + cu_seqlens
                    rotary = hf_vt.rot_pos_emb(inputs.image_grid_thw)
                    counts = torch.repeat_interleave(
                        inputs.image_grid_thw[:, 1] * inputs.image_grid_thw[:, 2], inputs.image_grid_thw[:, 0]
                    )
                    cu = counts.cumsum(dim=0, dtype=torch.int32)
                    cu = torch.nn.functional.pad(cu, (1, 0), value=0)

                    hf_b0 = hf_vt.blocks[0](hf_pe, cu_seqlens=cu, rotary_pos_emb=rotary)

                    # TT patch embed output [1,1,S,D] ttnn tensor
                    tt_x = visual.tt_model.patch_embed(inputs.pixel_values, inputs.image_grid_thw)
                    if isinstance(tt_x, torch.Tensor):
                        tt_x = ttnn.from_torch(tt_x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                        # Normalize to [1,1,S,D] for TT blocks
                        if len(tt_x.shape) == 3:
                            tt_x = ttnn.reshape(tt_x, (1, 1, tt_x.shape[1], tt_x.shape[2]))

                    rot_mats, cu_t = visual.tt_model.build_rot_mats_and_cu(
                        grid_thw=inputs.image_grid_thw, seq_len=int(tt_x.shape[2])
                    )

                    # Optional: validate RoPE tables (cos/sin) against HF.
                    try:
                        # HF rotary_pos_emb is `freqs` (not cos/sin), shaped [S, head_dim//2].
                        hf_freqs = hf_vt.rot_pos_emb(inputs.image_grid_thw)
                        hf_cos_half = hf_freqs.cos().float()
                        hf_sin_half = hf_freqs.sin().float()
                        # HF apply_rotary_pos_emb_vision repeats last dim twice to full head_dim.
                        hf_cos = hf_cos_half.unsqueeze(1).repeat(1, 1, 2).squeeze(1)  # [S, head_dim]
                        hf_sin = hf_sin_half.unsqueeze(1).repeat(1, 1, 2).squeeze(1)  # [S, head_dim]

                        tt_cos, tt_sin = rot_mats
                        composer = ttnn.ConcatMeshToTensor(mesh, dim=0)
                        tt_cos_t = ttnn.to_torch(tt_cos, mesh_composer=composer).to(torch.float32)
                        tt_sin_t = ttnn.to_torch(tt_sin, mesh_composer=composer).to(torch.float32)
                        if tt_cos_t.dim() == 4:
                            tt_cos_t = tt_cos_t.squeeze(0).squeeze(0)
                            tt_sin_t = tt_sin_t.squeeze(0).squeeze(0)
                        # Compare overlap
                        n = min(int(hf_cos.shape[0]), int(tt_cos_t.shape[0]))
                        d = min(int(hf_cos.shape[1]), int(tt_cos_t.shape[1]))
                        rope_cos_pcc = comp_pcc(hf_cos[:n, :d], tt_cos_t[:n, :d])
                        rope_sin_pcc = comp_pcc(hf_sin[:n, :d], tt_sin_t[:n, :d])
                        print(f"RoPE cos PCC: {rope_cos_pcc:.6f} (hf={tuple(hf_cos.shape)} tt={tuple(tt_cos_t.shape)})")
                        print(f"RoPE sin PCC: {rope_sin_pcc:.6f} (hf={tuple(hf_sin.shape)} tt={tuple(tt_sin_t.shape)})")
                    except Exception as exc_rope:
                        print(f"RoPE cos/sin PCC skipped: {type(exc_rope).__name__}: {exc_rope}")

                    # Optional: block0 component-level PCC to localize drift (norm1/attn/norm2/mlp).
                    try:
                        hf_blk0 = hf_vt.blocks[0]
                        tt_blk0 = visual.tt_model.blocks[0]

                        # HF norm1 output
                        hf_n1 = hf_blk0.norm1(hf_pe)

                        # TT norm1 output (ttnn tensor -> torch)
                        tt_n1 = tt_blk0.norm1(tt_x)
                        composer = ttnn.ConcatMeshToTensor(mesh, dim=0)
                        tt_n1_t = ttnn.to_torch(tt_n1, mesh_composer=composer).to(torch.float32)
                        if tt_n1_t.dim() == 4:
                            tt_n1_t = tt_n1_t.squeeze(0).squeeze(0)
                        n = min(int(hf_n1.shape[0]), int(tt_n1_t.shape[0]))
                        d = min(int(hf_n1.shape[1]), int(tt_n1_t.shape[1]))
                        print(f"Block0 norm1 PCC: {comp_pcc(hf_n1[:n,:d].float(), tt_n1_t[:n,:d].float()):.6f}")

                        # HF attention output (no residual)
                        hf_attn = hf_blk0.attn(hf_n1, cu_seqlens=cu, rotary_pos_emb=rotary)
                        # TT attention output (no residual)
                        tt_attn = tt_blk0.attention(tt_n1, rot_mats=rot_mats, cu_seqlens=cu_t)
                        tt_attn_t = ttnn.to_torch(tt_attn, mesh_composer=composer).to(torch.float32)
                        if tt_attn_t.dim() == 4:
                            tt_attn_t = tt_attn_t.squeeze(0).squeeze(0)
                        n = min(int(hf_attn.shape[0]), int(tt_attn_t.shape[0]))
                        d = min(int(hf_attn.shape[1]), int(tt_attn_t.shape[1]))
                        print(f"Block0 attn PCC: {comp_pcc(hf_attn[:n,:d].float(), tt_attn_t[:n,:d].float()):.6f}")

                        # Residual1
                        hf_r1 = hf_pe + hf_attn
                        tt_r1 = ttnn.add(tt_x, tt_attn)
                        tt_r1_t = ttnn.to_torch(tt_r1, mesh_composer=composer).to(torch.float32).squeeze(0).squeeze(0)
                        n = min(int(hf_r1.shape[0]), int(tt_r1_t.shape[0]))
                        d = min(int(hf_r1.shape[1]), int(tt_r1_t.shape[1]))
                        print(
                            f"Block0 after-attn residual PCC: {comp_pcc(hf_r1[:n,:d].float(), tt_r1_t[:n,:d].float()):.6f}"
                        )

                        # Norm2
                        hf_n2 = hf_blk0.norm2(hf_r1)
                        tt_n2 = tt_blk0.norm2(tt_r1)
                        tt_n2_t = ttnn.to_torch(tt_n2, mesh_composer=composer).to(torch.float32).squeeze(0).squeeze(0)
                        n = min(int(hf_n2.shape[0]), int(tt_n2_t.shape[0]))
                        d = min(int(hf_n2.shape[1]), int(tt_n2_t.shape[1]))
                        print(f"Block0 norm2 PCC: {comp_pcc(hf_n2[:n,:d].float(), tt_n2_t[:n,:d].float()):.6f}")

                        # MLP
                        hf_mlp = hf_blk0.mlp(hf_n2)
                        tt_mlp = tt_blk0.mlp(tt_n2)
                        tt_mlp_t = ttnn.to_torch(tt_mlp, mesh_composer=composer).to(torch.float32).squeeze(0).squeeze(0)
                        n = min(int(hf_mlp.shape[0]), int(tt_mlp_t.shape[0]))
                        d = min(int(hf_mlp.shape[1]), int(tt_mlp_t.shape[1]))
                        print(f"Block0 mlp PCC: {comp_pcc(hf_mlp[:n,:d].float(), tt_mlp_t[:n,:d].float()):.6f}")
                    except Exception as exc_blk:
                        print(f"Block0 component PCC skipped: {type(exc_blk).__name__}: {exc_blk}")

                    tt_b0 = visual.tt_model.blocks[0](tt_x, rot_mats=rot_mats, cu_seqlens=cu_t)

                    composer = ttnn.ConcatMeshToTensor(mesh, dim=0)
                    tt_b0_t = ttnn.to_torch(tt_b0, mesh_composer=composer).to(torch.float32)
                    if tt_b0_t.dim() == 4:
                        tt_b0_t = tt_b0_t.squeeze(0).squeeze(0)
                    elif tt_b0_t.dim() == 3:
                        tt_b0_t = tt_b0_t.reshape(-1, tt_b0_t.shape[-1])

                    n0 = min(int(hf_b0.shape[0]), int(tt_b0_t.shape[0]))
                    d0 = min(int(hf_b0.shape[1]), int(tt_b0_t.shape[1]))
                    b0_pcc = comp_pcc(hf_b0[:n0, :d0].float(), tt_b0_t[:n0, :d0].float())
                    print(f"Vision block0 PCC: {b0_pcc:.6f} (hf={tuple(hf_b0.shape)} tt={tuple(tt_b0_t.shape)})")
        except Exception as exc:
            print(f"Vision block0 PCC skipped: {type(exc).__name__}: {exc}")

        # Trunk vs Merger isolation:
        # - Compare hidden states right before merger (after 42 blocks + post_trunk_norm)
        # - Compare merger output given identical pre-merger hidden states
        try:
            hf_vt = getattr(ref.model, "vision_tower", None)
            if hf_vt is not None and hasattr(getattr(visual, "tt_model", None), "build_rot_mats_and_cu"):
                with torch.no_grad():
                    # ----- HF pre-merger hidden -----
                    hf_h = hf_vt.patch_embed(inputs.pixel_values, inputs.image_grid_thw)
                    if isinstance(hf_h, (tuple, list)):
                        hf_h = hf_h[0]
                    if hf_h.dim() == 4:
                        hf_h = hf_h.squeeze(0).squeeze(0)
                    elif hf_h.dim() == 3:
                        hf_h = hf_h.reshape(-1, hf_h.shape[-1])

                    rotary = hf_vt.rot_pos_emb(inputs.image_grid_thw)
                    counts = torch.repeat_interleave(
                        inputs.image_grid_thw[:, 1] * inputs.image_grid_thw[:, 2], inputs.image_grid_thw[:, 0]
                    )
                    cu = counts.cumsum(dim=0, dtype=torch.int32)
                    cu = torch.nn.functional.pad(cu, (1, 0), value=0)

                    for blk in hf_vt.blocks:
                        hf_h = blk(hf_h, cu_seqlens=cu, rotary_pos_emb=rotary)

                    if getattr(hf_vt.config, "post_norm", False) and hasattr(hf_vt, "post_trunk_norm"):
                        hf_h = hf_vt.post_trunk_norm(hf_h)

                    # ----- TT pre-merger hidden -----
                    tt_h = visual.tt_model.patch_embed(inputs.pixel_values, inputs.image_grid_thw)
                    if isinstance(tt_h, torch.Tensor):
                        tt_h = ttnn.from_torch(tt_h, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                    if len(tt_h.shape) == 3:
                        tt_h = ttnn.reshape(tt_h, (1, 1, tt_h.shape[1], tt_h.shape[2]))

                    rot_mats, cu_t = visual.tt_model.build_rot_mats_and_cu(
                        grid_thw=inputs.image_grid_thw, seq_len=int(tt_h.shape[2])
                    )
                    for blk in visual.tt_model.blocks:
                        tt_h = blk(tt_h, rot_mats=rot_mats, cu_seqlens=cu_t)
                    tt_h = visual.tt_model.norm(tt_h)

                    composer = ttnn.ConcatMeshToTensor(mesh, dim=0)
                    tt_h_t = ttnn.to_torch(tt_h, mesh_composer=composer).to(torch.float32)
                    if tt_h_t.dim() == 4:
                        tt_h_t = tt_h_t.squeeze(0).squeeze(0)
                    elif tt_h_t.dim() == 3:
                        tt_h_t = tt_h_t.reshape(-1, tt_h_t.shape[-1])

                    n1 = min(int(hf_h.shape[0]), int(tt_h_t.shape[0]))
                    d1 = min(int(hf_h.shape[1]), int(tt_h_t.shape[1]))
                    trunk_pcc = comp_pcc(hf_h[:n1, :d1].float(), tt_h_t[:n1, :d1].float())
                    print(
                        f"Vision trunk(pre-merger) PCC: {trunk_pcc:.6f} (hf={tuple(hf_h.shape)} tt={tuple(tt_h_t.shape)})"
                    )

                    # ----- Merger output PCC (HF merger vs TT merger) -----
                    hf_m = hf_vt.merger(hf_h)
                    if isinstance(hf_m, (tuple, list)):
                        hf_m = hf_m[0]
                    if hf_m.dim() == 4:
                        hf_m = hf_m.squeeze(0).squeeze(0)
                    elif hf_m.dim() == 3:
                        hf_m = hf_m.reshape(-1, hf_m.shape[-1])

                    tt_m = visual.tt_model.patch_merger(tt_h)
                    tt_m_t = ttnn.to_torch(tt_m, mesh_composer=composer).to(torch.float32)
                    if tt_m_t.dim() == 4:
                        tt_m_t = tt_m_t.squeeze(0).squeeze(0)
                    elif tt_m_t.dim() == 3:
                        tt_m_t = tt_m_t.reshape(-1, tt_m_t.shape[-1])

                    n2 = min(int(hf_m.shape[0]), int(tt_m_t.shape[0]))
                    d2 = min(int(hf_m.shape[1]), int(tt_m_t.shape[1]))
                    merger_pcc = comp_pcc(hf_m[:n2, :d2].float(), tt_m_t[:n2, :d2].float())
                    print(f"Vision merger PCC: {merger_pcc:.6f} (hf={tuple(hf_m.shape)} tt={tuple(tt_m_t.shape)})")

                    # Cross-check merger correctness independent of trunk mismatch:
                    # 1) Run TT merger on HF trunk hidden states.
                    try:
                        # HF trunk hidden [S,D] -> TT expects [1,1,S,D] in TILE.
                        hf_h_tt = ttnn.from_torch(
                            hf_h.to(torch.bfloat16).reshape(1, 1, hf_h.shape[0], hf_h.shape[1]),
                            device=mesh,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                        )
                        tt_m_from_hf = visual.tt_model.patch_merger(hf_h_tt)
                        tt_m_from_hf_t = ttnn.to_torch(tt_m_from_hf, mesh_composer=composer).to(torch.float32)
                        if tt_m_from_hf_t.dim() == 4:
                            tt_m_from_hf_t = tt_m_from_hf_t.squeeze(0).squeeze(0)
                        elif tt_m_from_hf_t.dim() == 3:
                            tt_m_from_hf_t = tt_m_from_hf_t.reshape(-1, tt_m_from_hf_t.shape[-1])
                        n3 = min(int(hf_m.shape[0]), int(tt_m_from_hf_t.shape[0]))
                        d3 = min(int(hf_m.shape[1]), int(tt_m_from_hf_t.shape[1]))
                        p3 = comp_pcc(hf_m[:n3, :d3].float(), tt_m_from_hf_t[:n3, :d3].float())
                        print(f"Vision merger PCC (TT merger on HF trunk): {p3:.6f}")
                    except Exception as exc2:
                        print(f"Vision merger cross-check skipped (TT on HF trunk): {type(exc2).__name__}: {exc2}")

                    # 2) Run HF merger on TT trunk hidden states.
                    try:
                        # ``tt_h_t`` was upcast to float32 for trunk PCC; HF ``merger`` weights use
                        # ``ref.model`` dtype (e.g. bfloat16). Match activations to parameter dtype.
                        _m_dtype = next(hf_vt.merger.parameters()).dtype
                        # TT trunk hidden torch [S,D] (tt_h_t) -> HF merger expects [S,D] then views internally.
                        hf_m_from_tt = hf_vt.merger(tt_h_t.to(_m_dtype))
                        if isinstance(hf_m_from_tt, (tuple, list)):
                            hf_m_from_tt = hf_m_from_tt[0]
                        if hf_m_from_tt.dim() == 4:
                            hf_m_from_tt = hf_m_from_tt.squeeze(0).squeeze(0)
                        elif hf_m_from_tt.dim() == 3:
                            hf_m_from_tt = hf_m_from_tt.reshape(-1, hf_m_from_tt.shape[-1])
                        n4 = min(int(hf_m_from_tt.shape[0]), int(tt_m_t.shape[0]))
                        d4 = min(int(hf_m_from_tt.shape[1]), int(tt_m_t.shape[1]))
                        p4 = comp_pcc(hf_m_from_tt[:n4, :d4].float(), tt_m_t[:n4, :d4].float())
                        print(f"Vision merger PCC (HF merger on TT trunk): {p4:.6f}")
                    except Exception as exc3:
                        print(f"Vision merger cross-check skipped (HF on TT trunk): {type(exc3).__name__}: {exc3}")
        except Exception as exc:
            print(f"Vision trunk/merger PCC skipped: {type(exc).__name__}: {exc}")

        hf_out = ref.vision_forward(inputs.pixel_values, inputs.image_grid_thw).float()
        tt_out = visual(inputs.pixel_values, inputs.image_grid_thw).float()

        # Compare overlapping rows if shapes differ slightly.
        n = min(hf_out.shape[0], tt_out.shape[0])
        d = min(hf_out.shape[1], tt_out.shape[1])
        pcc = comp_pcc(hf_out[:n, :d], tt_out[:n, :d])
        print(f"Vision tower PCC: {pcc:.6f} (hf={tuple(hf_out.shape)} tt={tuple(tt_out.shape)})")
        assert pcc > 0.88
    finally:
        close_dots_mesh_device(mesh)
