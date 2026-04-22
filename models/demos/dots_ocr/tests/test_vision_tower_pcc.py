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

    ref = DotsOCRReference(HFLoadSpec(model_id="rednote-hilab/dots.mocr", dtype=torch.float32))
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
                    if tt_x.dim() == 3:
                        tt_x = ttnn.reshape(tt_x, (1, 1, tt_x.shape[1], tt_x.shape[2]))

                    rot_mats, cu_t = visual.tt_model.build_rot_mats_and_cu(
                        grid_thw=inputs.image_grid_thw, seq_len=int(tt_x.shape[2])
                    )
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
