# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# End-to-end PCC test

import sys
from pathlib import Path

import pytest
import torch
import torchvision.transforms as T
from PIL import Image

import ttnn

_device = None
_torch_model = None
_tt_params = None
_ref_logits = None
_ref_boxes = None
_tt_logits = None
_tt_boxes = None
_sample_input = None
_ref_topk_ind = None

repo_path = Path(__file__).parent.parent / "RT-DETR" / "rtdetr_pytorch"
sys.path.insert(0, str(repo_path))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import YAMLConfig
from tt.hybrid_encoder import hybrid_encoder
from tt.postprocessor import postprocess
from tt.resnet_backbone import presnet50
from tt.rtdetr_decoder import decoder_layer
from tt.weight_utils import get_backbone_parameters, get_decoder_parameters, get_encoder_parameters

from models.common.utility_functions import comp_pcc

cfg_path = repo_path / "configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
ckpt_path = Path(__file__).parent.parent / "weights/rtdetr_r50vd.pth"

pcc_threshold = 0.90
score_threshold = 0.4


def _to_device(tensor_nchw, device):
    t = tensor_nchw.permute(0, 2, 3, 1).contiguous()
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if hasattr(device, "get_num_devices") else None,
    )


def _pull(tt_tensor, device):
    mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0) if hasattr(device, "get_num_devices") else None
    return ttnn.to_torch(
        tt_tensor,
        mesh_composer=mesh_composer,
    )[0:1].float()


def _run_torch_reference(torch_model, sample_input):
    """Clean PyTorch forward pass, returning logits, boxes, and the Top-K indices."""
    with torch.no_grad():
        out = torch_model(sample_input)

        feats = torch_model.backbone(sample_input)
        feats = torch_model.encoder(feats)
        memory_pt, spatial_shapes, _ = torch_model.decoder._get_encoder_input(feats)

        decoder = torch_model.decoder
        if decoder.eval_spatial_size is None:
            anchors, valid_mask = decoder._generate_anchors(spatial_shapes, device=memory_pt.device)
        else:
            anchors, valid_mask = decoder.anchors.to(memory_pt.device), decoder.valid_mask.to(memory_pt.device)

        memory_masked = valid_mask.to(memory_pt.dtype) * memory_pt
        output_memory = decoder.enc_output(memory_masked)
        enc_outputs_class = decoder.enc_score_head(output_memory)

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, decoder.num_queries, dim=1)

    return out["pred_logits"].detach(), out["pred_boxes"].detach(), topk_ind


def _run_tt_pipeline(sample_input, tt_params, torch_model, device, ref_topk_ind, return_debug=True):
    from src.zoo.rtdetr.utils import inverse_sigmoid

    from models.common.utility_functions import comp_pcc

    def check_spatial(name, pt_nchw, tt_tensor):
        """Handles both NCHW (4D) and flat NLC (3D) PyTorch tensors."""
        tt_raw = _pull(tt_tensor, device)  # always (1, 1, H*W, C) or (1, 1, seq, C)

        if pt_nchw.dim() == 4:
            # Standard spatial feature map: (1, C, H, W)
            _, C, H, W = pt_nchw.shape
            tt_nchw = tt_raw.squeeze(1).reshape(1, H, W, C).permute(0, 3, 1, 2).contiguous()
            pt_f = pt_nchw.float()
            tt_f = tt_nchw.float()

        elif pt_nchw.dim() == 3:
            pt_f = pt_nchw.float().reshape(-1)
            tt_f = tt_raw.squeeze(1).float().reshape(-1)

        else:
            raise ValueError(f"check_spatial: unexpected pt shape {pt_nchw.shape}")

        pcc, msg = comp_pcc(pt_f, tt_f, 0.96)
        print(f"  [Trace] {name:.<40} {' pass' if pcc else ' FAIL'} ({msg})")
        return pcc

    def check_flat(name, pt, tt_tensor_or_torch):
        if isinstance(tt_tensor_or_torch, torch.Tensor):
            tt_f = tt_tensor_or_torch.float()
        else:
            tt_f = _pull(tt_tensor_or_torch, device).float()
        pcc, msg = comp_pcc(pt.float().reshape(-1), tt_f.reshape(-1), 0.96)
        print(f"  [Trace] {name:.<40} {' pass' if pcc else ' FAIL'} ({msg})")
        return pcc

    print("\n" + "=" * 50)
    print("=== E2E Pipeline Debug Trace ===")

    h5, w5 = 20, 20

    with torch.no_grad():
        s_pt = torch_model.backbone(sample_input)
        s3_pt, s4_pt, s5_pt = s_pt[0], s_pt[1], s_pt[2]

        p_pt = torch_model.encoder(s_pt)
        p3_pt, p4_pt, p5_pt = p_pt[0], p_pt[1], p_pt[2]

        # FPN intermediate goldens
        with torch.no_grad():
            proj_feats = [torch_model.encoder.input_proj[i](feat) for i, feat in enumerate([s3_pt, s4_pt, s5_pt])]

            # Post-AIFI p5 (encoder[0] runs on proj_feats[2])
            h5_s, w5_s = proj_feats[2].shape[2:]
            src_flatten = proj_feats[2].flatten(2).permute(0, 2, 1)
            if torch_model.encoder.eval_spatial_size:
                pos_embed_pt = getattr(torch_model.encoder, "pos_embed2").to(src_flatten.device)
            else:
                pos_embed_pt = torch_model.encoder.build_2d_sincos_position_embedding(
                    w5_s,
                    h5_s,
                    torch_model.encoder.hidden_dim,
                    torch_model.encoder.pe_temperature,
                ).to(src_flatten.device)

            # Post-AIFI golden — keep as (1, 400, 256) flat, matching TTNN's (1, 1, 400, 256)
            post_aifi_p5_pt = torch_model.encoder.encoder[0](src_flatten, pos_embed=pos_embed_pt)

            post_aifi_p5_pt_nchw = post_aifi_p5_pt.permute(0, 2, 1).reshape(1, 256, 20, 20).contiguous()
            proj_feats[2] = post_aifi_p5_pt_nchw

            enc = torch_model.encoder

            # FPN intermediates — all 4D NCHW
            p5_lat_pt = torch.nn.functional.silu(enc.lateral_convs[0](proj_feats[2]))  # (1, 256, 20, 20)
            p5_up = torch.nn.functional.interpolate(p5_lat_pt, scale_factor=2.0, mode="nearest")
            p4_cat_pt = torch.cat([p5_up, proj_feats[1]], dim=1)  # (1, 512, 40, 40)
            p4_td_pt = enc.fpn_blocks[0](p4_cat_pt)  # (1, 256, 40, 40)
            p4_lat_pt = torch.nn.functional.silu(enc.lateral_convs[1](p4_td_pt))  # (1, 256, 40, 40)
            p4_up = torch.nn.functional.interpolate(p4_lat_pt, scale_factor=2.0, mode="nearest")
            p3_cat_pt = torch.cat([p4_up, proj_feats[0]], dim=1)  # (1, 512, 80, 80)
            p3_out_pt = enc.fpn_blocks[1](
                p3_cat_pt
            )  # (1, 256, 80, 80)                                     # (1, 256, 80, 80)

        # Intermediate AIFI goldens
        proj_feats_pt = [torch_model.encoder.input_proj[i](feat) for i, feat in enumerate([s3_pt, s4_pt, s5_pt])]
        pre_aifi_p5_pt_nchw = proj_feats_pt[2]  # (1, 256, 20, 20)

        src_flatten = proj_feats_pt[2].flatten(2).permute(0, 2, 1)  # (1, 400, 256)
        if torch_model.encoder.eval_spatial_size:
            pos_embed_pt = getattr(torch_model.encoder, "pos_embed2").to(src_flatten.device)
        else:
            pos_embed_pt = torch_model.encoder.build_2d_sincos_position_embedding(
                w5,
                h5,
                torch_model.encoder.hidden_dim,
                torch_model.encoder.pe_temperature,
            ).to(src_flatten.device)

        post_aifi_p5_pt = torch_model.encoder.encoder[0](src_flatten, pos_embed=pos_embed_pt)
        post_aifi_p5_pt_nchw = post_aifi_p5_pt.permute(0, 2, 1).reshape(1, 256, h5, w5).contiguous()

        decoder_pt = torch_model.decoder
        memory_pt, spatial_shapes, _ = decoder_pt._get_encoder_input(p_pt)

        if decoder_pt.eval_spatial_size is None:
            anchors, valid_mask = decoder_pt._generate_anchors(spatial_shapes, device=memory_pt.device)
        else:
            anchors = decoder_pt.anchors.to(memory_pt.device)
            valid_mask = decoder_pt.valid_mask.to(memory_pt.device)

        memory_masked = valid_mask.to(memory_pt.dtype) * memory_pt
        output_memory = decoder_pt.enc_output(memory_masked)
        enc_outputs_coord_unact = decoder_pt.enc_bbox_head(output_memory) + anchors

        init_ref_points_unact = enc_outputs_coord_unact.gather(
            dim=1, index=ref_topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1])
        ).detach()

        target_pt = output_memory.gather(
            dim=1, index=ref_topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        ).detach()

        # Decoder layer goldens
        layer_golden_outputs = []
        output_pt = target_pt
        ref_points_detach = torch.sigmoid(init_ref_points_unact)

        for i, layer in enumerate(decoder_pt.decoder.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = decoder_pt.query_pos_head(ref_points_detach)
            output_pt = layer(output_pt, ref_points_input, memory_pt, spatial_shapes, None, None, None, query_pos_embed)
            inter_ref_bbox = torch.sigmoid(decoder_pt.dec_bbox_head[i](output_pt) + inverse_sigmoid(ref_points_detach))
            layer_golden_outputs.append((output_pt.clone(), query_pos_embed.clone()))
            ref_points_detach = inter_ref_bbox

    # 1. TTNN Backbone

    x = _to_device(sample_input, device)
    s3, s4, s5 = presnet50(x, tt_params["backbone"], device)

    check_spatial("Backbone s3", s3_pt, s3)
    check_spatial("Backbone s4", s4_pt, s4)
    check_spatial("Backbone s5", s5_pt, s5)

    # 2. TTNN Encoder
    if return_debug:
        encoder_outs, debug_tt = hybrid_encoder(s3, s4, s5, tt_params["encoder"], device, return_debug=True)
    else:
        encoder_outs = hybrid_encoder(s3, s4, s5, tt_params["encoder"], device)

    p3, p4, p5 = encoder_outs

    check_spatial("Encoder p3", p3_pt, p3)
    check_spatial("Encoder p4", p4_pt, p4)
    check_spatial("Encoder p5", p5_pt, p5)

    if return_debug:
        check_spatial("FPN p5_lat", p5_lat_pt, debug_tt["p5_lat"])  # 4D
        check_spatial("FPN p4_cat", p4_cat_pt, debug_tt["p4_cat"])  # 4D, 512 ch
        check_spatial("FPN p4_td", p4_td_pt, debug_tt["p4_td"])  # 4D
        check_spatial("FPN p4_lat", p4_lat_pt, debug_tt["p4_lat"])  # 4D
        check_spatial("FPN p3_cat", p3_cat_pt, debug_tt["p3_cat"])  # 4D, 512 ch
        check_spatial("FPN p3_out", p3_out_pt, debug_tt["p3_out"])  # 4D

        for k, v in debug_tt.items():
            if v is not None:
                ttnn.deallocate(v)

    # 3. Encoder-to-Decoder Bridge

    p3_tt_eval = _pull(p3, device).squeeze(1).reshape(1, 80, 80, 256).permute(0, 3, 1, 2)
    p4_tt_eval = _pull(p4, device).squeeze(1).reshape(1, 40, 40, 256).permute(0, 3, 1, 2)
    p5_tt_eval = _pull(p5, device).squeeze(1).reshape(1, 20, 20, 256).permute(0, 3, 1, 2)

    with torch.no_grad():
        memory_tt_eval, _, _ = decoder_pt._get_encoder_input([p3_tt_eval, p4_tt_eval, p5_tt_eval])
        memory_masked_eval = valid_mask.to(memory_tt_eval.dtype) * memory_tt_eval
        output_memory_eval = decoder_pt.enc_output(memory_masked_eval)
        enc_outputs_coord_unact_eval = decoder_pt.enc_bbox_head(output_memory_eval) + anchors

        init_ref_points_unact_eval = enc_outputs_coord_unact_eval.gather(
            dim=1, index=ref_topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact_eval.shape[-1])
        ).detach()

        target_tt_eval = output_memory_eval.gather(
            dim=1, index=ref_topk_ind.unsqueeze(-1).repeat(1, 1, output_memory_eval.shape[-1])
        ).detach()

    check_flat("Bridge Memory", memory_pt, memory_tt_eval)
    check_flat("Bridge Target (Queries)", target_pt, target_tt_eval)
    check_flat("Bridge Init Ref Points", init_ref_points_unact, init_ref_points_unact_eval)

    # 4. TTNN Decoder

    n_levels = len(spatial_shapes)

    query_tt = ttnn.from_torch(
        target_tt_eval.unsqueeze(1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if hasattr(device, "get_num_devices") else None,
    )

    actual_decoder = decoder_pt.decoder
    ref_points_detach_eval = torch.sigmoid(init_ref_points_unact_eval.clone())

    if ref_points_detach_eval.dim() == 4:
        ref_points_detach_eval = ref_points_detach_eval[:, :, 0, :]

    for i, (torch_layer, tt_params_layer) in enumerate(zip(actual_decoder.layers, tt_params["decoder"])):
        with torch.no_grad():
            query_pos = decoder_pt.query_pos_head(ref_points_detach_eval)

        check_flat(f"Decoder L{i} Query Pos", layer_golden_outputs[i][1], query_pos)

        query_pos_tt = ttnn.from_torch(
            query_pos.reshape(1, 1, 300, 256).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if hasattr(device, "get_num_devices") else None,
        )

        ref_points_input = ref_points_detach_eval.unsqueeze(2).expand(-1, -1, n_levels, -1)

        query_tt = decoder_layer(
            query_tt,
            query_pos_tt,
            torch_layer,
            tt_params_layer,
            memory_tt_eval,
            ref_points_input,
            spatial_shapes,
            device,
            8,
            valid_mask=None,
        )

        query_torch_out = _pull(query_tt, device).view(1, 300, 256)

        check_flat(f"Decoder L{i} Output", layer_golden_outputs[i][0], query_torch_out)

        with torch.no_grad():
            inter_ref_bbox = torch.sigmoid(
                decoder_pt.dec_bbox_head[i](query_torch_out) + inverse_sigmoid(ref_points_detach_eval)
            )
        ref_points_detach_eval = inter_ref_bbox
        ttnn.deallocate(query_pos_tt)

    # 5. Prediction Heads

    with torch.no_grad():
        pred_logits = decoder_pt.dec_score_head[-1](query_torch_out)
        pred_boxes = ref_points_detach_eval

    print("=" * 50 + "\n")

    return pred_logits, pred_boxes


def _setup_all():
    global _device, _torch_model, _tt_params
    global _ref_logits, _ref_boxes, _tt_logits, _tt_boxes, _sample_input, _ref_topk_ind

    if _device is not None:
        return

    mesh_shape = ttnn.MeshShape(1, 2)
    _device = ttnn.open_mesh_device(mesh_shape, l1_small_size=16384)

    cfg = YAMLConfig(str(cfg_path))
    _torch_model = cfg.model
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    _torch_model.load_state_dict(ckpt["ema"]["module"])
    _torch_model.eval()

    _tt_params = {
        "backbone": get_backbone_parameters(_torch_model, _device),
        "encoder": get_encoder_parameters(_torch_model, _device),
        "decoder": get_decoder_parameters(_torch_model, _device),
    }

    img_path = Path(__file__).parent.parent / "demo" / "demo_images" / "sample.jpg"

    if img_path.exists():
        print(f"\n[setup] Loading real image from {img_path}")
        img = Image.open(img_path).convert("RGB").resize((640, 640))
        sample_input = T.ToTensor()(img).unsqueeze(0)
    else:
        print("\n[setup] WARNING: No sample image found. Falling back to zeros.")
        sample_input = torch.zeros(1, 3, 640, 640)

    print("\n[setup] running PyTorch reference...")
    _ref_logits, _ref_boxes, _ref_topk_ind = _run_torch_reference(_torch_model, sample_input)

    _sample_input = sample_input

    print("[setup] running TT pipeline...")
    _tt_logits, _tt_boxes = _run_tt_pipeline(sample_input, _tt_params, _torch_model, _device, _ref_topk_ind)

    print("[setup] done\n")


@pytest.fixture(scope="module", autouse=True)
def force_pipeline_setup():
    """This fixture runs automatically before any tests in this module."""
    print("\n[PyTest Fixture] Running setup...")
    _setup_all()
    yield
    print("\n[PyTest Fixture] Tearing down...")
    global _device
    if _device is not None:
        ttnn.close_mesh_device(_device)
        _device = None


def test_pred_logits_pcc():
    pcc, msg = comp_pcc(_ref_logits, _tt_logits, pcc_threshold)
    print(f"\npred_logits PCC: {msg}")
    assert pcc, f"pred_logits PCC below {pcc_threshold} - {msg}"


def test_pred_boxes_pcc():
    pcc, msg = comp_pcc(_ref_boxes, _tt_boxes, pcc_threshold)
    print(f"\npred_boxes PCC: {msg}")
    assert pcc, f"pred_boxes PCC below {pcc_threshold} - {msg}"


def test_top5_labels_and_scores():
    ref_scores, ref_labels = _ref_logits[0].sigmoid().max(-1)
    tt_scores, tt_labels = _tt_logits[0].sigmoid().max(-1)

    top5 = ref_scores.topk(5).indices
    for idx in top5:
        assert ref_labels[idx] == tt_labels[idx], (
            f"query {idx.item()}: label mismatch  ref={ref_labels[idx].item()}  " f"tt={tt_labels[idx].item()}"
        )
        diff = abs(ref_scores[idx].item() - tt_scores[idx].item())
        assert diff < 0.05, f"query {idx.item()}: score diff {diff:.4f} > 0.05"


def test_box_iou_fired_detections():
    from torchvision.ops import box_iou

    def to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)

    ref_scores = _ref_logits[0].sigmoid().max(-1).values
    keep = ref_scores > score_threshold

    if keep.sum() == 0:
        pytest.skip("no reference detections above threshold - lower score_threshold")

    iou = box_iou(to_xyxy(_ref_boxes[0][keep]), to_xyxy(_tt_boxes[0][keep])).diag()
    low = iou[iou < 0.9]
    assert len(low) == 0, f"{len(low)} boxes have IoU < 0.9  (min={iou.min():.4f})"


if __name__ == "__main__":
    _setup_all()

    print("\n=== Postprocessor Detections Sample ===")
    orig_size = torch.tensor([[640, 640]])

    tt_logits_tensor = ttnn.from_torch(
        _tt_logits.reshape(1, 1, 300, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(_device) if hasattr(_device, "get_num_devices") else None,
    )
    tt_boxes_tensor = ttnn.from_torch(
        _tt_boxes.reshape(1, 1, 300, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(_device) if hasattr(_device, "get_num_devices") else None,
    )

    results = postprocess(tt_logits_tensor, tt_boxes_tensor, orig_size, score_threshold)

    for i, r in enumerate(results):
        print(f"Image {i}: {len(r['scores'])} detections above threshold {score_threshold}")
        for s, l, b in zip(r["scores"], r["labels"], r["boxes"]):
            print(
                f"  [Class {l.item():3d}] Score: {s.item():.3f} | Box: [{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]"
            )

    if _device is not None:
        ttnn.close_mesh_device(_device)
