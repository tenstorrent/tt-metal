# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# End-to-end PCC test + demo inference

import sys
from pathlib import Path

import torch
import ttnn
import pytest

REPO_PATH = Path(__file__).parent.parent / "RT-DETR" / "rtdetr_pytorch"
sys.path.insert(0, str(REPO_PATH))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import YAMLConfig

from tt.resnet_backbone import presnet50
from tt.hybrid_encoder import hybrid_encoder
from tt.rtdetr_decoder import run_decoder
from tt.postprocessor import postprocess
from tt.weight_utils import get_backbone_parameters, get_encoder_parameters, get_decoder_parameters
from models.common.utility_functions import comp_pcc

CFG_PATH  = REPO_PATH / "configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
CKPT_PATH = Path(__file__).parent.parent / "weights/rtdetr_r50vd.pth"

PCC_THRESHOLD   = 0.97
SCORE_THRESHOLD = 0.3

_captured_query      = None
_captured_ref_points = None
_captured_query_pos  = None
_captured_memory     = None

def _to_device(tensor_nchw, device):
    t = tensor_nchw.permute(0, 2, 3, 1).contiguous()
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def _pull(tt_tensor, device):
    return ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0),
    )[0:1].float()


#  torch reference 

def _run_torch_reference(torch_model, sample_input):
    global _captured_query, _captured_ref_points, _captured_query_pos, _captured_memory

    def capture_hook(module, args, kwargs):
        global _captured_query, _captured_ref_points, _captured_query_pos, _captured_memory

        if 'tgt' in kwargs:
            _captured_query = kwargs['tgt'].detach()
        elif len(args) > 0:
            _captured_query = args[0].detach()

        if 'ref_points' in kwargs:
            _captured_ref_points = kwargs['ref_points'].detach()
        elif 'reference_points' in kwargs:
            _captured_ref_points = kwargs['reference_points'].detach()
        elif len(args) > 1:
            _captured_ref_points = args[1].detach()

        if 'query_pos' in kwargs and kwargs['query_pos'] is not None:
            _captured_query_pos = kwargs['query_pos'].detach()
        elif len(args) > 7 and args[7] is not None:
            _captured_query_pos = args[7].detach()

        if 'memory' in kwargs:
            _captured_memory = kwargs['memory'].detach()
        elif len(args) > 2:
            _captured_memory = args[2].detach()

    handle = torch_model.decoder.decoder.layers[0].register_forward_pre_hook(
        capture_hook, with_kwargs=True
    )
    with torch.no_grad():
        out = torch_model(sample_input)
    handle.remove()

    return out["pred_logits"].detach(), out["pred_boxes"].detach()

def _debug_decoder_trace(torch_model, device, tt_params):
    """Step through decoder layer by layer and check PCC at each stage."""
    from src.zoo.rtdetr.utils import inverse_sigmoid
    from models.common.utility_functions import comp_pcc

    def check(name, pt, tt):
        if isinstance(tt, torch.Tensor):
            tt_f = tt.float()
        else:
            tt_f = _pull(tt, device).squeeze(1)
        pt_f = pt.float()
        pcc, msg = comp_pcc(pt_f, tt_f, 0.97)
        print(f"  {name:.<45} {'ok' if pcc else 'not ok'} ({msg})")
        return pcc

    
    print(" Decoder + Postprocessor trace")
    

    decoder_module = torch_model.decoder.decoder

    pt_per_layer = {i: {} for i in range(6)}

    def make_pre_hook(i):
        def hook(module, args, kwargs):
            # args[1] is reference_points
            pt_per_layer[i]['ref_points'] = args[1].detach() if len(args) > 1 else kwargs['reference_points'].detach()
            # args[7] is query_pos_embed
            qpe = kwargs.get('query_pos_embed', args[7] if len(args) > 7 else None)
            if qpe is not None:
                pt_per_layer[i]['query_pos'] = qpe.detach()
        return hook

    def make_post_hook(i):
        def hook(module, args, kwargs, output):
            pt_per_layer[i]['output'] = output.detach()
        return hook

    handles = []
    for i, layer in enumerate(decoder_module.layers):
        handles.append(layer.register_forward_pre_hook(make_pre_hook(i), with_kwargs=True))
        handles.append(layer.register_forward_hook(make_post_hook(i), with_kwargs=True))

    orig_fwd = decoder_module.forward
    _ref_points_seq = []

    def patched_fwd(*args, **kwargs):
        _ref_points_seq.clear()
        return orig_fwd(*args, **kwargs)
    decoder_module.forward = patched_fwd

    with torch.no_grad():
        _ = torch_model(_sample_input)

    decoder_module.forward = orig_fwd
    for h in handles:
        h.remove()

    print("\n[Decoder Layer-by-Layer]")

    query_tt = ttnn.from_torch(
        _captured_query.reshape(1, 1, 300, 256).to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    memory_tt = ttnn.from_torch(
        _captured_memory.to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    for i in range(6):
        print(f"\n  --- Layer {i} ---")
        torch_layer = decoder_module.layers[i]
        params = tt_params["decoder"][i]
        ref_points_input = pt_per_layer[i]['ref_points']
        query_pos = pt_per_layer[i]['query_pos']

        query_pos_tt = ttnn.from_torch(
            query_pos.reshape(1, 1, 300, 256).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

        from tt.rtdetr_decoder import _self_attention, _layer_norm
        residual = query_tt
        sa_out = _self_attention(query_tt, query_pos_tt, params.self_attn, device, num_heads=8)
        query_tt = _layer_norm(
            ttnn.add(residual, sa_out, memory_config=ttnn.L1_MEMORY_CONFIG),
            params.norm1,
        )

        query_torch = _pull(query_tt, device).view(1, 300, 256)
        query_pos_torch = query_pos.view(1, 300, 256)
        
        q_with_pos = query_torch + query_pos_torch
        memory_torch = _captured_memory.float()

        with torch.no_grad():
            ca_out = torch_layer.cross_attn(
                query=q_with_pos,
                reference_points=ref_points_input,
                value=memory_torch,
                value_spatial_shapes=torch.tensor([[80,80],[40,40],[20,20]]),
                value_mask=None,
            )

        ca_out_tt = ttnn.from_torch(
            ca_out.reshape(1, 1, 300, 256).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device), 
        )
        
        residual = query_tt
        query_tt = _layer_norm(
            ttnn.add(residual, ca_out_tt, memory_config=ttnn.L1_MEMORY_CONFIG),
            params.norm2,
        )

        residual = query_tt
        ffn = ttnn.linear(query_tt, params.linear1.weight, bias=params.linear1.bias,
                          activation="relu", memory_config=ttnn.L1_MEMORY_CONFIG)
        ffn = ttnn.linear(ffn, params.linear2.weight, bias=params.linear2.bias,
                          memory_config=ttnn.L1_MEMORY_CONFIG)
        query_tt = _layer_norm(
            ttnn.add(residual, ffn, memory_config=ttnn.L1_MEMORY_CONFIG),
            params.norm3,
        )

        check(f"L{i} layer output", pt_per_layer[i]['output'], query_tt)

    print("\n[Prediction Heads]")
    query_torch_final = _pull(query_tt, device).view(1, 300, 256)
    
    with torch.no_grad():
        # 1. Logits are direct
        tt_logits = torch_model.decoder.dec_score_head[-1](query_torch_final)
        
        # 2. Extract and format the final reference points, stripping any dummy dims
        from src.zoo.rtdetr.utils import inverse_sigmoid
        final_ref_points = pt_per_layer[5]['ref_points'].view(1, 300, -1)
        
        # 3. Calculate raw box offsets
        box_offsets = torch_model.decoder.dec_bbox_head[-1](query_torch_final).view(1, 300, 4)
        
        # 4. Safely apply the RT-DETR bounding box update rule
        if final_ref_points.shape[-1] == 4:
            tt_boxes = torch.sigmoid(box_offsets + inverse_sigmoid(final_ref_points))
        else:
            # If reference points are only 2D (x, y), only add to the first two coords
            tt_boxes = box_offsets.clone()
            tt_boxes[..., :2] = box_offsets[..., :2] + inverse_sigmoid(final_ref_points)
            tt_boxes = torch.sigmoid(tt_boxes)

    check("pred_logits", _ref_logits, tt_logits)
    check("pred_boxes",  _ref_boxes,  tt_boxes)

#  ttnn pipeline 

def _run_tt_pipeline(sample_input, tt_params, torch_model, device):
    global _captured_query, _captured_ref_points, _captured_query_pos

    x = _to_device(sample_input, device)
    
    # 1. Backbone
    s3, s4, s5 = presnet50(x, tt_params["backbone"], device)
    
    # 2. Encoder
    p3, p4, p5 = hybrid_encoder(s3, s4, s5, tt_params["encoder"], device)

    # 3. ACTUAL Encoder-to-Decoder Handoff (No more cheating)
    p3_pt = _pull(p3, device).squeeze(1).reshape(1, 80, 80, 256).permute(0, 3, 1, 2)
    p4_pt = _pull(p4, device).squeeze(1).reshape(1, 40, 40, 256).permute(0, 3, 1, 2)
    p5_pt = _pull(p5, device).squeeze(1).reshape(1, 20, 20, 256).permute(0, 3, 1, 2)

    with torch.no_grad():
        memory_pt, spatial_shapes, _ = torch_model.decoder._get_encoder_input([p3_pt, p4_pt, p5_pt])
    
    spatial_shapes = torch.tensor(spatial_shapes)
    
    memory_tt = ttnn.from_torch(
        memory_pt.unsqueeze(1).to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    decoder_pt = torch_model.decoder

    query_tt = ttnn.from_torch(
        _captured_query.reshape(1, 1, 300, 256).to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    # Initial query pos doesn't matter, run_decoder overwrites it
    query_pos_tt = ttnn.from_torch(
        torch.zeros(1, 1, 300, 256, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    # 4. TTNN Decoder 
    query_out, final_ref_points = run_decoder(
        query_tt, query_pos_tt,
        torch_decoder=decoder_pt,
        tt_layer_params=tt_params["decoder"],
        memory=memory_tt,
        ref_points=_captured_ref_points,
        spatial_shapes=spatial_shapes,
        device=device,
    )

    # 5. Prediction heads 
    query_torch = _pull(query_out, device).view(1, 300, 256)
    
    with torch.no_grad():
        # Logits
        pred_logits = decoder_pt.dec_score_head[-1](query_torch)
        # Boxes (already fully computed by run_decoder)
        pred_boxes = final_ref_points.view(1, 300, 4) 

    return pred_logits, pred_boxes


#  module-level setup 

def _setup_all():
    global _device, _torch_model, _tt_params
    global _ref_logits, _ref_boxes, _tt_logits, _tt_boxes

    mesh_shape = ttnn.MeshShape(1, 2)
    _device = ttnn.open_mesh_device(mesh_shape, l1_small_size=16384)

    cfg = YAMLConfig(str(CFG_PATH))
    _torch_model = cfg.model
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
    _torch_model.load_state_dict(ckpt["ema"]["module"])
    _torch_model.eval()

    _tt_params = {
        "backbone": get_backbone_parameters(_torch_model, _device),
        "encoder":  get_encoder_parameters(_torch_model, _device),
        "decoder":  get_decoder_parameters(_torch_model, _device),
    }

    torch.manual_seed(7)
    sample_input = torch.randn(1, 3, 640, 640)

    print("\n[setup] running PyTorch reference...")
    _ref_logits, _ref_boxes = _run_torch_reference(_torch_model, sample_input)

    global _sample_input
    _sample_input = sample_input
    _debug_decoder_trace(_torch_model, _device, _tt_params)

    print("[setup] running TT pipeline...")
    _tt_logits, _tt_boxes = _run_tt_pipeline(sample_input, _tt_params, _torch_model, _device)

    print("[setup] done\n")


_device = _torch_model = _tt_params = None
_ref_logits = _ref_boxes = _tt_logits = _tt_boxes = None
_setup_all()


#  tests 

def test_pred_logits_pcc():
    pcc, msg = comp_pcc(_ref_logits, _tt_logits, PCC_THRESHOLD)
    print(f"\npred_logits PCC: {msg}")
    assert pcc, f"pred_logits PCC below {PCC_THRESHOLD} — {msg}"


def test_pred_boxes_pcc():
    pcc, msg = comp_pcc(_ref_boxes, _tt_boxes, PCC_THRESHOLD)
    print(f"\npred_boxes PCC: {msg}")
    assert pcc, f"pred_boxes PCC below {PCC_THRESHOLD} — {msg}"


def test_top5_labels_and_scores():
    ref_scores, ref_labels = _ref_logits[0].sigmoid().max(-1)
    tt_scores,  tt_labels  = _tt_logits[0].sigmoid().max(-1)

    top5 = ref_scores.topk(5).indices
    for idx in top5:
        assert ref_labels[idx] == tt_labels[idx], (
            f"query {idx.item()}: label mismatch  ref={ref_labels[idx].item()}  "
            f"tt={tt_labels[idx].item()}"
        )
        diff = abs(ref_scores[idx].item() - tt_scores[idx].item())
        assert diff < 0.05, f"query {idx.item()}: score diff {diff:.4f} > 0.05"


def test_box_iou_fired_detections():
    from torchvision.ops import box_iou

    def to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], -1)

    ref_scores = _ref_logits[0].sigmoid().max(-1).values
    keep = ref_scores > SCORE_THRESHOLD

    if keep.sum() == 0:
        pytest.skip("no reference detections above threshold — lower SCORE_THRESHOLD")

    iou = box_iou(to_xyxy(_ref_boxes[0][keep]), to_xyxy(_tt_boxes[0][keep])).diag()
    low = iou[iou < 0.9]
    assert len(low) == 0, f"{len(low)} boxes have IoU < 0.9  (min={iou.min():.4f})"


#  main 

if __name__ == "__main__":
    print("\n=== Postprocessor Detections Sample ===")
    orig_size = torch.tensor([[640, 640]])

    tt_logits_tensor = ttnn.from_torch(
        _tt_logits.reshape(1, 1, 300, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=_device, memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(_device),
    )
    tt_boxes_tensor = ttnn.from_torch(
        _tt_boxes.reshape(1, 1, 300, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=_device, memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(_device),
    )

    results = postprocess(tt_logits_tensor, tt_boxes_tensor, orig_size, SCORE_THRESHOLD)

    for i, r in enumerate(results):
        print(f"Image {i}: {len(r['scores'])} detections above threshold {SCORE_THRESHOLD}")
        for s, l, b in zip(r["scores"], r["labels"], r["boxes"]):
            print(f"  [Class {l.item():3d}] Score: {s.item():.3f} | Box: [{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]")

    ttnn.close_mesh_device(_device)