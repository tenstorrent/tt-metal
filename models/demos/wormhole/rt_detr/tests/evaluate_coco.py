# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import gc
from pathlib import Path

import torch
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.datasets import CocoDetection
from tqdm import tqdm

import ttnn

import warnings
warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).parent.resolve()
PROJECT = THIS_DIR.parent
REPO_PATH = PROJECT / "RT-DETR" / "rtdetr_pytorch"

sys.path.insert(0, str(REPO_PATH))
sys.path.insert(0, str(PROJECT))

from src.core import YAMLConfig
from src.zoo.rtdetr.utils import inverse_sigmoid

from tt.resnet_backbone import presnet50
from tt.hybrid_encoder import hybrid_encoder
from tt.rtdetr_decoder import run_decoder
from tt.weight_utils import get_backbone_parameters, get_encoder_parameters, get_decoder_parameters

COCO_CLASS_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]

# ==============================================================================
# GLOBAL MESH CACHE: Prevents C++ PyBind Memory Leaks
# ==============================================================================
_global_composer = None
_global_mapper = None

def _get_global_utils(device):
    global _global_composer, _global_mapper
    if _global_composer is None and hasattr(device, 'get_num_devices'):
        _global_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        _global_mapper = ttnn.ReplicateTensorToMesh(device)
    return _global_composer, _global_mapper

def _to_device(tensor, device, nchw_to_nhwc=False, mem_config=ttnn.DRAM_MEMORY_CONFIG):
    _, mapper = _get_global_utils(device)
    t = tensor.permute(0, 2, 3, 1).contiguous() if nchw_to_nhwc else tensor.contiguous()
    return ttnn.from_torch(
        t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=mem_config, 
        mesh_mapper=mapper,
    )

def _pull(tt_tensor, device):
    composer, _ = _get_global_utils(device)
    return ttnn.to_torch(
        tt_tensor, mesh_composer=composer,
    )[0:1].float()


def run_ttnn_rtdetr_inference(img, torch_model, tt_params, device):
    """End-to-End TTNN Forward Pass."""
    
    # 1. TTNN Backbone
    x_tt = _to_device(img, device, nchw_to_nhwc=True)
    s3_tt, s4_tt, s5_tt = presnet50(x_tt, tt_params["backbone"], device)

    # 2. TTNN Encoder
    p3_tt, p4_tt, p5_tt = hybrid_encoder(s3_tt, s4_tt, s5_tt, tt_params["encoder"], device)

    # 3. Pull and reshape for the Decoder handoff
    p3_pt = _pull(p3_tt, device).squeeze(1).reshape(1, 80, 80, 256).permute(0, 3, 1, 2)
    p4_pt = _pull(p4_tt, device).squeeze(1).reshape(1, 40, 40, 256).permute(0, 3, 1, 2)
    p5_pt = _pull(p5_tt, device).squeeze(1).reshape(1, 20, 20, 256).permute(0, 3, 1, 2)

    with torch.no_grad():
        # Apply encoder projection and get spatial shapes
        memory_pt, spatial_shapes, _ = torch_model.decoder._get_encoder_input([p3_pt, p4_pt, p5_pt])
        spatial_shapes = torch.tensor(spatial_shapes)
        
        # Get targets and unactivated reference points
        tgt, init_ref_unact, _, _ = torch_model.decoder._get_decoder_input(memory_pt, spatial_shapes)
        init_reference = init_ref_unact.sigmoid()

    # 4. Push Queries to TTNN Decoder (Memory stays on CPU!)
    query_tt = _to_device(tgt.reshape(1, 1, 300, 256), device, mem_config=ttnn.L1_MEMORY_CONFIG)

    # 5. Run TTNN Decoder Stack
    query_out, final_ref_points = run_decoder(
        query_tt=query_tt, 
        torch_decoder=torch_model.decoder,
        tt_layer_params=tt_params["decoder"],
        memory_torch=memory_pt,  
        ref_points=init_reference, 
        spatial_shapes=spatial_shapes,
        device=device
    )

    # 6. Run Prediction Heads 
    query_torch = _pull(query_out, device).view(1, 300, 256)
    with torch.no_grad():
        logits = torch_model.decoder.dec_score_head[-1](query_torch)
        boxes = final_ref_points.view(1, 300, 4) 

    tt_tensors = (x_tt, s3_tt, s4_tt, s5_tt, p3_tt, p4_tt, p5_tt, query_tt, query_out)
    return logits, boxes, tt_tensors


# ==============================================================================
# COCO Evaluation Loop
# ==============================================================================

def main():
    data_dir = PROJECT / "data/coco/val2017"
    ann_file = PROJECT / "data/coco/annotations/instances_val2017.json"
    config_path = PROJECT / "RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
    ckpt_path = PROJECT / "weights/rtdetr_r50vd.pth"
    res_file = PROJECT / "results.json"

    # --- CHECKPOINT RESUME LOGIC ---
    results = []
    processed_img_ids = set()
    if res_file.exists():
        try:
            with open(res_file, "r") as f:
                results = json.load(f)
                processed_img_ids = {r["image_id"] for r in results}
            print(f"\n[CHECKPOINT FOUND] Resuming evaluation! Skipping {len(processed_img_ids)} already completed images.")
        except Exception:
            print("\n[CHECKPOINT CORRUPT] Starting from scratch.")
            results = []
            processed_img_ids = set()

    mesh_shape = ttnn.MeshShape(1, 2)
    device = ttnn.open_mesh_device(mesh_shape, l1_small_size=16384)

    try:
        print("\nLoading PyTorch model...")
        cfg = YAMLConfig(str(config_path))
        model = cfg.model
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(ckpt["ema"]["module"])
        model.eval()

        print("Pushing weights to TTNN device...")
        tt_params = {
            "backbone": get_backbone_parameters(model, device),
            "encoder":  get_encoder_parameters(model, device),
            "decoder":  get_decoder_parameters(model, device),
        }

        transforms = T.Compose([
            T.Resize((640, 640)), 
            T.ToTensor()
        ])
        
        dataset = CocoDetection(root=str(data_dir), annFile=str(ann_file), transform=transforms)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        coco_gt = COCO(str(ann_file))

        print(f"\nRunning End-to-End TTNN Eval on {len(dataset)} images...")
        
        with torch.no_grad():
            for i, (img, _) in enumerate(tqdm(loader)):
                img_id = dataset.ids[i]
                
                # --- SKIP IF ALREADY PROCESSED ---
                if int(img_id) in processed_img_ids:
                    continue
                
                # Execute Full TTNN Pipeline
                logits, boxes, tt_tensors = run_ttnn_rtdetr_inference(img, model, tt_params, device)

                # Process Outputs
                logits = logits[0].sigmoid()
                boxes = boxes[0]

                img_info = coco_gt.loadImgs(int(img_id))[0]
                orig_h, orig_w = img_info["height"], img_info["width"]

                scores, labels = torch.max(logits, dim=-1)
                
                # COCO generally evaluates everything above a very low threshold
                keep = scores > 0.05

                # Force everything to primitive CPU numpy/lists immediately
                scores_cpu = scores[keep].cpu().tolist()
                labels_cpu = labels[keep].cpu().tolist()
                boxes_cpu  = boxes[keep].cpu().tolist()

                for score, label, box in zip(scores_cpu, labels_cpu, boxes_cpu):
                    cx, cy, w, h = box
                    x1 = (cx - w / 2) * orig_w
                    y1 = (cy - h / 2) * orig_h
                    w_abs = w * orig_w
                    h_abs = h * orig_h
                    
                    results.append(
                        {
                            "image_id": int(img_id),
                            "category_id": COCO_CLASS_IDS[int(label)], 
                            "bbox": [x1, y1, w_abs, h_abs],            
                            "score": score,
                        }
                    )
                
                # ==========================================
                # AGGRESSIVE GARBAGE COLLECTION
                # ==========================================
                for t in tt_tensors:
                    if t is not None:
                        ttnn.deallocate(t)
                del tt_tensors
                
                # Nuke massive host PyTorch objects
                del img, logits, boxes, scores, labels
                del scores_cpu, labels_cpu, boxes_cpu
                
                # Force PyBind & Python to sweep RAM
                gc.collect()

                processed_img_ids.add(int(img_id))
                if len(processed_img_ids) % 25 == 0:
                    with open(res_file, "w") as f:
                        json.dump(results, f)

        with open(res_file, "w") as f:
            json.dump(results, f)

        print("\nCalculating metrics...")
        coco_dt = coco_gt.loadRes(str(res_file))
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    finally:
        ttnn.close_mesh_device(device)

if __name__ == "__main__":
    main()