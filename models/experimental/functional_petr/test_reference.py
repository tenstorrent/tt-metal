import torch
import gc
from models.experimental.functional_petr.reference.petr import PETR
import torch.onnx
from models.experimental.functional_petr.reference.utils import LiDARInstance3DBoxes


def test_reference():
    device = torch.device("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    inputs = torch.load(
        "models/experimental/functional_petr/resources/golden_input_inputs_sample1.pt", weights_only=False
    )
    modified_batch_img_metas = torch.load(
        "models/experimental/functional_petr/resources/modified_input_batch_img_metas_sample1.pt", weights_only=False
    )
    print(f"Input type: {type(inputs)}")
    if isinstance(inputs, dict):
        print(f"Input keys: {inputs.keys()}")
        print(f"imgs shape: {inputs['imgs'].shape}")
        print(f"imgs dimensions: {inputs['imgs'].ndim}")
    model = PETR(use_grid_mask=True)
    weights_state_dict = torch.load(
        "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth", weights_only=False
    )["state_dict"]
    model.load_state_dict(weights_state_dict)
    model.eval()
    output = model.predict(inputs, modified_batch_img_metas)
    print("output", output)
    if output and len(output) > 0:
        print("Inference successful!")
        if "pts_bbox" in output[0]:
            pts_bbox = output[0]["pts_bbox"]
            print(f"  - Detected boxes: {pts_bbox.get('bboxes_3d', 'N/A')}")
            if "scores_3d" in pts_bbox:
                print(f"  - Scores shape: {pts_bbox['scores_3d'].shape}")
            if "labels_3d" in pts_bbox:
                print(f"  - Labels shape: {pts_bbox['labels_3d'].shape}")
    else:
        print("⚠ Warning: No detections or empty output")

    class ONNXWrapper(torch.nn.Module):
        def __init__(self, petr_model):
            super().__init__()
            self.backbone = petr_model.img_backbone.cpu()
            self.neck = petr_model.img_neck.cpu()
            self.pts_bbox_head = petr_model.pts_bbox_head.cpu()

        def forward(self, imgs):
            B, Nc, C, H, W = 1, 6, 3, 320, 800
            imgs_flat = imgs.view(-1, C, H, W)

            feats = self.backbone(imgs_flat)
            if not isinstance(feats, (list, tuple)):
                feats = [feats]

            fpn_feats = self.neck(feats)
            if isinstance(fpn_feats, (list, tuple)):
                fpn_feats = fpn_feats[0]

            # Reshape for detection head
            _, Cf, Hf, Wf = fpn_feats.shape
            fpn_feats = fpn_feats.view(B, Nc, Cf, Hf, Wf)

            # Create minimal metadata for detection head
            img_metas = []
            for i in range(B):
                img_metas.append(
                    {
                        "img_shape": [(H, W) for _ in range(Nc)],
                        "pad_shape": (H, W),
                        "ori_shape": (H, W, C),
                        "batch_input_shape": (H, W),
                        "cam2img": [torch.randn(3, 3, device=device) for _ in range(Nc)],
                        "lidar2cam": [torch.randn(4, 4, device=device) for _ in range(Nc)],
                        "lidar2img": [torch.randn(4, 4, device=device) for _ in range(Nc)],
                        "cam2lidar": [torch.randn(4, 4, device=device) for _ in range(Nc)],
                        "ego2global": torch.randn(4, 4, device=device),
                        "img_timestamp": [0.0] * Nc,
                        "img_aug_matrix": [torch.eye(4, device=device) for _ in range(Nc)],
                        "box_type_3d": LiDARInstance3DBoxes,
                        "scale_factor": 1.0,
                        "flip": False,
                        "lidar2ego": torch.eye(4, device=device),
                        "can_bus": torch.randn(18, device=device),
                    }
                )

            # Detection head
            outs = self.pts_bbox_head([fpn_feats], img_metas)

            if isinstance(outs, dict):
                if "all_cls_scores" in outs and "all_bbox_preds" in outs:
                    return outs["all_cls_scores"], outs["all_bbox_preds"]
                elif "cls_scores" in outs and "bbox_preds" in outs:
                    return outs["cls_scores"], outs["bbox_preds"]

            return fpn_feats, fpn_feats

            # return fpn_feats

    print("\n=== ONNX Export on CPU ===")
    wrapper = ONNXWrapper(model)
    wrapper.eval()
    # wrapper = wrapper.to(device)  # Ensure on CPU

    # Use CPU tensor for export
    input_tensor = inputs["imgs"].cpu()

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                input_tensor,
                "petr_full_model.onnx",
                input_names=["multi_camera_images"],
                # output_names=["features"],
                # opset_version=11,
                # do_constant_folding=False,  # Save memory
                # verbose=False,
                # export_params=True,
                output_names=["cls_scores", "bbox_preds"],
                dynamic_axes={
                    "multi_camera_images": {0: "batch_size"},
                    "cls_scores": {0: "batch_size"},
                    "bbox_preds": {0: "batch_size"},
                },
                opset_version=11,
                do_constant_folding=False,  # Save memory
                verbose=False,
                export_params=True,
            )
        print("Model exported to petr_full_model.onnx")

    except Exception as e:
        print(f"ONNX export failed: {e}")
