import torch
import torch.nn as nn
import ttnn
import transformers
from PIL import Image
import requests
from transformers import YolosImageProcessor, YolosForObjectDetection

from models.demos.yolos.tt.model_def import yolos, custom_preprocessor

def run_demo():
    # Load model and processor
    model_name = "hustvl/yolos-small"
    image_processor = YolosImageProcessor.from_pretrained(model_name)
    torch_model = YolosForObjectDetection.from_pretrained(model_name)
    torch_model.eval()

    # Load image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Preprocess
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values
    
    # Run PyTorch Model
    with torch.no_grad():
        torch_outputs = torch_model(**inputs)

    # Initialize TTNN
    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    # Convert parameters
    parameters = ttnn.model_preprocessing.preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    # Prepare inputs for TTNN
    # NHWC and BFloat16
    pixel_values_tt = torch.permute(pixel_values, (0, 2, 3, 1))
    pixel_values_tt = ttnn.from_torch(pixel_values_tt, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    
    # Config
    config = torch_model.config

    # Run TTNN Model
    cls_logits, bbox_pred = yolos(
        config,
        pixel_values_tt,
        parameters.cls_token,
        parameters.detection_tokens,
        parameters.position_embeddings,
        parameters=parameters
    )

    # Compare shapes
    print(f"PyTorch Logits: {torch_outputs.logits.shape}")
    print(f"TTNN Logits: {cls_logits.shape}")
    print(f"PyTorch Bboxes: {torch_outputs.pred_boxes.shape}")
    print(f"TTNN Bboxes: {bbox_pred.shape}")
    
    # Close device
    ttnn.close_device(device)
    
    print("Demo execution finished!")

if __name__ == "__main__":
    run_demo()
