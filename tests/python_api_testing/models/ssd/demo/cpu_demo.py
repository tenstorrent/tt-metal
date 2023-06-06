import torch
from loguru import logger
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


def test_cpu_demo(imagenet_sample_input, reset_seeds):
    image = imagenet_sample_input

    with torch.no_grad():
        torch_ssd = ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights
        )
        torch_ssd.eval()
        torch_output = torch_ssd(image)

        logger.info(f"CPU's predicted Output: {torch_output[0]['scores']}")
        logger.info(f"CPU's predicted Output: {torch_output[0]['labels']}")
        logger.info(f"CPU's predicted Output: {torch_output[0]['boxes']}")
