import os
import sys
from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import cv2
import torch
import torchvision
from loguru import logger
from datasets import load_dataset


def load_efficientnet_model():
    """
    Load the pre-trained EfficientNetB0 model.
    """
    model = torchvision.models.efficientnet_b0(pretrained=True)
    model.eval()
    return model


def preprocess():
    """
    Define the transform for the input image/frames.
    Resize, crop, convert to tensor, and apply ImageNet normalization stats.
    """
    transform =  torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    return transform


def read_classes():
    """
    Load the ImageNet class names.
    """
    with open(ROOT / "imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories


def download_images(img_path):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image.save(img_path)


def test_cpu_demo():
    img_path = ROOT / "input_image.jpg"
    download_images(img_path)

    model = load_efficientnet_model()
    categories = read_classes()
    transform = preprocess()

    image = cv2.imread(str(img_path))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Check the top 5 categories that are predicted.
    top5_prob, top5_catid = torch.topk(probabilities, 3)

    for i in range(top5_prob.size(0)):
        cv2.putText(image, f"{top5_prob[i].item()*100:.3f}%", (15, (i+1)*30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"{categories[top5_catid[i]]}", (160, (i+1)*30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        print(categories[top5_catid[i]], top5_prob[i].item())

    cv2.imwrite(str(ROOT / "out_image.jpg"), image)
