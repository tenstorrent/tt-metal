from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
from common import ImageNet

def prep_ImageNet(batch_size=64):
    root = "/mnt/MLPerf/pytorch_weka_data/imagenet/dataset/ILSVRC/Data/CLS-LOC"
    imagenet = ImageNet(root)

    dataloader = imagenet.get_dataset_loader(
        batch_size=batch_size, drop_last=True)

    return dataloader
