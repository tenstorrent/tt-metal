from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from resnetBlock import ResNet

from BasicBlock import BasicBlock
from Bottleneck import Bottleneck

from typing import Type, Union, List

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    state_dict,
    device = None,
    host = None,
    base_address = ""
) -> ResNet:
    with torch.no_grad():
        model = ResNet(block, layers,
                        device=device,
                        host=host,
                        state_dict=state_dict,
                        base_address=base_address)
    return model
