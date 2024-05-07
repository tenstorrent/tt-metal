from models.experimental.functional_yolov4.reference.downsample1 import DownSample1
from models.experimental.functional_yolov4.reference.downsample2 import DownSample2
from models.experimental.functional_yolov4.reference.downsample3 import DownSample3
from models.experimental.functional_yolov4.reference.downsample4 import DownSample4
from models.experimental.functional_yolov4.reference.downsample5 import DownSample5

import torch
import torch.nn as nn


class DownSamples(nn.Module):
    def __init__(self):
        super(DownSamples, self).__init__()
        self.downsample1 = DownSample1()
        self.downsample2 = DownSample2()
        self.downsample3 = DownSample3()
        self.downsample4 = DownSample4()
        self.downsample5 = DownSample5()

    def forward(self, input: torch.Tensor):
        output = self.downsample1(input)
        output = self.downsample2(output)
        output = self.downsample3(output)
        output = self.downsample4(output)
        output = self.downsample5(output)

        return output
