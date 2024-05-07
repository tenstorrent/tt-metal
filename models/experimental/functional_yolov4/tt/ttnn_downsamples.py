from models.experimental.functional_yolov4.tt.ttnn_downsample1 import TtDownSample1
from models.experimental.functional_yolov4.tt.ttnn_downsample2 import TtDownSample2
from models.experimental.functional_yolov4.tt.ttnn_downsample3 import TtDownSample3
from models.experimental.functional_yolov4.tt.ttnn_downsample4 import TtDownSample4
from models.experimental.functional_yolov4.tt.ttnn_downsample5 import TtDownSample5

import ttnn


class TtDownSamples:
    def __init__(self, parameters) -> None:
        self.downsample1 = TtDownSample1(parameters["downsample1"])
        self.downsample2 = TtDownSample2(parameters["downsample2"])
        self.downsample3 = TtDownSample3(parameters["downsample3"])
        self.downsample4 = TtDownSample4(parameters["downsample4"])
        self.downsample5 = TtDownSample5(parameters["downsample5"])

    def __call__(self, device, input_tensor):
        output_tensor = self.downsample1(device, input_tensor)
        output_tensor = self.downsample2(device, output_tensor)
        output_tensor = self.downsample3(device, output_tensor)
        output_tensor = self.downsample4(device, output_tensor)
        output_tensor = self.downsample5(device, output_tensor)

        return ttnn.from_device(output_tensor)
