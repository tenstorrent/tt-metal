import torch
import tt_lib
import warnings

from tt_lib.fallback_ops import fallback_ops
from python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from python_api_testing.models.yolov5.tt.yolov5_conv import TtYolov5Conv


class TtYolov5SPPF(torch.nn.Module):
    # Standard bottleneck
    def __init__(self, state_dict, base_address, device, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels

        # self.cv1 = Conv(c1, c_, 1, 1)
        self.cv1 = TtYolov5Conv(
            state_dict=state_dict,
            base_address=f"{base_address}.cv1",
            device=device,
            c1=c1,
            c2=c_,
            k=1,
            s=1,
        )

        # self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.cv2 = TtYolov5Conv(
            state_dict=state_dict,
            base_address=f"{base_address}.cv2",
            device=device,
            c1=c_ * 4,
            c2=c2,
            k=1,
            s=1,
        )

        # self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.m = fallback_ops.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(fallback_ops.concat((x, y1, y2, self.m(y2)), 1))
