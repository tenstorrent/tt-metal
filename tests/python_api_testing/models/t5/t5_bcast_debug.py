from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib


def test_T5Bcast_inference(device):
    py_variance = torch.randn((1, 32, 128, 32))
    tt_variance = (
        tt_lib.tensor.Tensor(
            py_variance.reshape(-1).tolist(),
            py_variance.size(),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )
        .to(tt_lib.tensor.Layout.TILE)
        .to(device)
    )

    tt_variance_epsilon_const = (
        tt_lib.tensor.Tensor(
            [0.000001] + [0.0 for _ in range(32 * 32 - 1)],
            [1, 1, 32, 32],
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )
        .to(tt_lib.tensor.Layout.TILE)
        .to(device)
    )

    # This operation hangs
    op_add = tt_lib.tensor.bcast(
        tt_variance,
        tt_variance_epsilon_const,
        tt_lib.tensor.BcastOpMath.ADD,
        tt_lib.tensor.BcastOpDim.H,
    )


if __name__ == "__main__":
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    test_T5Bcast_inference(device)
    tt_lib.device.CloseDevice(device)
