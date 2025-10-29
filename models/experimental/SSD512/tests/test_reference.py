# Create file: export_onnx.py
import torch
import torch.nn as nn
from models.experimental.SSD512.reference.ssd import build_ssd


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# Load trained model
net = build_ssd("test", 512, 21)
# net.load_weights('models/experimental/SSD512/reference/weights/vgg16_reducedfc.pth')
net.apply(weights_init)
net.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 512, 512)

# Export to ONNX
torch.onnx.export(
    net, dummy_input, "ssd512.onnx", input_names=["input"], output_names=["output"], opset_version=12, verbose=True
)
output = net(dummy_input)
print(output)

print("ONNX model exported as ssd512.onnx")
