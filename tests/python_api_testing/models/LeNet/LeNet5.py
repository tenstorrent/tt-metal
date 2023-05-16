from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from libs import tt_lib as ttl
from python_api_testing.fused_ops.linear import Linear as TtLinear
from python_api_testing.fused_ops.conv import conv as TtConv
from libs.tt_lib.utils import pad_weight
from python_api_testing.models.conv_on_device_utils import can_run_conv_on_device, run_conv_on_tt_device


batch_size = 64
num_classes = 10


def load_torch_LeNet(weka_path):

    model2 = LeNet5(num_classes).to('cpu')
    checkpoint = torch.load(weka_path, map_location=torch.device('cpu'))
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2.eval()
    return model2, checkpoint['model_state_dict']

def prep_data():
    train_dataset = torchvision.datasets.MNIST(root = './data',
                                            train = True,
                                            transform = transforms.Compose([
                                                    transforms.Resize((32,32)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
                                            download = True)

    test_dataset = torchvision.datasets.MNIST(root = './data',
                                            train = False,
                                            transform = transforms.Compose([
                                                    transforms.Resize((32,32)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),
                                            download=True)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)

    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)

    return test_dataset, test_loader


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


class TtLeNet5(nn.Module):
    def __init__(self, num_classes, device, host, state_dict):
        super().__init__()
        self.device = device
        self.host = host


        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        conv1_weight = state_dict["layer1.0.weight"]
        conv1_bias = state_dict["layer1.0.bias"].tolist()
        self.conv1_params = [6, 1, 5, 5, 1, 1, 0, 0, 1, 1]
        self.conv1_on_tt = TtConv(conv1_weight.reshape(-1).tolist(), self.conv1_params, self.device, conv1_bias)

        self.batch_norm1 = nn.BatchNorm2d(6)
        self.relu1 = ttl.tensor.relu
        self.maxp1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        conv2_weight = state_dict["layer2.0.weight"]
        conv2_bias = state_dict["layer2.0.bias"].tolist()
        self.conv2_params = [16, 6, 5, 5, 1, 1, 0, 0, 1, 1]
        self.conv2_on_tt = TtConv(conv2_weight.reshape(-1).tolist(), self.conv2_params, self.device, conv2_bias)

        self.batch_norm2 = nn.BatchNorm2d(16)
        self.relu2 = ttl.tensor.relu
        self.maxp2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv1.weight = nn.Parameter(state_dict["layer1.0.weight"])
        self.conv1.bias = nn.Parameter(state_dict["layer1.0.bias"])

        self.batch_norm1.weight = nn.Parameter(state_dict["layer1.1.weight"])
        self.batch_norm1.bias = nn.Parameter(state_dict["layer1.1.bias"])
        self.batch_norm1.running_mean = nn.Parameter(state_dict["layer1.1.running_mean"])
        self.batch_norm1.running_var = nn.Parameter(state_dict["layer1.1.running_var"])
        self.batch_norm1.num_batches_tracked = nn.Parameter(state_dict["layer1.1.num_batches_tracked"], requires_grad=False)

        self.conv2.weight = nn.Parameter(state_dict["layer2.0.weight"])
        self.conv2.bias = nn.Parameter(state_dict["layer2.0.bias"])

        self.batch_norm2.weight = nn.Parameter(state_dict["layer2.1.weight"])
        self.batch_norm2.bias = nn.Parameter(state_dict["layer2.1.bias"])
        self.batch_norm2.running_mean = nn.Parameter(state_dict["layer2.1.running_mean"])
        self.batch_norm2.running_var = nn.Parameter(state_dict["layer2.1.running_var"])
        self.batch_norm2.num_batches_tracked = nn.Parameter(state_dict["layer2.1.num_batches_tracked"], requires_grad=False)

        fc_weights = pad_weight(state_dict[f"fc.weight"])
        fc_weights = ttl.tensor.Tensor(fc_weights.reshape(-1).tolist(), fc_weights.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        fc_bias = pad_weight(state_dict[f"fc.bias"])
        fc_bias = ttl.tensor.Tensor(fc_bias.reshape(-1).tolist(), fc_bias.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        self.fc = TtLinear(416, 128, fc_weights, fc_bias, self.device) # 400, 120
        self.final_relu = ttl.tensor.relu

        fc1_weights = pad_weight(state_dict[f"fc1.weight"])
        fc1_weights = ttl.tensor.Tensor(fc1_weights.reshape(-1).tolist(), fc1_weights.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        fc1_bias = pad_weight(state_dict[f"fc1.bias"])
        fc1_bias = ttl.tensor.Tensor(fc1_bias.reshape(-1).tolist(), fc1_bias.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()
        self.fc1 = TtLinear(128, 96, fc1_weights, fc1_bias, self.device) # 120, 84

        self.final_relu1 = ttl.tensor.relu

        fc2_weights = pad_weight(state_dict[f"fc2.weight"])
        fc2_weights = ttl.tensor.Tensor(fc2_weights.reshape(-1).tolist(), fc2_weights.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        fc2_bias = pad_weight(state_dict[f"fc2.bias"])
        fc2_bias = ttl.tensor.Tensor(fc2_bias.reshape(-1).tolist(), fc2_bias.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        self.fc2 = TtLinear(96, 32, fc2_weights, fc2_bias, self.device) # 84, num_classes



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        #assert batch_size == 1
        # Layer1
        if(False and can_run_conv_on_device(list(x.size()), self.conv1_params)):
            print("Conv on tt device.")
            out = run_conv_on_tt_device(x, self.conv1_on_tt, self.conv1_params, self.device, self.host)
        else:
            print("Conv on CPU.")
            out = self.conv1(x) # HOST
        out = self.batch_norm1(out) # HOST
        # PAD
        tt_tensor = ttl.tensor.Tensor(
        out.reshape(-1).tolist(),
        out.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        )
        tt_tensor = tt_tensor.pad((batch_size, 6, 32, 32), (0, 0, 0, 0), 0)
        out = tt_tensor.to(ttl.tensor.Layout.TILE).to(self.device)

        out = self.relu1(out)
        # UNPAD
        out = out.to(self.host).to(ttl.tensor.Layout.ROW_MAJOR)
        out = out.unpad((0, 0, 0, 0), (batch_size - 1, 5, 27, 27))
        out  = torch.Tensor(out.data()).reshape(out.shape())

        out = self.maxp1(out) # HOST
        # Layer2
        if(False and can_run_conv_on_device(list(out.size()), self.conv2_params)):
            print("Conv on tt device.")
            out = run_conv_on_tt_device(out, self.conv2_on_tt, self.conv2_params, self.device, self.host)
        else:
            print("Conv on CPU.")
            out = self.conv2(out) # HOST
        out = self.batch_norm2(out) # HOST
        #PAD
        tt_tensor = ttl.tensor.Tensor(
        out.reshape(-1).tolist(),
        out.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        )
        out = tt_tensor.pad((batch_size, 16, 32, 32), (0, 0, 0, 0), 0)
        out = out.to(ttl.tensor.Layout.TILE).to(self.device)

        out = self.relu2(out)

        out = out.to(self.host).to(ttl.tensor.Layout.ROW_MAJOR)
        out = out.unpad((0, 0, 0, 0), (batch_size - 1, 15, 9, 9))
        out  = torch.Tensor(out.data()).reshape(out.shape())

        out = self.maxp2(out) # HOST
        # end of Layer 2
        out = out.reshape(out.size(0), 1, 1, -1) # madofied to have 4 dims
        # PAD
        tt_tensor = ttl.tensor.Tensor(
        out.reshape(-1).tolist(),
        out.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        )
        out = tt_tensor.pad((batch_size, 1, 32, 416), (0, 0, 0, 0), 0)
        out = out.to(ttl.tensor.Layout.TILE).to(self.device)

        out = self.fc(out)
        out = self.relu2(out)
        out = self.fc1(out)
        out = self.relu2(out)
        out = self.fc2(out)
        # UNPAD
        out = out.to(self.host).to(ttl.tensor.Layout.ROW_MAJOR)
        out = out.unpad((0, 0, 0, 0), (batch_size - 1, 0, 0, 9))
        out  = torch.Tensor(out.data()).reshape(out.shape())
        return out
