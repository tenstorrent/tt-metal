from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from libs import tt_lib as ttl
from python_api_testing.fused_ops.linear import Linear as TtLinear
from python_api_testing.fused_ops.softmax import softmax as TtSoftmax
from libs.tt_lib.utils import pad_weight


batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 10

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout2 = nn.Dropout(0.5) # commented out since makes no different in inference
        self.fc2 = nn.Linear(2304, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=1)
        # x = self.dropout2(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)



class TtConvNet(nn.Module):
    def __init__(self, device, host, state_dict):
        super().__init__()
        self.device = device
        self.host = host

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv1.weight = nn.Parameter(state_dict["conv1.weight"])
        self.conv1.bias = nn.Parameter(state_dict["conv1.bias"])

        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv2.weight = nn.Parameter(state_dict["conv2.weight"])
        self.conv2.bias = nn.Parameter(state_dict["conv2.bias"])

        # self.dropout2 = nn.Dropout(0.5)

        weights = pad_weight(state_dict[f"fc2.weight"])
        weights = ttl.tensor.Tensor(weights.reshape(-1).tolist(), weights.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()
        bias = pad_weight(state_dict[f"fc2.bias"])
        bias = ttl.tensor.Tensor(bias.reshape(-1).tolist(), bias.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        self.fc2 = TtLinear(2304, 32, weights, bias, self.device) # 2304, num_classes


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        # PAD
        tt_tensor = ttl.tensor.Tensor(
        out.reshape(-1).tolist(),
        out.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        )
        tt_tensor = tt_tensor.pad((batch_size, out.shape[1], 32, 32), (0, 0, 0, 0), 0)
        out = tt_tensor.to(ttl.tensor.Layout.TILE).to(self.device)

        out = ttl.tensor.relu(out)
        # UNPAD
        out = out.to(self.host).to(ttl.tensor.Layout.ROW_MAJOR)
        out = out.unpad((0, 0, 0, 0), (batch_size - 1, out.shape()[1] - 1, 29, 29))
        out  = torch.Tensor(out.data()).reshape(out.shape())

        out = F.max_pool2d(out, 2)
        out = self.conv2(out)

        # PAD
        tt_tensor = ttl.tensor.Tensor(
        out.reshape(-1).tolist(),
        out.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        )
        tt_tensor = tt_tensor.pad((batch_size, out.shape[1], 32, 32), (0, 0, 0, 0), 0)
        out = tt_tensor.to(ttl.tensor.Layout.TILE).to(self.device)

        out = ttl.tensor.relu(out)
        #UNPAD
        out = out.to(self.host).to(ttl.tensor.Layout.ROW_MAJOR)
        out = out.unpad((0, 0, 0, 0), (batch_size - 1, out.shape()[1] - 1, 12, 12))
        out  = torch.Tensor(out.data()).reshape(out.shape())

        out = F.max_pool2d(out, 2)
        out = torch.flatten(out, start_dim=1).unsqueeze(1).unsqueeze(1)

        # PAD
        tt_tensor = ttl.tensor.Tensor(
        out.reshape(-1).tolist(),
        out.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        )
        tt_tensor = tt_tensor.pad((batch_size, out.shape[1], 32, out.shape[-1]), (0, 0, 0, 0), 0)
        out = tt_tensor.to(ttl.tensor.Layout.TILE).to(self.device)

        out = self.fc2(out)
        out = out.to(self.host).to(ttl.tensor.Layout.ROW_MAJOR)
        out = out.unpad((0, 0, 0, 0), (batch_size - 1, out.shape()[1] - 1, 0, num_classes - 1))
        out  = torch.Tensor(out.data()).reshape(out.shape())

        tt_tensor = ttl.tensor.Tensor(
        out.reshape(-1).tolist(),
        out.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        )
        tt_tensor = tt_tensor.pad((batch_size, out.shape[1], 32, 32), (0, 0, 0, 0), -100*1000)
        out = tt_tensor.to(ttl.tensor.Layout.TILE).to(self.device)
        out = TtSoftmax(out)

        #UNPAD
        out = out.to(self.host).to(ttl.tensor.Layout.ROW_MAJOR)
        out = out.unpad((0, 0, 0, 0), (batch_size - 1, 0, 0, 9))
        out  = torch.Tensor(out.data()).reshape(out.shape())
        return out


def load_torch():
    weka_path = '/mnt/MLPerf/tt_dnn-models/MNIST/ConvNet/mnist_ConvNet.pt'
    model = ConvNet().to('cpu')
    checkpoint = torch.load(weka_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['model_state_dict']

def prep_data():
     #Loading the dataset and preprocessing
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
