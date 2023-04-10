from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from utility_functions import tt2torch_tensor, torch2tt_tensor

from libs import tt_lib as ttl
from python_api_testing.fused_ops.linear import Linear as TtLinear
from python_api_testing.fused_ops.softmax import softmax as TtSoftmax
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc
from libs.tt_lib.utils import pad_activation, pad_weight, print_diff_argmax


batch_size = 1
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

        self.fc2 = nn.Linear(2304, num_classes)
        self.fc2.weight = nn.Parameter(state_dict["fc2.weight"])
        self.fc2.bias = nn.Parameter(state_dict["fc2.bias"])

        # weights = pad_weight(state_dict[f"fc2.weights"])
        # weights = ttl.tensor.Tensor(weights.reshape(-1).tolist(), weights.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()
        # bias = pad_weight(state_dict[f"fc2.bias"])
        # bias = ttl.tensor.Tensor(bias.reshape(-1).tolist(), bias.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        # self.fc2 = TtLinear(2304, num_classes, weights, bias, self.device)


    def forward(self, x):
        x = tt2torch_tensor(x)
        x = self.conv1(x)
        x = F.pad(x, (0, 2, 0, 2), 'constant', 0)
        x = torch2tt_tensor(x, self.device)

        x = ttl.tensor.relu(x)
        x = tt2torch_tensor(x)
        # unpad
        x = x[:, :, :-2, :-2]
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.pad(x, (0, 19, 0, 19), 'constant', 0)
        x = torch2tt_tensor(x, self.device)

        x = ttl.tensor.relu(x)
        x = tt2torch_tensor(x)
        x = x[:, :, :13, :13] # unpad
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, start_dim=1)

        x = self.fc2(x)
        return F.softmax(x, dim=1)

        # x = F.pad(x, (0, 22, 0, 31), 'constant', -float('inf'))
        # x = torch2tt_tensor(x, self.device)
        # x = TtSoftmax(x)
        # x = tt2torch_tensor(x, self.host)
        # return x[:, :, 1, : num_classes]


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


if __name__ == "__main__":
    with torch.no_grad():
        torch.manual_seed(1234)
        # Initialize the device

        device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
        ttl.device.InitializeDevice(device)
        host = ttl.device.GetHost()

        #######

        torch_ConvNet, state_dict = load_torch()
        test_dataset, test_loader = prep_data()

        TtConvNet = TtConvNet(device, host, state_dict)

        for image, labels in test_loader:
            img = image[0, :, :, :].unsqueeze(0).to('cpu')
            label = labels[0]
            # print(img.shape, label.shape)
            torch_output = torch_ConvNet(img)
            _, torch_predicted = torch.max(torch_output.data, 1)

            tt_img = torch2tt_tensor(img, device)
            tt_output = TtConvNet(tt_img)
            # print(tt_output, torch_output)
            # tt_output = tt2torch_tensor(tt_output)

            print(comp_allclose_and_pcc(tt_output, torch_output))
            print(tt_output.shape, tt_output[0, :10])
            print(torch_output.shape, torch_output[0, :10])
            _, tt_predicted = torch.max(tt_output.data, 1)

            # print(f"Torch Predicted: {torch_predicted} \n   TT Predicted:  \n        Labels: {labels[0]}")

            print(f"Torch Predicted: {torch_predicted} \n   TT Predicted: {tt_predicted} \n        Labels: {labels}")
            break

    ttl.device.CloseDevice(device)
