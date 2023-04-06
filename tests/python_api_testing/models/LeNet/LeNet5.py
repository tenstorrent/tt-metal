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
import torchvision
import torchvision.transforms as transforms
from utils import tt2torch_tensor, torch2tt_tensor

from libs import tt_lib as ttl
from python_api_testing.fused_ops.linear import Linear as TtLinear
from libs.tt_lib.utils import pad_activation, pad_weight, print_diff_argmax


# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 10



#Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes):
#         super(ConvNeuralNet, self).__init__()
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0), # 0
            nn.BatchNorm2d(6),                                   # 1
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
        self.conv1.weight = nn.Parameter(state_dict["layer1.0.weight"])
        self.conv1.bias = nn.Parameter(state_dict["layer1.0.bias"])

        self.batch_norm1 = nn.BatchNorm2d(6)
        self.batch_norm1.weight = nn.Parameter(state_dict["layer1.1.weight"])
        self.batch_norm1.bias = nn.Parameter(state_dict["layer1.1.bias"])
        self.batch_norm1.running_mean = nn.Parameter(state_dict["layer1.1.running_mean"])
        self.batch_norm1.running_var = nn.Parameter(state_dict["layer1.1.running_var"])
        self.batch_norm1.num_batches_tracked = nn.Parameter(state_dict["layer1.1.num_batches_tracked"], requires_grad=False)

        self.relu1 = ttl.tensor.relu
        self.maxp1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.conv2.weight = nn.Parameter(state_dict["layer2.0.weight"])
        self.conv2.bias = nn.Parameter(state_dict["layer2.0.bias"])

        self.batch_norm2 = nn.BatchNorm2d(16)
        self.batch_norm2.weight = nn.Parameter(state_dict["layer2.1.weight"])
        self.batch_norm2.bias = nn.Parameter(state_dict["layer2.1.bias"])
        self.batch_norm2.running_mean = nn.Parameter(state_dict["layer2.1.running_mean"])
        self.batch_norm2.running_var = nn.Parameter(state_dict["layer2.1.running_var"])
        self.batch_norm2.num_batches_tracked = nn.Parameter(state_dict["layer2.1.num_batches_tracked"], requires_grad=False)

        self.relu2 = ttl.tensor.relu
        self.maxp2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        fc_weights = pad_weight(state_dict[f"fc.weight"])
        fc_weights = ttl.tensor.Tensor(fc_weights.reshape(-1).tolist(), fc_weights.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        fc_bias = pad_weight(state_dict[f"fc.bias"])
        fc_bias = ttl.tensor.Tensor(fc_bias.reshape(-1).tolist(), fc_bias.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        self.fc = TtLinear(400, 120, fc_weights, fc_bias, self.device)
        self.final_relu = ttl.tensor.relu

        fc1_weights = pad_weight(state_dict[f"fc1.weight"])
        fc1_weights = ttl.tensor.Tensor(fc1_weights.reshape(-1).tolist(), fc1_weights.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        fc1_bias = pad_weight(state_dict[f"fc1.bias"])
        fc1_bias = ttl.tensor.Tensor(fc1_bias.reshape(-1).tolist(), fc1_bias.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        self.fc1 = TtLinear(120, 84, fc1_weights, fc1_bias, self.device)

        self.final_relu1 = ttl.tensor.relu

        fc2_weights = pad_weight(state_dict[f"fc2.weight"])
        fc2_weights = ttl.tensor.Tensor(fc2_weights.reshape(-1).tolist(), fc2_weights.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        fc2_bias = pad_weight(state_dict[f"fc2.bias"])
        fc2_bias = ttl.tensor.Tensor(fc2_bias.reshape(-1).tolist(), fc2_bias.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).data()

        self.fc2 = TtLinear(84, num_classes, fc2_weights, fc2_bias, self.device)

# def torch2tt_tensor(py_tensor: torch.Tensor, tt_device):
# def tt2torch_tensor(tt_tensor):

    def forward(self, x):
        print(x.shape())
        x = tt2torch_tensor(x)
        # Layer1
        out = self.conv1(x) # HOST
        out = self.batch_norm1(out) #HOST
        print(out.shape)
        out = torch2tt_tensor(out, self.device)

        out = self.relu1(out)

        out = tt2torch_tensor(out)
        out = self.maxp1(out) # HOST
        # Layer2
        out = self.conv2(out) # HOST
        out = self.batch_norm2(out) # HOST
        out = torch2tt_tensor(out, self.device)

        out = self.relu2(out)

        out = tt2torch_tensor(out)
        out = self.maxp2(out) # HOST
        # end of Layer 2
        out = out.reshape(out.size(0), 1, 1, -1) # madofied to have 4 dims
        out = torch2tt_tensor(out, self.device)

        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

def load_torch_LeNet():

    PATH = 'tests/python_api_testing/models/LeNet/model.pt'
    model2 = LeNet5(num_classes).to('cpu')
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2.eval()
    return model2, checkpoint['model_state_dict']

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

        torch_LeNet, state_dict = load_torch_LeNet()
        test_dataset, test_loader = prep_data()

        TTLeNet = TtLeNet5(num_classes, device, host, state_dict)

        for image, labels in test_loader:
            img = image[0, :, :, :].unsqueeze(0).to('cpu')
            label = labels[0]
            # print(img.shape, label.shape)
            torch_output = torch_LeNet(img)
            _, torch_predicted = torch.max(torch_output.data, 1)

            tt_img = torch2tt_tensor(img, device)
            tt_output = TTLeNet(tt_img)
            tt_output = tt2torch_tensor(tt_output)
            _, tt_predicted = torch.max(tt_output.data, 1)


            print(f"Torch Predicted: {torch_predicted} \n   TT Predicted: {tt_predicted} \n        Labels: {labels}")
            break







    ttl.device.CloseDevice(device)




# snippet for visualization

# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)

# import matplotlib.pyplot as plt

# model2 = LeNet5(num_classes).to(device)
# checkpoint = torch.load(PATH)
# model2.load_state_dict(checkpoint['model_state_dict'])
# model2.eval()

# fig = plt.figure()
# for i in range(1):
#     plt.subplot(2,3,i+1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     print(example_data[i].shape)
#     output = model2(example_data[i].unsqueeze(0).to(device))
#     _, predicted = torch.max(output.data, 1)

#     plt.title("Truth: {} Model {}".format(example_targets[i], predicted))
#     plt.xticks([])
#     plt.yticks([])
# fig
