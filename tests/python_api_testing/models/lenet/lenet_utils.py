import torch
import torchvision
import torchvision.transforms as transforms
import tt_lib
from lenet.reference.lenet import LeNet5
from PIL import Image


def torch_to_tt_tensor(x: torch.Tensor) -> tt_lib.tensor.Tensor:
    return tt_lib.tensor.Tensor(
        x.reshape(-1).tolist(),
        x.shape,
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    )


def tt_tensor_to_torch(
    x: tt_lib.tensor.Tensor, host: tt_lib.device.Host
) -> torch.Tensor:
    x = x.to(host)
    return torch.Tensor(x.data()).reshape(x.shape())


def load_torch_lenet(weka_path, num_classes):
    model2 = LeNet5(num_classes).to("cpu")
    checkpoint = torch.load(weka_path, map_location=torch.device("cpu"))
    model2.load_state_dict(checkpoint["model_state_dict"])
    model2.eval()
    return model2, checkpoint["model_state_dict"]


def prepare_image(image: Image) -> torch.Tensor:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1325,), std=(0.3105,)),
        ]
    )(image).unsqueeze(0)


def prep_data(batch_size):
    # train_dataset = torchvision.datasets.MNIST(root = './data',
    #                                         train = True,
    #                                         transform = transforms.Compose([
    #                                                 transforms.Resize((32,32)),
    #                                                 transforms.ToTensor(),
    #                                                 transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
    #                                         download = True)

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1325,), std=(0.3105,)),
            ]
        ),
        download=True,
    )

    # train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
    #                                         batch_size = batch_size,
    #                                         shuffle = True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )

    return test_dataset, test_loader
