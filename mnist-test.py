import torch
import torch.nn as nn
import ttnn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


# --- Assume this is the PyTorch model class you defined and trained ---
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        input_size = 28 * 28
        hidden_size1 = 512
        hidden_size2 = 256
        hidden_size3 = 128
        num_classes = 10

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


# --- Step 1: Initialize the Tenstorrent Device ---
# This opens a connection to the first available Tenstorrent device.
device_id = 0
device = ttnn.open_device(device_id=device_id)
print("✅ Tenstorrent device initialized.")


# --- Step 2: Load Your Trained PyTorch Model on the Host ---
# Instantiate the PyTorch model structure
torch_model = MLP()

# Load the saved weights from your .pth file
model_path = "/home/ebanerjee/tt-metal/mnist_mlp_model_final.pth"  # IMPORTANT: Change this to your file path
torch_model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode. This is crucial as it disables dropout.
# For inference, dropout should always be turned off.
torch_model.eval()
print(f"✅ PyTorch model loaded from '{model_path}' and set to eval mode.")


# --- Step 3: Define the TTNN Equivalent of Your Model ---
class TTNN_MLP:
    def __init__(self, torch_model, device):
        self.device = device

        # Extract weights and biases from the torch model
        # and convert them to TTNN tensors on the Tenstorrent device.
        # Note: Weights for ttnn.linear must be transposed.

        # Layer 1
        self.W1 = ttnn.from_torch(torch_model.fc1.weight.T, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        self.b1 = ttnn.from_torch(torch_model.fc1.bias, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        # Layer 2
        self.W2 = ttnn.from_torch(torch_model.fc2.weight.T, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        self.b2 = ttnn.from_torch(torch_model.fc2.bias, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        # Layer 3
        self.W3 = ttnn.from_torch(torch_model.fc3.weight.T, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        self.b3 = ttnn.from_torch(torch_model.fc3.bias, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        # Output Layer 4
        self.W4 = ttnn.from_torch(torch_model.fc4.weight.T, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        self.b4 = ttnn.from_torch(torch_model.fc4.bias, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

        print("✅ Weights and biases extracted and moved to Tenstorrent device.")

    def __call__(self, x):
        # This is the forward pass using TTNN operations.
        # Note: Dropout is omitted because we are in inference mode.

        # Layer 1: Linear -> ReLU
        x = ttnn.linear(x, self.W1, bias=self.b1)
        x = ttnn.relu(x)

        # Layer 2: Linear -> ReLU
        x = ttnn.linear(x, self.W2, bias=self.b2)
        x = ttnn.relu(x)

        # Layer 3: Linear -> ReLU
        x = ttnn.linear(x, self.W3, bias=self.b3)
        x = ttnn.relu(x)

        # Layer 4: Output
        x = ttnn.linear(x, self.W4, bias=self.b4)

        return x


class MNISTDataset(Dataset):
    """
    Custom Dataset for MNIST CSV data.
    Modified to accept torchvision transforms for data augmentation.
    """

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # The first column is the label
        label = self.dataframe.iloc[idx, 0]

        # The rest are pixel values. Reshape to a 28x28 numpy array.
        # Ensure dtype is uint8, which is what PIL expects for grayscale images.
        image_np = self.dataframe.iloc[idx, 1:].values.astype(np.uint8).reshape(28, 28)

        # Convert numpy array to a PIL Image
        image_pil = Image.fromarray(image_np)

        # Apply transformations if they are provided
        if self.transform:
            image = self.transform(image_pil)
        else:
            # If no transform is provided, just convert to tensor.
            # ToTensor() also normalizes pixels from [0, 255] to [0.0, 1.0]
            image = transforms.ToTensor()(image_pil)

        # Ensure the label is a LongTensor as expected by CrossEntropyLoss
        return image, torch.tensor(label, dtype=torch.long)


try:
    test_df = pd.read_csv("/home/ebanerjee/tt-metal/mnist_test.csv")
    print("Successfully loaded test.csv")
except FileNotFoundError:
    print("Error: Could not find test.csv.")
    exit()

test_dataset = MNISTDataset(test_df)
batch_size = 1


test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# --- Step 4: Instantiate the TTNN Model ---
ttnn_model = TTNN_MLP(torch_model, device)
print("✅ TTNN model is ready for inference.")

# --- Step 5: Run Inference on the Tenstorrent Device ---
# Create a random sample input tensor (e.g., one 28x28 image)
# The input must have a batch size dimension.


correct = 0
total = 0

# Convert the torch tensor to a ttnn tensor and move it to the device
# The layout must match what the operation expects (TILE_LAYOUT for linear).

for images, labels in test_loader:
    single_image_tensor = images.view(images.shape[0], -1)
    single_label_tensor = labels.view(labels.shape[0], -1)

    ttnn_input = ttnn.from_torch(single_image_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output_row_major = ttnn.to_layout(ttnn_output, ttnn.ROW_MAJOR_LAYOUT)

    ttnn_argmax = ttnn.argmax(ttnn_output_row_major)

    torch_argmax = ttnn.to_torch(ttnn_argmax)

    predicted_label = torch_argmax.item()

    if predicted_label == single_label_tensor.item():
        correct += 1
    total += 1

    print("\n\n\n--- Inference Results ---")
    print(f"The predicted label for the {total}th image is {predicted_label}")
    print(f"The actual label for the {total}th image is {single_label_tensor.item()}")
    print("Input shape:", single_image_tensor.shape)
    print("Output shape:", ttnn_output.shape)
    print("Output tensor (first 10 values):", ttnn_output[0, :10])

    if total > 100:
        break

accuracy = 100 * correct / total
print("\n\n\n--------------------------------\n\n\n")
print(f"\nAccuracy of the network on the test images: {accuracy:.2f} %")

print("✅ Forward pass completed.")

# --- Step 6: Close the device connection ---
ttnn.close_device(device)
print("\n✅ Tenstorrent device closed.")
