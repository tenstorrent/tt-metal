import torch
import torch.nn as nn
import ttnn


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
model_path = "your_model.pth"  # IMPORTANT: Change this to your file path
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
        self.W1 = ttnn.from_torch(torch_model.fc1.weight.T, device=device, layout=ttnn.TILE_LAYOUT)
        self.b1 = ttnn.from_torch(torch_model.fc1.bias, device=device, layout=ttnn.TILE_LAYOUT)

        # Layer 2
        self.W2 = ttnn.from_torch(torch_model.fc2.weight.T, device=device, layout=ttnn.TILE_LAYOUT)
        self.b2 = ttnn.from_torch(torch_model.fc2.bias, device=device, layout=ttnn.TILE_LAYOUT)

        # Layer 3
        self.W3 = ttnn.from_torch(torch_model.fc3.weight.T, device=device, layout=ttnn.TILE_LAYOUT)
        self.b3 = ttnn.from_torch(torch_model.fc3.bias, device=device, layout=ttnn.TILE_LAYOUT)

        # Output Layer 4
        self.W4 = ttnn.from_torch(torch_model.fc4.weight.T, device=device, layout=ttnn.TILE_LAYOUT)
        self.b4 = ttnn.from_torch(torch_model.fc4.bias, device=device, layout=ttnn.TILE_LAYOUT)

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


# --- Step 4: Instantiate the TTNN Model ---
ttnn_model = TTNN_MLP(torch_model, device)
print("✅ TTNN model is ready for inference.")

# --- Step 5: Run Inference on the Tenstorrent Device ---
# Create a random sample input tensor (e.g., one 28x28 image)
# The input must have a batch size dimension.
sample_input_torch = torch.randn(1, 28 * 28)

# Convert the torch tensor to a ttnn tensor and move it to the device
# The layout must match what the operation expects (TILE_LAYOUT for linear).
ttnn_input = ttnn.from_torch(sample_input_torch, layout=ttnn.TILE_LAYOUT, device=device)

# Run the forward pass 🚀
ttnn_output = ttnn_model(ttnn_input)

# Convert the result back to a torch tensor on the host to view it
output_torch = ttnn.to_torch(ttnn_output)

print("\n--- Inference Results ---")
print("Input shape:", sample_input_torch.shape)
print("Output shape:", output_torch.shape)
print("Output tensor (first 5 values):", output_torch[0, :5])

# --- Step 6: Close the device connection ---
ttnn.close_device(device)
print("\n✅ Tenstorrent device closed.")
