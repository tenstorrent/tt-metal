import numpy as np
import torch

# Load reference
x_ref   = np.load("patchtsmixer_ref/ref_input_past_values.npy")  # (B, L, C)
y_ref   = np.load("patchtsmixer_ref/ref_predictions.npy")        # (B, H, C)

x_ref_t = torch.tensor(x_ref, dtype=torch.float32)

# Run your TTNN model:
y_ttnn = run_ttnn_patchtsmixer(x_ref_t)  # (B, H, C) from hardware

# Compare
mse = torch.mean((y_ttnn - torch.tensor(y_ref))**2).item()
mae = torch.mean(torch.abs(y_ttnn - torch.tensor(y_ref))).item()
corr = torch.corrcoef(
    torch.stack([y_ttnn.flatten(), torch.tensor(y_ref).flatten()])
)[0, 1].item()

print("MSE:", mse)
print("MAE:", mae)
print("corr:", corr)
