import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version PyTorch was compiled against: {torch.version.cuda}")

try:
    pass

    print("Import successful")
except Exception as e:
    print(f"Error: {e}")
