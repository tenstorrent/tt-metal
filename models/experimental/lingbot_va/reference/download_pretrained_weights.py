# download_model.py
from huggingface_hub import snapshot_download
import os

# Set the local directory where you want to save the model
local_dir = "models/experimental/lingbot_va/reference/checkpoints"

# Create directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

print(f"Downloading robbyant/lingbot-va-base to {local_dir}...")
snapshot_download(
    repo_id="robbyant/lingbot-va-base",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # Copy files instead of symlinks
    resume_download=True,  # Resume if interrupted
)
print(f"Download complete! Model saved to {local_dir}")
