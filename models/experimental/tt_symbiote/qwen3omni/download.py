from huggingface_hub import hf_hub_download

repo_name = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
filename = "model-00001-of-00015.safetensors"
download_path = "/home/ubuntu/tt-metal/models/experimental/qwen3omni/checkpoints"  # change to your desired directory

file_path = hf_hub_download(repo_id=repo_name, filename=filename, local_dir=download_path, local_dir_use_symlinks=False)

print(f"File downloaded to: {file_path}")
