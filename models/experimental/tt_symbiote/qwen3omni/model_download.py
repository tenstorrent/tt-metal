from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    filename="model.safetensors.index.json",
)

print("Downloaded to:", path)
