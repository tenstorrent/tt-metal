import os
from transformers import AutoModel

model_name = os.environ.get("MODEL_NAME", "falcon7b")
cache_dir = '/mnt/MLPerf/tt-dnn_models/hf_cache'

print(f"Downloading model: {model_name} to {cache_dir}")
AutoModel.from_pretrained(model_name, cache_dir=cache_dir)