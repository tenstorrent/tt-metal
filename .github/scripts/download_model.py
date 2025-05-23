from transformers import FalconConfig
from transformers.models.falcon.modeling_falcon import FalconForCausalLM

cache_dir = '/mnt/MLPerf/tt-dnn_models/hf_cache'
model_id = 'tiiuae/falcon-7b'

# Download config and modify if needed
config = FalconConfig.from_pretrained(model_id, cache_dir=cache_dir)
# Optional: customize layers, etc.
# config.num_hidden_layers = 12

# Download model weights
model = FalconForCausalLM.from_pretrained(model_id, config=config, cache_dir=cache_dir).eval()

print(f"Model {model_id} successfully downloaded to {cache_dir}")
