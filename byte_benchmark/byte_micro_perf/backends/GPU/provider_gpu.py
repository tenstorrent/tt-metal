import importlib.metadata
import traceback

GPU_PROVIDER = {}


# https://github.com/Dao-AILab/flash-attention
try:
    GPU_PROVIDER["flash_attn_v2"] = {"flash_attn_v2": importlib.metadata.version("flash_attn")}
except:
    pass


# https://github.com/Dao-AILab/flash-attention
try:
    GPU_PROVIDER["flash_attn_v3"] = {
        "flash_attn_v3": importlib.metadata.version("flash_attn"),
    }
except:
    pass


# https://github.com/vllm-project/vllm
try:
    GPU_PROVIDER["vllm"] = {
        "vllm": importlib.metadata.version("vllm"),
    }
except:
    pass


# https://github.com/flashinfer-ai/flashinfer
try:
    GPU_PROVIDER["flashinfer"] = {
        "flashinfer": importlib.metadata.version("flashinfer-python"),
    }
except:
    pass
