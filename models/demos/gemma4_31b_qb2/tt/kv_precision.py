"""Explicit diagnostic KV storage policy; BF8 remains default."""
import os
import ttnn

def selected_kv_dtypes():
    name=os.environ.get("GEMMA_EXPERIMENT_KV_DTYPE", "bf8kv")
    policies={"bf8kv":(ttnn.bfloat8_b,ttnn.bfloat8_b),
              "bf8k_bf4v":(ttnn.bfloat8_b,ttnn.bfloat4_b),
              "bf4kv":(ttnn.bfloat4_b,ttnn.bfloat4_b)}
    if name not in policies:
        raise ValueError("Unsupported explicit KV precision policy: "+name)
    return policies[name]

def tile_bytes(dtype):
    if dtype==ttnn.bfloat8_b:return 1088
    if dtype==ttnn.bfloat4_b:return 576
    raise ValueError("Unsupported KV cache dtype")
