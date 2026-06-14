"""ZAYA1-8B configuration constants + safetensors weight access.

Ground-truth values resolved in Phase 0 (see ../PORTING_SPEC.md). NOTE the
published config.json has two skews vs the fork modeling code; the values here
are the CORRECT runtime values (num_attention_heads=16, per-layer mlp_expansion).
"""
import glob
import json
import os

import torch


class ZayaConfig:
    dim = 2048
    n_layers = 80                  # strict alternation: even=ATT, odd=MoE
    vocab_size = 262272
    norm_eps = 1e-5

    # attention / CCA
    n_heads = 16                   # full head count (config.json's 8 is a skew)
    head_dim = 128
    n_kv_heads = 2
    cca_q_heads = 8                # compressed query heads
    rotary_dim = 64                # partial_rotary_factor 0.5 * head_dim
    rope_theta = 5_000_000.0
    cca_conv_kernel = 2            # cca_time0 == cca_time1 == 2
    cca_in_out_ch = cca_q_heads * head_dim + n_kv_heads * head_dim  # 1024+256=1280

    # MoE
    n_experts = 16
    router_topk = 1
    mlp_expansion = 256            # router latent dim D
    ffn_hidden_size = 4096         # swiglu fc1 out (glu halves to 2048)
    use_mod = True
    use_eda = True
    skip_expert_idx = n_experts    # MoD skip expert = index 16 (E=17 logits)

    residual_in_fp32 = True
    scale_residual_merge = True
    tie_word_embeddings = True

    @staticmethod
    def is_attention_layer(layer_idx: int) -> bool:
        return layer_idx % 2 == 0


def find_snapshot() -> str:
    """Locate the downloaded ZAYA1-8B snapshot dir inside the mounted HF cache."""
    roots = [
        os.environ.get("HF_HUB_CACHE", ""),
        os.environ.get("HF_HOME", ""),
        "/home/yito/work/hf_cache",
        os.path.expanduser("~/.cache/huggingface"),
    ]
    for r in roots:
        if not r:
            continue
        hits = glob.glob(os.path.join(r, "**/models--Zyphra--ZAYA1-8B/snapshots/*"), recursive=True)
        hits = [h for h in hits if os.path.isfile(os.path.join(h, "model.safetensors.index.json"))]
        if hits:
            return hits[0]
    raise FileNotFoundError("ZAYA1-8B snapshot not found in HF cache")


class ZayaWeights:
    """Lazy per-tensor access to the sharded safetensors checkpoint (torch tensors)."""

    def __init__(self, snapshot_dir: str = None):
        self.dir = snapshot_dir or find_snapshot()
        idx = json.load(open(os.path.join(self.dir, "model.safetensors.index.json")))
        self.weight_map = idx["weight_map"]
        self._handles = {}

    def _handle(self, shard):
        if shard not in self._handles:
            from safetensors import safe_open
            self._handles[shard] = safe_open(os.path.join(self.dir, shard), framework="pt")
        return self._handles[shard]

    def __contains__(self, name):
        return name in self.weight_map

    def get(self, name: str, dtype=torch.float32) -> torch.Tensor:
        t = self._handle(self.weight_map[name]).get_tensor(name)
        return t.to(dtype) if dtype is not None else t

    # convenience prefixes
    def att(self, layer, leaf):
        return self.get(f"model.layers.{layer}.{leaf}")

    def embed(self):
        return self.get("model.embed_tokens.weight")

    def final_norm(self):
        return self.get("model.final_norm.weight")
