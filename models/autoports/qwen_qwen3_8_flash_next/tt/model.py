# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen text autoregressive stack over the accepted TP4/EP4 decoder.

Setup reads one checkpoint layer at a time. Runtime preserves one complete
hyperconnection stream per rank, including between different layer kinds.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
from transformers import AutoConfig

import ttnn

from .decoder_components import rms
from .dram_projection import DramProjectionConfig, PhaseDramProjection, PhaseDramProjectionGroup
from .fused_decoder import merged_rms
from .multichip_decoder import (
    DRAM,
    Collectives,
    GatheredProjection,
    MeshExactProjection,
    MeshLoader,
    MeshPackedEmbedding,
    MeshWorkspace,
    MultichipDecoder,
)
from .precision import layer_policy, load_policy


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]

    def read(self, prefix, *, strip=False):
        keys = [k for k in self.index if k.startswith(prefix) and ".ngram_embedding.shard_" not in k]
        result = {}
        for filename in sorted({self.index[k] for k in keys}):
            with safe_open(self.path / filename, framework="pt") as source:
                for key in keys:
                    if self.index[key] == filename:
                        result[key.removeprefix(prefix) if strip else key] = source.get_tensor(key)
        return result

    def tensor(self, key):
        with safe_open(self.path / self.index[key], framework="pt") as source:
            return source.get_tensor(key)


class FinalMixer:
    """HF hyper_connection_mixer, whose use_combine=False omits injection."""

    def __init__(self, weights, mesh, ccl_dtype="bfloat16"):
        loader = MeshLoader(weights, mesh)
        self.ccl = Collectives(mesh, ccl_dtype)
        self.eps = 1e-6
        norm = weights["hc_norm.weight"].float().reshape(4, 2560) + 1
        self.norm = loader.shards([n.reshape(1, 1, 1, 2560) for n in norm], dim=3, dtype=ttnn.float32)
        self.down = GatheredProjection(
            MeshExactProjection(
                loader, list(weights["input_mix_weight_down.weight"].float().chunk(4, dim=0)), dtype=ttnn.float32
            ),
            self.ccl,
            valid_width=80,
        )
        self.up = MeshExactProjection(
            loader, list(weights["input_mix_weight_up.weight"].float().chunk(4, dim=0)), dtype=ttnn.float32, block=256
        )

    def __call__(self, residual):
        count = residual.shape[-2]
        normalized = (
            merged_rms(residual, self.norm, self.eps, rhs=True)
            if 1 <= count <= 32 and tuple(residual.shape) == (1, 1, count, 2560)
            else rms(residual, self.norm, self.eps)
        )
        down = ttnn.multiply(self.down(self.ccl.gather(normalized)), 0.25)
        mix = ttnn.typecast(ttnn.silu(ttnn.typecast(down, ttnn.float32)), ttnn.bfloat16)
        mix = ttnn.typecast(ttnn.sigmoid(ttnn.typecast(self.up(mix), ttnn.float32)), ttnn.bfloat16)
        return ttnn.typecast(ttnn.multiply(self.ccl.sum(ttnn.multiply(mix, normalized)), 0.25), ttnn.bfloat16)


@dataclass
class RequestCache:
    layers: list
    capacity: int
    position: int = 0  # Scheduler boundary metadata; device positions advance inside decoder.
    shared_kv: bool = False  # Shared serving pools move through page tables, never tensor copies.
    kv_bound: bool = True


def canonical_text_config(hf_config):
    """Translate checkpoint QSA labels without mutating scheduler-owned config."""
    config = deepcopy(getattr(hf_config, "text_config", hf_config))
    if "full_attention" in config.layer_types:
        indexer_fields = (
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        )
        if any(getattr(config, field, None) is None for field in indexer_fields):
            raise ValueError("Qwen full_attention aliases require the checkpoint's complete QSA indexer config")
        config.layer_types = ["indexed_attention" if kind == "full_attention" else kind for kind in config.layer_types]
    return config


class QwenModel:
    def __init__(self, checkpoint, mesh_device, *, packed_cache, layer_indices=None, precision=None, hf_config=None):
        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError("Qwen full model requires the accepted 1x4 mesh")
        self.mesh = mesh_device
        self.precision = load_policy(Path(__file__).resolve().parents[1], precision)
        self.residual_dtype = getattr(ttnn, self.precision["residual_dtype"])
        self.logits_dtype = getattr(ttnn, self.precision["logits_dtype"])
        source = Checkpoint(checkpoint)
        if hf_config is None:
            hf_config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        self.config = canonical_text_config(hf_config)
        self.layer_indices = tuple(range(self.config.num_hidden_layers) if layer_indices is None else layer_indices)
        print("SETUP_WORKSPACE", flush=True)
        self.workspace = MeshWorkspace(mesh_device)
        self.ccl = Collectives(mesh_device, self.precision["ccl_dtype"])
        print("SETUP_PACKED_EMBEDDING", flush=True)
        self.packed_embedding = (
            MeshPackedEmbedding.from_cache(packed_cache, mesh_device, self.ccl)
            if any(i + 1 in self.config.ple_layer_ids for i in self.layer_indices)
            else None
        )
        print("SETUP_TOKEN_EMBEDDING", flush=True)
        loader = MeshLoader({}, mesh_device)
        # Hidden-dimension sharding avoids four resident copies of the token table.
        embedding = source.tensor("model.language_model.embed_tokens.weight")
        self.embedding = loader.shards(list(embedding.chunk(4, dim=1)), dim=1, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.layers = []
        for index in self.layer_indices:
            print(f"LOADING_LAYER {index}", flush=True)
            weights = source.read(f"model.language_model.layers.{index}.", strip=True)
            self.layers.append(
                MultichipDecoder.from_state_dict(
                    weights,
                    hf_config=self.config,
                    layer_idx=index,
                    mesh_device=mesh_device,
                    packed_embedding=self.packed_embedding,
                    workspace=self.workspace,
                    precision=layer_policy(self.precision, index),
                )
            )
            print(f"LOADED_LAYER {index}", flush=True)
            del weights
        self.final_mixer = FinalMixer(
            source.read("model.language_model.hyper_connection_mixer.", strip=True),
            mesh_device,
            self.precision["ccl_dtype"],
        )
        head = embedding if self.config.tie_word_embeddings else source.tensor("lm_head.weight")
        self.compute = loader.compute
        # Prepare directly from HF partitions: never retain an additional
        # interleaved TT head beside the selected DRAM-bank-sharded weights.
        head_policy = DramProjectionConfig(
            cores=20,
            readers=3,
            block=2,
            dtype=self.precision["weight_groups"]["lm_head"],
            fidelity=self.precision["compute_fidelities"]["lm_head"],
            packed=False,
            activation_dtype=self.precision["activation_dtype"],
        )
        head_group = PhaseDramProjectionGroup(
            loader, [[part] for part in head.chunk(4, dim=0)], {"decode": head_policy, "rows": head_policy}
        )
        # Decoder projection helpers default packer accumulation on. Preserve
        # the full-model head's HiFi4/FP32 DEST/packer-off compute policy exactly.
        assert head_group.decode is head_group.rows
        head_group.decode.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, head_policy.fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.head_group = head_group
        self.lm_head = PhaseDramProjection(head_group, None)
        del head, embedding

    def precision_summary(self):
        """Read actual constructed operands/configs, not only requested JSON."""

        def compute(config):
            return {
                key: str(getattr(config, key)) if key == "math_fidelity" else getattr(config, key)
                for key in ("math_fidelity", "fp32_dest_acc_en", "packer_l1_acc", "math_approx_mode")
            }

        def projection(group):
            return {
                phase: {
                    "weight_dtypes": [str(w.dtype) for w in getattr(group, phase).weights],
                    "weight_shapes": [list(w.shape) for w in getattr(group, phase).weights],
                    "compute": compute(getattr(group, phase).compute),
                    "activation_dtype": getattr(group, phase).config.activation_dtype,
                    "output_dtype": "bfloat16",
                }
                for phase in ("decode", "rows")
            }

        layers = []
        for index, layer in zip(self.layer_indices, self.layers):
            layers.append(
                {
                    "layer": index,
                    "groups": {
                        name: [projection(g) for g in groups] for name, groups in layer.precision_groups.items()
                    },
                    "experts": {
                        "weights": {
                            name: sorted({str(w.dtype) for w in weights}) for name, weights in layer.mlp.experts.items()
                        },
                        "compute": compute(layer.mlp.expert_compute),
                        "active_experts_per_token": 10,
                        "local_experts": 128,
                        "execution": "routed moe_fused_swiglu",
                    },
                    "shared_down": {
                        "dtype": str(layer.mlp.shared_down.dtype),
                        "compute": compute(layer.mlp.shared_down_compute),
                    },
                    "ccl_dtype": str(layer.ccl.dtype),
                    "kv_dtype": str(layer.kv_dtype),
                }
            )
        return {
            "requested": self.precision,
            "layers": layers,
            "lm_head": projection(self.head_group),
            "top_level_ccl_dtype": str(self.ccl.dtype),
            "final_mixer_ccl_dtype": str(self.final_mixer.ccl.dtype),
            "fixed_contract_validation": "load_policy rejects unsupported residual/logits/sampling/fixed-numerics changes",
        }

    def create_cache(self, *, capacity=None, page_table=None, defer_kv_cache=False):
        capacity = self.config.max_position_embeddings if capacity is None else capacity
        if not 1 <= capacity <= self.config.max_position_embeddings:
            raise ValueError("Invalid context capacity")
        physical = ((capacity + 127) // 128) * 128
        return RequestCache(
            [
                layer.create_state(capacity=physical, page_table=page_table, defer_kv_cache=defer_kv_cache)
                for layer in self.layers
            ],
            capacity,
            kv_bound=not defer_kv_cache,
        )

    def embed(self, tokens):
        local = ttnn.embedding(tokens, self.embedding, layout=ttnn.TILE_LAYOUT, memory_config=DRAM)
        full = self.ccl.gather(local)
        return ttnn.reshape(full, (1, 1, tokens.shape[-1], self.config.hidden_size))

    def logits(self, residual):
        if residual.dtype != self.residual_dtype:
            raise ValueError("Residual stream violates the constructed precision policy")
        output = self.lm_head(self.final_mixer(residual))
        if output.dtype != self.logits_dtype:
            raise ValueError("Logits violate the constructed precision policy")
        return output

    def forward_chunk(self, tokens, token_words, *, cache, decode=False):
        if not cache.kv_bound:
            raise RuntimeError("Request KV storage must be bound before model execution")
        residual = self.embed(tokens)
        for layer, state in zip(self.layers, cache.layers, strict=True):
            forward = layer.decode_forward if decode else layer.prefill_forward
            residual = forward(residual, state=state, token_ids=token_words if layer.has_ple else None)
        return self.logits(residual)

    def decode_batch(self, tokens, token_words, *, caches):
        if any(not cache.kv_bound for cache in caches):
            raise RuntimeError("Request KV storage must be bound before model execution")
        residuals = tuple(self.embed(t) for t in tokens)
        for i, layer in enumerate(self.layers):
            residuals = layer.decode_forward(
                residuals, state=tuple(c.layers[i] for c in caches), token_ids=token_words if layer.has_ple else None
            )
        return tuple(self.logits(r) for r in residuals)
