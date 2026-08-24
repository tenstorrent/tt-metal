# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN weight preprocessing and shared layer helpers for the XTTS-v2 GPT core (Block 3).

Reference: models/experimental/xtts_v2/reference/xtts_gpt_ref.py
Architecture (HF GPT2, 30 blocks, causal, wpe nulled) + XTTS final_norm:

    inputs_embeds [1,S,1024]
      -> for each of 30 blocks:
           h = x + attn(ln_1(x))          # causal MHA, 16 heads, head_dim 64
           x = h + mlp(ln_2(h))           # c_fc(1024->4096) -> gelu_new -> c_proj(4096->1024)
      -> ln_f(x)                          # GPT2's final LayerNorm
      -> final_norm(x)                    # XTTS's extra LayerNorm
      = latents [1,S,1024]

TTNNGPTCore holds only what both halves of that loop share: the LayerNorm, linear and MLP helpers.
The loop itself lives in TTNNGPTTracedDecoder, written twice — once for the batched prefill and
once for the single-token decode step — because the two need different attention ops and different
matmul program configs.

GPT2 Conv1D weights are stored [in, out], which matches ttnn.linear's x[.,in]@W[in,out]
convention directly (no transpose).
"""

from dataclasses import dataclass

import ttnn

from models.experimental.xtts_v2.reference.xtts_gpt_ref import load_gpt_core_state
from models.experimental.xtts_v2.tt.ttnn_xtts_layernorm import TTNNLayerNorm


@dataclass
class TTNNGPTConfig:
    n_embd: int = 1024
    n_layer: int = 30
    n_head: int = 16
    n_inner: int = 4096
    layer_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


def _compute_config(math_fidelity=ttnn.MathFidelity.HiFi4):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def mesh_replicate_mapper(device):
    """Weight/state distribution for the (data-parallel) mesh: replicate to every device on a
    multi-device MeshDevice, or a no-op on a single card (MeshDevice(1,1) -> get_num_devices()==1),
    so the same code runs on one dev card and on a 1xN mesh unchanged."""
    n = getattr(device, "get_num_devices", lambda: 1)()
    return ttnn.ReplicateTensorToMesh(device) if n > 1 else None


def preprocess_gpt_parameters(device, ckpt_path=None, dtype=ttnn.bfloat16):
    """Load the transformer-core weights from the XTTS checkpoint into TTNN tensors."""
    core = load_gpt_core_state(ckpt_path) if ckpt_path else load_gpt_core_state()
    cfg = TTNNGPTConfig()
    mapper = mesh_replicate_mapper(device)

    def lin(w, b):
        return {
            "weight": ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper),
            "bias": ttnn.from_torch(
                b.reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper
            ),
        }

    def norm(w, b):
        return TTNNLayerNorm(device, w, b, cfg.n_embd, eps=cfg.layer_norm_eps, dtype=dtype, mesh_mapper=mapper)

    params = {"blocks": []}
    for i in range(cfg.n_layer):
        p = f"h.{i}."
        params["blocks"].append(
            {
                "ln_1": norm(core[p + "ln_1.weight"], core[p + "ln_1.bias"]),
                "c_attn": lin(core[p + "attn.c_attn.weight"], core[p + "attn.c_attn.bias"]),
                "attn_proj": lin(core[p + "attn.c_proj.weight"], core[p + "attn.c_proj.bias"]),
                "ln_2": norm(core[p + "ln_2.weight"], core[p + "ln_2.bias"]),
                "c_fc": lin(core[p + "mlp.c_fc.weight"], core[p + "mlp.c_fc.bias"]),
                "mlp_proj": lin(core[p + "mlp.c_proj.weight"], core[p + "mlp.c_proj.bias"]),
            }
        )
    params["ln_f"] = norm(core["ln_f.weight"], core["ln_f.bias"])
    params["final_norm"] = norm(core["final_norm.weight"], core["final_norm.bias"])
    return params


class TTNNGPTCore:
    def __init__(
        self,
        device,
        parameters,
        config: TTNNGPTConfig = None,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    ):
        self.device = device
        self.mesh_mapper = mesh_replicate_mapper(device)  # None on a single card; replicate on a mesh
        self.params = parameters
        self.config = config or TTNNGPTConfig()
        self.compute_kernel_config = _compute_config(math_fidelity)
        self.scale = 1.0 / (self.config.head_dim**0.5)
        # single-token decoders flip this to use the width-sharded LayerNorm path.
        self.ln_sharded = False

    def _layer_norm(self, x, p):
        # p is a TTNNLayerNorm. ln_sharded (set by the single-token decoders) selects the
        # width-sharded program-config path; the prefill core keeps the interleaved path.
        return p(x, sharded=self.ln_sharded, compute_kernel_config=self.compute_kernel_config)

    def _linear(self, x, p, prg=None):
        return ttnn.linear(
            x,
            p["weight"],
            bias=p["bias"],
            compute_kernel_config=self.compute_kernel_config,
            program_config=prg,
        )

    def _mlp(self, x, block, prg_fc=None, prg_proj=None):
        # c_fc + gelu_new folded into the matmul — matches the base TTIR->TTNN lowering the
        # compiler emits, is validated against the separate tanh-gelu by test_gpt_core_pcc, and avoids
        # a standalone elementwise gelu kernel. Decode carries it in prg_fc's fused_activation;
        # prefill has no prg, where the string form folds in on its own. Passing the string
        # ALONGSIDE an explicit program_config does NOT fuse — it adds a second kernel — so use
        # the string only when the config is not already carrying the activation.
        h = ttnn.linear(
            x,
            block["c_fc"]["weight"],
            bias=block["c_fc"]["bias"],
            activation=None if prg_fc is not None and prg_fc.fused_activation else "gelu",
            compute_kernel_config=self.compute_kernel_config,
            program_config=prg_fc,
        )
        return self._linear(h, block["mlp_proj"], prg=prg_proj)
