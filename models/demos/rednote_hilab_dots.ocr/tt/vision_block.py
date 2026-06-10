# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN vision transformer block for dots.ocr.

DotsVisionBlock (modeling_dots_vision): pre-norm residual block —
``x + attn(norm1(x))`` then ``h + mlp(norm2(h))`` — composing the already
brought-up sub-blocks TtVisionRMSNorm (eps=1e-5), TtVisionAttention (fused
QKV MHA, 12 heads x head_dim 128, 2D rope, windowed SDPA over cu_seqlens) and
TtVisionMLP (SwiGLU 1536 -> 4224 -> 1536). Mirrors reference_impl
models/demos/qwen25_vl/tt/vision_block.py: norm -> attention -> ttnn.add
residual -> norm -> mlp -> ttnn.add residual, DRAM interleaved throughout.

Parallelism plan (ARCHITECTURE.md): vision tower placement=replicate — all
sub-block weights are ``ReplicateTensorToMesh`` on the 1x4 mesh, activations
stay replicated, no CCL. On a single device the mesh_mapper degenerates
gracefully. Production (tt/ocr_model.py) overrides to tp_degree=4.

Occupancy REDO (production posture: bf16 tower, tp=4, 11264-row document,
1x4 BH mesh, queried grid 11x10=110, tracy under --traced): block traced
kernel time 8.03 -> 7.14 ms/device (-11.2%; 42 calls/image => ~37 ms off
the tower). The composed-block hotspots are the sub-blocks' recorded
ceilings (SDPA 2.57 ms @ 110/110 — chunk A/B ceiling; matmuls 1.47 ms @
88-100/110 — per-matmul A/B ceilings), so this tick's lever is the CCL
pair, 31.8%% of block time at 18-34 cores: it is LINK-bound at the 2/2-link
HW ceiling, so the remaining axis is wire BYTES — the tp>1 all-reduces in
attention (o_proj) and MLP (down) now emit bfloat8_b partials
(all_gather 1334 -> 742 us, reduce_scatter 1222 -> 1080 us/device; the
residual adds consume the bf8b branch output mixed-dtype, the bf16 residual
STREAM is untouched — the recorded 42-block compounding hazard governs).
reduce_scatter does not scale with bytes (local-reduction-bound, not
wire-bound). reduce_scatter_minimal_async + all_gather_async A/B at the
production shape LOSES (1.259 vs 1.050 ms median traced) — sync composite
kept. Residual adds/norms: 110/110 cores, DRAM placement per the recorded
L1-into-matmul stall hazard (QKV 97 -> 2963 us). Gates: block PCC 0.999917
at the tp=4 production parallelism (>0.99), gate tests unchanged
(block 0.999951 / attention 0.999693 / mlp 0.999991), e2e WER parity
re-run PASS (ttnn_wer <= hf_wer + 0.05).
"""

import importlib.util
import sys
from pathlib import Path

import ttnn
from models.common.lightweightmodule import LightweightModule

# Dir name contains a dot -> not importable as a package; load siblings by path.
_TT_DIR = Path(__file__).resolve().parent


def _load_sibling(stem):
    name = f"dots_ocr_tt_{stem}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _TT_DIR / f"{stem}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


TtVisionRMSNorm = _load_sibling("vision_rmsnorm").TtVisionRMSNorm
TtVisionAttention = _load_sibling("vision_attention").TtVisionAttention
TtVisionMLP = _load_sibling("vision_mlp").TtVisionMLP


class TtVisionBlock(LightweightModule):
    """dots.ocr vision block: x + attn(norm1(x)); h + mlp(norm2(h)).

    Args:
        mesh_device: ttnn mesh device handle (weights replicated).
        state_dict: torch tensors keyed norm1.weight, attn.qkv.weight,
            attn.proj.weight, norm2.weight, mlp.fc1.weight, mlp.fc2.weight,
            mlp.fc3.weight (HF keys vision_tower.blocks.N.* with the block
            prefix stripped).
        num_heads: attention heads (dots.ocr vision: 12, head_dim 128).
        dtype: on-device weight/activation dtype.
        eps: RMSNorm epsilon (dots.ocr vision uses 1e-5).
    """

    def __init__(self, mesh_device, state_dict, num_heads=12, dtype=ttnn.bfloat16, eps=1e-5, tp_degree=1):
        super().__init__()
        self.mesh_device = mesh_device

        self.norm1 = TtVisionRMSNorm(mesh_device, {"weight": state_dict["norm1.weight"]}, dtype=dtype, eps=eps)
        self.attn = TtVisionAttention(
            mesh_device,
            {"qkv.weight": state_dict["attn.qkv.weight"], "proj.weight": state_dict["attn.proj.weight"]},
            num_heads=num_heads,
            dtype=dtype,
            tp_degree=tp_degree,
        )
        self.norm2 = TtVisionRMSNorm(mesh_device, {"weight": state_dict["norm2.weight"]}, dtype=dtype, eps=eps)
        self.mlp = TtVisionMLP(
            mesh_device,
            {
                "fc1.weight": state_dict["mlp.fc1.weight"],
                "fc2.weight": state_dict["mlp.fc2.weight"],
                "fc3.weight": state_dict["mlp.fc3.weight"],
            },
            dtype=dtype,
            tp_degree=tp_degree,
        )

    # Host-side input prep delegates to the attention sub-block (rope tables
    # and cu_seqlens stay on host per ARCHITECTURE.md hybrid_notes).
    def prepare_rope(self, rotary_pos_emb, padded_seq):
        return self.attn.prepare_rope(rotary_pos_emb, padded_seq)

    def prepare_cu_seqlens(self, cu_seqlens):
        return self.attn.prepare_cu_seqlens(cu_seqlens)

    def forward(self, x_11SH: ttnn.Tensor, rot_mats, cu_seqlens: ttnn.Tensor) -> ttnn.Tensor:
        """x_11SH: [1, 1, padded_seq, dim] TILE_LAYOUT, replicated.

        rot_mats: (cos, sin) each [1, 1, padded_seq, head_dim] from prepare_rope.
        cu_seqlens: uint32 device tensor of unpadded window boundaries.
        Returns: [1, 1, padded_seq, dim], replicated (rows past the last
        cu_seqlens boundary are padding garbage — caller slices them off).

        The residual stream keeps the INPUT dtype: a bf16 caller gets the
        original all-bf16 behaviour, while the 42-layer full tower passes an
        fp32 residual (branch outputs stay bf16) to stop rounding error from
        compounding across blocks.
        """
        res_dtype = x_11SH.dtype
        attn_in = self.norm1(x_11SH)
        attn_out = self.attn(attn_in, rot_mats, cu_seqlens)
        ttnn.deallocate(attn_in)
        # Residual adds live in L1: their only consumers are RMSNorm reads and
        # the next residual add — never a matmul — so the L1 pin avoids the
        # L1-interleaved-into-matmul stall (tower tracy: DRAM add 44.2us vs
        # 13.8us for the same-sized L1 BinaryNg) while norm reads get faster.
        # Size-conditional: L1 win measured at seq<=896; document-scale
        # sequences (10k+ tokens, ~67 MB fp32 residual) exceed L1 — use DRAM.
        res_mem = ttnn.L1_MEMORY_CONFIG if x_11SH.shape[-2] <= 4096 else ttnn.DRAM_MEMORY_CONFIG
        h = ttnn.add(x_11SH, attn_out, memory_config=res_mem, dtype=res_dtype)
        ttnn.deallocate(attn_out)

        ff_in = self.norm2(h)
        ff_out = self.mlp(ff_in)
        ttnn.deallocate(ff_in)
        out = ttnn.add(h, ff_out, memory_config=res_mem, dtype=res_dtype)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return out
