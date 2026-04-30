# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen2.5-7B-Instruct — native TTTv2 stack (``Embedding1D``, ``RMSNorm1D``,
``Attention1D``, ``MLP1D``, ``RotarySetup1D``, ``LMHead1D``).

Tensor layout contracts:
  - **Prefill** hidden states: ``[1, 1, S, dim]`` TILE, ``S % 128 == 0``.
  - **Decode** hidden states: ``[1, 1, B, dim]`` TILE (``B`` padded to tile in modules).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.models.qwen25_7b import weight_utils
from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig
from models.common.modules.embedding.embedding_1d import Embedding1D, Embedding1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig
from models.common.modules.mlp.mlp_1d import MLP1D, MLP1DConfig
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from models.common.modules.rope.rope_1d import Rope1DConfig, RotarySetup1D, prepare_rot_idxs
from models.common.modules.tt_ccl import default_topology, get_tt_ccl


def _lazy(
    tensor: torch.Tensor,
    *,
    dtype: ttnn.DataType,
    cache: tuple[Path, str] | None,
) -> LazyWeight:
    """Minimal LazyWeight; ``Attention1D`` / ``MLP1D`` / ``Embedding1D`` resolvers fill mesh + memory."""
    return LazyWeight(source=tensor, dtype=dtype, cache_dir_weight_name=cache)


@dataclass
class Qwen25_7BTTTConfig:
    """Resolved hyper-parameters for a loaded HF Qwen2.5-7B checkpoint."""

    hf_model_id: str
    dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    num_hidden_layers: int
    max_batch_size: int
    max_seq_len: int
    rope_table_len: int


class Qwen25_7BDecoderLayer(LightweightModule):
    def __init__(
        self,
        *,
        input_layernorm: RMSNorm1D,
        self_attn: Attention1D,
        post_attention_layernorm: RMSNorm1D,
        mlp: MLP1D,
    ):
        super().__init__()
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp

    def prefill_forward(
        self,
        x: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        *,
        user_id: int = 0,
    ) -> ttnn.Tensor:
        r = self.input_layernorm.prefill_forward(x)
        r = self.self_attn.forward(r, None, rot_mats, mode="prefill", user_id=user_id)
        h = ttnn.add(x, r, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        r2 = self.post_attention_layernorm.prefill_forward(h)
        r2 = self.mlp.prefill_forward(r2)
        return ttnn.add(h, r2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        r = self.input_layernorm.forward(x, "decode")
        r = self.self_attn.forward(r, current_pos, rot_mats, mode="decode")
        h = ttnn.add(x, r, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        r2 = self.post_attention_layernorm.forward(h, "decode")
        r2 = self.mlp.forward(r2, "decode")
        return ttnn.add(h, r2, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class Qwen25_7BTTT(LightweightModule):
    """
    Full decoder for Qwen2.5-7B-Instruct (TTTv2 modules only).

    Call ``prefill_forward`` for a prompt chunk, then ``decode_forward`` one token
    at a time (``current_pos`` is the index of the token being predicted).
    """

    def __init__(
        self,
        cfg: Qwen25_7BTTTConfig,
        embed: Embedding1D,
        rope: RotarySetup1D,
        layers: List[Qwen25_7BDecoderLayer],
        norm: RMSNorm1D,
        lm_head: LMHead1D,
        mesh_device: ttnn.MeshDevice,
    ):
        super().__init__()
        self.cfg = cfg
        self.embed = embed
        self.rope = rope
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.mesh_device = mesh_device

    @classmethod
    def from_pretrained(
        cls,
        mesh_device: ttnn.MeshDevice,
        hf_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        revision: str | None = None,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        num_layers: int | None = None,
        cache_dir: Path | str | None = None,
        wqkv_dtype: ttnn.DataType = ttnn.bfloat16,
        mlp_w_dtype: ttnn.DataType = ttnn.bfloat8_b,
    ) -> Qwen25_7BTTT:
        """
        Load HF weights on host and build TTNN modules (weights materialize on first forward).

        Args:
            mesh_device: Open mesh device (N150 ``(1,1)``, N300 ``(1,2)``, …).
            hf_model_id: Hugging Face hub id.
            max_batch_size: Decode batch / KV allocation (tile-padded internally).
            max_seq_len: KV cache sequence budget (per layer).
            num_layers: If set, truncate stack for smoke tests.
            cache_dir: Optional directory for ``LazyWeight`` tensor caches.
        """
        ttnn.SetDefaultDevice(mesh_device)
        cache_path = Path(cache_dir) if cache_dir else None
        num_dev = mesh_device.get_num_devices()
        tt_ccl = get_tt_ccl(mesh_device) if num_dev > 1 else None
        topology = default_topology(mesh_device)

        hf_cfg = AutoConfig.from_pretrained(hf_model_id, revision=revision)
        n_heads_hf = hf_cfg.num_attention_heads
        n_kv_hf = hf_cfg.num_key_value_heads
        if num_dev > 1 and (n_heads_hf % num_dev != 0 or n_kv_hf % num_dev != 0):
            raise ValueError(
                f"This checkpoint requires num_attention_heads ({n_heads_hf}) and "
                f"num_key_value_heads ({n_kv_hf}) to each be divisible by the mesh device "
                f"count ({num_dev}) for Attention1D sharding. "
                f"Use a smaller mesh (e.g. 2 or 4 devices) or a model whose head counts divide the mesh."
            )
        torch_dtype = torch.bfloat16
        logger.info(f"Loading HF weights: {hf_model_id} (revision={revision})")
        hf = AutoModelForCausalLM.from_pretrained(hf_model_id, revision=revision, torch_dtype=torch_dtype)
        hf.eval()
        base = hf.model
        n_layers = num_layers if num_layers is not None else hf_cfg.num_hidden_layers
        dim = hf_cfg.hidden_size
        n_heads = hf_cfg.num_attention_heads
        n_kv = hf_cfg.num_key_value_heads
        head_dim = dim // n_heads
        inter = hf_cfg.intermediate_size
        vocab = hf_cfg.vocab_size
        rope_len = max(max_seq_len * 2, 8192)
        rope_len = (rope_len + 127) // 128 * 128

        qcfg = Qwen25_7BTTTConfig(
            hf_model_id=hf_model_id,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv,
            head_dim=head_dim,
            hidden_dim=inter,
            vocab_size=vocab,
            rms_norm_eps=hf_cfg.rms_norm_eps,
            rope_theta=getattr(hf_cfg, "rope_theta", 1_000_000.0),
            num_hidden_layers=n_layers,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            rope_table_len=rope_len,
        )

        emb_src = weight_utils.embed_tokens_torch(base.embed_tokens)
        emb = Embedding1D.from_config(
            Embedding1DConfig(
                weights=_lazy(
                    emb_src,
                    dtype=ttnn.bfloat16,
                    cache=(cache_path / "embedding", "tok_embeddings") if cache_path else None,
                ),
                mesh_device=mesh_device,
                embed_scale=1.0,
            )
        )

        cos_t, sin_t = weight_utils.build_rope_cos_sin_torch(base.rotary_emb, rope_len, head_dim, torch_dtype)
        cos_lw = _lazy(cos_t, dtype=ttnn.bfloat16, cache=(cache_path / "rope", "cos") if cache_path else None)
        sin_lw = _lazy(sin_t, dtype=ttnn.bfloat16, cache=(cache_path / "rope", "sin") if cache_path else None)
        rope = RotarySetup1D.from_config(
            Rope1DConfig(
                cos_matrix=cos_lw,
                sin_matrix=sin_lw,
                max_batch_size=max_batch_size,
                head_dim=head_dim,
                device=mesh_device,
                use_qk_fused=False,
            )
        )

        layers: list[Qwen25_7BDecoderLayer] = []
        for idx in range(n_layers):
            layer = base.layers[idx]
            prefix = f"layer{idx}"
            wqkv, wo, qn, kn, wqkv_b = weight_utils.attention_wqkv_wo_from_hf_layer(layer.self_attn, num_dev)
            lazy_wqkv = _lazy(
                wqkv,
                dtype=wqkv_dtype,
                cache=(cache_path / "attn", f"{prefix}_wqkv") if cache_path else None,
            )
            lazy_wo = _lazy(
                wo,
                dtype=wqkv_dtype,
                cache=(cache_path / "attn", f"{prefix}_wo") if cache_path else None,
            )
            q_norm_cfg = None
            k_norm_cfg = None
            if qn is not None:
                lazy_qn = _lazy(
                    qn.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch_dtype),
                    dtype=ttnn.bfloat16,
                    cache=(cache_path / "attn", f"{prefix}_qn") if cache_path else None,
                )
                q_norm_cfg = RMSNorm1DConfig(
                    weight=lazy_qn,
                    mesh_device=mesh_device,
                    eps=hf_cfg.rms_norm_eps,
                    decode_in_sharded=False,
                    decode_out_sharded=False,
                    prefill_distributed=False,
                    tt_ccl=tt_ccl,
                )
            if kn is not None:
                lazy_kn = _lazy(
                    kn.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch_dtype),
                    dtype=ttnn.bfloat16,
                    cache=(cache_path / "attn", f"{prefix}_kn") if cache_path else None,
                )
                k_norm_cfg = RMSNorm1DConfig(
                    weight=lazy_kn,
                    mesh_device=mesh_device,
                    eps=hf_cfg.rms_norm_eps,
                    decode_in_sharded=False,
                    decode_out_sharded=False,
                    prefill_distributed=False,
                    tt_ccl=tt_ccl,
                )
            bias_lw = (
                LazyWeight(
                    source=wqkv_b.to(torch_dtype),
                    dtype=ttnn.bfloat16,
                    cache_dir_weight_name=(cache_path / "attn", f"{prefix}_bias") if cache_path else None,
                )
                if wqkv_b is not None
                else None
            )

            attn_cfg = Attention1DConfig(
                wqkv=lazy_wqkv,
                wo=lazy_wo,
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                topology=topology,
                n_heads=n_heads,
                n_kv_heads=n_kv,
                head_dim=head_dim,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                q_norm_config=q_norm_cfg,
                k_norm_config=k_norm_cfg,
                wqkv_bias=bias_lw,
            )
            attn = Attention1D.from_config(attn_cfg)

            w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(layer.mlp)
            lazy_w1 = _lazy(w1, dtype=mlp_w_dtype, cache=(cache_path / "mlp", f"{prefix}_w1") if cache_path else None)
            lazy_w2 = _lazy(w2, dtype=mlp_w_dtype, cache=(cache_path / "mlp", f"{prefix}_w2") if cache_path else None)
            lazy_w3 = _lazy(w3, dtype=mlp_w_dtype, cache=(cache_path / "mlp", f"{prefix}_w3") if cache_path else None)
            mlp = MLP1D.from_config(
                MLP1DConfig(
                    w1=lazy_w1,
                    w2=lazy_w2,
                    w3=lazy_w3,
                    mesh_device=mesh_device,
                    tt_ccl=tt_ccl,
                    topology=topology,
                    max_batch_size=max_batch_size,
                )
            )

            def make_norm(w_flat: torch.Tensor, name: str) -> RMSNorm1D:
                lw = _lazy(
                    w_flat.to(torch_dtype),
                    dtype=ttnn.bfloat16,
                    cache=(cache_path / "norm", f"{prefix}_{name}") if cache_path else None,
                )
                return RMSNorm1D.from_config(
                    RMSNorm1DConfig(
                        weight=lw,
                        mesh_device=mesh_device,
                        eps=hf_cfg.rms_norm_eps,
                        max_batch_size=max_batch_size,
                        tt_ccl=tt_ccl,
                    )
                )

            layers.append(
                Qwen25_7BDecoderLayer(
                    input_layernorm=make_norm(weight_utils.rms_weight_torch(layer.input_layernorm), "pre_attn"),
                    self_attn=attn,
                    post_attention_layernorm=make_norm(
                        weight_utils.rms_weight_torch(layer.post_attention_layernorm), "post_attn"
                    ),
                    mlp=mlp,
                )
            )

        norm_lw = _lazy(
            weight_utils.rms_weight_torch(base.norm).to(torch_dtype),
            dtype=ttnn.bfloat16,
            cache=(cache_path / "norm", "final") if cache_path else None,
        )
        final_norm = RMSNorm1D.from_config(
            RMSNorm1DConfig(
                weight=norm_lw,
                mesh_device=mesh_device,
                eps=hf_cfg.rms_norm_eps,
                max_batch_size=max_batch_size,
                tt_ccl=tt_ccl,
            )
        )

        lm_w = hf.lm_head.weight.detach().to(torch_dtype).clone()
        lm_splits = weight_utils.build_lm_head_lazy_weights(
            mesh_device,
            lm_w,
            dim=dim,
            vocab_size=vocab,
            cache_dir=cache_path / "lm_head" if cache_path else None,
        )
        lm = LMHead1D.from_config(
            LMHead1DConfig(
                output_weights=lm_splits,
                mesh_device=mesh_device,
                dim=dim,
                max_batch_size=max_batch_size,
                lm_head_dtype=ttnn.bfloat8_b,
            )
        )

        del hf
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return cls(qcfg, emb, rope, layers, final_norm, lm, mesh_device)

    def prefill_forward(self, token_ids_tt: ttnn.Tensor, *, start_pos: int = 0, user_id: int = 0) -> ttnn.Tensor:
        """
        Run prefill on a token column.

        Args:
            token_ids_tt: ``uint32`` ids, shape ``[1, 1, 1, S]`` (``S`` divisible by 128).
            start_pos: Absolute position of the first token (usually 0).
        """
        x = self.embed.forward(token_ids_tt)
        seq_len = x.shape[2]
        assert seq_len % 128 == 0, "prefill seq_len must be divisible by 128"
        rot = self.rope.prefill_forward(start_pos, seq_len)
        h = x
        for layer in self.layers:
            h = layer.prefill_forward(h, rot, user_id=user_id)
        return self.norm.prefill_forward(h)

    def decode_forward(self, token_ids_tt: ttnn.Tensor, *, current_pos: int) -> ttnn.Tensor:
        """
        Single decode step (batch 1): ``token_ids_tt`` shape ``[1, 1, 1, 1]``, ``current_pos`` is KV index.
        """
        x = self.embed.forward(token_ids_tt)
        pos = torch.tensor([current_pos], dtype=torch.long)
        rot_idxs = prepare_rot_idxs(self.rope.config, pos, on_host=False)
        rot = self.rope.decode_forward(rot_idxs)
        cur = ttnn.from_torch(
            pos,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(self.mesh_device),
        )
        h = x
        for layer in self.layers:
            h = layer.decode_forward(h, cur, rot)
        return self.norm.forward(h, "decode")

    def lm_logits(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """Project last hidden to logits (vocab-sharded on multi-device)."""
        return self.lm_head.forward(hidden)
