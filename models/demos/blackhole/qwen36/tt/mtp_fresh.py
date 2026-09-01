# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fresh ttnn port of the Qwen3.6/3.8 MTP drafter head.

Independent rewrite of the drafter head (deliberately not reusing tt/mtp.py),
scoped to the actual architecture the checkpoint's mtp.* weights encode --
vLLM's Qwen3_5MultiTokenPredictor, per reference/mtp_torch.py::MTPTorchReference,
which this module must match numerically:

    x = fc(cat[rmsnorm(embed(tok)), rmsnorm(hidden)])   # [.,2H] -> [.,H]
    x = decoder_layer(x, position)                        # one full-attention layer, own KV
    out = rmsnorm(x); logits = lm_head(out)

This is NOT Gemma4's assistant-model architecture (models/demos/gemma4/tt/assistant/model.py):
that drafter owns its own attention/MLP weights end to end and cross-attends into the
TARGET's live KV cache (EAGLE-style). Our checkpoint's mtp.* weights carry no such
cross-attention parameters -- the head shares the target's embed_tokens/lm_head instead,
and keeps its own small paged KV cache. The two are different drafter designs dictated by
the checkpoint, not a style choice; only the coding conventions below (reuse the target's
own decoder-layer building blocks, load weights via a substate() prefix slice + optional
on-disk tensor cache, keep step() as the single atomic forward unit) are borrowed from
Gemma4's assistant model.

Scope, deliberately: single-user eager step() only. No traced replay windows, no batched
multi-user stepping, no on-device argmax -- tt/mtp.py's Qwen36MTPHead already carries that
performance engineering; this module is the from-scratch correctness baseline.
"""

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.tt.rms_norm import rms_norm_ttnn
from models.demos.blackhole.qwen36.utils.substate import substate


class _ReplicatedArgsView:
    """Forces num_devices=1 on the shared attention/MLP modules so the drafter's one
    layer runs fully replicated across a TP mesh instead of taking their sharded path --
    a 1-layer step is small enough that redundant per-device compute beats sharding it,
    and it keeps every device's copy of the drafter numerically independent (no CCL)."""

    num_devices = 1

    def __init__(self, args):
        object.__setattr__(self, "_args", args)

    def __getattr__(self, name):
        return getattr(self._args, name)


class Qwen36MTPDrafter:
    """The MTP drafter: fc-fused embed+hidden -> one full-attention decoder layer -> lm_head.

    Args:
        mesh_device: single device or TP mesh (replicated execution on a mesh).
        args: Qwen36ModelArgs (attention geometry / dims / eps).
        mtp_state_dict: mtp.*-stripped weights (weight_mapping.load_qwen36_mtp_state_dict).
        embedding: callable, replicated [1,1] uint32 device token -> replicated [1,1,dim]
            embedding (the target's own embedding table; a TP caller supplies its own
            replicated copy since the target's table is hidden-sharded).
        lm_head_weight: device weight shared with the target, [dim, vocab] ttnn.linear layout.
        rope: RoPE table object shared with the target (replicated on a mesh).
        lm_head_sharded: True when lm_head_weight is vocab-sharded across the mesh (step()
            then gathers the logit shards on host instead of reading one replica).
    """

    def __init__(
        self,
        mesh_device,
        args,
        mtp_state_dict,
        embedding,
        lm_head_weight,
        rope,
        tensor_cache_path=None,
        lm_head_sharded=False,
    ):
        self.device = mesh_device
        self.args = args
        self.embedding = embedding
        self.lm_head_weight = lm_head_weight
        self.rope = rope
        self.norm_eps = args.norm_eps
        self.num_devices = mesh_device.get_num_devices()
        self.lm_head_sharded = lm_head_sharded
        self._mesh_kwargs = dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)) if self.num_devices > 1 else {}
        layer_args = _ReplicatedArgsView(args) if self.num_devices > 1 else args

        cache_dir = (tensor_cache_path / "mtp_fresh") if tensor_cache_path else None

        def load_norm(name):
            # rms_norm_ttnn's zero-centered contract expects (1 + weight) pre-baked in.
            weight = mtp_state_dict[f"{name}.weight"] + 1.0
            return ttnn.as_tensor(
                weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=(cache_dir / f"{name}.weight_offset") if cache_dir else None,
                **self._mesh_kwargs,
            )

        self.norm_embed = load_norm("pre_fc_norm_embedding")
        self.norm_hidden = load_norm("pre_fc_norm_hidden")
        self.attn_norm = load_norm("layers.0.input_layernorm")
        self.ffn_norm = load_norm("layers.0.post_attention_layernorm")
        self.final_norm = load_norm("norm")

        # fc: HF [out, 2*dim] -> ttnn.linear's [in, out] convention.
        self.fc_weight = ttnn.as_tensor(
            mtp_state_dict["fc.weight"].T.contiguous(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(cache_dir / "fc.weight") if cache_dir else None,
            **self._mesh_kwargs,
        )

        # One ordinary full-attention decoder layer -- reuse the target's own building
        # blocks rather than reimplementing attention/MLP math for a single extra layer.
        self.attention = Qwen36GatedAttention(
            mesh_device,
            AttentionConfig.from_args(args),
            substate(mtp_state_dict, "layers.0.self_attn"),
            cache_dir,
        )
        self.feed_forward = Qwen36MLP(mesh_device, substate(mtp_state_dict, "layers.0.mlp"), cache_dir, args=layer_args)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self._k_cache = None
        self._v_cache = None
        self._page_tables = []

    def allocate_kv_cache(self, num_blocks_per_user, block_size=64, dtype=ttnn.bfloat16, users=1):
        """Allocate the drafter's own single-layer paged KV cache, identity page table per user.

        `num_blocks_per_user` blocks are reserved per user; user u's page table routes to
        blocks [u*num_blocks_per_user, (u+1)*num_blocks_per_user)."""
        assert self._k_cache is None, "KV cache already allocated"
        shape = [users * num_blocks_per_user, self.args.n_kv_heads, block_size, self.args.head_dim]
        self._k_cache = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        self._v_cache = ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        self.attention.set_paged_kv_cache(self._k_cache, self._v_cache)
        host_tables = torch.stack(
            [
                torch.arange(u * num_blocks_per_user, (u + 1) * num_blocks_per_user, dtype=torch.int32)
                for u in range(users)
            ]
        )
        self._page_tables = [
            ttnn.from_torch(
                host_tables[u : u + 1].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
            for u in range(users)
        ]

    def free_kv_cache(self):
        if self._k_cache is None:
            return
        for t in (self._k_cache, self._v_cache, *self._page_tables):
            ttnn.deallocate(t)
        self._k_cache = self._v_cache = None
        self._page_tables = []
        self.attention.paged_kv_cache_key = None
        self.attention.paged_kv_cache_value = None

    def step(self, token_id, hidden_row, position, user=0):
        """One drafter step: (hidden at `position`, token at `position`+1) -> logits for
        `position`+2, and the post-norm hidden to chain into the next step.

        token_id: int, target token at position+1 (or the drafter's own chained pick).
        hidden_row: torch [dim], target's post-final-norm hidden at `position` (or the
            drafter's own hidden from the previous chained step).
        """
        assert self._k_cache is not None, "call allocate_kv_cache first"
        tok = ttnn.from_torch(
            torch.tensor([[token_id]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            **self._mesh_kwargs,
        )
        hidden = ttnn.from_torch(
            hidden_row.reshape(1, 1, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            **self._mesh_kwargs,
        )
        cos, sin = self.rope.get_rot_mats(torch.tensor([[position]], dtype=torch.long))
        pos_tensor = ttnn.from_torch(
            torch.tensor([position], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            **self._mesh_kwargs,
        )

        logits, normed = self._forward(tok, hidden, pos_tensor, cos, sin, self._page_tables[user])
        for t in (tok, hidden, pos_tensor):
            ttnn.deallocate(t)

        logits_host = self._read_logits(logits)
        hidden_host = self._read_replicated(normed)
        ttnn.deallocate(logits)
        ttnn.deallocate(normed)
        return logits_host, hidden_host

    def _forward(self, tok, hidden, pos_tensor, cos, sin, page_table):
        """The drafter's op graph: fc(cat[rmsnorm(embed), rmsnorm(hidden)]) -> one decoder
        block -> final norm -> lm_head. Activations kept in DRAM throughout (a resident L1
        activation would pin the allocator watermark against the decoder block's static CB
        region -- see tt/mtp.py's _step_graph for the measured margin this avoids)."""
        dram = ttnn.DRAM_MEMORY_CONFIG

        embed = self.embedding(tok)
        embed_n = rms_norm_ttnn(embed, self.norm_embed, eps=self.norm_eps, memory_config=dram)
        ttnn.deallocate(embed)
        hidden_n = rms_norm_ttnn(hidden, self.norm_hidden, eps=self.norm_eps, memory_config=dram)

        fused = ttnn.concat([embed_n, hidden_n], dim=-1)  # embed first, matching vLLM's cat order
        ttnn.deallocate(embed_n)
        ttnn.deallocate(hidden_n)
        x = ttnn.linear(fused, self.fc_weight, compute_kernel_config=self.compute_kernel_config, memory_config=dram)
        ttnn.deallocate(fused)

        attn_in = rms_norm_ttnn(x, self.attn_norm, eps=self.norm_eps, memory_config=dram)
        attn_out = self.attention.forward(attn_in, cos, sin, position_tensor=pos_tensor, page_table=page_table)
        ttnn.deallocate(attn_in)
        residual1 = ttnn.add(x, attn_out, memory_config=dram)
        ttnn.deallocate(x)
        ttnn.deallocate(attn_out)

        ffn_in = rms_norm_ttnn(residual1, self.ffn_norm, eps=self.norm_eps, memory_config=dram)
        ffn_out = self.feed_forward.forward(ffn_in)
        ttnn.deallocate(ffn_in)
        residual2 = ttnn.add(residual1, ffn_out, memory_config=dram)
        ttnn.deallocate(residual1)
        ttnn.deallocate(ffn_out)

        normed = rms_norm_ttnn(residual2, self.final_norm, eps=self.norm_eps, memory_config=dram)
        ttnn.deallocate(residual2)
        logits = ttnn.linear(normed, self.lm_head_weight, compute_kernel_config=self.compute_kernel_config)
        return logits, normed

    def _read_replicated(self, t):
        """Host readback for a tensor replicated identically on every device."""
        if self.num_devices > 1:
            return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1).float()
        return ttnn.to_torch(t).reshape(-1).float()

    def _read_logits(self, logits):
        if self.num_devices > 1 and self.lm_head_sharded:
            return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=-1)).reshape(-1).float()
        return self._read_replicated(logits)
