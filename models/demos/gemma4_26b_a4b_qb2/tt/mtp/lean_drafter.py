# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The it-assistant drafter written against the port's decode ops (WS6).

Same math as ``models/demos/gemma4/tt/assistant/model.py`` / HF ``Gemma4AssistantForCausalLM`` (4 KV-shared
Gemma4 decoder layers, hidden 1024, dense GeGLU MLP), with fewer device programs per step (about 80 instead of
230 for the demo layers):

- Q heads: the q projection is padded with one zero K and one zero V head so the port's ``QkvHeadsNormRope``
  op does the head split, q RMSNorm and RoPE in one program and emits q in the layout SDPA reads; the zero
  K / V heads are discarded.
- SDPA: the target layer's own paged decode SDPA config (layers 28 / 29), reading that layer's cache.
- TP4 over heads and the MLP intermediate; the two row-parallel outputs all-reduce through the port's
  persistent collective buffers; hidden, norms and the pre / post projections are replicated.
- RoPE rows are gathered once per round (the drafter position is fixed for the K steps).
- The LM head stays vocab-sharded (``[1, 1, 1, vocab / 4]`` per chip, like the target's) for the greedy sampler.
"""

import os

import torch
from safetensors import safe_open

import ttnn
from models.demos.gemma4.custom_ops.concat_heads_decode import ConcatHeadsDecode
from models.demos.gemma4.custom_ops.qkv_heads_norm_rope import QkvHeadsNormRope
from models.demos.gemma4.tt.model import _get_lm_head_program_config
from models.demos.gemma4_26b_a4b_qb2.tt.mtp.drafter import DRAFTER_ID, DRAFTER_REVISION, shared_layers
from models.demos.gemma4_26b_a4b_qb2.tt.multichip_decoder import DecodeRopeRows

TP = 4
DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


class LeanDrafter:
    def __init__(self, model, *, lm_head_dtype=None, weight_dtype=None):
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig

        path = snapshot_download(DRAFTER_ID, revision=DRAFTER_REVISION, local_files_only=True)
        config = AutoConfig.from_pretrained(path)
        self.text = config.text_config
        if config.backbone_hidden_size != model.config.hidden_size or config.use_ordered_embeddings:
            raise ValueError("Unsupported drafter checkpoint")
        self.model, self.mesh = model, model.mesh
        self.hidden = self.text.hidden_size
        self.eps = self.text.rms_norm_eps
        self.types = list(self.text.layer_types)
        targets = shared_layers(model)
        # The target layer whose cache, SDPA config and RoPE table each drafter layer uses.
        self.target_layer = {kind: model.layers[number] for kind, number in targets.items()}
        self.cache_index = dict(targets)
        self.ccl_layer = model.layers[0]
        # Drafter weights are proposals the verify checks: their dtype moves only the acceptance.
        if lm_head_dtype is None:
            lm_head_dtype = DTYPES[os.environ.get("GEMMA4_26B_MTP_LM_HEAD", "bfp8")]
        if weight_dtype is None:
            weight_dtype = DTYPES[os.environ.get("GEMMA4_26B_MTP_WEIGHTS", "bfp8")]
        self.weight_dtype = weight_dtype
        self.programs = {}
        with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as file:
            state = {key: file.get_tensor(key) for key in file.keys()}
        self.layers = [self._layer(state, i) for i in range(self.text.num_hidden_layers)]
        self.pre_projection = self._replicated(state["pre_projection.weight"].T, weight_dtype)
        self.post_projection = self._replicated(state["post_projection.weight"].T, weight_dtype)
        self.final_norm = self._gamma(state["model.norm.weight"])
        self.lm_head = self._sharded(state["model.embed_tokens.weight"].T, -1, lm_head_dtype)
        self.compute = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.norm_compute = model.layers[0].layer.self_attn.compute
        # Residual stream width-sharded over 8x4 cores (one 32-column tile each): one-core interleaved RMSNorms of
        # the 1024-wide row cost ~17 us each, the sharded ones a few us.
        grid = (8, 4)
        per = self.hidden // 32 // (grid[0] * grid[1])
        self.shard = ttnn.create_sharded_memory_config(
            shape=(32, per * 32),
            core_grid=ttnn.CoreGrid(x=grid[0], y=grid[1]),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=grid, subblock_w=per, block_h=1, block_w=per, inplace=False
        )
        self.lm_head_dtype = lm_head_dtype

    # ---- weights ----------------------------------------------------------------------------------------------
    def _upload(self, value, mapper, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            value.contiguous()[None, None],
            dtype=dtype,
            layout=layout,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def _replicated(self, value, dtype=ttnn.bfloat16):
        return self._upload(value, ttnn.ReplicateTensorToMesh(self.mesh), dtype)

    def _sharded(self, value, dim, dtype=ttnn.bfloat16):
        return self._upload(value, ttnn.ShardTensorToMesh(self.mesh, dim=dim), dtype)

    def _gamma(self, weight):
        # Row-major bf16 gamma rows, the layout ttnn.rms_norm takes for an interleaved input.
        return self._upload(weight.reshape(-1, 32), ttnn.ReplicateTensorToMesh(self.mesh), layout=ttnn.ROW_MAJOR_LAYOUT)

    def _layer(self, state, i):
        prefix = f"model.layers.{i}."
        kind = self.types[i]
        q = state[prefix + "self_attn.q_proj.weight"]  # [heads * D, hidden]
        heads = self.text.num_attention_heads
        dim = q.shape[0] // heads
        per = heads // TP
        if per != self.target_layer[kind].layer.self_attn.source.config.num_attention_heads:
            raise ValueError("Drafter q heads per chip differ from the target's")
        # Per chip: [q heads of this chip | one zero k head | one zero v head], concatenated over chips.
        blocks = []
        for chip in range(TP):
            local = q[chip * per * dim : (chip + 1) * per * dim].T
            blocks.append(torch.cat([local, torch.zeros(self.hidden, 2 * dim, dtype=local.dtype)], dim=-1))
        o = state[prefix + "self_attn.o_proj.weight"]  # [hidden, heads * D]
        return {
            "kind": kind,
            "dim": dim,
            "heads": per,
            "input": self._gamma(state[prefix + "input_layernorm.weight"]),
            "post_attention": self._gamma(state[prefix + "post_attention_layernorm.weight"]),
            "pre_ff": self._gamma(state[prefix + "pre_feedforward_layernorm.weight"]),
            "post_ff": self._gamma(state[prefix + "post_feedforward_layernorm.weight"]),
            "scalar": float(state[prefix + "layer_scalar"].float().item()),
            "qkv": self._sharded(torch.cat(blocks, dim=-1), -1, self.weight_dtype),
            "q_gamma": self._replicated(state[prefix + "self_attn.q_norm.weight"].reshape(1, -1)),
            "k_gamma": self._replicated(torch.ones(1, dim, dtype=torch.bfloat16)),
            "o": self._sharded(o.T, -2, self.weight_dtype),
            "gate": self._sharded(state[prefix + "mlp.gate_proj.weight"].T, -1, self.weight_dtype),
            "up": self._sharded(state[prefix + "mlp.up_proj.weight"].T, -1, self.weight_dtype),
            "down": self._sharded(state[prefix + "mlp.down_proj.weight"].T, -2, self.weight_dtype),
        }

    # ---- ops ----------------------------------------------------------------------------------------------------
    def _norm(self, value, gamma):
        """Sharded RMSNorm of a sharded row; returns the sharded result."""
        return ttnn.rms_norm(
            value,
            epsilon=self.eps,
            weight=gamma,
            program_config=self.norm_program,
            compute_kernel_config=self.norm_compute,
            memory_config=self.shard,
        )

    def _flat(self, value):
        return ttnn.sharded_to_interleaved(value, ttnn.L1_MEMORY_CONFIG)

    def _reduce(self, value, role):
        # The all-gather half of the all-reduce writes the sharded residual layout directly.
        return self.ccl_layer.allreduce(value, role=role, output_memory=self.shard)

    def _linear(self, value, weight, dtype=ttnn.bfloat16, memory=ttnn.L1_MEMORY_CONFIG):
        # One-row-tile decode matmuls: in0 multicast, the weight columns split over the whole grid.
        key = (int(weight.shape[-2]), int(weight.shape[-1]))
        if key not in self.programs:
            self.programs[key] = _get_lm_head_program_config(self.mesh, 1, *key)
        return ttnn.linear(
            value,
            weight,
            dtype=dtype,
            compute_kernel_config=self.compute,
            memory_config=memory,
            program_config=self.programs[key],
        )

    def rope_rows(self, pos_u):
        """The position-gathered RoPE rows per layer type for one round (``pos_u``: [1, 32] uint32, row 0)."""
        return {kind: DecodeRopeRows.gather(self.model.rope_decode[kind], pos_u) for kind in sorted(set(self.types))}

    def step(self, token, hidden, *, cache, table, sdpa_pos, rope):
        """One drafter step. ``token`` [1, 1] uint32, ``hidden`` [1, 1, 1, 2816] bf16 (the seed or the previous
        step's ``next_hidden``), ``cache`` the target's KV cache list, ``sdpa_pos`` [1] int32.
        Returns (vocab-sharded logits [1, 1, 1, vocab / 4], next_hidden [1, 1, 1, 2816])."""
        embedded = self.model.embed(token)
        h = self._linear(ttnn.concat([embedded, hidden], dim=-1), self.pre_projection)
        h = ttnn.interleaved_to_sharded(h, self.shard)
        for layer in self.layers:
            kind, dim, heads = layer["kind"], layer["dim"], layer["heads"]
            attention = self.target_layer[kind].layer.self_attn
            qkv = self._linear(self._flat(self._norm(h, layer["input"])), layer["qkv"])
            q, _, _ = QkvHeadsNormRope.op(
                qkv,
                heads,
                1,
                dim,
                layer["q_gamma"],
                layer["k_gamma"],
                rope[kind].cos,
                rope[kind].sin,
                self.eps,
            )
            k_cache, v_cache = cache[self.cache_index[kind]]
            sdpa = attention.decode_sdpa
            attended = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                k_cache,
                v_cache,
                cur_pos_tensor=sdpa_pos,
                page_table_tensor=table,
                scale=1.0,
                sliding_window_size=sdpa.config.sliding_window if sdpa.config.is_sliding else None,
                program_config=sdpa.program,
                compute_kernel_config=sdpa.compute,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            projected = self._linear(ConcatHeadsDecode.op(attended, heads), layer["o"])
            reduced = self._reduce(projected, "mtp_attention")
            h = ttnn.add(h, self._norm(reduced, layer["post_attention"]), memory_config=self.shard)
            x = self._flat(self._norm(h, layer["pre_ff"]))
            gate = self._linear(x, layer["gate"])
            up = self._linear(x, layer["up"])
            mixed = ttnn.mul(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            reduced = self._reduce(self._linear(mixed, layer["down"]), "mtp_mlp")
            h = ttnn.add(
                h,
                self._norm(reduced, layer["post_ff"]),
                memory_config=self.shard,
                activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, layer["scalar"])],
            )
        normed = self._flat(self._norm(h, self.final_norm))
        # All 32 tile rows logical (row 0 is the draft): the argmax reductions then need no padding fills.
        rows = ttnn.reshape(normed, (1, 1, 32, self.hidden), (1, 1, 32, self.hidden))
        logits = self._linear(rows, self.lm_head, memory=ttnn.DRAM_MEMORY_CONFIG)
        next_hidden = self._linear(normed, self.post_projection, memory=ttnn.DRAM_MEMORY_CONFIG)
        return logits, next_hidden
