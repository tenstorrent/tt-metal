# SPDX-License-Identifier: Apache-2.0
"""Optimized single-device Qwen3.8 text decoder; weights and request state are explicit.

Runtime tensors stay on device. Setup and reference comparisons belong to callers.
Prefill accepts [B,S,5120], decode [B,1,5120]. Full-attention callers supply
RoPE cos/sin for the absolute positions and a paged cache; linear attention
callers supply convolution and FP32 recurrent state. State is updated in place.
Validation evidence and limitations are in doc/optimized_decoder/README.md.
"""

from dataclasses import dataclass
import math
import copy

import ttnn

from models.common.lightweightmodule import LightweightModule
from .fp32_recurrence import _KERNEL, _UPSTREAM


# Explicit model/SKU policy; generic matmul reader default remains one.
DEFAULT_POLICY = {'dram_sharded': True,
 'fidelity': 'LoFi',
 'mlp_dtype': 'bfloat4_b',
 'attention_dtype': 'bfloat4_b',
 'down_dtype': 'bfloat4_b',
 'block': 4,
 'sharded_norm': True,
 'carry_residual': True,
 'kv_dtype': 'bfloat8_b',
 'packed_mlp': True,
 'prefill_chunk': 1024,
 'prefill_grid': [11, 10],
 'prefill_block': 10,
 'inplace_recurrence': True,
 'mix': {'cores': 80, 'block': 2, 'readers': 2},
 'gateup': {'cores': 80, 'block': 2, 'readers': 2},
 'out': {'cores': 64, 'block': 3, 'prefill_block': 32, 'readers': 2},
 'down': {'cores': 68, 'block': 8, 'prefill_block': 32, 'readers': 2},
 'l1_intermediates': True,
 'sdpa_cores': 32,
 'separate_prefill': True,
 'prefill_sharded_norm': True,
 'prefill_norm_block': True,
 'prefill_sdpa_q': 128,
 'prefill_sdpa_k': 256,
 'prefill_sdpa_approx': True,
 'prefill_subh': 2,
 'history_tile_tail': True,
 'policy_name': 'p300c_bfp4_lofi_phase_specific_v1',
 'gate': {'cores': 80, 'block': 2, 'readers': 2},
 'up': {'cores': 80, 'block': 2, 'readers': 2},
 'prefill_norm_max_tokens': 1024,
 'l1_mlp': True,
 'prefill_unit_subw': 3}

LAYER_OVERRIDES = {"linear_attention": {"prefill_chunk": 1280, "prefill_norm_max_tokens": 1280, "prefill_unit_subw": 5}}


def default_policy(layer_type):
    policy = copy.deepcopy(DEFAULT_POLICY)
    policy.update(LAYER_OVERRIDES.get(layer_type, {}))
    return policy


@dataclass
class LinearState:
    conv: object  # [B,3,10240], BF16 row-major
    recurrent: object  # [B,48,128,128], FP32


@dataclass
class PagedState:
    key: object  # [physical_pages,4,32,256], BFP8 by default
    value: object


def reshape(x, shape):
    """Preserve logical ordering across tile boundaries."""
    return ttnn.reshape(x, shape)


class OptimizedDecoder(LightweightModule):
    chunk_size = 128
    page_size = 32

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, policy=None):
        import torch

        config = getattr(hf_config, "text_config", hf_config)
        assert config.hidden_size == 5120 and config.intermediate_size == 17408
        assert config.num_attention_heads == 24 and config.num_key_value_heads == 4
        assert config.linear_num_key_heads == 16 and config.linear_num_value_heads == 48
        assert mesh_device.get_num_devices() == 1
        self = cls()
        self.config, self.mesh_device = config, mesh_device
        self.policy = copy.deepcopy(policy) if policy else default_policy(config.layer_types[layer_idx])
        self.weight_roles = {}
        self.logical_widths = {}
        # This build exports the shared Wormhole/Blackhole config under the
        # Wormhole name; ttnn.types.BlackholeComputeKernelConfig is its alias.
        self.matmul_ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.policy.get("fidelity", "HiFi2")),
            math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)
        self.layer_idx, self.layer_type = layer_idx, config.layer_types[layer_idx]
        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )
        prefix = f"model.language_model.layers.{layer_idx}."
        if not any(k.startswith(prefix) for k in state_dict):
            prefix = ""  # layer-only HF state dict

        def weight(name):
            return state_dict[prefix + name].detach().float()

        def put(x, dtype=ttnn.bfloat16, role=None):
            mem = ttnn.DRAM_MEMORY_CONFIG
            logical_width = x.shape[-1]
            if self.policy.get("dram_sharded") and x.ndim == 2 and min(x.shape) >= 1024:
                grid = mesh_device.dram_grid_size()
                cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x-1, grid.y-1))])
                readers = self.policy.get(role, {}).get("readers", self.policy.get("readers", 1))
                width = math.ceil(x.shape[1] / (32 * grid.x * readers)) * 32 * readers
                if readers > 1:
                    x = torch.nn.functional.pad(x, (0, width * grid.x - logical_width))
                mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(cores, [x.shape[0], width], ttnn.ShardOrientation.ROW_MAJOR))
            tensor = ttnn.from_torch(x.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                   device=mesh_device, memory_config=mem)
            self.logical_widths[id(tensor)] = logical_width
            return tensor

        def linear(name):
            group = "down_dtype" if "down_proj" in name else "mlp_dtype" if name.startswith("mlp") else "attention_dtype"
            role = "down" if "down_proj" in name else "gate" if "gate_proj" in name else "up" if "up_proj" in name else "out"
            return put(weight(name + ".weight").T, getattr(ttnn, self.policy.get(group, "bfloat8_b")), role=role)

        self.in_norm = put(1 + weight("input_layernorm.weight"))
        self.post_norm = put(1 + weight("post_attention_layernorm.weight"))
        self.gate, self.up = linear("mlp.gate_proj"), linear("mlp.up_proj")
        self.down = linear("mlp.down_proj")
        if self.policy.get("packed_mlp"):
            self.gateup = put(torch.cat([weight("mlp.gate_proj.weight").T, weight("mlp.up_proj.weight").T], dim=-1),
                getattr(ttnn, self.policy.get("mlp_dtype", "bfloat8_b")), role="gateup")
            if not self.policy.get("separate_prefill"):
                self.gate = self.up = None
        if self.layer_type == "linear_attention":
            eye = torch.eye(32).reshape(1, 1, 32, 32)
            masks = torch.zeros(1, 1, 32, 96)
            masks[..., :16, :16] = 1
            masks[..., 16:, 48:64] = 1
            masks[..., 16:, 64:80] = 1
            self.gdn_constants = dict(eye=put(eye, ttnn.float32),
                tril=put(torch.ones_like(eye).tril(), ttnn.float32),
                ones=put(torch.ones_like(eye), ttnn.float32), masks=put(masks, ttnn.float32))
            self.l2_weight = put(torch.full((128,), 128**-0.5), ttnn.float32)
            self.conv_start = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
            self.mix = put(torch.cat([weight("linear_attn.in_proj_qkv.weight").T,
                weight("linear_attn.in_proj_z.weight").T,
                torch.nn.functional.pad(weight("linear_attn.in_proj_a.weight").T, (0, 16)),
                torch.nn.functional.pad(weight("linear_attn.in_proj_b.weight").T, (0, 16))], dim=-1), getattr(ttnn, self.policy.get("attention_dtype", "bfloat8_b")), role="mix")
            self.out = linear("linear_attn.out_proj")
            conv = weight("linear_attn.conv1d.weight")[:, 0, :]
            self.taps = [put(conv[:, i].reshape(1, 1, -1)) for i in range(4)]
            self.neg_A = put(-weight("linear_attn.A_log").exp().reshape(1, 1, 48), ttnn.float32)
            self.dt = put(weight("linear_attn.dt_bias").reshape(1, 1, 48), ttnn.float32)
            self.out_norm = put(weight("linear_attn.norm.weight"))
        else:
            qg = weight("self_attn.q_proj.weight").reshape(24, 2, 256, 5120)
            self.mix = put(torch.cat([qg[:, 0].reshape(6144, 5120).T,
                weight("self_attn.k_proj.weight").T, weight("self_attn.v_proj.weight").T,
                qg[:, 1].reshape(6144, 5120).T], dim=-1), getattr(ttnn, self.policy.get("attention_dtype", "bfloat8_b")), role="mix")
            self.out = linear("self_attn.o_proj")
            self.q_norm = put(1 + weight("self_attn.q_norm.weight"))
            self.k_norm = put(1 + weight("self_attn.k_norm.weight"))
            self.chunk_positions = [
                ttnn.from_torch(torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
                for position in range(0, config.max_position_embeddings, self.chunk_size)
            ]
        if self.policy.get("split_mix"):
            if self.layer_type == "linear_attention":
                pieces = [weight("linear_attn.in_proj_qkv.weight").T,
                    torch.cat([weight("linear_attn.in_proj_z.weight").T,
                        torch.nn.functional.pad(weight("linear_attn.in_proj_a.weight").T,(0,16)),
                        torch.nn.functional.pad(weight("linear_attn.in_proj_b.weight").T,(0,16))],dim=-1)]
            else:
                pieces = [qg[:,0].reshape(6144,5120).T, weight("self_attn.k_proj.weight").T,
                    weight("self_attn.v_proj.weight").T, qg[:,1].reshape(6144,5120).T]
            self.mix = tuple(put(part, getattr(ttnn,self.policy.get("attention_dtype","bfloat8_b")), role="mix") for part in pieces)
        self.weight_roles = {id(getattr(self, role)): role for role in ("mix", "out", "gate", "up", "down")}
        if isinstance(self.mix, tuple):
            self.weight_roles.update({id(part): "mix" for part in self.mix})
        if self.policy.get("packed_mlp"):
            self.weight_roles[id(self.gateup)] = "gateup"
        return self

    def allocate_state(self, *, batch_size, context_length=None):
        """Setup only. Owners release state after all traces referencing it retire."""
        def zeros(shape, dtype):
            return ttnn.zeros(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                              device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.layer_type == "linear_attention":
            return LinearState(ttnn.zeros([batch_size, 3, 10240], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                               device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                               zeros([batch_size, 48, 128, 128], ttnn.float32))
        if context_length is None or not 1 <= context_length <= self.config.max_position_embeddings:
            raise ValueError("Supply a valid context_length for the paged pool")
        # Flexible chunked attention reads page-table rows in 32-byte sticks:
        # round physical capacity to eight INT32 page IDs, not logical context.
        pages = batch_size * ((context_length + 255) // 256) * 8
        return PagedState(zeros([pages, 4, 32, 256], getattr(ttnn, self.policy.get("kv_dtype", "bfloat16"))),
                          zeros([pages, 4, 32, 256], getattr(ttnn, self.policy.get("kv_dtype", "bfloat16"))))

    def snapshot(self, state):
        """Setup/boundary operation: retain an independent resume point on device."""
        if isinstance(state, LinearState):
            return LinearState(ttnn.clone(state.conv), ttnn.clone(state.recurrent))
        return PagedState(ttnn.clone(state.key), ttnn.clone(state.value))

    def _linear(self, x, w, activation=None, *, packed_output=False, decode_batch=None):
        if isinstance(w, tuple):
            return ttnn.concat([self._linear(x, part) for part in w], dim=-1)
        role = self.weight_roles[id(w)]
        logical_n = self.logical_widths[id(w)]
        group = "mlp" if role in ("gate", "up", "gateup", "down") else "attention"
        cfg = self.policy.get(role, {})
        activation_name = self.policy.get(group + "_activation", "bfloat16")
        if self.policy.get("_prefill_phase"):
            activation_name = cfg.get("prefill_activation", activation_name)
        activation_dtype = getattr(ttnn, activation_name)
        if x.dtype != activation_dtype:
            x = ttnn.typecast(x, activation_dtype)
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, cfg.get("fidelity", self.policy.get("fidelity", "HiFi2"))),
            math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)
        act = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation else None
        if self.policy.get("dram_sharded") and (x.shape[1] == 1 or decode_batch is not None):
            batch, k = decode_batch or x.shape[0], x.shape[-1]
            x = ttnn.reshape(x, [1, 1, batch, k])
            cores = cfg.get("cores", self.policy.get("cores", 8))
            grid = ttnn.num_cores_to_corerangeset(cores, self.mesh_device.compute_with_storage_grid_size(), row_wise=True)
            assert k % (cores * 32) == 0
            shard_width = k // cores
            mem = ttnn.create_sharded_memory_config([math.ceil(batch/32)*32, shard_width], grid,
                ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
            x = ttnn.to_memory_config(x, mem)
            pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=cfg.get("block", self.policy.get("block", shard_width//32)),
                per_core_M=math.ceil(batch/32), per_core_N=math.ceil(w.shape[-1]/(32*cores)),
                fused_activation=act, num_workers_per_dram_bank=cfg.get("readers", self.policy.get("readers", 1)))
            y = ttnn.linear(x, w, program_config=pc, compute_kernel_config=ckc,
                dtype=ttnn.bfloat16, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
            if not (self.policy.get("carry_residual") and batch == 1 and role in (("out", "down", "gate", "up", "gateup") if self.policy.get("l1_mlp") else ("out", "down"))):
                y = ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG if self.policy.get("l1_intermediates") and role in ("mix", "gate", "up", "gateup") else ttnn.DRAM_MEMORY_CONFIG)
            if logical_n != w.shape[-1]:
                y = y[..., :logical_n]
            return ttnn.reshape(y, [1, batch, logical_n] if packed_output else [batch, 1, logical_n])
        use_l1 = self.policy.get("prefill_l1") and math.prod(list(x.shape)[:-1]) <= self.policy.get("prefill_l1_max_tokens", 512)
        if self.policy.get("_prefill_phase") and cfg.get("prefill_l1"):
            use_l1 = math.prod(list(x.shape)[:-1]) <= cfg.get("prefill_l1_max_tokens", 4096)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG if use_l1 else ttnn.DRAM_MEMORY_CONFIG)
        kwargs = {}
        if self.policy.get("dram_sharded") or self.policy.get("prefill_config"):
            gx, gy = cfg.get("prefill_grid", self.policy.get("prefill_grid", [8, 4]))
            m, n = math.ceil(x.shape[-2]/32), math.ceil(w.shape[-1]/32)
            pn = math.ceil(n/gx)
            sub = next(v for v in (4, 2, 1) if pn % v == 0)
            candidate_sub = self.policy.get("prefill_unit_subw", 1)
            if sub == 1 and pn % candidate_sub == 0:
                sub = candidate_sub
            sub = cfg.get('prefill_out_subblock_w', sub)
            while pn % sub:
                sub -= 1
            pm = math.ceil(m/gy)
            block_h = min(cfg.get('prefill_out_block_h', self.policy.get('prefill_out_block_h', pm)), pm)
            while pm % block_h:
                block_h -= 1
            sh = min(cfg.get("prefill_subh", self.policy.get("prefill_subh", 1)), block_h)
            while block_h % sh or sh * sub > 8:
                sh -= 1
            kwargs["program_config"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx,gy), in0_block_w=cfg.get("prefill_block", self.policy.get("prefill_block", 4)),
                out_subblock_h=sh, out_subblock_w=sub, per_core_M=pm, per_core_N=pn,
                transpose_mcast=False, fused_activation=act, fuse_batch=False,
                out_block_h=block_h)
        else:
            kwargs["activation"] = activation
            kwargs["core_grid"] = ttnn.CoreGrid(x=11,y=10)
        y = ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, **kwargs)
        return y[..., :logical_n] if logical_n != w.shape[-1] else y

    def _norm(self, x, w):
        if self.policy.get("sharded_norm") and x.shape[-1] == 5120 and (x.shape[-2] == 1 or (self.policy.get("prefill_sharded_norm") and math.prod(list(x.shape)[:-1]) <= self.policy.get("prefill_norm_max_tokens", 512))):
            shape = list(x.shape)
            batch = math.prod(shape[:-1])
            x = ttnn.reshape(x, [1, 1, batch, 5120])
            if shape[-2] > 1 and self.policy.get("prefill_norm_block"):
                mt = math.ceil(batch / 32)
                if mt > 8:
                    mt = math.ceil(mt / 8) * 8
                gy = min(8, mt)
                padded_rows = mt * 32
                if padded_rows != batch:
                    x = ttnn.pad(x, [(0,0), (0,0), (0,padded_rows-batch), (0,0)], value=0)
                shard_h = mt // gy * 32
                mem = ttnn.create_sharded_memory_config([shard_h, 640], ttnn.CoreGrid(x=8, y=gy),
                    ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
                pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(8,gy), subblock_w=4,
                    block_h=shard_h//32, block_w=20, inplace=False)
                x = ttnn.to_memory_config(x, mem)
            else:
                nc = self.policy.get("norm_cores", 8) if shape[-2] == 1 else self.policy.get("prefill_norm_cores", 80)
                if shape[-2] > 1:
                    # Secondary norm reducers traverse the bounding rectangle;
                    # avoid nonrectangular prefixes with uninitialized peers.
                    device_grid = self.mesh_device.compute_with_storage_grid_size()
                    nx = next(v for v in range(device_grid.x, 0, -1) if nc % v == 0 and nc // v <= device_grid.y)
                    norm_grid = (nx, nc // nx)
                    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(nx-1,nc//nx-1))])
                else:
                    norm_grid = (min(nc,11),math.ceil(nc/11))
                    grid = ttnn.num_cores_to_corerangeset(nc, self.mesh_device.compute_with_storage_grid_size(), row_wise=True)
                mem = ttnn.create_sharded_memory_config([math.ceil(batch/32)*32, 5120//nc], grid,
                    ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
                x = ttnn.to_memory_config(x, mem)
                pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=norm_grid, subblock_w=1,
                    block_h=math.ceil(batch/32), block_w=160//nc, inplace=False)
            x = ttnn.rms_norm(x, weight=w, epsilon=self.config.rms_norm_eps, program_config=pc,
                memory_config=mem, compute_kernel_config=self.ckc)
            if not (self.policy.get("carry_residual") and batch == 1):
                use_l1 = shape[-2] > 1 and self.policy.get("prefill_l1") and batch <= self.policy.get("prefill_l1_max_tokens", 512)
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG if use_l1 else ttnn.DRAM_MEMORY_CONFIG)
            if x.shape[-2] != batch:
                x = x[..., :batch, :]
            return ttnn.reshape(x,shape)
        return ttnn.rms_norm(x, weight=w, epsilon=self.config.rms_norm_eps,
                             compute_kernel_config=self.ckc)

    def _finish(self, residual, mixer):
        if mixer.is_sharded():
            residual = ttnn.to_memory_config(residual, mixer.memory_config())
        x = ttnn.add(residual, mixer, memory_config=mixer.memory_config())
        h = self._norm(x, self.post_norm)
        compact = self.policy.get("packed_mlp") and h.shape[1] == 1 and h.shape[0] > 1 and not self.policy.get("minimal_mlp")
        if self.policy.get("packed_mlp") and (h.shape[1] == 1 or not self.policy.get("separate_prefill")):
            if self.policy.get("minimal_mlp") and (h.shape[1] > 1 or self.policy.get("minimal_decode")):
                grid = self.mesh_device.compute_with_storage_grid_size()
                inter = ttnn.experimental.minimal_matmul(ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG),
                    self.gateup, fuse_swiglu=True,
                    config=ttnn.MinimalMatmulConfig(M_block_size=min(self.policy.get("minimal_m", 4), math.ceil(h.shape[-2]/32)),
                        K_block_size=self.policy.get("minimal_k", 8), N_block_size=self.policy.get("minimal_n", 8),
                        subblock_h=1, subblock_w=2, compute_with_storage_grid_size=grid),
                    compute_kernel_config=self.matmul_ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16)
            else:
                gu = self._linear(h, self.gateup, packed_output=compact)
                inter = ttnn.multiply(gu[..., :17408], gu[..., 17408:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                    memory_config=ttnn.L1_MEMORY_CONFIG if self.policy.get("l1_intermediates") and h.shape[1] == 1 else ttnn.DRAM_MEMORY_CONFIG)
        else:
            gate = self._linear(h, self.gate, activation="silu")
            inter = ttnn.multiply(gate, self._linear(h, self.up))
        mlp_decode_batch = h.shape[0] if compact else None
        if self.policy.get("prefill_l1"):
            del h
        mlp = self._linear(inter, self.down, decode_batch=mlp_decode_batch)
        if mlp.is_sharded():
            x = ttnn.to_memory_config(x, mlp.memory_config())
        return ttnn.add(x, mlp, memory_config=mlp.memory_config())

    def _delta(self, x, state):
        batch, length, _ = x.shape
        projected = self._linear(x, self.mix)
        physical_conv = ((length + 31) // 32) * 32
        padded_projected = ttnn.pad(projected, [(0, 0), (0, physical_conv-length), (0, 0)], value=0) if physical_conv != length else projected
        qs, ks, vs = [], [], []
        for bidx in range(batch):
            hist = ttnn.to_layout(state.conv[bidx:bidx+1], ttnn.ROW_MAJOR_LAYOUT)
            cq, ck, cv = ttnn.experimental.kda.qkv_causal_conv1d_silu(
                padded_projected[bidx:bidx+1], hist, *self.taps, 2048, 2048, 6144,
                actual_start=self.conv_start, predecessor_carry=hist,
                program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False))
            qs.append(cq[:, :length]); ks.append(ck[:, :length]); vs.append(cv[:, :length])
        qflat, kflat, vflat = [parts[0] if batch == 1 else ttnn.concat(parts, dim=0) for parts in (qs, ks, vs)]
        if self.policy.get("history_tile_tail") and length >= 32 and length % 32 == 0:
            # Slice a complete tile before untilize, then select logical rows.
            # Direct nonaligned tiled slicing would add an implicit retile.
            tail = projected[:, length-32:length, :10240]
            next_conv = ttnn.to_layout(tail, ttnn.ROW_MAJOR_LAYOUT)[:, -3:, :]
        else:
            next_conv = ttnn.to_layout(projected, ttnn.ROW_MAJOR_LAYOUT)[:, length-3:length, :10240] if length >= 3 else ttnn.concat(
                [state.conv[:, length:, :], ttnn.to_layout(projected[..., :10240], ttnn.ROW_MAJOR_LAYOUT)], dim=1)
        ttnn.copy(next_conv, state.conv)

        def l2(t):
            t = ttnn.typecast(t, ttnn.float32)
            return ttnn.rms_norm(t, weight=self.l2_weight, epsilon=1e-6/128, compute_kernel_config=self.ckc)

        a = ttnn.typecast(projected[..., 16384:16432], ttnn.float32)
        decay = ttnn.multiply(self.neg_A, ttnn.softplus(ttnn.add(a, self.dt)))
        beta = ttnn.sigmoid(ttnn.typecast(projected[..., 16448:16496], ttnn.float32))
        if length > 1:
            physical = ((length + 31) // 32) * 32
            def padded(t):
                return ttnn.pad(t, [(0, 0), (0, physical-length), (0, 0)], value=0) if physical != length else t
            chunk_outputs, chunk_states = [], []
            kernel_batch = (self.mesh_device.compute_with_storage_grid_size().x * self.mesh_device.compute_with_storage_grid_size().y) // 48
            pq, pk, pv, pg, pb = map(padded, (qflat, kflat, vflat, decay, beta))
            for begin in range(0, batch, kernel_batch):
                end = min(begin + kernel_batch, batch)
                out, final = ttnn.transformer.chunk_gated_delta_rule(
                    pq[begin:end], pk[begin:end], pv[begin:end], pg[begin:end], pb[begin:end],
                    initial_state=state.recurrent[begin:end], output_final_state=True,
                    chunk_size=32, output_head_major=True, compute_kernel_config=self.ckc, **self.gdn_constants)
                chunk_outputs.append(out); chunk_states.append(final)
            y = chunk_outputs[0] if len(chunk_outputs) == 1 else ttnn.concat(chunk_outputs, dim=0)
            new_state = chunk_states[0] if len(chunk_states) == 1 else ttnn.concat(chunk_states, dim=0)
            norm_gate = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                ttnn.typecast(y, ttnn.bfloat16), padded_projected, self.out_norm, 48,
                gate_column_offset=10240, epsilon=self.config.rms_norm_eps, output_dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False))
            ttnn.copy(new_state, state.recurrent)
            return self._linear(ttnn.multiply(norm_gate[:, :length], projected[..., 10240:16384]), self.out)
        else:
            q = l2(reshape(qflat, [batch, length, 16, 128]))
            k = l2(reshape(kflat, [batch, length, 16, 128]))
            v = ttnn.typecast(reshape(vflat, [batch, length, 48, 128]), ttnn.float32)
            # This kernel assigns one value head to each core. Split its batch axis,
            # retaining batched projections and an explicit state for every request.
            grid = self.mesh_device.compute_with_storage_grid_size()
            kernel_batch = (grid.x * grid.y) // 48
            ys, states = [], []
            for begin in range(0, batch, kernel_batch):
                end = min(begin + kernel_batch, batch)
                y_part, state_part = fp32_recurrence(
                    q[begin:end], k[begin:end], v[begin:end], decay[begin:end], beta[begin:end],
                    initial_state=state.recurrent[begin:end], output_final_state=True, compute_kernel_config=self.ckc,
                    inplace_state=self.policy.get("inplace_recurrence", False),
                )
                ys.append(y_part)
                states.append(state_part)
            y = ttnn.concat(ys, dim=0) if len(ys) > 1 else ys[0]
            new_state = ttnn.concat(states, dim=0) if len(states) > 1 else states[0]
        if new_state.buffer_address() != state.recurrent.buffer_address():
            ttnn.copy(new_state, state.recurrent)
        y = self._norm(ttnn.typecast(y, ttnn.bfloat16), self.out_norm)
        z = reshape(projected[..., 10240:16384], [batch, length, 48, 128])
        return self._linear(reshape(ttnn.multiply(z, y, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), [batch, length, 6144]), self.out)

    def _rope(self, x, cos, sin):
        rot, rest = x[..., :64], x[..., 64:]
        pieces = [ttnn.experimental.rotary_embedding(rot[b:b+1], cos[b:b+1], sin[b:b+1],
                    compute_kernel_config=self.ckc) for b in range(x.shape[0])]
        rotated = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=0)
        return ttnn.concat([rotated[..., :x.shape[-2], :], rest], dim=-1)

    def _attention(self, x, state, *, page_table, cos, sin, current_pos=None, chunk_start=0):
        batch, length, _ = x.shape
        projected = self._linear(x, self.mix)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            projected[..., :8192], num_heads=24, num_kv_heads=4, transpose_key=False)
        q = self._rope(self._norm(q, self.q_norm), cos, sin)
        k = self._rope(self._norm(k, self.k_norm), cos, sin)
        if current_pos is not None:
            # Decode kernel layout is [1,B,H,D]. Cache update requires height sharding.
            q = ttnn.permute(q, (2, 0, 1, 3))
            k, v = ttnn.permute(k, (2, 0, 1, 3)), ttnn.permute(v, (2, 0, 1, 3))
            mem = ttnn.create_sharded_memory_config(
                [32, 256], core_grid=ttnn.num_cores_to_corerangeset(batch, self.mesh_device.compute_with_storage_grid_size(), row_wise=True),
                strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            for cache, update in ((state.key, k), (state.value, v)):
                ttnn.experimental.paged_update_cache(cache, ttnn.to_memory_config(update, mem),
                                                     update_idxs_tensor=current_pos, page_table=page_table)
            y = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q, state.key, state.value, page_table_tensor=page_table, cur_pos_tensor=current_pos,
                scale=256**-0.5, compute_kernel_config=self.ckc,
                # The default is 32 tokens and one core/head: 8192 serial BF16
                # softmax-stat updates at full context. Bound the accumulation
                # chain with larger chunks and sequence-parallel cores.
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                    q_chunk_size=32, k_chunk_size=self.policy.get("sdpa_chunk", 256), max_cores_per_head_batch=self.policy.get("sdpa_cores", 16),
                    exp_approx_mode=False,
                ),
            )
            y = reshape(y, [batch, 1, 6144])
        else:
            fill_table = page_table[:, chunk_start // 32:]
            for user in range(batch):
                ttnn.experimental.paged_fill_cache(state.key, ttnn.typecast(k[user:user + 1], state.key.dtype), fill_table, batch_idx=user)
                ttnn.experimental.paged_fill_cache(state.value, ttnn.typecast(v[user:user + 1], state.value.dtype), fill_table, batch_idx=user)
            sdpa_options = {}
            if self.policy.get("prefill_sdpa_q"):
                qc = 32
                while qc * 2 <= min(self.policy["prefill_sdpa_q"], length):
                    qc *= 2
                while length % qc or (chunk_start and chunk_start % qc):
                    qc //= 2
                kc = self.policy.get("prefill_sdpa_k", 128)
                while chunk_start and chunk_start % kc:
                    kc //= 2
                sdpa_options["program_config"] = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                    q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=self.policy.get("prefill_sdpa_approx", False))
            y = ttnn.transformer.chunked_scaled_dot_product_attention(
                q, state.key, state.value, page_table_tensor=page_table, chunk_start_idx=None,
                chunk_start_idx_tensor=self.chunk_positions[chunk_start // self.chunk_size],
                compute_kernel_config=self.ckc, **sdpa_options,
            )
            y = ttnn.transformer.concatenate_heads(y)
        # Matches the installed HF reference. Gate semantics are independently audited.
        y = ttnn.multiply(projected[..., 8192:], y, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return self._linear(y, self.out)

    def decode_forward(self, hidden_states, *, state, page_table=None, current_pos=None, cos=None, sin=None,
                       active_mask=None):
        """One token per request; current_pos is a device INT32 tensor, never a host list."""
        assert hidden_states.shape[1] == 1
        h = self._norm(hidden_states, self.in_norm)
        if self.layer_type == "linear_attention":
            if active_mask is None:
                y = self._delta(h, state)
            else:
                if not getattr(self, "supports_prepared_prefill", False):
                    raise NotImplementedError("Inactive decode rows require masked-state support")
                y = self._delta(h, state, active_mask=active_mask)
        else:
            # Paged update and SDPA already skip rows whose current_pos is -1.
            y = self._attention(h, state, page_table=page_table, current_pos=current_pos, cos=cos, sin=sin)
        return self._finish(hidden_states, y)

    def prefill_forward(self, hidden_states, *, state, start_position=0, page_table=None, cos=None, sin=None,
                        prepared_metadata=None):
        """Chunked [B,S,H] prefill, all logical outputs returned; resume at a chunk boundary.

        cos/sin: [B,1,S,64] for absolute positions start_position..start_position+S.
        Linear state summarizes precisely the prefix; paged state owns the prefix pages.
        """
        batch, length, _ = hidden_states.shape
        if not 0 < length or start_position + length > self.config.max_position_embeddings:
            raise ValueError("Invalid logical sequence extent")
        if start_position % self.chunk_size:
            raise ValueError("Prefill continuation starts at a chunk boundary; decode handles individual tokens")
        if prepared_metadata is not None:
            # The full-model owner pads one physical chunk and supplies logical
            # validity as tensor data. No logical-length slices belong here.
            if not getattr(self, "supports_prepared_prefill", False):
                raise NotImplementedError("Prepared chunks require a decoder with masked-state support")
            if length != 1 and (length % 32 or length > 4096):
                raise ValueError("Prepared prefill expects physical length 1 or a 32-token bucket through4096")
            required = ("valid_mask", "history_indices", "fill_page_table", "chunk_start")
            if any(name not in prepared_metadata for name in required):
                raise ValueError("Prepared prefill metadata is incomplete")
            if tuple(prepared_metadata["valid_mask"].shape) != (batch, length, 1):
                raise ValueError("Prepared validity mask must match physical [B,P,1]")
            x = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
            h = self._norm(x, self.in_norm)
            if self.layer_type == "linear_attention":
                y = self._delta(h, state, prepared_metadata=prepared_metadata)
            else:
                c, s = cos, sin
                if length == 1:
                    # Preserve the inherited one-token prefill algorithm. Its
                    # attention kernel still owns a physical32-row query tile.
                    h = ttnn.pad(h, [(0, 0), (0, 31), (0, 0)], value=0)
                    c = ttnn.pad(c, [(0, 0), (0, 0), (0, 31), (0, 0)], value=0)
                    s = ttnn.pad(s, [(0, 0), (0, 0), (0, 31), (0, 0)], value=0)
                y = self._attention(h, state, page_table=page_table, cos=c, sin=s,
                                    prepared_metadata=prepared_metadata)
                if length == 1:
                    y = y[:, :1, :]
            if self.policy.get("prefill_l1"):
                del h
            return ttnn.to_memory_config(self._finish(x, y), ttnn.DRAM_MEMORY_CONFIG)
        outputs = []
        work_chunk = self.policy.get("prefill_chunk", self.chunk_size)
        for offset in range(0, length, work_chunk):
            n = min(work_chunk, length - offset)
            x = ttnn.to_layout(hidden_states[:, offset:offset + n, :], ttnn.TILE_LAYOUT)
            h = self._norm(x, self.in_norm)
            if self.layer_type == "linear_attention":
                y = self._delta(h, state)
            else:
                c, s = cos[:, :, offset:offset + n, :], sin[:, :, offset:offset + n, :]
                physical = ((n + 31) // 32) * 32
                if physical != n:
                    h = ttnn.pad(h, [(0, 0), (0, physical - n), (0, 0)], value=0)
                    c = ttnn.pad(c, [(0, 0), (0, 0), (0, physical - n), (0, 0)], value=0)
                    s = ttnn.pad(s, [(0, 0), (0, 0), (0, physical - n), (0, 0)], value=0)
                y = self._attention(h, state, page_table=page_table, cos=c, sin=s, chunk_start=start_position + offset)
                y = y[:, :n, :]
            if self.policy.get("prefill_l1"):
                del h
            outputs.append(ttnn.to_memory_config(self._finish(x, y), ttnn.DRAM_MEMORY_CONFIG))
        # Bound each concatenation's fan-in for advertised-context prompts.
        while len(outputs) > 1:
            outputs = [ttnn.concat(outputs[i:i + 16], dim=1) if len(outputs[i:i + 16]) > 1 else outputs[i]
                       for i in range(0, len(outputs), 16)]
        return outputs[0]


def fp32_recurrence(q, k, v, g, beta, *, initial_state, scale=None, output_final_state=True,
                    compute_kernel_config=None, inplace_state=False):
    """[B,T,Hq,128], [B,T,48,128], log g/beta [B,T,48] -> output, FP32 state.

    Only final-state mode is supported. The compute policy is fixed HiFi4/FP32;
    the optional compute_kernel_config argument preserves call-site compatibility.
    """
    batch, length, qheads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    if key_dim != 128 or value_dim != 128 or value_heads != 48 or qheads not in (16, 48):
        raise ValueError("FP32 recurrence supports the Qwen3.8-27B head dimensions only")
    if not output_final_state or initial_state is None:
        raise ValueError("An explicit initial state and final state output are required")
    device = q.device()
    grid = device.compute_with_storage_grid_size()
    heads = batch * value_heads
    if heads > grid.x * grid.y:
        raise ValueError("Split recurrence batches to at most the available cores / 48")
    for x in (q, k, v, g, beta, initial_state):
        if x.dtype != ttnn.float32:
            raise ValueError("Recurrence inputs and state must be FP32")

    def split(x, h, width):
        x = ttnn.reshape(ttnn.permute(x, (0, 2, 1, 3)), [batch * h, length, width])
        if h != value_heads:
            x = ttnn.repeat_interleave(x, value_heads // h, dim=0)
        return reshape(x, [heads * length, 1, width]) if length > 1 else x

    q = split(ttnn.multiply(q, key_dim ** -0.5 if scale is None else scale), qheads, key_dim)
    k = split(k, qheads, key_dim)
    v = split(v, value_heads, value_dim)

    def split_scalar(x):
        x = ttnn.reshape(ttnn.permute(x, (0, 2, 1)), [heads, length])
        return reshape(x, [heads * length, 1, 1])

    # Exponentiation remains the public FP32 TTNN op, as in the upstream wrapper.
    decay = split_scalar(ttnn.exp(g))
    beta = split_scalar(beta)
    state = ttnn.reshape(initial_state, [heads, key_dim, value_dim])
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([heads * length, 1, value_dim]), ttnn.float32,
                                        ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    # Each head reads its entire initial state once, before computing final state.
    # Final-only writer writes that same disjoint head after its computation.
    final = state if inplace_state else ttnn.allocate_tensor_on_device(
        ttnn.Shape([heads, key_dim, value_dim]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    coords = [ttnn.CoreCoord(h // grid.y, h % grid.y) for h in range(heads)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    kt, vt, tile_bytes = 4, 4, 4096
    cb_sizes = [kt, kt, vt, 2 if length > 1 else 1, 2 if length > 1 else 1,
                kt * vt, 2 * vt, 2 * kt * vt, kt * vt, kt * vt, kt * vt, vt, vt, kt, kt * vt, kt * vt]
    cbs = [ttnn.CBDescriptor(total_size=tiles * tile_bytes, core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index,
                                     data_format=ttnn.float32, page_size=tile_bytes)])
           for index, tiles in enumerate(cb_sizes)]
    reader_ct = [kt, vt, 0]
    # Ring indexing is disabled. A valid dummy accessor keeps the unused template
    # argument well-formed without any scratch buffer or new persistent state.
    for tensor in (q, k, v, decay, beta, state, state):
        reader_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    writer_ct = [kt, vt, 0]
    for tensor in (out, final):
        writer_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    rr, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for h, core in enumerate(coords):
        rr[core.x][core.y] = [h, length, q.buffer_address(), k.buffer_address(), v.buffer_address(),
                              decay.buffer_address(), beta.buffer_address(), state.buffer_address(), 0, 0, heads]
        wr[core.x][core.y] = [h, length, out.buffer_address(), final.buffer_address(), heads]
        cr[core.x][core.y] = [length]
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                          fp32_dest_acc_en=True, math_approx_mode=False)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    # Only copy/SFPU consumers get direct-to-DST FP32. CB 14/15 are separate
    # packed copies of state/sd for matmul; q/k/u/kcol remain matrix operands.
    for index in (2, 3, 4, 5, 8, 9, 10, 11):
        modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    descriptor = ttnn.ProgramDescriptor(kernels=[
        ttnn.KernelDescriptor(kernel_source=_UPSTREAM + "reader_fused_recurrent_gated_delta_rule.cpp",
                             core_ranges=cores, compile_time_args=reader_ct, runtime_args=rr,
                             config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=_UPSTREAM + "writer_fused_recurrent_gated_delta_rule.cpp",
                             core_ranges=cores, compile_time_args=writer_ct, runtime_args=wr,
                             config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=_KERNEL, core_ranges=cores, compile_time_args=[kt, vt, 0],
                             runtime_args=cr, config=config),
    ], cbs=cbs, semaphores=[])
    ttnn.generic_op([q, k, v, decay, beta, state, out, final], descriptor)
    out = ttnn.permute(ttnn.reshape(out, [batch, value_heads, length, value_dim]), (0, 2, 1, 3))
    return out, ttnn.reshape(final, [batch, value_heads, key_dim, value_dim])
