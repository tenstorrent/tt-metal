"""K2-Horizon TP4 autoregressive model over the reviewed multichip decoder.

Host work is restricted to checkpoint loading and public input preparation.
The residual remains rank-local through the entire decoder stack.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig

from .collective_buffers import DecodeCollectiveBuffers
from .full_model_policy import stage6_precision_policy
from .multichip_decoder import MultichipDecoder
from .optimized_decoder import PrecisionPolicy
from .optimized_full_model_policy import OptimizedFullModelDecoder
from .precision_config import layer_policy, load_precision_config

HF_MODEL = "IFM/K2-Horizon-7B"
HF_REVISION = "036114ce8d46c32b24c15423211069abb9c5d25e"


class Checkpoint:
    def __init__(self):
        self.config = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION, trust_remote_code=True)
        index = hf_hub_download(HF_MODEL, "model.safetensors.index.json", revision=HF_REVISION)
        self.mapping = json.loads(Path(index).read_text())["weight_map"]

    def load(self, prefix):
        names = [k for k in self.mapping if k.startswith(prefix)]
        result = {}
        for filename in sorted({self.mapping[k] for k in names}):
            path = hf_hub_download(HF_MODEL, filename, revision=HF_REVISION)
            with safe_open(path, framework="pt") as f:
                result.update({k: f.get_tensor(k) for k in names if self.mapping[k] == filename})
        return result


class K2Model:
    page_size = 32
    padded_vocab = 262144

    def __init__(
        self,
        mesh_device,
        *,
        override_num_layers=None,
        head_dtype=None,
        head_fidelity=None,
        head_split_size=8192,
        head_workers=1,
        head_k=4,
        decoder_policies=None,
        precision_config=None,
    ):
        self.mesh = mesh_device
        self.precision_config = load_precision_config(precision_config)
        precision = self.precision_config
        head_dtype = head_dtype or precision["head"]["weight_dtype"]
        head_fidelity = head_fidelity or precision["head"]["compute_fidelity"]
        self.logits_dtype = getattr(ttnn, precision["dtypes"]["logits"])
        self.token_dtype = getattr(ttnn, precision["dtypes"]["token_ids"])
        if list(mesh_device.shape) != [1, 4]:
            raise ValueError("K2-Horizon requires the reviewed TP4 MeshShape(1,4)")
        checkpoint = Checkpoint()
        self.config, self.tokenizer = checkpoint.config, checkpoint.tokenizer
        self.context = self.config.max_position_embeddings
        if self.context != precision["max_context"]:
            raise ValueError("Checkpoint and precision context contracts differ")
        self.vocab_size = self.config.vocab_size
        self.num_layers = override_num_layers or self.config.num_hidden_layers
        if not 1 <= self.num_layers <= self.config.num_hidden_layers:
            raise ValueError("Invalid layer count")
        if decoder_policies is None:
            decoder_policies = {}
        if not isinstance(decoder_policies, Mapping):
            raise TypeError("decoder_policies must map layer indices to PrecisionPolicy values")
        for index, policy in decoder_policies.items():
            if type(index) is not int or not 0 <= index < self.num_layers:
                raise ValueError("Decoder policy index is outside the constructed stack")
            if not isinstance(policy, PrecisionPolicy):
                raise TypeError("Decoder policies must be PrecisionPolicy values")
            if not policy.dram or not policy.prefill_fused:
                raise ValueError("The full model requires DRAM-sharded decode and fused prefill")
        self.pool = DecodeCollectiveBuffers(mesh_device)
        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(
                (MultichipDecoder if i in decoder_policies else OptimizedFullModelDecoder).from_state_dict(
                    checkpoint.load(f"model.layers.{i}."),
                    hf_config=self.config,
                    layer_idx=i,
                    mesh_device=mesh_device,
                    collective_buffers=self.pool,
                    policy=decoder_policies.get(i, layer_policy(precision, i)),
                    ccl_dtype=precision["dtypes"]["ccl"],
                    residual_dtype=precision["dtypes"]["residual"],
                    norm_fidelity=precision["accumulation"]["norm_fidelity"],
                    matmul_output_dtype=precision["dtypes"]["matmul_output"],
                    math_approx_mode=precision["accumulation"]["math_approx_mode"],
                    packer_l1_acc=precision["accumulation"]["packer_l1_acc"],
                )
            )
            print(f"K2 layer {i + 1}/{self.num_layers} loaded", flush=True)
        self.prefill_geometry = [
            self._prefill_descriptors(layer, decoder_policies.get(i, stage6_precision_policy(i)))
            for i, layer in enumerate(self.layers)
        ]
        emb = checkpoint.load("model.embed_tokens.")["model.embed_tokens.weight"]
        self.embedding = self.upload(
            emb, dtype=getattr(ttnn, precision["dtypes"]["embedding"]), layout=ttnn.ROW_MAJOR_LAYOUT, shard=-1
        )
        del emb
        gamma = checkpoint.load("model.norm.")["model.norm.weight"]
        self.norm_weight = self.upload(
            gamma.reshape(1, 1, 1, -1), dtype=getattr(ttnn, precision["dtypes"]["norm"]), shard=-1
        )
        head = checkpoint.load("lm_head.")["lm_head.weight"]
        head = torch.nn.functional.pad(head.T.contiguous(), (0, self.padded_vocab - self.vocab_size))
        self.head_dtype = head_dtype
        self.head_fidelity = head_fidelity
        if head_split_size not in (8192, 16384, 32768, 65536):
            raise ValueError("Unsupported LM-head split size")
        self.head = self.upload(head, dtype=getattr(ttnn, head_dtype), shard=-1)
        banks = mesh_device.dram_grid_size()
        bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks.x - 1, banks.y - 1))})
        self.head_input_memory = ttnn.create_sharded_memory_config(
            (32, 256),
            core_grid=ttnn.num_cores_to_corerangeset(16, mesh_device.compute_with_storage_grid_size(), True),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        self.head_program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=head_k, per_core_M=1, per_core_N=head_split_size // 1024, num_workers_per_dram_bank=head_workers
        )
        self.head_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, head_fidelity),
            math_approx_mode=precision["accumulation"]["math_approx_mode"],
            fp32_dest_acc_en=precision["accumulation"]["head_fp32"],
            packer_l1_acc=precision["accumulation"]["packer_l1_acc"],
        )
        head_parts = []
        bank_multiple = banks.x * 32 * head_workers
        physical_split = math.ceil(head_split_size / bank_multiple) * bank_multiple
        part_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(bank_grid, (4096, physical_split // banks.x), ttnn.ShardOrientation.ROW_MAJOR),
        )
        for offset in range(0, 65536, head_split_size):
            parts = torch.cat(
                [
                    torch.nn.functional.pad(
                        part[:, offset : offset + head_split_size], (0, physical_split - head_split_size)
                    )
                    for part in head.chunk(4, -1)
                ],
                -1,
            ).contiguous()
            head_parts.append(
                LazyWeight(
                    source=parts,
                    device=mesh_device,
                    dtype=getattr(ttnn, head_dtype),
                    mesh_mapper_config=ttnn.MeshMapperConfig(
                        placements=[ttnn.PlacementShard(-1)], mesh_shape_override=ttnn.MeshShape([4])
                    ),
                    memory_config=part_mem,
                    layout=ttnn.TILE_LAYOUT,
                )
            )
        self.head_decode = LMHead1D.from_config(
            LMHead1DConfig(
                output_weights=head_parts,
                mesh_device=mesh_device,
                dim=4096,
                program_configs=[self.head_program] * len(head_parts),
                weights_memcfgs=[part_mem] * len(head_parts),
                output_split_sizes=[head_split_size] * len(head_parts),
                input_memcfg=self.head_input_memory,
                output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                lm_head_dtype=self.logits_dtype,
                compute_kernel_config=self.head_compute,
            )
        )
        self.head_decode.load_device_weights()
        del head
        # Absolute-position lookup tables. FP32 generation matches HF exactly;
        # lookup and position advancement are device operations during decode.
        inv = 1.0 / (10000000.0 ** (torch.arange(0, 128, 2).float() / 128))
        angles = torch.outer(torch.arange(self.context).float(), inv)
        angles = torch.cat((angles, angles), -1)
        self.rope_tables = tuple(
            self.upload(
                fn(angles).bfloat16(), dtype=getattr(ttnn, precision["dtypes"]["rope"]), layout=ttnn.ROW_MAJOR_LAYOUT
            )
            for fn in (torch.cos, torch.sin)
        )

    def upload(self, x, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard=None, memory=None):
        return ttnn.from_torch(
            x.contiguous(),
            device=self.mesh,
            dtype=dtype,
            layout=layout,
            memory_config=memory or ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=(
                ttnn.ReplicateTensorToMesh(self.mesh) if shard is None else ttnn.ShardTensorToMesh(self.mesh, dim=shard)
            ),
        )

    def _prefill_descriptors(self, layer, policy):
        """Preserve reviewed short/unaligned prefill reduction geometry.

        The decoder also uses its DRAM-sharded programs for physical rows<=32
        and leading unaligned tokens. Those prefill reductions must not inherit
        tuning of the autoregressive decode programs. Weights and compute
        precision are shared; these are host-only immutable descriptors.
        """
        inputs, programs = {}, {}
        for role, weights in (
            ("qkv", (layer.wqkv,)),
            ("o", (layer.wo,)),
            ("mlp", (layer.wgate, layer.wup, layer.wgateup)),
            ("down", (layer.wdown,)),
        ):
            geometry = getattr(policy, role + "_geometry")
            for weight in weights:
                if weight is None:
                    continue
                key = id(weight)
                k, n = tuple(layer.decode_weights[key].shape)[-2:]
                inputs[key] = ttnn.create_sharded_memory_config(
                    (32, k // geometry.cores),
                    core_grid=ttnn.num_cores_to_corerangeset(
                        geometry.cores, self.mesh.compute_with_storage_grid_size(), True
                    ),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    use_height_and_width_as_shard_shape=True,
                )
                programs[key] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=geometry.block_w,
                    per_core_M=1,
                    per_core_N=n // (32 * geometry.cores),
                    num_workers_per_dram_bank=geometry.readers,
                )
        if policy.fused_gate:
            programs[id(layer.wgate)].fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
        return inputs, programs

    @contextmanager
    def _prefill_programs(self):
        original = [(layer.decode_inputs, layer.decode_programs) for layer in self.layers]
        try:
            for layer, (inputs, programs) in zip(self.layers, self.prefill_geometry):
                layer.decode_inputs, layer.decode_programs = inputs, programs
            yield
        finally:
            for layer, (inputs, programs) in zip(self.layers, original):
                layer.decode_inputs, layer.decode_programs = inputs, programs

    def allocate_cache(self, *, batch_size, capacity):
        if not 1 <= batch_size <= 32 or not 1 <= capacity <= self.context:
            raise ValueError("Cache requires batch 1..32 and capacity within the HF context")
        # Rounded SDPA read window owns initialized page-table entries, not only
        # allocator padding. Never expose an uninitialized page id to K128 SDPA.
        pages = math.ceil(capacity / 128) * 4
        table = torch.arange(batch_size * pages, dtype=torch.int32).reshape(batch_size, pages)
        # All ranks hold their own two KV heads. Mesh tensors represent local shape.
        shape = (batch_size * pages, 2, 32, 128)
        cache = [
            tuple(
                ttnn.zeros(
                    shape,
                    dtype=layer.kv_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            )
            for layer in self.layers
        ]
        return cache, table

    def final_logits(self, x, *, decode):
        first = self.layers[0]
        if decode:
            x = ttnn.to_memory_config(x, first.decode_residual_memory)
            x = ttnn.rms_norm(
                x,
                epsilon=first.eps,
                weight=self.norm_weight,
                program_config=first.decode_norm_program,
                compute_kernel_config=first.compute,
            )
        else:
            x = ttnn.rms_norm(x, epsilon=first.eps, weight=self.norm_weight, compute_kernel_config=first.compute)
        x = first._gather(x)
        # Decoder gather precision may be BFP8; retain the declared BF16
        # terminal norm/head input contract rather than changing head numerics.
        if x.dtype != self.norm_weight.dtype:
            x = ttnn.typecast(x, self.norm_weight.dtype)
        if decode:
            # Direct Tensor inputs do not consult LMHead1D's input_memcfg.
            # Use the head's measured 16-core working layout. Optimized decode
            # gathers to eight cores, so this boundary explicitly reshards it.
            return self.head_decode(ttnn.to_memory_config(x, self.head_input_memory))
        return ttnn.linear(
            x,
            self.head,
            compute_kernel_config=self.head_compute,
            dtype=self.logits_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def prefill_chunk(self, tokens, *, start_pos, page_table, kv_cache, last_only=False, cache_only=False):
        """One logical chunk for one slot. Caller owns chunking and prompt lengths."""
        length = len(tokens)
        tt_tokens = self.upload(
            torch.tensor(tokens).reshape(1, length), dtype=self.token_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        indices = self.upload(
            torch.arange(start_pos, start_pos + length).reshape(1, length),
            dtype=self.token_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        plan = self.layers[0].prepare_prefill(seq_len=length, start_pos=start_pos)
        return self.prefill_from_device(
            tt_tokens,
            indices,
            plan=plan,
            page_table=page_table,
            kv_cache=kv_cache,
            last_only=last_only,
            cache_only=cache_only,
        )

    def prefill_from_device(self, tokens, indices, *, plan, page_table, kv_cache, last_only=False, cache_only=False):
        """Prepared logical chunk; every input and plan tensor already resides on device."""
        length = plan.seq_len
        x = ttnn.reshape(ttnn.embedding(tokens, self.embedding, layout=ttnn.TILE_LAYOUT), (1, 1, length, 1024))
        rope = tuple(
            ttnn.reshape(ttnn.embedding(indices, table, layout=ttnn.TILE_LAYOUT), (1, 1, length, 128))
            for table in self.rope_tables
        )
        with self._prefill_programs():
            for layer, cache in zip(self.layers, kv_cache):
                x = layer.prefill_forward(x, rope=rope, kv_cache=cache, page_table=page_table, plan=plan)
            if cache_only:
                return None
            if last_only:
                x = x[:, :, -1:, :]
            return self.final_logits(x, decode=last_only)

    def decode(self, tokens, current_pos, rope_indices, *, page_table, kv_cache, batch_size):
        """Device-only trace body, stable token/position/page-table/cache inputs."""
        x = ttnn.embedding(ttnn.reshape(tokens, (1, 32)), self.embedding, layout=ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, (1, 1, 32, 1024))[:, :, :batch_size, :]
        rope = []
        for table in self.rope_tables:
            r = ttnn.embedding(rope_indices, table, layout=ttnn.TILE_LAYOUT)
            r = ttnn.reshape(r, (1, 1, 32, 128))[:, :, :batch_size, :]
            rope.append(ttnn.repeat(ttnn.reshape(r, (1, batch_size, 1, 128)), (1, 1, 32, 1)))
        pos = current_pos[:batch_size]
        for layer, cache in zip(self.layers, kv_cache):
            x = layer.decode_forward(x, rope=tuple(rope), kv_cache=cache, page_table=page_table, current_pos=pos)
        logits = self.final_logits(x, decode=True)
        # Sampling always uses one physical tile of fixed slots; inactive rows
        # have no valid cache position and are excluded from returned outputs.
        return self.sampler_logits(logits)

    @staticmethod
    def sampler_logits(logits):
        return (
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 32 - logits.shape[2]), (0, 0)], 0)
            if logits.shape[2] != 32
            else logits
        )
