# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Four-chip Qwen decoder candidates; acceptance is recorded in the stage README.

Weights are transformed only during construction. Residuals own one complete
hyperconnection stream per chip; request state is explicit and device resident.
"""

import math
from dataclasses import replace

import ttnn

from .causal_indexer_scores import causal_indexer_scores
from .chunk_output import write_token_chunk
from .decoder_components import WeightLoader, linear, rms
from .dram_projection import DramProjectionConfig, PhaseDramProjection, PhaseDramProjectionGroup
from .exact_projection_q1 import exact_projection_q1
from .exact_rank_sum import exact_rank_sum
from .exact_residual_write import exact_residual_write
from .functional_decoder import LinearState
from .fused_decoder import (
    _CONV_EXACT_COMPUTE,
    _CONV_EXACT_READER,
    _CONV_EXACT_WRITER,
    FusedChunkedQSA,
    FusedExactQKV,
    FusedGatedDeltaNet,
    FusedPreciseProjection,
    fused_selected_token_ids,
    merged_rms,
)
from .gdn_recurrence import _KERNEL_ROOT
from .ngram_hash import EOS, DeviceNGramHash
from .optimized_decoder import OptimizedDecoder
from .ple import PLE
from .qsa import QSA, QSAState
from .qsa_chunked import ChunkedQSA

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG

# Selected with real-weight whole-decoder TP4 runs. Both logical phases share
# immutable bank-sharded weights; every public count is padded/sliced internally.
MESH_DRAM_POLICIES = {
    "gr": DramProjectionConfig(cores=8, readers=3, block=5, dtype="bfloat16", packed=False),
    "gdn_zab": DramProjectionConfig(cores=8, readers=2, block=2, dtype="bfloat16", packed=True),
    "shared": DramProjectionConfig(cores=8, readers=3, block=5, dtype="bfloat8_b", packed=True),
    "qsa_qkv": DramProjectionConfig(cores=8, readers=3, block=2, dtype="bfloat8_b", packed=True),
    "qsa_out": DramProjectionConfig(cores=8, readers=3, block=2, dtype="bfloat8_b", packed=False),
}


def mesh_projection_group(loader, parts, role):
    policy = MESH_DRAM_POLICIES[role]
    precision = getattr(loader, "precision", None)
    if precision is not None:
        policy = replace(
            policy,
            dtype=precision["weight_groups"][role],
            fidelity=precision["compute_fidelities"][role],
            activation_dtype=precision["activation_dtype"],
        )
    result = PhaseDramProjectionGroup(loader, parts, {"decode": policy, "rows": policy})
    if hasattr(loader, "precision_groups"):
        loader.precision_groups.setdefault(role, []).append(result)
    return result


def mesh_projection(loader, parts, role):
    return PhaseDramProjection(mesh_projection_group(loader, parts, role), None)


class MeshLoader(WeightLoader):
    def shards(self, parts, dim=-1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        import torch

        dim %= parts[0].ndim
        return ttnn.from_torch(
            torch.cat(parts, dim=dim).contiguous(),
            dtype=dtype,
            layout=layout,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=dim),
            memory_config=DRAM,
        )

    def tensor(self, value, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            value.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )


class Collectives:
    """Serial CQ0 collective contract; callers prepare every physical shape."""

    def __init__(self, device, dtype="bfloat16"):
        self.device = device
        self.dtype = getattr(ttnn, dtype)
        self._sampling_ccl = None

    def prepare_sampling_collectives(self):
        """Own common async-sampling semaphores; call during sampler warmup.

        Decoder gather/sum retain their existing synchronous implementation.
        Lazy ownership avoids allocating unused pools in every decoder layer.
        The provider stays alive with this wrapper for the lifetime of traces;
        do not use the process-global device-id cache across mesh open/close.
        """
        if self._sampling_ccl is None:
            from models.common.tt_ccl import TT_CCL

            self._sampling_ccl = TT_CCL(self.device)
        return self._sampling_ccl

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        return self.prepare_sampling_collectives().get_and_cycle_barrier_semaphore_handle(cluster_axis)

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        return self.prepare_sampling_collectives().get_and_cycle_ag_semaphore_handles(cluster_axis)

    def get_num_links(self, cluster_axis=None):
        return self.prepare_sampling_collectives().get_num_links(cluster_axis)

    def gather(self, x, dim=-1):
        original_dtype = x.dtype
        if self.dtype == ttnn.bfloat8_b and original_dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT:
            x = ttnn.typecast(x, self.dtype)
        result = ttnn.all_gather(x, dim=dim % len(x.shape), memory_config=DRAM)
        return ttnn.typecast(result, original_dtype) if result.dtype != original_dtype else result

    def sum(self, x):
        # Ordered FP32 accumulation avoids BF16 rounding at intermediate hops.
        # Transport BF16 operands without widening their payload; the same
        # rank-ordered values are widened before accumulation on each chip.
        gathered = self.gather(x, dim=1)
        shape = tuple(x.shape)
        if (
            len(shape) == 4
            and shape[:2] == (1, 1)
            and 1 <= shape[2] <= 32
            and shape[3] == 2560
            and x.dtype == ttnn.bfloat16
            and x.layout == ttnn.TILE_LAYOUT
            and tuple(self.device.shape) == (1, 4)
            and self.device.arch() == ttnn.device.Arch.BLACKHOLE
            and tuple(gathered.shape) == (1, 4, shape[2], 2560)
            and tuple(gathered.padded_shape) == (1, 4, 32, 2560)
            and gathered.layout == ttnn.TILE_LAYOUT
            and gathered.memory_config() == DRAM
        ):
            tile = gathered.get_tile()
            if tuple(tile.tile_shape) == (32, 32) and not tile.transpose_within_face and not tile.transpose_of_faces:
                # The collective and rank ordering are unchanged. Native local
                # reduction preserves the selected FP32 H-sum tree and zeros.
                return exact_rank_sum(gathered)
        return ttnn.sum(ttnn.typecast(gathered, ttnn.float32), dim=1, keepdim=True)


class MeshExactProjection(FusedPreciseProjection):
    def __init__(
        self,
        loader,
        parts,
        *,
        dtype=ttnn.bfloat16,
        block=32,
        balanced=False,
        native_q1=False,
        native_q1_precast=False,
    ):
        import torch

        if type(native_q1) is not bool or type(native_q1_precast) is not bool:
            raise ValueError("Native Q1 projection options must be bool")
        if (native_q1 and balanced) or (native_q1_precast and not native_q1):
            raise ValueError("Native Q1 requires accurate sum; precast requires native Q1")
        self.native_q1, self.native_q1_precast = native_q1, native_q1_precast
        output, self.input_width = parts[0].shape
        self.output_width = math.ceil(output / 32) * 32
        self.parent_block = self.output_width
        padded = []
        for part in parts:
            target = torch.zeros(self.output_width, self.input_width, dtype=part.dtype)
            target[:output] = part
            padded.append(target.reshape(1, 1, self.output_width, self.input_width))
        self.weights = [loader.shards(padded, dim=2, dtype=dtype)]
        self.compute = loader.compute
        self.prefill_block = block
        self.chunk_prefill = True
        self.balanced_tree = balanced
        self.widen_in_unpack = True

    def __call__(self, value):
        count = value.shape[-2]
        if self.native_q1 and count == 1:
            # Only selected GR-up/GDN-out roles opt in. The resident weight,
            # accurate SFPU K order and explicit BF16 output cast are retained.
            native_value = (
                ttnn.typecast(value, ttnn.float32)
                if self.native_q1_precast and value.dtype == ttnn.bfloat16 and self.weights[0].dtype == ttnn.float32
                else value
            )
            return exact_projection_q1(native_value, self.weights[0], rows_per_worker=16, zero_once=True)
        if count > 32:
            # Keep the inherited prefill algorithm even for a final Q1 tail;
            # its recursive self(...) would otherwise enter native decode.
            return ttnn.concat(
                [
                    FusedPreciseProjection.__call__(self, value[:, :, start : start + 32, :])
                    for start in range(0, count, 32)
                ],
                dim=2,
            )
        return super().__call__(value)


class GatheredProjection(FusedPreciseProjection):
    def __init__(self, projection, ccl, *, before=False, valid_width=None):
        self.projection, self.ccl = projection, ccl
        self.before, self.valid_width = before, valid_width

    def __call__(self, x):
        if self.before:
            x = self.ccl.gather(x)
        result = self.projection(x)
        result = self.ccl.gather(result)
        if self.valid_width is not None:
            physical = result.shape[-1] // 4
            result = ttnn.concat(
                [result[..., r * physical : r * physical + self.valid_width] for r in range(4)], dim=-1
            )
        return result


class MeshInjectionProjection(FusedPreciseProjection):
    """Distribute the narrow10240K injection across ten FP32 partials."""

    def __init__(self, loader, host):
        import torch

        self.compute = loader.compute
        self.weight = loader.shards(
            [torch.nn.functional.pad(part.T, (0, 31)).reshape(1, 10, 1024, 32) for part in host.chunk(4, dim=0)],
            dim=3,
        )

    def __call__(self, x):
        n = x.shape[-2]
        chunks = ttnn.permute(ttnn.reshape(x, (1, n, 10, 1024)), (0, 2, 1, 3))
        partials = ttnn.matmul(
            chunks, self.weight, dtype=ttnn.float32, compute_kernel_config=self.compute, memory_config=DRAM
        )
        return ttnn.typecast(ttnn.sum(partials, dim=1, keepdim=True), ttnn.bfloat16)


class MeshIndexerProjection:
    """Independent ordered K partials, merged in the original FP32 order."""

    def __init__(self, original):
        self.weight = original.weight

    def __call__(self, hidden_states):
        from .parallel_ordered_projection import parallel_ordered_indexer_projection

        return parallel_ordered_indexer_projection(hidden_states, self.weight)


class MeshResidual:
    """One entire stream/chip; every norm still sees all2560 features."""

    def __init__(self, loader, prefix, config, ccl):
        w = loader.state_dict
        self.ccl, self.compute, self.eps = ccl, loader.compute, config.rms_norm_eps
        norm = w[prefix + "hc_norm.weight"].float().reshape(4, 2560) + 1
        self.weight = loader.shards([n.reshape(1, 1, 1, 2560) for n in norm], dim=3, dtype=ttnn.float32)
        down = w[prefix + "input_mix_weight_down.weight"].float()
        self.down = GatheredProjection(
            MeshExactProjection(loader, list(down.chunk(4, dim=0)), dtype=ttnn.float32), ccl, valid_width=80
        )
        up = w[prefix + "input_mix_weight_up.weight"].float()
        self.up = MeshExactProjection(
            loader,
            list(up.chunk(4, dim=0)),
            dtype=ttnn.float32,
            block=256,
            native_q1=True,
            native_q1_precast=True,
        )
        inject = w[prefix + "block_inject_weight.weight"]
        self.inject = mesh_projection(loader, [[part] for part in inject.chunk(4, dim=0)], "gr")

    def read(self, residual):
        count = residual.shape[-2]
        normalized = (
            merged_rms(residual, self.weight, self.eps, rhs=True)
            if 1 <= count <= 32 and tuple(residual.shape) == (1, 1, count, 2560)
            else rms(residual, self.weight, self.eps)
        )
        full = self.ccl.gather(normalized)
        down = ttnn.multiply(self.down(full), 0.25)
        mixing = ttnn.typecast(ttnn.silu(ttnn.typecast(down, ttnn.float32)), ttnn.bfloat16)
        mixing = ttnn.typecast(ttnn.sigmoid(ttnn.typecast(self.up(mixing), ttnn.float32)), ttnn.bfloat16)
        mixed = ttnn.typecast(ttnn.multiply(self.ccl.sum(ttnn.multiply(mixing, normalized)), 0.25), ttnn.bfloat16)
        injection = linear(full, self.inject, self.compute)[..., :1]
        injection = ttnn.multiply(
            ttnn.typecast(ttnn.sigmoid(ttnn.typecast(ttnn.multiply(injection, 0.25), ttnn.float32)), ttnn.bfloat16), 2
        )
        return mixed, injection

    def write(self, residual, block, injection):
        count = residual.shape[-2]
        if 1 <= count <= 32 and all(
            tuple(value.shape) == (1, 1, count, width)
            and value.dtype == ttnn.bfloat16
            and value.layout == ttnn.TILE_LAYOUT
            and value.memory_config() == DRAM
            for value, width in ((residual, 2560), (block, 2560), (injection, 1))
        ):
            return exact_residual_write(residual, block, injection)
        update = ttnn.typecast(
            ttnn.multiply(ttnn.typecast(block, ttnn.float32), ttnn.typecast(injection, ttnn.float32)), ttnn.bfloat16
        )
        return ttnn.typecast(
            ttnn.add(ttnn.typecast(residual, ttnn.float32), ttnn.typecast(update, ttnn.float32)), ttnn.bfloat16
        )


class MeshGDN(FusedGatedDeltaNet):
    def __init__(self, loader, config, ccl):
        import torch

        self.compute, self.eps = loader.compute, config.rms_norm_eps
        self.hkey, self.hvalue, self.dkey, self.dvalue = 4, 12, 128, 128
        self.key_width, self.value_width, self.conv_width = 512, 1536, 2560
        self.compact_recurrence_io = True
        self.fuse_recurrence_layout = True
        self.fuse_conv_exact = True
        w = loader.state_dict

        def qkv_parts(weight):
            q, k, v = weight.split([2048, 2048, 6144], dim=0)
            return [torch.cat([q.chunk(4)[r], k.chunk(4)[r], v.chunk(4)[r]], dim=0) for r in range(4)]

        qkv = FusedExactQKV.__new__(FusedExactQKV)
        qkv.output_width, qkv.input_width = 2560, 2560
        order = torch.tensor([j + 5 * (b + 16 * n) for j in range(5) for b in range(16) for n in range(32)])
        qkv.weight = loader.shards(
            [p[:, order].T.reshape(1, 1, 2560, 2560) for p in qkv_parts(w["linear_attn.in_proj_qkv.weight"])], dim=3
        )
        qkv.weights = [qkv.weight]
        self.qkv = qkv
        zab = [w["linear_attn.in_proj_" + role + ".weight"].chunk(4, dim=0) for role in ("z", "a", "b")]
        self.zab = mesh_projection(loader, [[torch.cat([part[r] for part in zab], dim=0)] for r in range(4)], "gdn_zab")
        self.A = loader.shards(
            [p.reshape(1, 1, 1, 12) for p in (-w["linear_attn.A_log"].float().exp()).chunk(4)],
            dim=3,
            dtype=ttnn.float32,
        )
        self.dt = loader.shards(
            [p.reshape(1, 1, 1, 12) for p in w["linear_attn.dt_bias"].float().chunk(4)], dim=3, dtype=ttnn.float32
        )
        self.norm = loader.tensor(w["linear_attn.norm.weight"].reshape(1, 1, 1, 128))
        self.conv = [
            loader.shards(
                [p.reshape(1, 1, 1, 2560) for p in qkv_parts(w["linear_attn.conv1d.weight"][:, 0, tap])],
                dim=3,
                dtype=ttnn.float32,
            )
            for tap in range(4)
        ]
        self.out = GatheredProjection(
            MeshExactProjection(loader, list(w["linear_attn.out_proj.weight"].chunk(4, dim=0)), native_q1=True),
            ccl,
            before=True,
        )

    def __call__(self, x, *, conv_state, recurrent_state):
        tokens = x.shape[-2]
        if tokens > 32 and (hasattr(self, "conv_taps") or getattr(self, "fuse_conv_exact", False)):
            return ttnn.concat(
                [
                    self(
                        x[:, :, start : min(start + 32, tokens), :],
                        conv_state=conv_state,
                        recurrent_state=recurrent_state,
                    )
                    for start in range(0, tokens, 32)
                ],
                dim=2,
            )
        packed = linear(x, self.zab, self.compute)
        z = packed[..., : self.value_width]
        a = packed[..., self.value_width : self.value_width + self.hvalue]
        b = packed[..., self.value_width + self.hvalue :]
        del packed
        projected = linear(x, self.qkv, self.compute)
        history = ttnn.concat((conv_state, projected), dim=2)
        if hasattr(self, "conv_taps"):
            # Consume only logical output rows; physical padding is private to the op.
            physical = ttnn.reshape(projected, (1, 32, self.conv_width), (1, 32, self.conv_width))
            carry = ttnn.reshape(ttnn.to_layout(conv_state, ttnn.ROW_MAJOR_LAYOUT), (1, 3, self.conv_width))
            q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
                physical,
                carry,
                *self.conv_taps,
                self.key_width,
                self.key_width,
                self.value_width,
                actual_start=self.conv_start,
                predecessor_carry=carry,
                program_config=self.conv_program,
                compute_kernel_config=self.conv_compute,
            )
            q = ttnn.reshape(q[:, :tokens, :], (1, tokens, self.hkey, self.dkey))
            k = ttnn.reshape(k[:, :tokens, :], (1, tokens, self.hkey, self.dkey))
            v = ttnn.reshape(v[:, :tokens, :], (1, tokens, self.hvalue, self.dvalue))
            ttnn.copy(history[:, :, -3:, :], conv_state)
        else:
            # HF Conv1d accumulates all four BF16 products before its BF16 output
            # rounding. Intermediate BF16 products/sums perturb the downstream gate.
            if getattr(self, "fuse_conv_exact", False):
                convolved = mesh_exact_conv_silu(history, self.conv, tokens)
            else:
                conv_input = ttnn.typecast(history, ttnn.float32)
                parts = [ttnn.multiply(conv_input[:, :, lane : lane + tokens, :], self.conv[lane]) for lane in range(4)]
                convolved = ttnn.typecast(
                    ttnn.add(ttnn.add(parts[0], parts[1]), ttnn.add(parts[2], parts[3])), ttnn.bfloat16
                )
                convolved = ttnn.typecast(ttnn.silu(ttnn.typecast(convolved, ttnn.float32)), ttnn.bfloat16)
            ttnn.copy(history[:, :, -3:, :], conv_state)
            q = ttnn.reshape(convolved[:, :, :, : self.key_width], (1, tokens, self.hkey, self.dkey))
            k = ttnn.reshape(convolved[:, :, :, self.key_width : 2 * self.key_width], (1, tokens, self.hkey, self.dkey))
            v = ttnn.reshape(convolved[:, :, :, 2 * self.key_width :], (1, tokens, self.hvalue, self.dvalue))

        def normalize(value):
            value = ttnn.typecast(value, ttnn.float32)
            return ttnn.multiply(
                value, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(value, value), dim=-1, keepdim=True), 1e-6))
            )

        q, k = normalize(q), normalize(k)
        decay = ttnn.multiply(self.A, ttnn.softplus(ttnn.add(ttnn.typecast(a, ttnn.float32), self.dt)))
        beta = ttnn.typecast(
            ttnn.typecast(ttnn.sigmoid(ttnn.typecast(b, ttnn.float32)), ttnn.bfloat16),
            ttnn.float32,
        )
        recurrent = (
            mesh_recurrent_gated_delta_rule
            if getattr(self, "fuse_recurrence_layout", False) or getattr(self, "compact_recurrence_io", False)
            else mesh_recurrent_gated_delta_rule
        )
        output, state = recurrent(
            q,
            k,
            ttnn.typecast(v, ttnn.float32),
            ttnn.reshape(decay, (1, tokens, self.hvalue)),
            ttnn.reshape(beta, (1, tokens, self.hvalue)),
            initial_state=recurrent_state,
            output_final_state=True,
            compute_kernel_config=self.compute,
        )
        ttnn.copy(state, recurrent_state)
        output = ttnn.typecast(output, ttnn.bfloat16)
        value = ttnn.typecast(output, ttnn.float32)
        scale = ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.multiply(value, value), dim=-1, keepdim=True), self.eps))
        output = ttnn.multiply(ttnn.typecast(ttnn.multiply(value, scale), ttnn.bfloat16), self.norm)
        gate = ttnn.reshape(z, (1, tokens, self.hvalue, self.dvalue))
        output = ttnn.typecast(
            ttnn.multiply(ttnn.typecast(output, ttnn.float32), ttnn.sigmoid(ttnn.typecast(gate, ttnn.float32))),
            ttnn.bfloat16,
        )
        return linear(ttnn.reshape(output, (1, 1, tokens, self.value_width)), self.out, self.compute)


def mesh_recurrent_gated_delta_rule(
    q, k, v, g, beta, *, initial_state, output_final_state=True, compute_kernel_config=None, precise_dot_products=True
):
    import ttnn as t

    count = q.shape[1]
    if tuple(q.shape) != (1, count, 4, 128) or tuple(k.shape) != tuple(q.shape):
        raise ValueError("Expected target q/k [1,T,16,128]")
    if tuple(v.shape) != (1, count, 12, 128) or tuple(g.shape) != (1, count, 12) or tuple(beta.shape) != tuple(g.shape):
        raise ValueError("Expected target v [1,T,12,128], g/beta [1,T,12]")
    if tuple(initial_state.shape) != (1, 12, 128, 128) or initial_state.dtype != t.float32:
        raise ValueError("A persistent FP32 [1,12,128,128] initial state is mandatory")
    if count < 1:
        raise ValueError("At least one actual token is required")
    device = q.device()

    def fp32(x):
        return x if x.dtype == t.float32 else t.typecast(x, t.float32)

    def reshape_rm(x, shape):
        return t.to_layout(t.reshape(t.to_layout(x, t.ROW_MAJOR_LAYOUT), shape), t.TILE_LAYOUT)

    def heads(x, number):
        return t.reshape(t.permute(fp32(x), (0, 2, 1, 3)), (number, count, 128))

    qh = heads(t.multiply(fp32(q), 128**-0.5), 4)
    kh = heads(k, 4)
    qh = t.repeat_interleave(qh, repeats=3, dim=0)
    kh = t.repeat_interleave(kh, repeats=3, dim=0)
    vh = heads(v, 12)
    if count > 1:
        qh, kh, vh = (reshape_rm(x, (12 * count, 1, 128)) for x in (qh, kh, vh))
    decay = t.exp(t.reshape(t.permute(fp32(g), (0, 2, 1)), (12, count)))
    betah = t.reshape(t.permute(fp32(beta), (0, 2, 1)), (12, count))
    decay = reshape_rm(decay, (12 * count, 1, 1))
    betah = reshape_rm(betah, (12 * count, 1, 1))
    state = t.reshape(initial_state, (12, 128, 128))
    output = t.empty(
        (12 * count, 1, 128), dtype=t.float32, layout=t.TILE_LAYOUT, device=device, memory_config=t.DRAM_MEMORY_CONFIG
    )
    final = t.empty(
        (12, 128, 128), dtype=t.float32, layout=t.TILE_LAYOUT, device=device, memory_config=t.DRAM_MEMORY_CONFIG
    )
    tensors = [qh, kh, vh, decay, betah, state, output, final]
    for tensor in tensors:
        if (
            tensor.layout != t.TILE_LAYOUT
            or tensor.dtype != t.float32
            or tensor.memory_config() != t.DRAM_MEMORY_CONFIG
        ):
            raise ValueError("Recurrence requires interleaved DRAM FP32 TILE operands")
    grid = device.compute_with_storage_grid_size()
    if grid.x * grid.y < 12:
        raise ValueError("Target recurrence requires48 compute cores")
    cores = [t.CoreCoord(h // grid.y, h % grid.y) for h in range(12)]
    ranges = t.CoreRangeSet([t.CoreRange(core, core) for core in cores])
    reader_rt, writer_rt, compute_rt = t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs()
    addresses = [value.buffer_address() for value in tensors]
    for h, core in enumerate(cores):
        reader_rt[core.x][core.y] = [h, count, *addresses[:6], 0, 0, 12]
        writer_rt[core.x][core.y] = [h, count, *addresses[6:], 12]
        compute_rt[core.x][core.y] = [count]
    reader_ct = [4, 4, 0]
    for value in tensors[:6]:
        reader_ct.extend(t.TensorAccessorArgs(value).get_compile_time_args())
    writer_ct = [4, 4, 0]
    for value in tensors[6:]:
        writer_ct.extend(t.TensorAccessorArgs(value).get_compile_time_args())
    compute_config = t.ComputeConfigDescriptor(
        math_fidelity=t.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=False, math_approx_mode=False
    )
    modes = [t.UnpackToDestMode.Default] * 64
    # Direct copies in decay, residual and rank1 seeding must retain all FP32
    # mantissa bits. q/k/matmul operands remain native unless explicitly copied.
    # State CBs8/9/10 are also FPU operands: direct-DST modes cannot be
    # applied to those indices. Separate aliases share SRAM but expose an
    # exact-copy view; their read pointers are synchronized in the kernel.
    exact_aliases = {8: 17, 9: 18, 10: 19}
    for cb in (2, 3, 4, 5, 11, *exact_aliases.values()):
        modes[cb] = t.UnpackToDestMode.UnpackToDestFp32
    if precise_dot_products:
        # The SFPU variant has no FPU source consumers: all arithmetic inputs
        # load FP32 directly into DST, including q/k, state and delta.
        exact_aliases = {}
        for cb in (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12):
            modes[cb] = t.UnpackToDestMode.UnpackToDestFp32
    compute_config.unpack_to_dest_mode = modes
    specs = [
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 2 if count > 1 else 1),
        (4, 2 if count > 1 else 1),
        (5, 16),
        (6, 8),
        (7, 32),
        (8, 16),
        (9, 16),
        (10, 16),
        (11, 4),
        (12, 4),
        (13, 4),
    ]
    cbs = [
        t.CBDescriptor(
            total_size=tiles * 4096,
            core_ranges=ranges,
            format_descriptors=[
                t.CBFormatDescriptor(buffer_index=view, data_format=t.float32, page_size=4096)
                for view in ([index, exact_aliases[index]] if index in exact_aliases else [index])
            ],
        )
        for index, tiles in specs
    ]
    descriptors = []
    for filename, ct, rt, config in (
        ("gdn_reader.cpp", reader_ct, reader_rt, t.ReaderConfigDescriptor()),
        ("gdn_writer.cpp", writer_ct, writer_rt, t.WriterConfigDescriptor()),
        ("gdn_compute_sfpu.cpp" if precise_dot_products else "gdn_compute.cpp", [4, 4, 0], compute_rt, compute_config),
    ):
        descriptors.append(
            t.KernelDescriptor(
                kernel_source=str(_KERNEL_ROOT / filename),
                source_type=t.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=ranges,
                compile_time_args=ct,
                runtime_args=rt,
                config=config,
            )
        )
    program = t.ProgramDescriptor(kernels=descriptors, cbs=cbs, semaphores=[])
    t.generic_op(tensors, program)
    result = t.to_layout(
        t.permute(t.reshape(t.to_layout(output, t.ROW_MAJOR_LAYOUT), (1, 12, count, 128)), (0, 2, 1, 3)), t.TILE_LAYOUT
    )
    return result, t.reshape(final, (1, 12, 128, 128)) if output_final_state else None


def mesh_exact_conv_silu(history, taps, count):
    """Four exact FP32 products/pairwise adds, BF16 RNE, FP32 SiLU, BF16 RNE.

    Native SFPU primitives are fused within one generic program. Four shifted
    BF16 slices remain device operations; the compute kernel never uses the FPU.
    """
    if not 1 <= count <= 32 or tuple(history.shape) != (1, 1, count + 3, 2560):
        raise ValueError("Expected BF16 GDN history [1,1,logical_count+3,2560], count1..32")
    device = history.device()
    if history.dtype != ttnn.bfloat16 or history.layout != ttnn.TILE_LAYOUT:
        raise ValueError("Exact fused convolution requires BF16 tiled history")
    windows = [history[:, :, i : i + count, :] for i in range(4)]
    for tap in taps:
        if tuple(tap.shape) != (1, 1, 1, 2560) or tap.dtype != ttnn.float32 or tap.layout != ttnn.TILE_LAYOUT:
            raise ValueError("Exact fused convolution requires four FP32 tiled taps")
    if len(taps) != 4 or any(
        value.memory_config() != ttnn.DRAM_MEMORY_CONFIG or value.device() != device for value in [*windows, *taps]
    ):
        raise ValueError("Exact fused convolution requires same-device interleaved DRAM")
    output = ttnn.empty(
        (1, 1, count, 2560),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grid = device.compute_with_storage_grid_size()
    cores = min(grid.x * grid.y, 80)
    locations = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(cores)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in locations])
    reader, compute, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    addresses = [value.buffer_address() for value in [*windows, *taps]]
    start = 0
    for i, core in enumerate(locations):
        number = 80 // cores + int(i < 80 % cores)
        reader[core.x][core.y] = [*addresses, start, number]
        compute[core.x][core.y] = [number]
        writer[core.x][core.y] = [output.buffer_address(), start, number]
        start += number
    read_ct = []
    for value in [*windows, *taps]:
        read_ct.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )
        for source, ct, rt, config in (
            (_CONV_EXACT_READER, read_ct, reader, ttnn.ReaderConfigDescriptor()),
            (
                _CONV_EXACT_COMPUTE,
                [],
                compute,
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=False,
                    math_approx_mode=False,
                ),
            ),
            (
                _CONV_EXACT_WRITER,
                ttnn.TensorAccessorArgs(output).get_compile_time_args(),
                writer,
                ttnn.WriterConfigDescriptor(),
            ),
        )
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * size,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
        )
        for index, dtype, size in ((0, ttnn.bfloat16, 2048), (1, ttnn.float32, 4096), (2, ttnn.bfloat16, 2048))
    ]
    ttnn.generic_op([*windows, *taps, output], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return output


class MeshWorkspace:
    """Reusable sequential-layer expert scratch; share across a decoder stack."""

    def __init__(self, mesh_device):
        import torch

        self.device = mesh_device
        self.rope = None
        self.rope_key = None
        self.index_arrays = {}
        # A32-token group has at most320 selected experts; every active expert
        # needs one32-row tile. This is the tight worst-case dispatch extent.
        self.capacity = 32 * min(512, 10 * 32)
        self.expert_output = ttnn.zeros(
            (self.capacity, 2560), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.combine_zeros = ttnn.zeros(
            (320, 2560), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.score_zeros = ttnn.zeros((32, 10), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.id_zeros = ttnn.from_torch(
            torch.zeros(32, 10, dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        loader = MeshLoader({}, mesh_device)
        self.rank_lo, self.rank_hi = (
            loader.shards(
                [torch.full((1, 1, 1, 1), rank * 128 + offset, dtype=torch.float32) for rank in range(4)],
                dim=0,
                dtype=ttnn.float32,
            )
            for offset in (0, 128)
        )


class MeshMoe:
    """EP4 grouped execution: each token executes only its selected experts."""

    def __init__(self, loader, ccl, expert_dtype, workspace):
        import torch

        self.loader, self.ccl, self.compute = loader, ccl, loader.compute
        self.workspace = workspace
        w = loader.state_dict
        self.router = GatheredProjection(MeshExactProjection(loader, list(w["mlp.gate.weight"].chunk(4, dim=0))), ccl)
        self.experts = {}
        gate, up = w["mlp.experts.gate_up_proj"].chunk(2, dim=1)
        down = w["mlp.experts.down_proj"]
        for name, host in [("gate", gate), ("up", up), ("down", down)]:
            self.experts[name] = [
                loader.shards([host[r * 128 + j].T for r in range(4)], dim=0, dtype=expert_dtype) for j in range(128)
            ]
        self.expert_compute = ttnn.init_device_compute_kernel_config(
            loader.mesh_device.arch(),
            math_fidelity=getattr(
                ttnn.MathFidelity, getattr(loader, "precision", {}).get("compute_fidelities", {}).get("experts", "LoFi")
            ),
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=False,
        )
        self.ids = loader.shards(
            [torch.arange(r * 128, (r + 1) * 128, dtype=torch.int32) for r in range(4)],
            dim=0,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        table = torch.full((4, 512), -1, dtype=torch.int32)
        for r in range(4):
            table[r, r * 128 : (r + 1) * 128] = 0
        self.reduce_table = loader.shards(
            list(table.split(1, dim=0)), dim=0, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.dispatch_table = ttnn.from_torch(
            torch.zeros(1, 512, dtype=torch.int32),
            device=loader.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.zero = ttnn.from_torch(
            torch.zeros(1, 1, 32, 512, dtype=torch.bfloat16), device=loader.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.one = ttnn.from_torch(
            torch.ones(1, 1, 32, 10, dtype=torch.bfloat16), device=loader.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.capacity, self.output = workspace.capacity, workspace.expert_output
        # Shared expert TP4. Output partials are reduced once, including the gate.
        shared_parts = [w["mlp.shared_expert." + role + "_proj.weight"].chunk(4, dim=0) for role in ("gate", "up")]
        self.shared = []  # No retained interleaved duplicates of prepared weights.
        self.dram_shared = mesh_projection_group(
            loader, [[part[r] for part in shared_parts] for r in range(4)], "shared"
        )
        self.shared_down = loader.shards(
            list(w["mlp.shared_expert.down_proj.weight"].T.chunk(4, dim=0)),
            dim=0,
            dtype=getattr(
                ttnn, getattr(loader, "precision", {}).get("weight_groups", {}).get("shared_down", "bfloat8_b")
            ),
        )
        self.shared_down_compute = ttnn.init_device_compute_kernel_config(
            loader.mesh_device.arch(),
            math_fidelity=getattr(
                ttnn.MathFidelity,
                getattr(loader, "precision", {}).get("compute_fidelities", {}).get("shared_down", "HiFi4"),
            ),
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )
        self.shared_route = loader.tensor(w["mlp.shared_expert_gate.weight"].T, ttnn.bfloat8_b)

    def __call__(self, x):
        from .router_topk import aten_router_topk

        n = x.shape[-2]
        logits = ttnn.typecast(self.router(x), ttnn.float32)
        scores, ids = aten_router_topk(logits)
        scores = ttnn.typecast(
            ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=self.compute), ttnn.bfloat16
        )
        selected = ttnn.to_layout(
            ttnn.scatter(self.zero[:, :, :n, :], dim=-1, index=ids, src=self.one[:, :, :n, :]), ttnn.TILE_LAYOUT
        )
        counts = ttnn.sum(ttnn.typecast(selected, ttnn.float32), dim=2, keepdim=True)
        aligned = ttnn.multiply(ttnn.ceil(ttnn.multiply(counts, 1 / 32)), 32)
        offsets = ttnn.subtract(ttnn.cumsum(aligned, dim=3), aligned)

        def ints(value, shape):
            return ttnn.reshape(ttnn.to_layout(ttnn.typecast(value, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), shape)

        counts, offsets = ints(counts, (1, 512)), ints(offsets, (1, 512))
        ids_rm = ttnn.to_layout(ttnn.typecast(ids, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT)
        routed, metadata = ttnn.experimental.deepseek_prefill.dispatch(
            ttnn.reshape(x, (1, n, 2560)),
            ttnn.reshape(ids_rm, (1, n, 10)),
            offsets,
            self.dispatch_table,
            dispatch_group_size=1,
            experts_per_chip=512,
            num_routed_experts=512,
            num_experts_per_tok=10,
            metadata_len=3,
            max_dispatch_buffer_token_size=self.capacity,
            cluster_axis=0,
        )
        down = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            ttnn.reshape(routed, (self.capacity, 2560)),
            self.experts["gate"],
            self.experts["up"],
            self.experts["down"],
            ttnn.reshape(counts, (512,)),
            self.ids,
            input_m_tiles=math.ceil(n / 32),
            compute_kernel_config=self.expert_compute,
            core_grid=ttnn.CoreCoord(10, 8),
            output=self.output,
            expert_region_offsets=ttnn.reshape(offsets, (512,)),
            read_x_at_offset=True,
            stochastic_rounding=False,
        )
        # Native TILE combine corrupts the first active region on this target.
        # Explicit untilization is exact; convert only the static worst-case
        # extent for these logical rows (no routing readback is needed).
        combine_capacity = 32 * min(512, 10 * n)
        combined = ttnn.experimental.deepseek_prefill.combine(
            ttnn.reshape(
                ttnn.to_layout(down[:combine_capacity, :], ttnn.ROW_MAJOR_LAYOUT), (1, 1, combine_capacity, 2560)
            ),
            metadata[..., :combine_capacity, :],
            ttnn.reshape(counts, (1, 1, 512)),
            ttnn.reshape(offsets, (1, 1, 512)),
            dispatch_group_size=1,
            experts_per_chip=512,
            num_experts_per_tok=10,
            seq_len_per_chip=n,
            cluster_axis=0,
        )
        weights = ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT)
        routed = self._reduce(combined, weights, ids_rm, n)
        shared = self.shared_forward(x)
        return ttnn.typecast(self.ccl.sum(ttnn.add(routed, shared)), ttnn.bfloat16)

    def shared_forward(self, x):
        gate, up = self.dram_shared(x)
        middle = ttnn.multiply(ttnn.silu(gate), up)
        shared = linear(middle, self.shared_down, self.shared_down_compute)
        return ttnn.multiply(shared, ttnn.sigmoid(linear(x, self.shared_route, self.compute)))

    def _reduce(self, combined, weights, ids_rm, n):
        if n == 1:
            return self._reduce_single(combined, weights, ids_rm)
        if n < 32:
            # Only the hardware reduction is padded. Routing, expert counts and
            # mutable attention state still see exactly n logical tokens.
            combined = ttnn.reshape(
                ttnn.concat(
                    [
                        ttnn.reshape(combined, (n * 10, 2560)),
                        self.workspace.combine_zeros[: (32 - n) * 10, :],
                    ],
                    dim=0,
                ),
                (1, 1, 32, 10, 2560),
            )
            weights = ttnn.reshape(
                ttnn.concat(
                    [
                        ttnn.reshape(weights, (n, 10)),
                        self.workspace.score_zeros[: 32 - n, :],
                    ],
                    dim=0,
                ),
                (1, 1, 32, 10),
            )
            ids_rm = ttnn.reshape(
                ttnn.concat(
                    [
                        ttnn.reshape(ids_rm, (n, 10)),
                        self.workspace.id_zeros[: 32 - n, :],
                    ],
                    dim=0,
                ),
                (1, 1, 32, 10),
            )
        routed = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            combined, weights, ids_rm, self.reduce_table, expert_dim=3, output_memory_config=DRAM
        )
        return routed[:, :, :n, :]

    def _reduce_single(self, combined, weights, ids_rm):
        # Native post-combine reduction assigns this bucket one core. A
        # feature-parallel matmul avoids its 32-token padding and serial BF16
        # accumulation. Keep native grouped reduction for larger token groups.
        ids = ttnn.to_layout(ttnn.typecast(ttnn.reshape(ids_rm, (1, 1, 1, 10)), ttnn.float32), ttnn.TILE_LAYOUT)
        owned = ttnn.logical_and(ttnn.ge(ids, self.workspace.rank_lo), ttnn.lt(ids, self.workspace.rank_hi))
        owned = ttnn.reshape(ttnn.typecast(owned, ttnn.bfloat16), (1, 1, 10, 1))
        values = ttnn.to_layout(ttnn.reshape(combined, (1, 1, 10, 2560)), ttnn.TILE_LAYOUT)
        # Nonowned slots can be stale, including NaNs. Select zero before any
        # arithmetic: multiplying those slots by zero would not mask NaNs.
        values = ttnn.where(owned, values, 0.0)
        weights = ttnn.to_layout(ttnn.reshape(weights, (1, 1, 1, 10)), ttnn.TILE_LAYOUT)
        return ttnn.matmul(weights, values, dtype=ttnn.bfloat16, compute_kernel_config=self.compute, memory_config=DRAM)


class MultichipDecoder(OptimizedDecoder):
    """1x4 target implementation under validation against OptimizedDecoder."""

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        expert_dtype=ttnn.bfloat4_b,
        packed_embedding=None,
        workspace=None,
        precision=None,
        **kwargs,
    ):
        if tuple(mesh_device.shape) != (1, 4):
            raise ValueError("This implementation targets the complete1x4 Blackhole mesh")
        if kwargs:
            raise TypeError(f"Unexpected construction options: {sorted(kwargs)}")
        cfg = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
        prefix = f"model.language_model.layers.{layer_idx}."
        w = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)} or state_dict
        self = cls()
        self.mesh_device, self.config, self.layer_idx = mesh_device, cfg, layer_idx
        self.kind = cfg.layer_types[layer_idx]
        self.has_ple = layer_idx + 1 in cfg.ple_layer_ids
        self.embedding = packed_embedding
        if self.has_ple and (packed_embedding is None or packed_embedding.rows != 320001536):
            raise ValueError("PLE requires the full mesh-sharded packed table")
        self.precision = precision
        self.kv_dtype = getattr(ttnn, precision["kv_cache_dtype"]) if precision else ttnn.bfloat16
        self.ccl = Collectives(mesh_device, precision["ccl_dtype"] if precision else "bfloat16")
        self.workspace = workspace if workspace is not None else MeshWorkspace(mesh_device)
        if self.workspace.device != mesh_device:
            raise ValueError("Workspace must belong to the target mesh")
        loader = MeshLoader(w, mesh_device)
        if precision is not None:
            loader.precision = precision
            expert_dtypes = {precision["weight_groups"]["expert_" + group] for group in ("gate", "up", "down")}
            if len(expert_dtypes) != 1:
                raise ValueError("moe_fused_swiglu requires gate/up/down to share dtype")
            expert_dtype = getattr(ttnn, expert_dtypes.pop())
        loader.precision_groups = {}
        self.precision_groups = loader.precision_groups
        self.ple = MeshPLE(loader, cfg) if self.has_ple else None
        self.attn_mix = MeshResidual(loader, "attn_hyper_connection.", cfg, self.ccl)
        self.mlp_mix = MeshResidual(loader, "mlp_hyper_connection.", cfg, self.ccl)
        if self.kind == "linear_attention":
            self.attn = MeshGDN(loader, cfg, self.ccl)
            self.chunked_attn = None
        elif self.kind == "indexed_attention":
            self.chunked_attn = MeshQSA.build(loader, cfg, self.ccl, self.workspace)
            self.attn = self.chunked_attn.attention
        else:
            raise ValueError(f"Unsupported layer kind: {self.kind}")
        self.mlp = MeshMoe(loader, self.ccl, expert_dtype, self.workspace)
        return self

    def _create_base_state(
        self, *, capacity=None, page_table=None, hash_lengths=tuple(range(1, 33)), defer_kv_cache=False
    ):
        if self.kind == "indexed_attention":
            import torch

            capacity = self.config.max_position_embeddings if capacity is None else capacity
            if not (128 <= capacity <= self.config.max_position_embeddings and capacity % 128 == 0):
                raise ValueError("capacity must be a multiple of128 within the advertised context")
            if page_table is None:
                page_table = torch.arange(capacity // 32, dtype=torch.int32).reshape(1, -1)
            if isinstance(page_table, torch.Tensor):
                if tuple(page_table.shape) != (1, capacity // 32) or not torch.equal(
                    page_table.sort().values.flatten().long(), torch.arange(capacity // 32)
                ):
                    raise ValueError("setup page table must permute the allocated pages")
                page_table = ttnn.from_torch(page_table.int(), device=self.mesh_device, dtype=ttnn.int32)

            def zeros(shape, dtype=ttnn.bfloat16):
                return ttnn.zeros(shape, device=self.mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

            if capacity not in self.workspace.index_arrays:
                self.workspace.index_arrays[capacity] = tuple(
                    self.attn.loader.tensor(torch.arange(width).float().reshape(1, 1, 1, width), ttnn.float32)
                    for width in (capacity // 4, capacity)
                )
            blocks, tokens = self.workspace.index_arrays[capacity]
            state = QSAState(
                capacity=capacity,
                page_size=32,
                position=ttnn.from_torch(torch.zeros(1, dtype=torch.int32), device=self.mesh_device, dtype=ttnn.int32),
                page_table=page_table,
                k_cache=None if defer_kv_cache else zeros((capacity // 32, 1, 32, 256), self.kv_dtype),
                v_cache=None if defer_kv_cache else zeros((capacity // 32, 1, 32, 256), self.kv_dtype),
                raw_history=zeros((1, 1, 3, 128)),
                compressed_keys=zeros((1, 1, capacity // 4, 128)),
                block_numbers=blocks,
                token_numbers=tokens,
            )
            return LinearState(attention=state)
        return LinearState(
            ttnn.zeros((1, 1, 3, 2560), device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            ttnn.zeros((1, 12, 128, 128), device=self.mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
        )

    def create_state(self, *, capacity=None, page_table=None, hash_lengths=tuple(range(1, 33)), defer_kv_cache=False):
        import torch

        state = self._create_base_state(capacity=capacity, page_table=page_table, defer_kv_cache=defer_kv_cache)
        if self.has_ple:

            def integer(value):
                return ttnn.from_torch(value, device=self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

            state.ple_conv = ttnn.zeros(
                (1, 1, 9, 2560), device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            history = torch.zeros(1, 16, dtype=torch.int32)
            history[:, :2] = EOS
            state.token_history = integer(history)
            state.hash_operations = {}
            for n in sorted(set(hash_lengths)):
                if not 1 <= n <= 32:
                    raise ValueError("Expected physical hash length1..32")
                control = torch.zeros(1, 16, dtype=torch.int32)
                control[:, 0] = n
                op = DeviceNGramHash(
                    tokens=integer(torch.zeros(1, n, 16, dtype=torch.int32)),
                    control=integer(control),
                    history=state.token_history,
                    output=integer(torch.zeros(1, n, 16, dtype=torch.int32)),
                )
                op.prepare()
                state.hash_operations[n] = op
        return state

    def _attention_chunk(self, x, state, token_ids=None):
        if self.has_ple:
            n = x.shape[-2]
            if token_ids is None or tuple(token_ids.shape) != (1, n, 16):
                raise ValueError("PLE requires token words [1,T,16]")
            op = state.hash_operations[n]
            ttnn.copy(token_ids, op.tensors[0])
            op.forward()
            embedding = self.embedding(ttnn.reshape(op.tensors[-1], (1, 1, n, 16)))
            x = ttnn.add(x, self.ple(x, embedding, conv_state=state.ple_conv))
        mixed, injection = self.attn_mix.read(x)
        if self.chunked_attn is None:
            block = self.attn(mixed, conv_state=state.conv, recurrent_state=state.recurrent)
        else:
            block = self.chunked_attn.forward_chunk(mixed, state=state.attention)
            self.chunked_attn.advance(state.attention, x.shape[-2])
        x = self.attn_mix.write(x, block, injection)
        mixed, injection = self.mlp_mix.read(x)
        return x, mixed, injection

    def _chunk(self, x, state, token_ids=None):
        x, mixed, injection = self._attention_chunk(x, state, token_ids)
        return self.mlp_mix.write(x, self.mlp(mixed), injection)

    def prefill_forward(self, hidden_states, *, state, token_ids=None):
        if isinstance(hidden_states, (tuple, list)):
            self._validate_batch(hidden_states, state, token_ids)
            return tuple(
                self.prefill_forward(x, state=s, token_ids=None if token_ids is None else token_ids[i])
                for i, (x, s) in enumerate(zip(hidden_states, state, strict=True))
            )
        n = hidden_states.shape[-2]
        if not 0 < n <= self.config.max_position_embeddings or tuple(hidden_states.shape) != (1, 1, n, 2560):
            raise ValueError("Expected a logical residual shard [1,1,T>0,2560]")
        if n <= 32:
            return self._chunk(hidden_states, state, token_ids)
        output = ttnn.empty(
            tuple(hidden_states.shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=DRAM,
        )
        for start in range(0, n, 32):
            chunk = self._chunk(
                hidden_states[:, :, start : min(start + 32, n), :],
                state,
                None if token_ids is None else token_ids[:, start : min(start + 32, n), :],
            )
            write_token_chunk(chunk, output, start)
            del chunk
        return output

    def decode_forward(self, hidden_states, *, state, token_ids=None):
        if isinstance(hidden_states, (tuple, list)):
            self._validate_batch(hidden_states, state, token_ids)
            if any(tuple(x.shape) != (1, 1, 1, 2560) for x in hidden_states):
                raise ValueError("Expected one local residual row per decode request")
            # Attention has independent request caches. Route all MLP rows together
            # so expert weights are reused by the active requests in one dispatch.
            prepared = [
                self._attention_chunk(x, state[i], None if token_ids is None else token_ids[i])
                for i, x in enumerate(hidden_states)
            ]
            mixed = prepared[0][1] if len(prepared) == 1 else ttnn.concat([p[1] for p in prepared], dim=2)
            blocks = self.mlp(mixed)
            return tuple(
                self.mlp_mix.write(x, blocks[:, :, i : i + 1, :], injection)
                for i, (x, _, injection) in enumerate(prepared)
            )
        if tuple(hidden_states.shape) != (1, 1, 1, 2560):
            raise ValueError("Expected [1,1,1,2560] decode residual shard")
        return self._chunk(hidden_states, state, token_ids)


class MeshQSA(FusedChunkedQSA):
    def _selected_ids(self, query, *, state, positions):
        # Read the live compressed prefix from device positions. Preserve the
        # selected SFPU sum order and selector ordering without scoring padding.
        scores = causal_indexer_scores(self.attention, query, state.compressed_keys, positions, packed_decode=True)
        return fused_selected_token_ids(scores, positions)

    @classmethod
    def build(cls, loader, config, ccl, workspace):
        import torch

        base = QSA(loader, config)
        rope_key = (config.max_position_embeddings, base.rotary_dim, config.rope_parameters["rope_theta"])
        if workspace.rope is None:
            workspace.rope_key, workspace.rope = rope_key, (base.cos, base.sin)
        elif workspace.rope_key != rope_key:
            raise ValueError("Shared RoPE tables must have identical context and frequency parameters")
        else:
            base.cos, base.sin = workspace.rope
        result = FusedChunkedQSA.from_component(
            ChunkedQSA(base),
            gate=True,
            concat=True,
            pack=False,
            rope=True,
            score_sum=True,
            merge_score_heads=True,
            valid_rows=True,
            selector=True,
        )
        result.__class__ = cls
        result.ccl = ccl
        result.gather = local_selected_paged_kv
        if getattr(loader, "precision", {}).get("kv_cache_dtype") == "bfloat8_b":
            from .bfp8_selected_paged_kv import local_selected_paged_kv_bfp8

            result.gather = local_selected_paged_kv_bfp8
        a = result.attention
        a.project_index_qk = MeshIndexerProjection(a.project_index_qk)
        w = loader.state_dict
        q = w["self_attn.q_proj.weight"].chunk(4, dim=0)
        k = w["self_attn.k_proj.weight"].chunk(2, dim=0)
        v = w["self_attn.v_proj.weight"].chunk(2, dim=0)
        result.packed_qkv = mesh_projection(
            loader, [[torch.cat([q[r], k[r // 2], v[r // 2]], dim=0)] for r in range(4)], "qsa_qkv"
        )
        a.out = mesh_projection(loader, [[part] for part in w["self_attn.o_proj.weight"].chunk(4, dim=1)], "qsa_out")
        # QKV/WO keep their independently selected HiFi4 DRAM policies. This
        # config controls the remaining attention work, including SDPA.
        a.compute = ttnn.init_device_compute_kernel_config(
            loader.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )
        del a.q, a.k, a.v
        result.sdpa_program = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=32,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        return result

    def forward_chunk(self, hidden_states, *, state, return_selected_ids=False):
        """Consume 1..32 rows, preserving state.position for caller advancement."""
        t, a = self.t, self.attention
        count = hidden_states.shape[-2]
        if tuple(hidden_states.shape) != (1, 1, count, 2560) or not 1 <= count <= 32:
            raise ValueError("Expected [1,1,1..32,2560] valid token rows")
        pos = t.to_layout(t.reshape(t.typecast(state.position, t.float32), (1, 1, 1, 1)), t.TILE_LAYOUT)
        positions = t.add(pos, self.query_offsets[:, :, :count, :])
        group = t.floor(t.multiply(positions, 0.25))
        group_starts = t.multiply(group, 4)
        write_groups = t.where(t.eq(t.subtract(positions, group_starts), 3), group, -1.0)

        packed = a.linear(hidden_states, self.packed_qkv, a.compute)
        q_gate = t.permute(t.reshape(packed[..., :3072], (1, count, 6, 512)), (0, 2, 1, 3))
        key = t.permute(t.reshape(packed[..., 3072:3328], (1, count, 1, 256)), (0, 2, 1, 3))
        value = t.permute(t.reshape(packed[..., 3328:3584], (1, count, 1, 256)), (0, 2, 1, 3))
        del packed
        query, gate = q_gate[..., :256], q_gate[..., 256:]
        query = self._rope(a.rms(query, a.q_norm, a.eps), positions)
        key = self._rope(a.rms(key, a.k_norm, a.eps), positions)

        index_qk = a.project_index_qk(hidden_states)
        index_query = t.permute(t.reshape(index_qk[..., :512], (1, count, 4, 128)), (0, 2, 1, 3))
        index_query = self._rope(a.rms(index_query, a.index_q_norm, a.eps), positions)
        raw = index_qk[..., 512:]
        history = t.concat([state.raw_history, raw], dim=2)
        # Sliding four-row sums must accumulate FP32, then round to BF16 before
        # the HF k-layernorm. Both arbitrary continuation phase and short tails
        # use exactly the same device expression.
        history_fp32 = t.typecast(history, t.float32)
        if getattr(self, "pool_mean", False):
            pooled = t.typecast(
                t.mean(
                    t.concat([history_fp32[:, :, offset : offset + count, :] for offset in range(4)], dim=0),
                    dim=0,
                    keepdim=True,
                    compute_kernel_config=a.compute,
                ),
                t.bfloat16,
            )
        else:
            pooled = t.add(history_fp32[:, :, :count, :], history_fp32[:, :, 1 : count + 1, :])
            pooled = t.add(pooled, history_fp32[:, :, 2 : count + 2, :])
            pooled = t.add(pooled, history_fp32[:, :, 3 : count + 3, :])
            pooled = t.typecast(t.multiply(pooled, 0.25), t.bfloat16)
        pooled = self._rope(a.rms(pooled, a.index_k_norm, a.eps), group_starts)

        for row in range(count):
            position = t.reshape(
                t.to_layout(t.typecast(positions[:, :, row : row + 1, :], t.int32), t.ROW_MAJOR_LAYOUT), (1,)
            )
            group_position = t.reshape(
                t.to_layout(t.typecast(write_groups[:, :, row : row + 1, :], t.int32), t.ROW_MAJOR_LAYOUT), (1,)
            )
            a._update(state.k_cache, t.permute(key[:, :, row : row + 1, :], (0, 2, 1, 3)), position, state.page_table)
            a._update(state.v_cache, t.permute(value[:, :, row : row + 1, :], (0, 2, 1, 3)), position, state.page_table)
            a._update(state.compressed_keys, pooled[:, :, row : row + 1, :], group_position)
        t.copy(history[:, :, -3:, :], state.raw_history)

        selected_ids = self._selected_ids(index_query, state=state, positions=positions)
        selected_k, selected_v = self.gather(state.k_cache, state.v_cache, state.page_table, selected_ids)
        sentinel = t.reshape(self.invalid_ids[:, :, :count, :], (count, 2080))
        allowed = t.ne(selected_ids, sentinel)
        mask = t.reshape(t.typecast(t.where(allowed, 0.0, -1.0e9), t.bfloat16), (count, 1, 1, 2080))
        attention = t.transformer.scaled_dot_product_attention(
            t.permute(query, (2, 1, 0, 3)),
            selected_k,
            selected_v,
            is_causal=False,
            attn_mask=mask,
            scale=256**-0.5,
            program_config=self.sdpa_program,
            compute_kernel_config=a.compute,
            memory_config=t.DRAM_MEMORY_CONFIG,
        )
        attention = t.permute(attention, (2, 1, 0, 3))
        attention = (
            t.multiply(attention, gate, input_tensor_b_activations=[t.UnaryOpType.SIGMOID])
            if self.fuse_gate
            else t.multiply(attention, t.sigmoid(gate))
        )
        attention = t.reshape(t.transformer.concatenate_heads(attention), (1, 1, count, 1536))
        output = t.typecast(self.ccl.sum(a.linear(attention, a.out, a.compute)), t.bfloat16)
        return (output, selected_ids) if return_selected_ids else output


_LOCAL_ADDRESS_SOURCE = r"""
inline uint32_t qsa_face_element(uint32_t row, uint32_t column) {
    return (((row >> 4) * 2u + (column >> 4)) * 256u) + (row & 15u) * 16u + (column & 15u);
}
inline uint32_t qsa_cache_tile(uint32_t physical_page, uint32_t head,
                               uint32_t page_size, uint32_t token_in_page, uint32_t channel_tile) {
    return (((physical_page + head) * (page_size / 32u) + token_in_page / 32u) * 8u) + channel_tile;
}
"""


def local_kv_kernel_source():
    return (
        '#include <stdint.h>\n#include "api/dataflow/dataflow_api.h"\n'
        + _LOCAL_ADDRESS_SOURCE
        + r"""
void kernel_main() {
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t PHYSICAL_PAGES = get_compile_time_arg_val(1);
    constexpr uint32_t LOGICAL_PAGES = get_compile_time_arg_val(2);
    constexpr uint32_t QUERIES = get_compile_time_arg_val(3);
    constexpr uint32_t SLOT_TILES = 65; // 2080 selected slots
    constexpr auto ka = TensorAccessorArgs<4>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto koa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto voa = TensorAccessorArgs<koa.next_compile_time_args_offset()>();
    const auto keys = TensorAccessor(ka, get_arg_val<uint32_t>(0), 2048);
    const auto values = TensorAccessor(va, get_arg_val<uint32_t>(1), 2048);
    const auto pages = TensorAccessor(pa, get_arg_val<uint32_t>(2));
    const auto ids = TensorAccessor(ia, get_arg_val<uint32_t>(3), 4096);
    const auto kout = TensorAccessor(koa, get_arg_val<uint32_t>(4), 2048);
    const auto vout = TensorAccessor(voa, get_arg_val<uint32_t>(5), 2048);
    const uint32_t begin = get_arg_val<uint32_t>(6);
    const uint32_t end = get_arg_val<uint32_t>(7);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    const uint32_t kp = scratch;
    const uint32_t vp = scratch + 16384;
    const uint32_t ip = scratch + 32768;
    const uint32_t pp = scratch + 36864;
    const uint32_t lp = scratch + 36928;
    volatile tt_l1_ptr uint32_t* kw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kp);
    volatile tt_l1_ptr uint32_t* vw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vp);
    volatile tt_l1_ptr uint16_t* kh = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(kp);
    volatile tt_l1_ptr uint16_t* vh = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(vp);
    volatile tt_l1_ptr uint32_t* selected = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ip);
    volatile tt_l1_ptr uint32_t* page_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pp);
    volatile tt_l1_ptr uint16_t* lines = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(lp);
    uint64_t previous_line[32];
    for (uint32_t group = begin; group < end; ++group) {
        const uint32_t query = group / SLOT_TILES;
        const uint32_t head = 0;
        const uint32_t slot_tile = group % SLOT_TILES;
        ASSERT(query < QUERIES);
        for (uint32_t i = 0; i < 4096; ++i) { kw[i] = 0; vw[i] = 0; }
        for (uint32_t i = 0; i < 32; ++i) previous_line[i] = ~uint64_t(0);
        uint32_t previous_page_line = 0xffffffffu;
        // Q<=32, therefore selected_ids has exactly one physical tile row.
        noc_async_read(ids.get_noc_addr(slot_tile), ip, 4096);
        noc_async_read_barrier();
        for (uint32_t row = 0; row < 32; ++row) {
            const uint32_t token = selected[qsa_face_element(query, row)];
            if (token == 0xffffffffu) continue;
            ASSERT(token < LOGICAL_PAGES * PAGE_SIZE);
            const uint32_t logical_page = token / PAGE_SIZE;
            const uint32_t page_line = (logical_page * 4u) & ~63u;
            if (page_line != previous_page_line) {
                noc_async_read(pages.get_noc_addr(0) + page_line, pp, 64);
                noc_async_read_barrier();
                previous_page_line = page_line;
            }
            const uint32_t physical_page = page_words[logical_page & 15u];
            ASSERT(physical_page < PHYSICAL_PAGES);
            const uint32_t local_token = token % PAGE_SIZE;
            const uint32_t source_row = local_token & 31u;
            // Eight channels and two faces are read in parallel for K and V.
            // Consecutive selected tokens often share their aligned source
            // lines; retain each channel/face line until its address changes.
            for (uint32_t channel = 0; channel < 8; ++channel) {
                const uint32_t tile = qsa_cache_tile(physical_page, head, PAGE_SIZE, local_token, channel);
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t offset = qsa_face_element(source_row, half * 16) * 2u;
                    const uint32_t aligned = offset & ~63u;
                    const uint32_t line = channel * 2 + half;
                    const uint64_t kaddr = keys.get_noc_addr(tile) + aligned;
                    const uint64_t vaddr = values.get_noc_addr(tile) + aligned;
                    if (kaddr != previous_line[line]) {
                        noc_async_read(kaddr, lp + line * 64, 64);
                        previous_line[line] = kaddr;
                    }
                    if (vaddr != previous_line[line + 16]) {
                        noc_async_read(vaddr, lp + (line + 16) * 64, 64);
                        previous_line[line + 16] = vaddr;
                    }
                }
            }
            noc_async_read_barrier();
            for (uint32_t channel = 0; channel < 8; ++channel) {
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t line = channel * 2 + half;
                    const uint32_t source_offset = (qsa_face_element(source_row, half * 16) * 2u & 63u) / 2u;
                    const uint32_t dest = channel * 1024 + qsa_face_element(row, half * 16);
                    for (uint32_t element = 0; element < 16; ++element) {
                        kh[dest + element] = lines[line * 32 + source_offset + element];
                        vh[dest + element] = lines[(line + 16) * 32 + source_offset + element];
                    }
                }
            }
        }
        const uint32_t first_output_tile = group * 8;
        for (uint32_t channel = 0; channel < 8; ++channel) {
            noc_async_write(kp + channel * 2048, kout.get_noc_addr(first_output_tile + channel), 2048);
            noc_async_write(vp + channel * 2048, vout.get_noc_addr(first_output_tile + channel), 2048);
        }
        noc_async_write_barrier();
    }
}
"""
    )


class LocalSelectedPagedKV:
    def __init__(self, *, k_cache, v_cache, page_table, selected_ids, keys, values):
        import ttnn

        self._ttnn = ttnn
        self.tensors = [k_cache, v_cache, page_table, selected_ids, keys, values]
        shape = tuple(k_cache.shape)
        if len(shape) != 4 or shape[1] != 1 or shape[3] != 256 or shape[2] % 32:
            raise ValueError("Expected paged caches [physical_pages,1,page_size,256], page_size divisible by32")
        if tuple(v_cache.shape) != shape or min(shape) < 1:
            raise ValueError("K/V cache shapes must agree and be nonempty")
        if len(page_table.shape) != 2 or page_table.shape[0] != 1:
            raise ValueError("Page table must be [1,P]")
        if page_table.shape[1] < 1 or page_table.dtype not in (ttnn.int32, ttnn.uint32):
            raise ValueError("Page table must have INT32/UINT32 entries")
        if page_table.shape[1] % 16 and k_cache.device().arch() != ttnn.device.Arch.BLACKHOLE:
            raise ValueError("Short unaligned page-table sticks currently require Blackhole DRAM alignment")
        # Blackhole NOC_DRAM_READ_ALIGNMENT_BYTES is64. Buffer::aligned_size()
        # reserves every RM page rounded to that alignment, and TensorAccessorArgs
        # serializes aligned_page_size. Thus the final64-byte page-table read is
        # within the physical allocation even for logical P=1 or4. Other targets
        # are required to provide a full64-byte-multiple logical stick above.
        if page_table.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("Page table must be row-major")
        if len(selected_ids.shape) != 2 or selected_ids.shape[1] != 2080 or not 1 <= selected_ids.shape[0] <= 32:
            raise ValueError("selected_ids must be [Q,2080], 1<=Q<=32")
        if selected_ids.dtype != ttnn.uint32 or selected_ids.layout != ttnn.TILE_LAYOUT:
            raise ValueError("selected_ids must be tiled UINT32")
        queries = selected_ids.shape[0]
        out_shape = (queries, 1, 2080, 256)
        if tuple(keys.shape) != out_shape or tuple(values.shape) != out_shape:
            raise ValueError("Output shapes must be [Q,1,2080,256]")
        for tensor in (k_cache, v_cache, keys, values):
            if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
                raise ValueError("K/V cache and outputs must be BF16 TILE")
        for tensor in self.tensors:
            if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                raise ValueError("All tensors must be interleaved DRAM")
        addresses = [tensor.buffer_address() for tensor in self.tensors]
        if addresses[4] in addresses[:4] or addresses[5] in addresses[:5]:
            raise ValueError("Output buffers must not alias input buffers or each other")
        group_count = queries * 65
        grid = k_cache.device().compute_with_storage_grid_size()
        core_count = min(group_count, grid.x * grid.y)
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(core_count)]
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
        runtime_args = ttnn.RuntimeArgs()
        for number, core in enumerate(cores):
            runtime_args[core.x][core.y] = [
                *addresses,
                group_count * number // core_count,
                group_count * (number + 1) // core_count,
            ]
        compile_args = [shape[2], shape[0], page_table.shape[1], queries]
        for tensor in self.tensors:
            compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        self.program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=local_kv_kernel_source(),
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ranges,
                    compile_time_args=compile_args,
                    runtime_args=runtime_args,
                    config=ttnn.WriterConfigDescriptor(),
                )
            ],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=40960,
                    core_ranges=ranges,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=40960)
                    ],
                )
            ],
            semaphores=[],
        )

    def prepare(self):
        self._ttnn.experimental.prepare_generic_op(self.tensors, self.program)

    def forward(self):
        output = self._ttnn.generic_op(self.tensors, self.program)
        return self.tensors[-2], output


def local_selected_paged_kv(k_cache, v_cache, page_table, selected_ids, *, keys=None, values=None):
    """Functional device-only gather; optional outputs must be stable owned buffers."""
    import ttnn

    shape = (selected_ids.shape[0], 1, 2080, 256)
    if keys is None:
        keys = ttnn.empty(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=k_cache.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if values is None:
        values = ttnn.empty(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=k_cache.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    operation = LocalSelectedPagedKV(
        k_cache=k_cache, v_cache=v_cache, page_table=page_table, selected_ids=selected_ids, keys=keys, values=values
    )
    return operation.forward()


def _repack_ple_table(root, meta, filename, columns, words_per_row):
    """Host-only bounded partition of a complete packed cache; tile-pad each rank."""
    import hashlib
    import os

    import numpy as np

    # Setup-only bounded repack. Avoid retaining three full25.6GB code
    # copies while redistributing the original32-stripe table.
    fingerprint = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()[:16]
    target = root / f"mesh4_interleaved_v2_{fingerprint}_{filename}"
    local_rows = meta["rows"] // 4
    local_columns = math.ceil(local_rows * words_per_row / 1024) * 32
    expected_bytes = 32 * 4 * local_columns * 4
    if not target.exists() or target.stat().st_size != expected_bytes:
        source = np.memmap(root / filename, mode="r", dtype=np.uint32, shape=(32, columns))
        temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
        output = np.memmap(temporary, mode="w+", dtype=np.uint32, shape=(32, 4 * local_columns))
        for rank in range(4):
            for start in range(0, local_rows, 262144):
                end = min(start + 262144, local_rows)
                rows = 4 * np.arange(start, end, dtype=np.int64) + rank
                indices = rows[:, None] * words_per_row + np.arange(words_per_row)[None, :]
                block = source[indices % 32, indices // 32]
                begin = rank * local_columns + start * words_per_row // 32
                finish = rank * local_columns + end * words_per_row // 32
                output[:, begin:finish] = block.reshape(-1, 32).T
        output.flush()
        del output, source
        temporary.replace(target)
    prepared = np.memmap(target, mode="c", dtype=np.uint32, shape=(32, 4 * local_columns))
    return prepared


class MeshPackedEmbedding:
    """Interleaved row ownership r=id%4; exact integer local address=id>>2."""

    def __init__(self, local, rank, ccl):
        self.local, self.rank, self.ccl = local, rank, ccl
        self.rows = local.rows * 4

    @classmethod
    def from_cache(cls, cache_dir, mesh_device, ccl):
        import json
        from pathlib import Path

        import torch

        from .packed_embedding import PackedEmbedding
        from .packed_embedding_cpu import quantization_spec

        root = Path(cache_dir)
        meta = json.loads((root / "manifest.json").read_text())
        if meta["rows"] != 320001536 or meta["scale_encoding"] != "uint8_fixed_log_lut":
            raise ValueError("Expected complete group10 scale8 PLE cache")
        loader = MeshLoader({}, mesh_device)

        def table(filename, columns, words_per_row):
            prepared = _repack_ple_table(root, meta, filename, columns, words_per_row)
            return ttnn.from_torch(
                torch.from_numpy(prepared),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=DRAM,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
            )

        words = table("words.bin", meta["words_columns"], 20)
        scales = table("scales.bin", meta["scales_columns"], 4)
        spec = quantization_spec(
            quantization=meta["quantization"],
            group_size=meta["group_size"],
            codebook_values=meta["codebook_values"],
            scale_codebook_values=meta["scale_codebook_values"],
        )
        local = PackedEmbedding(
            words,
            scales,
            loader.tensor(
                torch.tensor(tuple(spec["codebook_values"]) + (0.0,) * 16).reshape(1, 32).expand(32, -1), ttnn.float32
            ),
            rows=meta["rows"] // 4,
            spec=spec,
            lanes=loader.tensor(torch.arange(20).reshape(1, 20), ttnn.uint32),
            scale_lanes=loader.tensor(torch.arange(4).reshape(1, 4), ttnn.uint32),
            scale_indices=loader.tensor((torch.arange(160) // 10).reshape(1, 160), ttnn.uint32),
            scale_codebook=loader.tensor(
                torch.tensor(spec["scale_codebook_values"], dtype=torch.bfloat16).reshape(1, 256).expand(32, -1)
            ),
        )
        from .pipelined_striped_gather import pipelined_striped_gather

        local._lookup_striped = pipelined_striped_gather
        rank = loader.shards(
            [torch.full((1, 1, 1, 16), r, dtype=torch.int32) for r in range(4)], dim=0, dtype=ttnn.uint32
        )
        return cls(local, rank, ccl)

    def __call__(self, ids):
        ids = ttnn.to_layout(ids, ttnn.TILE_LAYOUT)
        owned = ttnn.eq(ttnn.bitwise_and(ids, 3), self.rank)
        local_ids = ttnn.logical_right_shift(ids, 2)
        values = self.local(local_ids)
        n = ids.shape[-2]
        mask = ttnn.reshape(ttnn.typecast(owned, ttnn.bfloat16), (1, n * 16, 1, 1))
        values = ttnn.reshape(values, (1, n * 16, 1, 160))
        result = ttnn.reshape(ttnn.multiply(values, mask), (1, 1, n, 2560))
        return ttnn.typecast(self.ccl.sum(result), ttnn.bfloat16)


class MeshPLE(PLE):
    def __init__(self, loader, config):
        self.width, self.streams, self.eps = 2560, 1, config.rms_norm_eps
        self.dilation, self.kernel, self.state_len = 3, 4, 9
        self.compute = loader.compute
        w = loader.state_dict
        self.key = loader.shards([p.T for p in w["ple.key_proj.weight"].chunk(4, dim=0)], dim=1, dtype=ttnn.bfloat8_b)
        # HF broadcasts one2560-wide value across the four stream-specific gates.
        self.value = loader.tensor(w["ple.value_proj.weight"].T, ttnn.bfloat8_b)
        for role in ("key", "query", "conv"):
            values = w["ple.norm_" + role + ".weight"].float().reshape(4, 2560) + 1
            setattr(
                self,
                "norm_" + role,
                loader.shards([v.reshape(1, 1, 1, 2560) for v in values], dim=3, dtype=ttnn.float32),
            )
        self.conv = [
            loader.shards([p.reshape(1, 1, 1, 2560) for p in w["ple.conv1d.weight"][:, 0, tap].chunk(4)], dim=3)
            for tap in range(4)
        ]
