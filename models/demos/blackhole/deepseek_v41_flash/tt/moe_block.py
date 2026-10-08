# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash MoE block (router + 384 routed experts + 1 shared expert), decode only.

A thin composition of the repo's generic modules:

    x ──TTMoEGate──▶ (weights, indices) ──TTMoEDecode──▶ routed + shared expert output

* EP=32: experts are sharded over every device on the expert dimension (12 per device), tokens are
  dispatched along ``cluster_axis`` (mesh rows) and the output is reduce-scattered over the other axis,
  so each device ends with ``[1, 1, tokens_per_device, hidden / mesh_cols]`` -- the same hidden-sharded
  residual layout GPT-OSS uses.
* The checkpoint's swiglu clamp is not applied (``moe_compute`` SILU has none); see moe_weights.py.
"""

from pathlib import Path

import ttnn
from models.common.modules.moe.tt_moe_decode import TTMoEDecode
from models.common.modules.moe.tt_moe_decode_config import TTMoEDecodeConfig
from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig
from models.demos.blackhole.deepseek_v41_flash.tt.router import DSV41Gate


class _TailDecode(TTMoEDecode):
    """TTMoEDecode whose tilize + fast_reduce tail is one fused program (moe_tail.py) when T < 32; set DSV41_MOE_TAIL=0 for the stock path."""

    def forward(self, tt_x, tt_scores, tt_indices, layer_id: int = 0):
        import os

        cfg = self.config
        if (
            os.environ.get("DSV41_MOE_TAIL", "1") == "0"
            or cfg.batch_per_device >= ttnn.TILE_SIZE
            or cfg.num_fast_reduce_outputs != 1
            or self._needs_fast_reduce_padding
            or cfg.num_shared_experts != 0
        ):
            return super().forward(tt_x, tt_scores, tt_indices, layer_id)
        from models.demos.blackhole.deepseek_v41_flash.tt.moe_tail import make_col_tensor, moe_tail

        if not hasattr(self, "_col"):
            self._col = make_col_tensor(self._mesh)
        (x_t, d_x), (i_t, d_i), (s_t, d_s) = self._format_dispatch_inputs(tt_x, tt_indices, tt_scores)
        sparse, o_idx, o_sc = ttnn.experimental.all_to_all_dispatch_metadata(
            x_t,
            i_t,
            s_t,
            self.expert_state.tt_expert_mapping,
            **cfg.dispatch.model_dump(),
            output_tensors=self.buffers.tt_dispatch_output_tensors,
            cross_device_semaphore=self.buffers.dispatch_global_semaphore,
        )
        if d_x:
            ttnn.deallocate(x_t)
        if d_s:
            ttnn.deallocate(s_t)
        _, _, _, l1_out, _, combine = ttnn.experimental.moe_compute(
            sparse,
            o_idx,
            o_sc,
            self.expert_state.tt_expert_mapping,
            self.expert_state.tt_w0_w1,
            self.expert_state.tt_w2,
            layer_id=layer_id,
            **cfg.compute.model_dump(),
            optional_output_tensor=self.buffers.tt_combine_output,
            optional_cross_device_semaphore=self.buffers.combine_global_semaphore,
        )
        ttnn.deallocate(l1_out)
        red = moe_tail(combine, tt_scores, tt_indices, self.expert_state.tt_expert_mapping, self._col)
        if d_i:
            ttnn.deallocate(i_t)
        out = ttnn.reduce_scatter(red, **cfg.reduce_scatter.model_dump())
        ttnn.deallocate(red)
        return out


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "deepseek_v41_flash.yaml"


ROUTE_LOG = [] if __import__("os").environ.get("DSV41_ROUTE_CAPTURE") == "1" else None

ROUTE_ON = [False]  # set by the caller around EAGER decode steps only (never during trace capture)


def _read_rows(md, t):
    """[B, k] torch int64 of a row-sharded (dim 0) / column-replicated tensor."""
    import torch

    ttnn.synchronize_device(md)
    rows, cols = tuple(md.shape)
    devs = ttnn.get_device_tensors(t)
    return torch.cat([ttnn.to_torch(devs[r * cols]).reshape(-1, t.shape[-1]).long() for r in range(rows)])


class DSV41MoEBlock:
    def __init__(
        self,
        mesh_device,
        weights: dict,
        topology=ttnn.Topology.Linear,
        batch_per_device: int = 4,
        gate_bias_shift=0.0,
        shared_in_moe: bool = False,
        buffers=None,
        expert_state=None,
    ):
        """``shared_in_moe=False`` (default) leaves the shared expert out of ``moe_compute`` (bfp4-only) -- the layer
        adds it separately in higher precision, see shared_expert.py. ``expert_state``: the already uploaded routed-expert weights of this layer
        (``_TTMoEDecodeExpertState`` of an earlier block, batch independent): they are reused instead of read from the cache again (Model.reconfigure).
        """
        self._topology, self._shared_in_moe = topology, shared_in_moe
        decode_cfg, gate_cfg = self._configs(mesh_device, batch_per_device, topology, shared_in_moe)

        self.mesh_device = mesh_device
        self.decode_config = decode_cfg
        # bf16-exact selection bias + fp32 residual folded into the score (see router.py)
        self.gate = DSV41Gate(
            mesh_device,
            gate_cfg,
            torch_gate_weight=weights["gate_weight"],
            torch_gate_bias=weights["gate_bias"],
            bias_shift=gate_bias_shift,
        )
        if expert_state is not None:
            self.decode = self._weightless_decode(mesh_device, decode_cfg, buffers, expert_state=expert_state)
        elif __import__("os").environ.get("DSV41_UNI_NODECODE") == "1":
            # PREFILL-ONLY measurement mode (DSV41_PREFILL_MOE=unified): no moe_compute expert weights on the device (DRAM would not fit them next to the
            # unified-layout copy); decode through this block is not possible. Scratch buffers / config as TTMoEDecode.__init__ builds them.
            self.decode = self._weightless_decode(mesh_device, decode_cfg, buffers)
        else:
            self._build_decode(mesh_device, decode_cfg, weights, shared_in_moe, buffers)
        self.decode._mesh = mesh_device

    @staticmethod
    def _configs(mesh_device, batch_per_device, topology, shared_in_moe):
        """(decode config, gate config) for ``batch_per_device`` tokens per device."""
        text = CONFIG_PATH.read_text()
        # the derived memory configs (dispatch input shards, ...) are computed from batch_per_device when the config is parsed,
        # so the batch size has to be in the YAML text itself (updating the field afterwards leaves them sized for 4 users)
        text = text.replace("batch_per_device: 4 ", f"batch_per_device: {batch_per_device} ", 1)
        if not shared_in_moe:
            text = text.replace("num_shared_experts: 1", "num_shared_experts: 0").replace(
                "  shared_expert_ids_to_devices: fully_replicated\n", ""
            )
        mesh_shape = tuple(mesh_device.shape)
        decode_cfg = TTMoEDecodeConfig.from_yaml(text, topology=topology)
        if decode_cfg.mesh_shape != mesh_shape:
            decode_cfg = decode_cfg.with_mesh_shape(mesh_shape)
        if decode_cfg.batch_per_device != batch_per_device:
            decode_cfg = decode_cfg.model_copy(update={"batch_per_device": batch_per_device})
        if decode_cfg.num_fast_reduce_outputs == 1:
            # Generic ttnn.reduce_scatter path (non-Ring topology): its CCL address generator rejects the
            # ND_SHARDED L1 default of the fast-reduce output, so hand it interleaved DRAM instead.
            decode_cfg = decode_cfg.model_copy(
                update={
                    "reduce": decode_cfg.reduce.model_copy(update={"output_memory_config": ttnn.DRAM_MEMORY_CONFIG})
                }
            )
        gate_cfg = TTMoEGateConfig.from_yaml(text).model_copy(update={"batch_per_device": batch_per_device})
        return decode_cfg, gate_cfg

    def for_batch(self, batch_per_device, buffers=None):
        """A block for ANOTHER number of tokens per device (a decode bucket, tt/decode_buckets.py) that shares everything batch independent with this one (routed-expert weights, the
        router: its exact path reads the token count from its input) and owns the decode scratch buffers / config of the new size (``buffers``: those of an earlier block of the
        same bucket). The expert-weight state of this block is reused."""
        import copy

        nb = copy.copy(self)
        decode_cfg, _ = self._configs(self.mesh_device, batch_per_device, self._topology, self._shared_in_moe)
        nb.decode_config = decode_cfg
        nb.decode = self._weightless_decode(
            self.mesh_device, decode_cfg, buffers, expert_state=self.decode.expert_state
        )
        nb.decode._mesh = self.mesh_device
        return nb

    @staticmethod
    def _weightless_decode(mesh_device, cfg, buffers, expert_state=None):
        from types import SimpleNamespace

        from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size

        from models.common.modules.moe.tt_moe_decode import _TTMoEDecodeBuffers

        dec = object.__new__(_TailDecode)
        dec.config = cfg
        dec.expert_state = (
            expert_state
            if expert_state is not None
            else SimpleNamespace(tt_expert_mapping=None, tt_w0_w1=None, tt_w2=None)
        )
        if buffers is None:
            bd = cfg.buffers.model_dump()
            bd["compute_tilize_drain_core"] = ttnn.experimental.get_moe_tilize_drain_core(
                mesh_device,
                cfg.compute.output_height_shard_dim,
                auto_output_width_shard_dim(cfg.hidden_size, matmul_ring_size=effective_matmul_ring_size(mesh_device)),
                cfg.hidden_size,
                mux_core_range_set=cfg.compute.mux_core_range_set,
            )
            buffers = _TTMoEDecodeBuffers(mesh_device, **bd)
        dec.buffers = buffers
        return dec

    def _build_decode(self, mesh_device, decode_cfg, weights, shared_in_moe, buffers):
        self.decode = _TailDecode(
            mesh_device=mesh_device,
            config=decode_cfg,
            torch_w0=weights["w0"],
            torch_w1=weights["w1"],
            torch_w2=weights["w2"],
            shared_id_to_torch_w0=weights["shared_w0"] if shared_in_moe else None,
            shared_id_to_torch_w1=weights["shared_w1"] if shared_in_moe else None,
            shared_id_to_torch_w2=weights["shared_w2"] if shared_in_moe else None,
            weight_cache_dir=weights.get("cache_dir"),
            buffers=buffers,
        )

    def warmup(self):
        """Compile the MoE programs once, steering where ``moe_compute``'s persistent semaphore lands in L1.

        ``moe_compute`` creates a global semaphore (a 320 B/bank L1 allocation owned by its cached program) on its
        first compile, AFTER its ~650 KB/bank L1 outputs were allocated. The L1 allocator is top-down first-fit, so
        the semaphore lands just below those outputs, permanently capping the static circular-buffer region of every
        other program at ~770 KB (SDPA decode at head_dim 512 then fails to launch). To put it at the top of L1
        instead, leave a small free hole just under the persistent allocations before the first compile: the big
        outputs cannot fit in it, the semaphore can.
        """
        md = self.mesh_device
        if __import__("os").environ.get("DSV41_UNI_NODECODE") == "1":
            return  # prefill-only unified mode: no moe_compute weights, nothing to compile
        nb = ttnn.get_memory_view(md, ttnn.BufferType.L1).num_banks
        tile_row = lambda n_tiles_per_bank: ttnn.empty(
            [1, 1, 32, 32 * nb * n_tiles_per_bank],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self._l1("warmup: start")
        hole, fence = tile_row(1), tile_row(2)  # 2 KB then 4 KB per bank, adjacent below the persistent allocations
        self._l1("warmup: hole+fence allocated")
        ttnn.deallocate(hole)  # -> a 2 KB hole at the top of L1
        self._l1("warmup: hole freed")
        T = self.decode_config.batch_per_device
        zeros = lambda shape, lay, dt: ttnn.from_torch(
            __import__("torch").zeros(shape),
            device=md,
            dtype=dt,
            layout=lay,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        x_gate = zeros([1, 1, T, self.decode_config.hidden_size], ttnn.TILE_LAYOUT, ttnn.bfloat16)
        x_tok = zeros([T, 1, 1, self.decode_config.hidden_size], ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)
        out = self.forward(x_gate, x_tok)
        ttnn.synchronize_device(md)
        self._l1("warmup: after forward (out alive)")
        ttnn.deallocate(out)
        ttnn.deallocate(fence)
        self._l1("warmup: end")

    def forward(self, tt_x_gate: ttnn.Tensor, tt_x_tokens: ttnn.Tensor, forced_routing=None) -> ttnn.Tensor:
        """tt_x_gate: [1, 1, tokens_per_device, hidden] tile layout (router input).
        tt_x_tokens: [tokens_per_device, 1, 1, hidden] row-major bf16 DRAM (dispatch input).
        Returns [1, 1, tokens_per_device, hidden / mesh_cols] per device."""
        self._l1("start")
        if (
            forced_routing is not None
        ):  # (scores bf16 RM [T,1,1,k], indices uint16 RM [T,1,1,k]) -- bypass the router (diagnostics)
            tt_scores, tt_indices = forced_routing
        else:
            tt_scores, tt_indices = self.gate.forward(tt_x_gate)
        self.last_routing = (tt_scores, tt_indices)  # for debugging / error-budget tests
        if (
            ROUTE_LOG is not None and ROUTE_ON[0]
        ):  # DSV41_ROUTE_CAPTURE=1 (eager decode only): host copy of the REAL routed expert ids, one entry per call
            ROUTE_LOG.append(_read_rows(self.mesh_device, tt_indices))
        self._l1("after gate")
        if tt_indices.dtype != ttnn.uint16:
            tt_indices = ttnn.typecast(tt_indices, ttnn.uint16)
        if tt_indices.layout != ttnn.ROW_MAJOR_LAYOUT:
            tt_indices = ttnn.to_layout(tt_indices, ttnn.ROW_MAJOR_LAYOUT)
        if tt_scores.dtype != ttnn.bfloat16:
            tt_scores = ttnn.typecast(tt_scores, ttnn.bfloat16)
        if tt_scores.layout != ttnn.ROW_MAJOR_LAYOUT:
            tt_scores = ttnn.to_layout(tt_scores, ttnn.ROW_MAJOR_LAYOUT)
        self._l1("after typecasts")
        out = self.decode.forward(tt_x=tt_x_tokens, tt_scores=tt_scores, tt_indices=tt_indices, layer_id=0)
        self._l1("after decode")
        return out

    def _l1(self, tag):  # debug: DSV_L1_TRACE=<file> appends L1 allocator readings
        import os

        path = os.environ.get("DSV_L1_TRACE")
        if path:
            ttnn.synchronize_device(self.mesh_device)
            mv = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.L1)
            with open(path, "a") as f:
                f.write(
                    f"MOE {tag:16s} allocated/bank {mv.total_bytes_allocated_per_bank:7d} largest_free {mv.largest_contiguous_bytes_free_per_bank:8d}\n"
                )
