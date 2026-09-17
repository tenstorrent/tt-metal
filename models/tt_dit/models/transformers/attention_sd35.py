# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import os

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole

from ...layers.linear import ColParallelLinear
from ...layers.module import Module
from ...layers.normalization import RMSNorm
from ...utils.padding import pad_weight_tensor
from ...utils.substate import pop_substate, rename_substate


def is_fused_tp(parallel_config, ccl_manager) -> bool:
    """Ring TP fuses the spatial gathers into the projections (AGMM / MMRS); Linear TP cannot.

    ``SD35_FUSED_TP=0`` forces the explicit-gather path on a ring as well, for same-fabric A/B runs.
    """
    return (
        parallel_config.tensor_parallel.factor > 1
        and ccl_manager is not None
        and ccl_manager.topology == ttnn.Topology.Ring
        and os.environ.get("SD35_FUSED_TP", "1") != "0"
    )


def flatten_batch(x: ttnn.Tensor) -> ttnn.Tensor:
    """[1, B, N, D] -> [1, 1, B*N, D]. The fused all-gather-matmul / matmul-reduce-scatter ops need a
    unit batch, and every row-wise op is indifferent to the fold; N is a tile multiple so this is a
    view. The CFG pair (B=2) becomes one tall M (uncond rows first, then cond)."""
    one, b, n, d = x.shape
    if b == 1:
        return x
    return ttnn.reshape(x, (1, 1, b * n, d))


def unflatten_batch(x: ttnn.Tensor, like_shape) -> ttnn.Tensor:
    """Inverse of flatten_batch: [1, 1, B*N, D'] -> [1, B, N, D'] with B, N taken from ``like_shape``."""
    one, b, n, _ = like_shape
    if b == 1:
        return x
    return ttnn.reshape(x, (1, b, n, x.shape[-1]))


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py
class SD35JointAttention(Module):
    # Map from (is_blackhole, sp_factor, tp_factor) -> (q_chunk_size, k_chunk_size)
    sdpa_chunk_size_map = {
        (False, 2, 2): (256, 512),
        (False, 4, 4): (256, 512),
        (True, 2, 2): (256, 512),
        (True, 4, 4): (128, 512),
        # tp4 column, no SP: joint SDPA on the full 12x10 grid with streaming compute + KV chain
        # forwarding (q256 k512: 799 us/call vs 1290 legacy; k256 869)
        (True, 1, 4): (256, 512),
    }
    default_sdpa_chunk_size = (256, 512)

    def __init__(
        self,
        query_dim,
        head_dim,
        heads,
        out_dim=None,
        bias=False,
        out_bias=True,
        context_pre_only=None,
        eps=1e-5,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
        padding_config=None,
        quant_config=None,
    ):
        super().__init__()

        self.query_dim = query_dim
        self.head_dim = head_dim
        self.heads = heads
        self.padding_config = padding_config
        self.padded_heads = padding_config.target_heads if padding_config is not None else heads

        self.out_dim = out_dim if out_dim is not None else query_dim
        # Note: added_kv_proj_dim should be passed as parameter, using query_dim as default
        self.added_kv_proj_dim = query_dim
        self.context_pre_only = context_pre_only
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.quant_config = quant_config
        # Quantized projections (to_qkv / add_qkv_proj) and the bf16-carved output projections
        # (to_out / to_add_out) get their dtype/activation/compute-fidelity from the quant profile;
        # a None profile leaves every kwarg dict empty => the unquantized bf16 model.
        _mm_cc = quant_config.mm_compute_config(mesh_device.arch()) if quant_config is not None else None
        _qkv_q = dict(quant_config.qkv_linear_kwargs()) if quant_config is not None else {}
        _out_q = dict(quant_config.out_linear_kwargs()) if quant_config is not None else {}
        if _mm_cc is not None:
            _qkv_q["compute_kernel_config"] = _mm_cc
            _out_q["compute_kernel_config"] = _mm_cc
        self._sdpa_input_dtype = quant_config.sdpa_input_dtype if quant_config is not None else None
        # Narrow the TP attention-output gathers to bf8 when activations are quantized (None => bf16).
        self._ag_dtype = quant_config.activation_dtype if quant_config is not None else None

        self.n_local_heads = self.padded_heads // self.parallel_config.tensor_parallel.factor

        # Ring TP fuses the spatial activation gathers into the projections: to_qkv and to_out run
        # as all-gather-matmuls on the TP-fractured input (strided fabric-bound op for the swept
        # shapes, all_gather_minimal_matmul_async otherwise), and to_out also takes the block's
        # gated residual in its epilogue. Linear TP keeps the explicit gathers (the fused ops need
        # a ring). The prompt projections stay on the gather + matmul path: at M ~ 160 they are
        # memory-bound and the fused ops buy nothing.
        self.fused_tp = is_fused_tp(parallel_config, ccl_manager)

        self.inner_dim = out_dim if out_dim is not None else head_dim * self.heads
        self.padded_inner_dim = head_dim * self.padded_heads
        rms_kwargs = {
            "embedding_dim": head_dim,
            "norm_eps": eps,
            "norm_elementwise_affine": True,
            "bias": False,
            "mesh_device": mesh_device,
        }

        self.norm_q = RMSNorm(**rms_kwargs)
        self.norm_k = RMSNorm(**rms_kwargs)

        # Fused QKV projection. ccl_manager lets the fused-TP path run it as an all-gather-matmul.
        self.to_qkv = ColParallelLinear(
            query_dim,
            3 * self.padded_inner_dim,
            bias=bias,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            **_qkv_q,
        )

        # Implementing joint attention
        self.add_qkv_proj = ColParallelLinear(
            self.added_kv_proj_dim,
            3 * self.padded_inner_dim,
            bias=bias,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            **_qkv_q,
        )

        self.to_out = ColParallelLinear(
            self.padded_inner_dim,
            self.out_dim,
            bias=out_bias,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            **_out_q,
        )

        if self.context_pre_only is not None and not self.context_pre_only:
            # TODO: Use `out_context_dim` parameter if given
            self.to_add_out = ColParallelLinear(
                self.padded_inner_dim,
                self.out_dim,
                bias=out_bias,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                **_out_q,
            )

        self.norm_added_q = RMSNorm(**rms_kwargs)
        self.norm_added_k = RMSNorm(**rms_kwargs)

        full_grid = self.mesh_device.compute_with_storage_grid_size()
        # The reserved row is for the ring SDPA's CCL workers; without sequence parallelism the
        # joint SDPA can use the full grid (12% faster on the tp4 column, 1396 vs 1584 us per call).
        if self.parallel_config.sequence_parallel.factor > 1:
            self.sdpa_worker_grid = (full_grid.x, full_grid.y - 1)
        else:
            self.sdpa_worker_grid = (full_grid.x, full_grid.y)
        ring_sdpa_chunk_size = self.sdpa_chunk_size_map.get(
            (
                is_blackhole(),
                self.parallel_config.sequence_parallel.factor,
                self.parallel_config.tensor_parallel.factor,
            ),
            self.default_sdpa_chunk_size,
        )
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_worker_grid,
            q_chunk_size=ring_sdpa_chunk_size[0],
            k_chunk_size=ring_sdpa_chunk_size[1],
            exp_approx_mode=False,  # NOTE: False is more correct
        )
        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,  # NOTE: Set to True if there's a correctness issue
        )

        # Experimental ring joint SDPA (head-segment scheduler, as used by MiniMax-H3): overlaps the
        # K/V ring exchange with attention compute. Only meaningful under sequence parallelism.
        # SD35_EXP_RING=0 keeps ring_joint_scaled_dot_product_attention. The op walks
        # B * heads * segments as serial passes over the SDPA rows; the factory caps the pass count
        # (kMaxPasses, 3 in main), overridable here for a rebuilt kernel via SD35_EXP_RING_MAX_PASSES.
        self.full_grid = full_grid
        # SD35_EXP_RING_SP1=1 also tries the op without sequence parallelism (ring of one device), as
        # an alternative joint-SDPA implementation (head-segment scheduler) on the tp4 layout.
        self.use_exp_ring = (
            (self.parallel_config.sequence_parallel.factor > 1 or os.environ.get("SD35_EXP_RING_SP1", "0") == "1")
            and is_blackhole()
            and os.environ.get("SD35_EXP_RING", "1") == "1"
        )
        self.exp_ring_max_passes = int(os.environ.get("SD35_EXP_RING_MAX_PASSES", "3"))
        self.exp_ring_max_k_chunk = 512
        self._exp_ring_configs: dict[tuple[int, int, int], ttnn.SDPAProgramConfig | None] = {}
        # Optional private, deeper semaphore pool for the exp op (the CCL manager's is 2-deep):
        # SD35_EXP_RING_SEM_DEPTH=n rotates over n sets of num_links global semaphores. Diagnostic
        # knob for the call-to-call state leak seen at 8 passes (2026-09-17).
        self._exp_sem_depth = int(os.environ.get("SD35_EXP_RING_SEM_DEPTH", "2"))
        self._exp_sem_pool = None
        self._exp_sem_idx = 0

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    # ---- experimental ring joint SDPA configuration (adapted from attention_minimax_h3.py) ----
    _EXP_L1_TILE_BYTES = 2048
    _EXP_USABLE_L1_BYTES = 1_312_640
    _EXP_DST_TILES = 8
    _EXP_SUBBLOCKS = (
        (2, 4), (4, 2), (1, 8), (8, 1), (1, 7), (7, 1), (2, 3), (3, 2), (1, 6), (6, 1),
        (1, 5), (5, 1), (2, 2), (1, 4), (4, 1), (1, 3), (3, 1), (1, 2), (2, 1), (1, 1),
    )  # fmt: skip

    def _exp_streaming_compute_enabled(self, sq_t: int, sk_t: int) -> bool:
        """Mirrors `use_streaming_compute` in exp_ring_joint_sdpa_program_factory.cpp (the compute
        kernel static_asserts on it, so an ineligible shape fails to build rather than falling back)."""
        dst = self._EXP_DST_TILES
        for h, w in self._EXP_SUBBLOCKS:
            if h * w <= dst and sq_t % h == 0 and sk_t % w == 0:
                return h <= 2 and sk_t % (dst // h) == 0 and sq_t // h > 1
        return False

    def _exp_sdpa_l1_bytes(self, sq_t: int, sk_t: int, p: int, resident_q: bool = True) -> int:
        """L1 the exp op's circular buffers need for (q, k, passes), mirroring its CB table."""
        dh_t = self.head_dim // ttnn.TILE_SIZE
        tiles = (
            (p if resident_q else 1) * sq_t * dh_t
            + 4 * sk_t * dh_t
            + 7
            + 2 * p * sq_t
            + p * sq_t * dh_t
            + sq_t
            + 16
            + sq_t * sk_t
            + 2 * sq_t * dh_t
            + 4 * sq_t
            + sq_t
        )
        return tiles * self._EXP_L1_TILE_BYTES

    def _exp_ring_program_config(self, n_local: int, joint_len: int, batch: int):
        """Search (cols, segs_per_head, q_chunk, k_chunk) for the exp ring op, or None if nothing fits.

        A head's Q chunks (local N chunks plus the joint/prompt chunks) must fill whole SDPA rows:
        ceil(n_local / q) + ceil(joint_len / q) == cols * segs. Passes = ceil(B * heads * segs / rows)
        must stay within the factory's cap. Score: lightest per-core load (passes * q_chunk), then the
        largest k_chunk, then the widest grid.
        """
        key = (n_local, joint_len, batch)
        if key in self._exp_ring_configs:
            return self._exp_ring_configs[key]
        tile = ttnn.TILE_SIZE
        rows = self.full_grid.y - (self.full_grid.y % 2)  # SDPA rows must be even
        # The op requires the (tile-padded) joint length to be a whole number of Q chunks.
        joint_padded = math.ceil(joint_len / tile) * tile
        best = None
        for cols in range(self.full_grid.x - 1, 2, -1):
            for segs in (1, 2, 3):
                chunks = cols * segs
                q_cands = [
                    q
                    for q in range(tile, n_local + tile, tile)
                    if math.ceil(n_local / q) + math.ceil(joint_len / q) == chunks
                    and (joint_padded == 0 or joint_padded % q == 0)
                ]
                if not q_cands:
                    continue
                q_chunk = q_cands[0]
                passes = math.ceil(batch * self.n_local_heads * segs / rows)
                if passes > self.exp_ring_max_passes:
                    continue
                for k_chunk in range(self.exp_ring_max_k_chunk, 0, -tile):
                    sq_t, sk_t = q_chunk // tile, k_chunk // tile
                    fits = (
                        self._exp_sdpa_l1_bytes(sq_t, sk_t, passes) <= self._EXP_USABLE_L1_BYTES
                        or self._exp_sdpa_l1_bytes(sq_t, sk_t, passes, resident_q=False) <= self._EXP_USABLE_L1_BYTES
                    )
                    if fits and self._exp_streaming_compute_enabled(sq_t, sk_t):
                        score = (passes * q_chunk, -k_chunk, -cols)
                        if best is None or score < best[0]:
                            best = (score, cols, q_chunk, k_chunk, passes, segs)
                        break
        cfg = None
        if best is not None:
            _, cols, q_chunk, k_chunk, passes, segs = best
            cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(cols + 1, rows),
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            )
            logger.info(
                f"SD35 exp ring SDPA: N_local={n_local} L={joint_len} B={batch} heads/dev={self.n_local_heads} -> "
                f"grid {cols + 1}x{rows} (sdpa {cols}x{rows}), q_chunk={q_chunk}, k_chunk={k_chunk}, segs={segs}, passes={passes}"
            )
        else:
            logger.warning(
                f"SD35 exp ring SDPA: no config for N_local={n_local} L={joint_len} B={batch} "
                f"heads/dev={self.n_local_heads} within {self.exp_ring_max_passes} passes; using ring_joint SDPA"
            )
        self._exp_ring_configs[key] = cfg
        return cfg

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        def reshape_and_merge_qkv(q_state, k_state, v_state):
            # Rearrange QKV projections such column-fracturing shards the heads
            def _merge_tensors(q, k, v):
                n_dev = self.parallel_config.tensor_parallel.factor
                q, k, v = q.T, k.T, v.T
                # Pad QKV weights and biases to match the padded heads
                if self.padding_config is not None:
                    q = pad_weight_tensor(q, self.padding_config, pad_output_dim=True)
                    k = pad_weight_tensor(k, self.padding_config, pad_output_dim=True)
                    v = pad_weight_tensor(v, self.padding_config, pad_output_dim=True)
                q = q.reshape(q.shape[0], n_dev, self.n_local_heads, self.head_dim)
                k = k.reshape(k.shape[0], n_dev, self.n_local_heads, self.head_dim)
                v = v.reshape(v.shape[0], n_dev, self.n_local_heads, self.head_dim)
                qkv = torch.cat([q, k, v], dim=2)
                qkv = qkv.reshape(qkv.shape[0], 3 * self.padded_heads * self.head_dim)
                qkv = qkv.T
                return qkv

            weight = _merge_tensors(q_state["weight"], k_state["weight"], v_state["weight"])

            out_state = {"weight": weight}
            if "bias" in q_state:
                bias = _merge_tensors(
                    q_state["bias"].unsqueeze(-1), k_state["bias"].unsqueeze(-1), v_state["bias"].unsqueeze(-1)
                )
                bias = bias.squeeze(-1)
                out_state["bias"] = bias
            return out_state

        qkv_state = reshape_and_merge_qkv(
            pop_substate(state, "to_q"), pop_substate(state, "to_k"), pop_substate(state, "to_v")
        )
        state["to_qkv.weight"] = qkv_state["weight"]
        if "bias" in qkv_state:
            state["to_qkv.bias"] = qkv_state["bias"]

        add_qkv_state = reshape_and_merge_qkv(
            pop_substate(state, "add_q_proj"),
            pop_substate(state, "add_k_proj"),
            pop_substate(state, "add_v_proj"),
        )
        state["add_qkv_proj.weight"] = add_qkv_state["weight"]
        if "bias" in add_qkv_state:
            state["add_qkv_proj.bias"] = add_qkv_state["bias"]

        rename_substate(state, "to_out.0", "to_out")

        if self.padding_config is not None:
            if "to_out.weight" in state:
                weight = state["to_out.weight"].T
                weight = pad_weight_tensor(weight, self.padding_config, pad_input_dim=True)
                state["to_out.weight"] = weight.T
            if "to_add_out.weight" in state:
                weight = state["to_add_out.weight"].T
                weight = pad_weight_tensor(weight, self.padding_config, pad_input_dim=True)
                state["to_add_out.weight"] = weight.T

    def _ag_tp(self, x):
        """All-gather ``x`` on the TP feature axis (dim=3), narrowing to bf8 when activations are
        quantized so the collective moves half the bytes. None ``_ag_dtype`` => original bf16 gather."""
        axis = self.parallel_config.tensor_parallel.mesh_axis
        if self._ag_dtype is not None:
            x = ttnn.typecast(x, self._ag_dtype)
        return ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                x.shape, 3, axis, dtype=self._ag_dtype or ttnn.bfloat16
            ),
            dim=3,
            multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(axis),
            num_links=self.ccl_manager.num_links,
            topology=self.ccl_manager.topology,
            cluster_axis=axis,
            **self.ccl_manager.get_ag_hyperparams(x.shape),
        )

    def forward(self, spatial_1BND, prompt_1BLD, N, spatial_residual=None, spatial_gate=None):
        """
        Inputs are replicated (Linear TP) — or, under fused Ring TP, ``spatial_1BND`` is still
        TP-fractured on D and ``prompt_1BLD`` replicated.
        Outputs are width-fractured.

        ``spatial_residual`` / ``spatial_gate`` (fused TP only): the block's residual stream
        [1, B, N, D/tp] and its attention gate at every flattened row [1, 1, B*N, D/tp]. When given,
        the spatial output is ``residual + gate * to_out(attn)`` computed in the to_out epilogue,
        i.e. the updated residual stream rather than the bare projection.
        """

        if self.fused_tp:
            qkv_flat = self.to_qkv(flatten_batch(spatial_1BND), parallel_config=self.parallel_config)
            qkv_1BNF = unflatten_batch(qkv_flat, spatial_1BND.shape)
        else:
            qkv_1BNF = self.to_qkv(spatial_1BND)
        local_heads = self.n_local_heads
        q_BHNE, k_BHNE, v_BHNE = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.squeeze(qkv_1BNF, 0), num_heads=local_heads, transpose_key=False
        )

        q_BHNE = self.norm_q(q_BHNE)
        k_BHNE = self.norm_k(k_BHNE)

        add_qkv_1BLF = self.add_qkv_proj(prompt_1BLD)
        add_q_BHLE, add_k_BHLE, add_v_BHLE = ttnn.transformer.split_query_key_value_and_split_heads(
            ttnn.squeeze(add_qkv_1BLF, 0), num_heads=local_heads, transpose_key=False
        )
        add_q_BHLE = self.norm_added_q(add_q_BHLE)
        add_k_BHLE = self.norm_added_k(add_k_BHLE)

        # Narrow SDPA inputs to bf8 when activations are quantized: shrinks the ring KV all-gather
        # payload and the attention feed. SDPA math stays HiFi2 (only the inputs narrow).
        if self._sdpa_input_dtype is not None:
            q_BHNE = ttnn.typecast(q_BHNE, self._sdpa_input_dtype)
            k_BHNE = ttnn.typecast(k_BHNE, self._sdpa_input_dtype)
            v_BHNE = ttnn.typecast(v_BHNE, self._sdpa_input_dtype)
            add_q_BHLE = ttnn.typecast(add_q_BHLE, self._sdpa_input_dtype)
            add_k_BHLE = ttnn.typecast(add_k_BHLE, self._sdpa_input_dtype)
            add_v_BHLE = ttnn.typecast(add_v_BHLE, self._sdpa_input_dtype)

        exp_cfg = None
        if self.use_exp_ring:
            exp_cfg = self._exp_ring_program_config(q_BHNE.shape[2], add_q_BHLE.shape[2], q_BHNE.shape[0])
        if exp_cfg is not None:
            sp_axis = self.parallel_config.sequence_parallel.mesh_axis
            nbuf = int(os.environ.get("SD35_EXP_RING_NBUF", "32"))

            def _exp_sems():
                if self._exp_sem_depth <= 2:
                    return self.ccl_manager.get_exp_ring_ping_pong_semaphore(sp_axis)
                if self._exp_sem_pool is None:
                    ttnn.synchronize_device(self.mesh_device)
                    self._exp_sem_pool = [
                        [
                            ttnn.create_global_semaphore(self.mesh_device, self.ccl_manager.ccl_cores, 0)
                            for _ in range(self.ccl_manager.num_links)
                        ]
                        for _ in range(self._exp_sem_depth)
                    ]
                    ttnn.synchronize_device(self.mesh_device)
                sems = self._exp_sem_pool[self._exp_sem_idx]
                self._exp_sem_idx = (self._exp_sem_idx + 1) % self._exp_sem_depth
                return sems

            def _exp(q, k, v, jq, jk, jv, cfg):
                return ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    jq,
                    jk,
                    jv,
                    persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                        k.shape, 2, sp_axis, dtype=self._sdpa_input_dtype or ttnn.bfloat16
                    ),
                    persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                        v.shape, 2, sp_axis, dtype=self._sdpa_input_dtype or ttnn.bfloat16
                    ),
                    joint_strategy="rear",
                    logical_n=N,
                    program_config=cfg,
                    compute_kernel_config=self.sdpa_compute_kernel_config,
                    dim=2,
                    multi_device_global_semaphore=_exp_sems(),
                    num_links=self.ccl_manager.num_links,
                    cluster_axis=sp_axis,
                    mesh_device=self.mesh_device,
                    topology=self.ccl_manager.topology,
                    subdevice_id=self.ccl_manager.ccl_sub_device_id,
                    num_workers_per_link=cfg.compute_with_storage_grid_size.y // 2,
                    num_buffers_per_channel=nbuf,
                )

            if os.environ.get("SD35_EXP_RING_SPLIT_CFG", "0") == "1" and q_BHNE.shape[0] > 1:
                # One SDPA call per CFG branch (halves the head instances per call, so half the passes).
                cfg1 = self._exp_ring_program_config(q_BHNE.shape[2], add_q_BHLE.shape[2], 1)
                outs = []
                for b in range(q_BHNE.shape[0]):
                    sl = lambda t: t[b : b + 1]
                    outs.append(
                        _exp(sl(q_BHNE), sl(k_BHNE), sl(v_BHNE), sl(add_q_BHLE), sl(add_k_BHLE), sl(add_v_BHLE), cfg1)
                    )
                spatial_BHNE = ttnn.concat([o[0] for o in outs], dim=0)
                prompt_BHLE = ttnn.concat([o[1] for o in outs], dim=0)
            else:
                spatial_BHNE, prompt_BHLE, _lse = _exp(
                    q_BHNE, k_BHNE, v_BHNE, add_q_BHLE, add_k_BHLE, add_v_BHLE, exp_cfg
                )
        elif self.parallel_config.sequence_parallel.factor > 1:
            spatial_BHNE, prompt_BHLE, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                add_q_BHLE,
                add_k_BHLE,
                add_v_BHLE,
                persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                    k_BHNE.shape,
                    2,
                    self.parallel_config.sequence_parallel.mesh_axis,
                    dtype=self._sdpa_input_dtype or ttnn.bfloat16,
                ),
                persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                    v_BHNE.shape,
                    2,
                    self.parallel_config.sequence_parallel.mesh_axis,
                    dtype=self._sdpa_input_dtype or ttnn.bfloat16,
                ),
                joint_strategy="rear",
                logical_n=N,
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.sequence_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
                mesh_device=self.mesh_device,
                topology=self.ccl_manager.topology,
                subdevice_id=self.ccl_manager.ccl_sub_device_id,
                ccl_core_grid_offset=(0, self.sdpa_worker_grid[1]),
            )
        else:
            spatial_BHNE, prompt_BHLE = ttnn.transformer.joint_scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                add_q_BHLE,
                add_k_BHLE,
                add_v_BHLE,
                joint_strategy="rear",
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )

        spatial_1BND = ttnn.transformer.concatenate_heads(spatial_BHNE)
        spatial_1BND = ttnn.unsqueeze(spatial_1BND, 0)

        if self.fused_tp:
            attn_flat = flatten_batch(spatial_1BND)
            if spatial_residual is not None:
                out_flat = self.to_out(
                    attn_flat,
                    parallel_config=self.parallel_config,
                    addcmul_a=flatten_batch(spatial_residual),
                    addcmul_b=spatial_gate,
                    addcmul_scalar=1.0,
                )
            else:
                out_flat = self.to_out(attn_flat, parallel_config=self.parallel_config)
            spatial_1BND = unflatten_batch(out_flat, spatial_1BND.shape)
        else:
            if self.parallel_config.tensor_parallel.factor > 1:
                spatial_1BND = self._ag_tp(spatial_1BND)

            spatial_1BND = self.to_out(spatial_1BND)

        prompt_out = None
        if self.context_pre_only is not None and not self.context_pre_only:
            prompt_1BLD = ttnn.transformer.concatenate_heads(prompt_BHLE)
            prompt_1BLD = ttnn.unsqueeze(prompt_1BLD, 0)
            if self.parallel_config.tensor_parallel.factor > 1:
                prompt_1BLD = self._ag_tp(prompt_1BLD)
            prompt_1BLD = self.to_add_out(prompt_1BLD)
            prompt_out = prompt_1BLD

        return spatial_1BND, prompt_out
