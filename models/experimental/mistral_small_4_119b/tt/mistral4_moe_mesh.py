# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral4 **routed** MoE on a mesh — TTNN expert matmuls; optional HF-identical routing on host for PCC.

Checkpoint tensors are converted with PyTorch in ``__init__`` only for ``ttnn.from_torch``.
Hub checkpoints may ship MoE weights as FP8; use :func:`mistral4_mlp_state_dict_bf16_match_hf` so TTNN
loads the same bf16 tensors as ``Mistral4MoE`` after ``load_state_dict`` + ``.to(torch.bfloat16)`` (raw
``.to(bfloat16)`` on FP8 buffers is **not** HF-equivalent).

* **Weights**: ``experts.gate_up_proj`` / ``experts.down_proj`` may be **sharded** on dim ``0``
  (expert index) across the mesh when ``shard_routed_experts`` is enabled (default if
  ``n_routed_experts % num_mesh_devices == 0``). Each device runs only its local experts, then
  :func:`~models.tt_transformers.tt.ccl.tt_all_reduce` sums partial outputs (same contract as
  ``constants.expert_index_ranges_per_mesh_device``). Set ``shard_routed_experts=False`` to keep the
  older **replicated** expert tables (higher DRAM, no CCL combine).
* **Router**: replicated ``gate``; ``ttnn.linear`` → ``ttnn.softmax`` → ``ttnn.topk`` (``k=4``).
* **Routing on device**: ``n_group == 1`` and ``topk_group == 1`` only (hub Small 4 config).
* **Combine**: sum partial routed outputs across devices (reduce_scatter + all_gather on 1×N meshes;
  :func:`~models.tt_transformers.tt.ccl.tt_all_reduce` on wider meshes) when experts are sharded and
  ``num_mesh_devices > 1``.

:class:`TtMistral4SharedExpertsMlpTtnn` implements HF ``Mistral4MoE.shared_experts`` on TTNN for use
alongside the routed skeleton inside :class:`~models.experimental.mistral_small_4_119b.tt.text_backbone.TtMistral4DecoderLayer`
(``use_ttnn_moe=True``).

``route_tokens_to_experts_torch`` matches HF routing for ``n_group=1`` / ``topk_group=1`` (unit tests and
callers that pass host ``topk`` into :meth:`TtMistral4MoeRoutedExpertParallelSkeleton.forward`).
"""

from __future__ import annotations

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_reduce


def _mistral4_moe_ccl_cluster_axis(mesh_device) -> int:
    """
    ``reduce_scatter_minimal_async`` uses ``get_topological_dimension(tensor, cluster_axis)``.

    Fully **replicated** activations on a **1×N** mesh report ring size 1 for ``cluster_axis=0``
    (row mesh size is 1); use ``cluster_axis=1`` so the ring follows the N-wide column axis.
    """
    mesh_shape = list(mesh_device.shape)
    rows, cols = int(mesh_shape[0]), int(mesh_shape[1])
    if cols > 1 and rows == 1:
        return 1
    if rows > 1 and cols == 1:
        return 0
    # 2D mesh: prefer the larger mesh extent (heuristic for replicated tensors).
    return 1 if cols >= rows else 0


def _mistral4_routed_moe_combine_across_mesh(
    mesh_device,
    tt_ccl,
    input_tensor: ttnn.Tensor,
    *,
    cluster_axis: int | None = None,
    dim: int = 3,
) -> ttnn.Tensor:
    """
    Sum identical logical activations ``[1,1,T,H]`` across the mesh (MoE routed partials).

    :func:`~models.tt_transformers.tt.ccl.tt_all_reduce` returns **reduce_scatter only** when
    ``1 in mesh_device.shape`` (e.g. 1×4), which changes tensor volume and breaks a final
    ``reshape`` to ``[B,1,S,H]``. This matches the library's **composite** path: reduce_scatter
    then all_gather, then restore the input rank-4 shape.

    ``cluster_axis`` defaults to :func:`_mistral4_moe_ccl_cluster_axis` so replicated tensors on
    1×N meshes use the column ring (``cluster_axis=1``).
    """
    mesh_shape = list(mesh_device.shape)
    if mesh_shape == [1, 1] or int(mesh_device.get_num_devices()) <= 1:
        return input_tensor

    if cluster_axis is None:
        cluster_axis = _mistral4_moe_ccl_cluster_axis(mesh_device)

    num_reduce_scatter_links = tt_ccl.get_num_links(cluster_axis)
    num_all_gather_links = tt_ccl.get_num_links(cluster_axis)
    topology = ttnn.Topology.Linear
    chunks_per_sync = 10
    num_workers_per_link = 2
    subdevice_id = None

    original_shape = input_tensor.shape

    if 1 in mesh_shape:
        in_t = input_tensor
        if original_shape[0] != 1 or original_shape[1] != 1:
            in_t = ttnn.reshape(
                in_t,
                (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1]),
            )
        input_mem_cfg = in_t.memory_config()
        reduced_tensor = ttnn.experimental.reduce_scatter_minimal_async(
            in_t,
            persistent_output_buffers=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=num_reduce_scatter_links,
            cluster_axis=cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            chunks_per_sync=chunks_per_sync,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=2,
            subdevice_id=subdevice_id,
        )
        in_t.deallocate(True)
        reduced_tensor = ttnn.experimental.all_gather_async(
            reduced_tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=num_all_gather_links,
            cluster_axis=cluster_axis,
            topology=topology,
            memory_config=input_mem_cfg,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=chunks_per_sync,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=2,
            subdevice_id=subdevice_id,
        )
        return ttnn.reshape(reduced_tensor, original_shape)

    return tt_all_reduce(
        input_tensor,
        mesh_device,
        tt_ccl,
        cluster_axis=cluster_axis,
        dim=dim,
        topology=topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        sharded=False,
    )


def route_tokens_to_experts_torch(router_logits: torch.Tensor, text_config) -> tuple[torch.Tensor, torch.Tensor]:
    """HF ``Mistral4MoE.route_tokens_to_experts`` for ``n_group=1`` / ``topk_group=1`` (full HF when those hold)."""
    n_group = int(getattr(text_config, "n_group", None) or 1)
    n_routed_experts = int(text_config.n_routed_experts)
    topk_group = int(getattr(text_config, "topk_group", None) or 1)
    top_k = int(text_config.num_experts_per_tok)
    norm_topk_prob = bool(getattr(text_config, "norm_topk_prob", True))
    routed_scaling_factor = float(getattr(text_config, "routed_scaling_factor", 1.0))

    router_logits = router_logits.softmax(-1)
    group_scores = router_logits.view(-1, n_group, n_routed_experts // n_group).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(-1, n_group, n_routed_experts // n_group).reshape(-1, n_routed_experts)
    scores_for_choice = router_logits.masked_fill(~score_mask.bool(), 0.0)
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = router_logits.gather(1, topk_indices)
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights /= denominator
    topk_weights = topk_weights * routed_scaling_factor
    return topk_indices, topk_weights


def mistral4_mlp_state_dict_bf16_match_hf(text_config, mlp_state_dict: dict) -> dict:
    """
    Return an ``mlp`` state dict (keys like ``gate.weight``, ``experts.gate_up_proj``, …) in bf16
    matching HF ``Mistral4MoE`` after ``load_state_dict(mlp_state_dict)`` and ``.to(torch.bfloat16)``.

    Hub FP8 expert weights must be dequantized the same way as Transformers; do not pass raw FP8
    tensors through ``tensor.to(torch.bfloat16)`` for TTNN uploads.
    """
    try:
        from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE
    except ImportError as exc:  # pragma: no cover
        raise ImportError("mistral4_mlp_state_dict_bf16_match_hf requires transformers with Mistral4MoE.") from exc

    moe = Mistral4MoE(text_config).eval()
    moe.load_state_dict(mlp_state_dict, strict=True)
    moe = moe.to(torch.bfloat16)
    return {k: v.detach().contiguous() for k, v in moe.state_dict().items()}


def _routing_ttnn_simple_group(
    router_logits_tt: ttnn.Tensor,
    *,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
    top_k: int,
    use_fp32_softmax: bool = False,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Same as HF ``route_tokens_to_experts`` when ``n_group==1`` and ``topk_group==1``."""
    if use_fp32_softmax:
        # FP32 softmax then bf16 ``topk`` (``ttnn.topk`` requires bf16/bfloat8) — closer to HF torch
        # on real checkpoints than bf16-only softmax.
        rl32 = ttnn.typecast(router_logits_tt, dtype=ttnn.float32)
        probs32 = ttnn.softmax(rl32, dim=-1)
        ttnn.deallocate(rl32)
        probs = ttnn.typecast(probs32, dtype=ttnn.bfloat16)
        ttnn.deallocate(probs32)
    else:
        probs = ttnn.softmax(router_logits_tt, dim=-1)
    topk_vals, topk_idx = ttnn.topk(probs, top_k, dim=-1, largest=True, sorted=False)
    ttnn.deallocate(probs)
    if norm_topk_prob:
        denom = ttnn.sum(topk_vals, dim=-1, keepdim=True)
        eps = ttnn.full(
            denom.shape,
            1e-20,
            dtype=denom.dtype,
            layout=denom.layout,
            device=denom.device(),
        )
        denom = ttnn.add(denom, eps)
        topk_w = ttnn.div(topk_vals, denom)
        ttnn.deallocate(eps)
        ttnn.deallocate(denom)
        ttnn.deallocate(topk_vals)
    else:
        topk_w = topk_vals
    if routed_scaling_factor != 1.0:
        scale = ttnn.full(
            topk_w.shape,
            float(routed_scaling_factor),
            dtype=topk_w.dtype,
            layout=topk_w.layout,
            device=topk_w.device(),
        )
        topk_w = ttnn.mul(topk_w, scale)
        ttnn.deallocate(scale)
    return topk_idx, topk_w


class TtMistral4MoeRoutedExpertParallelSkeleton(LightweightModule):
    """
    Routed MoE: TTNN router (optional) and TTNN experts.

    Expert ``gate_up`` / ``down`` tables are **sharded** on expert dim ``0`` across the mesh by
    default when ``n_routed_experts`` divides ``get_num_devices()``; otherwise they stay **replicated**.

    :meth:`forward` may receive ``topk_indices_torch`` / ``topk_weights_torch`` to skip device routing
    (HF-identical ``topk`` computed on host) while keeping expert matmuls on device.

    Args:
        tt_ccl: ``TT_CCL`` instance required when ``get_num_devices() > 1``.
        shard_routed_experts: ``True`` / ``False`` to force sharding, or ``None`` to auto-select when
            ``n_routed_experts % num_mesh_devices == 0`` (shard) else replicate.
        use_fp32_router_softmax: if ``True``, softmax runs in FP32 then probabilities are cast to bf16
            before ``ttnn.topk`` (closer to HF torch routing on real checkpoints). Default ``False``
            keeps the strict PCC path in ``tests/test_mistral4_moe_mesh_routed_pcc.py`` unchanged.
    """

    def __init__(
        self,
        mesh_device,
        text_config,
        mlp_state_dict: dict,
        *,
        tt_ccl,
        weight_dtype=ttnn.bfloat16,
        use_fp32_router_softmax: bool = False,
        shard_routed_experts: bool | None = None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.text_config = text_config
        self.tt_ccl = tt_ccl
        self._use_fp32_router_softmax = bool(use_fp32_router_softmax)
        self.num_mesh_devices = int(mesh_device.get_num_devices())

        n_group = int(getattr(text_config, "n_group", None) or 1)
        topk_group = int(getattr(text_config, "topk_group", None) or 1)
        if n_group != 1 or topk_group != 1:
            raise NotImplementedError(
                f"TTNN routing supports only n_group=1 and topk_group=1; got {n_group=} {topk_group=}."
            )

        if self.num_mesh_devices > 1 and tt_ccl is None:
            raise ValueError("tt_ccl is required when mesh has more than one device.")

        self._compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self._inter = int(text_config.moe_intermediate_size)
        self._hidden = int(text_config.hidden_size)
        self._n_routed = int(text_config.n_routed_experts)
        self._top_k = int(text_config.num_experts_per_tok)
        self._norm_topk = bool(getattr(text_config, "norm_topk_prob", True))
        self._routed_scale = float(getattr(text_config, "routed_scaling_factor", 1.0))

        if shard_routed_experts is None:
            self._shard_experts = self._n_routed % self.num_mesh_devices == 0
        else:
            self._shard_experts = bool(shard_routed_experts)
        if self._shard_experts and (self._n_routed % self.num_mesh_devices) != 0:
            raise ValueError(
                f"shard_routed_experts requires n_routed_experts divisible by num mesh devices; "
                f"got {self._n_routed=} and {self.num_mesh_devices=}."
            )
        self._local_num = self._n_routed // self.num_mesh_devices if self._shard_experts else self._n_routed

        if "gate.weight" not in mlp_state_dict:
            raise KeyError("mlp_state_dict must contain 'gate.weight'")
        if "experts.gate_up_proj" not in mlp_state_dict or "experts.down_proj" not in mlp_state_dict:
            raise KeyError("mlp_state_dict must contain experts.gate_up_proj and experts.down_proj")

        gate_w = mlp_state_dict["gate.weight"].detach().to(torch.bfloat16).contiguous()
        gate_w_linear = gate_w.t().contiguous()
        self.gate_weight_tt = ttnn.from_torch(
            gate_w_linear,
            device=mesh_device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        gu = mlp_state_dict["experts.gate_up_proj"].detach().to(torch.bfloat16).contiguous()
        dn = mlp_state_dict["experts.down_proj"].detach().to(torch.bfloat16).contiguous()
        expert_mapper = (
            ttnn.ShardTensorToMesh(mesh_device, dim=0)
            if self._shard_experts
            else ttnn.ReplicateTensorToMesh(mesh_device)
        )
        self._gu_weights = ttnn.from_torch(
            gu,
            device=mesh_device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=expert_mapper,
        )
        self._dn_weights = ttnn.from_torch(
            dn,
            device=mesh_device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=expert_mapper,
        )

        self._expert_id_eq_tt: list[ttnn.Tensor] = []
        self._expert_base_shard: ttnn.Tensor | None = None
        self._local_slot_uint_tt: list[ttnn.Tensor] = []

        if self._shard_experts:
            starts = torch.arange(0, self._n_routed, self._local_num, dtype=torch.int32)
            self._expert_base_shard = ttnn.from_torch(
                starts.reshape(self.num_mesh_devices, 1, 1, 1),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            )
            for le in range(self._local_num):
                tid = torch.tensor([[[[le]]]], dtype=torch.int32)
                self._local_slot_uint_tt.append(
                    ttnn.from_torch(
                        tid,
                        device=mesh_device,
                        dtype=ttnn.uint32,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    )
                )
        else:
            for e in range(self._n_routed):
                tid = torch.tensor([[[[e]]]], dtype=torch.int32)
                self._expert_id_eq_tt.append(
                    ttnn.from_torch(
                        tid,
                        device=mesh_device,
                        dtype=ttnn.uint32,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    )
                )

    def _expert_routing_weight(self, topk_idx: ttnn.Tensor, topk_w: ttnn.Tensor, expert_id: int) -> ttnn.Tensor:
        t_len = int(topk_idx.shape[2])
        k = int(self._top_k)
        acc = ttnn.zeros((1, 1, t_len, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        id_tt = self._expert_id_eq_tt[expert_id]
        idx_cast = ttnn.typecast(topk_idx, dtype=ttnn.uint32)
        for slot in range(k):
            idx_k = ttnn.slice(idx_cast, [0, 0, 0, slot], [1, 1, t_len, slot + 1], [1, 1, 1, 1])
            w_k = ttnn.slice(topk_w, [0, 0, 0, slot], [1, 1, t_len, slot + 1], [1, 1, 1, 1])
            match = ttnn.eq(idx_k, id_tt)
            match_bf16 = ttnn.typecast(match, dtype=ttnn.bfloat16)
            contrib = ttnn.mul(w_k, match_bf16)
            acc = ttnn.add(acc, contrib)
            ttnn.deallocate(idx_k)
            ttnn.deallocate(w_k)
            ttnn.deallocate(match)
            ttnn.deallocate(match_bf16)
            ttnn.deallocate(contrib)
        ttnn.deallocate(idx_cast)
        return acc

    def _expert_routing_weight_id_tensor(
        self, topk_idx: ttnn.Tensor, topk_w: ttnn.Tensor, id_tt: ttnn.Tensor
    ) -> ttnn.Tensor:
        """Same as :meth:`_expert_routing_weight` but ``id_tt`` is the global expert id to match (``uint32``)."""
        t_len = int(topk_idx.shape[2])
        k = int(self._top_k)
        acc = ttnn.zeros((1, 1, t_len, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        idx_cast = ttnn.typecast(topk_idx, dtype=ttnn.uint32)
        for slot in range(k):
            idx_k = ttnn.slice(idx_cast, [0, 0, 0, slot], [1, 1, t_len, slot + 1], [1, 1, 1, 1])
            w_k = ttnn.slice(topk_w, [0, 0, 0, slot], [1, 1, t_len, slot + 1], [1, 1, 1, 1])
            match = ttnn.eq(idx_k, id_tt)
            match_bf16 = ttnn.typecast(match, dtype=ttnn.bfloat16)
            contrib = ttnn.mul(w_k, match_bf16)
            acc = ttnn.add(acc, contrib)
            ttnn.deallocate(idx_k)
            ttnn.deallocate(w_k)
            ttnn.deallocate(match)
            ttnn.deallocate(match_bf16)
            ttnn.deallocate(contrib)
        ttnn.deallocate(idx_cast)
        return acc

    def _expert_forward_one(self, x_flat_tt: ttnn.Tensor, *, row_index: int) -> ttnn.Tensor:
        """``row_index`` is the row into ``_gu_weights`` / ``_dn_weights`` (global expert id if replicated, local if sharded)."""
        two_i = 2 * self._inter
        h = self._hidden
        t_len = int(x_flat_tt.shape[2])

        gu_row = ttnn.slice(
            self._gu_weights,
            [row_index, 0, 0],
            [row_index + 1, two_i, h],
            [1, 1, 1],
        )
        gu_2ih = ttnn.squeeze(gu_row, 0)
        ttnn.deallocate(gu_row)
        gu_w = ttnn.transpose(gu_2ih, -2, -1)
        ttnn.deallocate(gu_2ih)

        gu_out = ttnn.linear(
            x_flat_tt,
            gu_w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        ttnn.deallocate(gu_w)

        gate_part = ttnn.slice(gu_out, [0, 0, 0, 0], [1, 1, t_len, self._inter], [1, 1, 1, 1])
        up_part = ttnn.slice(gu_out, [0, 0, 0, self._inter], [1, 1, t_len, two_i], [1, 1, 1, 1])
        ttnn.deallocate(gu_out)
        silu_g = ttnn.silu(gate_part)
        ttnn.deallocate(gate_part)
        hid = ttnn.mul(silu_g, up_part)
        ttnn.deallocate(silu_g)
        ttnn.deallocate(up_part)

        dn_row = ttnn.slice(
            self._dn_weights,
            [row_index, 0, 0],
            [row_index + 1, h, self._inter],
            [1, 1, 1],
        )
        dn_h_i = ttnn.squeeze(dn_row, 0)
        ttnn.deallocate(dn_row)
        dn_w = ttnn.transpose(dn_h_i, -2, -1)
        ttnn.deallocate(dn_h_i)

        out = ttnn.linear(
            hid,
            dn_w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        ttnn.deallocate(hid)
        ttnn.deallocate(dn_w)
        return out

    def forward(
        self,
        hidden_11SH: ttnn.Tensor,
        *,
        topk_indices_torch: torch.Tensor | None = None,
        topk_weights_torch: torch.Tensor | None = None,
    ) -> ttnn.Tensor:
        if hidden_11SH.shape[0] != 1 or hidden_11SH.shape[1] != 1:
            raise ValueError(f"expected leading dims [1,1,...], got {tuple(hidden_11SH.shape)}")

        b, _, s, h = hidden_11SH.shape
        t_len = int(b * s)
        # Own a flat copy: ``reshape`` may alias ``hidden_11SH``; we ``deallocate`` the flat buffer at
        # the end of this forward and must not invalidate the caller's tensor.
        x_flat_tt = ttnn.clone(ttnn.reshape(hidden_11SH, (1, 1, t_len, h)))

        if topk_indices_torch is not None or topk_weights_torch is not None:
            if topk_indices_torch is None or topk_weights_torch is None:
                raise ValueError("topk_indices_torch and topk_weights_torch must both be set or both omitted.")
            if topk_indices_torch.shape != (t_len, self._top_k) or topk_weights_torch.shape != (t_len, self._top_k):
                raise ValueError(
                    f"expected host topk shapes ({t_len}, {self._top_k}), got "
                    f"{tuple(topk_indices_torch.shape)} and {tuple(topk_weights_torch.shape)}"
                )
            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
            topk_idx = ttnn.from_torch(
                topk_indices_torch.to(torch.int32).reshape(1, 1, t_len, self._top_k),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            topk_w = ttnn.from_torch(
                topk_weights_torch.reshape(1, 1, t_len, self._top_k).to(torch.bfloat16),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
        else:
            router_logits_tt = ttnn.linear(
                x_flat_tt,
                self.gate_weight_tt,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self._compute_kernel_config,
            )
            topk_idx, topk_w = _routing_ttnn_simple_group(
                router_logits_tt,
                norm_topk_prob=self._norm_topk,
                routed_scaling_factor=self._routed_scale,
                top_k=self._top_k,
                use_fp32_softmax=self._use_fp32_router_softmax,
            )
            ttnn.deallocate(router_logits_tt)

        acc_local = ttnn.zeros((1, 1, t_len, h), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)

        if self._shard_experts:
            assert self._expert_base_shard is not None
            for li in range(self._local_num):
                global_id_tt = ttnn.add(self._expert_base_shard, self._local_slot_uint_tt[li])
                rw = self._expert_routing_weight_id_tensor(topk_idx, topk_w, global_id_tt)
                ttnn.deallocate(global_id_tt)
                y_e = self._expert_forward_one(x_flat_tt, row_index=li)
                y_w = ttnn.mul(y_e, rw)
                ttnn.deallocate(y_e)
                ttnn.deallocate(rw)
                acc_local = ttnn.add(acc_local, y_w)
                ttnn.deallocate(y_w)
        else:
            for e in range(self._n_routed):
                rw = self._expert_routing_weight(topk_idx, topk_w, e)
                y_e = self._expert_forward_one(x_flat_tt, row_index=e)
                y_w = ttnn.mul(y_e, rw)
                ttnn.deallocate(y_e)
                ttnn.deallocate(rw)
                acc_local = ttnn.add(acc_local, y_w)
                ttnn.deallocate(y_w)

        ttnn.deallocate(topk_idx)
        ttnn.deallocate(topk_w)
        ttnn.deallocate(x_flat_tt)

        out_tt = acc_local
        if self._shard_experts and self.num_mesh_devices > 1:
            out_tt = _mistral4_routed_moe_combine_across_mesh(
                self.mesh_device,
                self.tt_ccl,
                acc_local,
                dim=3,
            )
        return ttnn.reshape(out_tt, (b, 1, s, h))


class TtMistral4SharedExpertsMlpTtnn(LightweightModule):
    """
    HF ``Mistral4MoE.shared_experts`` (SiLU gate × up, ``down``) on TTNN.

    Checkpoint tensors are read with PyTorch in ``__init__`` only for ``ttnn.from_torch``.
    ``forward`` uses only TTNN ops (same contract as :class:`TtMistral4MoeRoutedExpertParallelSkeleton`).
    """

    def __init__(
        self,
        mesh_device,
        text_config,
        mlp_state_dict: dict,
        *,
        weight_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self._hidden = int(text_config.hidden_size)
        self._compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        for k in (
            "shared_experts.gate_proj.weight",
            "shared_experts.up_proj.weight",
            "shared_experts.down_proj.weight",
        ):
            if k not in mlp_state_dict:
                raise KeyError(f"mlp_state_dict must contain {k!r}")

        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        mem = ttnn.DRAM_MEMORY_CONFIG
        layout = ttnn.TILE_LAYOUT

        def _lin_w(pytorch_w: torch.Tensor) -> ttnn.Tensor:
            # ``ttnn.linear(x, W)`` with x [*, H] uses W [H, out] (see routed gate upload).
            w = pytorch_w.detach().to(torch.bfloat16).contiguous()
            return ttnn.from_torch(
                w.t().contiguous(),
                device=mesh_device,
                dtype=weight_dtype,
                layout=layout,
                memory_config=mem,
                mesh_mapper=mapper,
            )

        self._gate_w = _lin_w(mlp_state_dict["shared_experts.gate_proj.weight"])
        self._up_w = _lin_w(mlp_state_dict["shared_experts.up_proj.weight"])
        self._down_w = _lin_w(mlp_state_dict["shared_experts.down_proj.weight"])

    def forward(self, hidden_11SH: ttnn.Tensor) -> ttnn.Tensor:
        if hidden_11SH.shape[0] != 1 or hidden_11SH.shape[1] != 1:
            raise ValueError(f"expected leading dims [1,1,...], got {tuple(hidden_11SH.shape)}")
        b, _, s, h = hidden_11SH.shape
        if h != self._hidden:
            raise ValueError(f"hidden dim {h} != config hidden_size {self._hidden}")
        t_len = int(b * s)
        x = ttnn.reshape(hidden_11SH, (1, 1, t_len, h))
        gate = ttnn.linear(
            x,
            self._gate_w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        up = ttnn.linear(
            x,
            self._up_w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        silu_g = ttnn.silu(gate)
        ttnn.deallocate(gate)
        hid = ttnn.mul(silu_g, up)
        ttnn.deallocate(silu_g)
        ttnn.deallocate(up)
        out = ttnn.linear(
            hid,
            self._down_w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        ttnn.deallocate(hid)
        return ttnn.reshape(out, (b, 1, s, h))
