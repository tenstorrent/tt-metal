# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode-frame routed experts on the decode MoE kernel (all_to_all_dispatch_metadata -> moe_compute -> fused
score-weighted reduce -> reduce-scatter), models/common/moe/TTMoEDecode with the gemma_4_26b config.

Layout: the frame's rows (all lanes, gathered on every chip) are split over the mesh rows (the dispatch axis,
cluster_axis 0); every column holds 32 experts (4 per chip on the Galaxy, 16 on a (2,4) block), the same placement
the sparse path uses. Each column computes its experts' contributions for every token in its row-chip's slice, the
reduce-scatter over the columns sums them, and two all-gathers restore the replicated [frame, hidden] tensor.
Expert weights are the kernel's bfloat4_b DRAM-sharded layout (a second copy next to the sparse path's).

Enabled with GEMMA4_MOE_DECODE_KERNEL=1 (see experts/__init__.py); GEMMA4_MOE_DECODE_FRAME sets the frame rows.
GEMMA4_MOE_DECODE_TILE_TAIL=1 keeps the kernel output in TILE layout through the two all-gathers (drops the untilize /
re-tilize pair of the frame tail); the packed routing of GEMMA4_MOE_FUSED=1 builds the dispatch inputs with
``custom_ops/moe_frame_prep`` (one program instead of the pad / layout / partition / typecast / copy glue).
"""
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.experts.decode import PackedRouting

try:  # main moved the shared decode MoE library to models/common/moe (TTTv2 left tt-metal)
    from models.common.moe.tt_moe_decode import TTMoEDecode
    from models.common.moe.tt_moe_decode_config import TTMoEDecodeConfig

    _MOE_LIB_DIR = Path(__file__).resolve().parents[4] / "common" / "moe"
except ImportError:  # builds before that move keep the old package
    from models.common.modules.moe.tt_moe_decode import TTMoEDecode
    from models.common.modules.moe.tt_moe_decode_config import TTMoEDecodeConfig

    _MOE_LIB_DIR = Path(__file__).resolve().parents[4] / "common" / "modules" / "moe"

CONFIG_YAML = _MOE_LIB_DIR / "configs" / "gemma_4_26b.yaml"


def moe_decode_kernel_enabled():
    return os.environ.get("GEMMA4_MOE_DECODE_KERNEL", "0") == "1"


def moe_compute_custom_enabled() -> bool:
    """GEMMA4_MOE_COMPUTE_CUSTOM_OP=1: custom_ops/moe_compute (per-expert resident weights, the standalone fabric
    combine, GEMMA4_MOE_COMPUTE_SKIP_ZERO_SCORES) in place of the shared op's moe_compute - on decode frames only."""
    return os.environ.get("GEMMA4_MOE_COMPUTE_CUSTOM_OP", "0") == "1"


def kernel_decode_frame(mesh_config, rows: int) -> bool:
    """Whether a decode MoE kernel call carries decode frames (the traced step) rather than a prefill's rows.

    Under lanes the single-user owner prefill is marked by ``mesh_config._g4_single_lane_prefill_owner``: its 128
    real rows gather to the same R as the 128-user decode frame, so R cannot tell them apart. On the tp-only shapes
    decode rows are at most one row tile (32).
    """
    if mesh_config is not None and getattr(mesh_config, "lane_sharded", False):
        return getattr(mesh_config, "_g4_single_lane_prefill_owner", None) is None
    return int(rows) <= 32


def moe_compute_impl(decode_frame: bool, custom_enabled: bool) -> str:
    """Which moe_compute runs a kernel call: the custom op on decode frames when enabled, the shared op otherwise.

    Galaxy window-2 arms (2026-10-24, eager 128-token lane prefill warm median / traced 128-user step): base
    371 ms / 46.2, fused_all 532 / 32.9, fused_all without the custom op 371 / 40.7, custom op without the skip 529
    / 45.6, fused_all without the TILE tail 538 / 34.4, GLUE only 373 / 44.4. The custom op (with the skip and the
    TILE tail) is the decode win and, applied to the 128-row owner prefill, the +160 ms prefill regression: it is
    scoped to decode frames, the prefill rows keep the shared op.
    """
    return "custom" if (decode_frame and custom_enabled) else "shared"


def tile_tail_enabled(decode_frame: bool, flag: bool, out_is_tile: bool, batch_per_device: int) -> bool:
    """GEMMA4_MOE_DECODE_TILE_TAIL=1 (TILE-layout gathers in the kernel tail) applies to decode frames only: on the
    prefill rows the ROW_MAJOR tail stays (the window-2 no_tail arm: 1.5 ms/step of the decode win, prefill-neutral
    only together with the shared op)."""
    return bool(decode_frame and flag and out_is_tile and int(batch_per_device) % 32 == 0)


# One TTMoEDecode per mesh, shared by every layer: the module keeps per-instance L1 scratch and semaphores (a
# per-layer instance x 30 left ~200 KB of resident L1 and its static circular buffers then clashed), and its weights
# are layered ([L, E, ...], forward(layer_id)). Layers register their HF tensors as they are built; the module is
# built when the last one arrives. Keyed by the mesh's own id (``MeshDevice.id()``, a process-wide counter that is
# never reused), never ``id(mesh_device)``: a mesh closed and reopened between two calls (every pytest case) can
# reuse the Python address, and the new mesh's layers would then find a module built on the closed one.
_SHARED = {}


class Gemma4MoEDecode:
    def __init__(self, mesh_device, config, gate_up_proj, down_proj, layer_idx=0, num_layers=1, frame_rows=None):
        """gate_up_proj [E, 2I, H] (gate rows first), down_proj [E, H, I]: HF tensors of this layer."""
        self.mesh_device = mesh_device
        rows, cols = mesh_device.shape
        self.rows, self.cols = rows, cols
        self.hidden = config.hidden_size
        self.top_k = config.top_k
        self.num_experts = config.num_experts
        self.inter = config.moe_intermediate_size
        self.layer_idx = int(layer_idx)
        self.num_layers = int(num_layers)
        self.frame_rows = int(frame_rows or os.environ.get("GEMMA4_MOE_DECODE_FRAME", str(rows * ttnn.TILE_SIZE)))
        assert self.frame_rows % rows == 0, f"frame {self.frame_rows} must split over {rows} mesh rows"
        self.batch_per_device = self.frame_rows // rows
        key = int(mesh_device.id())
        reg = _SHARED.setdefault(key, {"layers": {}, "module": None, "config": None})
        reg["layers"][self.layer_idx] = (gate_up_proj, down_proj)
        self._reg = reg
        if reg["module"] is None and len(reg["layers"]) >= self.num_layers:
            self._build(config)

    def _build(self, config):
        import yaml

        reg = self._reg
        data = yaml.safe_load(CONFIG_YAML.read_text())
        data["mesh_shape"] = [self.rows, self.cols]
        data["batch_per_device"] = self.batch_per_device
        data["hidden_size"] = self.hidden
        data["select_experts_k"] = self.top_k
        data["num_routed_experts"] = self.num_experts
        data["compute"]["intermediate_size"] = self.inter
        # The lane axis has 4 chips, so the module takes the generic ``ttnn.reduce_scatter`` path (the fused DeepSeek
        # reduce-scatter needs 8). That path rejects the fast reduce's default ND-sharded L1 output on current main
        # (sharding_addrgen_helper: only block/height/width sharding), so the fast reduce writes interleaved DRAM.
        data.setdefault("reduce", {})["output_memory_config"] = {
            "buffer_type": 0,  # DRAM
            "memory_layout": 0,  # INTERLEAVED
            "created_with_nd_shard_spec": False,
            "per_core_allocation": False,
        }
        tt_config = TTMoEDecodeConfig.from_yaml(yaml.safe_dump(data, sort_keys=False), topology=self._topology())
        I = self.inter
        layers = [reg["layers"][l] for l in range(self.num_layers)]
        # kernel layout: w0 = gate [L, E, H, N], w1 = up [L, E, H, N], w2 = down [L, E, N, H]
        gate = torch.stack([gu[:, :I, :].to(torch.bfloat16).transpose(1, 2).contiguous() for gu, _ in layers], dim=0)
        up = torch.stack([gu[:, I:, :].to(torch.bfloat16).transpose(1, 2).contiguous() for gu, _ in layers], dim=0)
        down = torch.stack([d.to(torch.bfloat16).transpose(1, 2).contiguous() for _, d in layers], dim=0)
        _MoECls = TTMoEDecode
        if moe_compute_custom_enabled():
            # custom_ops/moe_compute: per-expert resident weights + the standalone fabric combine (README there);
            # decode frames only (moe_compute_impl): the prefill rows run the shared op on the same instance. The
            # port refuses through its unsupported_reason (Blackhole-only, the standalone staging capacity: a
            # (2,4) block's 16 experts x 128 frame rows); then the shared op serves every call, with one warning.
            from models.demos.gemma4.custom_ops.moe_compute.moe_decode_custom import TTMoEDecodeCustom, custom_reason

            reason = custom_reason(self.mesh_device, tt_config, self.num_experts)
            if reason is None:
                _MoECls = TTMoEDecodeCustom
            else:
                logger.warning(f"GEMMA4_MOE_COMPUTE_CUSTOM_OP=1 ignored, the shared moe_compute serves: {reason}")
        reg["module"] = _MoECls(self.mesh_device, tt_config, gate, up, down)
        # the shared op's pipeline on the same instance and weights (TTMoEDecodeCustom only swaps the compute call)
        reg["shared_forward"] = TTMoEDecode.forward
        reg["config"] = tt_config
        reg["layers"].clear()
        logger.info(
            f"Gemma4MoEDecode: shared module for {self.num_layers} layers on mesh {self.rows}x{self.cols}, frame "
            f"{self.frame_rows} rows ({self.batch_per_device}/chip), post_combine_tilize="
            f"{getattr(tt_config, 'use_post_combine_tilize', None)}, topology={tt_config.topology}"
        )

    @staticmethod
    def _topology():
        """Dispatch / combine topology from the mesh's 1-D fabric: Ring on FABRIC_1D_RING (the Galaxy's 8-row dispatch
        axis), Linear on a FABRIC_1D line (the (2,4) block, whose p150_x8 MESH descriptor has no wraparound: a ring
        fabric cannot even open there). GEMMA4_MOE_DECODE_TOPOLOGY=ring|linear overrides the detection."""
        forced = os.environ.get("GEMMA4_MOE_DECODE_TOPOLOGY", "").strip().lower()
        if forced in ("ring", "linear"):
            return ttnn.Topology.Ring if forced == "ring" else ttnn.Topology.Linear
        try:
            fabric = ttnn.get_fabric_config()
        except Exception:  # noqa: BLE001 - no query on this build: the served default
            return ttnn.Topology.Ring
        name = getattr(fabric, "name", None) or str(fabric).rsplit(".", 1)[-1]
        return ttnn.Topology.Linear if name == "FABRIC_1D" else ttnn.Topology.Ring

    @property
    def module(self):
        assert self._reg["module"] is not None, "decode MoE module not built: not every layer registered"
        return self._reg["module"]

    def _pad_index_rows(self, pad):
        """[1, 1, pad, top_k] uint16 ROW_MAJOR rows of 0..top_k-1, replicated: distinct expert ids for the
        weight-0 frame padding of the sparse routing (cached per pad size)."""
        cache = self.__dict__.setdefault("_pad_index_rows_cache", {})
        if pad not in cache:
            rows = torch.arange(self.top_k, dtype=torch.int32).repeat(pad, 1).reshape(1, 1, pad, self.top_k)
            cache[pad] = ttnn.from_torch(
                rows,
                device=self.mesh_device,
                dtype=ttnn.uint16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        return cache[pad]

    def _rowbase(self):
        """Per-chip first frame row of this mesh row (custom_ops/moe_frame_prep), built once."""
        if getattr(self, "_rowbase_t", None) is None:
            from models.demos.gemma4.custom_ops.moe_frame_prep import MoeFramePrep

            self._rowbase_t = MoeFramePrep.make_rowbase(self.mesh_device, self.batch_per_device)
        return self._rowbase_t

    def _forward_packed(self, hidden, routing):
        """``PackedRouting`` rows (GEMMA4_MOE_FUSED=1): ``hidden`` [1,1,R,H] TILE and ``packed`` [1,1,R,2k] ROW_MAJOR
        (R a multiple of 32, every lane's rows) -> the kernel's x / ids / scores in their dispatch memory configs by
        one data-movement program (the frame padding rows are synthesized there), then the kernel."""
        from models.demos.gemma4.custom_ops.moe_frame_prep import MoeFramePrep

        R = int(hidden.shape[2])
        assert R <= self.frame_rows and R % ttnn.TILE_SIZE == 0, f"{R} rows (frame {self.frame_rows})"
        assert self.batch_per_device % ttnn.TILE_SIZE == 0, "the frame-prep op reads whole row tiles per device"
        cfg = self.module.config
        x, ids, scores = MoeFramePrep.op(
            hidden,
            routing.packed,
            self._rowbase(),
            routing.k,
            x_memory_config=cfg.dispatch_input_memory_config,
            scores_memory_config=cfg.dispatch_input_expert_scores_memory_config,
            B=self.batch_per_device,
        )
        out = self._run_module(x, scores, ids, decode=True)  # packed routing only feeds decode frames
        ttnn.deallocate(x)
        ttnn.deallocate(scores)
        ttnn.deallocate(ids)
        return self._tail(out, R, decode=True)

    def _run_module(self, x, scores, indices, decode: bool):
        """The kernel's compute: ``moe_compute_impl`` picks the custom op on decode frames, the shared op's forward on
        prefill rows (the custom module keeps the parent's weights, so the shared pipeline runs on the same instance).
        """
        module = self.module
        if moe_compute_impl(decode, getattr(module, "_custom_op", None) is not None) == "shared":
            shared = self._reg.get("shared_forward")
            if shared is not None and getattr(module, "_custom_op", None) is not None:
                return shared(module, x, scores, indices, layer_id=self.layer_idx)
        return module.forward(x, scores, indices, layer_id=self.layer_idx)

    def _tail(self, out, R, decode: bool = True):
        """The kernel's per-chip [1,1,B,H/cols] -> the replicated [1,1,R,H] frame rows: all-gather over the columns
        (hidden chunks) and the rows (token blocks). GEMMA4_MOE_DECODE_TILE_TAIL=1 gathers in TILE layout (B is a
        whole row tile) on decode frames: no untilize before / tilize after the gathers."""
        B = self.batch_per_device
        tile_tail = tile_tail_enabled(
            decode, os.environ.get("GEMMA4_MOE_DECODE_TILE_TAIL", "0") == "1", out.layout == ttnn.TILE_LAYOUT, B
        )
        if not tile_tail and out.layout != ttnn.ROW_MAJOR_LAYOUT:
            out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)  # strips any tile padding of a B < 32 slice
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(out, (1, 1, B, self.hidden // self.cols))
        if self.cols > 1:
            full = ttnn.all_gather(out, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)  # [1,1,B,H]
            ttnn.deallocate(out)
        else:
            full = out
        if self.rows > 1:
            frame = ttnn.all_gather(full, dim=2, cluster_axis=0, topology=ttnn.Topology.Linear)  # [1,1,frame,H]
            ttnn.deallocate(full)
        else:
            frame = full
        if not tile_tail:
            frame = ttnn.to_layout(frame, ttnn.TILE_LAYOUT)
        if R < self.frame_rows:
            frame = ttnn.slice(frame, [0, 0, 0, 0], [1, 1, R, self.hidden])
        return frame

    def forward(self, hidden, routing, decode: bool = True):
        """hidden [1,1,R,H] replicated on every chip, R <= frame_rows (padded here); routing is either the dense
        [1,1,R,E] matrix or the router's sparse ``(values [1,1,R,K] bf16, indices [1,1,R,K] uint16)``, which skip
        the dense pad / layout / per-row top-k here. Returns [1,1,R,H] replicated: the top-k weighted expert sum.
        ``decode`` = the rows are decode frames (the traced step): the custom moe_compute op and the TILE tail apply;
        a prefill's rows (the single-user owner prefill) keep the shared op and the ROW_MAJOR tail."""
        if isinstance(routing, PackedRouting):
            return self._forward_packed(hidden, routing)
        R = int(hidden.shape[2])
        assert R <= self.frame_rows, f"{R} rows > frame {self.frame_rows}"
        sparse = isinstance(routing, tuple)
        values, indices = routing if sparse else (routing, None)
        pad = self.frame_rows - R
        if pad > 0:
            hidden = ttnn.pad(hidden, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0)
            values = ttnn.pad(values, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0)
        B = self.batch_per_device
        # Row-major before the per-row split: a slice of B < 32 rows is not tile-aligned, reshapes of uint16 tensors
        # are only views in row-major, and the gathers below must not carry tile padding between chips.
        hidden_rm = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.mesh_partition(hidden_rm, dim=2, cluster_axis=0)  # [1,1,B,H] per row-chip
        ttnn.deallocate(hidden_rm)
        if sparse:
            dbg = os.environ.get("GEMMA4_SPARSE_DEBUG") == "1"
            if pad > 0:
                # Pad rows carry weight 0 but must still name DISTINCT experts per k: the dispatch kernel
                # keys its per-token slots by expert id, and a row of eight identical ids hangs the mesh
                # (the dense path's topk over an all-zero row yields ids 0..7 for the same reason).
                indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
                indices = ttnn.concat([indices, self._pad_index_rows(pad)], dim=2)

            def _ck(tag, t):
                if dbg:
                    ttnn.synchronize_device(self.mesh_device)
                    logger.info(
                        f"[sparse-dbg] {tag}: shape={t.shape} dtype={t.dtype} layout={t.layout} mem={t.memory_config()}"
                    )

            _ck("values(in)", values)
            _ck("indices(in)", indices)
            values_rm = ttnn.to_layout(values, ttnn.ROW_MAJOR_LAYOUT)
            _ck("values_rm", values_rm)
            indices_rm = (
                indices if indices.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
            )
            _ck("indices_rm", indices_rm)
            scores = ttnn.reshape(ttnn.mesh_partition(values_rm, dim=2, cluster_axis=0), (B, 1, 1, self.top_k))
            _ck("scores", scores)
            indices = ttnn.reshape(ttnn.mesh_partition(indices_rm, dim=2, cluster_axis=0), (B, 1, 1, self.top_k))
            _ck("indices", indices)
            if dbg:
                logger.info(
                    f"[sparse-dbg] indices host row0: {ttnn.to_torch(ttnn.get_device_tensors(indices)[0])[0, 0, 0].tolist()}"
                )
                logger.info(
                    f"[sparse-dbg] scores host row0: {ttnn.to_torch(ttnn.get_device_tensors(scores)[0])[0, 0, 0].tolist()}"
                )
            ttnn.deallocate(values_rm)
            ttnn.deallocate(indices_rm)
        else:
            routing_rm = ttnn.to_layout(values, ttnn.ROW_MAJOR_LAYOUT)
            r = ttnn.mesh_partition(routing_rm, dim=2, cluster_axis=0)  # [1,1,B,E]
            ttnn.deallocate(routing_rm)
            r = ttnn.to_layout(r, ttnn.TILE_LAYOUT)
            scores, indices = ttnn.topk(r, k=self.top_k, dim=-1)  # [1,1,B,K] bf16 / uint16 (TILE)
            ttnn.deallocate(r)
            scores = ttnn.reshape(ttnn.to_layout(scores, ttnn.ROW_MAJOR_LAYOUT), (B, 1, 1, self.top_k))
            indices = ttnn.reshape(ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT), (B, 1, 1, self.top_k))
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        x = ttnn.reshape(x, (B, 1, 1, self.hidden))
        if os.environ.get("GEMMA4_MOE_DECODE_L1_DUMP") and not getattr(self, "_dumped", False):
            # which L1 buffers are resident when the module's static circular buffers are laid out
            self._dumped = True
            prefix = os.environ["GEMMA4_MOE_DECODE_L1_DUMP"].rstrip("/") + "/moe_decode_pre_"
            os.makedirs(os.path.dirname(prefix), exist_ok=True)
            ttnn.dump_device_memory_state(self.mesh_device, prefix)
            logger.info(f"Gemma4MoEDecode: device memory state dumped to {prefix}*")
        if sparse and os.environ.get("GEMMA4_SPARSE_DEBUG") == "1":
            logger.info("[sparse-dbg] entering kernel forward")
        out = self._run_module(x, scores, indices, decode=decode)  # [1,1,B,H/cols] per chip
        if sparse and os.environ.get("GEMMA4_SPARSE_DEBUG") == "1":
            ttnn.synchronize_device(self.mesh_device)
            logger.info("[sparse-dbg] kernel forward done")
        ttnn.deallocate(x)
        ttnn.deallocate(scores)
        ttnn.deallocate(indices)
        return self._tail(out, R, decode=decode)
