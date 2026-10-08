# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Flat streamed routed experts: TtRoutedExpert's interface on the ``flat_routed_expert`` op (Blackhole).

One program per chip runs every local expert's SwiGLU FFN as a spatial pipeline: 16 DRAM-bank reader cores stream
each expert's weights once into gate/up cores, x relays read and tilize the row-major bf16 dispatch buffer and
multicast it, and down cores write bfp8 y tiles at each expert's region of a dispatch-shaped output. Per-expert token
counts and region offsets are read on device, so one cached program serves every routing.

Drop-in for TtRoutedExpert.forward: same inputs (the ROW_MAJOR bf16 dispatch buffer, counts, region offsets) and
the same output (bf8 TILE, expert rows at their regions), which combine already accepts.

Weights. The op reads its own bank layout, a pure TILE PERMUTATION of the per-expert (gate [H, I], up [H, I],
down [I, H]) tensors TtRoutedExpert uses: ``flat_tile_maps`` says where each tile goes, and the
``flat_routed_expert_gather_tiles`` helper copies the packed tiles there. A block-float tile carries its own shared
exponents, so the per-expert quantization is reused exactly (bit-identical weights) and building a layer is a memcpy,
not an unpack / re-quantize (a 384-expert Kimi-K2.7 layer lays out in ~5 s, about what loading it takes).

Nothing is ever written to disk. The weights come from the torch weights when given, else from TtRoutedExpert's
per-expert cache, read only; the flat layout is rebuilt in memory on every construction rather than cached, since that
costs no more than loading a cached copy would. ``TtMoe.check_cache_complete`` and the per-model cache markers keep
their meaning.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping

# The op's activation ids (se3_compute.cpp SE_ACT). SituGlu's betas are fixed in the kernel at Kimi-K3's 4 / 25.
FLAT_ACTIVATIONS = {
    ttnn.RoutedExpertActivation.Silu: 0,
    ttnn.RoutedExpertActivation.SwiGluOai: 1,
    ttnn.RoutedExpertActivation.SituGlu: 2,
    ttnn.RoutedExpertActivation.ClampedSiluGlu: 3,
}
FLAT_WEIGHT_DTYPES = {ttnn.bfloat4_b: "bf4", ttnn.bfloat8_b: "bf8"}
# The op is built for >= 256 tokens per expert of dispatch capacity (its helper relays); below that, use unified.
FLAT_MIN_TOKENS_PER_EXPERT = 256
FLAT_MAX_EXPERTS_PER_CHIP = 64  # the op's plan streams 1..64 local experts
_KBLK = 8  # K tiles per gate/up weight block, fixed by the op
_PROJS = ("gate", "up", "down")  # source index of local expert e, projection p: 3 * e + p

ROUTED_EXPERT_IMPL_ENV = "TT_DS_PREFILL_ROUTED_EXPERT_IMPL"


def resolve_routed_expert_impl(model_cfg) -> str:
    """The routed-expert op a model runs: ``$TT_DS_PREFILL_ROUTED_EXPERT_IMPL`` ("unified" / "flat") when set, for
    A/B runs, else the model config's ``ROUTED_EXPERT_IMPL``, else "unified"."""
    impl = os.environ.get(ROUTED_EXPERT_IMPL_ENV) or getattr(model_cfg, "ROUTED_EXPERT_IMPL", "unified")
    if impl not in ("unified", "flat"):
        raise ValueError(f"routed expert impl must be 'unified' or 'flat', got {impl!r} (${ROUTED_EXPERT_IMPL_ENV})")
    return impl


def flat_routed_expert_supported(
    mesh_device,
    activation,
    weights_dtype,
    emb_dim: int,
    hidden_dim: int,
    max_tokens: int,
    has_biases: bool,
    experts_per_chip: int,
    num_routed_experts: int,
) -> Optional[str]:
    """None when the flat op can run this configuration, else the reason it cannot. Ends by asking the op's planner
    itself (cached, host only), so a shape it has no layout for falls back instead of failing in the constructor."""
    if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
        return "Blackhole only"
    if activation not in FLAT_ACTIVATIONS:
        return f"activation {activation} not implemented"
    if weights_dtype not in FLAT_WEIGHT_DTYPES:
        return f"weights dtype {weights_dtype} (bfloat4_b / bfloat8_b only)"
    if has_biases:
        return "expert biases not implemented"
    if emb_dim % 256 != 0:
        return f"emb_dim {emb_dim} not a multiple of 256"
    if hidden_dim % 32 != 0:
        return f"hidden_dim {hidden_dim} not a multiple of 32"
    if not 1 <= experts_per_chip <= FLAT_MAX_EXPERTS_PER_CHIP:
        return f"{experts_per_chip} local experts (1..{FLAT_MAX_EXPERTS_PER_CHIP})"
    if max_tokens < FLAT_MIN_TOKENS_PER_EXPERT:
        return f"max tokens per expert {max_tokens} < {FLAT_MIN_TOKENS_PER_EXPERT}"
    try:
        ttnn._ttnn.operations.experimental.flat_routed_expert_plan(
            mesh_device,
            emb_dim,
            hidden_dim,
            experts_per_chip,
            num_routed_experts,
            max_tokens,
            weights_bf8=weights_dtype == ttnn.bfloat8_b,
        )
    except RuntimeError as error:
        return f"no layout for this shape ({str(error).splitlines()[0]})"
    return None


def crs_rects(cores):
    """Cores packed into maximal rectangles (greedy: widest x run, then grow in y), so that dispatch can multicast
    to them instead of writing every core."""
    left = {(c.x, c.y) for c in cores}
    rects = []
    for x0, y0 in sorted(left, key=lambda t: (t[1], t[0])):
        if (x0, y0) not in left:
            continue
        x1 = x0
        while (x1 + 1, y0) in left:
            x1 += 1
        y1 = y0
        while all((x, y1 + 1) in left for x in range(x0, x1 + 1)):
            y1 += 1
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                left.discard((x, y))
        rects.append(ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)))
    return ttnn.CoreRangeSet(rects)


def _kd_of(pcd, it):
    """K tiles per down-weight block of a down core with pcd output columns (the op's plan picks the same)."""
    return max(k for k in (8, 4, 2, 1) if k * pcd <= 16 and it % k == 0)


def _src(proj, e, kt, nt, n_tiles):
    """tile_map entries for local expert e's projection `proj` at tile rows kt / columns nt (broadcast), in a source
    with n_tiles tile columns: (source index << 32) | row-major tile index."""
    return (np.int64(3 * e + proj) << 32) | (kt * n_tiles + nt).astype(np.int64)


def flat_tile_maps(lay, experts_per_chip: int, ht: int, it: int) -> dict:
    """Where every tile of the op's weight tensors comes from. Per tensor name: (tile_map, shard tile rows), the map
    in row-major tile order of a [shard rows, banks] tile grid -- one tile column per DRAM bank -- with entries
    (source index << 32) | source tile, or -1 for a zero (padding) tile. Sources are the per-expert tensors in
    (e, gate / up / down) order: gate / up [H, I] (ht x it tiles), down [I, H] (it x ht tiles).

    Regions, each a run of tiles in one bank column (region j: bank j % banks, slot j // banks):
    * w_gu, reader r: per expert, per K-block c of 8 tiles, per pair-set pl < RG: for k < 8, for p < NP, gate then up
      at tile (c * 8 + k, ((r % n_rd_sg) * RG + pl) * NP + p);
    * w_d, down core d: per expert, per K-block c of kd = kd_of(pcds[d]) tiles: the [kd x pcds[d]] tiles of down at
      (c * kd + k, col0s[d] + q), zero-padded to the widest core's region;
    * w_rd, reader tail i: per expert, per K-block c of kd_r tiles: [kd_r x pcd_r] tiles of down at
      (c * kd_r + k, rem_cols + (i % n_rdn) * pcd_r + q).
    """
    E, RG, NP = experts_per_chip, lay["rg"], lay["np"]
    k = np.arange(_KBLK)
    regions = {"w_gu": [], "w_d": [], "w_rd": []}
    for r in range(lay["n_rd"]):
        runs = []
        for e in range(E):
            for c in range(lay["nk_gu"]):
                for pl in range(RG):
                    cols = ((r % lay["n_rd_sg"]) * RG + pl) * NP + np.arange(NP)
                    kt = (c * _KBLK + k)[:, None, None]
                    gu = np.stack([_src(0, e, kt, cols[None, :, None], it), _src(1, e, kt, cols[None, :, None], it)])
                    # [2, 8, NP, 1] -> order k, p, (gate, up)
                    runs.append(gu[..., 0].transpose(1, 2, 0).reshape(-1))
        regions["w_gu"].append(np.concatenate(runs))
    for d, (pcd, col0) in enumerate(zip(lay["pcds"], lay["col0s"])):
        kd = _kd_of(pcd, it)
        q = np.arange(pcd)
        regions["w_d"].append(
            np.concatenate(
                [
                    _src(2, e, (c * kd + np.arange(kd))[:, None], (col0 + q)[None, :], ht).reshape(-1)
                    for e in range(E)
                    for c in range(it // kd)
                ]
            )
        )
    if lay["rdown"]:
        kd_r, n_rdn, pcd_r, rem = lay["kd_r"], lay["n_rdn"], lay["pcd_r"], lay["rem_cols"]
        q = np.arange(pcd_r)
        for i in range(n_rdn * lay["nsg"]):
            col = rem + (i % n_rdn) * pcd_r
            regions["w_rd"].append(
                np.concatenate(
                    [
                        _src(2, e, (c * kd_r + np.arange(kd_r))[:, None], (col + q)[None, :], ht).reshape(-1)
                        for e in range(E)
                        for c in range(lay["nblk_r"])
                    ]
                )
            )

    banks = lay["banks"]
    maps = {}
    for name, regs in regions.items():
        if not regs:
            continue
        tiles = max(len(r_) for r_ in regs)  # down regions pad to the widest core
        per = -(-len(regs) // banks)
        grid = np.full((per * tiles, banks), -1, dtype=np.int64)
        for j, r_ in enumerate(regs):
            h, b = divmod(j, banks)
            grid[h * tiles : h * tiles + len(r_), b] = r_
        maps[name] = (grid.reshape(-1), per * tiles)
    return maps


def _bank_memory_config(shard_tile_rows, banks):
    """Width-sharded DRAM, one 32-wide column of tiles per bank."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (shard_tile_rows * 32, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )


# The down coordinators' done words: zero, and left zero by every launch, so one tensor serves every layer that runs
# the same plan one after another. One uint32 per down core / reader tail of a subgrid (<= 43 on a Galaxy chip; the
# op checks the fit), so 64 words: 256 B, the allocator's 64 B granularity. The L1 allocator gives a sharded buffer one
# address range on every core, so whatever this pins at the top of L1 is lost to every later op's static CBs
# (the 2 KB tile this used to be is the likely cause of ring_mla's 448 B static-CB clash on Kimi-K3 L24 chunked).
DONE_WORDS_PER_CORE = 64
_DONE_WORDS = {}


def _done_words(mesh_device, coords):
    key = (id(mesh_device), tuple(coords))
    if key not in _DONE_WORDS:
        cores = [ttnn.CoreCoord(x, y) for x, y in coords]
        _DONE_WORDS[key] = ttnn.from_torch(
            torch.zeros(len(cores), DONE_WORDS_PER_CORE, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs_rects(cores), (1, DONE_WORDS_PER_CORE), ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
    return _DONE_WORDS[key]


def _expert_cache_file(cache_path, cache_name_prefix, local_expert, proj, weights_dtype) -> Path:
    """The tensorbin TtRoutedExpert's ttnn.as_tensor writes for one local expert's projection."""
    name = f"{cache_name_prefix}.local_{local_expert}_{proj}_dtype_{weights_dtype.name}_layout_TILE.tensorbin"
    return Path(cache_path) / name


def _expert_cache_complete(cache_path, cache_name_prefix, experts_per_chip, weights_dtype) -> bool:
    """Every per-expert tensorbin present. Checked file by file rather than through TtRoutedExpert's
    check_cache_complete, whose directory listing is a process-wide snapshot taken before this layer ran."""
    return all(
        _expert_cache_file(cache_path, cache_name_prefix, e, proj, weights_dtype).is_file()
        for e in range(experts_per_chip)
        for proj in _PROJS
    )


class TtFlatRoutedExpert(LightweightModule):
    """TtRoutedExpert on the flat_routed_expert op. See the module docstring."""

    def __init__(
        self,
        mesh_device,
        experts_per_chip: int,
        num_routed_experts: int,
        dispatch_group_size: int,
        num_dispatch_groups: int,
        emb_dim: int,
        hidden_dim: int,
        max_tokens: int,
        torch_weights: Optional[list[dict]] = None,
        weights_dtype=ttnn.bfloat4_b,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
        *,
        activation: "ttnn.RoutedExpertActivation",
        pin: int = 1,
    ):
        """
        Args:
            experts_per_chip / num_routed_experts / dispatch_group_size / num_dispatch_groups: the EP placement
                (ExpertMapping, column-major: the same global ids TtRoutedExpert's global_expert_idx_table holds).
            emb_dim, hidden_dim: the routed expert's H (routed_emb_dim under LatentMoE) and I.
            max_tokens: dispatch capacity per expert (max_dispatched_tokens_per_expert); >= 256.
            torch_weights: per global expert {'gate_proj', 'up_proj', 'down_proj'} in HF [out, in] layout, or None to
                build from the per-expert cache.
            weight_cache_path / cache_name_prefix: TtRoutedExpert's cache location, whose per-expert tensorbins are
                READ when torch_weights is None. Nothing is ever written there. With neither weights nor a complete
                cache the weights are UNINITIALIZED (perf runs).
            activation: the routed expert's GLU activation.
            pin: pin the largest expert's weights in chunks of >= pin sub-blocks (0: off).
        """
        super().__init__()
        reason = flat_routed_expert_supported(
            mesh_device,
            activation,
            weights_dtype,
            emb_dim,
            hidden_dim,
            max_tokens,
            False,
            experts_per_chip,
            num_routed_experts,
        )
        if reason is not None:
            raise NotImplementedError(f"TtFlatRoutedExpert: {reason}")
        rows, cols = tuple(mesh_device.shape)
        assert (rows, cols) == (dispatch_group_size, num_dispatch_groups), (
            f"flat routed expert expects one dispatch group per mesh column: mesh {(rows, cols)}, "
            f"groups {(dispatch_group_size, num_dispatch_groups)}"
        )
        self.mesh_device = mesh_device
        self.experts_per_chip = experts_per_chip
        self.emb_dim, self.hidden_dim = emb_dim, hidden_dim
        self.max_tokens = max_tokens
        self.weights_dtype = weights_dtype
        self.activation = FLAT_ACTIVATIONS[activation]
        self.pin = pin

        table = ExpertMapping.create_global_expert_idx_table(
            experts_per_chip=experts_per_chip,
            dispatch_group_size=dispatch_group_size,
            num_dispatch_groups=num_dispatch_groups,
        )
        # gids[d]: row-major device d's local experts' global ids, in local order.
        self.gids = [[int(g) for g in table[c, r]] for r in range(rows) for c in range(cols)]

        self.plan = dict(
            ttnn._ttnn.operations.experimental.flat_routed_expert_plan(
                mesh_device,
                emb_dim,
                hidden_dim,
                experts_per_chip,
                num_routed_experts,
                max_tokens,
                weights_bf8=weights_dtype == ttnn.bfloat8_b,
                pin=pin,
            )
        )
        names = ("w_gu", "w_d", "w_rd") if self.plan["rdown"] else ("w_gu", "w_d")
        weights = self._build(names, torch_weights, weight_cache_path, cache_name_prefix)
        self.w_gu, self.w_d = weights["w_gu"], weights["w_d"]
        self.w_rd = weights.get("w_rd")

        gid_host = torch.tensor(self.gids, dtype=torch.int32).reshape(rows * cols, 1, experts_per_chip)
        self.gidx = ttnn.from_torch(
            gid_host,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        self.done = _done_words(mesh_device, self.plan["coords"])

    def _expert_sources(self, torch_weights, weight_cache_path, cache_name_prefix):
        """The per-expert host tensors, (e, gate / up / down) order, each mesh-distributed as TtRoutedExpert lays it
        out (gate / up [H, I], down [I, H]: x @ W), so device d's shard holds its local expert e. None when there are
        neither torch weights nor a complete per-expert cache. Reads only: nothing is written to the cache.

        From torch weights they are quantized exactly as TtRoutedExpert's ttnn.as_tensor quantizes them, so the flat
        weights are bit-identical to the unified op's either way."""
        if torch_weights is not None:
            return self._sources_from_torch(torch_weights)
        cached = weight_cache_path is not None and cache_name_prefix is not None
        if not cached or not _expert_cache_complete(
            weight_cache_path, cache_name_prefix, self.experts_per_chip, self.weights_dtype
        ):
            return None
        # The files ttnn.as_tensor wrote for TtRoutedExpert, loaded directly rather than through as_tensor, which on
        # a miss or a failed load writes the placeholder it is handed as the weights.
        return [
            ttnn.load_tensor(_expert_cache_file(weight_cache_path, cache_name_prefix, e, p, self.weights_dtype))
            for e in range(self.experts_per_chip)
            for p in _PROJS
        ]

    def _sources_from_torch(self, torch_weights):
        rows, cols = tuple(self.mesh_device.shape)
        n = rows * cols * self.experts_per_chip
        assert len(torch_weights) == n, f"expected {n} expert weights, got {len(torch_weights)}"
        mapper = ExpertMapping.get_weights_mesh_mapper(self.mesh_device)
        sources = []
        for e in range(self.experts_per_chip):
            per_proj = ExpertMapping.gather_weights_for_mesh_distribution(
                torch_weights, e, rows, cols, self.experts_per_chip
            )
            for ws in per_proj:  # gate, up, down; HF [out, in] -> [in, out]
                stacked = torch.stack([w.T.contiguous() for w in ws]).reshape(rows, cols, *ws[0].T.shape)
                sources.append(
                    ttnn.from_torch(stacked, dtype=self.weights_dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                )
        return sources

    def _build(self, names, torch_weights, weight_cache_path, cache_name_prefix):
        sources = self._expert_sources(torch_weights, weight_cache_path, cache_name_prefix)
        if sources is None:
            return self._placeholder_weights(names, cache_name_prefix)
        logger.info(f"TtFlatRoutedExpert: laying out {cache_name_prefix or 'expert'} weights (tile gather)")
        banks = self.plan["banks"]
        maps = flat_tile_maps(self.plan, self.experts_per_chip, self.emb_dim // 32, self.hidden_dim // 32)
        out = {}
        for n in names:
            tile_map, shard_tile_rows = maps[n]
            mc = _bank_memory_config(shard_tile_rows, banks)
            host = ttnn._ttnn.operations.experimental.flat_routed_expert_gather_tiles(
                sources, tile_map, ttnn.Shape([1, shard_tile_rows * 32, banks * 32]), mc
            )
            out[n] = host.to(self.mesh_device, mc)
        return out

    def _placeholder_weights(self, names, cache_name_prefix):
        """Uninitialized weights in the op's layout, for runs that have neither torch weights nor a routed-expert
        cache -- the perf legs, which build the experts from placeholders on purpose. Allocated on device directly."""
        logger.warning(
            f"TtFlatRoutedExpert {cache_name_prefix or ''}: no torch weights and no routed-expert cache; "
            "running UNINITIALIZED expert weights (fine for perf, wrong for accuracy)"
        )
        banks = self.plan["banks"]
        maps = flat_tile_maps(self.plan, self.experts_per_chip, self.emb_dim // 32, self.hidden_dim // 32)
        return {
            n: ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, maps[n][1] * 32, banks * 32]),
                self.weights_dtype,
                ttnn.TILE_LAYOUT,
                self.mesh_device,
                _bank_memory_config(maps[n][1], banks),
            )
            for n in names
        }

    def forward(
        self,
        dispatched_buffer: ttnn.Tensor,
        expert_token_counts: ttnn.Tensor,
        expert_region_offsets: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Args:
            dispatched_buffer: (max_dispatch_buffer_token_size, emb_dim) bf16 ROW_MAJOR, DRAM interleaved.
            expert_token_counts / expert_region_offsets: (1, num_routed_experts) uint32 ROW_MAJOR (offset_cumsum).

        Returns:
            (max_dispatch_buffer_token_size, emb_dim) bf8 TILE, each local expert's rows at its region (the rest
            unwritten; combine reads only the counted rows).
        """
        if dispatched_buffer.layout != ttnn.ROW_MAJOR_LAYOUT or dispatched_buffer.dtype != ttnn.bfloat16:
            dispatched_buffer = ttnn.to_layout(dispatched_buffer, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.tracy_message("`TT_SIGNPOST: FlatRoutedExpert`")
        return ttnn.experimental.deepseek_prefill.flat_routed_expert(
            dispatched_buffer,
            expert_token_counts,
            expert_region_offsets,
            self.gidx,
            self.w_gu,
            self.w_d,
            reader_down_weights=self.w_rd,
            done_words=self.done,
            intermediate=self.hidden_dim,
            max_tokens_per_expert=self.max_tokens,
            activation=self.activation,
            pin=self.pin,
        )
