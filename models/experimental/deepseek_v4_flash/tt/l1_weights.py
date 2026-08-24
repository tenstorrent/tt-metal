# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pack two layers' resident weights into ONE height-sharded L1 tensor.

The L1 allocator hands out the same address range on every core, so a tensor
that occupies only a zone's cores still reserves (and wastes) that range on the
rest of the chip. Instead of one width-sharded tensor per weight, everything in
``l1_placement`` is fused into a single ``bfloat4_b`` tensor that is HEIGHT
sharded across **all 120 cores** with one equal-sized shard per core:

* A core's shard is the concatenation of the *tile streams* of every weight
  slab that ``l1_placement`` assigns to that core, in ``WEIGHT_ORDER``, layer 0
  first. A slab's tile stream is its 32x32 tiles in row-major order -- the same
  order the width-sharded / prefetched paths deliver, so a matmul consuming a
  region of the shard sees tiles exactly as it does today.
* Shards are zero-padded at the end to the size of the largest zone's shard, so
  every core carries the same allocation (``shard_layout().shard_tiles`` tiles).
* The shard is one tile wide: shape ``[shard_tiles * 32, 32]``. Tile t of a
  core's stream is rows ``[t*32, (t+1)*32)``. The full tensor is
  ``[120 * shard_tiles * 32, 32]`` and ``ttnn.from_torch`` with the
  HEIGHT_SHARDED config below places shard ``i`` on grid core ``i`` (row-major).

Consumers find a weight inside the shard through :func:`shard_layout`, which
maps ``(layer, weight)`` to a tile offset -- the same for every core of the
weight's zone. The norms and other bf16 tensors are NOT here (one tensor has
one dtype); they stay small separate allocations.

Host-only except :func:`build_l1_weight_tensor` / :func:`l1_memory_config`,
which import ttnn lazily. Self-test::

    python models/experimental/deepseek_v4_flash/tt/l1_weights.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .l1_placement import (
    GRID_H,
    GRID_W,
    NUM_CORES,
    TILE,
    ZONES,
    WeightPlacement,
    core_range_set,
    placements_for_layer,
)

# The order weight regions appear inside a shard (per layer; layer 0's regions
# precede layer 1's). Chosen to match the decode step's consumption order so a
# layer's reads walk the shard forward.
WEIGHT_ORDER = (
    "q_a_proj",
    "q_b_proj",
    "kv_proj",
    "compressor.gate_proj",
    "compressor.kv_proj",
    "o_a_proj",
    "o_b_proj",
    "shared_gate_proj",
    "shared_up_proj",
    "shared_down_proj",
    "router_gate",
    "attn_hc.fn",
    "ffn_hc.fn",
)

DEFAULT_LAYER_TYPES = ("heavily_compressed_attention", "compressed_sparse_attention")


@dataclass(frozen=True)
class ShardRegion:
    """One weight's slab inside every shard of its zone."""

    layer: int
    name: str
    zone: str
    tile_offset: int  # first tile of the region within the shard
    num_tiles: int  # slab size; region rows are [tile_offset*32, (tile_offset+num_tiles)*32)
    spec: WeightPlacement


@dataclass(frozen=True)
class ShardLayout:
    shard_tiles: int  # equal for every core, padding included
    regions: tuple[ShardRegion, ...]

    @property
    def shard_shape(self) -> tuple[int, int]:
        return self.shard_tiles * TILE, TILE

    def region(self, layer: int, name: str) -> ShardRegion:
        for r in self.regions:
            if r.layer == layer and r.name == name:
                return r
        raise KeyError(f"no region for layer {layer} weight {name!r}")


def packed_weight_spec(layout: ShardLayout, layer: int, name: str):
    """The matmul_decode descriptor for one region of a packed tensor."""
    import ttnn

    region = layout.region(layer, name)
    spec = region.spec
    return ttnn._ttnn.operations.experimental.MatmulDecodePackedWeightSpec(
        tile_offset=region.tile_offset,
        K=spec.K,
        N=spec.N,
        cores=core_range_set(spec.zone),
        k_blocks=spec.k_blocks or 1,
        batch=spec.batch or 1,
        b_blocks=spec.b_blocks or 1,
    )


def placement_weights_from_decoder_layer(weights: dict, layer_type: str) -> dict[str, torch.Tensor]:
    """Materialize a decoder layer's checkpoint weights under placement names."""

    def get(key):
        value = weights[key]
        return value() if callable(value) else value

    out = {
        "q_a_proj": get("self_attn.q_a_proj.weight"),
        "q_b_proj": get("self_attn.q_b_proj.weight"),
        "kv_proj": get("self_attn.kv_proj.weight"),
        "o_a_proj": get("self_attn.o_a_proj.weight"),
        "o_b_proj": get("self_attn.o_b_proj.weight"),
        "shared_gate_proj": get("mlp.shared_experts.gate_proj.weight"),
        "shared_up_proj": get("mlp.shared_experts.up_proj.weight"),
        "shared_down_proj": get("mlp.shared_experts.down_proj.weight"),
        "router_gate": get("mlp.gate.weight"),
        "attn_hc.fn": get("attn_hc.fn"),
        "ffn_hc.fn": get("ffn_hc.fn"),
    }
    if layer_type != "sliding_attention":
        out["compressor.gate_proj"] = get("self_attn.compressor.gate_proj.weight")
        out["compressor.kv_proj"] = get("self_attn.compressor.kv_proj.weight")
    return out


def shard_layout(
    layer_types: tuple[str, ...] = DEFAULT_LAYER_TYPES,
    weight_names_by_layer: tuple[frozenset[str], ...] | None = None,
) -> ShardLayout:
    """Tile offsets of every weight region, plus the common (padded) shard size."""
    if weight_names_by_layer is not None and len(weight_names_by_layer) != len(layer_types):
        raise ValueError("weight_names_by_layer must have one name set per layer")
    cursor = {zone: 0 for zone in ZONES}
    regions = []
    for layer, lt in enumerate(layer_types):
        placements = placements_for_layer(lt)
        for name in WEIGHT_ORDER:
            if weight_names_by_layer is not None and name not in weight_names_by_layer[layer]:
                continue
            spec = placements.get(name)
            if spec is None:  # e.g. no compressor on a sliding layer
                continue
            rows, cols = spec.shard_shape
            num_tiles = (rows // TILE) * (cols // TILE)
            regions.append(ShardRegion(layer, name, spec.zone, cursor[spec.zone], num_tiles, spec))
            cursor[spec.zone] += num_tiles
    return ShardLayout(shard_tiles=max(cursor.values()), regions=tuple(regions))


# --------------------------------------------------------------------------- #
# Host-side packing
# --------------------------------------------------------------------------- #


def _op_layout(name: str, w: torch.Tensor, spec: WeightPlacement) -> torch.Tensor:
    """Torch ``[out, in]`` checkpoint weight -> the op's ``[K, N]`` (or ``[batch, K, N]``)."""
    w = w.t().contiguous()  # [K, N_total]
    if spec.batch is not None:
        # Grouped projection: [K, batch * N] -> [batch, K, N].
        return w.reshape(spec.K, spec.batch, spec.N).permute(1, 0, 2).contiguous()
    if w.shape[1] < spec.N:  # e.g. hc fn's 24 outputs padded to one tile of 32
        w = torch.nn.functional.pad(w, (0, spec.N - w.shape[1]))
    if tuple(w.shape) != (spec.K, spec.N):
        raise ValueError(f"{name}: weight is {list(w.shape)}, placement wants [{spec.K}, {spec.N}]")
    return w


def _core_slab(w: torch.Tensor, spec: WeightPlacement, pos: int) -> torch.Tensor:
    """The ``[rows, cols]`` slab of zone-relative core ``pos`` (row-major, as
    decode_weight_layout assigns: core = kb * n_blocks + nb / bb * n_blocks + nb)."""
    kc, nc = spec.shard_shape[0], spec.shard_shape[1]
    nb = pos % spec.n_blocks
    if spec.batch is not None:
        bb = pos // spec.n_blocks
        bc = spec.batch // spec.b_blocks
        groups = w[bb * bc : (bb + 1) * bc, :, nb * nc : (nb + 1) * nc]  # [Bc, K, Nc]
        return groups.reshape(bc * spec.K, nc)
    if spec.k_blocks is not None:
        kb = pos // spec.n_blocks
        return w[kb * kc : (kb + 1) * kc, nb * nc : (nb + 1) * nc]
    return w[:, nb * nc : (nb + 1) * nc]


def _tile_stream(slab: torch.Tensor) -> torch.Tensor:
    """``[R, C]`` -> ``[R*C/32, 32]``: the slab's 32x32 tiles, row-major, stacked
    one tile wide. Tilizing the result reproduces the original tiles byte-for-byte."""
    r, c = slab.shape
    rt, ct = r // TILE, c // TILE
    return slab.reshape(rt, TILE, ct, TILE).permute(0, 2, 1, 3).reshape(rt * ct * TILE, TILE)


def pack_host_tensor(
    weights_by_layer: list[dict[str, torch.Tensor]],
    layer_types: tuple[str, ...] = DEFAULT_LAYER_TYPES,
    weight_names_by_layer: tuple[frozenset[str], ...] | None = None,
) -> tuple[torch.Tensor, ShardLayout]:
    """Fuse two layers' weights into the ``[120 * shard_rows, 32]`` host tensor.

    ``weights_by_layer[i]`` maps placement names to torch checkpoint-orientation
    (``[out, in]``) tensors for layer ``i`` (``o_a_proj`` as its packed
    ``[o_groups * o_lora_rank, hidden]`` GroupedLinear weight).
    """
    layout = shard_layout(layer_types, weight_names_by_layer)
    ops: dict[tuple[int, str], torch.Tensor] = {}
    for region in layout.regions:
        ops[(region.layer, region.name)] = _op_layout(
            region.name, weights_by_layer[region.layer][region.name], region.spec
        )

    shard_rows = layout.shard_tiles * TILE
    fused = torch.zeros(NUM_CORES * shard_rows, TILE, dtype=torch.bfloat16)
    for zone, (start, count) in ZONES.items():
        zone_regions = [r for r in layout.regions if r.zone == zone]
        for pos in range(count):
            core = start + pos
            base = core * shard_rows
            for r in zone_regions:
                stream = _tile_stream(_core_slab(ops[(r.layer, r.name)], r.spec, pos))
                row0 = base + r.tile_offset * TILE
                fused[row0 : row0 + stream.shape[0]] = stream
    return fused, layout


# --------------------------------------------------------------------------- #
# Device side (lazy ttnn)
# --------------------------------------------------------------------------- #


def l1_memory_config(shard_tiles: int):
    """HEIGHT_SHARDED over the full 12x10 grid: shard ``i`` -> core ``i`` row-major."""
    import ttnn

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_W - 1, GRID_H - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_tiles * TILE, TILE], ttnn.ShardOrientation.ROW_MAJOR),
    )


def build_l1_weight_tensor(
    weights_by_layer: list[dict[str, torch.Tensor]] | None,
    device,
    layer_types: tuple[str, ...] = DEFAULT_LAYER_TYPES,
    cache_file_name=None,
    weight_names_by_layer: tuple[frozenset[str], ...] | None = None,
):
    """The fused resident weight tensor, on ``device`` in bf4, plus its layout."""
    import ttnn

    if weights_by_layer is None:
        if cache_file_name is None:
            raise ValueError("weights_by_layer may be None only when loading a cached fused tensor")
        layout = shard_layout(layer_types, weight_names_by_layer)
        fused = None
    else:
        fused, layout = pack_host_tensor(weights_by_layer, layer_types, weight_names_by_layer)
    tensor = ttnn.as_tensor(
        fused,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=l1_memory_config(layout.shard_tiles),
        cache_file_name=cache_file_name,
    )
    return tensor, layout


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.manual_seed(0)

    def dummy_layer(layer_type: str) -> dict[str, torch.Tensor]:
        out = {}
        for name, spec in placements_for_layer(layer_type).items():
            n_out = 24 if name.endswith("hc.fn") else (spec.batch or 1) * spec.N
            # bf16, the dtype pack_host_tensor stores: comparing against float32
            # sources would fail on the (intended) downcast rather than on layout.
            out[name] = torch.randn(n_out, spec.K, dtype=torch.bfloat16)
        return out

    layer_types = DEFAULT_LAYER_TYPES
    weights = [dummy_layer(lt) for lt in layer_types]
    fused, layout = pack_host_tensor(weights, layer_types)

    shard_rows = layout.shard_tiles * TILE
    assert fused.shape == (NUM_CORES * shard_rows, TILE)

    # Every region on every core of its zone must round-trip to the slab
    # computed straight from the source weight.
    for region in layout.regions:
        op = _op_layout(region.name, weights[region.layer][region.name], region.spec)
        start, count = ZONES[region.zone]
        for pos in (0, count - 1):
            base = (start + pos) * shard_rows + region.tile_offset * TILE
            expect = _tile_stream(_core_slab(op, region.spec, pos))
            got = fused[base : base + expect.shape[0]]
            assert torch.equal(got, expect), f"mismatch: layer {region.layer} {region.name} pos {pos}"

    # Padding beyond each zone's last region must be zero.
    for zone, (start, count) in ZONES.items():
        used = max((r.tile_offset + r.num_tiles for r in layout.regions if r.zone == zone), default=0)
        pad = fused[start * shard_rows + used * TILE : start * shard_rows + shard_rows]
        assert torch.all(pad == 0)

    bf4_bytes = layout.shard_tiles * 576
    print(f"layer types: {layer_types}")
    print(f"shard: {layout.shard_tiles} tiles = [{shard_rows}, {TILE}] = {bf4_bytes / 1024:.0f} KB/core in bf4")
    print(
        f"fused tensor: [{fused.shape[0]}, {fused.shape[1]}], {NUM_CORES * bf4_bytes / 2**20:.2f} MB over {NUM_CORES} cores"
    )
    print("\nregions (zone, layer, name, tile offset, tiles):")
    for r in sorted(layout.regions, key=lambda r: (r.zone, r.layer, r.tile_offset)):
        print(f"  {r.zone}  L{r.layer}  {r.name:<22} @ {r.tile_offset:>5} + {r.num_tiles:>4} tiles")
    for zone, (start, count) in ZONES.items():
        used = max((r.tile_offset + r.num_tiles for r in layout.regions if r.zone == zone), default=0)
        print(f"{zone}: {used} tiles used, {layout.shard_tiles - used} padded")
    print("\nself-test passed")
