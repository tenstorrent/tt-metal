# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse MLA perf and determinism cases: the box this host is (BOX), the gate cases and their device params
(gate_params), and one built layer with its caches (build_sparse_perf_case).

Boxes, from the device count and TT_MESH_GRAPH_DESC_PATH:
  * Galaxy: 8x4, SP=8 on mesh axis 0, torus XY, 5k chunks.
  * Galaxy 4x4: one 4x4 prefill stage on a tray half (tests/run_4x4_stage_tests.sh; [LINE, RING] descriptor): SP=4
    on the axis-1 ring, TP=4 on the axis-0 line, torus X, 2.5k chunks over the same 50k / 500k contexts.
  * Galaxy 4x4 subtorus: the same stage on the 16-chip subtorus (single_bh_galaxy_subtorus_xy4 descriptor; its header
    names the chips to make visible): rings on both axes, torus XY.
  * LoudBox: 2x4, SP on axis 0, Fabric2D (QuietBox 2x2 for the exploration matrix).
"""

import copy
import gc
import json
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import ttnn
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import random_mla_weights
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.reference import pretrained_mla_weights
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    STAGE_4X4_TOPOLOGY,
    fabric2d_device_params,
    mesh_descriptor_topologies,
    model_fabric_payload,
    torus_x_device_params,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_mesh import detect_num_devices
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import make_hidden
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import read_sharded_rows

GALAXY_SP = 8
# Galaxy-global scenario sizes: a 5k chunk over a 50k (warm) or 500k (long) cache. A 4x4 stage runs 2.5k chunks.
CHUNK_TOKENS = 5120
STAGE_4X4_CHUNK_TOKENS = 2560
WARM_CACHE_TOKENS = 51200
LONG_CACHE_TOKENS = 512000
SCENARIO_CACHE_TOKENS = {"warm": WARM_CACHE_TOKENS, "long": LONG_CACHE_TOKENS}
# Gate cases: (variant, shared_layer). GLM-5.3 also gates a shared layer, which reuses a full layer's top-k and runs
# no indexer.
GATE_CASES = {
    "kimi_k3_1": ("kimi_k3_1", False),
    "glm_5_3": ("glm_5_3", False),
    "glm_5_3-no_indexer": ("glm_5_3", True),
}

# Real-data cases: GLM-5.3 traces (chunked_group_a_v1 layout) that hold the full layer's input for every prompt
# position (decoder_io/decoder_output_layer_<layer - 1>). name -> (trace dir, cache tokens); the timed chunk is the
# last one, rows [cache, cache + CHUNK_TOKENS).
GLM53_TRACES = Path("/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/golden_traces")
REAL_DATA_CASES = {
    "55k": (GLM53_TRACES / "vllm-glm53-indexer-kcache-55k", 51200),
    # The 1M trace's prompt (code_debug), prefilled on a Galaxy with test_glm_prefill_transformer_chunked_no_pcc and
    # TT_PREFILL_DUMP_LAYERS=73 (layer 74's input, 102,400 rows).
    "64k": (GLM53_TRACES / "glm53-real-prompt-100k", 61440),
}
GLM53_CHECKPOINT = Path(os.environ.get("GLM53_HF_MODEL", "/mnt/weka/model-weights/llm/zai-org/GLM-5.3-fp8-aca966e4"))


@dataclass(frozen=True)
class RealData:
    """A prompt's real inputs to one layer: its pretrained weights (from `checkpoint`) and the previous layer's output
    for every position (from `trace_dir`), normalized by the layer's input RMSNorm as the decoder block does."""

    trace_dir: Path
    checkpoint: Path = GLM53_CHECKPOINT

    def _shards(self, prefix: str) -> list[str]:
        weight_map = json.loads((self.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        return sorted({str(self.checkpoint / f) for k, f in weight_map.items() if k.startswith(prefix)})

    def weights(self, config, layer: int) -> dict:
        return pretrained_mla_weights(
            config, layer=layer, checkpoint_path=self._shards(f"model.layers.{layer}.self_attn.")
        )

    def layer_input(self, config, layer: int, start: int, end: int) -> torch.Tensor:
        """Rows [start, end) of `layer`'s attention input, [1, 1, end - start, hidden] bf16."""
        key = f"model.layers.{layer}.input_layernorm.weight"
        with safe_open(self._shards(key)[0], framework="pt") as f:
            gamma = f.get_tensor(key).float()
        name = f"decoder_output_layer_{layer - 1}"
        h = read_sharded_rows(self.trace_dir / "decoder_io" / name, name, start, end)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps) * gamma
        return h.to(torch.bfloat16).reshape(1, 1, end - start, -1)


@dataclass(frozen=True)
class PerfBox:
    """Where the gate runs: `name` keys the targets; meshes are (rows, cols), with SP on `sp_axis`."""

    name: str
    mesh: tuple[int, int]
    device_params: object
    sp_axis: int
    chunk: int
    # Box-local cache = Galaxy cache x cache_scale: LoudBox / QuietBox keep Galaxy's per-chip depth (SP/8); a 4x4
    # stage holds the whole context (1).
    cache_scale: Fraction

    @property
    def id(self) -> str:
        base, mesh = self.name.lower().replace(" ", "-"), f"{self.mesh[0]}x{self.mesh[1]}"
        return base if mesh in base else f"{base}-{mesh}"

    def cache_tokens(self, galaxy_cache: int) -> int:
        return int(galaxy_cache * self.cache_scale)


def _scaled_box(name, mesh, device_params):
    """An SP-on-axis-0 box with fewer chips than Galaxy, scaled to Galaxy's per-chip work."""
    sp = mesh[0]
    return PerfBox(name, mesh, device_params, 0, CHUNK_TOKENS * sp // GALAXY_SP, Fraction(sp, GALAXY_SP))


def _galaxy_box() -> PerfBox:
    """The Galaxy layout TT_MESH_GRAPH_DESC_PATH selects: a 4x4 stage or the full 8x4."""
    descriptor = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    topologies = set(mesh_descriptor_topologies(descriptor)) if descriptor else set()
    if topologies == {STAGE_4X4_TOPOLOGY}:
        return PerfBox("Galaxy 4x4", (4, 4), torus_x_device_params, 1, STAGE_4X4_CHUNK_TOKENS, Fraction(1))
    if topologies == {("4", "4", "RING", "RING")}:
        return PerfBox("Galaxy 4x4 subtorus", (4, 4), torus_xy_device_params, 1, STAGE_4X4_CHUNK_TOKENS, Fraction(1))
    assert not any(t[:2] == ("4", "4") for t in topologies), f"{descriptor}: unsupported 4x4 layout {topologies}"
    return PerfBox("Galaxy", (8, 4), torus_xy_device_params, 0, CHUNK_TOKENS, Fraction(1))


def detect_box(num_devices: int, matrix: bool = False) -> PerfBox | None:
    if not ttnn.device.is_blackhole():  # DSA ops are Blackhole-only; an 8-chip Wormhole T3K is not a LoudBox
        return None
    if num_devices == 32:
        return _galaxy_box()
    if num_devices == 8:
        return _scaled_box("LoudBox", (2, 4), fabric2d_device_params)
    if num_devices == 4 and matrix:
        return _scaled_box("QuietBox", (2, 2), fabric2d_device_params)
    return None


@dataclass
class SparsePerfCase:
    mla: ttMLA
    x: ttnn.Tensor
    rope: dict
    kvpe_cache: ttnn.Tensor
    index_kv_cache: ttnn.Tensor | None
    cache_layer_idx: int
    # ShardTensor2dMesh dims of a chunk's input: sequence over SP, hidden over TP.
    input_shard_dims: tuple
    # A GLM-5.3 shared layer runs no indexer: it takes a full layer's top-k indices.
    indexer_indices: ttnn.Tensor | None = None
    # The previous layer's record of the prefix it pushed for this layer (timed in the full-layer case).
    prefix_entry: tuple | None = None

    def forward(self, start: int) -> ttnn.Tensor:
        """One chunk forward at `start`; the caller owns (and deallocates) the output."""
        if self.prefix_entry is not None:  # restored each forward, so this layer pushes only the rest
            self.mla.tt_ccl.sparse_kv_push_planner.prefetched[self.mla.layer_idx] = self.prefix_entry
        return self.mla.forward(
            self.x,
            self.rope,
            self.kvpe_cache,
            actual_start=start,
            cache_layer_idx=self.cache_layer_idx,
            index_kv_cache=self.index_kv_cache,
            indexer_indices=self.indexer_indices,
        )


def build_sparse_perf_case(
    mesh_device,
    variant,
    config_only,
    *,
    cache: int,
    chunk: int,
    topology,
    attn_mode: str = "sparse",
    kv_cache_format=MlaKvCacheFormat.SCALED_FP8,
    tp_shard_kv: bool = True,
    shared_layer: bool = False,
    sp_axis: int = 0,
    real_data: RealData | None = None,
) -> SparsePerfCase:
    """ttMLA, its caches and one chunk of input for forwards over a `cache + chunk` prefix.

    sparse: DSA indexer + sparse SDPA over `kv_cache_format`, caches SPxTP-sharded when `tp_shard_kv`.
    dense: the v3.1 baseline, no indexer, full-prefix ring MLA over the tiled BFP8 cache.
    GLM-5.3 sparse layers run inside a stage, as in the model: the full layer builds the push plan and pushes the next
    layer's prefix beside its sparse SDPA.
    shared_layer (GLM-5.3, sparse): the shared layer after it; it reuses the full layer's top-k, pushes only the rest of
    its KV and pushes the next layer's prefix beside its sparse SDPA.
    real_data (GLM-5.3, sparse): the full layer gets its pretrained weights and a real prompt's input, and runs every
    chunk before the timed one to fill its caches, so its indexer scores real keys and its top-k (which the shared
    layer reuses) is the model's. The shared layer keeps random weights: it runs no indexer, so they don't change
    which rows it reads.
    """
    has_indexer = attn_mode == "sparse"
    tp_axis = 1 - sp_axis
    sp, tp = mesh_device.shape[sp_axis], mesh_device.shape[tp_axis]
    kv_tp_axis = tp_axis if tp_shard_kv else None
    total = cache + chunk
    # The cold loop steps by `chunk`, and the indexed rope table needs total to be a whole number of chunks.
    assert cache % chunk == 0, f"cache {cache} must be a whole number of {chunk}-token chunks"
    assert total % sp == 0 and (total // sp) % 32 == 0, f"total {total} must be tile-aligned per SP={sp} chip"
    if tp_shard_kv:
        for name, n in (("cache", total), ("chunk", chunk)):
            assert (
                n % (sp * tp) == 0 and (n // (sp * tp)) % 32 == 0
            ), f"tp_shard_kv: {name} {n} must split into tile-aligned per-chip stripes over sp {sp} x tp {tp}"
    assert config_only.num_attention_heads % tp == 0 and config_only.index_n_heads % tp == 0, (
        f"{variant.name} heads (MLA={config_only.num_attention_heads}, index={config_only.index_n_heads}) "
        f"must be divisible by TP={tp}"
    )

    config = copy.deepcopy(config_only)
    config.max_seq_len = total
    full_indexer_layers = num_full_indexer_layers(config) if has_indexer else None
    index_cache_layers = 1
    mla_layer_idx, mla_layer_num, cache_layer_idx, first_layer_idx = 0, 1, 0, None
    if full_indexer_layers:
        full_layer_indices = [i for i, mode in enumerate(config.indexer_types) if mode == "full"]
        assert len(full_layer_indices) == full_indexer_layers
        # GLM-5.3: the last full layer in a stage from the previous full layer to its successor, so the ring score reads
        # a real index-K slot (1 of 2) and the layer pushes its successor's prefix.
        first_layer_idx, mla_layer_idx = full_layer_indices[-2:]
        cache_layer_idx = mla_layer_idx - first_layer_idx
        mla_layer_num, index_cache_layers = cache_layer_idx + 2, 2
    # Kimi-K3.1: the last DSA layer's slot of a 24-slot cache.
    if variant.name == "kimi_k3_1" and has_indexer:
        mla_layer_ids = variant.model_config.mla_layer_ids()
        mla_layer_num, mla_layer_idx = len(mla_layer_ids), mla_layer_ids[-1]
        cache_layer_idx = variant.model_config.mla_kv_slot(mla_layer_idx)
        index_cache_layers = mla_layer_num

    if real_data is not None:
        assert has_indexer and full_indexer_layers, f"real_data needs a GLM-5.3 sparse layer, got {variant.name}"
    mla = ttMLA(
        config,
        dict(real_data.weights(config, mla_layer_idx) if real_data else random_mla_weights(config)),
        mesh_device,
        topology=topology,
        layer_idx=mla_layer_idx,
        seq_len=total,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        active_seq_len=chunk,
        layer_num=mla_layer_num,
        first_layer_idx=first_layer_idx,
        has_indexer=has_indexer,
        sparse_kv_cache_format=kv_cache_format if has_indexer else MlaKvCacheFormat.BF16_RM,
    )
    rope = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).get_rope_tensors_indexed(total, chunk)
    kvpe_cache = init_mla_kv_cache(
        cache_format=kv_cache_format if has_indexer else MlaKvCacheFormat.BFP8_TILE,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=total,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=mla_layer_num if has_indexer else 1,
        tp_axis=kv_tp_axis if has_indexer else None,
    )
    # Both caches start at zero: the ring indexer always covers the full prefix, so cache contents don't change
    # op shapes or timing. (cold fills them with real chunk forwards.)
    index_kv_cache = None
    if has_indexer:
        index_kv_cache = init_kvpe_cache(
            kvpe_cache_head_dim=mla._indexer.index_args.index_head_dim,
            mesh_device=mesh_device,
            seq_len=total,
            mesh_shape=list(mesh_device.shape),
            sp_axis=sp_axis,
            num_kvpe_cache_layers=index_cache_layers,
            num_users=1,
            dtype=ttnn.bfloat8_b,
            tp_axis=kv_tp_axis,
        )
    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2

    def to_device(hidden):
        return ttnn.from_torch(
            hidden,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        )

    if real_data is None:
        x = to_device(make_hidden(chunk, config.hidden_size, seed=42)[:, :chunk].unsqueeze(0))
    else:
        # Fill the caches the way the prompt did: one forward per chunk before the timed one.
        for start in range(0, cache, chunk):
            x = to_device(real_data.layer_input(config, mla_layer_idx, start, start + chunk))
            ttnn.deallocate(
                mla.forward(
                    x,
                    rope,
                    kvpe_cache,
                    actual_start=start,
                    cache_layer_idx=cache_layer_idx,
                    index_kv_cache=index_kv_cache,
                )
            )
            ttnn.deallocate(x)
        x = to_device(real_data.layer_input(config, mla_layer_idx, cache, cache + chunk))
    case = SparsePerfCase(mla, x, rope, kvpe_cache, index_kv_cache, cache_layer_idx, tuple(shard_dims))
    if not shared_layer:
        return case

    assert has_indexer and full_indexer_layers, f"{variant.name} has no shared (indexer-reuse) layers"
    out, indices = mla.forward(
        x,
        rope,
        kvpe_cache,
        actual_start=cache,
        cache_layer_idx=cache_layer_idx,
        index_kv_cache=index_kv_cache,
        return_indexer_indices=True,
    )
    ttnn.deallocate(out)
    planner = mla.tt_ccl.sparse_kv_push_planner  # None where the push is off: the shared layer gathers
    prefix_entry = planner.prefetched.get(mla_layer_idx + 1) if planner is not None else None
    # Free the full layer before building the shared one: ttMLA holds reference cycles, so without a collection its
    # buffers would linger and could be freed mid-measurement.
    ttnn.deallocate(index_kv_cache)
    ttnn.deallocate(kvpe_cache.storage)
    del case, mla, kvpe_cache, index_kv_cache
    gc.collect()
    # The shared layer after the full layer, in the same stage extended by one layer, so it has a shared successor.
    assert config.indexer_types[mla_layer_idx + 1 : mla_layer_idx + 3] == ["shared"] * 2
    shared = ttMLA(
        config,
        dict(random_mla_weights(config)),
        mesh_device,
        topology=topology,
        layer_idx=mla_layer_idx + 1,
        seq_len=total,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        active_seq_len=chunk,
        layer_num=mla_layer_num + 1,
        first_layer_idx=first_layer_idx,
        has_indexer=True,
        sparse_kv_cache_format=kv_cache_format,
    )
    shared_kvpe_cache = init_mla_kv_cache(
        cache_format=kv_cache_format,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=total,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        num_kvpe_cache_layers=mla_layer_num + 1,
        tp_axis=kv_tp_axis,
    )
    return SparsePerfCase(
        shared,
        x,
        rope,
        shared_kvpe_cache,
        None,
        cache_layer_idx + 1,
        tuple(shard_dims),
        indexer_indices=indices,
        prefix_entry=prefix_entry,
    )


BOX = detect_box(detect_num_devices())


def gate_params():
    if BOX is None:
        return [
            pytest.param(
                (8, 4),
                {},
                "kimi_k3_1",
                "kimi_k3_1",
                marks=pytest.mark.skip(reason="sparse MLA perf gate runs on Blackhole Galaxy and LoudBox only"),
            )
        ]
    params = []
    for case, (variant, _) in GATE_CASES.items():
        dp = BOX.device_params(
            fabric_payload_size=model_fabric_payload(variant),
            worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
            trace_region_size=16 * 1024 * 1024,
        )
        params.append(pytest.param(BOX.mesh, dp, variant, case, id=f"{BOX.id}-{case}"))
    return params


def real_data_params():
    """GLM-5.3 gate cases x REAL_DATA_CASES, on the 8x4 Galaxy only: the traces hold Galaxy-global positions, which a
    scaled box (smaller chunk and cache) or a 4x4 stage would not line up with."""
    if BOX is None or BOX.name != "Galaxy":
        return [
            pytest.param(
                (8, 4),
                {},
                "glm_5_3",
                "glm_5_3",
                next(iter(REAL_DATA_CASES)),
                marks=pytest.mark.skip(reason="real-data sparse MLA perf runs on the Blackhole Galaxy 8x4 only"),
            )
        ]
    return [
        pytest.param(*p.values, data, id=f"{p.id}-{data}")
        for p in gate_params()
        if p.values[2] == "glm_5_3"
        for data in REAL_DATA_CASES
    ]
