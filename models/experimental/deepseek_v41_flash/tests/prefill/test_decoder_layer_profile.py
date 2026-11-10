# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device-op profile of one ``DeepSeekV41PrefillDecoderLayer`` over one prefill chunk, on real weights, as the model
runs it: on a ``1 x 4`` TP4 submesh of the Galaxy (fabric as in the prefill demo), bf8 attention, bf4 routed experts.

The device half only: nothing is compared. The streams, ``pre_mix`` and Engram rows are random and replicated over
the submesh, as the model's are. A warm-up chunk fills the program cache; the profiled chunk, run on a fresh state,
sits between ``PREFILL_LAYER_START`` / ``PREFILL_LAYER_END`` signposts.

    python -m tracy -p -r -v -m pytest models/experimental/deepseek_v41_flash/tests/prefill/test_decoder_layer_profile.py
"""

import pytest
import torch
from tracy import signpost

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _build_rope,
    _checkpoint_available,
)
from models.experimental.deepseek_v41_flash.tests.prefill.test_full_model_prefill_demo import (
    _ATTENTION_WEIGHT_DTYPE,
    _EXPERT_DTYPE,
    _TP_SIZE,
)
from models.experimental.deepseek_v41_flash.tt.config import (
    CACHE_DIR,
    DEFAULT_MODEL_DIR,
    decoder_layer_weights,
    expert_provider,
    load_config,
)
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts
from models.experimental.deepseek_v41_flash.tt.prefill.attention import PrefillShared
from models.experimental.deepseek_v41_flash.tt.prefill.decoder_layer import DeepSeekV41PrefillDecoderLayer
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("mesh_device", [(8, _TP_SIZE)], indirect=["mesh_device"], ids=["galaxy_8x4"])
@pytest.mark.parametrize("chunk", (1024,))
@pytest.mark.parametrize(
    "layer",
    (0, 1, 2, 20),
    ids=("window", "window_engram", "ratio2_source", "ratio1_source"),
)
@torch.no_grad()
def test_decoder_layer_prefill_profile(mesh_device, layer, chunk):
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    hc = config.hc_mult
    set_active_system_config(load_system_config(mesh_device=mesh_device).log())
    device = mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(0, 0))

    layer_cache = WeightCache(CACHE_DIR).sub(f"layers.{layer}")
    experts = DeepSeekV41PreloadedExperts(
        config,
        expert_provider(loader, layer),
        device,
        dtype=_EXPERT_DTYPE,
        cache=layer_cache.sub("ffn"),
        tp_size=_TP_SIZE,
        balance_dram_banks=True,
    )
    decoder_layer = DeepSeekV41PrefillDecoderLayer(
        config,
        layer,
        decoder_layer_weights(loader, layer, config),
        device,
        _build_rope(config, loader.snapshot_dir, chunk),
        experts,
        cache=layer_cache,
        weight_dtype=_ATTENTION_WEIGHT_DTYPE,
        tp_size=_TP_SIZE,
    )

    def to_device(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    torch.manual_seed(0)
    streams = to_device(torch.randn(1, hc, chunk, config.hidden_size))
    pre_mix = to_device(torch.rand(1, chunk, 1, hc))
    engram_rows = None
    if decoder_layer.engram is not None:
        engram_rows = to_device(torch.randn(1, 1, chunk, decoder_layer.engram.kin))

    for profiled in (False, True):
        state, shared = decoder_layer.new_state(), PrefillShared()
        ttnn.synchronize_device(device)
        if profiled:
            signpost("PREFILL_LAYER_START")
        out, out_pre = decoder_layer(streams, pre_mix, state, shared, engram_rows=engram_rows)
        ttnn.synchronize_device(device)
        if profiled:
            signpost("PREFILL_LAYER_END")
        # The MoE kernel opens per-token zones; one layer's markers fill the per-core profiler buffer, so drain
        # it after every run instead of only at teardown, where an overflowed buffer aborts the process.
        ttnn.ReadDeviceProfiler(device)
        for tensor in (out, out_pre, shared.entry_idx, shared.candidates):
            if tensor is not None:
                ttnn.deallocate(tensor)
