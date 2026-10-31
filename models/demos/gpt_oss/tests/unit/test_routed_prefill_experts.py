# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Routed top-k prefill experts (tt/experts/prefill_routed.py) vs the dense all-expert prefill path and the HF reference.

Real GPT-OSS weights of one layer (router + MXFP4 experts dequantized from the checkpoint), 1xN Blackhole mesh.
Checks, per sequence length and layer:
  * routed MoE (router + experts) vs dense MoE: PCC >= 0.99;
  * routed experts vs the HF experts on the same routing: PCC >= the experts threshold;
  * routed MoE vs the HF MLP: PCC >= the mlp threshold;
  * sum of per-expert token counts == k * S, no NaN, repeat runs identical;
  * capacity worst case: every token routed to experts 0..k-1 (one device holds all of them), nothing dropped;
  * multi-chunk: S=4096 with GPT_OSS_ROUTED_PREFILL_CHUNK=1024 (4 chunks), same checks and thresholds;
  * a full chunk plus a 64-token last chunk: S=16448 at the default chunk (16384), S=8256 with
    GPT_OSS_ROUTED_PREFILL_CHUNK=8192; S=8256 at the default chunk runs as one chunk shorter than the operator;
  * pair positions exact at S=1088 (one chunk shorter than the shared prefix operator; the dense path cannot run
    1088, so no MoE case), at the full default chunk (16384) and at a full chunk plus a 64-token chunk (16448);
  * combine token blocks: S=2112 (a full 2048-token combine block plus a 64-token last block), routed vs HF only
    (the dense path cannot run a 2112-token chunk);
  * supports(): batch > 1 and non-multiple-of-64 lengths run dense.

Run: pytest models/demos/gpt_oss/tests/unit/test_routed_prefill_experts.py -k 1x4
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.common.utility_functions import comp_pcc

from ...tt.experts.prefill_routed import (
    ROUTED_PREFILL_CHUNK_ENV,
    ROUTED_PREFILL_ENV,
    ROUTED_PREFILL_NUM_EXPERTS,
    ROUTED_PREFILL_PLACEMENT_ENV,
    routed_prefill_mesh_supported,
)
from ...tt.mlp import MLP
from ..test_factory import TestFactory, parametrize_mesh_with_fabric, real_checkpoint_path, skip_1x4_on_galaxy

THRESHOLDS = json.loads((Path(__file__).resolve().parents[2] / "unit_test_thresholds.json").read_text())
ROUTED_VS_DENSE_PCC = 0.99
# Lengths of test_routed_prefill_experts that the dense prefill path can also run (the routed-vs-dense check).
DENSE_REF_LENGTHS = (128, 1024, 4096, 8256, 16448)


def _routed_setup(mesh_device):
    """TestFactory setup and the HF checkpoint directory, or skip: the routed path needs a single-row Blackhole mesh and
    gpt-oss-20b, and these checks need the real checkpoint weights (CI's unit tests only have the config)."""
    if not routed_prefill_mesh_supported(mesh_device):
        pytest.skip("routed prefill needs a single-row Blackhole mesh with > 1 device")
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    if setup["config"].num_local_experts != ROUTED_PREFILL_NUM_EXPERTS:
        pytest.skip(f"the routed prefill path is built for {ROUTED_PREFILL_NUM_EXPERTS} experts (gpt-oss-20b)")
    model_path = real_checkpoint_path(os.environ.get("HF_MODEL", setup["model_args"].model_path))
    if model_path is None:
        pytest.skip("needs the real checkpoint weights (HF_MODEL with model.safetensors.index.json)")
    return setup, model_path


def _load_layer_moe(model_path, layer_idx):
    """Router + dequantized expert weights of one layer, in the MLP state-dict layout (HF names)."""
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.layers.{layer_idx}.mlp."
    raw = {}
    for name, fname in index.items():
        if name.startswith(prefix):
            with safe_open(str(Path(model_path) / fname), framework="pt") as f:
                raw[name[len(prefix) :]] = f.get_tensor(name)
    experts = {
        "gate_up_proj": convert_moe_packed_tensors(
            raw["experts.gate_up_proj_blocks"], raw["experts.gate_up_proj_scales"]
        ),
        "gate_up_proj_bias": raw["experts.gate_up_proj_bias"].to(torch.bfloat16),
        "down_proj": convert_moe_packed_tensors(raw["experts.down_proj_blocks"], raw["experts.down_proj_scales"]),
        "down_proj_bias": raw["experts.down_proj_bias"].to(torch.bfloat16),
    }
    router = {"weight": raw["router.weight"].to(torch.bfloat16), "bias": raw["router.bias"].to(torch.bfloat16)}
    return router, experts


def _bf4_round_trip(experts):
    """Expert weights as the device holds them (bfloat4_b per [K, N] matrix), back in bf16."""

    def q(w):  # w: [E, K, N]
        t = ttnn.from_torch(w.float(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
        return ttnn.to_torch(t).to(torch.bfloat16)

    gu = experts["gate_up_proj"]
    gate, up = q(gu[..., ::2].contiguous()), q(gu[..., 1::2].contiguous())
    gu_q = torch.stack([gate, up], dim=-1).reshape(gu.shape)
    return {**experts, "gate_up_proj": gu_q, "down_proj": q(experts["down_proj"])}


def _load_norm_gamma(model_path, layer_idx):
    index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())["weight_map"]
    name = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
    with safe_open(str(Path(model_path) / index[name]), framework="pt") as f:
        return f.get_tensor(name).float()


def _reference_mlp(config, router, experts):
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP

    config._experts_implementation = "eager"
    mlp = GptOssMLP(config).eval().to(torch.bfloat16)
    state = {f"router.{k}": v for k, v in router.items()}
    state.update({f"experts.{k}": v for k, v in experts.items()})
    mlp.load_state_dict(state, strict=True)
    return mlp.float()


def _to_tt(t, mesh_device, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _first_device(t, mesh_device):
    """Replicated output -> torch, from device 0."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _pcc(a, b, threshold):
    passing, pcc = comp_pcc(a.float(), b.float(), threshold)
    return passing, pcc


def _gain(ref, out):
    """Least-squares scale of out onto ref (PCC is blind to a uniform scale error; the residual stream is not)."""
    ref, out = ref.double().flatten(), out.double().flatten()
    return float((out * ref).sum() / (ref * ref).sum())


def _routed_counts_total(routed, tt_indices, mesh_device):
    """Sum over devices of each device's own experts' token counts (the dispatch/unified-op view)."""
    _, counts, _ = routed._routing(tt_indices)
    per_device = [
        ttnn.to_torch(t).reshape(-1)[: routed.num_experts].to(torch.int64) for t in ttnn.get_device_tensors(counts)
    ]
    epc = routed.experts_per_chip
    own = [c[d * epc : (d + 1) * epc] for d, c in enumerate(per_device)]
    return int(sum(o.sum() for o in own)), torch.cat(own)


@pytest.fixture
def routed_env(monkeypatch):
    monkeypatch.setenv(ROUTED_PREFILL_ENV, "1")
    monkeypatch.setenv("GPT_OSS_ROUTED_PREFILL_MIN_SEQ", "64")


@skip_1x4_on_galaxy
@parametrize_mesh_with_fabric([(1, 4)])
@pytest.mark.parametrize(
    "seq_len, chunk",
    [(128, None), (1024, None), (2112, None), (4096, None), (4096, 1024), (8256, None), (8256, 8192), (16448, None)],
    ids=["S128", "S1024", "S2112", "S4096", "S4096-chunk1024", "S8256", "S8256-chunk8192", "S16448"],
)
@pytest.mark.parametrize("layer_idx", [0, 12], ids=["layer0", "layer12"])
def test_routed_prefill_experts(
    mesh_device, device_params, seq_len, chunk, layer_idx, routed_env, monkeypatch, reset_seeds
):
    setup, model_path = _routed_setup(mesh_device)
    if chunk is not None:
        # Multi-chunk: the sequence runs as seq_len / chunk routed chunks, concatenated before the all-reduce.
        monkeypatch.setenv(ROUTED_PREFILL_CHUNK_ENV, str(chunk))
    config = setup["config"]
    thresholds = THRESHOLDS[setup["model_args"].model_name]["prefill"]

    router_sd, experts_sd = _load_layer_moe(model_path, layer_idx)
    reference = _reference_mlp(config, router_sd, experts_sd)
    state_dict = {f"router.{k}": v for k, v in router_sd.items()}
    state_dict.update({f"experts.{k}": v for k, v in experts_sd.items()})

    tt_mlp = MLP(
        mesh_device,
        config,
        state_dict,
        setup["ccl_manager"],
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        mesh_config=setup["mesh_config"],
        use_throughput_experts=False,
    )
    routed = tt_mlp.routed_prefill
    assert routed is not None, "routed prefill experts were not built"
    H = config.hidden_size
    assert routed.supports((1, 1, seq_len, H))
    assert not routed.supports((1, 1, 96, H)) and not routed.supports(
        (1, 1, seq_len + 32, H)
    ), "non-multiple-of-64 lengths must run dense"
    assert not routed.supports((2, 1, seq_len, H)) and not routed.supports(
        (1, 2, seq_len, H)
    ), "batch > 1 must run dense"
    if chunk is not None:
        assert routed.chunk_size == chunk and seq_len > chunk, f"expected a multi-chunk run, chunk {chunk}"
    if seq_len == 16448:
        # The default chunk: one full chunk on the [chunk, chunk] operator plus a 64-token chunk on the small one.
        assert routed.chunk_size == 16384, f"S16448 expects the 16384 default chunk, got {routed.chunk_size}"
    if seq_len % routed.chunk_size != 0:
        # The last (or only) chunk is shorter than the shared prefix operator and uses its top-left block.
        assert seq_len % routed.chunk_size < routed._prefix_tri.cmax

    hidden = torch.randn(1, seq_len, config.hidden_size)

    def tt_hidden():
        # Fresh device input per call: the dense prefill path deallocates its input.
        return _to_tt(hidden.unsqueeze(0), mesh_device, ttnn.bfloat8_b)

    hidden_q = ttnn.to_torch(ttnn.get_device_tensors(tt_hidden())[0]).float().reshape(seq_len, config.hidden_size)

    # --- full MoE: routed vs dense vs HF ---
    out_routed = _first_device(tt_mlp(tt_hidden(), is_decode=False), mesh_device).reshape(seq_len, -1)
    out_routed2 = _first_device(tt_mlp(tt_hidden(), is_decode=False), mesh_device).reshape(seq_len, -1)
    # S2112 (one routed chunk: a full ROUTED_COMBINE_BLOCK of 2048 tokens plus a 64-token last block) has no dense
    # reference: the dense path splits each chunk into 1024-token pieces and cannot run a 2112-token chunk.
    dense_ref = seq_len in DENSE_REF_LENGTHS
    out_dense = None
    if dense_ref:
        min_seq = routed.min_seq
        routed.min_seq = 1 << 30  # force the dense path
        out_dense = _first_device(tt_mlp(tt_hidden(), is_decode=False), mesh_device).reshape(seq_len, -1)
        routed.min_seq = min_seq
    with torch.no_grad():
        ref_mlp, _ = reference(hidden_q.reshape(1, seq_len, -1))
    ref_mlp = ref_mlp.reshape(seq_len, -1)

    assert not torch.isnan(out_routed).any(), "NaN in routed MoE output"
    assert torch.equal(out_routed, out_routed2), "routed MoE output differs between identical runs"
    ok_rd, pcc_rd = _pcc(out_dense, out_routed, ROUTED_VS_DENSE_PCC) if dense_ref else (True, None)
    ok_rm, pcc_rm = _pcc(ref_mlp, out_routed, thresholds["mlp"])
    pcc_dm = _pcc(ref_mlp, out_dense, 0.0)[1] if dense_ref else None
    logger.info(
        f"S={seq_len} layer={layer_idx}: MoE routed-vs-dense {pcc_rd} routed-vs-HF {pcc_rm} dense-vs-HF {pcc_dm}"
    )

    # --- experts on the same (HF) routing: routed vs HF experts ---
    with torch.no_grad():
        logits = torch.nn.functional.linear(hidden_q, reference.router.weight, reference.router.bias)
        top_vals, top_idx = torch.topk(logits, config.num_experts_per_tok, dim=-1, sorted=True)
        top_w = torch.softmax(top_vals, dim=-1)
        ref_experts = reference.experts(hidden_q, router_indices=top_idx, routing_weights=top_w).reshape(seq_len, -1)
    tt_idx = _to_tt(top_idx.to(torch.int32), mesh_device, ttnn.uint16)
    tt_w = _to_tt(top_w, mesh_device, ttnn.bfloat16)
    out_experts = _first_device(routed(tt_hidden(), tt_idx, tt_w), mesh_device).reshape(seq_len, -1)
    ok_e, pcc_e = _pcc(ref_experts, out_experts, thresholds["experts"])
    total, own_counts = _routed_counts_total(routed, tt_idx, mesh_device)
    expected_counts = torch.bincount(top_idx.reshape(-1), minlength=config.num_local_experts)
    logger.info(
        f"S={seq_len} layer={layer_idx}: experts routed-vs-HF {pcc_e}; counts total {total} "
        f"(max/expert {int(own_counts.max())}, per-device {[int(c.sum()) for c in own_counts.split(routed.experts_per_chip)]})"
    )
    assert total == config.num_experts_per_tok * seq_len, f"sum(counts) {total} != k*S"
    assert torch.equal(own_counts, expected_counts), "device token counts differ from the routing histogram"
    assert not torch.isnan(out_experts).any(), "NaN in routed experts output"

    # --- capacity worst case: all tokens to experts 0..k-1 (all on device 0) ---
    k = config.num_experts_per_tok
    skew_idx = torch.arange(k).repeat(seq_len, 1)
    skew_w = torch.softmax(torch.randn(seq_len, k), dim=-1)
    with torch.no_grad():
        ref_skew = reference.experts(hidden_q, router_indices=skew_idx, routing_weights=skew_w).reshape(seq_len, -1)
    out_skew = _first_device(
        routed(
            tt_hidden(),
            _to_tt(skew_idx.to(torch.int32), mesh_device, ttnn.uint16),
            _to_tt(skew_w, mesh_device, ttnn.bfloat16),
        ),
        mesh_device,
    ).reshape(seq_len, -1)
    ok_s, pcc_s = _pcc(ref_skew, out_skew, thresholds["experts"])
    logger.info(f"S={seq_len} layer={layer_idx}: skewed (all tokens -> experts 0..{k-1}) routed-vs-HF {pcc_s}")
    assert not torch.isnan(out_skew).any(), "NaN in skewed routed output"
    # A dropped token leaves a zero row; every row of the reference is nonzero.
    zero_rows = int((out_skew.abs().sum(dim=-1) == 0).sum())
    assert zero_rows == 0, f"{zero_rows} tokens dropped in the capacity worst case"

    assert ok_rd, f"routed vs dense MoE PCC {pcc_rd} < {ROUTED_VS_DENSE_PCC}"
    assert ok_rm, f"routed MoE vs HF MLP PCC {pcc_rm} < {thresholds['mlp']}"
    assert ok_e, f"routed experts vs HF experts PCC {pcc_e} < {thresholds['experts']}"
    assert ok_s, f"skewed routed experts vs HF PCC {pcc_s} < {thresholds['experts']}"


@skip_1x4_on_galaxy
@parametrize_mesh_with_fabric([(1, 4)])
@pytest.mark.parametrize("seq_len", [1024], ids=["S1024"])
def test_routed_prefill_layers_back_to_back(mesh_device, device_params, seq_len, routed_env, reset_seeds):
    """Several layers' MoE on one device, interleaved: every layer has the same shapes, so each op after the first
    layer runs from the program cache with that layer's own weights, biases and tables."""
    setup, model_path = _routed_setup(mesh_device)
    config = setup["config"]
    layers = [0, 12, 23]
    mlps, refs, ref_bf16s, ref_qs = [], [], [], []
    for layer_idx in layers:
        router_sd, experts_sd = _load_layer_moe(model_path, layer_idx)
        refs.append(_reference_mlp(config, router_sd, experts_sd))
        ref_bf16s.append(_reference_mlp(config, router_sd, experts_sd).to(torch.bfloat16))
        ref_qs.append(_reference_mlp(config, router_sd, _bf4_round_trip(experts_sd)))
        state_dict = {f"router.{k}": v for k, v in router_sd.items()}
        state_dict.update({f"experts.{k}": v for k, v in experts_sd.items()})
        mlps.append(
            MLP(
                mesh_device,
                config,
                state_dict,
                setup["ccl_manager"],
                dtype=ttnn.bfloat8_b,
                tensor_cache_path=None,
                mesh_config=setup["mesh_config"],
                use_throughput_experts=False,
            )
        )
    # One [chunk, chunk] prefix operator per device, shared by every layer.
    holders = {id(m.routed_prefill._prefix_tri) for m in mlps}
    assert len(holders) == 1, f"{len(holders)} prefix operators for {len(mlps)} layers, expected one shared"
    # The MoE input is RMSNorm(x) * gamma: unit-variance rows scaled by the layer's post-attention norm weight.
    hiddens = [torch.randn(1, seq_len, config.hidden_size) * _load_norm_gamma(model_path, l) for l in layers]
    results = []
    for rnd in range(2):
        for layer_idx, tt_mlp, ref, hidden in zip(layers, mlps, refs, hiddens):

            def tt_hidden():
                return _to_tt(hidden.unsqueeze(0), mesh_device, ttnn.bfloat8_b)

            hidden_q = ttnn.to_torch(ttnn.get_device_tensors(tt_hidden())[0]).float().reshape(1, seq_len, -1)
            out_routed = _first_device(tt_mlp(tt_hidden(), is_decode=False), mesh_device).reshape(seq_len, -1)
            tt_mlp.routed_prefill.min_seq = 1 << 30
            out_dense = _first_device(tt_mlp(tt_hidden(), is_decode=False), mesh_device).reshape(seq_len, -1)
            tt_mlp.routed_prefill.min_seq = 64
            with torch.no_grad():
                ref_out = ref(hidden_q)[0].reshape(seq_len, -1)
                ref_bf16 = ref_bf16s[layers.index(layer_idx)](hidden_q.bfloat16())[0].float().reshape(seq_len, -1)
                ref_q = ref_qs[layers.index(layer_idx)](hidden_q)[0].reshape(seq_len, -1)
            ok, pcc = _pcc(out_dense, out_routed, ROUTED_VS_DENSE_PCC)
            _, pcc_hf = _pcc(ref_out, out_routed, 0.0)
            _, pcc_dhf = _pcc(ref_out, out_dense, 0.0)
            logger.info(
                f"round {rnd} layer {layer_idx}: routed-vs-dense {pcc} routed-vs-HF {pcc_hf} dense-vs-HF {pcc_dhf} "
                f"gain routed/HF {_gain(ref_out, out_routed):.4f} dense/HF {_gain(ref_out, out_dense):.4f} "
                f"HFbf16/HF {_gain(ref_out, ref_bf16):.4f} | vs bf4-weight fp32 ref: routed gain "
                f"{_gain(ref_q, out_routed):.4f} pcc {_pcc(ref_q, out_routed, 0.0)[1]:.5f} dense gain "
                f"{_gain(ref_q, out_dense):.4f} pcc {_pcc(ref_q, out_dense, 0.0)[1]:.5f}"
            )
            results.append((rnd, layer_idx, ok, pcc))
    bad = [r for r in results if not r[2]]
    assert not bad, f"routed vs dense below {ROUTED_VS_DENSE_PCC}: {bad}"


@skip_1x4_on_galaxy
@parametrize_mesh_with_fabric([(1, 4)])
@pytest.mark.parametrize(
    "seq_len", [128, 1024, 1088, 8192, 16384, 16448], ids=["S128", "S1024", "S1088", "S8192", "S16384", "S16448"]
)
def test_routed_pair_positions_exact(mesh_device, device_params, seq_len, routed_env, reset_seeds):
    """Dispatch-buffer row of every (token, slot) pair equals the integer reference on every device: region start of
    the pair's expert + its rank among that expert's pairs (token order) for local pairs, the spare tile otherwise.
    Random routing and the worst case (every token -> experts 0..k-1, ranks up to C - 1). A sequence longer than the
    routed chunk is split as RoutedPrefillExperts.__call__ splits it, and every chunk is checked on its own."""
    setup, model_path = _routed_setup(mesh_device)
    config = setup["config"]
    router_sd, experts_sd = _load_layer_moe(model_path, 0)
    state_dict = {f"router.{k}": v for k, v in router_sd.items()}
    state_dict.update({f"experts.{k}": v for k, v in experts_sd.items()})
    tt_mlp = MLP(
        mesh_device,
        config,
        state_dict,
        setup["ccl_manager"],
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        mesh_config=setup["mesh_config"],
        use_throughput_experts=False,
    )
    routed = tt_mlp.routed_prefill
    E, k, epc = routed.num_experts, routed.top_k, routed.experts_per_chip
    patterns = {
        "random": torch.stack([torch.randperm(E)[:k] for _ in range(seq_len)]),
        "skew": torch.arange(k).repeat(seq_len, 1),
    }
    C_max = min(routed.chunk_size, seq_len)
    chunks = [(s, min(s + C_max, seq_len)) for s in range(0, seq_len, C_max)]
    for (name, idx_full), (start, end) in ((p, c) for p in patterns.items() for c in chunks):
        idx = idx_full[start:end]
        C = end - start
        name = name if len(chunks) == 1 else f"{name} chunk [{start}, {end})"
        tt_idx = _to_tt(idx.to(torch.int32), mesh_device, ttnn.uint16)
        _, counts, region_offsets = routed._routing(tt_idx)
        consts = routed._chunk_constants(C)
        pos_tm, pos_sm = routed._pair_positions(tt_idx, region_offsets, consts)
        pos_d = [ttnn.to_torch(t).reshape(-1).to(torch.int64) for t in ttnn.get_device_tensors(pos_tm)]
        # slot-major [1, k*C] -> token-major order
        pos_sm_d = [
            ttnn.to_torch(t).reshape(k, C).t().reshape(-1).to(torch.int64) for t in ttnn.get_device_tensors(pos_sm)
        ]
        reg_d = [ttnn.to_torch(t).reshape(-1)[:E].to(torch.int64) for t in ttnn.get_device_tensors(region_offsets)]
        cnt_d = [ttnn.to_torch(t).reshape(-1)[:E].to(torch.int64) for t in ttnn.get_device_tensors(counts)]
        onehot = torch.nn.functional.one_hot(idx, E).sum(dim=1)  # [C, E]
        rank = torch.cumsum(onehot, dim=0) - onehot  # pairs of e from earlier tokens
        for d in range(routed.num_devices):
            local = (idx >= d * epc) & (idx < (d + 1) * epc)
            ref = torch.where(
                local,
                reg_d[d][idx] + torch.gather(rank, 1, idx),
                torch.full_like(idx, consts["spare"]),
            ).reshape(-1)
            for order, got in (("token-major", pos_d[d]), ("slot-major", pos_sm_d[d])):
                mism = int((got != ref).sum())
                assert (
                    mism == 0
                ), f"{name} S={seq_len} device {d}: {mism} {order} pair positions differ from the reference"
            own = list(range(d * epc, (d + 1) * epc))
            ends = [int(reg_d[d][e] + cnt_d[d][e]) for e in own]
            starts = [int(reg_d[d][e]) for e in own]
            assert all(ends[i] <= starts[i + 1] for i in range(epc - 1)), f"{name} device {d}: regions overlap"
            assert ends[-1] <= consts["spare"], f"{name} device {d}: last region runs into the spare tile"
        logger.info(f"S={seq_len} {name}: pair positions exact on {routed.num_devices} devices")


@skip_1x4_on_galaxy
@parametrize_mesh_with_fabric([(1, 4)])
@pytest.mark.parametrize("seq_len", [1024, 8192], ids=["S1024", "S8192"])
def test_routed_prefill_placement(mesh_device, device_params, seq_len, routed_env, monkeypatch, reset_seeds):
    """Calibrated expert -> device placement (prefill_routed_placement.py) vs the identity placement, layer 12.

    Same weights and inputs. The routed router emits slot ids (expert order[v] as id v): the same experts as the
    identity router up to bf16 logit ties. Both MoE outputs match the HF MLP equally well; on HF routing (mapped to slot
    ids) the experts match the HF experts, including the capacity worst case (every token to the k experts of
    device 0), and the per-slot token counts are the slot-id histogram."""
    if not routed_prefill_mesh_supported(mesh_device) or mesh_device.shape[1] != 4:
        pytest.skip("the calibrated placement is for a 1x4 Blackhole mesh")
    setup, model_path = _routed_setup(mesh_device)
    config = setup["config"]
    thresholds = THRESHOLDS[setup["model_args"].model_name]["prefill"]
    layer_idx = 12
    E, k = config.num_local_experts, config.num_experts_per_tok
    router_sd, experts_sd = _load_layer_moe(model_path, layer_idx)
    reference = _reference_mlp(config, router_sd, experts_sd)
    state_dict = {f"router.{name}": v for name, v in router_sd.items()}
    state_dict.update({f"experts.{name}": v for name, v in experts_sd.items()})

    def build(mode):
        monkeypatch.setenv(ROUTED_PREFILL_PLACEMENT_ENV, mode)
        return MLP(
            mesh_device,
            config,
            state_dict,
            setup["ccl_manager"],
            dtype=ttnn.bfloat8_b,
            tensor_cache_path=None,
            mesh_config=setup["mesh_config"],
            use_throughput_experts=False,
            layer_idx=layer_idx,
        )

    mlp_id, mlp_cal = build("identity"), build("calibrated")
    routed = mlp_cal.routed_prefill
    assert mlp_id.routed_prefill.expert_order is None
    order = routed.expert_order
    assert order is not None and sorted(order) == list(range(E)) and order != sorted(order)
    epc = routed.experts_per_chip

    hidden = torch.randn(1, seq_len, config.hidden_size) * _load_norm_gamma(model_path, layer_idx)

    def tt_hidden():
        return _to_tt(hidden.unsqueeze(0), mesh_device, ttnn.bfloat8_b)

    hidden_q = ttnn.to_torch(ttnn.get_device_tensors(tt_hidden())[0]).float().reshape(1, seq_len, -1)
    out_id = _first_device(mlp_id(tt_hidden(), is_decode=False), mesh_device).reshape(seq_len, -1)
    out_cal = _first_device(mlp_cal(tt_hidden(), is_decode=False), mesh_device).reshape(seq_len, -1)
    out_cal2 = _first_device(mlp_cal(tt_hidden(), is_decode=False), mesh_device).reshape(seq_len, -1)
    with torch.no_grad():
        ref = reference(hidden_q)[0].reshape(seq_len, -1)
    # Not bit-identical: the router logits are bfloat8_b (shared exponent per 16 columns), so the permuted columns
    # round near-ties differently and ~4.5% of tokens swap one top-4 expert for a near-tied one (measured PCC 0.9939
    # vs identity, the same PCC vs HF); each device also rounds its partial sum to bf8 over another expert grouping.
    ok_ic, pcc_ic = _pcc(out_id, out_cal, ROUTED_VS_DENSE_PCC)
    ok_ref, pcc_ref = _pcc(ref, out_cal, thresholds["mlp"])
    _, pcc_ref_id = _pcc(ref, out_id, 0.0)

    # Router ids: the routed router's top-k sets are the identity router's, renumbered, up to bf8 logit rounding.
    idx_id, _ = mlp_id.router(tt_hidden(), True)
    idx_cal, _ = mlp_cal.routed_router(tt_hidden(), True)
    ids_id = ttnn.to_torch(ttnn.get_device_tensors(idx_id)[0]).to(torch.int64).reshape(seq_len, -1)
    ids_cal = ttnn.to_torch(ttnn.get_device_tensors(idx_cal)[0]).to(torch.int64).reshape(seq_len, -1)
    mapped = torch.as_tensor(order)[ids_cal]
    same_sets = float((mapped.sort(-1).values == ids_id.sort(-1).values).all(-1).float().mean())
    same_slots = float((mapped == ids_id).float().mean())
    set_equal = (mapped.sort(-1).values == ids_id.sort(-1).values).all(-1)
    # A token whose top-k set changed should have a tie between its k-th and (k+1)-th logit in the identity router's
    # logits (the same linear call the router makes); the untied rest is bounded below.
    lg = ttnn.linear(
        ttnn.reshape(tt_hidden(), (-1, config.hidden_size)),
        mlp_id.router.weight,
        bias=mlp_id.router.bias,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    lg = ttnn.to_torch(ttnn.get_device_tensors(lg)[0]).float().reshape(seq_len, E)
    top_sorted = lg.sort(-1, descending=True).values
    boundary_tie = top_sorted[:, k - 1] == top_sorted[:, k]
    untied_changes = int((~set_equal & ~boundary_tie).sum())
    # Diagnosis of untied changes: which router's top-k set differs from the exact top-k of the same bf16 logits.
    exact = lg.topk(k, dim=-1).indices.sort(-1).values
    off_id = int((~(ids_id.sort(-1).values == exact).all(-1) & ~boundary_tie).sum())
    off_cal = int((~(mapped.sort(-1).values == exact).all(-1) & ~boundary_tie).sum())
    pos = torch.empty(E, dtype=torch.long)
    pos[torch.as_tensor(order)] = torch.arange(E)  # HF expert id -> slot id

    # HF routing (as slot ids) straight into the routed experts: per-slot counts and the experts of each device.
    hq = hidden_q.reshape(seq_len, -1)
    with torch.no_grad():
        logits = torch.nn.functional.linear(hq, reference.router.weight, reference.router.bias)
        top_vals, top_idx = torch.topk(logits, k, dim=-1, sorted=True)
        top_w = torch.softmax(top_vals, dim=-1)
        ref_experts = reference.experts(hq, router_indices=top_idx, routing_weights=top_w).reshape(seq_len, -1)
    slot_idx = pos[top_idx]
    tt_idx = _to_tt(slot_idx.to(torch.int32), mesh_device, ttnn.uint16)
    out_experts = _first_device(
        routed(tt_hidden(), tt_idx, _to_tt(top_w, mesh_device, ttnn.bfloat16)), mesh_device
    ).reshape(seq_len, -1)
    ok_e, pcc_e = _pcc(ref_experts, out_experts, thresholds["experts"])
    _, counts, _ = routed._routing(tt_idx)
    cnt = [ttnn.to_torch(t).reshape(-1)[:E].to(torch.int64) for t in ttnn.get_device_tensors(counts)]
    hist = torch.bincount(slot_idx.reshape(-1), minlength=E)  # per slot
    loads_id = torch.bincount(top_idx.reshape(-1), minlength=E).reshape(4, epc).sum(-1)
    loads_cal = hist.reshape(4, epc).sum(-1)

    # Capacity worst case: every token to the k experts held by device 0 under this placement (slot ids 0..k-1).
    skew_idx = torch.as_tensor(order[:k]).repeat(seq_len, 1)
    skew_w = torch.softmax(torch.randn(seq_len, k), dim=-1)
    with torch.no_grad():
        ref_skew = reference.experts(hq, router_indices=skew_idx, routing_weights=skew_w).reshape(seq_len, -1)
    out_skew = _first_device(
        routed(
            tt_hidden(),
            _to_tt(pos[skew_idx].to(torch.int32), mesh_device, ttnn.uint16),
            _to_tt(skew_w, mesh_device, ttnn.bfloat16),
        ),
        mesh_device,
    ).reshape(seq_len, -1)
    ok_s, pcc_s = _pcc(ref_skew, out_skew, thresholds["experts"])
    logger.info(
        f"S={seq_len}: MoE calibrated-vs-identity {pcc_ic}, calibrated-vs-HF {pcc_ref}, identity-vs-HF {pcc_ref_id}; "
        f"experts-vs-HF {pcc_e}, skew {pcc_s}; router top-k sets equal through the placement {same_sets:.4f} "
        f"(slots {same_slots:.4f}; boundary ties {int(boundary_tie.sum())}, changed sets without a tie {untied_changes}, untied sets off the exact top-k: identity {off_id} calibrated {off_cal}); device loads identity {loads_id.tolist()} calibrated {loads_cal.tolist()}"
    )
    assert torch.equal(out_cal, out_cal2), "calibrated placement output differs between identical runs"
    assert not torch.isnan(out_cal).any() and not torch.isnan(out_experts).any() and not torch.isnan(out_skew).any()
    assert all(torch.equal(c, hist) for c in cnt), "device token counts differ from the routing histogram"
    assert int((out_skew.abs().sum(dim=-1) == 0).sum()) == 0, "tokens dropped in the capacity worst case"
    # A changed set without a tie in the identity logits comes from the permuted router's own bf8 rounding (its logits
    # differ by up to one bf16 ulp; measured 1 of 1024 and 31 of 8192 tokens), so the bound is 1% of the tokens.
    assert untied_changes <= seq_len // 100, f"{untied_changes} tokens changed their top-k experts without a tie"
    assert abs(pcc_ref - pcc_ref_id) < 0.002, f"MoE vs HF PCC: calibrated {pcc_ref} vs identity {pcc_ref_id}"
    assert ok_ic, f"calibrated vs identity placement MoE PCC {pcc_ic} < {ROUTED_VS_DENSE_PCC}"
    assert ok_ref, f"calibrated placement MoE vs HF MLP PCC {pcc_ref} < {thresholds['mlp']}"
    assert ok_e, f"calibrated placement experts vs HF experts PCC {pcc_e} < {thresholds['experts']}"
    assert ok_s, f"calibrated placement skewed experts vs HF PCC {pcc_s} < {thresholds['experts']}"


@pytest.mark.parametrize("nb", [2, 32, 33], ids=["nb2", "nb32", "nb33"])
def test_prefix_bmm_config_one_block_per_core(nb):
    """_prefix_bmm_config guards a ttnn matmul bug: a MatmulMultiCoreReuseProgramConfig with more than one output block
    per core returned wrong sums (tt-ops runs/gpt-oss-20b-prefill-b32/team-note-matmul-multiblock-wrong-sums.md). The
    blocked prefix sum's within / totals configs (shapes as in RoutedPrefillExperts) must exist while the nb blocks fit
    one per core of the grid (nb 2..32 on the 8x8 / 8x4 grids) and fall back to the ttnn default (None) above that."""
    from ...tt.experts import prefill_routed as pr

    device_grid = ttnn.CoreCoord(11, 10)  # Blackhole compute grid
    bt, et = pr.ROUTED_PREFIX_BLOCK // ttnn.TILE_SIZE, 1  # E = 32 experts: one tile column
    within = pr._prefix_bmm_config(pr._PREFIX_WITHIN_GRID, device_grid, nb, bt, bt, et, pr._PREFIX_WITHIN_PCM)
    totals = pr._prefix_bmm_config(pr._PREFIX_TOTALS_GRID, device_grid, nb, 1, bt, et, 1)
    if nb > 32:
        assert within is None and totals is None, f"nb={nb}: blocks exceed one per core but a config was returned"
    else:
        assert within is not None and totals is not None, f"nb={nb}: blocks fit one per core but no config"
        assert within.per_core_M == pr._PREFIX_WITHIN_PCM and totals.per_core_M == 1
