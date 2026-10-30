# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decode PCC of ``DeepSeekV41DecoderLayer`` against the checkpoint's own ``Block``, on real weights.

A case decodes a group of layers token by token from position 0, each layer on its own random
streams and ``pre_mix`` and in layer order -- the order their shared compressed KV / top-k needs.
The reference (``tests/reference.py``) steps the same blocks the same way, an Engram layer's
``model.Engram`` ahead of its block as ``Transformer.forward`` does.

The cases cover every kind of layer: window-only, a ratio-2 kv + index source and its consumer, a
ratio-1 source, consumer and index-only source, and both Engram layers (1: window-only, 14: a ratio-2
source). An Engram case reads that layer's n-gram table into host RAM (~101 GB), skipping without
1.25x that free, and feeds random token ids.

The streams carry the input through the hyper-connection mix almost unchanged, so the output alone
says little about the sublayers. Each checked step therefore compares, by PCC, what the layer adds
to the streams (``out - x``) and the attention output, and by absolute error the ``pre_mix`` the
layer hands on (4 values, mostly the learned bias). The MoE output is compared over all checked
steps together, since a near-tied 6th expert can swap on a single token (see ``test_moe.py``).
Every step is logged; only the checked ones are asserted.

``test_decoder_layer_decode`` runs on one chip, ``test_decoder_layer_decode_tp4`` the same cases
TP4 on a 1x4 submesh of an 8x4 Galaxy; there the outputs are replicated and rank 0's are checked.
``test_decoder_layer_decode_s5`` feeds the same positions as ``S``-row blocks, streams ``[1, S, hc, D]``
(DSpark's parallel verify), against the same token-by-token reference, row by row.

The converted ttnn weights are cached under ``CACHE_DIR`` (``$DEEPSEEK_V4_CACHE_DIR``, default ``../cache``, per checkpoint),
shared with ``test_attention.py``, ``test_moe.py`` and ``test_engram.py``. The routed experts are
nearly all of a layer's weights; ``tools/fill_expert_cache.py`` converts them ahead of time, on the
host and across processes: by default the TP4 test's, and with ``--tp-size 1`` the TP1 test's.

    python models/experimental/deepseek_v41_flash/tools/fill_expert_cache.py --layers 20 21 24
    pytest models/experimental/deepseek_v41_flash/tests/decode/test_decoder_layer.py
"""

import contextlib
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tests.reference import (
    load_reference,
    reference_blocks,
    reference_engram,
    reference_engram_hash,
)
from models.experimental.deepseek_v41_flash.tt.config import (
    CACHE_DIR,
    DEFAULT_MODEL_DIR,
    decoder_layer_weights,
    expert_provider,
    load_config,
)
from models.experimental.deepseek_v41_flash.tt.decode.attention import build_decode_caches
from models.experimental.deepseek_v41_flash.tt.decode.decoder_layer import DeepSeekV41DecoderLayer
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts
from models.experimental.deepseek_v41_flash.tt.engram_lookup import (
    EngramHostLookup,
    engram_ram_needed,
    load_engram_table,
)
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config
from models.experimental.deepseek_v4_flash.tt.decode.hyperconnection import _HC_COLLAPSE_CORES
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session

PCC = 0.95
# The carried pre is ~sigmoid(bias), moving a few hundredths between tokens; bf16 alone costs ~1e-3.
# Attention's own error reaches it through the streams: ~0.022 at pos 0 of layer 20 (ratio 1).
PRE_MIX_ATOL = 0.05
# An Engram layer's streams carry Engram's own error into the FFN hyper-connection: ~0.015 at pos 127 of layer 1,
# ~0.03 at pos 1 of layer 14.
ENGRAM_PRE_MIX_ATOL = 0.04
TP_SIZE = 4
PARENT_MESH = (8, 4)
CASES = pytest.mark.parametrize(
    "layers, seq_len",
    (
        ((0,), 160),  # window only; the ring wraps
        ((1,), 160),  # window only, Engram
        ((2, 3), 160),  # ratio 2: kv + index source, consumer
        ((14,), 160),  # ratio 2 kv + index source, Engram
        ((20, 21, 24), 576),  # ratio 1: source, consumer, index-only source; top-k drops 64 of 576
    ),
    ids=("window", "window_engram", "ratio2", "ratio2_engram", "ratio1"),
)
# DSpark's block: the token plus four drafts.
S = 5
CASES_S5 = pytest.mark.parametrize(
    "layers, seq_len",
    (
        # Blocks start at multiples of 5: both ratio-2 parities, one block across the causal
        # switch (125..129 holds window - 1) and the ring wrap.
        ((0,), 160),  # window only
        ((1,), 160),  # window only, Engram
        ((2, 3), 160),  # ratio 2: kv + index source, consumer
        ((14,), 160),  # ratio 2 kv + index source, Engram
        ((20, 21, 24), 580),  # ratio 1: source, consumer, index-only source; top-k drops 68 of 580
    ),
    ids=("window", "window_engram", "ratio2", "ratio2_engram", "ratio1"),
)


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


def _available_bytes():
    with open("/proc/meminfo") as f:
        return next(int(line.split()[1]) * 1024 for line in f if line.startswith("MemAvailable:"))


def _engram_tables(loader, layer_ids) -> dict:
    """The n-gram tables of ``layer_ids`` in host RAM; skips without 1.25x their size free."""
    nbytes = engram_ram_needed(loader, layer_ids)
    if _available_bytes() < 1.25 * nbytes:
        pytest.skip(f"needs {nbytes / 1e9:.0f} GB of free memory for the Engram tables")
    return {i: load_engram_table(loader, i) for i in layer_ids}


def _keep_output(outputs: dict, key: str):
    """Forward hook storing the module's output in ``outputs[key]``."""

    def hook(module, args, out):
        outputs[key] = out

    return hook


def _spy(module, method: str, outputs: dict, key: str, arg: int | None = None) -> None:
    """Have ``module.<method>`` also store a copy of each output -- or, given ``arg``, of that
    positional argument as passed in -- in ``outputs[key]``.

    A copy, since the layer frees the output, and on device, so the step dispatches without host
    syncs. ``clone`` cannot leave a sharded layout, hence the DRAM move.
    """
    call = getattr(module, method)

    def copy(t):
        if isinstance(t, (tuple, list)):
            return [copy(u) for u in t]
        return ttnn.clone(ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG))

    def spied(*args, **kwargs):
        if arg is not None:
            outputs[key] = copy(args[arg])
        out = call(*args, **kwargs)
        if arg is None:
            outputs[key] = copy(out)
        return out

    setattr(module, method, spied)


def _ffn_pre_attribution(block, h, attn, h_dev, attn_dev, post_dev, comb_dev, mid_dev) -> dict:
    """Where the FFN ``pre`` error comes from, as max abs errors.

    The reference's FFN ``pre`` is recomputed with one input of the attention-side mix (``hc_post``)
    swapped for the device's at a time; ``comb.T`` checks the mixing convention, and ``mix op``
    compares the device's mixed streams ``mid_dev`` against the reference mix of all device inputs.
    """
    _, post, comb = block.hc_mixes(h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
    post_dev, comb_dev, attn_dev = (
        post_dev.reshape(post.shape),
        comb_dev.reshape(comb.shape),
        attn_dev.reshape(attn.shape),
    )

    def ffn_pre(mid):
        return block.hc_mixes(mid, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)[0]

    expected = ffn_pre(block.hc_post(attn, h, post, comb))
    all_dev = ffn_pre(block.hc_post(attn_dev, h_dev, post_dev, comb_dev))
    swaps = {
        "attention": block.hc_post(attn_dev, h, post, comb),
        "Engram": block.hc_post(attn, h_dev, post, comb),
        "post": block.hc_post(attn, h, post_dev, comb),
        "comb": block.hc_post(attn, h, post, comb_dev),
        "comb.T": block.hc_post(attn, h, post, comb.transpose(-1, -2)),
    }
    report = {name: (ffn_pre(mid) - expected).abs().max().item() for name, mid in swaps.items()}
    report["all device inputs"] = (all_dev - expected).abs().max().item()
    report["mix op"] = (ffn_pre(mid_dev) - all_dev).abs().max().item()
    report["|comb_dev - comb|"] = (comb_dev - comb).abs().max().item()
    report["|comb_dev - comb.T|"] = (comb_dev - comb.transpose(-1, -2)).abs().max().item()
    report["|post_dev - post|"] = (post_dev - post).abs().max().item()
    return report


def _streams_attribution(block, h, x, expected, attn, ffn, mid_dev, ffn_dev, post_dev, comb_dev) -> dict:
    """Where the streams error comes from, as PCCs of ``out - x``.

    The reference's output is recomputed with one input of the FFN-side mix (``hc_post``) at a time
    swapped for the device's: the attention-side streams ``mid`` it reads, the MoE output, and the FFN
    hyper-connection's ``post`` / ``comb``. ``mix op`` compares the device's output against the
    reference mix of all device inputs, by max abs error.
    """
    _, post, comb = block.hc_mixes(h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
    mid = block.hc_post(attn, h, post, comb)
    _, ffn_post, ffn_comb = block.hc_mixes(mid, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
    mid_dev, ffn_dev, post_dev, comb_dev = (
        t.reshape(r.shape) for t, r in ((mid_dev, mid), (ffn_dev, ffn), (post_dev, ffn_post), (comb_dev, ffn_comb))
    )
    swaps = {
        "none": block.hc_post(ffn, mid, ffn_post, ffn_comb),
        "attention-side streams": block.hc_post(ffn, mid_dev, ffn_post, ffn_comb),
        "MoE": block.hc_post(ffn_dev, mid, ffn_post, ffn_comb),
        "FFN post": block.hc_post(ffn, mid, post_dev, ffn_comb),
        "FFN comb": block.hc_post(ffn, mid, ffn_post, comb_dev),
        "all": block.hc_post(ffn_dev, mid_dev, post_dev, comb_dev),
    }
    report = {name: comp_pcc((expected - x).reshape(-1), (out - x).reshape(-1))[1] for name, out in swaps.items()}
    report["|FFN comb_dev - comb|"] = (comb_dev - ffn_comb).abs().max().item()
    report["|FFN post_dev - post|"] = (post_dev - ffn_post).abs().max().item()
    return report, swaps["all"]


def _to_torch(t: ttnn.Tensor) -> torch.Tensor:
    """Rank 0's copy of a tensor replicated over the mesh, as fp32."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _rank_spread(t: ttnn.Tensor) -> float:
    """Max abs difference between any rank's copy of a replicated tensor and rank 0's (0 on one chip)."""
    copies = [ttnn.to_torch(c).float() for c in ttnn.get_device_tensors(t)]
    return max((c - copies[0]).abs().max().item() for c in copies)


def _log_pcc(expected, actual, what: str) -> str | None:
    """Log the PCC of ``actual`` against ``expected``; the failure message if it is below ``PCC``."""
    passing, pcc = comp_pcc(expected.reshape(-1), actual.reshape(-1), PCC)
    logger.info(f"{what}: PCC {pcc}")
    return None if passing else f"{what}: PCC {pcc} < {PCC}"


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@CASES
@torch.no_grad()
def test_decoder_layer_decode(device, layers, seq_len):
    _check_decode(device, layers, seq_len, tp_size=1)


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@CASES
@torch.no_grad()
def test_decoder_layer_decode_tp4(mesh_device, layers, seq_len):
    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"needs an {PARENT_MESH[0]}x{PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, TP_SIZE), ttnn.MeshCoordinate(0, 0))
    _check_decode(submesh, layers, seq_len, tp_size=TP_SIZE)


def _check_decode(device, layers, seq_len, tp_size: int) -> None:
    """Decode ``layers`` for ``seq_len`` steps on ``device`` (a 1x``tp_size`` mesh) against the reference."""
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    engram_layers = [i for i in layers if i in config.engram_layer_ids]
    tables = _engram_tables(loader, engram_layers)

    model = load_reference(snapshot)
    reference = reference_blocks(model, snapshot, loader, layers, seq_len)
    expected_parts = {i: {} for i in layers}
    for i, block in reference.items():
        block.attn.register_forward_hook(_keep_output(expected_parts[i], "attn"))
        block.ffn.register_forward_hook(_keep_output(expected_parts[i], "ffn"))

    cache = WeightCache(CACHE_DIR)
    # Only Engram's wkv is prefetched (the rest of the layer copies its weights per call); its ring
    # is private, so the device's mapping starts empty and the first Engram layer builds it.
    prefetch_buffers = None
    if engram_layers:
        if not ttnn.experimental.is_tensor_prefetcher_supported(device):
            pytest.skip("Engram's wkv needs the tensor prefetcher (programmable DRAM cores)")
        prefetch_buffers = {}
    decoder_layers, parts = {}, {i: {} for i in layers}
    for i in layers:
        layer_cache = cache.sub(f"layers.{i}")
        experts = DeepSeekV41PreloadedExperts(
            config,
            expert_provider(loader, i),
            device,
            dtype=ttnn.bfloat4_b,
            cache=layer_cache.sub("ffn"),
            tp_size=tp_size,
        )
        layer = DeepSeekV41DecoderLayer(
            config,
            i,
            decoder_layer_weights(loader, i, config),
            device,
            experts,
            cache=layer_cache,
            weight_dtype=ttnn.bfloat4_b,
            tp_size=tp_size,
            prefetch_buffers=prefetch_buffers,
        )
        _spy(layer.self_attn, "decode", parts[i], "attn")
        _spy(layer.mlp, "decode_static", parts[i], "ffn")
        _spy(layer.ffn_hc, "forward", parts[i], "ffn_hc_in", arg=0)
        _spy(layer.ffn_hc, "forward", parts[i], "ffn_hc")
        _spy(layer.attn_hc, "forward", parts[i], "attn_hc")
        if layer.engram is not None:
            _spy(layer.engram, "forward", parts[i], "engram")
        decoder_layers[i] = layer
    caches = build_decode_caches(device, config, layers, seq_len)
    rope = {i: (reference[i].attn.freqs_cis.real, reference[i].attn.freqs_cis.imag) for i in layers}

    torch.manual_seed(0)
    hc = config.hc_mult
    streams = torch.randn(len(layers), seq_len, 1, 1, hc, config.hidden_size).bfloat16().float()
    pre_mixes = torch.rand(len(layers), seq_len, 1, 1, hc).bfloat16().float()
    if engram_layers:
        tokenizer = AutoTokenizer.from_pretrained(snapshot)
        input_ids = torch.randint(len(tokenizer), (1, seq_len))
        reference_hash = reference_engram_hash(model, snapshot, tokenizer, seq_len)
        reference_engrams = {i: reference_engram(model, snapshot, loader, i, *tables[i]) for i in engram_layers}
        lookup = EngramHostLookup(config, snapshot, tables, 1, seq_len, tokenizer)

    window = config.sliding_window
    checked = {0, 1, window - 2, window - 1, window} | set(range(seq_len - 4, seq_len))
    moe_outs = {i: ([], []) for i in layers}  # (expected, actual) per checked step
    with contextlib.ExitStack() as session:
        if prefetch_buffers is not None:
            session.enter_context(tensor_prefetcher_session(device))
            ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        for pos in range(seq_len):
            if engram_layers:
                token = input_ids[:, pos : pos + 1]
                rows = lookup(token, pos)  # {layer: [1, 1, n_hash_cols * head_dim]}
                hashes = reference_hash(token, pos)  # [1, 1, n_engram_layers, n_hash_cols]
            for j, i in enumerate(layers):
                x, pre_mix = streams[j, pos], pre_mixes[j, pos]
                h = x
                if i in engram_layers:
                    h = reference_engrams[i](x, hashes[:, :, reference_hash.layout.layer_ids.index(i)])
                expected, expected_pre = reference[i](h, pos, pre_mix, None)

                layer = decoder_layers[i]
                # Rows go in before the step: the step can sync the host (a CCL's first call allocates a
                # global semaphore with a blocking write), which would wait forever on rows not yet sent.
                if layer.engram is not None:
                    layer.engram.write_rows(rows[i])
                    layer.engram.wkv.fetch_weights()
                out, out_pre = layer.decode(
                    ttnn.from_torch(
                        x,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device,
                        memory_config=width_sharded_l1_config(
                            hc, config.hidden_size, device, num_cores=_HC_COLLAPSE_CORES, tile_height=1
                        ),
                    ),
                    ttnn.from_torch(
                        pre_mix.reshape(1, 1, 1, hc), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    ),
                    caches[i],
                    **layer.self_attn.decode_inputs(pos, *rope[i]),
                )

                # Every step is logged in full, then only the checked ones are asserted.
                check = pos in checked
                what = f"layer {i} (ratio {config.compress_ratios[i]}) pos {pos}"
                failures = [
                    _log_pcc(expected - x, _to_torch(out) - x, f"{what} streams (out - x)"),
                    _log_pcc(expected_parts[i]["attn"], _to_torch(parts[i]["attn"]), f"{what} attention"),
                ]
                if tp_size > 1:
                    logger.info(
                        f"{what} max abs spread over the ranks: streams {_rank_spread(out)}, attention "
                        f"{_rank_spread(parts[i]['attn'])}, MoE {_rank_spread(parts[i]['ffn'])}, attention comb "
                        f"{_rank_spread(parts[i]['attn_hc'][1])}, FFN input {_rank_spread(parts[i]['ffn_hc_in'])}, "
                        f"FFN comb {_rank_spread(parts[i]['ffn_hc'][1])}"
                    )
                if layer.engram is not None:
                    engram_delta = _to_torch(parts[i]["engram"]) - x
                    logger.info(
                        f"{what} Engram (out - x): PCC {comp_pcc((h - x).reshape(-1), engram_delta.reshape(-1))[1]}"
                    )
                # The reference's FFN pre, from the streams the device's FFN hyper-connection was given: the
                # device's pre against it is that op's own error, it against expected_pre what came in with them.
                block = reference[i]
                ffn_hc_in = _to_torch(parts[i]["ffn_hc_in"])
                pre_given_input = block.hc_mixes(ffn_hc_in, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)[0]
                actual_pre, expected_pre, pre_given_input = (
                    t.reshape(-1) for t in (_to_torch(out_pre), expected_pre, pre_given_input)
                )
                pre_error = (actual_pre - expected_pre).abs().max().item()
                logger.info(
                    f"{what} pre_mix: max abs error {pre_error} (hyper-connection op "
                    f"{(actual_pre - pre_given_input).abs().max().item()}, from its input "
                    f"{(pre_given_input - expected_pre).abs().max().item()}); "
                    f"expected {expected_pre.tolist()}, actual {actual_pre.tolist()}"
                )
                post_dev, comb_dev = (_to_torch(t) for t in parts[i]["attn_hc"][:2])
                attribution = _ffn_pre_attribution(
                    block,
                    h,
                    expected_parts[i]["attn"],
                    _to_torch(parts[i]["engram"]) if layer.engram is not None else x,
                    _to_torch(parts[i]["attn"]),
                    post_dev,
                    comb_dev,
                    ffn_hc_in,
                )
                logger.info(
                    f"{what} FFN pre error by device input: "
                    + ", ".join(f"{k} {v:.4f}" for k, v in attribution.items())
                )
                pre_atol = ENGRAM_PRE_MIX_ATOL if layer.engram is not None else PRE_MIX_ATOL
                if pre_error > pre_atol:
                    failures.append(f"{what} pre_mix: max abs error {pre_error} > {pre_atol}")
                moe_expected, moe_actual = expected_parts[i]["ffn"].reshape(-1), _to_torch(parts[i]["ffn"]).reshape(-1)
                _, pcc = comp_pcc(moe_expected, moe_actual, PCC)
                logger.info(f"{what} MoE: PCC {pcc}")
                out_dev = _to_torch(out)
                streams_report, out_given_inputs = _streams_attribution(
                    block,
                    h,
                    x,
                    expected,
                    expected_parts[i]["attn"],
                    expected_parts[i]["ffn"],
                    ffn_hc_in,
                    moe_actual,
                    *(_to_torch(t) for t in parts[i]["ffn_hc"][:2]),
                )
                streams_report["mix op"] = (out_dev - out_given_inputs.reshape(out_dev.shape)).abs().max().item()
                logger.info(
                    f"{what} streams PCC by device input: "
                    + ", ".join(f"{k} {v:.4f}" for k, v in streams_report.items())
                )
                if check:
                    moe_outs[i][0].append(moe_expected)
                    moe_outs[i][1].append(moe_actual)
                    failures = [f for f in failures if f is not None]
                    assert not failures, "; ".join(failures)

    for i, (moe_expected, moe_actual) in moe_outs.items():
        failure = _log_pcc(torch.stack(moe_expected), torch.stack(moe_actual), f"layer {i} MoE, checked steps")
        assert failure is None, failure


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@CASES_S5
@torch.no_grad()
def test_decoder_layer_decode_s5(device, layers, seq_len):
    assert seq_len % S == 0
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    engram_layers = [i for i in layers if i in config.engram_layer_ids]
    tables = _engram_tables(loader, engram_layers)

    model = load_reference(snapshot)
    reference = reference_blocks(model, snapshot, loader, layers, seq_len)
    expected_parts = {i: {} for i in layers}
    for i, block in reference.items():
        block.attn.register_forward_hook(_keep_output(expected_parts[i], "attn"))
        block.ffn.register_forward_hook(_keep_output(expected_parts[i], "ffn"))

    cache = WeightCache(CACHE_DIR)
    prefetch_buffers = None
    if engram_layers:
        if not ttnn.experimental.is_tensor_prefetcher_supported(device):
            pytest.skip("Engram's wkv needs the tensor prefetcher (programmable DRAM cores)")
        prefetch_buffers = {}
    decoder_layers, parts = {}, {i: {} for i in layers}
    for i in layers:
        layer_cache = cache.sub(f"layers.{i}")
        experts = DeepSeekV41PreloadedExperts(
            config, expert_provider(loader, i), device, dtype=ttnn.bfloat4_b, cache=layer_cache.sub("ffn")
        )
        layer = DeepSeekV41DecoderLayer(
            config,
            i,
            decoder_layer_weights(loader, i, config),
            device,
            experts,
            cache=layer_cache,
            weight_dtype=ttnn.bfloat4_b,
            prefetch_buffers=prefetch_buffers,
            verify_rows=S,
        )
        _spy(layer.self_attn, "decode", parts[i], "attn")
        _spy(layer.mlp, "decode_static", parts[i], "ffn")
        decoder_layers[i] = layer
    caches = build_decode_caches(device, config, layers, seq_len, verify_rows=S)
    rope = {i: (reference[i].attn.freqs_cis.real, reference[i].attn.freqs_cis.imag) for i in layers}

    torch.manual_seed(0)
    hc, hidden = config.hc_mult, config.hidden_size
    streams = torch.randn(len(layers), seq_len, 1, 1, hc, hidden).bfloat16().float()
    pre_mixes = torch.rand(len(layers), seq_len, 1, 1, hc).bfloat16().float()
    if engram_layers:
        tokenizer = AutoTokenizer.from_pretrained(snapshot)
        input_ids = torch.randint(len(tokenizer), (1, seq_len))
        reference_hash = reference_engram_hash(model, snapshot, tokenizer, seq_len)
        reference_engrams = {i: reference_engram(model, snapshot, loader, i, *tables[i]) for i in engram_layers}
        lookup = EngramHostLookup(config, snapshot, tables, 1, seq_len, tokenizer)
    # The layer's streams, as the Engram takes a verify block's.
    streams_config = width_sharded_l1_config(S * hc, hidden, device, num_cores=_HC_COLLAPSE_CORES, tile_height=1)

    window = config.sliding_window
    checked = {0, 1, window - 2, window - 1, window} | set(range(seq_len - 4, seq_len))
    moe_outs = {i: ([], []) for i in layers}  # (expected, actual) per checked row
    writer = ThreadPoolExecutor(max_workers=1)
    with contextlib.ExitStack() as session:
        # A failed step can leave a write blocked on the device for good; do not wait on it.
        session.callback(writer.shutdown, wait=False)
        if prefetch_buffers is not None:
            session.enter_context(tensor_prefetcher_session(device))
            ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        for start in range(0, seq_len, S):
            positions = range(start, start + S)
            if engram_layers:
                # One lookup over the block, as a verify step hashes its run of one session.
                rows = lookup(input_ids[:, start : start + S], start)  # {layer: [1, S, n_hash_cols * head_dim]}
            # The reference shares the group's compressed state, so it steps position-major, token by token.
            expected = {i: [] for i in layers}
            for pos in positions:
                if engram_layers:
                    hashes = reference_hash(input_ids[:, pos : pos + 1], pos)
                for j, i in enumerate(layers):
                    x, pre_mix = streams[j, pos], pre_mixes[j, pos]
                    h = x
                    if i in engram_layers:
                        h = reference_engrams[i](x, hashes[:, :, reference_hash.layout.layer_ids.index(i)])
                    out, out_pre = reference[i](h, pos, pre_mix, None)
                    expected[i].append((out, out_pre, expected_parts[i]["attn"], expected_parts[i]["ffn"]))

            for j, i in enumerate(layers):
                layer = decoder_layers[i]
                write = None
                if layer.engram is not None:
                    # The block's rows outnumber the socket's pages, so they go in while the step is dispatched.
                    write = writer.submit(layer.engram.write_rows, rows[i])
                    layer.engram.wkv.fetch_weights()
                out, out_pre = layer.decode(
                    ttnn.from_torch(
                        streams[j, start : start + S].reshape(1, S, hc, hidden),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device,
                        memory_config=streams_config,
                    ),
                    ttnn.from_torch(
                        pre_mixes[j, start : start + S].reshape(1, S, 1, hc),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    ),
                    caches[i],
                    **layer.self_attn.decode_inputs(start, *rope[i], rows=S),
                )
                if write is not None:
                    write.result()
                assert list(out.shape) == [1, S, hc, hidden], list(out.shape)
                assert list(out_pre.shape) == [1, S, 1, hc], list(out_pre.shape)
                out_dev, pre_dev = _to_torch(out), _to_torch(out_pre)
                attn_dev, ffn_dev = (_to_torch(parts[i][k]).reshape(S, -1) for k in ("attn", "ffn"))

                # Every row is logged, then only the checked ones are asserted.
                pre_atol = ENGRAM_PRE_MIX_ATOL if layer.engram is not None else PRE_MIX_ATOL
                for k, pos in enumerate(positions):
                    x = streams[j, pos]
                    expected_out, expected_pre, expected_attn, expected_ffn = expected[i][k]
                    what = f"layer {i} (ratio {config.compress_ratios[i]}) pos {pos} (row {k} of the block at {start})"
                    failures = [
                        _log_pcc(expected_out - x, out_dev[:, k : k + 1] - x, f"{what} streams (out - x)"),
                        _log_pcc(expected_attn, attn_dev[k], f"{what} attention"),
                    ]
                    pre_error = (pre_dev[:, k].reshape(-1) - expected_pre.reshape(-1)).abs().max().item()
                    logger.info(f"{what} pre_mix: max abs error {pre_error}")
                    if pre_error > pre_atol:
                        failures.append(f"{what} pre_mix: max abs error {pre_error} > {pre_atol}")
                    _, pcc = comp_pcc(expected_ffn.reshape(-1), ffn_dev[k], PCC)
                    logger.info(f"{what} MoE: PCC {pcc}")
                    if pos in checked:
                        moe_outs[i][0].append(expected_ffn.reshape(-1))
                        moe_outs[i][1].append(ffn_dev[k])
                        failures = [f for f in failures if f is not None]
                        assert not failures, "; ".join(failures)

    for i, (moe_expected, moe_actual) in moe_outs.items():
        failure = _log_pcc(torch.stack(moe_expected), torch.stack(moe_actual), f"layer {i} MoE, checked rows")
        assert failure is None, failure
